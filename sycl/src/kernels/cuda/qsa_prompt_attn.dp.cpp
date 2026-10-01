// src/kernels/cuda/qsa_prompt_attn.cu - see include/strata/kernels/qsa_prompt_attn.hpp.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/qsa_prompt_attn_xmx.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_q4.hpp"

#include <cmath>

#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace strata::kernels {
namespace {

constexpr int HD = 256;           // head_dim
constexpr int G = 12;             // query heads per KV head
#ifndef D1_CH
#define D1_CH 32
#endif
constexpr int CH = D1_CH;         // cells per chunk
constexpr int THREADS = 128;      // 4 warps: scores by cell (8 each), p.v by dimension (64 each = one int8 scale group)
constexpr int QS = HD + 8;        // q row stride in halves (bank-conflict-free fragment loads)

// The MMA below needs sm_75 or newer (Turing runs it as two k=8 steps); cp.async needs sm_80. Builds for pre-sm_75
// cards compile the MMA to a trap; qsa_prompt_attn_batch refuses such a device at run time, so the old kernel runs
// there.  Turing compiles cp_async16 to a trap as well and takes the v1 kernel instead of launch_i8.
#if defined(__HIPCC__)          // AMD: no mma.sync / cp.async; the host keeps the old kernel (below)
#define STRATA_PA_SM80 0
#elif 0   // SYCL: inline PTX (mma/ldmatrix/cp.async) - the XMX port is pending; see tools/fixups.py
#define STRATA_PA_SM80 1
#else
#define STRATA_PA_SM80 0
#endif

// m16n8k16 with f16 inputs needs sm_80.  Turing (sm_75) has m16n8k8 with the SAME A/B/C register mapping, so the
// k=16 step is two k=8 steps on the fragments as they are already laid out: a[0]/a[1] are rows gid/gid+8 at k columns
// 2*tig..2*tig+1 (b[0]'s k rows), a[2]/a[3] the same rows at k columns 2*tig+8..2*tig+9 (b[1]'s k rows).  The
// products then add into the same FP32 C registers in the order hi-part-0, hi-part-1, which is the order the k16
// instruction accumulates in as well - but the sum now rounds twice, so the two paths do not agree bit for bit.
__dpct_inline__ void mma16816(float *c, const uint32_t *a, const uint32_t *b) {
#if !STRATA_PA_SM80 &&                                                         \
    (defined(__HIPCC__) || !defined(DPCT_COMPATIBILITY_TEMP) ||                \
     DPCT_COMPATIBILITY_TEMP < 750)
    /* unreachable on SYCL: the launcher refuses this device */   // AMD and pre-Turing builds: no mma.sync (the host keeps the old kernel there)
#elif 0   // SYCL: Turing's m16n8k8 PTX (unreachable: the launcher refuses this device)
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(b[0]));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[2]), "r"(a[3]), "r"(b[1]));
#else
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
                   "r"(b[1]));
#else
    {
        volatile void *d_mat_frag_ct1[4] = {&c[0], &c[1], &c[2], &c[3]};
        sycl::vec<uint32_t, 4> a_mat_frag_ct1(a[0], a[1], a[2], a[3]);
        sycl::vec<uint32_t, 2> b_mat_frag_ct1(b[0], b[1]);
        sycl::vec<float, 4> c_mat_frag_ct1(c[0], c[1], c[2], c[3]);
        dpct::experimental::matrix::mma<16, 8, 16, sycl::half, float>(
            reinterpret_cast<volatile void **>(d_mat_frag_ct1), &a_mat_frag_ct1,
            &b_mat_frag_ct1, &c_mat_frag_ct1);
    }
#endif
#endif
}

// Two int8 codes (low byte first) as an exact half2: 1024 + (c + 128) built in the mantissa, minus 1152.
__dpct_inline__ uint32_t i8x2_to_h2(uint32_t x) {
    uint32_t y = ((x & 0xffu) | ((x & 0xff00u) << 8)) ^ 0x00800080u;
    y |= 0x64006400u;
    sycl::half2 h = *reinterpret_cast<sycl::half2 *>(&y);
    h = h - sycl::half2(
                sycl::vec<float, 1>(1152.f)
                    .convert<sycl::half, sycl::rounding_mode::automatic>()[0],
                sycl::vec<float, 1>(1152.f)
                    .convert<sycl::half, sycl::rounding_mode::automatic>()[0]);
    return *reinterpret_cast<uint32_t*>(&h);
}

__dpct_inline__ uint32_t pack_h2(float lo_k,
                                 float hi_k) { // element k in the low half
    sycl::half2 h = sycl::float2(lo_k, hi_k)
                        .convert<sycl::half, sycl::rounding_mode::rte>();
    return *reinterpret_cast<uint32_t*>(&h);
}

// KV_MODE 1: int8 codes + fp16 scale per 64 values. KV_MODE 0: fp16 values (scales 1).
// KV_MODE 3 (hybrid K8V4): K as mode 1, V as mode 0 - the row's q4_0 blocks are dequantized to fp16 at
// gather, so everything downstream of the load is the mode-0 V path; the caller un-rotates the output.
template <int KV_MODE>
struct Smem {
    using KElem =
        typename std::conditional<KV_MODE == 0, sycl::half, int8_t>::type;
    using VElem =
        typename std::conditional<KV_MODE == 1, int8_t, sycl::half>::type;
    static constexpr int KROW = KV_MODE == 0 ? HD + 8 : HD + 16;   // elements; 16-byte aligned rows, banks spread
    static constexpr int VROW = KV_MODE == 1 ? HD + 16 : HD + 8;
    sycl::half qh[16][QS];
    sycl::half ql[16][QS];
    KElem k[CH][KROW];
    VElem v[CH][VROW];
    float ks[CH][4];
    float vs[CH][4];
    float s[16][CH + 1];
    float qmax[THREADS / 32];
    float alpha[16];
    float lsum[16];
    float mrow[16];
    long long row[CH];
};

template <int KV_MODE>
/*
DPCT1110: The total declared local variable size in device function
prompt_attn_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
prompt_attn_kernel(const float *__restrict__ q, QsaAttnPools p,
                   const int32_t *__restrict__ ids,
                   const int32_t *__restrict__ steps, int n_kv_heads,
                   int page_size, float scale_log2, float *__restrict__ attn,
                   int cap, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto smem_raw = (unsigned char *)dpct_local;
    Smem<KV_MODE>& S = *reinterpret_cast<Smem<KV_MODE>*>(smem_raw);
    const int qi = item_ct1.get_group(2), kvh = item_ct1.get_group(1);
    const int n_head = n_kv_heads * G;
    q += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    attn += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    ids += (size_t) qi * cap;
    /*
    DPCT1098: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    const int n = *(steps + (size_t)qi * kStepCount + kStepWidth);
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int gid = lane >> 2, tig = lane & 3;

    // q: 12 heads + 4 zero rows, scaled by a power of two that puts its largest value near 2^14 (exact, and the
    // lo halves stay out of FP16's subnormal range), then split into hi + lo halves
    float qm = 0.0f;
#pragma unroll
    for (int i = t; i < G * HD; i += THREADS)
        qm = sycl::fmax(qm, sycl::fabs(q[i]));
#pragma unroll
    /*
    DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = 16; o > 0; o >>= 1) qm = sycl::fmax(
        qm, dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                qm, o));
    if (lane == 0) S.qmax[warp] = qm;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    qm = sycl::fmax(sycl::fmax(S.qmax[0], S.qmax[1]),
                    sycl::fmax(S.qmax[2], S.qmax[3]));
    int qe = 0;
    /*
    DPCT1017: The sycl::frexp call is used instead of the frexpf call. These
    two calls do not provide exactly the same functionality. Check the potential
    precision and/or performance issues for the generated code.
    */
    if (qm > 0.0f) sycl::frexp(
        qm, sycl::address_space_cast<sycl::access::address_space::generic_space,
                                     sycl::access::decorated::yes>(
                &qe)); // qm < 2^qe
    const float qup = sycl::ldexp(1.0f, 14 - qe),
                qdown = sycl::ldexp(scale_log2, qe - 14);
    for (int i = t; i < 16 * HD; i += THREADS) {
        const int h = i / HD, d = i % HD;
        const float x = h < G ? q[(size_t) h * HD + d] * qup : 0.0f;
        const sycl::half hi =
            sycl::vec<float, 1>(x)
                .convert<sycl::half, sycl::rounding_mode::rte>()[0];
        S.qh[h][d] = hi;
        S.ql[h][d] =
            sycl::vec<float, 1>(
                x - sycl::vec<sycl::half, 1>(hi)
                        .convert<float, sycl::rounding_mode::automatic>()[0])
                .convert<sycl::half, sycl::rounding_mode::rte>()[0];
    }
    if (t < 16) { S.mrow[t] = -INFINITY; S.lsum[t] = 0.0f; }

    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;

    for (int c0 = 0; c0 < n; c0 += CH) {
        const int nh = sycl::min(CH, n - c0);
        if (t < CH) {
            long long r = -1;
            if (t < nh) {
                const int cell = ids[c0 + t];
                const long long page = (long long) p.page_table[cell / page_size];
                r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
            }
            S.row[t] = r;
        }
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier(); // rows ready; the previous chunk's p.v is done with
                            // k, v, s
        // gather the chunk's K and V rows (16-byte pieces; K8V4's V as q4_0 blocks dequantized to fp16)
        // and their scales
        {
            constexpr int KPIECES = HD * (int) sizeof(typename Smem<KV_MODE>::KElem) / 16;   // per K row
            for (int i = t; i < CH * KPIECES; i += THREADS) {
                const int c = i / KPIECES, pc = i % KPIECES;
                const long long r = S.row[c];
                sycl::uint4 kx = sycl::uint4(0, 0, 0, 0);
                if (r >= 0) {
                    if constexpr (KV_MODE == 0)
                        /*
                        DPCT1098: The '*' expression is used instead of the
                        __ldg call. These two expressions do not provide the
                        exact same functionality. Check the generated code for
                        potential precision and/or performance issues.
                        */
                        kx = *(reinterpret_cast<const sycl::uint4 *>(p.k_pool +
                                                                     r * HD) +
                               pc);
                    else   // modes 1 and 3: the K side is INT8
                        /*
                        DPCT1098: The '*' expression is used instead of the
                        __ldg call. These two expressions do not provide the
                        exact same functionality. Check the generated code for
                        potential precision and/or performance issues.
                        */
                        kx = *(reinterpret_cast<const sycl::uint4 *>(p.k_q +
                                                                     r * HD) +
                               pc);
                }
                *reinterpret_cast<sycl::uint4 *>(
                    reinterpret_cast<unsigned char *>(&S.k[c][0]) + pc * 16) =
                    kx;
            }
            if constexpr (KV_MODE == 3) {   // V: dequantize the row's q4_0 blocks straight into the fp16 V row
                constexpr int BLKS = HD / QK4_0;
                constexpr int BYTES = BLKS * (int) sizeof(block_q4_0);
                for (int i = t; i < CH * BLKS; i += THREADS) {
                    const int c = i / BLKS, b = i % BLKS;
                    const long long r = S.row[c];
#pragma unroll
                    for (int j = 0; j < QK4_0; ++j)
                        S.v[c][b * QK4_0 + j] = sycl::half(0);
                    if (r >= 0) {
                        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + r * BYTES) + b;
                        const float d =
                            sycl::vec<sycl::half, 1>(
                                sycl::bit_cast<sycl::half, unsigned short>(
                                    blk->d))
                                .convert<float,
                                         sycl::rounding_mode::automatic>()[0];
#pragma unroll
                        for (int j = 0; j < QK4_0 / 2; ++j) {
                            S.v[c][b * QK4_0 + j] =
                                sycl::vec<float, 1>(
                                    (float)((int)(blk->qs[j] & 0x0F) - 8) * d)
                                    .convert<sycl::half,
                                             sycl::rounding_mode::rte>()[0];
                            S.v[c][b * QK4_0 + j + QK4_0 / 2] =
                                sycl::vec<float, 1>(
                                    (float)((int)(blk->qs[j] >> 4) - 8) * d)
                                    .convert<sycl::half,
                                             sycl::rounding_mode::rte>()[0];
                        }
                    }
                }
            } else {
                constexpr int VPIECES = HD * (int) sizeof(typename Smem<KV_MODE>::VElem) / 16;   // per V row
                for (int i = t; i < CH * VPIECES; i += THREADS) {
                    const int c = i / VPIECES, pc = i % VPIECES;
                    const long long r = S.row[c];
                    sycl::uint4 vx = sycl::uint4(0, 0, 0, 0);
                    if (r >= 0) {
                        if constexpr (KV_MODE == 1)
                            /*
                            DPCT1098: The '*' expression is used instead of
                            the __ldg call. These two expressions do not provide
                            the exact same functionality. Check the generated
                            code for potential precision and/or performance
                            issues.
                            */
                            vx = *(reinterpret_cast<const sycl::uint4 *>(
                                       p.v_q + r * HD) +
                                   pc);
                        else
                            /*
                            DPCT1098: The '*' expression is used instead of
                            the __ldg call. These two expressions do not provide
                            the exact same functionality. Check the generated
                            code for potential precision and/or performance
                            issues.
                            */
                            vx = *(reinterpret_cast<const sycl::uint4 *>(
                                       p.v_pool + r * HD) +
                                   pc);
                    }
                    *reinterpret_cast<sycl::uint4 *>(
                        reinterpret_cast<unsigned char *>(&S.v[c][0]) +
                        pc * 16) = vx;
                }
            }
            for (int i = t; i < CH * 4; i += THREADS) {
                const int c = i / 4, g = i % 4;
                const long long r = S.row[c];
                float a = 0.0f, b = 0.0f;
                if (r >= 0) {
                    if constexpr (KV_MODE == 1) {
                        a = sycl::vec<sycl::half, 1>(
                                sycl::bit_cast<sycl::half, unsigned short>(
                                    p.k_scale[r * (HD / KV_Q8_GROUP) + g]))
                                .convert<float,
                                         sycl::rounding_mode::automatic>()[0];
                        b = sycl::vec<sycl::half, 1>(
                                sycl::bit_cast<sycl::half, unsigned short>(
                                    p.v_scale[r * (HD / KV_Q8_GROUP) + g]))
                                .convert<float,
                                         sycl::rounding_mode::automatic>()[0];
                    } else if constexpr (KV_MODE == 3) {   // K as int8, V dequantized to fp16 (scale 1)
                        a = sycl::vec<sycl::half, 1>(
                                sycl::bit_cast<sycl::half, unsigned short>(
                                    p.k_scale[r * (HD / KV_Q8_GROUP) + g]))
                                .convert<float,
                                         sycl::rounding_mode::automatic>()[0];
                        b = 1.0f;
                    } else {
                        a = b = 1.0f;
                    }
                }
                S.ks[c][g] = a;
                S.vs[c][g] = b;
            }
        }
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        // scores: warp w takes cells 8w..8w+7 (one n-tile) over all 256 dims, per 64-dim scale group
#pragma unroll
        for (int nt = 0; nt < CH / 32; ++nt) {
            const int cb = (warp + 4 * nt) * 8;
            float sc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                float tg[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const int k0 = (g * 4 + kk) * 16;
                    uint32_t ah[4], al[4], b[2];
                    ah[0] = *reinterpret_cast<const uint32_t*>(&S.qh[gid][k0 + 2 * tig]);
                    ah[1] = *reinterpret_cast<const uint32_t*>(&S.qh[gid + 8][k0 + 2 * tig]);
                    ah[2] = *reinterpret_cast<const uint32_t*>(&S.qh[gid][k0 + 2 * tig + 8]);
                    ah[3] = *reinterpret_cast<const uint32_t*>(&S.qh[gid + 8][k0 + 2 * tig + 8]);
                    al[0] = *reinterpret_cast<const uint32_t*>(&S.ql[gid][k0 + 2 * tig]);
                    al[1] = *reinterpret_cast<const uint32_t*>(&S.ql[gid + 8][k0 + 2 * tig]);
                    al[2] = *reinterpret_cast<const uint32_t*>(&S.ql[gid][k0 + 2 * tig + 8]);
                    al[3] = *reinterpret_cast<const uint32_t*>(&S.ql[gid + 8][k0 + 2 * tig + 8]);
                    if constexpr (KV_MODE != 0) {   // modes 1 and 3: the K side is INT8 codes
                        b[0] = i8x2_to_h2(*reinterpret_cast<const uint16_t*>(&S.k[cb + gid][k0 + 2 * tig]));
                        b[1] = i8x2_to_h2(*reinterpret_cast<const uint16_t*>(&S.k[cb + gid][k0 + 2 * tig + 8]));
                    } else {
                        b[0] = *reinterpret_cast<const uint32_t*>(&S.k[cb + gid][k0 + 2 * tig]);
                        b[1] = *reinterpret_cast<const uint32_t*>(&S.k[cb + gid][k0 + 2 * tig + 8]);
                    }
                    mma16816(tg, ah, b);
#ifndef D1_NO_QLO
                    mma16816(tg, al, b);
#endif
                }
                const float s0 = S.ks[cb + 2 * tig][g], s1 = S.ks[cb + 2 * tig + 1][g];
                sc[0] = sycl::fma(tg[0], (float)s0, sc[0]);
                sc[1] = sycl::fma(tg[1], (float)s1, sc[1]);
                sc[2] = sycl::fma(tg[2], (float)s0, sc[2]);
                sc[3] = sycl::fma(tg[3], (float)s1, sc[3]);
            }
            const int c = cb + 2 * tig;
            S.s[gid][c] = c < nh ? sc[0] * qdown : -INFINITY;
            S.s[gid][c + 1] = c + 1 < nh ? sc[1] * qdown : -INFINITY;
            S.s[gid + 8][c] = c < nh ? sc[2] * qdown : -INFINITY;
            S.s[gid + 8][c + 1] = c + 1 < nh ? sc[3] * qdown : -INFINITY;
        }
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        // online softmax: row t/8, 4 cells per thread, 8 threads per row (lanes 8r..8r+7 of a warp)
        {
            constexpr int PER = CH / 8;
            const int r = t >> 3, sub = t & 7;
            float x[PER], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                x[j] = S.s[r][sub * PER + j]; mx = sycl::fmax(mx, x[j]);
            }
#pragma unroll
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            for (int o = 1; o < 8; o <<= 1) mx = sycl::fmax(
                mx,
                dpct::experimental::permute_sub_group_by_xor(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), mx, o));
            const float m_old = S.mrow[r];
            const float m_new = sycl::fmax(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                const float e =
                    x[j] == -INFINITY ? 0.0f : sycl::exp2(x[j] - m_new);
                S.s[r][sub * PER + j] = e;
                sum += e;
            }
#pragma unroll
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            for (int o = 1; o < 8; o <<= 1) sum +=
                dpct::experimental::permute_sub_group_by_xor(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), sum, o);
            sycl::group_barrier(
                sycl::ext::oneapi::this_work_item::get_sub_group());
            if (sub == 0) {
                const float a =
                    m_old == -INFINITY ? 0.0f : sycl::exp2(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = sycl::fma(S.lsum[r], (float)a, sum);
                S.mrow[r] = m_new;
            }
        }
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        // p.v: warp w owns dims [64w, 64w+64), which is int8 scale group w. The scale is folded into p relative to
        // the chunk's largest, times 2^14 (p' <= 2^14: its lo half stays out of FP16's subnormal range); the chunk's
        // sum is then added to the running one in FP32 with the factor taken back out
        {
            float vmax = 0.0f;
#pragma unroll
            for (int c = lane; c < CH; c += 32)
                vmax = sycl::fmax(vmax, S.vs[c][warp]);
#pragma unroll
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            for (int o = 16; o > 0; o >>= 1) vmax = sycl::fmax(
                vmax, dpct::experimental::permute_sub_group_by_xor(
                          0xffffffffu,
                          sycl::ext::oneapi::this_work_item::get_sub_group(),
                          vmax, o));
            const float vup = vmax > 0.0f ? 16384.0f / vmax : 0.0f, vdown = vmax * (1.0f / 16384.0f);
            float tmp[8][4];
#pragma unroll
            for (int j = 0; j < 8; ++j) tmp[j][0] = tmp[j][1] = tmp[j][2] = tmp[j][3] = 0.0f;
#pragma unroll
            for (int ks = 0; ks < CH / 16; ++ks) {
                const int cA = ks * 16 + 2 * tig, cB = cA + 8;
                const float w0 = S.vs[cA][warp] * vup, w1 = S.vs[cA + 1][warp] * vup, w2 = S.vs[cB][warp] * vup,
                            w3 = S.vs[cB + 1][warp] * vup;
                const float p00 = S.s[gid][cA] * w0, p01 = S.s[gid][cA + 1] * w1;
                const float p10 = S.s[gid + 8][cA] * w0, p11 = S.s[gid + 8][cA + 1] * w1;
                const float p02 = S.s[gid][cB] * w2, p03 = S.s[gid][cB + 1] * w3;
                const float p12 = S.s[gid + 8][cB] * w2, p13 = S.s[gid + 8][cB + 1] * w3;
                uint32_t ah[4], al[4];
                ah[0] = pack_h2(p00, p01);
                ah[1] = pack_h2(p10, p11);
                ah[2] = pack_h2(p02, p03);
                ah[3] = pack_h2(p12, p13);
                {
                    const sycl::half2 *h =
                        reinterpret_cast<const sycl::half2 *>(ah);
                    sycl::float2 f;
                    f = h[0].template convert<float,
                                              sycl::rounding_mode::automatic>();
                        al[0] = pack_h2(p00 - f.x(), p01 - f.y());
                    f = h[1].template convert<float,
                                              sycl::rounding_mode::automatic>();
                        al[1] = pack_h2(p10 - f.x(), p11 - f.y());
                    f = h[2].template convert<float,
                                              sycl::rounding_mode::automatic>();
                        al[2] = pack_h2(p02 - f.x(), p03 - f.y());
                    f = h[3].template convert<float,
                                              sycl::rounding_mode::automatic>();
                        al[3] = pack_h2(p12 - f.x(), p13 - f.y());
                }
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int d = warp * 64 + j * 8 + gid;
                    uint32_t b[2];
                    if constexpr (KV_MODE == 1) {
                        const uint32_t x0 = (uint8_t) S.v[cA][d] | ((uint32_t) (uint8_t) S.v[cA + 1][d] << 8);
                        const uint32_t x1 = (uint8_t) S.v[cB][d] | ((uint32_t) (uint8_t) S.v[cB + 1][d] << 8);
                        b[0] = i8x2_to_h2(x0);
                        b[1] = i8x2_to_h2(x1);
                    } else {
                        const sycl::half2 h0 =
                            sycl::half2(S.v[cA][d], S.v[cA + 1][d]);
                        const sycl::half2 h1 =
                            sycl::half2(S.v[cB][d], S.v[cB + 1][d]);
                        b[0] = *reinterpret_cast<const uint32_t*>(&h0);
                        b[1] = *reinterpret_cast<const uint32_t*>(&h1);
                    }
                    mma16816(tmp[j], ah, b);
#ifndef D1_NO_PLO
                    mma16816(tmp[j], al, b);
#endif
                }
            }
            const float a0 = S.alpha[gid], a1 = S.alpha[gid + 8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[j][0] = sycl::fma(acc[j][0], (float)a0, tmp[j][0] * vdown);
                acc[j][1] = sycl::fma(acc[j][1], (float)a0, tmp[j][1] * vdown);
                acc[j][2] = sycl::fma(acc[j][2], (float)a1, tmp[j][2] * vdown);
                acc[j][3] = sycl::fma(acc[j][3], (float)a1, tmp[j][3] * vdown);
            }
        }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const float l0 = S.lsum[gid], l1 = S.lsum[gid + 8];
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int d = warp * 64 + j * 8 + 2 * tig;
        *reinterpret_cast<sycl::float2 *>(attn + (size_t)gid * HD + d) =
            sycl::float2(acc[j][0] * i0, acc[j][1] * i0);
        if (gid + 8 < G)
            *reinterpret_cast<sycl::float2 *>(attn + (size_t)(gid + 8) * HD +
                                              d) =
                sycl::float2(acc[j][2] * i1, acc[j][3] * i1);
    }
}

// ---- v2 (int8 KV): warp w owns dims [64w, 64w+64) for both q.k and p.v, which is also int8 scale group w. So a
// warp needs only its own 64-byte slice of each K and V row: it gathers it itself with cp.async into its own
// double-buffered stage while it computes the previous chunk, and q stays in registers. Only the q.k partial sums
// (one per dim group) cross warps, and they are added in a fixed order: the result is deterministic.
constexpr int CH2 = 32;

struct Smem2 {
    int8_t kv[2][4][2][CH2][64];   // stage, warp, K/V, cell, 64 dims in 16-byte pieces XOR-swizzled by the cell
    float sc[2][4][2][CH2];        // stage, warp, K/V scale of the cell for the warp's group
    float part[4][16][CH2 + 1];    // q.k per dim group
    float p[16][CH2 + 1];
    float qmax[4];
    float alpha[16];
    float lsum[16];
    float mrow[16];
};

__dpct_inline__ int
swz(int cell, int byte) { // byte offset of (cell, byte) in a stage slice
    return cell * 64 + ((((byte >> 4) ^ (cell >> 1)) & 3) << 4) + (byte & 15);
}
__dpct_inline__ void cp_async16(void *smem, const void *gmem, bool valid) {
#if !STRATA_PA_SM80
    /* unreachable on SYCL: the launcher refuses this device */
#else
    auto sa = smem;
    /*
    DPCT1053: Migration of device assembly code is not supported.
    */
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa),
                 "l"(gmem), "r"(valid ? 16 : 0));
#endif
}
__dpct_inline__ void cp_async_commit() {
#if STRATA_PA_SM80
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    asm volatile("cp.async.commit_group;\n" ::);
#else

#endif
#endif
}
__dpct_inline__ void cp_async_wait1() {
#if STRATA_PA_SM80
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    asm volatile("cp.async.wait_group 1;\n" ::);
#else

#endif
#endif
}

/*
DPCT1110: The total declared local variable size in device function
prompt_attn_i8_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
prompt_attn_i8_kernel(const float *__restrict__ q, QsaAttnPools p,
                      const int32_t *__restrict__ ids,
                      const int32_t *__restrict__ steps, int n_kv_heads,
                      int page_size, float scale_log2, float *__restrict__ attn,
                      int cap, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto smem_raw = (unsigned char *)dpct_local;
    Smem2& S = *reinterpret_cast<Smem2*>(smem_raw);
    const int qi = item_ct1.get_group(2), kvh = item_ct1.get_group(1);
    const int n_head = n_kv_heads * G;
    q += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    attn += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    ids += (size_t) qi * cap;
    /*
    DPCT1098: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    const int n = *(steps + (size_t)qi * kStepCount + kStepWidth);
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int gid = lane >> 2, tig = lane & 3;
    const int dim0 = warp * 64;

    // q: the power-of-two prescale over all 12 heads (as v1), then this warp's 64 dims as hi/lo A fragments
    float qm = 0.0f;
#pragma unroll
    for (int i = t; i < G * HD; i += THREADS)
        qm = sycl::fmax(qm, sycl::fabs(q[i]));
#pragma unroll
    /*
    DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = 16; o > 0; o >>= 1) qm = sycl::fmax(
        qm, dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                qm, o));
    if (lane == 0) S.qmax[warp] = qm;
    if (t < 16) { S.mrow[t] = -INFINITY; S.lsum[t] = 0.0f; }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    qm = sycl::fmax(sycl::fmax(S.qmax[0], S.qmax[1]),
                    sycl::fmax(S.qmax[2], S.qmax[3]));
    int qe = 0;
    /*
    DPCT1017: The sycl::frexp call is used instead of the frexpf call. These
    two calls do not provide exactly the same functionality. Check the potential
    precision and/or performance issues for the generated code.
    */
    if (qm > 0.0f) sycl::frexp(
        qm, sycl::address_space_cast<sycl::access::address_space::generic_space,
                                     sycl::access::decorated::yes>(&qe));
    const float qup = sycl::ldexp(1.0f, 14 - qe),
                qdown = sycl::ldexp(scale_log2, qe - 14);
    uint32_t qh[4][4], ql[4][4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = gid + (r & 1) * 8, col = dim0 + kk * 16 + 2 * tig + (r >> 1) * 8;
            sycl::float2 x = sycl::float2(0.f, 0.f);
            if (row < G) x = *reinterpret_cast<const sycl::float2 *>(
                q + (size_t)row * HD + col);
            x.x() *= qup;
            x.y() *= qup;
            const sycl::half2 hi =
                sycl::float2(x.x(), x.y())
                    .convert<sycl::half, sycl::rounding_mode::rte>();
            const sycl::float2 hf =
                hi.template convert<float, sycl::rounding_mode::automatic>();
            const sycl::half2 lo =
                sycl::float2(x.x() - hf.x(), x.y() - hf.y())
                    .convert<sycl::half, sycl::rounding_mode::rte>();
            qh[kk][r] = *reinterpret_cast<const uint32_t*>(&hi);
            ql[kk][r] = *reinterpret_cast<const uint32_t*>(&lo);
        }
    }

    // the chunk pipeline: cells two chunks ahead, their pool rows one chunk ahead, the data (cp.async) one ahead
    const int n_chunks = (n + CH2 - 1) / CH2;
    /*
    DPCT1098: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    auto cell_of = [&](int c) -> int {
                                       return c < n ? *(ids + c) : -1;
    };
    auto row_of = [&](int cell) -> long long {
        if (cell < 0) return -1;
        /*
        DPCT1098: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        const long long page = (long long)*(p.page_table + cell / page_size);
        return (page * n_kv_heads + kvh) * page_size + (cell % page_size);
    };
    auto issue = [&](long long r, int st, float& ksr, float& vsr) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = lane + 32 * j, cell = idx >> 2, pc = idx & 3;
            /*
            DPCT1108: '__shfl_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            /*
            DPCT1121: Make sure that the "r" which is used in the SYCL group
            function/algorithm is initialized.
            */
            const long long rr = dpct::experimental::select_from_sub_group(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                r, cell);
            const bool ok = rr >= 0;
            const size_t off = ok ? (size_t) rr * HD + dim0 + pc * 16 : 0;
            cp_async16(&S.kv[st][warp][0][0][0] + swz(cell, pc * 16), p.k_q + off, ok);
            cp_async16(&S.kv[st][warp][1][0][0] + swz(cell, pc * 16), p.v_q + off, ok);
        }
        /*
        DPCT1098: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        ksr = r >= 0 ? sycl::vec<sycl::half, 1>(
                           sycl::bit_cast<sycl::half, unsigned short>(
                               *(p.k_scale + r * (HD / KV_Q8_GROUP) + warp)))
                           .convert<float, sycl::rounding_mode::automatic>()[0]
                     : 0.0f;
        /*
        DPCT1098: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        vsr = r >= 0 ? sycl::vec<sycl::half, 1>(
                           sycl::bit_cast<sycl::half, unsigned short>(
                               *(p.v_scale + r * (HD / KV_Q8_GROUP) + warp)))
                           .convert<float, sycl::rounding_mode::automatic>()[0]
                     : 0.0f;
    };
    float ksn, vsn;
    issue(row_of(cell_of(lane)), 0, ksn, vsn);
    cp_async_commit();
    S.sc[0][warp][0][lane] = ksn;
    S.sc[0][warp][1][lane] = vsn;
    long long r_next = row_of(cell_of(CH2 + lane));
    int cell_next2 = cell_of(2 * CH2 + lane);

    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;

    for (int ci = 0; ci < n_chunks; ++ci) {
        const int st = ci & 1, c0 = ci * CH2;
        const bool more = ci + 1 < n_chunks;
        if (more) issue(r_next, st ^ 1, ksn, vsn);
        cp_async_commit();
        r_next = row_of(cell_next2);
        cell_next2 = cell_of((ci + 3) * CH2 + lane);
        cp_async_wait1();
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
        const int8_t* K = &S.kv[st][warp][0][0][0];
        const int8_t* V = &S.kv[st][warp][1][0][0];
        // q.k over this warp's 64 dims, times the cell's K scale for this group
#pragma unroll
        for (int nt = 0; nt < CH2 / 8; ++nt) {
            float tg[4] = {0.f, 0.f, 0.f, 0.f};
            const int cell = nt * 8 + gid;
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                uint32_t b[2];
                b[0] = i8x2_to_h2(*reinterpret_cast<const uint16_t*>(K + swz(cell, kk * 16 + 2 * tig)));
                b[1] = i8x2_to_h2(*reinterpret_cast<const uint16_t*>(K + swz(cell, kk * 16 + 2 * tig + 8)));
                mma16816(tg, qh[kk], b);
                mma16816(tg, ql[kk], b);
            }
            const int c = nt * 8 + 2 * tig;
            const float s0 = S.sc[st][warp][0][c], s1 = S.sc[st][warp][0][c + 1];
            S.part[warp][gid][c] = tg[0] * s0;
            S.part[warp][gid][c + 1] = tg[1] * s1;
            S.part[warp][gid + 8][c] = tg[2] * s0;
            S.part[warp][gid + 8][c + 1] = tg[3] * s1;
        }
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        // online softmax over the four groups' sum (fixed order): row t/8, 4 cells per thread
        {
            const int r = t >> 3, sub = t & 7;
            float x[4], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = sub * 4 + j;
                x[j] = c0 + c < n ? (((S.part[0][r][c] + S.part[1][r][c]) + S.part[2][r][c]) + S.part[3][r][c]) * qdown
                                  : -INFINITY;
                mx = sycl::fmax(mx, x[j]);
            }
#pragma unroll
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            for (int o = 1; o < 8; o <<= 1) mx = sycl::fmax(
                mx,
                dpct::experimental::permute_sub_group_by_xor(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), mx, o));
            const float m_old = S.mrow[r];
            const float m_new = sycl::fmax(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float e =
                    x[j] == -INFINITY ? 0.0f : sycl::exp2(x[j] - m_new);
                S.p[r][sub * 4 + j] = e;
                sum += e;
            }
#pragma unroll
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            for (int o = 1; o < 8; o <<= 1) sum +=
                dpct::experimental::permute_sub_group_by_xor(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), sum, o);
            sycl::group_barrier(
                sycl::ext::oneapi::this_work_item::get_sub_group());
            if (sub == 0) {
                const float a =
                    m_old == -INFINITY ? 0.0f : sycl::exp2(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = sycl::fma(S.lsum[r], (float)a, sum);
                S.mrow[r] = m_new;
            }
        }
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        // p.v over this warp's 64 dims (as v1)
        {
            float vmax = S.sc[st][warp][1][lane];
#pragma unroll
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            for (int o = 16; o > 0; o >>= 1) vmax = sycl::fmax(
                vmax, dpct::experimental::permute_sub_group_by_xor(
                          0xffffffffu,
                          sycl::ext::oneapi::this_work_item::get_sub_group(),
                          vmax, o));
            const float vup = vmax > 0.0f ? 16384.0f / vmax : 0.0f, vdown = vmax * (1.0f / 16384.0f);
            float tmp[8][4];
#pragma unroll
            for (int j = 0; j < 8; ++j) tmp[j][0] = tmp[j][1] = tmp[j][2] = tmp[j][3] = 0.0f;
#pragma unroll
            for (int ks = 0; ks < CH2 / 16; ++ks) {
                const int cA = ks * 16 + 2 * tig, cB = cA + 8;
                const float w0 = S.sc[st][warp][1][cA] * vup, w1 = S.sc[st][warp][1][cA + 1] * vup,
                            w2 = S.sc[st][warp][1][cB] * vup, w3 = S.sc[st][warp][1][cB + 1] * vup;
                const float p00 = S.p[gid][cA] * w0, p01 = S.p[gid][cA + 1] * w1;
                const float p10 = S.p[gid + 8][cA] * w0, p11 = S.p[gid + 8][cA + 1] * w1;
                const float p02 = S.p[gid][cB] * w2, p03 = S.p[gid][cB + 1] * w3;
                const float p12 = S.p[gid + 8][cB] * w2, p13 = S.p[gid + 8][cB + 1] * w3;
                uint32_t ah[4], al[4];
                ah[0] = pack_h2(p00, p01);
                ah[1] = pack_h2(p10, p11);
                ah[2] = pack_h2(p02, p03);
                ah[3] = pack_h2(p12, p13);
                {
                    const sycl::half2 *h =
                        reinterpret_cast<const sycl::half2 *>(ah);
                    sycl::float2 f;
                    f = h[0].template convert<float,
                                              sycl::rounding_mode::automatic>();
                        al[0] = pack_h2(p00 - f.x(), p01 - f.y());
                    f = h[1].template convert<float,
                                              sycl::rounding_mode::automatic>();
                        al[1] = pack_h2(p10 - f.x(), p11 - f.y());
                    f = h[2].template convert<float,
                                              sycl::rounding_mode::automatic>();
                        al[2] = pack_h2(p02 - f.x(), p03 - f.y());
                    f = h[3].template convert<float,
                                              sycl::rounding_mode::automatic>();
                        al[3] = pack_h2(p12 - f.x(), p13 - f.y());
                }
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int d = j * 8 + gid;
                    const uint32_t x0 = (uint8_t) V[swz(cA, d)] | ((uint32_t) (uint8_t) V[swz(cA + 1, d)] << 8);
                    const uint32_t x1 = (uint8_t) V[swz(cB, d)] | ((uint32_t) (uint8_t) V[swz(cB + 1, d)] << 8);
                    uint32_t b[2];
                    b[0] = i8x2_to_h2(x0);
                    b[1] = i8x2_to_h2(x1);
                    mma16816(tmp[j], ah, b);
                    mma16816(tmp[j], al, b);
                }
            }
            const float a0 = S.alpha[gid], a1 = S.alpha[gid + 8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[j][0] = sycl::fma(acc[j][0], (float)a0, tmp[j][0] * vdown);
                acc[j][1] = sycl::fma(acc[j][1], (float)a0, tmp[j][1] * vdown);
                acc[j][2] = sycl::fma(acc[j][2], (float)a1, tmp[j][2] * vdown);
                acc[j][3] = sycl::fma(acc[j][3], (float)a1, tmp[j][3] * vdown);
            }
        }
        if (more) {
            S.sc[st ^ 1][warp][0][lane] = ksn;
            S.sc[st ^ 1][warp][1][lane] = vsn;
        }
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const float l0 = S.lsum[gid], l1 = S.lsum[gid + 8];
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int d = dim0 + j * 8 + 2 * tig;
        *reinterpret_cast<sycl::float2 *>(attn + (size_t)gid * HD + d) =
            sycl::float2(acc[j][0] * i0, acc[j][1] * i0);
        if (gid + 8 < G)
            *reinterpret_cast<sycl::float2 *>(attn + (size_t)(gid + 8) * HD +
                                              d) =
                sycl::float2(acc[j][2] * i1, acc[j][3] * i1);
    }
}

bool launch_i8(const float *q, const QsaAttnPools &pools, const int32_t *ids,
               const int32_t *steps, int64_t cap, const QsaShapes &s,
               float *attn, int64_t n_q, dpct::queue_ptr st) try {
    static bool attr[64] = {};   // the shared-memory opt-in is per device (a layer split runs this on several)
    int dev = 0;
    dev = dpct::get_current_device_id();
    /*
    DPCT1083: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    const int bytes = (int)sizeof(Smem2);
    if (dev < 0 || dev >= 64) return false;
    if (!attr[dev]) {
        /*
        DPCT1027: The call to cudaFuncSetAttribute was replaced with 0
        because SYCL currently does not support corresponding setting.
        */
        if (0 != 0) {
            /*
            DPCT1026: The call to cudaGetLastError was removed because this
            functionality is redundant in SYCL.
            */
            return false;
        }
        attr[dev] = true;
    }
    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};
            dpct::has_capability_or_fail(st->get_device(),
                                         {sycl::aspect::fp16});

            st->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range(bytes), cgh);

                auto q_q0_s_n_head_HD_ct0 = q + q0 * s.n_head * HD;
                auto ids_q0_cap_ct2 = ids + q0 * cap;
                auto steps_q0_kStepCount_ct3 = steps + q0 * kStepCount;
                auto attn_q0_s_n_head_HD_ct7 = attn + q0 * s.n_head * HD;

                cgh.parallel_for<
                    dpct_kernel_name<class prompt_attn_i8_kernel_c3d463>>(
                    sycl::nd_range<3>(
                        sycl::range(1, (unsigned)s.n_head_kv, (unsigned)nb) *
                            sycl::range(1, 1, THREADS),
                        sycl::range(1, 1, THREADS)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        32)]] {
                        prompt_attn_i8_kernel(
                            q_q0_s_n_head_HD_ct0, pools, ids_q0_cap_ct2,
                            steps_q0_kStepCount_ct3, (int)s.n_head_kv,
                            (int)s.page_size, scale_log2,
                            attn_q0_s_n_head_HD_ct7, (int)cap,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
        }
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;

    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

template <int KV_MODE>
bool launch(const float *q, const QsaAttnPools &pools, const int32_t *ids,
            const int32_t *steps, int64_t cap, const QsaShapes &s, float *attn,
            int64_t n_q, dpct::queue_ptr st) try {
    static bool attr[64] = {};   // per device, as above
    int dev = 0;
    dev = dpct::get_current_device_id();
    /*
    DPCT1083: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    const int bytes = (int)sizeof(Smem<KV_MODE>);
    if (dev < 0 || dev >= 64) return false;
    if (!attr[dev]) {
        /*
        DPCT1027: The call to cudaFuncSetAttribute was replaced with 0
        because SYCL currently does not support corresponding setting.
        */
        if (0 != 0) {
            /*
            DPCT1026: The call to cudaGetLastError was removed because this
            functionality is redundant in SYCL.
            */
            return false;
        }
        attr[dev] = true;
    }
    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};
            dpct::has_capability_or_fail(st->get_device(),
                                         {sycl::aspect::fp16});

            st->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range(bytes), cgh);

                auto q_q0_s_n_head_HD_ct0 = q + q0 * s.n_head * HD;
                auto ids_q0_cap_ct2 = ids + q0 * cap;
                auto steps_q0_kStepCount_ct3 = steps + q0 * kStepCount;
                auto attn_q0_s_n_head_HD_ct7 = attn + q0 * s.n_head * HD;

                cgh.parallel_for<
                    dpct_kernel_name<class prompt_attn_kernel_e09b15,
                                     dpct_kernel_scalar<KV_MODE>>>(
                    sycl::nd_range<3>(
                        sycl::range(1, (unsigned)s.n_head_kv, (unsigned)nb) *
                            sycl::range(1, 1, THREADS),
                        sycl::range(1, 1, THREADS)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        32)]] {
                        prompt_attn_kernel<KV_MODE>(
                            q_q0_s_n_head_HD_ct0, pools, ids_q0_cap_ct2,
                            steps_q0_kStepCount_ct3, (int)s.n_head_kv,
                            (int)s.page_size, scale_log2,
                            attn_q0_s_n_head_HD_ct7, (int)cap,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
        }
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;

    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

}  // namespace

bool qsa_prompt_attn_batch(const float *q, const QsaAttnPools &pools,
                           const int32_t *ids, const int32_t *steps,
                           int64_t cap, const QsaShapes &s, float *attn,
                           int64_t n_q, void *stream) try {
    if (n_q <= 0) return true;
    bool turing = false;   // per call, from the CURRENT device (a layer split can mix Turing with newer cards)
    {   // sm_75 or newer: the MMA above compiles for both.  sm_80+ runs the cp.async kernel (launch_i8); Turing has
        // no cp.async, so it runs the v1 kernel (launch<1>, same accuracy, another summation order).  An older card
        // keeps the old kernel.
        static int cc_major[64] = {};
        int dev = 0;
        /*
        DPCT1026: The call to cudaGetLastError was removed because this
        functionality is redundant in SYCL.
        */
        if (DPCT_CHECK_ERROR(dev = dpct::get_current_device_id()) != 0 ||
            dev < 0 || dev >= 64) {
            ; return false;
        }
        if (cc_major[dev] == 0) {
            int major = 0;
            if (DPCT_CHECK_ERROR(
                    major = dpct::get_device(dev).get_major_version()) != 0) {
                /*
                DPCT1026: The call to cudaGetLastError was removed because
                this functionality is redundant in SYCL.
                */
                return false;
            }
            cc_major[dev] = major;
        }
        (void) cc_major[dev];
        // SYCL: the mma.sync kernel below is not ported (upstream's Turing/sm_80 split - `turing` - does not
        // apply); its XMX port takes the same arguments
        if (qsa_prompt_attn_xmx(q, pools, ids, steps, cap, s, attn, n_q, stream)) return true;
        return false;   // the caller takes the older kernel
    }
#if defined(__HIPCC__)
    return false;   // the tensor-core kernel is compiled out on AMD (its major version is not a CUDA sm)
#endif
    if (pools.k_q4 != nullptr || s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !ids ||
        !steps || !pools.page_table)
        return false;
    dpct::queue_ptr st = strata::q_of(stream);
    if (pools.k_q != nullptr && pools.v_q4 != nullptr) {   // hybrid K8V4: int8 K + dequantized-q4 V
        if (!pools.k_scale) return false;
        return launch<3>(q, pools, ids, steps, cap, s, attn, n_q, st);
    }
    if (pools.k_q != nullptr) {
        if (!pools.v_q || !pools.k_scale || !pools.v_scale) return false;
        // STRATA_PROMPT_ATTN_V1=1 (debug): the first version, same accuracy, another summation order - the control
        // for how far the model amplifies an FP32-level change.  Turing always takes it: v2's cp.async does not
        // exist before sm_80.
        static const bool v1 = std::getenv("STRATA_PROMPT_ATTN_V1") != nullptr;
        if (v1 || turing) return launch<1>(q, pools, ids, steps, cap, s, attn, n_q, st);
        return launch_i8(q, pools, ids, steps, cap, s, attn, n_q, st);
    }
    if (!pools.k_pool || !pools.v_pool) return false;
    return launch<0>(q, pools, ids, steps, cap, s, attn, n_q, st);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

}  // namespace strata::kernels
