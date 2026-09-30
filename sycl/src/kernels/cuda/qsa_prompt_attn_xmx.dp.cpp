// SYCL port: the prompt attention on XMX (joint_matrix) - the port of qsa_prompt_attn.cu's tensor-core kernel.
//
// One work-group per (query position, KV head): 16 query rows (the 12 heads of the group, 4 zero rows) against the
// position's selected cells, 16 cells per chunk. Scores S = Q K^T and the update O += P V run on joint_matrix
// (FP16 x FP16 -> FP32, 16x16x16, sub-group 16); Q and P are split into hi + lo FP16 halves exactly as the CUDA kernel
// does, and the per-64-dim INT8 scales of K and V are applied outside the matrix products the same way (K's after the
// group's product, V's folded into P). The online softmax and the accumulator merge are scalar over local memory.
// KV modes: 0 FP16 K/V, 1 INT8 K/V with scales, 3 INT8 K + q4_0 V dequantized to FP16.
#include "strata/kernels/qsa_prompt_attn_xmx.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/sycl_queue.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <dpct/dpct.hpp>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace strata::kernels {
namespace {
namespace jm = sycl::ext::oneapi::experimental::matrix;
constexpr int HD = 256, G = 12, CH = 16, SG = 16, NSG = 8, THREADS = SG * NSG, QS = HD + 8, PS = CH + 8;
constexpr int QK4_0 = 32;
struct q4_0_block { uint16_t d; uint8_t qs[QK4_0 / 2]; };

struct Smem {
    sycl::half* qh;      // [16][QS]
    sycl::half* ql;      // [16][QS]
    sycl::half* kp;      // packed K^T: (k/2)*(CH*2) + c*2 + (k&1), k < HD, c < CH
    sycl::half* vp;      // packed V:   (c/2)*(HD*2) + d*2 + (c&1)
    sycl::half* pv;      // [4 groups][2 hi/lo][16][PS] P tiles
    float* ks;           // [CH][4]
    float* vs;           // [CH][4]
    float* s;            // [16][PS] scores, then probabilities
    float* acc;          // [16][HD]
    float* tmp;          // [NSG][16][16]
    float* misc;         // qmax[8], mrow[16], lsum[16], alpha[16], vup[4], vdown[4]
    long long* row;      // [CH]
};
constexpr size_t kSmemBytes = (size_t) 2 * 16 * QS * 2 + (size_t) 2 * CH * HD * 2 + (size_t) 4 * 2 * 16 * PS * 2 +
                              (size_t) 2 * CH * 4 * 4 + (size_t) 16 * PS * 4 + (size_t) 16 * HD * 4 + (size_t) NSG * 256 * 4 +
                              (size_t) 72 * 4 + (size_t) CH * 8 + 64;

__dpct_inline__ Smem carve(uint8_t* base) {
    Smem m;
    uint8_t* p = base;
    auto take = [&](size_t bytes) { uint8_t* r = p; p += (bytes + 15) & ~(size_t) 15; return r; };
    m.qh = (sycl::half*) take(16 * QS * 2);
    m.ql = (sycl::half*) take(16 * QS * 2);
    m.kp = (sycl::half*) take((size_t) CH * HD * 2);
    m.vp = (sycl::half*) take((size_t) CH * HD * 2);
    m.pv = (sycl::half*) take((size_t) 4 * 2 * 16 * PS * 2);
    m.ks = (float*) take(CH * 4 * 4);
    m.vs = (float*) take(CH * 4 * 4);
    m.s = (float*) take(16 * PS * 4);
    m.acc = (float*) take((size_t) 16 * HD * 4);
    m.tmp = (float*) take((size_t) NSG * 256 * 4);
    m.misc = (float*) take(72 * 4);
    m.row = (long long*) take(CH * 8);
    return m;
}

template <typename T>
__dpct_inline__ auto lptr(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p);
}

__dpct_inline__ float sg_max(sycl::sub_group sg, float v) {
    return sycl::reduce_over_group(sg, v, sycl::maximum<float>());
}

template <int KV_MODE>
void prompt_attn_xmx_kernel(const float* __restrict__ q, QsaAttnPools p, const int32_t* __restrict__ ids,
                            const int32_t* __restrict__ steps, int n_kv_heads, int page_size, float scale_log2,
                            float* __restrict__ attn, int cap, uint8_t* smem) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto sg = it.get_sub_group();
    Smem S = carve(smem);
    const int qi = (int) it.get_group(2), kvh = (int) it.get_group(1);
    const int n_head = n_kv_heads * G;
    q += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    attn += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    ids += (size_t) qi * cap;
    const int n = steps[(size_t) qi * kStepCount + kStepWidth];
    const int t = (int) it.get_local_id(2), lane = (int) sg.get_local_id()[0], sgid = (int) sg.get_group_id()[0];
    float* qmax = S.misc;
    float* mrow = S.misc + 8;
    float* lsum = S.misc + 24;
    float* alpha = S.misc + 40;
    float* vup = S.misc + 56;
    float* vdown = S.misc + 60;
    // q: 12 heads + 4 zero rows, scaled so its largest value sits near 2^14 (exact), split into hi + lo halves
    float qm = 0.0f;
    for (int i = t; i < G * HD; i += THREADS) qm = sycl::fmax(qm, sycl::fabs(q[i]));
    qm = sg_max(sg, qm);
    if (lane == 0) qmax[sgid] = qm;
    it.barrier(sycl::access::fence_space::local_space);
    qm = 0.0f;
    for (int i = 0; i < NSG; ++i) qm = sycl::fmax(qm, qmax[i]);
    int qe = 0;   // frexp's exponent (qm < 2^qe) from the float's bits: normal numbers only, qm > 0 is
    if (qm > 0.0f) qe = (int) ((sycl::bit_cast<uint32_t>(qm) >> 23) & 0xFFu) - 126;
    const float qup = sycl::ldexp(1.0f, 14 - qe), qdown = sycl::ldexp(scale_log2, qe - 14);
    for (int i = t; i < 16 * HD; i += THREADS) {
        const int h = i / HD, d = i % HD;
        const float x = h < G ? q[(size_t) h * HD + d] * qup : 0.0f;
        const sycl::half hi = sycl::half(x);
        S.qh[h * QS + d] = hi;
        S.ql[h * QS + d] = sycl::half(x - (float) hi);
    }
    for (int i = t; i < 16 * HD; i += THREADS) S.acc[i] = 0.0f;
    if (t < 16) { mrow[t] = -INFINITY; lsum[t] = 0.0f; }
    // the K^T tile is the same for every chunk position: B(k, c) packed by k pairs
    for (int c0 = 0; c0 < n; c0 += CH) {
        const int nh = sycl::min(CH, n - c0);
        it.barrier(sycl::access::fence_space::local_space);   // the previous chunk is done with kp, vp, s, tmp
        if (t < CH) {
            long long r = -1;
            if (t < nh) {
                const int cell = ids[c0 + t];
                const long long page = (long long) p.page_table[cell / page_size];
                r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
            }
            S.row[t] = r;
        }
        it.barrier(sycl::access::fence_space::local_space);
        // K rows -> packed K^T (raw INT8 codes as halves in modes 1 and 3; the scale comes after the product)
        for (int i = t; i < CH * (HD / 8); i += THREADS) {
            const int c = i / (HD / 8), piece = i % (HD / 8);
            const long long r = S.row[c];
            sycl::half h[8];
            if (r < 0) {
                for (int j = 0; j < 8; ++j) h[j] = sycl::half(0.0f);
            } else if constexpr (KV_MODE == 0) {
                const sycl::vec<uint16_t, 8> v = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(p.k_pool + r * HD + piece * 8);
                for (int j = 0; j < 8; ++j) h[j] = sycl::bit_cast<sycl::half>(v[j]);
            } else {
                const sycl::vec<int8_t, 8> v = *reinterpret_cast<const sycl::vec<int8_t, 8>*>(p.k_q + r * HD + piece * 8);
                for (int j = 0; j < 8; ++j) h[j] = sycl::half((float) v[j]);
            }
            for (int j = 0; j < 8; j += 2) {
                const int k = piece * 8 + j;
                sycl::half2* dst = reinterpret_cast<sycl::half2*>(S.kp + (k >> 1) * (CH * 2) + c * 2);
                *dst = sycl::half2(h[j], h[j + 1]);
            }
        }
        // V rows -> packed V (pairs of cells)
        if constexpr (KV_MODE == 3) {
            constexpr int BLKS = HD / QK4_0;
            constexpr int BYTES = BLKS * (int) sizeof(q4_0_block);
            for (int i = t; i < CH * BLKS; i += THREADS) {
                const int c = i / BLKS, b = i % BLKS;
                const long long r = S.row[c];
                float d = 0.0f;
                const q4_0_block* blk = nullptr;
                if (r >= 0) {
                    blk = reinterpret_cast<const q4_0_block*>(p.v_q4 + r * BYTES) + b;
                    d = (float) sycl::bit_cast<sycl::half>(blk->d);
                }
                for (int j = 0; j < QK4_0 / 2; ++j) {
                    const float lo = blk ? (float) ((int) (blk->qs[j] & 0x0F) - 8) * d : 0.0f;
                    const float hi = blk ? (float) ((int) (blk->qs[j] >> 4) - 8) * d : 0.0f;
                    const int d0 = b * QK4_0 + j, d1 = d0 + QK4_0 / 2;
                    S.vp[(c >> 1) * (HD * 2) + d0 * 2 + (c & 1)] = sycl::half(lo);
                    S.vp[(c >> 1) * (HD * 2) + d1 * 2 + (c & 1)] = sycl::half(hi);
                }
            }
        } else {
            for (int i = t; i < CH * (HD / 8); i += THREADS) {
                const int c = i / (HD / 8), piece = i % (HD / 8);
                const long long r = S.row[c];
                sycl::half h[8];
                if (r < 0) {
                    for (int j = 0; j < 8; ++j) h[j] = sycl::half(0.0f);
                } else if constexpr (KV_MODE == 1) {
                    const sycl::vec<int8_t, 8> v = *reinterpret_cast<const sycl::vec<int8_t, 8>*>(p.v_q + r * HD + piece * 8);
                    for (int j = 0; j < 8; ++j) h[j] = sycl::half((float) v[j]);
                } else {
                    const sycl::vec<uint16_t, 8> v = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(p.v_pool + r * HD + piece * 8);
                    for (int j = 0; j < 8; ++j) h[j] = sycl::bit_cast<sycl::half>(v[j]);
                }
                for (int j = 0; j < 8; ++j) S.vp[(c >> 1) * (HD * 2) + (piece * 8 + j) * 2 + (c & 1)] = h[j];
            }
        }
        for (int i = t; i < CH * 4; i += THREADS) {
            const int c = i / 4, g = i % 4;
            const long long r = S.row[c];
            float a = 0.0f, b = 0.0f;
            if (r >= 0) {
                if constexpr (KV_MODE == 1) {
                    a = (float) sycl::bit_cast<sycl::half>(p.k_scale[r * (HD / KV_Q8_GROUP) + g]);
                    b = (float) sycl::bit_cast<sycl::half>(p.v_scale[r * (HD / KV_Q8_GROUP) + g]);
                } else if constexpr (KV_MODE == 3) {
                    a = (float) sycl::bit_cast<sycl::half>(p.k_scale[r * (HD / KV_Q8_GROUP) + g]);
                    b = 1.0f;
                } else {
                    a = b = 1.0f;
                }
            }
            S.ks[c * 4 + g] = a;
            S.vs[c * 4 + g] = b;
        }
        it.barrier(sycl::access::fence_space::local_space);
        // scores: sub-group sg takes dim group g = sg & 3 with the hi (sg < 4) or lo (sg >= 4) half of q
        {
            const int g = sgid & 3;
            const sycl::half* qa = (sgid < 4 ? S.qh : S.ql);
            jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 16, 16> C;
            jm::joint_matrix_fill(sg, C, 0.0f);
            for (int kk = 0; kk < 4; ++kk) {
                const int k0 = (g * 4 + kk) * 16;
                jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, 16, 16, jm::layout::row_major> A;
                jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, 16, jm::layout::ext_intel_packed> B;
                jm::joint_matrix_load(sg, A, lptr(qa + k0), (size_t) QS);
                jm::joint_matrix_load(sg, B, lptr(S.kp + (k0 >> 1) * (CH * 2)), (size_t) (CH * 2));
                jm::joint_matrix_mad(sg, C, A, B, C);
            }
            jm::joint_matrix_store(sg, C, lptr(S.tmp + sgid * 256), (size_t) 16, jm::layout::row_major);
        }
        it.barrier(sycl::access::fence_space::local_space);
        for (int i = t; i < 16 * CH; i += THREADS) {
            const int r = i / CH, c = i % CH;
            float v = 0.0f;
            for (int g = 0; g < 4; ++g) v += (S.tmp[g * 256 + r * 16 + c] + S.tmp[(g + 4) * 256 + r * 16 + c]) * S.ks[c * 4 + g];
            S.s[r * PS + c] = c < nh ? v * qdown : -INFINITY;
        }
        it.barrier(sycl::access::fence_space::local_space);
        // online softmax: row t/8, 2 cells per thread, 8 lanes per row (within one sub-group)
        {
            constexpr int PER = CH / 8;
            const int r = t >> 3, sub = t & 7;
            float x[PER], mx = -INFINITY;
            for (int j = 0; j < PER; ++j) { x[j] = S.s[r * PS + sub * PER + j]; mx = sycl::fmax(mx, x[j]); }
            for (int o = 1; o < 8; o <<= 1) mx = sycl::fmax(mx, sycl::permute_group_by_xor(sg, mx, o));
            const float m_old = mrow[r];
            const float m_new = sycl::fmax(m_old, mx);
            float sum = 0.0f;
            for (int j = 0; j < PER; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : sycl::exp2(x[j] - m_new);
                S.s[r * PS + sub * PER + j] = e;
                sum += e;
            }
            for (int o = 1; o < 8; o <<= 1) sum += sycl::permute_group_by_xor(sg, sum, o);
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : sycl::exp2(m_old - m_new);
                alpha[r] = a;
                lsum[r] = sycl::fma(lsum[r], a, sum);
                mrow[r] = m_new;
            }
        }
        // the V scale per dim group, folded into P relative to the chunk's largest (times 2^14, as in CUDA)
        if (t < 4) {
            float vmax = 0.0f;
            for (int c = 0; c < CH; ++c) vmax = sycl::fmax(vmax, S.vs[c * 4 + t]);
            vup[t] = vmax > 0.0f ? 16384.0f / vmax : 0.0f;
            vdown[t] = vmax * (1.0f / 16384.0f);
        }
        it.barrier(sycl::access::fence_space::local_space);
        // P tiles: pv[g][hi/lo][r][c] = s[r][c] * vs[c][g] * vup[g], split into hi + lo halves
        for (int i = t; i < 4 * 16 * CH; i += THREADS) {
            const int g = i / (16 * CH), rc = i % (16 * CH), r = rc / CH, c = rc % CH;
            const float pw = S.s[r * PS + c] * S.vs[c * 4 + g] * vup[g];
            const sycl::half hi = sycl::half(pw);
            S.pv[((g * 2 + 0) * 16 + r) * PS + c] = hi;
            S.pv[((g * 2 + 1) * 16 + r) * PS + c] = sycl::half(pw - (float) hi);
        }
        it.barrier(sycl::access::fence_space::local_space);
        // p.v: sub-group sg owns dims [32 sg, 32 sg + 32) = two B tiles of scale group sg / 2
        {
            const int g = sgid >> 1;
            const sycl::half* ah = S.pv + ((g * 2 + 0) * 16) * PS;
            const sycl::half* al = S.pv + ((g * 2 + 1) * 16) * PS;
            jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, 16, 16, jm::layout::row_major> Ah, Al;
            jm::joint_matrix_load(sg, Ah, lptr(ah), (size_t) PS);
            jm::joint_matrix_load(sg, Al, lptr(al), (size_t) PS);
            const float a_vdown = vdown[g];
            for (int dt = 0; dt < 2; ++dt) {
                const int d0 = sgid * 32 + dt * 16;
                jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, 16, jm::layout::ext_intel_packed> B;
                jm::joint_matrix_load(sg, B, lptr(S.vp + d0 * 2), (size_t) (HD * 2));
                jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 16, 16> C;
                jm::joint_matrix_fill(sg, C, 0.0f);
                jm::joint_matrix_mad(sg, C, Ah, B, C);
                jm::joint_matrix_mad(sg, C, Al, B, C);
                jm::joint_matrix_store(sg, C, lptr(S.tmp + sgid * 256), (size_t) 16, jm::layout::row_major);
                sycl::group_barrier(sg);
                for (int i = lane; i < 256; i += SG) {
                    const int r = i >> 4, c = i & 15;
                    float* a = S.acc + r * HD + d0 + c;
                    *a = sycl::fma(*a, alpha[r], S.tmp[sgid * 256 + i] * a_vdown);
                }
                sycl::group_barrier(sg);
            }
        }
    }
    it.barrier(sycl::access::fence_space::local_space);
    for (int i = t; i < G * HD; i += THREADS) {
        const int h = i / HD, d = i % HD;
        const float l = lsum[h];
        attn[(size_t) h * HD + d] = l > 0.0f ? S.acc[h * HD + d] / l : 0.0f;
    }
}

template <int KV_MODE>
bool launch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
            const QsaShapes& s, float* attn, int64_t n_q, dpct::queue_ptr st) {
    const float scale_log2 = 1.4426950408889634f / std::sqrt((float) HD);
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        const float* qq = q + q0 * s.n_head * HD;
        const int32_t* idq = ids + q0 * cap;
        const int32_t* stq = steps + q0 * kStepCount;
        float* aq = attn + q0 * s.n_head * HD;
        const int nkv = (int) s.n_head_kv, ps = (int) s.page_size, capi = (int) cap;
        st->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<uint8_t, 1> slm(sycl::range<1>(kSmemBytes), cgh);
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, (size_t) nkv, (size_t) nb * THREADS), sycl::range<3>(1, 1, THREADS)),
                             [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(16)]] {
                                 prompt_attn_xmx_kernel<KV_MODE>(qq, pools, idq, stq, nkv, ps, scale_log2, aq, capi,
                                                                 slm.get_multi_ptr<sycl::access::decorated::no>().get());
                             });
        });
    }
    return true;
}
}  // namespace

bool qsa_prompt_attn_xmx(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                         const QsaShapes& s, float* attn, int64_t n_q, void* stream) {
    // Off unless asked for: correct (qsa_prompt_attn_parity: 1e-6 of scale, like the CUDA kernel) but 3x slower than the
    // FP32 fallback on the B70 (18.7 vs 5.5 ms per chunk at 32k context, 2026-09-30) - the chunk staging and packing
    // dominate, not the products. Kept as the base for a tuned version (bigger chunks, vector packing).
    static const bool on = std::getenv("STRATA_PROMPT_ATTN_XMX") && std::getenv("STRATA_PROMPT_ATTN_XMX")[0] == '1';
    if (!on || n_q <= 0) return false;
    if (pools.k_q4 != nullptr || s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !ids || !steps ||
        !pools.page_table)
        return false;
    dpct::queue_ptr st = strata::q_of(stream);
    static const bool fits = [&] {
        const size_t have = st->get_device().get_info<sycl::info::device::local_mem_size>();
        if (have < kSmemBytes) std::fprintf(stderr, "strata: prompt attention on XMX needs %zu bytes of local memory, the device has %zu\n", kSmemBytes, have);
        return have >= kSmemBytes;
    }();
    if (!fits) return false;
    if (pools.k_q != nullptr && pools.v_q4 != nullptr) {
        if (!pools.k_scale) return false;
        return launch<3>(q, pools, ids, steps, cap, s, attn, n_q, st);
    }
    if (pools.k_q != nullptr) {
        if (!pools.v_q || !pools.k_scale || !pools.v_scale) return false;
        return launch<1>(q, pools, ids, steps, cap, s, attn, n_q, st);
    }
    if (!pools.k_pool || !pools.v_pool) return false;
    return launch<0>(q, pools, ids, steps, cap, s, attn, n_q, st);
}
}  // namespace strata::kernels
