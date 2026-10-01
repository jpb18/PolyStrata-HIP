// src/kernels/cuda/qsa_select.cu - see include/strata/kernels/qsa_select.hpp.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/core/emulate.hpp"
#include <cstdlib>
#include <cstring>
#include "strata/kernels/qsa_select.hpp"

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {
namespace {

constexpr int IDX_DIM = 128, IDX_HEADS = 4, R = 4;
constexpr int SCORE_WARPS = 8;
constexpr int TOPK_T = 256;

__dpct_inline__ uint32_t order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    const uint32_t b = sycl::bit_cast<unsigned int>(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__dpct_inline__ void block_scores_kernel(const float *__restrict__ pooled,
                                         const float *__restrict__ dead,
                                         const float *__restrict__ q_idx,
                                         const int32_t *__restrict__ steps,
                                         int64_t max_blocks,
                                         float *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t qi = item_ct1.get_group(1);
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid];
    const int64_t b = (int64_t)item_ct1.get_group(2) * SCORE_WARPS +
                      (item_ct1.get_local_id(2) >> 5);
    if (b > n_bid || b >= max_blocks) return;
    const int lane = item_ct1.get_local_id(2) & 31;
    const float* key = (b == n_bid) ? dead : pooled + b * IDX_DIM;
    const sycl::float4 k4 =
        *reinterpret_cast<const sycl::float4 *>(key + lane * 4);
    const float* q = q_idx + qi * IDX_HEADS * IDX_DIM + lane * 4;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < IDX_HEADS; ++h) {
        const sycl::float4 q4 =
            *reinterpret_cast<const sycl::float4 *>(q + h * IDX_DIM);
        float d = k4.x() * q4.x() + k4.y() * q4.y() + k4.z() * q4.z() +
                  k4.w() * q4.w();
#pragma unroll
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        for (int o = 16; o > 0; o >>= 1) d +=
            dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                d, o);
        score += d > 0.0f ? d : 0.0f;
    }
    if (lane == 0) {
        if (b == n_bid && n_kv % R != 0) score += 1e9f;
        out[qi * max_blocks + b] = score;
    }
}

/*
DPCT1110: The total declared local variable size in device function
block_topk_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void block_topk_kernel(const float *__restrict__ scores,
                                       const int32_t *__restrict__ steps,
                                       int64_t max_blocks, int64_t cap,
                                       int32_t *__restrict__ ids) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &hist = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[256]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_a =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[TOPK_T]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_b =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[TOPK_T]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_digit = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_above = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int64_t qi = item_ct1.get_group(2);
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = item_ct1.get_local_id(2);
    if (n_kv <= width) {                               // everything is selected: the identity, ascending
#pragma unroll
        for (int64_t j = t; j < n_kv; j += TOPK_T) out[j] = (int32_t)j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;                      // blocks 0..n_bid, the last possibly empty
    const int64_t per = (nb + TOPK_T - 1) / TOPK_T;
    const int64_t b0 = (int64_t) t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    // ---- radix select: the largest key thr with (cells with key >= thr) >= width, 8 bits at a time
    uint32_t prefix = 0;
    int above = 0;                                     // cells strictly above the digits fixed so far
    for (int shift = 24; shift >= 0; shift -= 8) {
#pragma unroll
        for (int i = t; i < 256; i += TOPK_T) hist[i] = 0;
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
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int64_t b = b0; b < b1; ++b) {
            const int w = weight(b);
            if (w == 0) continue;
            const uint32_t k = order_key(sc[b]);
            if ((k & hi_mask) == (prefix & hi_mask)) dpct::atomic_fetch_add<
                sycl::access::address_space::generic_space>(
                &hist[(k >> shift) & 255], w);
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
        if (t == 0) {
            int cum = above, d = 255;
#pragma unroll
            for (; d > 0; --d) {
                if (cum + hist[d] >= width) break;
                cum += hist[d];
            }
            s_digit = d;
            s_above = cum;
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
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
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
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;          // cells equal to thr that fit, lowest index first
    // ---- per-thread counts of cells above and at the threshold, then their exclusive prefixes
    int gt = 0, eq = 0;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
    s_a[t] = gt;
    s_b[t] = eq;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (t == 0) {
        int ag = 0, ae = 0;
        for (int i = 0; i < TOPK_T; ++i) {
            const int g = s_a[i], e = s_b[i];
            s_a[i] = ag; s_b[i] = ae;
            ag += g; ae += e;
        }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const int64_t eq_before = s_b[t];
    int64_t my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + (int) my_eq;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    s_a[t] = sel;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (t == 0) {
        int a = 0;
        for (int i = 0; i < TOPK_T; ++i) { const int c = s_a[i]; s_a[i] = a; a += c; }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    int64_t wpos = s_a[t];
    int64_t eq_left = my_eq;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) {
#pragma unroll
            for (int c = 0; c < w; ++c) out[wpos++] = (int32_t)(b * R + c);
        } else if (k == thr) {
#pragma unroll
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left)
                out[wpos++] = (int32_t)(b * R + c);
        }
    }
}


// ---- QSA select on tensor cores (perf-review, after D-1): the block scores of many queries are one GEMM,
// rows (query, indexer head) x columns (blocks), K = 128, with relu per head summed. 3xTF32 (each operand split
// into a TF32 hi and lo part, hi*hi + hi*lo + lo*hi) keeps FP32-level accuracy; the summation order differs from
// the warp kernel, so a score can move in its last bits and a near-tie can select differently (not bitwise).
// Blocks < n_bid only; the tail block n_bid (the `dead` key, +1e9) is scored by the warp kernel's own code.
constexpr int TC_QT = 16;                 // queries per CTA (one m16 tile per indexer head)
constexpr int TC_NB = 32;                 // blocks per tile (4 warps x n8)
constexpr int TC_ITER = 4;                // tiles per CTA (the query tile is loaded once)
constexpr int TC_QS = IDX_HEADS * IDX_DIM + 4;   // query row stride in floats
constexpr int TC_KS = IDX_DIM + 4;               // key row stride

// TF32 conversion and MMA need sm_80: below it they compile to a trap and qsa_block_scores_tc refuses the device
#if defined(__HIPCC__)          // AMD: no mma.sync / cp.async; the host keeps the warp kernel (below)
#define STRATA_SEL_SM80 0
#elif 0   // SYCL: inline PTX (mma/ldmatrix/cp.async) - the XMX port is pending; see tools/fixups.py
#define STRATA_SEL_SM80 1
#else
#define STRATA_SEL_SM80 0
#endif
__dpct_inline__ uint32_t tf32_hi(float x) {
#if STRATA_SEL_SM80
    uint32_t r;
    /*
    DPCT1053: Migration of device assembly code is not supported.
    */
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
#else
    return sycl::bit_cast<uint32_t>(x);
#endif
}
__dpct_inline__ void mma_tf32(float *c, const uint32_t *a, const uint32_t *b) {
#if !STRATA_SEL_SM80
    /* unreachable on SYCL: the launcher refuses this device */
#else
    /*
    DPCT1053: Migration of device assembly code is not supported.
    */
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
                   "r"(b[1]));
#endif
}

/*
DPCT1110: The total declared local variable size in device function
block_scores_tc_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void block_scores_tc_kernel(
    const float *__restrict__ pooled, const float *__restrict__ q_idx,
    const int32_t *__restrict__ steps, int64_t nq, int64_t max_blocks,
    int64_t reach, float *__restrict__ out, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto sm = (float *)dpct_local;
    float* sQ = sm;                                   // [TC_QT][TC_QS]
    float* sK = sm + TC_QT * TC_QS;                   // [TC_NB][TC_KS]
    auto &s_nbid =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[TC_QT]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5,
              gid = lane >> 2, tig = lane & 3;
    const int64_t q0 = (int64_t)item_ct1.get_group(1) * TC_QT;
    for (int i = t; i < TC_QT * IDX_HEADS * IDX_DIM / 4; i += 128) {
        const int r = i / (IDX_HEADS * IDX_DIM / 4), c = i % (IDX_HEADS * IDX_DIM / 4);
        sycl::float4 v = sycl::float4(0.f, 0.f, 0.f, 0.f);
        if (q0 + r < nq) v = reinterpret_cast<const sycl::float4 *>(
            q_idx + (q0 + r) * IDX_HEADS * IDX_DIM)[c];
        *reinterpret_cast<sycl::float4 *>(sQ + r * TC_QS + c * 4) = v;
    }
    if (t < TC_QT) s_nbid[t] = q0 + t < nq ? steps[(q0 + t) * kStepCount + kStepNBid] : 0;
    int lo_nbid = 0x7fffffff, hi_nbid = 0;
    item_ct1.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int i = 0; i < TC_QT; ++i) {
        if (q0 + i >= nq) break;
        lo_nbid = sycl::min(lo_nbid, s_nbid[i]);
        hi_nbid = sycl::max(hi_nbid, s_nbid[i]);
    }
    for (int it = 0; it < TC_ITER; ++it) {
        const int64_t b0 =
            ((int64_t)item_ct1.get_group(2) * TC_ITER + it) * TC_NB;
        if (b0 >= reach || b0 >= hi_nbid) break;      // blocks >= every query's n_bid: nothing to score
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier(); // the previous tile's reads are done
        for (int i = t; i < TC_NB * IDX_DIM / 4; i += 128) {
            const int r = i / (IDX_DIM / 4), c = i % (IDX_DIM / 4);
            sycl::float4 v = sycl::float4(0.f, 0.f, 0.f, 0.f);
            if (b0 + r < hi_nbid) v = reinterpret_cast<const sycl::float4 *>(
                pooled + (b0 + r) * IDX_DIM)[c];
            *reinterpret_cast<sycl::float4 *>(sK + r * TC_KS + c * 4) = v;
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
        float acc[IDX_HEADS][4];
#pragma unroll
        for (int h = 0; h < IDX_HEADS; ++h) acc[h][0] = acc[h][1] = acc[h][2] = acc[h][3] = 0.f;
        const float* kr = sK + (warp * 8 + gid) * TC_KS;
#pragma unroll 4
        for (int k0 = 0; k0 < IDX_DIM; k0 += 8) {
            const float kx0 = kr[k0 + tig], kx1 = kr[k0 + tig + 4];
            uint32_t bh[2], bl[2];
            bh[0] = tf32_hi(kx0);
            bh[1] = tf32_hi(kx1);
            bl[0] = tf32_hi(kx0 - sycl::bit_cast<float>(bh[0]));
            bl[1] = tf32_hi(kx1 - sycl::bit_cast<float>(bh[1]));
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const float* qa = sQ + h * IDX_DIM + k0 + tig;
                const float x0 = qa[gid * TC_QS], x1 = qa[(gid + 8) * TC_QS];
                const float x2 = qa[gid * TC_QS + 4], x3 = qa[(gid + 8) * TC_QS + 4];
                uint32_t ah[4], al[4];
                ah[0] = tf32_hi(x0); ah[1] = tf32_hi(x1); ah[2] = tf32_hi(x2); ah[3] = tf32_hi(x3);
                al[0] = tf32_hi(x0 - sycl::bit_cast<float>(ah[0]));
                al[1] = tf32_hi(x1 - sycl::bit_cast<float>(ah[1]));
                al[2] = tf32_hi(x2 - sycl::bit_cast<float>(ah[2]));
                al[3] = tf32_hi(x3 - sycl::bit_cast<float>(ah[3]));
                mma_tf32(acc[h], al, bh);
                mma_tf32(acc[h], ah, bl);
                mma_tf32(acc[h], ah, bh);
            }
        }
        // relu per head, heads added in order (as the warp kernel), written where the block completed for the query
        const int64_t bc = b0 + warp * 8 + 2 * tig;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int qr = gid + half * 8;
            const int64_t qi = q0 + qr;
            if (qi >= nq) continue;
            const int nb = s_nbid[qr];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int64_t b = bc + j;
                if (b >= nb || b >= max_blocks) continue;
                float score = 0.0f;
#pragma unroll
                for (int h = 0; h < IDX_HEADS; ++h) {
                    const float d = acc[h][half * 2 + j];
                    score += d > 0.0f ? d : 0.0f;
                }
                out[qi * max_blocks + b] = score;
            }
        }
    }
}

#if defined(__HIPCC__)
// ---- gfx12 (RDNA4: gfx1200 / gfx1201) scorer on v_wmma_f32_16x16x16_bf16 (wave32).  Same GEMM as above: for a tile of 16
// queries, every block's score is sum over the 4 indexer heads of relu(q_h . k), K = 128.  FP32-level accuracy from a
// three-way bf16 split of both operands (x = hi + mid + lo EXACTLY - the top 8, next 8 and last 8 bits of the fp32
// mantissa, so bf16's fp32-sized exponent range needs no scaling) and the six products of order <= 2, smallest first;
// like the TF32 kernel it sums in another order than the warp kernel (not bitwise).  The tail block n_bid is the warp
// kernel's arithmetic (block_scores_tail_kernel).
//   D = A x B with A = 16 key blocks (row l%16 of a lane, k = 8*(l/16)+i), B = 16 queries (column l%16, same k):
//   a lane ends up with 8 consecutive blocks of ONE query, written as one run.  No LDS for keys: a lane reads its
//   32-byte slice of the key row from global (the 16 query tiles of a launch re-read them from L2); the queries are
//   split once per CTA into LDS.
#if defined(__gfx1200__) || defined(__gfx1201__)
#define STRATA_SEL_GFX12 1
#else
#define STRATA_SEL_GFX12 0
#endif
typedef short sel_s8 __attribute__((ext_vector_type(8)));
typedef float sel_f8 __attribute__((ext_vector_type(8)));
typedef uint32_t sel_u4 __attribute__((ext_vector_type(4)));
constexpr int WQT = 16;                    // queries per CTA (the N of one WMMA)
constexpr int WITER = 4;                   // key tiles per warp (the CTA covers 4 warps * WITER * 16 blocks)
constexpr int WQS = IDX_DIM + 8;           // bf16 elements per LDS row: 272 bytes, conflict-free 16-byte reads

__device__ __forceinline__ sel_f8 wmma_bf16(const sel_s8& a, const sel_s8& b, const sel_f8& c) {
#if STRATA_SEL_GFX12
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
#else
    /* unreachable on SYCL: the launcher refuses this device */
    return c;
#endif
}
// the bf16 pieces of x as fp32 bit patterns whose upper 16 bits are the bf16 (lower 16 are zero)
__device__ __forceinline__ void split3(float x, uint32_t& hi, uint32_t& mid, uint32_t& lo) {
    hi = sycl::bit_cast<uint32_t>(x) & 0xffff0000u;
    const float r1 = x - __uint_as_float(hi);                   // exact
    mid = sycl::bit_cast<uint32_t>(r1) & 0xffff0000u;
    lo = sycl::bit_cast<uint32_t>(r1 - __uint_as_float(mid)) & 0xffff0000u;   // exact: at most 8 significant bits left
}
__device__ __forceinline__ uint32_t pack_bf16x2(uint32_t a, uint32_t b) {   // low half = a's bf16, high half = b's
#if STRATA_SEL_GFX12
    return __builtin_amdgcn_perm(b, a, 0x07060302u);
#else
    return (a >> 16) | (b & 0xffff0000u);
#endif
}

__global__ void __launch_bounds__(128) block_scores_wmma_kernel(const float* __restrict__ pooled,
                                                                const float* __restrict__ q_idx,
                                                                const int32_t* __restrict__ steps, int64_t nq,
                                                                int64_t max_blocks, int64_t reach,
                                                                float* __restrict__ out) {
    __shared__ __align__(16) uint16_t sq[3][IDX_HEADS][WQT][WQS];   // hi / mid / lo, [head][query][dim]
    __shared__ int s_nbid[WQT];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, l16 = lane & 15, g = lane >> 4;
    const int64_t q0 = (int64_t) blockIdx.y * WQT;
    for (int i = t; i < WQT * IDX_HEADS * IDX_DIM / 4; i += 128) {
        const int row = i / (IDX_DIM / 4), c = i % (IDX_DIM / 4);
        const int qr = row / IDX_HEADS, h = row % IDX_HEADS;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (q0 + qr < nq) v = reinterpret_cast<const float4*>(q_idx + (q0 + qr) * IDX_HEADS * IDX_DIM)[h * (IDX_DIM / 4) + c];
        const float x[4] = {v.x, v.y, v.z, v.w};
        uint32_t hb[4], mb[4], lb[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) split3(x[j], hb[j], mb[j], lb[j]);
        *reinterpret_cast<uint2*>(&sq[0][h][qr][c * 4]) = make_uint2(pack_bf16x2(hb[0], hb[1]), pack_bf16x2(hb[2], hb[3]));
        *reinterpret_cast<uint2*>(&sq[1][h][qr][c * 4]) = make_uint2(pack_bf16x2(mb[0], mb[1]), pack_bf16x2(mb[2], mb[3]));
        *reinterpret_cast<uint2*>(&sq[2][h][qr][c * 4]) = make_uint2(pack_bf16x2(lb[0], lb[1]), pack_bf16x2(lb[2], lb[3]));
    }
    if (t < WQT) s_nbid[t] = q0 + t < nq ? steps[(q0 + t) * kStepCount + kStepNBid] : 0;
    __syncthreads();
    int hi_nbid = 0;
#pragma unroll
    for (int i = 0; i < WQT; ++i) hi_nbid = max(hi_nbid, s_nbid[i]);
    const int64_t qi = q0 + l16;
    const int nb_q = s_nbid[l16];
    for (int it = 0; it < WITER; ++it) {
        const int64_t b0 = (((int64_t) blockIdx.x * WITER + it) * 4 + warp) * 16;
        if (b0 >= reach || b0 >= hi_nbid) break;          // warp-uniform; later tiles start higher
        const int64_t row = b0 + l16;
        const bool rv = row < hi_nbid && row < max_blocks;
        const float* kp = pooled + (rv ? row : 0) * IDX_DIM + g * 8;
        // acc: the hi*hi products; cor: the five smaller ones (<= 2^-8 of it). Kept apart and added once at the end: a
        // correction summed into the big accumulator is rounded to ITS ulp at every one of the 48 steps (measured: 8e-7
        // of the score scale; apart, near the warp kernel's)
        sel_f8 acc[IDX_HEADS], cor[IDX_HEADS];
#pragma unroll
        for (int h = 0; h < IDX_HEADS; ++h) acc[h] = cor[h] = sel_f8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll 2
        for (int kk = 0; kk < IDX_DIM / 16; ++kk) {
            float4 k0 = make_float4(0.f, 0.f, 0.f, 0.f), k1 = k0;
            if (rv) {
                k0 = *reinterpret_cast<const float4*>(kp + kk * 16);
                k1 = *reinterpret_cast<const float4*>(kp + kk * 16 + 4);
            }
            const float x[8] = {k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w};
            uint32_t hb[8], mb[8], lb[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) split3(x[j], hb[j], mb[j], lb[j]);
            sel_u4 ah, am, al;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                ah[j] = pack_bf16x2(hb[2 * j], hb[2 * j + 1]);
                am[j] = pack_bf16x2(mb[2 * j], mb[2 * j + 1]);
                al[j] = pack_bf16x2(lb[2 * j], lb[2 * j + 1]);
            }
            const sel_s8 Ah = __builtin_bit_cast(sel_s8, ah), Am = __builtin_bit_cast(sel_s8, am),
                         Al = __builtin_bit_cast(sel_s8, al);
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const sel_s8 Bh = *reinterpret_cast<const sel_s8*>(&sq[0][h][l16][kk * 16 + g * 8]);
                const sel_s8 Bm = *reinterpret_cast<const sel_s8*>(&sq[1][h][l16][kk * 16 + g * 8]);
                const sel_s8 Bl = *reinterpret_cast<const sel_s8*>(&sq[2][h][l16][kk * 16 + g * 8]);
                cor[h] = wmma_bf16(Al, Bh, cor[h]);
                cor[h] = wmma_bf16(Ah, Bl, cor[h]);
                cor[h] = wmma_bf16(Am, Bm, cor[h]);
                cor[h] = wmma_bf16(Ah, Bm, cor[h]);
                cor[h] = wmma_bf16(Am, Bh, cor[h]);
                acc[h] = wmma_bf16(Ah, Bh, acc[h]);
            }
        }
        // relu per head, heads added in order (as the warp kernel); this lane: query qi, blocks b0 + 8g .. 8g+7
        if (qi < nq) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int64_t b = b0 + g * 8 + i;
                if (b >= nb_q || b >= max_blocks) continue;
                float score = 0.0f;
#pragma unroll
                for (int h = 0; h < IDX_HEADS; ++h) {
                    const float d = acc[h][i] + cor[h][i];
                    score += d > 0.0f ? d : 0.0f;
                }
                out[qi * max_blocks + b] = score;
            }
        }
    }
}

// the gfx12 kernel needs gfx1200/gfx1201 code objects and a gfx12 device (gfx1100 has WMMA too, with another layout)
bool sel_gfx12_device() {
    static int ok[64] = {};   // per device: 0 unknown, 1 yes, 2 no
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (ok[dev] == 0) {
        cudaDeviceProp prop;
        ok[dev] = (cudaGetDeviceProperties(&prop, dev) == cudaSuccess &&
                   (std::strncmp(prop.gcnArchName, "gfx1200", 7) == 0 || std::strncmp(prop.gcnArchName, "gfx1201", 7) == 0))
                      ? 1 : 2;
        cudaGetLastError();
    }
    return ok[dev] == 1;
}
#endif  // __HIPCC__

// the tail block n_bid of each query: exactly block_scores_kernel's arithmetic for that block
__dpct_inline__ void block_scores_tail_kernel(const float *__restrict__ dead,
                                              const float *__restrict__ q_idx,
                                              const int32_t *__restrict__ steps,
                                              int64_t max_blocks,
                                              float *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t qi = item_ct1.get_group(2);
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid];
    if (n_bid >= max_blocks) return;
    const int lane = item_ct1.get_local_id(2) & 31;
    const sycl::float4 k4 =
        *reinterpret_cast<const sycl::float4 *>(dead + lane * 4);
    const float* q = q_idx + qi * IDX_HEADS * IDX_DIM + lane * 4;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < IDX_HEADS; ++h) {
        const sycl::float4 q4 =
            *reinterpret_cast<const sycl::float4 *>(q + h * IDX_DIM);
        float d = k4.x() * q4.x() + k4.y() * q4.y() + k4.z() * q4.z() +
                  k4.w() * q4.w();
#pragma unroll
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        for (int o = 16; o > 0; o >>= 1) d +=
            dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                d, o);
        score += d > 0.0f ? d : 0.0f;
    }
    if (lane == 0) {
        if (n_kv % R != 0) score += 1e9f;
        out[qi * max_blocks + n_bid] = score;
    }
}

// ---- the same top-k with each query's keys read once: 1,024 threads hold up to TK_PER consecutive blocks' keys in
// registers (contexts up to 4 * 1024 * TK_PER cells), per-warp histograms, block-wide scans. The selection rule is
// block_topk_kernel's (radix threshold, ties to the lowest index, cells ascending): identical ids.
constexpr int TK_T = 1024;
constexpr int TK_PER = 33;
#if defined(__HIPCC__)
// AMD (RDNA, wave32): a 1,024-thread block may use 192 VGPRs per lane, so the keys of 66 blocks per thread fit - contexts
// up to 4 * 1024 * 66 = 270,336 cells (the 262,144 --max-context) keep the register kernel. NVIDIA's 64 registers per
// thread at 1,024 threads allow only TK_PER.
constexpr int TK_PER_MAX = 66;
#else
constexpr int TK_PER_MAX = TK_PER;
#endif

__dpct_inline__ int block_excl_scan(int v, int *s_warp, int &total) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = item_ct1.get_local_id(2) & 31,
              warp = item_ct1.get_local_id(2) >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        /*
        DPCT1108: '__shfl_up_sync' was migrated with the experimental feature
        masked sub_group function which may not be supported by all compilers or
        runtimes. You may need to adjust the code.
        */
        const int y = dpct::experimental::shift_sub_group_right(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), x,
            o);
        if (lane >= o) x += y;
    }
    if (lane == 31) s_warp[warp] = x;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (warp == 0) {
        int w = s_warp[lane];
        int z = w;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            /*
            DPCT1108: '__shfl_up_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const int y = dpct::experimental::shift_sub_group_right(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                z, o);
            if (lane >= o) z += y;
        }
        s_warp[lane] = z - w;               // exclusive per warp
        if (lane == 31) s_warp[32] = z;     // total
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const int r = s_warp[warp] + x - v;
    total = s_warp[32];
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    return r;
}

template <int PER>
/*
DPCT1110: The total declared local variable size in device function
block_topk_reg_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void block_topk_reg_kernel(const float *__restrict__ scores,
                                           const int32_t *__restrict__ steps,
                                           int64_t max_blocks, int64_t cap,
                                           int32_t *__restrict__ ids) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &hist =
    *sycl::ext::oneapi::group_local_memory_for_overwrite<int[TK_T / 32][256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_warp =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[33]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_digit = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_above = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int64_t qi = item_ct1.get_group(2);
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    if (n_kv <= width) {
#pragma unroll
        for (int64_t j = t; j < n_kv; j += TK_T) out[j] = (int32_t)j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;
    const int64_t per = (nb + TK_T - 1) / TK_T;       // <= PER (the caller checks)
    const int64_t b0 = (int64_t) t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    uint32_t key[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) key[j] = (b0 + j < b1) ? order_key(sc[b0 + j]) : 0u;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    uint32_t prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
#pragma unroll
        for (int i = lane; i < 256; i += 32) hist[warp][i] = 0;
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int64_t b = b0 + j;
            if (b >= b1) break;
            const int w = weight(b);
            if (w == 0) continue;
            if ((key[j] & hi_mask) == (prefix & hi_mask))
                dpct::atomic_fetch_add<
                    sycl::access::address_space::generic_space>(
                    &hist[warp][(key[j] >> shift) & 255], w);
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
        if (t < 256) {                                // fold the warps' histograms into warp 0's
            int s = 0;
#pragma unroll
            for (int w2 = 0; w2 < TK_T / 32; ++w2) s += hist[w2][t];
            hist[0][t] = s;
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
        if (t == 0) {
            int cum = above, d = 255;
#pragma unroll
            for (; d > 0; --d) {
                if (cum + hist[0][d] >= width) break;
                cum += hist[0][d];
            }
            s_digit = d;
            s_above = cum;
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
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
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
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;
    int gt = 0, eq = 0;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t b = b0 + j;
        if (b >= b1) break;
        const int w = weight(b);
        if (w == 0) continue;
        if (key[j] > thr) gt += w;
        else if (key[j] == thr) eq += w;
    }
    int tot;
    const int eq_before = block_excl_scan(eq, s_warp, tot);
    int64_t my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + (int) my_eq;
    int64_t wpos = block_excl_scan(sel, s_warp, tot);
    int64_t eq_left = my_eq;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t b = b0 + j;
        if (b >= b1) break;
        const int w = weight(b);
        if (w == 0) continue;
        if (key[j] > thr) {
#pragma unroll
            for (int c = 0; c < w; ++c) out[wpos++] = (int32_t)(b * R + c);
        } else if (key[j] == thr) {
#pragma unroll
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left)
                out[wpos++] = (int32_t)(b * R + c);
        }
    }
}


// Block scores with every key block read ONCE for all of a call's queries (block_scores_kernel's grid is
// (max_blocks / 8) x nq: ~24,600 mostly-idle blocks per layer at a decode window, each key re-read per query).  A fixed
// grid strides over the blocks; per (block, query) the same arithmetic in the same order as block_scores_kernel.
// qsa_block_scores takes it for every call without an active-block count and at most MQ queries: the captured decode
// window, the uncaptured decode, and prefill's pooled16 call.
constexpr int MQ = 8;
/*
DPCT1110: The total declared local variable size in device function
block_scores_multi_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
__dpct_inline__ void block_scores_multi_kernel(
    const float *__restrict__ pooled, const float *__restrict__ dead,
    const float *__restrict__ q_idx, const int32_t *__restrict__ steps, int nq,
    int64_t max_blocks, float *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &qs = *sycl::ext::oneapi::group_local_memory_for_overwrite<
    float[MQ * IDX_HEADS * IDX_DIM]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_nkv =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int64_t[MQ]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_nbid =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int64_t[MQ]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < nq * IDX_HEADS * IDX_DIM;
         i += item_ct1.get_local_range(2)) qs[i] = q_idx[i];
    if (item_ct1.get_local_id(2) < nq) {
        s_nkv[item_ct1.get_local_id(2)] =
            steps[item_ct1.get_local_id(2) * kStepCount + kStepNKv];
        s_nbid[item_ct1.get_local_id(2)] =
            steps[item_ct1.get_local_id(2) * kStepCount + kStepNBid];
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    int64_t top = 0;
#pragma unroll
    for (int q = 0; q < nq; ++q) top = s_nbid[q] > top ? s_nbid[q] : top;
    const int lane = item_ct1.get_local_id(2) & 31;
    const int64_t wstride = (int64_t)item_ct1.get_group_range(2) * SCORE_WARPS;
    for (int64_t b = (int64_t)item_ct1.get_group(2) * SCORE_WARPS +
                     (item_ct1.get_local_id(2) >> 5);
         b <= top && b < max_blocks; b += wstride) {
        const sycl::float4 kp = *reinterpret_cast<const sycl::float4 *>(
            pooled + b * IDX_DIM + lane * 4);
        const sycl::float4 kd =
            *reinterpret_cast<const sycl::float4 *>(dead + lane * 4);
        for (int qi = 0; qi < nq; ++qi) {
            const int64_t n_bid = s_nbid[qi];
            if (b > n_bid) continue;
            const sycl::float4 k4 = (b == n_bid) ? kd : kp;
            const float* q = qs + qi * IDX_HEADS * IDX_DIM + lane * 4;
            float score = 0.0f;
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const sycl::float4 q4 =
                    *reinterpret_cast<const sycl::float4 *>(q + h * IDX_DIM);
                float d = k4.x() * q4.x() + k4.y() * q4.y() + k4.z() * q4.z() +
                          k4.w() * q4.w();
#pragma unroll
                /*
                DPCT1108: '__shfl_xor_sync' was migrated with the
                experimental feature masked sub_group function which may not be
                supported by all compilers or runtimes. You may need to adjust
                the code.
                */
                for (int o = 16; o > 0; o >>= 1) d +=
                    dpct::experimental::permute_sub_group_by_xor(
                        0xffffffffu,
                        sycl::ext::oneapi::this_work_item::get_sub_group(), d,
                        o);
                score += d > 0.0f ? d : 0.0f;
            }
            if (lane == 0) {
                if (b == n_bid && s_nkv[qi] % R != 0) score += 1e9f;
                out[qi * max_blocks + b] = score;
            }
        }
    }
}
}  // namespace

void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                      int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t active_blocks) {
    if (nq <= 0) return;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || nq > 65535) {
        std::fprintf(stderr, "qsa_block_scores: unsupported indexer geometry\n");
        std::exit(1);
    }
    // a block past a query's n_bid returns at once: the grid need only reach the batch's largest n_bid (C-1)
    static const bool multi = [] { const char* v = std::getenv("STRATA_SCORES_MULTI"); return v == nullptr || std::atoi(v) != 0; }();
    if (multi && nq <= MQ && active_blocks <= 0) {   // no active count: decode (captured or not) and prefill's pooled16
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            strata::q_of(stream)
                ->parallel_for<
                    dpct_kernel_name<class block_scores_multi_kernel_dea7d4>>(
                    sycl::nd_range<3>(sycl::range(1, 1, 256) *
                                          sycl::range(1, 1, SCORE_WARPS * 32),
                                      sycl::range(1, 1, SCORE_WARPS * 32)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            block_scores_multi_kernel(pooled, dead, q_idx,
                                                      steps, (int)nq,
                                                      max_blocks, scores);
                        });
        }
        /*
        DPCT1010: SYCL uses exceptions to report errors and does not use the
        error codes. The cudaGetLastError function call was replaced with 0. You
        need to rewrite this code.
        */
        const dpct::err0 e = 0;
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */

        return;
    }
    const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
    const dpct::dim3 grid((unsigned)((reach + SCORE_WARPS - 1) / SCORE_WARPS),
                          (unsigned)nq);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class block_scores_kernel_925c9d>>(
                sycl::nd_range<3>(grid * sycl::range(1, 1, SCORE_WARPS * 32),
                                  sycl::range(1, 1, SCORE_WARPS * 32)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        block_scores_kernel(pooled, dead, q_idx, steps,
                                            max_blocks, scores);
                    });
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
}

bool qsa_block_scores_tc(const float *pooled, const float *dead,
                         const float *q_idx, const int32_t *steps, int64_t nq,
                         int64_t max_blocks, const QsaShapes &s, float *scores,
                         void *stream, int64_t active_blocks) try {
    if (nq <= 0) return true;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || nq > 65535 * TC_QT) return false;
#if defined(__HIPCC__)
    // AMD: the gfx12 (RDNA4) WMMA scorer, opt-in (STRATA_SELECT_WMMA=1): it selects slightly differently from the warp
    // kernel (254/256 queries the same), so the default keeps the warp kernel; every other target keeps it too (false)
    static const bool wmma_on = [] {
        const char* v = std::getenv("STRATA_SELECT_WMMA");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    if (!wmma_on || !sel_gfx12_device()) return false;
    {
        const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
        const int64_t per = (int64_t) 4 * WITER * 16;
        const dim3 grid((unsigned) ((reach + per - 1) / per), (unsigned) ((nq + WQT - 1) / WQT));
        block_scores_wmma_kernel<<<grid, 128, 0, (cudaStream_t) stream>>>(pooled, q_idx, steps, nq, max_blocks, reach, scores);
        block_scores_tail_kernel<<<(unsigned) nq, 32, 0, (cudaStream_t) stream>>>(dead, q_idx, steps, max_blocks, scores);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores_tc: %s\n", cudaGetErrorString(e)); std::exit(1); }
        return true;
    }
#else
    {   // sm_80 or newer (TF32 MMA); an older card keeps the warp kernel
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
            // STRATA_QSA_WARP=1|select (an A/B arm): the pre-sm_80 kernels on any card, as RTX 20 runs them
            const char* w = std::getenv("STRATA_QSA_WARP");
            cc_major[dev] = w && (!std::strcmp(w, "1") || !std::strcmp(w, "select")) ? 7 : strata::cc_major_of(major);
        }
        (void) cc_major[dev];
        return false;   // SYCL: the tensor-core kernel is not ported yet; the caller takes the older kernel
    }
    static bool attr[64] = {};   // the shared-memory opt-in is per device (a layer split runs it on several)
    int adev = 0;
    adev = dpct::get_current_device_id();
    /*
    DPCT1083: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    const int bytes = (TC_QT * TC_QS + TC_NB * TC_KS) * (int)sizeof(float);
    if (!attr[adev]) {
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
        attr[adev] = true;
    }
    const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
    const int64_t per = (int64_t) TC_NB * TC_ITER;
    const dpct::dim3 grid((unsigned)((reach + per - 1) / per),
                          (unsigned)((nq + TC_QT - 1) / TC_QT));
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range(bytes), cgh);

                cgh.parallel_for<
                    dpct_kernel_name<class block_scores_tc_kernel_bc5f07>>(
                    sycl::nd_range<3>(grid * sycl::range(1, 1, 128),
                                      sycl::range(1, 1, 128)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        block_scores_tc_kernel(
                            pooled, q_idx, steps, nq, max_blocks, reach, scores,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class block_scores_tail_kernel_70dfaa>>(
                sycl::nd_range<3>(sycl::range(1, 1, (unsigned)nq) *
                                      sycl::range(1, 1, 32),
                                  sycl::range(1, 1, 32)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        block_scores_tail_kernel(dead, q_idx, steps, max_blocks,
                                                 scores);
                    });
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */

    return true;
#endif
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void qsa_block_topk_ref(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                        const QsaShapes& s, int32_t* ids, void* stream) {
    if (nq <= 0) return;
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s)) {
        std::fprintf(stderr, "qsa_block_topk: unsupported geometry or cap\n");
        std::exit(1);
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class block_topk_kernel_213104>>(
                sycl::nd_range<3>(sycl::range(1, 1, (unsigned)nq) *
                                      sycl::range(1, 1, TOPK_T),
                                  sycl::range(1, 1, TOPK_T)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    block_topk_kernel(scores, steps, max_blocks, cap, ids);
                });
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
}

void qsa_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                    const QsaShapes& s, int32_t* ids, void* stream, int64_t active_blocks) {
    // keys in registers when every query's blocks fit (contexts up to ~135K cells); the same ids. STRATA_TOPK_OLD=1:
    // the kernel that reads them from memory on every pass
    static const bool old = std::getenv("STRATA_TOPK_OLD") != nullptr;
    if (nq <= 0) return;
    // the blocks a query can have: the call's active count when the caller knows it (the prompt path), else the capacity.
    // Decode (no count) keeps the capacity rule and the original register width: nothing changes there.
#if defined(__HIPCC__)
    const bool counted = active_blocks > 0;
#else
    // CUDA keeps 0.1.32's capacity rule: #337's dispatch was measured on RDNA4 only, and on the RTX 5070 the 64K
    // prompts read 1-3% slower with it
    const bool counted = false;
    (void) active_blocks;
#endif
    const int64_t reach = counted && active_blocks < max_blocks ? active_blocks : max_blocks;
    const int64_t fit = (int64_t) TK_T * (counted ? TK_PER_MAX : TK_PER);
#if defined(__HIPCC__)
    constexpr int64_t kRegMinBlocks = 7168;   // gfx1201: below ~28K cells the 1,024-thread kernel's fixed cost loses to the ref
    const bool too_small = counted && reach < kRegMinBlocks;
#else
    const bool too_small = false;
#endif
    if (old || too_small || reach > fit) {
        qsa_block_topk_ref(scores, steps, nq, max_blocks, cap, s, ids, stream);
        return;
    }
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s)) {
        std::fprintf(stderr, "qsa_block_topk: unsupported geometry or cap\n");
        std::exit(1);
    }
    if (reach <= (int64_t) TK_T * TK_PER)
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class block_topk_reg_kernel_652097,
                                            dpct_kernel_scalar<TK_PER>>>(
                sycl::nd_range<3>(sycl::range(1, 1, (unsigned)nq) *
                                      sycl::range(1, 1, TK_T),
                                  sycl::range(1, 1, TK_T)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        block_topk_reg_kernel<TK_PER>(scores, steps, max_blocks,
                                                      cap, ids);
                    });
    } else
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed
    the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class block_topk_reg_kernel_652098,
                                            dpct_kernel_scalar<TK_PER_MAX>>>(
                sycl::nd_range<3>(sycl::range(1, 1, (unsigned)nq) *
                                      sycl::range(1, 1, TK_T),
                                  sycl::range(1, 1, TK_T)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        block_topk_reg_kernel<TK_PER_MAX>(scores, steps,
                                                          max_blocks, cap, ids);
                    });
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
}

}  // namespace strata::kernels
