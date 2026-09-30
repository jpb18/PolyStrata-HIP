// src/kernels/cuda/qsa_select.cu - see include/strata/kernels/qsa_select.hpp.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/qsa_select.hpp"

#include <cfloat>
#include <cstdio>
#include <cstdlib>

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

__dpct_inline__ int block_excl_scan(int v, int *s_warp, int &total) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = item_ct1.get_local_id(2) & 31,
              warp = item_ct1.get_local_id(2) >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        /*
        DPCT1108: '__shfl_up_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
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
    const int64_t per = (nb + TK_T - 1) / TK_T;       // <= TK_PER (the caller checks)
    const int64_t b0 = (int64_t) t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    uint32_t key[TK_PER];
#pragma unroll
    for (int j = 0; j < TK_PER; ++j) key[j] = (b0 + j < b1) ? order_key(sc[b0 + j]) : 0u;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    uint32_t prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
#pragma unroll
        for (int i = lane; i < 256; i += 32) hist[warp][i] = 0;
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
#pragma unroll
        for (int j = 0; j < TK_PER; ++j) {
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
    for (int j = 0; j < TK_PER; ++j) {
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
    for (int j = 0; j < TK_PER; ++j) {
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

}  // namespace

void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                      int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t active_blocks) {
    if (nq <= 0) return;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || nq > 65535) {
        std::fprintf(stderr, "qsa_block_scores: unsupported indexer geometry\n");
        std::exit(1);
    }
    // a block past a query's n_bid returns at once: the grid need only reach the batch's largest n_bid (C-1)
    const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
    const dpct::dim3 grid((unsigned)((reach + SCORE_WARPS - 1) / SCORE_WARPS),
                          (unsigned)nq);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class block_scores_kernel_6f5935>>(
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
            cc_major[dev] = major;
        }
        (void) cc_major[dev];
        return false;   // SYCL: the tensor-core kernel is not ported yet; the caller takes the older kernel
    }
#if defined(__HIPCC__)
    return false;   // the tensor-core kernel is compiled out on AMD (its major version is not a CUDA sm)
#endif
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
                    dpct_kernel_name<class block_scores_tc_kernel_f08b24>>(
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
                dpct_kernel_name<class block_scores_tail_kernel_8303e7>>(
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
            ->parallel_for<dpct_kernel_name<class block_topk_kernel_409264>>(
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
                    const QsaShapes& s, int32_t* ids, void* stream) {
    // keys in registers when every query's blocks fit (contexts up to ~135K cells); the same ids. STRATA_TOPK_OLD=1:
    // the kernel that reads them from memory on every pass
    static const bool old = std::getenv("STRATA_TOPK_OLD") != nullptr;
    if (nq <= 0) return;
    if (old || max_blocks > (int64_t) TK_T * TK_PER) {
        qsa_block_topk_ref(scores, steps, nq, max_blocks, cap, s, ids, stream);
        return;
    }
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s)) {
        std::fprintf(stderr, "qsa_block_topk: unsupported geometry or cap\n");
        std::exit(1);
    }
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class block_topk_reg_kernel_a867d1>>(
                sycl::nd_range<3>(sycl::range(1, 1, (unsigned)nq) *
                                      sycl::range(1, 1, TK_T),
                                  sycl::range(1, 1, TK_T)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        block_topk_reg_kernel(scores, steps, max_blocks, cap,
                                              ids);
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
