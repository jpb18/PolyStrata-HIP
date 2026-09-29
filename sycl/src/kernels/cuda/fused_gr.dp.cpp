// src/kernels/cuda/fused_gr.cu - see include/strata/kernels/fused_gr.hpp.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int N = 2560;         // n_embd
constexpr int HC = 4;           // streams
constexpr int D = N * HC;       // 10240
constexpr int LR = 320;         // hc_lr
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int DOWN_BLOCKS = LR / WARPS;          // 40 blocks of 8 rows; one more for the inject rows
constexpr int UP_COLS = 32;                      // columns d per `up` block (x 4 streams = 128 rows)
constexpr int UP_BLOCKS = N / UP_COLS;           // 80

__dpct_inline__ float warp_sum(float v) {
#pragma unroll
    /*
    DPCT1108:88: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = 16; o > 0; o >>= 1) v +=
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), v,
            o);
    return v;
}
__dpct_inline__ float sigmoidf_(float x) {
    return 1.0f / (1.0f + sycl::native::exp(-x));
}

// 8 bf16 packed in a uint4 against 8 floats.
__dpct_inline__ float dot8(const sycl::uint4 w, const float *x) {
    float acc = 0.0f;
    const uint32_t v[4] = {w.x(), w.y(), w.z(), w.w()};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        acc = sycl::fma(sycl::bit_cast<float>(v[j] << 16), (float)(x[2 * j]),
                        acc);
        acc = sycl::fma(sycl::bit_cast<float>(v[j] & 0xffff0000u),
                        (float)(x[2 * j + 1]), acc);
    }
    return acc;
}

/*
DPCT1110:89: The total declared local variable size in device function
gr_down_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void gr_down_kernel(FusedGrArgs a) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &xn = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[D]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &part =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[WARPS][HC]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_rs =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[HC]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    float gw[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    // 1. R' * w_norm into shared memory, and the per-stream sums of squares of R'.
    float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = t * 4; i < D; i += THREADS * 4) {
        const int c = i / N, d = i - c * N;
        sycl::float4 r = *reinterpret_cast<const sycl::float4 *>(a.R + i);
        if (a.apply) {
            const sycl::float4 b =
                *reinterpret_cast<const sycl::float4 *>(a.bo_prev + d);
            r.x() = sycl::fma((float)(b.x()), gw[c], r.x());
                r.y() = sycl::fma((float)(b.y()), gw[c], r.y());
            r.z() = sycl::fma((float)(b.z()), gw[c], r.z());
                r.w() = sycl::fma((float)(b.w()), gw[c], r.w());
        }
        const sycl::float4 g =
            *reinterpret_cast<const sycl::float4 *>(a.w_norm + i);
        float sq =
            r.x() * r.x() + r.y() * r.y() + r.z() * r.z() + r.w() * r.w();
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) if (cc == c) ss[cc] += sq;
        *reinterpret_cast<sycl::float4 *>(xn + i) = sycl::float4(
            r.x() * g.x(), r.y() * g.y(), r.z() * g.z(), r.w() * g.w());
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float v = warp_sum(ss[c]);
        if (lane == 0) part[warp][c] = v;
    }
    /*
    DPCT1065:457: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (t < HC) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) s += part[w][t];
        s_rs[t] = sycl::rsqrt(s / (float)N + a.eps);
        if (item_ct1.get_group(2) == 0) a.rs[t] = s_rs[t];
    }
    /*
    DPCT1065:458: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#pragma unroll
    for (int i = t; i < D; i += THREADS) xn[i] *= s_rs[i / N];
    /*
    DPCT1065:459: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // 2. one warp per output row: 10240 bf16 = 1280 chunks of 8, 40 per lane.
    const bool inject_block = item_ct1.get_group(2) == DOWN_BLOCKS;
    const int row = inject_block ? warp : item_ct1.get_group(2) * WARPS + warp;
    if (inject_block && (a.w_inject == nullptr || warp >= HC)) return;
    const uint16_t* wrow = (inject_block ? a.w_inject : a.w_down) + (size_t) row * D;
    const sycl::uint4 *w4 = reinterpret_cast<const sycl::uint4 *>(wrow);
    float acc = 0.0f;
#pragma unroll 4
    /*
    DPCT1098:460: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    for (int j = lane; j < D / 8; j += 32) acc += dot8(*(w4 + j), xn + j * 8);
    acc = warp_sum(acc);
    if (lane != 0) return;
    if (inject_block) {
        a.inject_out[row] = acc;
    } else {
        const float x = acc / (float) HC;
        a.lo[row] = x / (1.0f + sycl::native::exp(-x));
    }
}

__dpct_inline__ void gr_up_kernel(FusedGrArgs a) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &lo = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[LR]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &g = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[HC][UP_COLS]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int d0 = item_ct1.get_group(2) * UP_COLS;
#pragma unroll
    for (int k = t; k < LR; k += THREADS) lo[k] = a.lo[k];
    /*
    DPCT1065:461: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // 128 rows (4 streams x 32 columns), 16 per warp: 320 bf16 = 40 chunks of 8.
    for (int r = warp; r < HC * UP_COLS; r += WARPS) {
        const int c = r / UP_COLS, dd = r - c * UP_COLS, i = c * N + d0 + dd;
        const sycl::uint4 *w4 =
            reinterpret_cast<const sycl::uint4 *>(a.w_up + (size_t)i * LR);
        /*
        DPCT1098:463: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        float acc = dot8(*(w4 + lane), lo + lane * 8);
        /*
        DPCT1098:464: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        if (lane < LR / 8 - 32) acc +=
            dot8(*(w4 + 32 + lane), lo + (32 + lane) * 8);
        acc = warp_sum(acc);
        if (lane == 0) {
            float rv = a.R[i];
            if (a.apply) {
                rv = sycl::fma((float)(a.bo_prev[d0 + dd]),
                               2.0f * sigmoidf_(a.inj_prev[c] / (float)HC), rv);
                a.R_out[i] = rv;                       // this block owns column d0+dd of every stream
            }
            const float x = rv * a.w_norm[i] * a.rs[c];
            g[c][dd] = x * sigmoidf_(acc);
        }
    }
    /*
    DPCT1065:462: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (t < UP_COLS) {
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[c][t];
        a.mixed[d0 + t] = s / (float) HC;
    }
}

// ================================ plan v0.3 P6: T tokens, one weight read ================================
struct GrMulti {
    FusedGrArgs a[kFusedGrMaxT];
    float* xn;
    int T;
};

// Step 1 of `gr_down_kernel`, one block per token, same threads and reduction order: rs[t] and xn[t] to global.
/*
DPCT1110:90: The total declared local variable size in device function
gr_norm_multi_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void gr_norm_multi_kernel(GrMulti m) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part =
    *sycl::ext::oneapi::group_local_memory_for_overwrite<float[WARPS][HC]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_rs =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[HC]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const FusedGrArgs &a = m.a[item_ct1.get_group(2)];
    float *xn = m.xn + (size_t)item_ct1.get_group(2) * D;
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    float gw[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = t * 4; i < D; i += THREADS * 4) {
        const int c = i / N, d = i - c * N;
        sycl::float4 r = *reinterpret_cast<const sycl::float4 *>(a.R + i);
        if (a.apply) {
            const sycl::float4 b =
                *reinterpret_cast<const sycl::float4 *>(a.bo_prev + d);
            r.x() = sycl::fma((float)(b.x()), gw[c], r.x());
                r.y() = sycl::fma((float)(b.y()), gw[c], r.y());
            r.z() = sycl::fma((float)(b.z()), gw[c], r.z());
                r.w() = sycl::fma((float)(b.w()), gw[c], r.w());
        }
        const sycl::float4 g =
            *reinterpret_cast<const sycl::float4 *>(a.w_norm + i);
        float sq =
            r.x() * r.x() + r.y() * r.y() + r.z() * r.z() + r.w() * r.w();
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) if (cc == c) ss[cc] += sq;
        *reinterpret_cast<sycl::float4 *>(xn + i) = sycl::float4(
            r.x() * g.x(), r.y() * g.y(), r.z() * g.z(), r.w() * g.w());
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float v = warp_sum(ss[c]);
        if (lane == 0) part[warp][c] = v;
    }
    /*
    DPCT1065:465: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (t < HC) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) s += part[w][t];
        s_rs[t] = sycl::rsqrt(s / (float)N + a.eps);
        a.rs[t] = s_rs[t];
    }
    /*
    DPCT1065:466: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#pragma unroll
    for (int i = t; i < D; i += THREADS) xn[i] *= s_rs[i / N];
}

constexpr int TILE = 2560;             // xn floats per token staged at a time: 320 chunks of 8, 10 per lane
constexpr int TQ = TILE / 8 / 32;      // uint4 weight chunks per lane per tile

// Step 2 of `gr_down_kernel` for T tokens.  One warp per row (so each lane accumulates the same chunks in the
// same order as the single-token kernel); per tile the lane's 10 weight chunks are loaded BEFORE the activation
// tile is staged, so the DRAM and L2 traffic are in flight together.
/*
DPCT1110:91: The total declared local variable size in device function
gr_down_multi_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void gr_down_multi_kernel(GrMulti m, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto tile = (float *)dpct_local; // [T][TILE]
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int T = m.T;
    const bool inject_block = item_ct1.get_group(2) == DOWN_BLOCKS;
    const int row = inject_block ? warp : item_ct1.get_group(2) * WARPS + warp;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || warp >= HC));
    const uint16_t* wrow = (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t) (active ? row : 0) * D;
    const sycl::uint4 *w4 = reinterpret_cast<const sycl::uint4 *>(wrow);
    float acc[kFusedGrMaxT];
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) acc[k] = 0.0f;
    for (int base = 0; base < D; base += TILE) {
        sycl::uint4 wv[TQ];
        if (active) {
#pragma unroll
            /*
            DPCT1098:469: The '*' expression is used instead of the __ldg call.
            These two expressions do not provide the exact same functionality.
            Check the generated code for potential precision and/or performance
            issues.
            */
            for (int q = 0; q < TQ; ++q)
                wv[q] = *(w4 + base / 8 + lane + 32 * q);
        }
        /*
        DPCT1118:92: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:467: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier(); // the previous tile is consumed
        const sycl::float4 *src4 = reinterpret_cast<const sycl::float4 *>(m.xn);
        sycl::float4 *tile4 = reinterpret_cast<sycl::float4 *>(tile);
        for (int i = t; i < T * (TILE / 4); i += THREADS) {
            const int k = i / (TILE / 4), off = i - k * (TILE / 4);
            tile4[i] = src4[((size_t) k * D + base) / 4 + off];
        }
        /*
        DPCT1118:93: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:468: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (!active) continue;
#pragma unroll
        for (int q = 0; q < TQ; ++q) {
            const int j = lane + 32 * q;
#pragma unroll
            for (int k = 0; k < kFusedGrMaxT; ++k)
                if (k < T) acc[k] += dot8(wv[q], tile + k * TILE + j * 8);
        }
    }
    if (!active) return;
    float s[kFusedGrMaxT];
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) s[k] = k < T ? warp_sum(acc[k]) : 0.0f;
    // lane k writes token k (every lane holds every sum after the xor reduction)
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) {
        if (k >= T || lane != k) continue;
        if (inject_block) {
            m.a[k].inject_out[row] = s[k];
        } else {
            const float x = s[k] / (float) HC;
            m.a[k].lo[row] = x / (1.0f + sycl::native::exp(-x));
        }
    }
}

constexpr int UPM_COLS = 16;                      // columns per block (x 4 streams = 64 rows, 8 per warp)
constexpr int UPM_BLOCKS = N / UPM_COLS;          // 160

// `gr_up_kernel` for T tokens: each row of w_up read once; the T dots reduced by xor so every lane holds every
// sum, and lane k runs token k's epilogue - the T epilogues in parallel instead of one after another.
/*
DPCT1110:94: The total declared local variable size in device function
gr_up_multi_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void gr_up_multi_kernel(GrMulti m) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &lo = *sycl::ext::oneapi::group_local_memory_for_overwrite<
    float[kFusedGrMaxT][LR]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &g = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[kFusedGrMaxT][HC][UPM_COLS]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int T = m.T;
    const int d0 = item_ct1.get_group(2) * UPM_COLS;
#pragma unroll
    for (int i = t; i < T * LR; i += THREADS)
        lo[i / LR][i % LR] = m.a[i / LR].lo[i % LR];
    /*
    DPCT1065:470: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int r = warp; r < HC * UPM_COLS; r += WARPS) {
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const sycl::uint4 *w4 =
            reinterpret_cast<const sycl::uint4 *>(m.a[0].w_up + (size_t)i * LR);
        /*
        DPCT1098:472: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        const sycl::uint4 wa = *(w4 + lane);
        /*
        DPCT1098:473: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        const sycl::uint4 wb =
            lane < LR / 8 - 32 ? *(w4 + 32 + lane) : sycl::uint4(0, 0, 0, 0);
        // the epilogue inputs of this lane's token, fetched while the dots run
        float rv = 0.0f, wn = 0.0f, rsc = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < T) {
            const FusedGrArgs& a = m.a[lane];
            rv = a.R[i];
            wn = a.w_norm[i];
            rsc = a.rs[c];
            apply = a.apply;
            if (apply) { bo = a.bo_prev[d0 + dd]; ip = a.inj_prev[c]; }
        }
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k) {
            if (k >= T) break;
            float acc = dot8(wa, lo[k] + lane * 8);
            if (lane < LR / 8 - 32) acc += dot8(wb, lo[k] + (32 + lane) * 8);
            acc = warp_sum(acc);
            if (lane == k) mine = acc;
        }
        if (lane < T) {
            if (apply) {
                rv = sycl::fma(bo, 2.0f * sigmoidf_(ip / (float)HC), rv);
                m.a[lane].R_out[i] = rv;
            }
            const float x = rv * wn * rsc;
            g[lane][c][dd] = x * sigmoidf_(mine);
        }
    }
    /*
    DPCT1065:471: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[k][c][col];
        m.a[k].mixed[d0 + col] = s / (float) HC;
    }
}

}  // namespace

void fused_gr_read_multi(const FusedGrArgs* a, int n_tok, float* xn_scratch, void* stream, unsigned long long* stamp_buf,
                         int stamp_i0) {
    if (n_tok < 1 || n_tok > kFusedGrMaxT || xn_scratch == nullptr) {
        std::fprintf(stderr, "fused_gr_read_multi: invalid arguments\n");
        std::exit(1);
    }
    GrMulti m;
    for (int t = 0; t < n_tok; ++t) {
        m.a[t] = a[t];
        const FusedGrArgs& x = a[t];
        if (!x.R || !x.w_norm || !x.w_down || !x.w_up || !x.lo || !x.rs || !x.mixed || (x.w_inject && !x.inject_out) ||
            (x.apply && (!x.bo_prev || !x.inj_prev || !x.R_out)) || x.w_down != a[0].w_down || x.w_up != a[0].w_up ||
            x.w_inject != a[0].w_inject || x.w_norm != a[0].w_norm) {
            std::fprintf(stderr, "fused_gr_read_multi: invalid arguments for token %d\n", t);
            std::exit(1);
        }
    }
    m.xn = xn_scratch;
    m.T = n_tok;
    dpct::queue_ptr st = strata::q_of(stream);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<dpct_kernel_name<class gr_norm_multi_kernel_bf1e13>>(
            sycl::nd_range<3>(sycl::range(1, 1, n_tok) *
                                  sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gr_norm_multi_kernel(m);
            });
    }
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, stream);
    // the shared-memory opt-in is a per-DEVICE setting: once per device, not once per process (a layer split
    // runs this kernel on two cards)
    static bool attr[64] = {};
    int dev = 0;
    dev = dpct::get_current_device_id();
    if (dev < 0 || dev >= 64 || !attr[dev]) {
        // at most what the card allows (Turing: 64 KB - enough for windows of up to 6 tokens)
        int optin = 0;
        /*
        DPCT1019:474: local_mem_size in SYCL is not a complete equivalent of
        cudaDevAttrMaxSharedMemoryPerBlockOptin in CUDA. You may need to adjust
        the code.
        */
        optin = dpct::get_device(dev).get_local_mem_size();
        int want = (int) (kFusedGrMaxT * TILE * sizeof(float));
        if (optin > 0 && want > optin) want = optin;
        /*
        DPCT1026:475: The call to cudaFuncSetAttribute was removed because SYCL
        currently does not support corresponding setting.
        */
        /*
        DPCT1026:476: The call to cudaGetLastError was removed because this
        functionality is redundant in SYCL.
        */
        if (dev >= 0 && dev < 64) attr[dev] = true;
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->submit([&](sycl::handler &cgh) {
            /*
            DPCT1083:1166: The size of local memory in the migrated code may be
            different from the original code. Check that the allocated memory
            size in the migrated code is correct.
            */
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range((size_t)n_tok * TILE * sizeof(float)), cgh);

            cgh.parallel_for<
                dpct_kernel_name<class gr_down_multi_kernel_e2a6b0>>(
                sycl::nd_range<3>(sycl::range(1, 1, DOWN_BLOCKS + 1) *
                                      sycl::range(1, 1, THREADS),
                                  sycl::range(1, 1, THREADS)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gr_down_multi_kernel(
                            m, dpct_local_acc_ct1
                                   .get_multi_ptr<sycl::access::decorated::no>()
                                   .get());
                    });
        });
    }
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, stream);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<dpct_kernel_name<class gr_up_multi_kernel_e2528e>>(
            sycl::nd_range<3>(sycl::range(1, 1, UPM_BLOCKS) *
                                  sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gr_up_multi_kernel(m);
            });
    }
    /*
    DPCT1010:477: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
}

bool fused_gr_supported(int64_t n_embd, int64_t hc, int64_t hc_lr) {
    return n_embd == N && hc == HC && hc_lr == LR;
}

void fused_gr_read(const FusedGrArgs& a, void* stream) {
    if (!a.R || !a.w_norm || !a.w_down || !a.w_up || !a.lo || !a.rs || !a.mixed ||
        (a.w_inject && !a.inject_out) || (a.apply && (!a.bo_prev || !a.inj_prev || !a.R_out)) ||
        (a.apply && a.inj_prev == a.inject_out)) {
        std::fprintf(stderr, "fused_gr_read: invalid arguments\n");
        std::exit(1);
    }
    dpct::queue_ptr st = strata::q_of(stream);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<dpct_kernel_name<class gr_down_kernel_5d714c>>(
            sycl::nd_range<3>(sycl::range(1, 1, DOWN_BLOCKS + 1) *
                                  sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gr_down_kernel(a);
            });
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<dpct_kernel_name<class gr_up_kernel_86511f>>(
            sycl::nd_range<3>(sycl::range(1, 1, UP_BLOCKS) *
                                  sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gr_up_kernel(a);
            });
    }
    /*
    DPCT1010:479: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
}

}  // namespace strata::kernels
