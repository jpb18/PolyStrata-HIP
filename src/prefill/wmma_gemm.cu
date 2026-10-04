// src/prefill/wmma_gemm.cu - RDNA3 WMMA FP16 & BF16 GEMM for Strata prefill (gfx1100).
//
// Computes Y[t, n] = beta * Y[t, n] + sum_k W[n, k] * X[t, k] using RDNA3 WMMA intrinsics:
//   * X is T x K row-major (leading dim K), fp16 or bf16
//   * W is N x K row-major (leading dim K), fp16 or bf16
//   * Y is T x N row-major with leading dimension ldy >= N, fp32
//
// Hardware: AMD RDNA3 (gfx1100, e.g. RX 7900 XTX)
//   * v_wmma_f32_16x16x16_f16_w32 intrinsic (__builtin_amdgcn_wmma_f32_16x16x16_f16_w32)
//   * v_wmma_f32_16x16x16_bf16_w32 intrinsic (__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32)
//   * Wave32 doubled input fragment: lane t (lane_lo = t & 15) holds row lane_lo of A (16 elements along K)
//     and column lane_lo of B (16 elements along K). Lanes 16..31 duplicate lanes 0..15.
//   * Wave32 C output mapping: lane t holds column lane_lo of 16x16 output tile,
//     with 8 elements alternating rows: row m = 2*i + lane_hi (lane_hi = t >> 4).
//   * Tile variants:
//       1. gemm_wmma_64x64_4w: 4 waves (128 threads), 64T x 64N x 16K tile, double-buffered LDS for W.
//       2. gemm_wmma_16x16_1w: 1 wave (32 threads), 16T x 16N x 16K tile, zero LDS/barriers, for small T/N.

#include "wmma_gemm.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#include <cstdlib>
#include <cstring>

// STRATA_WMMA_GFX11 is defined by the BUILD (CMakeLists.txt, from CMAKE_HIP_ARCHITECTURES), not inferred
// from compiler macros: measured on this toolchain the HOST pass of a HIP compile does not define
// __gfx1100__ but does define __HIP_DEVICE_COMPILE__, so a compiler-macro guard here silently selected the
// "return false" stub at the bottom of this file for the very symbol the engine links - the WMMA path then
// never ran, while the same file compiled with hipcc (as the probes do) took the real branch.  One
// build-defined macro is uniform across the host and device passes.
#if defined(STRATA_WMMA_GFX11)

using v8fp32 = float __attribute__((ext_vector_type(8)));

template <typename ElemT>
struct WmmaTraits;

template <>
struct WmmaTraits<_Float16> {
    using vec_t = _Float16 __attribute__((ext_vector_type(16)));
    __device__ static inline v8fp32 mma(vec_t a, vec_t b, v8fp32 c) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || defined(__gfx1151__)
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
        (void) a; (void) b; return c;   // not a gfx11 device pass: never launched (runtime gate)
#endif
    }
};

template <>
struct WmmaTraits<__bf16> {
    using vec_t = __bf16 __attribute__((ext_vector_type(16)));
    __device__ static inline v8fp32 mma(vec_t a, vec_t b, v8fp32 c) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || defined(__gfx1151__)
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
#else
        (void) a; (void) b; return c;   // not a gfx11 device pass: never launched (runtime gate)
#endif
    }
};

// ===========================================================================
// Variant 1: 16x16_1w - Single-wave kernel for small T / N.
// Zero LDS, zero barrier synchronization, full register residency.
// ===========================================================================
template <typename ElemT>
__global__ void gemm_wmma_16x16_1w(
    const uint16_t* __restrict__ X,
    const uint16_t* __restrict__ W,
    float* __restrict__ Y,
    int64_t T, int64_t N, int64_t K, int64_t ldy, float beta) {

    using vec_t = typename WmmaTraits<ElemT>::vec_t;
    const int m_tile = blockIdx.y * 16;
    const int n_tile = blockIdx.x * 16;
    if (m_tile >= T || n_tile >= N) return;

    const int lane = threadIdx.x;   // 0..31
    const int lane_lo = lane & 15;  // 0..15
    const int lane_hi = lane >> 4;  // 0 or 1

    v8fp32 c_acc = {0, 0, 0, 0, 0, 0, 0, 0};

    const int m_row = m_tile + lane_lo;
    const int n_row = n_tile + lane_lo;

    for (int k_tile = 0; k_tile < K; k_tile += 16) {
        vec_t a_frag, b_frag;

        if (m_row < T) {
            __builtin_memcpy(&a_frag, X + (int64_t)m_row * K + k_tile, sizeof(a_frag));
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) a_frag[i] = 0;
        }

        if (n_row < N) {
            __builtin_memcpy(&b_frag, W + (int64_t)n_row * K + k_tile, sizeof(b_frag));
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) b_frag[i] = 0;
        }

        c_acc = WmmaTraits<ElemT>::mma(a_frag, b_frag, c_acc);
    }

    const int out_n = n_tile + lane_lo;
    if (out_n < N) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int out_m = m_tile + 2 * i + lane_hi;
            if (out_m < T) {
                float* dst = Y + (int64_t)out_m * ldy + out_n;
                if (beta == 0.0f) {
                    *dst = c_acc[i];
                } else {
                    *dst = beta * (*dst) + c_acc[i];
                }
            }
        }
    }
}

// ===========================================================================
// Variant 2: 64x64_4w - 4 waves per block (128 threads), 64T x 64N tile.
// Double-buffered LDS tile for W (4 KB LDS total), cooperative 128-bit global loads.
// ===========================================================================
template <typename ElemT>
__global__ void gemm_wmma_64x64_4w(
    const uint16_t* __restrict__ X,
    const uint16_t* __restrict__ W,
    float* __restrict__ Y,
    int64_t T, int64_t N, int64_t K, int64_t ldy, float beta) {

    using vec_t = typename WmmaTraits<ElemT>::vec_t;
    const int m_tile = blockIdx.y * 64;
    const int n_tile = blockIdx.x * 64;
    if (m_tile >= T || n_tile >= N) return;

    const int tid = threadIdx.x;   // 0..127
    const int wave_id = tid >> 5;  // 0..3
    const int lane = tid & 31;     // 0..31
    const int lane_lo = lane & 15; // 0..15
    const int lane_hi = lane >> 4; // 0 or 1

    // 4 accumulators per wave covering 4 x 16 N-subtiles
    v8fp32 c_acc0 = {0, 0, 0, 0, 0, 0, 0, 0};
    v8fp32 c_acc1 = {0, 0, 0, 0, 0, 0, 0, 0};
    v8fp32 c_acc2 = {0, 0, 0, 0, 0, 0, 0, 0};
    v8fp32 c_acc3 = {0, 0, 0, 0, 0, 0, 0, 0};

    // Double-buffered LDS tile: 64 rows of N x 16 elements of K (2 KB per buffer)
    alignas(16) __shared__ ElemT b_lds[2][64][16];

    // Thread mapping for cooperative loading of W into LDS (128 threads load 64x16 elements)
    // Each thread loads 8 halfs (16 bytes = uint4)
    const int row_in_tile = tid >> 1;     // 0..63
    const int k_sub = (tid & 1) << 3;     // 0 or 8
    const int actual_n = n_tile + row_in_tile;

    auto load_w_into_lds = [&](int buf, int k_tile) {
        const int actual_k = k_tile + k_sub;
        uint4 val = {0, 0, 0, 0};
        if (actual_n < N && actual_k < K) {
            const void* ptr = reinterpret_cast<const void*>(W + (int64_t)actual_n * K + actual_k);
            val = *reinterpret_cast<const uint4*>(ptr);
        }
        *reinterpret_cast<uint4*>(&b_lds[buf][row_in_tile][k_sub]) = val;
    };

    // Pre-fill buffer 0
    load_w_into_lds(0, 0);
    __syncthreads();

    const int m_row = m_tile + wave_id * 16 + lane_lo;
    int cur_buf = 0;

    for (int k_tile = 0; k_tile < K; k_tile += 16) {
        const int next_buf = 1 - cur_buf;
        const int k_next = k_tile + 16;

        // Prefetch next K-tile of W into LDS
        if (k_next < K) {
            load_w_into_lds(next_buf, k_next);
        }

        // Load A fragment from X for current wave's 16 M-rows
        vec_t a_frag;
        if (m_row < T) {
            __builtin_memcpy(&a_frag, X + (int64_t)m_row * K + k_tile, sizeof(a_frag));
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) a_frag[i] = 0;
        }

        // Read B fragments from LDS and compute WMMA
        vec_t b_frag0, b_frag1, b_frag2, b_frag3;
        __builtin_memcpy(&b_frag0, &b_lds[cur_buf][0 + lane_lo][0], sizeof(vec_t));
        __builtin_memcpy(&b_frag1, &b_lds[cur_buf][16 + lane_lo][0], sizeof(vec_t));
        __builtin_memcpy(&b_frag2, &b_lds[cur_buf][32 + lane_lo][0], sizeof(vec_t));
        __builtin_memcpy(&b_frag3, &b_lds[cur_buf][48 + lane_lo][0], sizeof(vec_t));

        c_acc0 = WmmaTraits<ElemT>::mma(a_frag, b_frag0, c_acc0);
        c_acc1 = WmmaTraits<ElemT>::mma(a_frag, b_frag1, c_acc1);
        c_acc2 = WmmaTraits<ElemT>::mma(a_frag, b_frag2, c_acc2);
        c_acc3 = WmmaTraits<ElemT>::mma(a_frag, b_frag3, c_acc3);

        __syncthreads();
        cur_buf = next_buf;
    }

    // Store C to Y (coalesced 64-byte writes per wave half)
    const int m_tile_wave = m_tile + wave_id * 16;
    auto store_acc = [&](const v8fp32& acc, int n_base) {
        const int out_n = n_base + lane_lo;
        if (out_n < N) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int out_m = m_tile_wave + 2 * i + lane_hi;
                if (out_m < T) {
                    float* dst = Y + (int64_t)out_m * ldy + out_n;
                    if (beta == 0.0f) {
                        *dst = acc[i];
                    } else {
                        *dst = beta * (*dst) + acc[i];
                    }
                }
            }
        }
    };

    store_acc(c_acc0, n_tile + 0);
    store_acc(c_acc1, n_tile + 16);
    store_acc(c_acc2, n_tile + 32);
    store_acc(c_acc3, n_tile + 48);
}

template <typename ElemT>
static inline bool strata_wmma_gemm_dispatch(const uint16_t* X, const uint16_t* W, float* Y,
                                             int64_t T, int64_t N, int64_t K, int64_t ldy, float beta,
                                             void* stream) {
    if (!X || !W || !Y) return false;
    if (T <= 0 || N <= 0 || K <= 0) return false;
    // Runtime gate (review): run only on gfx11 devices, whatever this build compiled for.  The build-time
    // STRATA_WMMA_GFX11 macro controls whether the intrinsics compile; this check controls whether they run.
    {
        int dev = 0;
        hipDeviceProp_t prop{};
        if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&prop, dev) != hipSuccess) return false;
        if (std::strncmp(prop.gcnArchName, "gfx11", 5) != 0) return false;
    }
    if (K % 16 != 0) return false;
    if (beta != 0.0f && beta != 1.0f) return false;
    if (ldy <= 0) ldy = N;
    if (ldy < N) return false;

    hipStream_t s = static_cast<hipStream_t>(stream);

    if (T >= 32 && N >= 32) {
        dim3 block(128);
        dim3 grid((uint32_t)((N + 63) / 64), (uint32_t)((T + 63) / 64), 1);
        gemm_wmma_64x64_4w<ElemT><<<grid, block, 0, s>>>(X, W, Y, T, N, K, ldy, beta);
    } else {
        dim3 block(32);
        dim3 grid((uint32_t)((N + 15) / 16), (uint32_t)((T + 15) / 16), 1);
        gemm_wmma_16x16_1w<ElemT><<<grid, block, 0, s>>>(X, W, Y, T, N, K, ldy, beta);
    }
    return true;
}

bool strata_wmma_gemm_f16(const uint16_t* X, const uint16_t* W, float* Y,
                          int64_t T, int64_t N, int64_t K, int64_t ldy, float beta,
                          void* stream) {
    return strata_wmma_gemm_dispatch<_Float16>(X, W, Y, T, N, K, ldy, beta, stream);
}

bool strata_wmma_gemm_bf16(const uint16_t* X, const uint16_t* W, float* Y,
                           int64_t T, int64_t N, int64_t K, int64_t ldy, float beta,
                           void* stream) {
    return strata_wmma_gemm_dispatch<__bf16>(X, W, Y, T, N, K, ldy, beta, stream);
}

// ===========================================================================
// S23 (opt-in STRATA_PF_GEMM=1): the prompt projections' FP16 GEMM on gfx11 (gfx1151: 30-33 TFLOPS on the 16K-row
// projection shapes where the tuned hipBLASLt reaches 25-27; s23/gemm_probe4.hip).  Block 128 x BN (BN 256 or 128),
// BK 32, 8 waves of 64 x BN/4, LDS double buffer with the next stage prefetched into named registers (arrays there
// went to scratch), grouped tile order (8 row tiles sweep all column tiles: the weights stream from DRAM once per
// group).  FP32 accumulation, K in another order than hipBLASLt: rounding-level, quality-gated.
// ===========================================================================
namespace pfg {
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));
constexpr int BM = 128, BK = 32, LDK = BK + 8, GM = 8;
__device__ __forceinline__ h16 frag(const _Float16* p) {
    const uint4 a = *reinterpret_cast<const uint4*>(p), b = *reinterpret_cast<const uint4*>(p + 8);
    const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    return __builtin_bit_cast(h16, w);
}
template <int BN, int WN>
__global__ void __launch_bounds__(256) kernel(const _Float16* __restrict__ X, const _Float16* __restrict__ W,
                                              float* __restrict__ Y, int M, int N, int K, int ldy, int accumulate,
                                              int ldx, int ldw) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || defined(__gfx1151__)
    constexpr int TN = WN / 16, NB = BN / 64;
    __shared__ __align__(16) _Float16 sA[2][BM][LDK];
    __shared__ __align__(16) _Float16 sB[2][BN][LDK];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    const int wm = wave & 1, wn = wave >> 1;
    const int num_m = (M + BM - 1) / BM, num_n = (N + BN - 1) / BN, b = blockIdx.x;
    const int group = b / (GM * num_n), first_m = group * GM, gsize = min(GM, num_m - first_m);
    const int m0 = (first_m + (b % (GM * num_n)) % gsize) * BM, n0 = ((b % (GM * num_n)) / gsize) * BN;
    const int sr = tid >> 2, sq = tid & 3;
    auto ldA = [&](int k0, int r) -> uint4 {
        return *reinterpret_cast<const uint4*>(X + (size_t) min(m0 + r, M - 1) * ldx + k0 + 8 * sq);
    };
    auto ldB = [&](int k0, int r) -> uint4 {
        return *reinterpret_cast<const uint4*>(W + (size_t) min(n0 + r, N - 1) * ldw + k0 + 8 * sq);
    };
    f8 acc[4][TN];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < TN; ++j) acc[i][j] = f8{0, 0, 0, 0, 0, 0, 0, 0};
    uint4 ra0 = ldA(0, sr), ra1 = ldA(0, sr + 64), rb0 = ldB(0, sr), rb1 = ldB(0, sr + 64), rb2 = rb0, rb3 = rb0;
    if constexpr (NB > 2) { rb2 = ldB(0, sr + 128); rb3 = ldB(0, sr + 192); }
    int buf = 0;
    *reinterpret_cast<uint4*>(&sA[0][sr][8 * sq]) = ra0; *reinterpret_cast<uint4*>(&sA[0][sr + 64][8 * sq]) = ra1;
    *reinterpret_cast<uint4*>(&sB[0][sr][8 * sq]) = rb0; *reinterpret_cast<uint4*>(&sB[0][sr + 64][8 * sq]) = rb1;
    if constexpr (NB > 2) { *reinterpret_cast<uint4*>(&sB[0][sr + 128][8 * sq]) = rb2; *reinterpret_cast<uint4*>(&sB[0][sr + 192][8 * sq]) = rb3; }
    __syncthreads();
    for (int k0 = 0; k0 < K; k0 += BK) {
        const bool more = k0 + BK < K;
        if (more) {
            ra0 = ldA(k0 + BK, sr); ra1 = ldA(k0 + BK, sr + 64); rb0 = ldB(k0 + BK, sr); rb1 = ldB(k0 + BK, sr + 64);
            if constexpr (NB > 2) { rb2 = ldB(k0 + BK, sr + 128); rb3 = ldB(k0 + BK, sr + 192); }
        }
#pragma unroll
        for (int ks = 0; ks < BK; ks += 16) {
            h16 a[4], bb[TN];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = frag(&sA[buf][64 * wm + 16 * i + l16][ks]);
#pragma unroll
            for (int j = 0; j < TN; ++j) bb[j] = frag(&sB[buf][WN * wn + 16 * j + l16][ks]);
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i], bb[j], acc[i][j]);
        }
        if (more) {
            const int nb = buf ^ 1;
            *reinterpret_cast<uint4*>(&sA[nb][sr][8 * sq]) = ra0; *reinterpret_cast<uint4*>(&sA[nb][sr + 64][8 * sq]) = ra1;
            *reinterpret_cast<uint4*>(&sB[nb][sr][8 * sq]) = rb0; *reinterpret_cast<uint4*>(&sB[nb][sr + 64][8 * sq]) = rb1;
            if constexpr (NB > 2) { *reinterpret_cast<uint4*>(&sB[nb][sr + 128][8 * sq]) = rb2; *reinterpret_cast<uint4*>(&sB[nb][sr + 192][8 * sq]) = rb3; }
            __syncthreads();
            buf = nb;
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int n = n0 + WN * wn + 16 * j + l16;
            if (n >= N) continue;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int m = m0 + 64 * wm + 16 * i + 2 * e + hi;
                if (m < M) {
                    float* y = Y + (size_t) m * ldy + n;
                    *y = accumulate ? *y + acc[i][j][e] : acc[i][j][e];
                }
            }
        }
#endif
}
// The 128 x 256 kernel with a 64-k LDS tile (single buffer, the next tile in 12 named staging registers, 2 barriers
// per tile: half the loop trips and barriers per k of the BK 32 double buffer). Each accumulator takes the same
// WMMAs in the same k order as `kernel`: the same bits (s23/gemm_probe11.hip, memcmp). gfx1151, M 16384, both
// operands padded: 38.0 -> 41.3 TFLOPS (N 2560 K 6144), 38.9 -> 40.9 (N 12288 K 2560), 40.3 -> 40.4 (N 10240).
// K a multiple of 64.
__global__ void __launch_bounds__(256) kernel_bk64(const _Float16* __restrict__ X, const _Float16* __restrict__ W,
                                                   float* __restrict__ Y, int M, int N, int K, int ldy, int accumulate,
                                                   int ldx, int ldw) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || defined(__gfx1151__)
    constexpr int BN = 256, WN = 64, BK64 = 64, LDK64 = BK64 + 8;
    __shared__ __align__(16) _Float16 sA[BM][LDK64];
    __shared__ __align__(16) _Float16 sB[BN][LDK64];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    const int wm = wave & 1, wn = wave >> 1;
    const int num_m = (M + BM - 1) / BM, num_n = (N + BN - 1) / BN, b = blockIdx.x;
    const int group = b / (GM * num_n), first_m = group * GM, gsize = min(GM, num_m - first_m);
    const int m0 = (first_m + (b % (GM * num_n)) % gsize) * BM, n0 = ((b % (GM * num_n)) / gsize) * BN;
    const int sr = tid >> 3, sq = tid & 7;            // 32 rows per pass, 8 x 8 halves per row
    auto ldA = [&](int k0, int r) -> uint4 {
        return *reinterpret_cast<const uint4*>(X + (size_t) min(m0 + r, M - 1) * ldx + k0 + 8 * sq);
    };
    auto ldB = [&](int k0, int r) -> uint4 {
        return *reinterpret_cast<const uint4*>(W + (size_t) min(n0 + r, N - 1) * ldw + k0 + 8 * sq);
    };
    uint4 ra0, ra1, ra2, ra3, rb0, rb1, rb2, rb3, rb4, rb5, rb6, rb7;
    auto load = [&](int k0) {
        ra0 = ldA(k0, sr); ra1 = ldA(k0, sr + 32); ra2 = ldA(k0, sr + 64); ra3 = ldA(k0, sr + 96);
        rb0 = ldB(k0, sr); rb1 = ldB(k0, sr + 32); rb2 = ldB(k0, sr + 64); rb3 = ldB(k0, sr + 96);
        rb4 = ldB(k0, sr + 128); rb5 = ldB(k0, sr + 160); rb6 = ldB(k0, sr + 192); rb7 = ldB(k0, sr + 224);
    };
    auto store = [&]() {
        *reinterpret_cast<uint4*>(&sA[sr][8 * sq]) = ra0; *reinterpret_cast<uint4*>(&sA[sr + 32][8 * sq]) = ra1;
        *reinterpret_cast<uint4*>(&sA[sr + 64][8 * sq]) = ra2; *reinterpret_cast<uint4*>(&sA[sr + 96][8 * sq]) = ra3;
        *reinterpret_cast<uint4*>(&sB[sr][8 * sq]) = rb0; *reinterpret_cast<uint4*>(&sB[sr + 32][8 * sq]) = rb1;
        *reinterpret_cast<uint4*>(&sB[sr + 64][8 * sq]) = rb2; *reinterpret_cast<uint4*>(&sB[sr + 96][8 * sq]) = rb3;
        *reinterpret_cast<uint4*>(&sB[sr + 128][8 * sq]) = rb4; *reinterpret_cast<uint4*>(&sB[sr + 160][8 * sq]) = rb5;
        *reinterpret_cast<uint4*>(&sB[sr + 192][8 * sq]) = rb6; *reinterpret_cast<uint4*>(&sB[sr + 224][8 * sq]) = rb7;
    };
    f8 acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = f8{0, 0, 0, 0, 0, 0, 0, 0};
    load(0);
    store();
    __syncthreads();
    const int ar = 64 * wm + l16, br = WN * wn + l16;
    for (int k0 = 0; k0 < K; k0 += BK64) {
        const bool more = k0 + BK64 < K;
        if (more) load(k0 + BK64);
#pragma unroll
        for (int ks = 0; ks < BK64; ks += 16) {
            const h16 a0 = frag(&sA[ar][ks]), a1 = frag(&sA[ar + 16][ks]), a2 = frag(&sA[ar + 32][ks]),
                      a3 = frag(&sA[ar + 48][ks]);
            const h16 b0 = frag(&sB[br][ks]), b1 = frag(&sB[br + 16][ks]), b2 = frag(&sB[br + 32][ks]),
                      b3 = frag(&sB[br + 48][ks]);
#define PFG_ROW(i, a)                                                                                                 \
    acc[i][0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b0, acc[i][0]);                                         \
    acc[i][1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b1, acc[i][1]);                                         \
    acc[i][2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b2, acc[i][2]);                                         \
    acc[i][3] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b3, acc[i][3]);
            PFG_ROW(0, a0) PFG_ROW(1, a1) PFG_ROW(2, a2) PFG_ROW(3, a3)
#undef PFG_ROW
        }
        if (more) {
            __syncthreads();   // every wave is done reading the tile
            store();
            __syncthreads();
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n = n0 + WN * wn + 16 * j + l16;
            if (n >= N) continue;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int m = m0 + 64 * wm + 16 * i + 2 * e + hi;
                if (m < M) {
                    float* y = Y + (size_t) m * ldy + n;
                    *y = accumulate ? *y + acc[i][j][e] : acc[i][j][e];
                }
            }
        }
#endif
}
}  // namespace pfg

bool strata_pf_gemm_f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                        float beta, void* stream) {
    return strata_pf_gemm_f16_ld(X, K, W, K, Y, T, N, K, ldy, beta, stream);
}

bool strata_pf_gemm_f16_ld(const uint16_t* X, int64_t ldx, const uint16_t* W, int64_t ldw, float* Y, int64_t T,
                           int64_t N, int64_t K, int64_t ldy, float beta, void* stream) {
    if (!X || !W || !Y || T < 64 || N < 512 || K % pfg::BK != 0 || K < pfg::BK) return false;
    if (beta != 0.0f && beta != 1.0f) return false;
    if (ldy <= 0) ldy = N;
    if (ldx < K || ldw < K || ldx % 8 != 0 || ldw % 8 != 0 || ldx > (1LL << 30) || ldw > (1LL << 30)) return false;
    if (ldy < N || T > (1LL << 30) || N > (1LL << 30) || K > (1LL << 30)) return false;
    static const bool gfx11 = [] {
        int dev = 0;
        hipDeviceProp_t prop{};
        if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&prop, dev) != hipSuccess) return false;
        return std::strncmp(prop.gcnArchName, "gfx11", 5) == 0;
    }();
    if (!gfx11) return false;
    hipStream_t s = static_cast<hipStream_t>(stream);
    const int acc = beta == 1.0f ? 1 : 0;
    const int64_t mt = (T + pfg::BM - 1) / pfg::BM;
    // STRATA_PF_BK64=0: the BK 32 kernel for the wide shapes too (the A/B; the same bits)
    static const bool bk64 = [] { const char* v = std::getenv("STRATA_PF_BK64"); return !v || v[0] != '0'; }();
    if (N >= 1024 && bk64 && K % 64 == 0) {
        const unsigned grid = (unsigned) (mt * ((N + 255) / 256));
        pfg::kernel_bk64<<<grid, 256, 0, s>>>((const _Float16*) X, (const _Float16*) W, Y, (int) T, (int) N, (int) K,
                                              (int) ldy, acc, (int) ldx, (int) ldw);
    } else if (N >= 1024) {
        const unsigned grid = (unsigned) (mt * ((N + 255) / 256));
        pfg::kernel<256, 64><<<grid, 256, 0, s>>>((const _Float16*) X, (const _Float16*) W, Y, (int) T, (int) N, (int) K,
                                                  (int) ldy, acc, (int) ldx, (int) ldw);
    } else {
        const unsigned grid = (unsigned) (mt * ((N + 127) / 128));
        pfg::kernel<128, 32><<<grid, 256, 0, s>>>((const _Float16*) X, (const _Float16*) W, Y, (int) T, (int) N, (int) K,
                                                  (int) ldy, acc, (int) ldx, (int) ldw);
    }
    return hipGetLastError() == hipSuccess;
}

#else
// Non-HIP / Non-GFX11 compilation fallback
bool strata_pf_gemm_f16(const uint16_t*, const uint16_t*, float*, int64_t, int64_t, int64_t, int64_t, float, void*) {
    return false;
}
bool strata_pf_gemm_f16_ld(const uint16_t*, int64_t, const uint16_t*, int64_t, float*, int64_t, int64_t, int64_t,
                           int64_t, float, void*) {
    return false;
}
bool strata_wmma_gemm_f16(const uint16_t*, const uint16_t*, float*,
                          int64_t, int64_t, int64_t, int64_t, float,
                          void*) {
    return false;
}

bool strata_wmma_gemm_bf16(const uint16_t*, const uint16_t*, float*,
                           int64_t, int64_t, int64_t, int64_t, float,
                           void*) {
    return false;
}
#endif
