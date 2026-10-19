// src/poly/kernels.cu - see the header.  The reference for every formula is llama.cpp's graph of the model and the
// ggml ops it builds.  Where a kernel takes several tokens, blockIdx.y is the token unless it says otherwise.
#include "strata/poly/kernels.hpp"
#include "strata/poly/gpu_failure.hpp"
#include "strata/kernels/q8_1_finite.hpp"
#if defined(STRATA_USE_HIP)
#include "strata/kernels/native_mmvq.hpp"
#endif

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::poly {
namespace {

constexpr int WARP = 32;
constexpr int HC = 4;   // hyper-connection streams; the model's hc_mult, checked at load
constexpr int KEY_BLOCK = 16;   // cache rows one block of dsa_keys_kernel copies
// attn_score_kernel: a block's warps, a row and a group each; the values of a key a thread reads at a step; the
// queries a key's values are kept for
constexpr int SCORE_WARPS = 8, SCORE_VALUES = 4, SCORE_QUERIES = 4;
// attn_value_kernel: a block's value dimensions, a sum's shares, a block's queries
constexpr int VALUE_DIMS = 32, VALUE_SHARES = 32, VALUE_QUERIES = 8;
// The attention of a call of a few tokens at Qwen3.5/3.6's shape (attend(), STRATA_POLY_ATTN): the cache rows a
// warp of attn_score_rows_kernel takes, and the rows of a part of attn_value_part_kernel's span, counted from row 0
constexpr int SCORE_ROWS = 16, VALUE_PART = 512;
constexpr int KDA_TERMS = 16, KDA_COLUMNS = 16;   // kda_state_kernel: a thread's values, a block's columns

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffff, x, o, WARP);
    return x;
}

// One value per thread summed over the block.  blockDim.x is a multiple of 32, and `part` has a float a warp: 8
// where a block is 256 threads at most, 32 where it can be 1024.
__device__ __forceinline__ float block_sum(float x, float* part) {
    x = warp_sum(x);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = x;
    __syncthreads();
    float s = 0.0f;
    for (int k = 0; k < int(blockDim.x) / WARP; ++k) s += part[k];
    __syncthreads();   // part is free again
    return s;
}

// A value summed over each run of `lanes` threads of a warp; lanes is a power of two.
__device__ __forceinline__ float lanes_sum(float x, int lanes) {
    for (int o = lanes / 2; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffff, x, o, WARP);
    return x;
}

__device__ __forceinline__ float block_max(float x, float* part) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, o, WARP));
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = x;
    __syncthreads();
    float m = part[0];
    for (int k = 1; k < int(blockDim.x) / WARP; ++k) m = fmaxf(m, part[k]);
    __syncthreads();
    return m;
}

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float ldw(const float* w, size_t i) { return w[i]; }
__device__ __forceinline__ float ldw(const uint16_t* w, size_t i) { return __uint_as_float(uint32_t(w[i]) << 16); }

template<typename W>
__global__ void gemv_kernel(const W* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, int n_in,
                            int n_out, int rows_per_x) {
    __shared__ float part[32];
    const int o = blockIdx.x;
    const size_t tok = blockIdx.y;
    const W* wr = w + size_t(o) * n_in;
    const float* xr = x + (tok * size_t(n_out / rows_per_x) + size_t(o / rows_per_x)) * n_in;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n_in; i += blockDim.x) acc += ldw(wr, i) * xr[i];
    acc = block_sum(acc, part);
    if (threadIdx.x == 0) y[tok * n_out + o] = acc;
}

// The threads of a block that sums over n values.  A thread reads its values one after the other, and a kernel
// this small takes as long as those reads in a row: four a thread at most, while a block has the threads.  With
// 256 threads for 4096 values, GLM 5.3 Flash's layer norms, small products and stream mixing were 2.7 of the 11
// ms a token its card worked before a layer's router had its scores.
int sum_threads(int n) {
    int threads = 128;
    while (threads < 1024 && threads * 4 < n) threads *= 2;
    return threads;
}

template<typename W>
void gemv(const W* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, int rows, void* stream) {
    const int threads = n_in >= 128 ? sum_threads(n_in) : 32;
    gemv_kernel<W><<<dim3(n_out, rows), threads, 0, (cudaStream_t) stream>>>(w, x, y, n_in, n_out, rows_per_x);
}

__global__ void bf16_to_f32_kernel(const uint16_t* __restrict__ w, float* __restrict__ out, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = ldw(w, i);
}

__global__ void rms_norm_kernel(const float* __restrict__ x, const float* __restrict__ w, float* y, int n,
                                float eps, int y_stride) {
    __shared__ float part[32];
    const float* xr = x + size_t(blockIdx.x) * n;
    float* yr = y + size_t(blockIdx.x) * y_stride;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, part);
    const float scale = 1.0f / sqrtf(ss / float(n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = w ? xr[i] * scale * w[i] : xr[i] * scale;
}

__global__ void layer_norm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                  const float* __restrict__ b, float* y, int n, float eps) {
    __shared__ float part[8];
    const float* xr = x + size_t(blockIdx.x) * n;
    float* yr = y + size_t(blockIdx.x) * n;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i];
    const float mean = block_sum(s, part) / float(n);
    float v = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float d = xr[i] - mean;
        v += d * d;
    }
    const float scale = 1.0f / sqrtf(block_sum(v, part) / float(n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = (xr[i] - mean) * scale * w[i] + b[i];
}

// mix[o] = fn row o . r for o < n_mix; mix[n_mix] = sum of r^2 (the flat RMS norm's input)
__global__ void hc_mix_kernel(const float* __restrict__ r, const float* __restrict__ fn, float* __restrict__ mix,
                              int n, int n_mix) {
    __shared__ float part[32];
    const int o = blockIdx.x;
    r += size_t(blockIdx.y) * n;
    mix += size_t(blockIdx.y) * (n_mix + 1);
    float acc = 0.0f;
    if (o < n_mix) {
        const float* fr = fn + size_t(o) * n;
        for (int i = threadIdx.x; i < n; i += blockDim.x) acc += fr[i] * r[i];
    } else {
        for (int i = threadIdx.x; i < n; i += blockDim.x) acc += r[i] * r[i];
    }
    acc = block_sum(acc, part);
    if (threadIdx.x == 0) mix[o] = acc;
}

__device__ void comb_norm_cols(float* comb, float eps) {
    for (int dst = 0; dst < HC; ++dst) {
        float sum = eps;
        for (int src = 0; src < HC; ++src) sum += comb[dst + HC * src];
        const float inv = 1.0f / sum;
        for (int src = 0; src < HC; ++src) comb[dst + HC * src] *= inv;
    }
}

__device__ void comb_norm_rows(float* comb, float eps) {
    for (int src = 0; src < HC; ++src) {
        float sum = eps;
        for (int dst = 0; dst < HC; ++dst) sum += comb[dst + HC * src];
        const float inv = 1.0f / sum;
        for (int dst = 0; dst < HC; ++dst) comb[dst + HC * src] *= inv;
    }
}

__global__ void hc_read_kernel(const float* __restrict__ r, const float* __restrict__ mix,
                               const float* __restrict__ scale, const float* __restrict__ base,
                               float* __restrict__ hcw, float* __restrict__ cur, int n_embd, float norm_eps,
                               float hc_eps, int iters) {
    __shared__ float pre[HC];
    r += size_t(blockIdx.y) * HC * n_embd;
    mix += size_t(blockIdx.y) * (HC * (2 + HC) + 1);
    hcw += size_t(blockIdx.y) * (HC + HC * HC);
    cur += size_t(blockIdx.y) * n_embd;
    if (threadIdx.x == 0) {
        const float rs = 1.0f / sqrtf(mix[HC * (2 + HC)] / float(HC * n_embd) + norm_eps);
        for (int h = 0; h < HC; ++h) pre[h] = sigmoid(mix[h] * rs * scale[0] + base[h]) + hc_eps;
        if (blockIdx.x == 0) {
            for (int h = 0; h < HC; ++h) hcw[h] = 2.0f * sigmoid(mix[HC + h] * rs * scale[1] + base[HC + h]);
            float comb[HC * HC];
            for (int src = 0; src < HC; ++src) {   // softmax over dst
                float mx = -INFINITY;
                for (int dst = 0; dst < HC; ++dst) {
                    const int k = dst + HC * src;
                    comb[k] = mix[2 * HC + k] * rs * scale[2] + base[2 * HC + k];
                    mx = fmaxf(mx, comb[k]);
                }
                float sum = 0.0f;
                for (int dst = 0; dst < HC; ++dst) {
                    const int k = dst + HC * src;
                    comb[k] = expf(comb[k] - mx);
                    sum += comb[k];
                }
                const float inv = 1.0f / sum;
                for (int dst = 0; dst < HC; ++dst) comb[dst + HC * src] = comb[dst + HC * src] * inv + hc_eps;
            }
            comb_norm_cols(comb, hc_eps);
            for (int it = 1; it < iters; ++it) {
                comb_norm_rows(comb, hc_eps);
                comb_norm_cols(comb, hc_eps);
            }
            for (int k = 0; k < HC * HC; ++k) hcw[HC + k] = comb[k];
        }
    }
    __syncthreads();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    float s = 0.0f;
    for (int h = 0; h < HC; ++h) s += r[size_t(h) * n_embd + i] * pre[h];
    cur[i] = s;
}

__global__ void hc_post_kernel(const float* __restrict__ x, const float* __restrict__ hcw, float* r, int n_embd) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    x += size_t(blockIdx.y) * n_embd;
    hcw += size_t(blockIdx.y) * (HC + HC * HC);
    r += size_t(blockIdx.y) * HC * n_embd;
    float in[HC], out[HC];
    for (int s = 0; s < HC; ++s) in[s] = r[size_t(s) * n_embd + i];
    const float xi = x[i];
    for (int dst = 0; dst < HC; ++dst) {
        float v = xi * hcw[dst];
        for (int src = 0; src < HC; ++src) v += in[src] * hcw[HC + dst + HC * src];
        out[dst] = v;
    }
    for (int s = 0; s < HC; ++s) r[size_t(s) * n_embd + i] = out[s];
}

__global__ void hc_mean_kernel(const float* __restrict__ r, float* __restrict__ y, int n_embd) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    r += size_t(blockIdx.y) * HC * n_embd;
    float acc = r[i];
    for (int s = 1; s < HC; ++s) acc += r[size_t(s) * n_embd + i];
    y[size_t(blockIdx.y) * n_embd + i] = acc * (1.0f / HC);
}

// d_conv is 4: three past inputs per channel, then the new one.  A thread is a channel and walks the tokens.
// The state as it is after each of the first n_snap tokens goes to snap, a state `stride` floats after the other.
__global__ void kda_conv_kernel(const float* __restrict__ proj, float* __restrict__ state,
                                const float* __restrict__ wq, const float* __restrict__ wk,
                                const float* __restrict__ wv, float* __restrict__ out, int d_inner, int rows,
                                float* __restrict__ snap, size_t stride, int n_snap) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= 3 * d_inner) return;
    const int m = t / d_inner, c = t % d_inner;
    const float* w = (m == 0 ? wq : m == 1 ? wk : wv) + 4 * size_t(c);
    float* st = state + size_t(t) * 3;
    float s0 = st[0], s1 = st[1], s2 = st[2];
    const size_t first = size_t(m) * rows * d_inner + c;
    for (int row = 0; row < rows; ++row) {
        const size_t at = first + size_t(row) * d_inner;
        const float x = proj[at];
        const float y = s0 * w[0] + s1 * w[1] + s2 * w[2] + x * w[3];
        s0 = s1;
        s1 = s2;
        s2 = x;
        out[at] = y * sigmoid(y);
        if (row < n_snap) {
            float* m = snap + size_t(row) * stride + size_t(t) * 3;
            m[0] = s0;
            m[1] = s1;
            m[2] = s2;
        }
    }
    st[0] = s0;
    st[1] = s1;
    st[2] = s2;
}

// One block per head, and a column of the state to every `lanes` threads of it: a thread holds every lanes-th
// value of its column, so a column's two sums are a few terms a thread and a sum over its threads.  A thread a
// column read the column's 128 values one after the other, twice, and that was 25 microseconds a token.  The
// block walks the tokens.  s[i + hd * j] holds key dimension i of column j.  blockDim.x is hd * lanes.
__global__ void kda_step_kernel(const float* __restrict__ qkv, const float* __restrict__ g1,
                                const float* __restrict__ a, const float* __restrict__ beta, float* __restrict__ s,
                                const float* __restrict__ g2, const float* __restrict__ o_norm,
                                float* __restrict__ out, int hd, int n_head, int rows, float lower_bound,
                                float norm_eps, float* __restrict__ snap, size_t stride, int n_snap) {
    __shared__ float part[32];
    __shared__ float qn[256], kn[256], decay[256];
    const int lanes = int(blockDim.x) / hd;
    const int h = blockIdx.x, j = threadIdx.x / lanes, lane = threadIdx.x % lanes;
    const bool first = lane == 0;   // the thread that stands for its column in the sums over the head
    const size_t d_inner = size_t(hd) * n_head, plane = d_inner * rows;
    float* col = s + (size_t(h) * hd + j) * hd;
    for (int row = 0; row < rows; ++row) {
        const size_t at = size_t(row) * d_inner + size_t(h) * hd + j;
        const float q = qkv[at], k = qkv[plane + at], v = qkv[2 * plane + at];
        const float qs = block_sum(first ? q * q : 0.0f, part);
        const float ks = block_sum(first ? k * k : 0.0f, part);
        if (first) {
            qn[j] = q / sqrtf(qs + 1e-6f) * (1.0f / sqrtf(float(hd)));
            kn[j] = k / sqrtf(ks + 1e-6f);
            // the decay gate of key dimension j: exp(lower_bound * sigmoid(-(g1 * a)))
            decay[j] = expf(lower_bound / (1.0f + expf(g1[at] * a[h])));
        }
        __syncthreads();
        // Which product is fused into which sum is spelled out, here and in kda_state_kernel: left to the
        // compiler, the two came out with other last bits.
        float sk = 0.0f;
        for (int i = lane; i < hd; i += lanes) sk = fmaf(__fmul_rn(col[i], decay[i]), kn[i], sk);
        sk = lanes_sum(sk, lanes);
        const float d = (v - sk) * sigmoid(beta[size_t(row) * n_head + h]);
        float o = 0.0f;
        float* m = row < n_snap ? snap + size_t(row) * stride + (size_t(h) * hd + j) * hd : nullptr;
        for (int i = lane; i < hd; i += lanes) {
            const float n = fmaf(col[i], decay[i], __fmul_rn(kn[i], d));
            col[i] = n;
            if (m) m[i] = n;
            o = fmaf(n, qn[i], o);
        }
        o = lanes_sum(o, lanes);
        // the sum's last barrier also ends this token's reads of qn, kn and decay
        const float os = block_sum(first ? o * o : 0.0f, part);
        if (first) out[at] = o / sqrtf(os / float(hd) + norm_eps) * o_norm[j] * sigmoid(g2[at]);
    }
}

// kda_step_kernel in three for a prompt batch.  There a head's block walks the tokens with every column's values
// in the card's memory, and waits for its warps three times a token, for sums that need no token before them.
// Here the norms of q and k come first for every token at once, the state's columns walk the tokens in registers,
// a few columns a block so that the blocks spread over the card, and the output's norm comes last.  The sums add
// in kda_step_kernel's order, so a token comes out with the same bits.

// The sum over a head's values that kda_step_kernel's block_sum makes of them, a thread a value here.  There they
// lie `lanes` threads apart, so a warp sums WARP / lanes of them (`group`) and the block adds the warps' sums in
// turn.  part has a float a group.
__device__ __forceinline__ float kda_head_sum(float x, float* part, int group) {
    x = lanes_sum(x, group);
    if (threadIdx.x % group == 0) part[threadIdx.x / group] = x;
    __syncthreads();
    float s = 0.0f;
    for (int k = 0; k < int(blockDim.x) / group; ++k) s += part[k];
    __syncthreads();   // part is free again
    return s;
}

// One block a token and head, a thread a value: q and k normed in place, and g1 made the decay of its key
// dimension.
__global__ void kda_gates_kernel(float* __restrict__ qkv, float* __restrict__ g1, const float* __restrict__ a,
                                 int n_head, int rows, float lower_bound, int group) {
    __shared__ float part[32];
    const int hd = blockDim.x;
    const size_t at = size_t(blockIdx.x) * hd + threadIdx.x, plane = size_t(hd) * n_head * rows;
    const float q = qkv[at], k = qkv[plane + at];
    // the squares are rounded before they are added, as there
    const float qs = kda_head_sum(__fmul_rn(q, q), part, group);
    const float ks = kda_head_sum(__fmul_rn(k, k), part, group);
    qkv[at] = q / sqrtf(qs + 1e-6f) * (1.0f / sqrtf(float(hd)));
    qkv[plane + at] = k / sqrtf(ks + 1e-6f);
    g1[at] = expf(lower_bound / (1.0f + expf(g1[at] * a[blockIdx.x % n_head])));
}

// The state's update with what kda_gates_kernel left.  A column has `lanes` threads as in kda_step_kernel, each
// with KDA_TERMS of its values, and a block has KDA_COLUMNS of a head's columns.  out gets the output before its
// norm.
__global__ void kda_state_kernel(const float* __restrict__ qkv, const float* __restrict__ decay,
                                 const float* __restrict__ beta, float* __restrict__ s, float* __restrict__ out,
                                 int n_head, int rows) {
    const int lanes = int(blockDim.x) / KDA_COLUMNS, hd = lanes * KDA_TERMS;
    const int per_head = hd / KDA_COLUMNS;   // blocks a head
    const int h = blockIdx.x / per_head, j = (blockIdx.x % per_head) * KDA_COLUMNS + threadIdx.x / lanes;
    const int lane = threadIdx.x % lanes;
    const size_t d_inner = size_t(hd) * n_head, plane = d_inner * rows;
    float* col = s + (size_t(h) * hd + j) * hd + lane;
    float c[KDA_TERMS];
#pragma unroll
    for (int t = 0; t < KDA_TERMS; ++t) c[t] = col[t * lanes];
    for (int row = 0; row < rows; ++row) {
        const size_t head = size_t(row) * d_inner + size_t(h) * hd;
        const float* qn = qkv + head + lane;
        const float* kn = qkv + plane + head + lane;
        const float* dc = decay + head + lane;
        const float v = qkv[2 * plane + head + j];
        float sk = 0.0f;   // kda_step_kernel's fused products
#pragma unroll
        for (int t = 0; t < KDA_TERMS; ++t) sk = fmaf(__fmul_rn(c[t], dc[t * lanes]), kn[t * lanes], sk);
        sk = lanes_sum(sk, lanes);
        const float d = (v - sk) * sigmoid(beta[size_t(row) * n_head + h]);
        float o = 0.0f;
#pragma unroll
        for (int t = 0; t < KDA_TERMS; ++t) {
            const float n = fmaf(c[t], dc[t * lanes], __fmul_rn(kn[t * lanes], d));
            c[t] = n;
            o = fmaf(n, qn[t * lanes], o);
        }
        o = lanes_sum(o, lanes);
        if (lane == 0) out[head + j] = o;
    }
#pragma unroll
    for (int t = 0; t < KDA_TERMS; ++t) col[t * lanes] = c[t];
}

// One block a token and head, a thread a value: the output's norm, its weights and the gate.
__global__ void kda_out_kernel(float* __restrict__ out, const float* __restrict__ g2,
                               const float* __restrict__ o_norm, float norm_eps, int group) {
    __shared__ float part[32];
    const int hd = blockDim.x;
    const size_t at = size_t(blockIdx.x) * hd + threadIdx.x;
    const float o = out[at];
    const float os = kda_head_sum(__fmul_rn(o, o), part, group);
    out[at] = o / sqrtf(os / float(hd) + norm_eps) * o_norm[threadIdx.x] * sigmoid(g2[at]);
}

__global__ void store_f16_kernel(const float* __restrict__ x, half* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(x[i]);
}

// Rows from the position the card holds when the launch runs.  SPILL: the rows from `split` on go to `host`.
template <bool SPILL>
__global__ void store_f16_at_kernel(const float* __restrict__ x, half* __restrict__ cache, int n, int rows,
                                    const long long* __restrict__ at, int ring, half* __restrict__ host,
                                    long long split) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * rows) return;
    const long long row = ring > 0 ? (*at + i / n) % ring : *at + i / n;
    if constexpr (SPILL)
        if (row >= split) {
            host[size_t(row - split) * n + i % n] = __float2half(x[i]);
            return;
        }
    cache[size_t(row) * n + i % n] = __float2half(x[i]);
}

// A row of a cache split at `split` (KvSpill): in VRAM below it, in pinned host memory from it on.
__device__ __forceinline__ const half* spill_row(const half* dev, const half* host, long long split, long long row,
                                                 size_t width) {
    return row < split ? dev + size_t(row) * width : host + size_t(row - split) * width;
}

// Rows [row0, row0 + m) of a split cache, one after the other into `out` (VRAM): 16 bytes a thread.
__global__ void kv_gather_kernel(const half* __restrict__ dev, const half* __restrict__ host, long long split,
                                 long long row0, int m, int width, half* __restrict__ out) {
    const int per = width / 8;   // uint4 a row
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(m) * per) return;
    const long long row = row0 + (long long) (i / per);
    const uint4* src = reinterpret_cast<const uint4*>(spill_row(dev, host, split, row, width));
    reinterpret_cast<uint4*>(out)[i] = src[i % per];
}

// How many cache rows token `tok` of a call from position pos attends to: every row up to its own while it is
// dense, then its list.
__host__ __device__ inline int dsa_keys(int tok, long long pos, int dense_rows, int top_rows, int kpool) {
    const long long n = pos + tok + 1;
    return tok >= dense_rows ? top_rows + int(n % kpool) : int(n);
}

// The keys of token first + blockIdx.y as float rows, n_keys of them; the rows past its last key are zero.  A
// block copies KEY_BLOCK rows.
__global__ void dsa_keys_kernel(const half* __restrict__ cache, const int32_t* __restrict__ sel_all, int sel_stride,
                                int top_rows, int kpool, long long pos, int first, int dense_rows,
                                float* __restrict__ keys, int rank, int n_keys) {
    const int tok = first + blockIdx.y;
    const int n_sel = dsa_keys(tok, pos, dense_rows, top_rows, kpool);
    const int32_t* sel = tok >= dense_rows ? sel_all + size_t(tok - dense_rows) * sel_stride : nullptr;
    float* dst = keys + size_t(blockIdx.y) * n_keys * rank;
    const int j0 = blockIdx.x * KEY_BLOCK;
    for (int i = threadIdx.x; i < KEY_BLOCK * rank; i += blockDim.x) {
        const int j = j0 + i / rank, r = i % rank;
        if (j >= n_keys) return;
        dst[size_t(j) * rank + r] = j < n_sel ? __half2float(cache[size_t(sel ? sel[j] : j) * rank + r]) : 0.0f;
    }
}

// Softmax in place over a head's scores of its token's keys; the entries past the last key become zero.  One
// block per head and token.
__global__ void dsa_softmax_kernel(float* score, int n_keys, int top_rows, int kpool, long long pos, int first,
                                   int dense_rows) {
    __shared__ float part[8];
    const int n_sel = dsa_keys(first + blockIdx.y, pos, dense_rows, top_rows, kpool);
    float* row = score + (size_t(blockIdx.y) * gridDim.x + blockIdx.x) * n_keys;
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < n_sel; j += blockDim.x) mx = fmaxf(mx, row[j]);
    mx = block_max(mx, part);
    float sum = 0.0f;
    for (int j = threadIdx.x; j < n_sel; j += blockDim.x) {
        row[j] = expf(row[j] - mx);
        sum += row[j];
    }
    const float inv = 1.0f / block_sum(sum, part);
    for (int j = threadIdx.x; j < n_keys; j += blockDim.x) row[j] = j < n_sel ? row[j] * inv : 0.0f;
}

// One block per token; `first` is the first token's cache row.
__global__ void idx_store_kernel(const float* __restrict__ key, const float* __restrict__ gate,
                                 float* __restrict__ first, int dim) {
    const int d = threadIdx.x;
    if (d >= dim) return;
    float* row = first + size_t(blockIdx.x) * 3 * dim;
    row[d] = key[size_t(blockIdx.x) * dim + d];
    row[dim + d] = gate[size_t(blockIdx.x) * dim + d];
}

// One block per token from position pos; the ones that do not end a pool have nothing to do.
__global__ void idx_pool_kernel(float* cache, const float* __restrict__ ape, int dim, int kpool, long long pos) {
    const int d = threadIdx.x;
    const long long last = pos + blockIdx.x;
    if (d >= dim || (last + 1) % kpool != 0) return;
    const float* first = cache + size_t(last - kpool + 1) * 3 * dim;
    float mx = -INFINITY;
    for (int k = 0; k < kpool; ++k) mx = fmaxf(mx, first[size_t(k) * 3 * dim + dim + d] + ape[d + dim * k]);
    float sum = 0.0f, acc = 0.0f;
    for (int k = 0; k < kpool; ++k) {
        const float* row = first + size_t(k) * 3 * dim;
        const float e = expf(row[dim + d] + ape[d + dim * k] - mx);
        sum += e;
        acc += e * row[d];
    }
    cache[size_t(last) * 3 * dim + 2 * dim + d] = acc / sum;
}

// One block per pool and token, 32 threads over the indexer heads.
__global__ void idx_score_kernel(const float* __restrict__ q, const float* __restrict__ w,
                                 const float* __restrict__ cache, float* __restrict__ score, int score_stride,
                                 int dim, int n_head, int kpool, long long pos, float wscale) {
    const size_t tok = blockIdx.y;
    if (blockIdx.x >= (pos + tok + 1) / kpool) return;   // not a whole pool at this token yet
    const float* key = cache + (size_t(blockIdx.x) * kpool + kpool - 1) * 3 * dim + 2 * dim;
    q += tok * n_head * dim;
    w += tok * n_head;
    float acc = 0.0f;
    for (int h = threadIdx.x; h < n_head; h += WARP) {
        const float* qh = q + size_t(h) * dim;
        float dot = 0.0f;
        for (int d = 0; d < dim; ++d) dot += qh[d] * key[d];
        acc += fmaxf(dot, 0.0f) * (w[h] * wscale);
    }
    acc = warp_sum(acc);
    if (threadIdx.x == 0) score[tok * score_stride + blockIdx.x] = acc;
}

// idx_score_kernel's sum over the heads for a prompt batch, whose products cuBLAS made: dot holds q[h] . pooled(p)
// for a token and head's n_pool pools together.  A thread a pool, blockIdx.y the token.
__global__ void idx_weigh_kernel(const float* __restrict__ dot, const float* __restrict__ w,
                                 float* __restrict__ score, int score_stride, int n_pool, int n_head, int kpool,
                                 long long pos, float wscale) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t tok = blockIdx.y;
    if (p >= (pos + tok + 1) / kpool) return;   // not a whole pool at this token yet
    dot += tok * n_head * n_pool + p;
    w += tok * n_head;
    float acc = 0.0f;
    for (int h = 0; h < n_head; ++h) acc += fmaxf(dot[size_t(h) * n_pool], 0.0f) * (w[h] * wscale);
    score[tok * score_stride + p] = acc;
}

// A score as an integer: a higher score is a larger one, and -0 is +0, as the host compares them.
__device__ __forceinline__ uint32_t idx_key(float s) {
    const uint32_t b = __float_as_uint(s + 0.0f);
    return b & 0x80000000u ? ~b : b | 0x80000000u;
}

// One value per thread summed over the block, as block_sum for counts.
__device__ __forceinline__ int block_count(int x, int* part) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffff, x, o, WARP);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = x;
    __syncthreads();
    int s = 0;
    for (int k = 0; k < int(blockDim.x) / WARP; ++k) s += part[k];
    __syncthreads();
    return s;
}

// A token's n_top pools with the highest scores, the lower pool first among equals, as rows of its list in that
// order, then the rows no pool has yet: what the host's partial sort makes of the scores.  One block of 256 a
// token; a thread has a run of the token's pools.  The block's shared memory holds the chosen pools and then
// their keys, n_top each.
__global__ void idx_select_kernel(const float* __restrict__ score, int score_stride, int32_t* __restrict__ sel,
                                  int sel_stride, int n_top, int kpool, long long pos) {
    extern __shared__ uint32_t chosen[];
    __shared__ int part[8], above[256], level[256];
    const long long at = pos + blockIdx.x;
    const int n_pool = int((at + 1) / kpool);
    const float* s = score + size_t(blockIdx.x) * score_stride;
    const int per = (n_pool + int(blockDim.x) - 1) / int(blockDim.x);
    const int lo = threadIdx.x * per, hi = min(lo + per, n_pool);
    // the cut: the largest key that n_top pools reach, a bit at a time
    uint32_t cut = 0;
    for (int b = 31; b >= 0; --b) {
        const uint32_t c = cut | (1u << b);
        int n = 0;
        for (int i = lo; i < hi; ++i) n += idx_key(s[i]) >= c;
        if (block_count(n, part) >= n_top) cut = c;
    }
    // the pools above the cut, and of those on it the lowest, in the order of the pools
    int g = 0, e = 0;
    for (int i = lo; i < hi; ++i) {
        const uint32_t k = idx_key(s[i]);
        g += k > cut;
        e += k == cut;
    }
    above[threadIdx.x] = g;
    level[threadIdx.x] = e;
    __syncthreads();
    int g0 = 0, e0 = 0, g_all = 0;   // in the threads before this one, and above the cut in all
    for (int k = 0; k < int(blockDim.x); ++k) {
        if (k < int(threadIdx.x)) g0 += above[k], e0 += level[k];
        g_all += above[k];
    }
    const int need = n_top - g_all;
    int out = g0 + min(e0, need);
    for (int i = lo; i < hi; ++i) {
        const uint32_t k = idx_key(s[i]);
        if (k > cut || (k == cut && e0++ < need)) {
            chosen[out] = i;
            chosen[n_top + out++] = k;
        }
    }
    __syncthreads();
    // a chosen pool's place: behind those with a higher score and the lower pools with its own
    for (int c = threadIdx.x; c < n_top; c += blockDim.x) {
        const uint32_t mine = chosen[n_top + c];
        int rank = 0;
        for (int k = 0; k < n_top; ++k) rank += chosen[n_top + k] > mine || (chosen[n_top + k] == mine && k < c);
        int32_t* row = sel + size_t(blockIdx.x) * sel_stride + size_t(rank) * kpool;
        for (int j = 0; j < kpool; ++j) row[j] = int32_t(chosen[c]) * kpool + j;
    }
    if (threadIdx.x != 0) return;
    int32_t* tail = sel + size_t(blockIdx.x) * sel_stride + size_t(n_top) * kpool;
    for (long long p = (long long) n_pool * kpool; p <= at; ++p) *tail++ = int32_t(p);
}

__global__ void swiglu_clamp_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                                    float* __restrict__ out, int n, float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = fminf(gate[i], limit);
    const float u = fminf(fmaxf(up[i], -limit), limit);
    out[i] = g / (1.0f + expf(-g)) * u;
}

__global__ void axpy_kernel(const float* __restrict__ x, float a, float* __restrict__ acc, int n, bool add) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    acc[i] = add ? acc[i] + x[i] * a : x[i] * a;
}

// The sums may be one and the same, so a thread adds its value of each vector in turn.
__global__ void add_scaled_kernel(ScaledRows rows, int count, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int k = 0; k < count; ++k) rows.acc[k][i] = rows.acc[k][i] + rows.x[k][i] * rows.w[k];
}

__global__ void swiglu_clamp_rows_kernel(const float* __restrict__ gate_up, float* __restrict__ out, int n,
                                         float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* row = gate_up + size_t(blockIdx.y) * 2 * n;
    const float g = fminf(row[i], limit);
    const float u = fminf(fmaxf(row[n + i], -limit), limit);
    out[size_t(blockIdx.y) * n + i] = g / (1.0f + expf(-g)) * u;
}

__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x)));
}

__global__ void geglu_kernel(const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ out,
                             int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = gelu(gate[i]) * up[i];
}

__global__ void geglu_rows_kernel(const float* __restrict__ gate_up, float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* row = gate_up + size_t(blockIdx.y) * 2 * n;
    out[size_t(blockIdx.y) * n + i] = gelu(row[i]) * row[n + i];
}

__global__ void moe_sum_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ acc,
                               int n, int n_used) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const size_t first = size_t(blockIdx.y) * n_used;
    float s = 0.0f;
    for (int k = 0; k < n_used; ++k) s += w[first + k] * x[(first + k) * n + i];
    acc[size_t(blockIdx.y) * n + i] += s;
}

__global__ void add_bias_kernel(const float* __restrict__ b, float* __restrict__ acc, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) acc[size_t(blockIdx.y) * n + i] += b[i];
}

// One block per head and token.  The pairs of the rotation are i and i + n_rot / 2 for i below n_rot / 2.  With
// axes, a token has a position on each of three axes, and pair j turns by the one on axis (j % sections) % 3.
// With `at`, the first token's position is what the card holds there when the launch runs.
__global__ void head_norm_rope_kernel(const float* __restrict__ x, int x_stride, const float* __restrict__ w,
                                      float* __restrict__ y, int dim, int heads, int n_rot, float theta_scale,
                                      const float* __restrict__ freq, long long pos, float eps,
                                      const int* __restrict__ axes, int sections,
                                      const long long* __restrict__ at) {
    __shared__ float part[8];
    const float* xr = x + size_t(blockIdx.x) * x_stride;
    float* yr = y + size_t(blockIdx.x) * dim;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, part);
    const float scale = 1.0f / sqrtf(ss / float(dim) + eps);
    const int tok = blockIdx.x / heads;
    const float own = float((at ? *at : pos) + tok);
    const int half = n_rot / 2;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        if (i >= n_rot) {
            yr[i] = w ? xr[i] * scale * w[i] : xr[i] * scale;
            continue;
        }
        const int j = i < half ? i : i - half;
        const float p = axes ? float(axes[size_t(tok) * 3 + (j % sections) % 3]) : own;
        const float turn = p * powf(theta_scale, float(j));
        const float theta = freq ? turn / freq[j] : turn;
        const float x0 = w ? xr[j] * scale * w[j] : xr[j] * scale;
        const float x1 = w ? xr[j + half] * scale * w[j + half] : xr[j + half] * scale;
        yr[i] = i < half ? x0 * cosf(theta) - x1 * sinf(theta) : x0 * sinf(theta) + x1 * cosf(theta);
    }
}

// The rows the token at position p attends: how many, and the first of them.  They end at its own row, or at row
// `last` when that is given: the tokens of the call then see each other.
__device__ __forceinline__ int attn_rows(long long p, int window, long long last, long long* lo) {
    *lo = window > 0 && p >= window ? p - window + 1 : 0;
    return int((last < 0 ? p : last) - *lo) + 1;
}

// The rows the `tokens` tokens from position p attend between them: how many, and the first of them.  A launch's
// scores lie in that order, so that its tokens read a row of the caches once between them.
__device__ __forceinline__ int attn_span(long long p, int tokens, int window, long long last, long long* lo) {
    long long lo_last;
    const int n = attn_rows(p + tokens - 1, window, last, &lo_last);
    attn_rows(p, window, last, lo);
    return int(lo_last - *lo) + n;
}

// The scores of the key k with N queries that lie one after the other from q, for the warp of attn_score_kernel.
// score: where the first of them has its score of the key's row; the others lie n_keys floats apart.
template <int N>
__device__ __forceinline__ void attn_scores(const float* __restrict__ q, const half* __restrict__ k,
                                            float* __restrict__ score, int n_keys, int dim, float scale) {
    static_assert(SCORE_VALUES == 4, "a step is four halves of the key in one read and a float4 of a query");
    const int lane = threadIdx.x % WARP;
    const float* qi[N];
    float acc[N];
#pragma unroll
    for (int i = 0; i < N; ++i) {
        qi[i] = q + size_t(i) * dim;
        acc[i] = 0.0f;
    }
    for (int d = SCORE_VALUES * lane; d < dim; d += SCORE_VALUES * WARP) {
        const unsigned long long bits = *reinterpret_cast<const unsigned long long*>(k + d);
        float x[SCORE_VALUES];
#pragma unroll
        for (int j = 0; j < SCORE_VALUES; ++j)
            x[j] = __half2float(__ushort_as_half((unsigned short) (bits >> 16 * j)));
#pragma unroll
        for (int i = 0; i < N; ++i) {
            const float4 y = *reinterpret_cast<const float4*>(qi[i] + d);
            acc[i] += y.x * x[0];
            acc[i] += y.y * x[1];
            acc[i] += y.z * x[2];
            acc[i] += y.w * x[3];
        }
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
        for (int off = WARP / 2; off > 0; off >>= 1) acc[i] += __shfl_down_sync(0xffffffffu, acc[i], off);
        if (lane == 0) score[size_t(i) * n_keys] = acc[i] * scale;
    }
}

// attn_scores() for n queries: N at a time, then what is left by halves of N.
template <int N>
__device__ __forceinline__ void attn_scores_by(int n, const float* __restrict__ q, const half* __restrict__ k,
                                               float* __restrict__ score, int n_keys, int dim, float scale) {
    for (; n >= N; n -= N, q += size_t(N) * dim, score += size_t(N) * n_keys)
        attn_scores<N>(q, k, score, n_keys, dim, scale);
    if constexpr (N > 1) attn_scores_by<N / 2>(n, q, k, score, n_keys, dim, scale);
}

// score[token][head][i] = scale * (q[token][head] . the key of the head's group in the i-th row of the launch's
// span).  A warp has a row and one of the row's groups of heads.  Its threads read the key side by side,
// SCORE_VALUES values each a step, so that a step reads one place of the cache, and the launch's tokens read it
// once between them: the cache of a long context is what the kernel waits for.  A thread sums its products with
// a query, for up to SCORE_QUERIES heads of a token at a time, and a query's 32 sums are added in pairs.  A
// token gets scores for the rows of the span it does not attend as well, which the softmax leaves out.
// SPILL: the rows from `split` on are in khost (KvSpill).
template <bool SPILL>
__global__ void attn_score_kernel(const float* __restrict__ q, const half* __restrict__ kcache,
                                  float* __restrict__ score, int n_keys, int dim, int n_head, int kv_heads,
                                  long long pos, int first, int tokens, float scale, int window, int ring,
                                  long long last, const long long* __restrict__ at, const half* __restrict__ khost,
                                  long long split) {
    const int unit = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;   // a row and a group
    const int key = unit / kv_heads, g = unit % kv_heads, per = n_head / kv_heads;
    long long lo;
    if (key >= attn_span((at ? *at : pos) + first, tokens, window, last, &lo)) return;
    const long long row = ring > 0 ? (lo + key) % ring : lo + key;
    const half* k;
    if constexpr (SPILL) k = spill_row(kcache, khost, split, row, size_t(kv_heads) * dim) + size_t(g) * dim;
    else k = kcache + (size_t(row) * kv_heads + g) * dim;
    q += (size_t(first) * n_head + g * per) * dim;   // the group's heads of the launch's first token
    score += size_t(g) * per * n_keys + key;         // and their scores of the row
    for (int t = 0; t < tokens; ++t, q += size_t(n_head) * dim, score += size_t(n_head) * n_keys)
        attn_scores_by<SCORE_QUERIES>(per, q, k, score, n_keys, dim, scale);
}

// Softmax in place over a head's scores of the rows its token attends; the other rows of the launch's span get
// zero, which adds nothing to the token's values.  One block per head and token.
__global__ void attn_softmax_kernel(float* score, int n_keys, long long pos, int first, int tokens, int window,
                                    long long last, const long long* __restrict__ at) {
    __shared__ float part[8];
    const long long p = (at ? *at : pos) + first;
    long long lo, lo0;
    const int span = attn_span(p, tokens, window, last, &lo0);
    const int n = attn_rows(p + blockIdx.y, window, last, &lo), own = int(lo - lo0);
    float* all = score + (size_t(blockIdx.y) * gridDim.x + blockIdx.x) * n_keys;
    float* row = all + own;   // the token's rows of the span
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < n; j += blockDim.x) mx = fmaxf(mx, row[j]);
    mx = block_max(mx, part);
    float sum = 0.0f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        row[j] = expf(row[j] - mx);
        sum += row[j];
    }
    const float inv = 1.0f / block_sum(sum, part);
    for (int j = threadIdx.x; j < n; j += blockDim.x) row[j] *= inv;
    for (int j = threadIdx.x; j < own; j += blockDim.x) all[j] = 0.0f;
    for (int j = own + n + threadIdx.x; j < span; j += blockDim.x) all[j] = 0.0f;
}

// A thread's sums of attn_value_kernel for N queries: over its share of the span's rows from `row` of the cache
// on, of a row's value with each query's weight there.  w: the weights of the lane's query; the warp's first N
// lanes have the N queries between them.
// SPILL: the rows from `split` on are in vh, at the same place in a row as v's (KvSpill); no ring.
template <int N, bool SPILL>
__device__ __forceinline__ void attn_values(const float* __restrict__ w, const half* __restrict__ v, size_t width,
                                            long long row, int ring, int span, float* sums,
                                            const half* __restrict__ vh, long long split) {
    float acc[N];
#pragma unroll
    for (int i = 0; i < N; ++i) acc[i] = 0.0f;
    if constexpr (SPILL) {
        for (int j = threadIdx.x / VALUE_DIMS; j < span; j += VALUE_SHARES, row += VALUE_SHARES) {
            const float x = __half2float(*spill_row(v, vh, split, row, width)), weight = w[j];
#pragma unroll
            for (int i = 0; i < N; ++i) acc[i] += __shfl_sync(0xffffffffu, weight, i) * x;
        }
    } else {
        v += size_t(row) * width;
        for (int j = threadIdx.x / VALUE_DIMS; j < span; j += VALUE_SHARES) {
            const float x = __half2float(*v), weight = w[j];
#pragma unroll
            for (int i = 0; i < N; ++i) acc[i] += __shfl_sync(0xffffffffu, weight, i) * x;
            v += VALUE_SHARES * width;
            row += VALUE_SHARES;
            if (ring > 0 && row >= ring) row -= ring, v -= size_t(ring) * width;
        }
    }
#pragma unroll
    for (int i = 0; i < N; ++i) sums[i] = acc[i];
}

// attn_values() for n queries, with N if no smaller power of two holds them.
template <int N, bool SPILL>
__device__ __forceinline__ void attn_values_for(int n, const float* __restrict__ w, const half* __restrict__ v,
                                                size_t width, long long row, int ring, int span, float* sums,
                                                const half* __restrict__ vh, long long split) {
    if constexpr (N > 1)
        if (n <= N / 2) return attn_values_for<N / 2, SPILL>(n, w, v, width, row, ring, span, sums, vh, split);
    attn_values<N, SPILL>(w, v, width, row, ring, span, sums, vh, split);
}

// out[token][head][d] = sum over the rows of score[token][head][row] * the value of the head's group in that
// row.  A block has a group of heads, VALUE_DIMS value dimensions (blockIdx.x) and up to VALUE_QUERIES of the
// group's queries (blockIdx.y): the group's heads of the launch's first token, then of its second, and so on.
// A thread has one of the dimensions and one of VALUE_SHARES shares of the span's rows, every VALUE_SHARES-th
// row from its own, so that many reads of the cache are on their way at a time.  It reads a row's value once
// for the block's queries; their weights of the row come in one read as well, a lane a query, and the warp
// passes them around.  A token's shares are counted from its own first row and added in their order, so that it
// comes out the same whatever tokens the launch has beside it.
template <bool SPILL>
__global__ void attn_value_kernel(const float* __restrict__ score, const half* __restrict__ vcache,
                                  float* __restrict__ out, int n_keys, int dim, int n_head, int kv_heads,
                                  long long pos, int first, int tokens, int window, int ring, long long last,
                                  const long long* __restrict__ at, const half* __restrict__ vhost, long long split) {
    static_assert(VALUE_DIMS == WARP && VALUE_QUERIES <= WARP, "a warp is a share and reads a query's weight a lane");
    __shared__ float share[VALUE_QUERIES][VALUE_SHARES][VALUE_DIMS];
    const int per = n_head / kv_heads, dims = (dim + VALUE_DIMS - 1) / VALUE_DIMS;
    const int g = blockIdx.x / dims, q0 = blockIdx.y * VALUE_QUERIES;
    const int live = min(VALUE_QUERIES, tokens * per - q0);   // the block's queries
    const int lane = threadIdx.x % VALUE_DIMS, s = threadIdx.x / VALUE_DIMS;
    const int d = (blockIdx.x % dims) * VALUE_DIMS + lane;
    const long long p = (at ? *at : pos) + first;
    long long lo;
    const int span = attn_span(p, tokens, window, last, &lo);
    // the lane's query, and for a lane past the head's last value a value to read all the same
    const int mine = q0 + min(lane % VALUE_QUERIES, live - 1);
    float sums[VALUE_QUERIES];
    const size_t in_row = size_t(g) * dim + min(d, dim - 1);
    attn_values_for<VALUE_QUERIES, SPILL>(live, score + (size_t(mine / per) * n_head + g * per + mine % per) * n_keys,
                                         vcache + in_row, size_t(kv_heads) * dim,
                                         ring > 0 ? (lo + s) % ring : lo + s, ring, span, sums,
                                         SPILL ? vhost + in_row : nullptr, split);
    for (int i = 0; i < live; ++i) {
        long long own;   // the first row of the query's token: the thread's rows are this share of the token's
        attn_rows(p + (q0 + i) / per, window, last, &own);
        share[i][(s + VALUE_SHARES - int((own - lo) % VALUE_SHARES)) % VALUE_SHARES][lane] = sums[i];
    }
    __syncthreads();
    if (s != 0 || d >= dim) return;
    for (int i = 0; i < live; ++i) {
        float sum = share[i][0][lane];
        for (int k = 1; k < VALUE_SHARES; ++k) sum += share[i][k][lane];
        const int query = q0 + i;
        out[((size_t(first) + query / per) * n_head + g * per + query % per) * dim + d] = sum;
    }
}

// The attention of a call of a few tokens for groups of 8 heads of 256 values (Qwen3.5/3.6), with no sliding
// window, ring or split cache: attn_score_rows_kernel, attn_softmax_kernel, attn_value_part_kernel and
// attn_value_sum_kernel in place of attn_score_kernel, attn_softmax_kernel and attn_value_kernel
// (STRATA_POLY_ATTN=0 keeps those).  Measured on an RX 9060 XT at 45K rows (poly_kernel_bench attn-decode), those
// took 2 times what reading the keys and values takes for a token and 5.7 times for four.  attn_score_kernel's warp
// read 8 KiB of a token's queries for each row of 512 bytes of keys and added each head's products across its
// lanes with 5 shuffles; attn_value_kernel's block had one token's queries, so a window read the values once for
// each of its tokens, and passed every row's weights around the warp, a shuffle a query; and a token alone had
// 16 blocks for the card's 32 compute units.  With these, on the same card: 45K rows, 708 -> 432 us for a token and
// 1,902 -> 891 us for four; 8K, 106 -> 55 and 314 -> 138; 2K, 31 -> 35 and 83 -> 66.  A token's scores and values
// come out the same whatever tokens the call has beside it, as they did: each of its sums runs over its own rows
// and values in an order that does not depend on the others.  The values' sums are added in another order than
// attn_value_kernel adds them, so the last bits differ from its (1.3e-8 at most on values up to 0.04).

// The 8 sums a lane holds, one per head, each added over the warp's 32 lanes: at each of the first three steps a
// lane keeps half of its heads and gets its partner's half of them, so the eight take 9 shuffles where one at a
// time takes 40.  Lane l ends with the sum of head ((l >> 4) & 1) * 4 + ((l >> 3) & 1) * 2 + ((l >> 2) & 1),
// which its four lanes l & ~3 to l | 3 all have.
__device__ __forceinline__ float heads_sum(float (&a)[8], int lane) {
    float b[4], c[2];
    const bool hi16 = lane & 16, hi8 = lane & 8, hi4 = lane & 4;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float send = hi16 ? a[k] : a[4 + k];
        b[k] = (hi16 ? a[4 + k] : a[k]) + __shfl_xor_sync(0xffffffffu, send, 16);
    }
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        const float send = hi8 ? b[k] : b[2 + k];
        c[k] = (hi8 ? b[2 + k] : b[k]) + __shfl_xor_sync(0xffffffffu, send, 8);
    }
    float x = (hi4 ? c[1] : c[0]) + __shfl_xor_sync(0xffffffffu, hi4 ? c[0] : c[1], 4);
    x += __shfl_xor_sync(0xffffffffu, x, 2);
    x += __shfl_xor_sync(0xffffffffu, x, 1);
    return x;
}

// scores: a warp takes SCORE_ROWS rows of a group, and keeps a token's queries of the group in its registers while
// it reads them, where attn_score_kernel read them again for each row.  Its lanes read a row side by side, four
// values each a step; each lane's products of a head are added in the order of its steps, then over the lanes by
// heads_sum().
__global__ void __launch_bounds__(SCORE_WARPS * WARP)
    attn_score_rows_kernel(const float* __restrict__ q, const half* __restrict__ kcache, float* __restrict__ score,
                           int n_keys, int n_head, int kv_heads, long long pos, int first, int tokens, float scale,
                           const long long* __restrict__ at) {
    constexpr int dim = 256, per = 8, steps = dim / (SCORE_VALUES * WARP);
    const int unit = blockIdx.x * SCORE_WARPS + threadIdx.x / WARP;   // a group and SCORE_ROWS rows
    const int g = unit % kv_heads, key0 = unit / kv_heads * SCORE_ROWS;
    long long lo;
    const int span = attn_span((at ? *at : pos) + first, tokens, 0, -1, &lo);
    if (key0 >= span) return;
    const int keys = min(SCORE_ROWS, span - key0), lane = threadIdx.x % WARP;
    const int mine = ((lane >> 4) & 1) * 4 + ((lane >> 3) & 1) * 2 + ((lane >> 2) & 1);   // heads_sum()'s head
    for (int t = 0; t < tokens; ++t) {
        const float* qt = q + ((size_t(first) + t) * n_head + size_t(g) * per) * dim;
        float4 y[per][steps];
#pragma unroll
        for (int h = 0; h < per; ++h)
#pragma unroll
            for (int st = 0; st < steps; ++st)
                y[h][st] = *reinterpret_cast<const float4*>(qt + size_t(h) * dim + SCORE_VALUES * (lane + st * WARP));
        float* sc = score + (size_t(t) * n_head + size_t(g) * per + mine) * n_keys + key0;
        for (int r = 0; r < keys; ++r) {
            const half* k = kcache + ((size_t(lo) + key0 + r) * kv_heads + g) * dim;
            float x[steps][SCORE_VALUES];
#pragma unroll
            for (int st = 0; st < steps; ++st) {
                const unsigned long long bits =
                    *reinterpret_cast<const unsigned long long*>(k + SCORE_VALUES * (lane + st * WARP));
#pragma unroll
                for (int j = 0; j < SCORE_VALUES; ++j)
                    x[st][j] = __half2float(__ushort_as_half((unsigned short) (bits >> 16 * j)));
            }
            float acc[per];
#pragma unroll
            for (int h = 0; h < per; ++h) {
                acc[h] = 0.0f;
#pragma unroll
                for (int st = 0; st < steps; ++st) {
                    acc[h] += y[h][st].x * x[st][0];
                    acc[h] += y[h][st].y * x[st][1];
                    acc[h] += y[h][st].z * x[st][2];
                    acc[h] += y[h][st].w * x[st][3];
                }
            }
            const float sum = heads_sum(acc, lane);
            if ((lane & 3) == 0) sc[r] = sum * scale;
        }
    }
}

// values: a block has a group, up to Q queries (the group's heads of up to Q / 8 tokens, blockIdx.z) and a part of
// VALUE_PART rows of the span (blockIdx.y), counted from row 0.  Its 8 warps are 8 shares of the part's rows, every
// 8th row from the share's own, and a warp's lanes read a row's 256 values side by side, 8 each.  The block reads
// the queries' weights of VALUE_SUB rows at a time into shared memory, a row's together, from where a warp takes a
// row's at once.  A thread adds its 8 dimensions of each row's value times each query's weight there, its share's
// rows in order; the block adds its shares in their order into the part's sum (part_out), and
// attn_value_sum_kernel adds the parts in their order.  The rows a token does not attend have weight zero, and a
// part past its last row is zero.  attn_value_kernel's blocks had 32 of the dimensions each, so the weights, as
// many bytes as the values for four tokens, were read 8 times, and a window read the values once a token.
constexpr int VALUE_SUB = 128, VALUE_WARPS = 8, VALUE_ROUND = 2;   // queries the shares are added for at a time
template <int Q>
__global__ void __launch_bounds__(VALUE_WARPS * WARP)
    attn_value_part_kernel(const float* __restrict__ score, const half* __restrict__ vcache,
                           float* __restrict__ part_out, int n_keys, int dim, int n_head, int kv_heads, long long pos,
                           int first, int tokens, const long long* __restrict__ at) {
    static_assert(VALUE_SUB % VALUE_WARPS == 0 && VALUE_PART % VALUE_SUB == 0, "whole rounds of the shares");
    static_assert(Q % 4 == 0 && Q % VALUE_ROUND == 0, "a row's weights are read four at a time");
    constexpr int D = 256, E = D / WARP;   // a group's values, and a lane's
    constexpr int QP = Q + 4;              // a row's weights in shared memory, and room against bank conflicts
    constexpr int words = VALUE_SUB * QP > VALUE_ROUND * VALUE_WARPS * D ? VALUE_SUB * QP : VALUE_ROUND * VALUE_WARPS * D;
    __shared__ __align__(16) float mem[words];
    float(*wt)[QP] = reinterpret_cast<float(*)[QP]>(mem);
    float(*share)[VALUE_WARPS][D] = reinterpret_cast<float(*)[VALUE_WARPS][D]>(mem);
    const int per = n_head / kv_heads, all = tokens * per;
    const int g = blockIdx.x, q0 = blockIdx.z * Q, live = min(Q, all - q0);
    const int lane = threadIdx.x % WARP, s = threadIdx.x / WARP, tid = threadIdx.x;
    long long lo;
    const int span = attn_span((at ? *at : pos) + first, tokens, 0, -1, &lo);
    const int part0 = blockIdx.y * VALUE_PART, rows = max(0, min(span - part0, VALUE_PART));
    const size_t width = size_t(kv_heads) * dim;
    const half* v = vcache + (size_t(lo) + part0) * width + size_t(g) * dim + E * lane;
    float acc[Q][E];
#pragma unroll
    for (int i = 0; i < Q; ++i)
#pragma unroll
        for (int e = 0; e < E; ++e) acc[i][e] = 0.0f;
    for (int sub = 0; sub < rows; sub += VALUE_SUB) {
        const int n = min(VALUE_SUB, rows - sub);
        __syncthreads();   // the weights before these are read
        for (int i = tid; i < Q * VALUE_SUB; i += VALUE_WARPS * WARP) {
            const int j = i / VALUE_SUB, r = i % VALUE_SUB, qi = q0 + min(j, live - 1);
            wt[r][j] = r < n ? score[(size_t(qi / per) * n_head + size_t(g) * per + qi % per) * n_keys + part0 + sub + r]
                             : 0.0f;
        }
        __syncthreads();
        for (int r = s; r < n; r += VALUE_WARPS) {
            const uint4 raw = *reinterpret_cast<const uint4*>(v + size_t(sub + r) * width);
            const half2* h2 = reinterpret_cast<const half2*>(&raw);
            float x[E];
#pragma unroll
            for (int j = 0; j < E / 2; ++j) {
                const float2 f = __half22float2(h2[j]);
                x[2 * j] = f.x;
                x[2 * j + 1] = f.y;
            }
#pragma unroll
            for (int c = 0; c < Q / 4; ++c) {
                const float4 w = *reinterpret_cast<const float4*>(&wt[r][4 * c]);
#pragma unroll
                for (int e = 0; e < E; ++e) {
                    acc[4 * c][e] += w.x * x[e];
                    acc[4 * c + 1][e] += w.y * x[e];
                    acc[4 * c + 2][e] += w.z * x[e];
                    acc[4 * c + 3][e] += w.w * x[e];
                }
            }
        }
    }
    float* dst = part_out + ((size_t(blockIdx.y) * kv_heads + g) * all + q0) * dim;
#pragma unroll
    for (int c = 0; c < Q; c += VALUE_ROUND) {   // the shares of VALUE_ROUND queries at a time
        if (c >= live) break;
        __syncthreads();
#pragma unroll
        for (int i = 0; i < VALUE_ROUND; ++i)
#pragma unroll
            for (int e = 0; e < E; ++e) share[i][s][E * lane + e] = acc[c + i][e];
        __syncthreads();
        for (int i = tid / D; i < VALUE_ROUND; i += VALUE_WARPS * WARP / D) {
            if (c + i >= live) break;
            const int d = tid % D;
            float sum = share[i][0][d];
#pragma unroll
            for (int k = 1; k < VALUE_WARPS; ++k) sum += share[i][k][d];
            dst[size_t(c + i) * dim + d] = sum;
        }
    }
}

// The parts' sums of attn_value_part_kernel in their order: a thread a token, head and dimension.  The parts up
// to the one with the call's last row: the others have no row of the call.
__global__ void attn_value_sum_kernel(const float* __restrict__ part_out, float* __restrict__ out, int dim,
                                      int n_head, int kv_heads, long long pos, int first, int tokens,
                                      const long long* __restrict__ at) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tokens * n_head * dim) return;
    const int d = i % dim, h = i / dim % n_head, t = i / (dim * n_head);
    const int per = n_head / kv_heads, all = tokens * per;
    long long lo;
    const int span = attn_span((at ? *at : pos) + first, tokens, 0, -1, &lo);
    const int parts = (span + VALUE_PART - 1) / VALUE_PART;
    const size_t at0 = ((size_t(h / per)) * all + size_t(t) * per + h % per) * dim + d,
                 stride = size_t(kv_heads) * all * dim;
    float sum = part_out[at0];
    for (int p = 1; p < parts; ++p) sum += part_out[at0 + p * stride];
    out[((size_t(first) + t) * n_head + h) * dim + d] = sum;
}

// A chunk's queries and its results lie in two orders: the engine's (token, head) and the one a product per
// group of heads takes (group, token, head of the group).  This is where the engine's `row` lies in the second.
__device__ __forceinline__ size_t attn_by_group(int row, int n_head, int kv_heads, int tokens) {
    const int per = n_head / kv_heads, t = row / n_head, h = row % n_head;
    return (size_t(h / per) * tokens + t) * per + h % per;
}

// The queries to the order of the groups, as the halves the products take.  blockIdx.y is the token and the head.
__global__ void attn_group_kernel(const float* __restrict__ src, half* __restrict__ dst, int dim, int n_head,
                                  int kv_heads, int tokens) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    const float x = src[size_t(blockIdx.y) * dim + i];
    dst[attn_by_group(blockIdx.y, n_head, kv_heads, tokens) * dim + i] = __float2half(x);
}

// The results back to the engine's order.
__global__ void attn_ungroup_kernel(const float* __restrict__ src, float* __restrict__ dst, int dim, int n_head,
                                    int kv_heads, int tokens) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    dst[size_t(blockIdx.y) * dim + i] = src[attn_by_group(blockIdx.y, n_head, kv_heads, tokens) * dim + i];
}

// A softmax over a chunk's rows a block of rows at a time, for scores in the order of the groups: blockIdx.x
// counts (group, token, head of the group), and the block's rows start at row0 of the sequence.  A score's weight
// is its exp against the largest its token and head have had (top), zero for a row the token does not attend, as
// the half the values' product takes.  sum is the sum of the weights against that top, and shrink what the
// weights of the blocks before are worth against it.  With `fresh` there was no block before.
__global__ void attn_softmax_block_kernel(const float* __restrict__ score, half* __restrict__ weight, int n_keys,
                                          long long row0, long long pos, int first, int window, int tokens, int per,
                                          long long last, float* top, float* sum, float* shrink, bool fresh) {
    __shared__ float part[8];
    const int tok = first + (blockIdx.x % (tokens * per)) / per;
    long long lo;
    const int n = attn_rows(pos + tok, window, last, &lo);
    const long long end = lo + n;   // the token attends the sequence's rows from lo to before end
    const int j0 = lo > row0 ? int(lo - row0) : 0, j1 = end - row0 < n_keys ? int(end - row0) : n_keys;
    const float* row = score + size_t(blockIdx.x) * n_keys;
    half* w = weight + size_t(blockIdx.x) * n_keys;
    float mx = -INFINITY;
    for (int j = j0 + threadIdx.x; j < j1; j += blockDim.x) mx = fmaxf(mx, row[j]);
    mx = block_max(mx, part);
    const float before = fresh ? -INFINITY : top[blockIdx.x], now = fmaxf(before, mx);
    float s = 0.0f;
    for (int j = threadIdx.x; j < n_keys; j += blockDim.x) {
        w[j] = __float2half(j >= j0 && j < j1 ? expf(row[j] - now) : 0.0f);
        s += __half2float(w[j]);   // the sum is of the weights as the values get them
    }
    s = block_sum(s, part);
    if (threadIdx.x != 0) return;
    const float f = mx == -INFINITY ? 1.0f : expf(before - now);   // none of the block's rows: nothing changes
    top[blockIdx.x] = now;
    sum[blockIdx.x] = (fresh ? 0.0f : sum[blockIdx.x] * f) + s;
    shrink[blockIdx.x] = f;
}

// A chunk's results in the order of the groups, `dim` values a token and head: times its factor, or divided by it.
// A factor of exactly 1 (the largest score did not change, so exp(0)) leaves every value as it is, bit for bit
// (x * 1.0f is x), so the block returns without reading or writing the values.
__global__ void attn_scale_kernel(float* out, const float* __restrict__ factor, int dim, bool divide) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    const float f = factor[blockIdx.y];
    const float m = divide ? 1.0f / f : f;
    if (m == 1.0f) return;
    out[size_t(blockIdx.y) * dim + i] *= m;
}

__global__ void softcap_kernel(float* x, size_t n, float cap) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) x[i] = tanhf(x[i] * (1.0f / cap)) * cap;
}

// One block of 1024; a thread has every 1024th value.
__global__ void likeliest_kernel(const float* __restrict__ x, int n, Likeliest* __restrict__ out) {
    __shared__ float part[32];
    __shared__ int first[32];
    float m = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, x[i]);
    m = block_max(m, part);
    int at = n;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (x[i] == m) at = min(at, i);
        if (x[i] > m - 20.0f) sum += expf(x[i] - m);
    }
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) at = min(at, __shfl_xor_sync(0xffffffff, at, o, WARP));
    if ((threadIdx.x & 31) == 0) first[threadIdx.x >> 5] = at;
    sum = block_sum(sum, part);   // and first is written
    if (threadIdx.x != 0) return;
    for (int k = 1; k < int(blockDim.x) / WARP; ++k) at = min(at, first[k]);
    out->token = at < n ? at : 0;
    out->prob = 1.0f / sum;
}

__global__ void mul_sigmoid_kernel(float* __restrict__ acc, const float* __restrict__ g, int n, int g_stride,
                                   bool one_gate) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    acc[size_t(blockIdx.y) * n + i] *= sigmoid(g[size_t(blockIdx.y) * g_stride + (one_gate ? 0 : i)]);
}

// Where a token leaves the recurrent state as it is after it: `snaps` has a bit for each of a call's tokens that
// does, and their states lie in the tokens' order, `stride` floats apart.  Null for a token that leaves none.
__device__ __forceinline__ float* snap_of(float* snap, size_t stride, unsigned snaps, int row) {
    return row < 32 && (snaps >> row & 1u) ? snap + size_t(__popc(snaps & ((1u << row) - 1u))) * stride : nullptr;
}

// Three past inputs per channel, then the new one.  A thread is a channel and walks the tokens.  The state as it
// is after a token of `snaps` goes to snap (snap_of()).
__global__ void conv4_silu_kernel(const float* __restrict__ x, float* __restrict__ state,
                                  const float* __restrict__ w, float* __restrict__ out, int channels, int rows,
                                  float* __restrict__ snap, size_t stride, unsigned snaps) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const float* wc = w + 4 * size_t(c);
    float* st = state + size_t(c) * 3;
    float s0 = st[0], s1 = st[1], s2 = st[2];
    for (int row = 0; row < rows; ++row) {
        const size_t at = size_t(row) * channels + c;
        const float v = x[at];
        const float y = s0 * wc[0] + s1 * wc[1] + s2 * wc[2] + v * wc[3];
        s0 = s1;
        s1 = s2;
        s2 = v;
        out[at] = y * sigmoid(y);
        if (float* m = snap_of(snap, stride, snaps, row)) {
            m += size_t(c) * 3;
            m[0] = s0;
            m[1] = s1;
            m[2] = s2;
        }
    }
    st[0] = s0;
    st[1] = s1;
    st[2] = s2;
}

// The gated delta net's step, in four launches.  A token's queries and keys are normed without the state, its
// decay and its share of the error are a head's alone, and its output is normed without the tokens after it: all
// of that is done for every token at once, and only the state's update walks the tokens.
//
// One block a token and key head: the head's q and k, normed in place.
__global__ void gdn_norm_kernel(float* __restrict__ qkv, int dim, int k_heads, size_t row_n, float norm_eps) {
    __shared__ float part[8];
    const int hk = blockIdx.x % k_heads, j = threadIdx.x;
    float* x = qkv + size_t(blockIdx.x / k_heads) * row_n;
    float* qp = x + size_t(hk) * dim + j;
    float* kp = x + size_t(dim) * k_heads + size_t(hk) * dim + j;
    const float q = *qp, k = *kp;
    const float qs = block_sum(q * q, part);
    const float ks = block_sum(k * k, part);
    *qp = q / sqrtf(qs + norm_eps) * (1.0f / sqrtf(float(dim)));
    *kp = k / sqrtf(ks + norm_eps);
}

// A token's decay and its share of the error for each head, in place of alpha and beta.
__global__ void gdn_gates_kernel(float* __restrict__ alpha, float* __restrict__ beta, const float* __restrict__ dt,
                                 const float* __restrict__ a, int n_head, int n) {
    const int at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at >= n) return;
    const int h = at % n_head;
    const float g = alpha[at] + dt[h];
    alpha[at] = expf((g > 20.0f ? g : logf(1.0f + expf(g))) * a[h]);
    beta[at] = sigmoid(beta[at]);
}

// The state's update, a warp a column: the blocks of a head share its columns among their warps, and a lane holds
// every 32nd value of its warp's column, so a column is read and written as whole lines and its two sums are a
// few terms a lane and a sum over the warp.  The block walks the tokens, and no warp waits for another.
// s[i + dim * j] holds key dimension i of column j.  out gets the output before its norm.  decay and share are
// what gdn_gates_kernel left.  dim is a multiple of 32.
__global__ void gdn_state_kernel(const float* __restrict__ qkv, const float* __restrict__ decay,
                                 const float* __restrict__ share, float* __restrict__ s, float* __restrict__ out,
                                 int dim, int n_head, int k_heads, int rows, float* __restrict__ snap, size_t stride,
                                 unsigned snaps) {
    const int per_head = dim / int(blockDim.x / WARP);   // blocks a head
    const int h = blockIdx.x / per_head, hk = h % k_heads;
    const int j = (blockIdx.x % per_head) * int(blockDim.x / WARP) + int(threadIdx.x / WARP);
    const int lane = threadIdx.x & (WARP - 1);
    if (j >= dim) return;   // a whole warp: the last block of a head whose columns do not fill it
    const size_t keys = size_t(dim) * k_heads, d_inner = size_t(dim) * n_head, row_n = 2 * keys + d_inner;
    const size_t mine = (size_t(h) * dim + j) * dim;
    float* col = s + mine;
    for (int row = 0; row < rows; ++row) {
        const float* x = qkv + size_t(row) * row_n;
        const float* qn = x + size_t(hk) * dim;
        const float* kn = x + keys + size_t(hk) * dim;
        const float v = x[2 * keys + size_t(h) * dim + j];
        const float dc = decay[size_t(row) * n_head + h];
        float sk = 0.0f;
        for (int i = lane; i < dim; i += WARP) sk += col[i] * dc * kn[i];
        sk = warp_sum(sk);
        const float d = (v - sk) * share[size_t(row) * n_head + h];
        float o = 0.0f;
        float* m = snap_of(snap, stride, snaps, row);
        if (m) m += mine;
        for (int i = lane; i < dim; i += WARP) {
            const float n = col[i] * dc + kn[i] * d;
            col[i] = n;
            if (m) m[i] = n;
            o += n * qn[i];
        }
        o = warp_sum(o);
        if (lane == 0) out[size_t(row) * d_inner + size_t(h) * dim + j] = o;
    }
}

// gdn_state_kernel for a prompt batch, which leaves no snapshots: each column's sums are the same, in the same order,
// so the same bits (poly_kernel_bench gdn checks it), but the column stays in registers from the first token to the last (DIM / WARP values a lane),
// and a warp has COLS columns, whose sums go side by side.  gdn_state_kernel reads and writes its column in memory
// at each token, and a token's sums wait for that: 6.6 ms of a 1,024-token call's layer on an RX 9060 XT, a quarter
// of an 8K prompt's card time (bench/results/2026-10-13-hip-prefill-kernels).  A token's inputs are read while the
// token before it is summed.
template <int DIM, int COLS>
__global__ void __launch_bounds__(256) gdn_state_reg_kernel(const float* __restrict__ qkv,
                                                            const float* __restrict__ decay,
                                                            const float* __restrict__ share, float* __restrict__ s,
                                                            float* __restrict__ out, int n_head, int k_heads, int rows) {
    constexpr int PER = DIM / WARP;
    const int warp = int(blockIdx.x) * int(blockDim.x / WARP) + int(threadIdx.x / WARP);
    const int h = warp / (DIM / COLS);
    if (h >= n_head) return;   // a whole warp
    const int j0 = (warp % (DIM / COLS)) * COLS, hk = h % k_heads, lane = threadIdx.x & (WARP - 1);
    const size_t keys = size_t(DIM) * k_heads, d_inner = size_t(DIM) * n_head, row_n = 2 * keys + d_inner;
    float* col = s + (size_t(h) * DIM + j0) * DIM + lane;
    float c[COLS][PER];
#pragma unroll
    for (int k = 0; k < COLS; ++k)
#pragma unroll
        for (int t = 0; t < PER; ++t) c[k][t] = col[size_t(k) * DIM + t * WARP];
    // a token's inputs: its key head's q and k, the column's values, its decay and share
    float qn[PER], kn[PER], v[COLS], dc = 0.0f, sh = 0.0f;
    const auto load = [&](int row) {
        const float* x = qkv + size_t(row) * row_n;
#pragma unroll
        for (int t = 0; t < PER; ++t) {
            qn[t] = x[size_t(hk) * DIM + lane + t * WARP];
            kn[t] = x[keys + size_t(hk) * DIM + lane + t * WARP];
        }
#pragma unroll
        for (int k = 0; k < COLS; ++k) v[k] = x[2 * keys + size_t(h) * DIM + j0 + k];
        dc = decay[size_t(row) * n_head + h];
        sh = share[size_t(row) * n_head + h];
    };
    if (rows > 0) load(0);
    for (int row = 0; row < rows; ++row) {
        float q[PER], kk[PER], vv[COLS];
#pragma unroll
        for (int t = 0; t < PER; ++t) {
            q[t] = qn[t];
            kk[t] = kn[t];
        }
#pragma unroll
        for (int k = 0; k < COLS; ++k) vv[k] = v[k];
        const float d0 = dc, s0 = sh;
        if (row + 1 < rows) load(row + 1);
        float sk[COLS];
#pragma unroll
        for (int k = 0; k < COLS; ++k) {
            sk[k] = 0.0f;
#pragma unroll
            for (int t = 0; t < PER; ++t) sk[k] += c[k][t] * d0 * kk[t];
        }
#pragma unroll
        for (int k = 0; k < COLS; ++k) sk[k] = warp_sum(sk[k]);
        float o[COLS];
#pragma unroll
        for (int k = 0; k < COLS; ++k) {
            const float d = (vv[k] - sk[k]) * s0;
            o[k] = 0.0f;
#pragma unroll
            for (int t = 0; t < PER; ++t) {
                // gdn_state_kernel's contraction, written out: with c * d0 at hand from sk the compiler would
                // otherwise fuse the other product, and the bits would differ
                const float n = __fmaf_rn(c[k][t], d0, __fmul_rn(kk[t], d));
                c[k][t] = n;
                o[k] += n * q[t];
            }
        }
#pragma unroll
        for (int k = 0; k < COLS; ++k) o[k] = warp_sum(o[k]);
        if (lane == 0)
#pragma unroll
            for (int k = 0; k < COLS; ++k) out[size_t(row) * d_inner + size_t(h) * DIM + j0 + k] = o[k];
    }
#pragma unroll
    for (int k = 0; k < COLS; ++k)
#pragma unroll
        for (int t = 0; t < PER; ++t) col[size_t(k) * DIM + t * WARP] = c[k][t];
}

// One block a token and value head: the output's norm, its weights and the gate.
__global__ void gdn_out_kernel(float* __restrict__ out, const float* __restrict__ z,
                               const float* __restrict__ o_norm, int dim, float norm_eps) {
    __shared__ float part[8];
    const size_t at = size_t(blockIdx.x) * dim + threadIdx.x;
    const float o = out[at];
    const float os = block_sum(o * o, part);
    const float gate = z[at];
    out[at] = o / sqrtf(os / float(dim) + norm_eps) * o_norm[threadIdx.x] * (gate * sigmoid(gate));
}

// ---- STRATA_POLY_FUSE (kernels.hpp): kernels that each do the work of two or three of the ones above, so that a
// call of a few tokens has fewer launches.  Each value comes from the same expression, in the same thread and the
// same order of sums, as in the kernels it stands for, so the bits are theirs.  The Q8_1 blocks are
// native_quantize_q8_1_kernel's: a warp has a block of 32 values in lane order, their largest magnitude and their
// sum over the warp, so a kernel here quantizes only where every warp has 32 values in a row.
struct Q81 {
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q81) == 36, "native_mmvq's block_q8_1");

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, o, WARP));
    return x;
}

// Value i of a run of values quantized in place in y: every lane of the warp has one, i in lane order.
__device__ __forceinline__ void q8_1_store(float xi, Q81* y, size_t i) {
    const float amax = warp_max(fabsf(xi));
    const float sum = warp_sum(xi);
    const float d = kernels::q8_1_finite(amax / 127.0f);
    y[i / 32].qs[i % 32] = kernels::q8_1_quant(xi, d, amax);
    if (i % 32 == 0) y[i / 32].ds = kernels::q8_1_ds(d, sum);
}

// rms_norm_kernel and the quantization of its output.  With `add`, axpy_kernel(add, a, x) first, into x.
__global__ void rms_norm_q8_kernel(float* x, const float* __restrict__ add, float a, const float* __restrict__ w,
                                   float* y, Q81* __restrict__ yq, int n, float eps) {
    __shared__ float part[32];
    float* xr = x + size_t(blockIdx.x) * n;
    float* yr = y + size_t(blockIdx.x) * n;
    float ss = 0.0f;
    if (add) {
        const float* ar = add + size_t(blockIdx.x) * n;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const float v = xr[i] + ar[i] * a;
            xr[i] = v;
            ss += v * v;
        }
    } else {
        for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    }
    ss = block_sum(ss, part);
    const float scale = 1.0f / sqrtf(ss / float(n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = w ? xr[i] * scale * w[i] : xr[i] * scale;
        yr[i] = v;
        q8_1_store(v, yq, size_t(blockIdx.x) * n + i);
    }
}

// swiglu_clamp_kernel (gate and up) or swiglu_clamp_rows_kernel (gate_up: rows of n gate values, then n up values),
// and the quantization of its output.  blockIdx.y is the row.
__global__ void swiglu_clamp_q8_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                                       const float* __restrict__ gate_up, float* __restrict__ out,
                                       Q81* __restrict__ q, int n, float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;   // n is a multiple of 32: only whole warps return
    const size_t at = size_t(blockIdx.y) * n + i;
    const float* row = gate_up + size_t(blockIdx.y) * 2 * n;
    const float g = fminf(gate_up ? row[i] : gate[at], limit);
    const float u = fminf(fmaxf(gate_up ? row[n + i] : up[at], -limit), limit);
    const float v = g / (1.0f + expf(-g)) * u;
    out[at] = v;
    q8_1_store(v, q, at);
}

// mul_sigmoid_kernel with a gate a value, and the quantization of its output.
__global__ void mul_sigmoid_q8_kernel(float* __restrict__ acc, const float* __restrict__ g, Q81* __restrict__ q,
                                      int n, int g_stride) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;   // n is a multiple of 32
    const size_t at = size_t(blockIdx.y) * n + i;
    acc[at] *= sigmoid(g[size_t(blockIdx.y) * g_stride + i]);
    q8_1_store(acc[at], q, at);
}

// gdn_out_kernel and the quantization of its output.
__global__ void gdn_out_q8_kernel(float* __restrict__ out, const float* __restrict__ z,
                                  const float* __restrict__ o_norm, Q81* __restrict__ q, int dim, float norm_eps) {
    __shared__ float part[8];
    const size_t at = size_t(blockIdx.x) * dim + threadIdx.x;
    const float o = out[at];
    const float os = block_sum(o * o, part);
    const float gate = z[at];
    const float v = o / sqrtf(os / float(dim) + norm_eps) * o_norm[threadIdx.x] * (gate * sigmoid(gate));
    out[at] = v;
    q8_1_store(v, q, at);
}

// gemv_kernel for the gated delta net's alpha and beta (n_head rows each, blockIdx.x: alpha's rows, then beta's), and
// gdn_gates_kernel's work on each output: alpha's becomes the decay, beta's the share of the error.
__global__ void gdn_ab_kernel(const float* __restrict__ wa, const float* __restrict__ wb, const float* __restrict__ x,
                              float* __restrict__ alpha, float* __restrict__ beta, const float* __restrict__ dt,
                              const float* __restrict__ a, int n_in, int n_head) {
    __shared__ float part[32];
    const bool second = int(blockIdx.x) >= n_head;
    const int h = second ? int(blockIdx.x) - n_head : int(blockIdx.x);
    const size_t tok = blockIdx.y;
    const float* wr = (second ? wb : wa) + size_t(h) * n_in;
    const float* xr = x + tok * n_in;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n_in; i += blockDim.x) acc += ldw(wr, i) * xr[i];
    acc = block_sum(acc, part);
    if (threadIdx.x != 0) return;
    const size_t at = tok * n_head + h;
    if (second) {
        beta[at] = sigmoid(acc);
    } else {
        const float g = acc + dt[h];
        alpha[at] = expf((g > 20.0f ? g : logf(1.0f + expf(g))) * a[h]);
    }
}

inline cudaStream_t cs(void* stream) { return (cudaStream_t) stream; }
inline int grid(size_t n, int block) { return int((n + block - 1) / block); }

// #247/#325 (Windows HIP, seen on gfx1201): hipBLAS can return success with the right FP16/BF16 product and still
// leave hipErrorInvalidValue as the thread's last error, which the next launch check (Mmq's) would take for its own
// failure.  So on Windows HIP a product's own success clears that one value, and any other error is fatal, as
// src/prefill/gemm.cu does for `strata`.  So that this cannot hide a real hipErrorInvalidValue from a launch before
// the product, the last error is looked at before every HIP product too (blas_before). CUDA keeps its own path.
#if defined(STRATA_USE_HIP)
void blas_before(const char* what) {
    hip_launch_check(what);
}
#define STRATA_BLAS_BEFORE(what) blas_before(what)
#else
#define STRATA_BLAS_BEFORE(what) ((void) 0)
#endif

void blas_ck(cublasStatus_t s, const char* what) {
#if defined(STRATA_USE_HIP)
    if (s != CUBLAS_STATUS_SUCCESS)
        gpu_fatal(what, "hipBLAS failure", (int) s, "hipblas_status");
#else
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(what) + ": cuBLAS status " + std::to_string((int) s));
#endif
#if defined(STRATA_USE_HIP) && defined(_WIN32)
    const hipError_t sticky = hipGetLastError();
    if (sticky != hipSuccess && sticky != hipErrorInvalidValue)
        hip_check(sticky, what);
#endif
}

}  // namespace

void* blas_create(void* stream) {
    cublasHandle_t h = nullptr;
    STRATA_BLAS_BEFORE("cublasCreate");
    blas_ck(cublasCreate(&h), "cublasCreate");
    if (const cublasStatus_t s = cublasSetStream(h, cs(stream)); s != CUBLAS_STATUS_SUCCESS) {
#if defined(STRATA_USE_HIP)
        blas_ck(s, "cublasSetStream");
#else
        cublasDestroy(h);
        blas_ck(s, "cublasSetStream");
#endif
    }
    // cuBLAS takes 75 ms over a process's first product of halves, whatever its size: a prompt batch's attention
    // would wait for it.  A product that fails here fails again where its status is read.
    const int n = 16;
    void* p = nullptr;
    const cudaError_t allocated = cudaMalloc(&p, n * n * (2 * sizeof(half) + sizeof(float)));
#if defined(STRATA_USE_HIP)
    if (allocated != cudaErrorMemoryAllocation) hip_check(allocated, "BLAS warmup allocation");
    else hip_clear_expected(allocated, "BLAS warmup allocation");
#endif
    if (allocated != cudaSuccess) return h;
#if defined(STRATA_USE_HIP)
    hip_check(cudaMemsetAsync(p, 0, n * n * (2 * sizeof(half) + sizeof(float)), cs(stream)), "BLAS warmup zero");
#else
    cudaMemsetAsync(p, 0, n * n * (2 * sizeof(half) + sizeof(float)), cs(stream));
#endif
    const float one = 1.0f, zero = 0.0f;
    const half* a = (const half*) p;
    const cublasStatus_t warm =
        cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &one, a, CUDA_R_16F, n, n * n, a + n * n,
                                   CUDA_R_16F, n, n * n, &zero, (float*) (a + 2 * n * n), CUDA_R_32F, n, n * n, 1,
                                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#if defined(STRATA_USE_HIP)
    blas_ck(warm, "the first product of halves");
    // hipBLAS took 6.4 s over a process's first float product on an RX 9060 XT under Windows (poly_kernel_bench),
    // which a prompt batch's float projections (gemm_f32) felt in the first prompt; a small one here takes it.
    const float* f = (const float*) p;
    const cublasStatus_t warm_f =
        cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &one, f, n, f, n, &zero, (float*) (a + 2 * n * n), n);
    // Consume hipBLAS's documented Windows stale InvalidValue only after success,
    // before a general launch check. Every execution/synchronization error is fatal.
    blas_ck(warm_f, "the first float product");
    hip_sync(cs(stream), "BLAS warmup");
    hip_check(cudaFree(p), "BLAS warmup free");
#else
    cudaStreamSynchronize(cs(stream));
    cudaFree(p);
    (void) warm;
#endif
    return h;
}

void blas_destroy(void* blas) {
    if (blas) cublasDestroy((cublasHandle_t) blas);
}

void gemv_f32(const float* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, int rows, void* stream) {
    gemv(w, x, y, n_in, n_out, rows_per_x, rows, stream);
}

void gemv_bf16(const uint16_t* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, int rows,
               void* stream) {
    gemv(w, x, y, n_in, n_out, rows_per_x, rows, stream);
}

// A buffer of rows is its transpose to cuBLAS, so y[row] = w x[row] is Y = W^T X.  A group's rows of w, its part
// of every token's x and of every token's y are each a matrix with a stride.
void gemm_f32(void* blas, const float* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, int rows) {
    const cublasHandle_t h = (cublasHandle_t) blas;
    const float one = 1.0f, zero = 0.0f;
    const int groups = n_out / rows_per_x;
    if (groups == 1) {
        STRATA_BLAS_BEFORE("a batch's projection");
        blas_ck(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, n_out, rows, n_in, &one, w, n_in, x, n_in, &zero, y, n_out),
                "a batch's projection");
        return;
    }
    STRATA_BLAS_BEFORE("a batch's projections");
    blas_ck(cublasSgemmStridedBatched(h, CUBLAS_OP_T, CUBLAS_OP_N, rows_per_x, rows, n_in, &one, w, n_in,
                                      (long long) rows_per_x * n_in, x, groups * n_in, n_in, &zero, y, n_out,
                                      rows_per_x, groups),
            "a batch's projections");
}

#if defined(STRATA_USE_HIP)
namespace {
__global__ void copy_words_kernel(const uint32_t* __restrict__ src, uint32_t* __restrict__ dst, size_t n) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        dst[i] = src[i];
}
}  // namespace

void copy_words(const void* src, void* dst, size_t n, void* stream) {
    if (n == 0) return;
    copy_words_kernel<<<std::min(grid(n, 256), 1024), 256, 0, cs(stream)>>>((const uint32_t*) src, (uint32_t*) dst,
                                                                             n);
}

namespace {
__global__ void add_scaled_dev_kernel(ScaledRows rows, const float* __restrict__ w, int count, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int k = 0; k < count; ++k) rows.acc[k][i] = rows.acc[k][i] + rows.x[k][i] * w[k];
}

// The likelier of two (probability, expert): the lower number among equals, as the host's scan finds it.
__device__ __forceinline__ void likelier(float& p, int& e, float q, int f) {
    if (q > p || (q == p && f < e)) p = q, e = f;
}

constexpr int kRouteThreads = 256;

__global__ void __launch_bounds__(kRouteThreads) route_topk_softmax_kernel(
    const float* __restrict__ scores, int n_expert, int n_used, float min_sum, const uint8_t* const* __restrict__ table,
    size_t down_off, kernels::MmvqSlots* gu, kernels::MmvqSlots* dn, float* w_dev, CardRoute out) {
    // a block a token: its scores, its slots and its share of `out`, n_used (8 at most) a token
    const int row = blockIdx.x;
    scores += (size_t) row * n_expert;
    gu += row;
    dn += row;
    w_dev += row * 8;
    out.ids += row * 8;
    out.w += row * 8;
    out.cached += row * 8;
    out.x += (size_t) row * out.n_x;
    out.x_host += (size_t) row * out.n_x;
    out.need += row;
    __shared__ float prob[1024];
    __shared__ uint8_t taken[1024];
    __shared__ float wp[kRouteThreads / 32];
    __shared__ int we[kRouteThreads / 32];
    __shared__ float mx_s;
    __shared__ int pick[8];
    __shared__ float pw[8];
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    for (int i = tid; i < out.n_x; i += kRouteThreads) out.x_host[i] = out.x[i];   // the experts' input, for the CPU
    float mx = -INFINITY;
    for (int e = tid; e < n_expert; e += kRouteThreads) mx = fmaxf(mx, scores[e]);
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o, 32));
    if (lane == 0) wp[warp] = mx;
    __syncthreads();
    if (tid == 0) {
        float m = wp[0];
        for (int w = 1; w < kRouteThreads / 32; ++w) m = fmaxf(m, wp[w]);
        mx_s = m;
    }
    __syncthreads();
    for (int e = tid; e < n_expert; e += kRouteThreads) prob[e] = expf(scores[e] - mx_s), taken[e] = 0;
    __syncthreads();
    for (int k = 0; k < n_used; ++k) {
        float p = -2.f;
        int best = n_expert;
        for (int e = tid; e < n_expert; e += kRouteThreads)
            if (!taken[e]) likelier(p, best, prob[e], e);
        for (int o = 16; o > 0; o >>= 1) {
            const float q = __shfl_xor_sync(0xffffffff, p, o, 32);
            const int f = __shfl_xor_sync(0xffffffff, best, o, 32);
            likelier(p, best, q, f);
        }
        if (lane == 0) wp[warp] = p, we[warp] = best;
        __syncthreads();
        if (tid == 0) {
            for (int w = 1; w < kRouteThreads / 32; ++w) likelier(p, best, wp[w], we[w]);
            pick[k] = best;
            pw[k] = p;
            taken[best] = 1;
        }
        __syncthreads();
    }
    if (tid != 0) return;
    // the host's sum: in the experts' order, in double
    double all = 0.0;
    for (int e = 0; e < n_expert; ++e) all += (double) prob[e];
    const float inv = (float) (1.0 / all);
    float sum = 0.f;
    for (int k = 0; k < n_used; ++k) sum += pw[k] = pw[k] * inv;
    const float denom = fmaxf(sum, min_sum);
    int need = 0;
    for (int k = 0; k < 8; ++k) {
        const int e = k < n_used ? pick[k] : 0;
        const uint8_t* s = k < n_used ? table[e] : nullptr;
        const float w = k < n_used ? pw[k] / denom : 0.f;
        gu->w[k] = s;
        dn->w[k] = s ? s + down_off : nullptr;
        w_dev[k] = s ? w : 0.f;
        if (k < n_used) out.ids[k] = e, out.w[k] = w, out.cached[k] = s != nullptr, need |= s == nullptr;
    }
    *out.need = need;
    if (out.ready) {   // after everything above, for the host that polls it: of several tokens, by the last block
        __threadfence_system();
        if (gridDim.x > 1) {
            if (atomicAdd(out.blocks, 1u) != gridDim.x - 1) return;
            *out.blocks = 0;   // for the next layer that uses it: every block has added
            __threadfence_system();
        }
        *(volatile uint32_t*) out.ready = *out.seq;
        __threadfence_system();
    }
}

// The CPU's experts of a layer into fo: when the routing picked one (need), once the host says their sum is in
// `cpu` (go == seq).  A wait longer than `ticks` of wall_clock64() gives up and sets *failed.
__global__ void wait_add_kernel(const int* need, const uint32_t* go, const uint32_t* seq, const float* cpu, float* fo,
                                int n, int rows, long long ticks, uint32_t* failed, unsigned long long* waited) {
    int any = 0;   // of each token's need: a token that picked no expert outside VRAM gets nothing added
    for (int r = 0; r < rows; ++r) any |= need[r];
    if (!any) return;
    __shared__ int ok;
    if (threadIdx.x == 0) {
        const uint32_t want = *seq;
        const long long t0 = wall_clock64();
        ok = 1;
        while (__hip_atomic_load(go, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != want) {
            // a layer before this one gave up: the token is lost, and the host stops (Engine::card_host())
            if (__hip_atomic_load(failed, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) != 0) {
                ok = 0;
                break;
            }
            if (wall_clock64() - t0 > ticks) {
                ok = 0;
                *(volatile uint32_t*) failed = 1;
                __threadfence_system();
                break;
            }
            __builtin_amdgcn_s_sleep(2);
        }
        if (waited) atomicAdd(waited, (unsigned long long) (wall_clock64() - t0));
    }
    __syncthreads();
    if (!ok) return;
    for (int r = 0; r < rows; ++r)
        if (need[r])
            for (int i = threadIdx.x; i < n; i += blockDim.x) fo[(size_t) r * n + i] += cpu[(size_t) r * n + i];
}

// STRATA_POLY_FUSE: add_scaled_dev_kernel, wait_add_kernel and, with `resid`, axpy_kernel(fo, a, resid) in one
// launch.  A thread has value i of every row, as in the first and the last, and a block waits for the host's sum
// on its own, as wait_add_kernel's one block did (only block 0 counts the wait).  Each value goes through the same
// sums in the same order as in the three launches.
__global__ void card_tail_kernel(ScaledRows rows, const float* __restrict__ w, int count, const int* need,
                                 const uint32_t* go, const uint32_t* seq, const float* cpu, float* fo, float* resid,
                                 float a, int n, int n_rows, long long ticks, uint32_t* failed,
                                 unsigned long long* waited) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        for (int k = 0; k < count; ++k) rows.acc[k][i] = rows.acc[k][i] + rows.x[k][i] * w[k];
    int any = 0;
    for (int r = 0; r < n_rows; ++r) any |= need[r];
    if (any) {
        __shared__ int ok;
        if (threadIdx.x == 0) {
            const uint32_t want = *seq;
            const long long t0 = wall_clock64();
            ok = 1;
            while (__hip_atomic_load(go, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != want) {
                if (__hip_atomic_load(failed, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) != 0) {
                    ok = 0;
                    break;
                }
                if (wall_clock64() - t0 > ticks) {
                    ok = 0;
                    *(volatile uint32_t*) failed = 1;
                    __threadfence_system();
                    break;
                }
                __builtin_amdgcn_s_sleep(2);
            }
            if (waited && blockIdx.x == 0) atomicAdd(waited, (unsigned long long) (wall_clock64() - t0));
        }
        __syncthreads();
        if (ok && i < n)
            for (int r = 0; r < n_rows; ++r)
                if (need[r]) fo[(size_t) r * n + i] += cpu[(size_t) r * n + i];
    }
    if (resid && i < n)
        for (int r = 0; r < n_rows; ++r) {
            const size_t at = (size_t) r * n + i;
            resid[at] = resid[at] + fo[at] * a;
        }
}

__global__ void set_word_kernel(uint32_t* p, uint32_t v) { *p = v; }

__global__ void set_slot_kernel(const uint8_t** table, size_t i, const uint8_t* slot) { table[i] = slot; }
}  // namespace

void add_scaled_dev(const ScaledRows& rows, const float* w, int count, int n, void* stream) {
    if (count < 1 || count > 32) throw std::invalid_argument("add_scaled_dev: 1 to 32 vectors");
    add_scaled_dev_kernel<<<grid(n, 256), 256, 0, cs(stream)>>>(rows, w, count, n);
}

void route_topk_softmax(const float* scores, int n_expert, int n_used, float min_sum, const uint8_t* const* table,
                        size_t down_off, void* gate_up, void* down, float* w_dev, const CardRoute& out, int rows,
                        void* stream) {
    if (n_expert < 1 || n_expert > 1024 || n_used < 1 || n_used > 8 || n_used > n_expert)
        throw std::invalid_argument("route_topk_softmax: 1 to 1024 experts, 1 to 8 of them used");
    if (rows < 1 || rows > 8 || (rows > 1 && out.ready && !out.blocks))
        throw std::invalid_argument("route_topk_softmax: 1 to 8 tokens, several with a block count");
    route_topk_softmax_kernel<<<rows, kRouteThreads, 0, cs(stream)>>>(
        scores, n_expert, n_used, min_sum, table, down_off, (kernels::MmvqSlots*) gate_up, (kernels::MmvqSlots*) down,
        w_dev, out);
}

void wait_add(const int* need, const uint32_t* go, const uint32_t* seq, const float* cpu, float* fo, int n,
              int rows, long long ticks, uint32_t* failed, void* stream, unsigned long long* waited) {
    wait_add_kernel<<<1, 256, 0, cs(stream)>>>(need, go, seq, cpu, fo, n, rows, ticks, failed, waited);
}

void card_tail(const ScaledRows& sum, const float* w, int count, const int* need, const uint32_t* go,
               const uint32_t* seq, const float* cpu, float* fo, float* resid, int n, int rows, long long ticks,
               uint32_t* failed, void* stream, unsigned long long* waited) {
    if (count < 1 || count > 32) throw std::invalid_argument("card_tail: 1 to 32 vectors");
    card_tail_kernel<<<grid(n, 256), 256, 0, cs(stream)>>>(sum, w, count, need, go, seq, cpu, fo, resid, 1.f, n, rows,
                                                         ticks, failed, waited);
}

void set_word(uint32_t* p, uint32_t v, void* stream) { set_word_kernel<<<1, 1, 0, cs(stream)>>>(p, v); }

void set_slot(const uint8_t** table, size_t i, const uint8_t* slot, void* stream) {
    set_slot_kernel<<<1, 1, 0, cs(stream)>>>(table, i, slot);
}
#endif

void bf16_to_f32(const uint16_t* w, float* out, size_t n, void* stream) {
    bf16_to_f32_kernel<<<grid(n, 256), 256, 0, cs(stream)>>>(w, out, n);
}

void rms_norm(const float* x, const float* w, float* y, int n, int rows, float eps, void* stream, int y_stride) {
    rms_norm_kernel<<<rows, sum_threads(n), 0, cs(stream)>>>(x, w, y, n, eps, y_stride ? y_stride : n);
}

void layer_norm(const float* x, const float* w, const float* b, float* y, int n, int rows, float eps, void* stream) {
    layer_norm_kernel<<<rows, 128, 0, cs(stream)>>>(x, w, b, y, n, eps);
}

void hc_pre(const float* r, const float* fn, const float* scale, const float* base, float* mix, float* hcw,
            float* cur, int n_embd, int rows, float norm_eps, float hc_eps, int sinkhorn_iters, void* stream) {
    const int n_mix = HC * (2 + HC);
    hc_mix_kernel<<<dim3(n_mix + 1, rows), sum_threads(HC * n_embd), 0, cs(stream)>>>(r, fn, mix, HC * n_embd, n_mix);
    hc_read_kernel<<<dim3(grid(n_embd, 256), rows), 256, 0, cs(stream)>>>(r, mix, scale, base, hcw, cur, n_embd,
                                                                         norm_eps, hc_eps, sinkhorn_iters);
}

void hc_post(const float* x, const float* hcw, float* r, int n_embd, int rows, void* stream) {
    hc_post_kernel<<<dim3(grid(n_embd, 256), rows), 256, 0, cs(stream)>>>(x, hcw, r, n_embd);
}

void hc_mean(const float* r, float* y, int n_embd, int rows, void* stream) {
    hc_mean_kernel<<<dim3(grid(n_embd, 256), rows), 256, 0, cs(stream)>>>(r, y, n_embd);
}

void kda_conv(const float* proj, float* state, const float* wq, const float* wk, const float* wv, float* out,
              int d_inner, int rows, void* stream, float* snap, size_t stride, int n_snap) {
    kda_conv_kernel<<<grid(3 * d_inner, 256), 256, 0, cs(stream)>>>(proj, state, wq, wk, wv, out, d_inner, rows,
                                                                    snap, stride, snap ? n_snap : 0);
}

void kda_step(float* qkv, float* g1, const float* a, const float* beta, float* s, const float* g2,
              const float* o_norm, float* out, int head_dim, int n_head, int rows, float gate_lower_bound,
              float norm_eps, void* stream, float* snap, size_t stride, int n_snap) {
    if (head_dim % WARP != 0 || head_dim > 256)
        throw std::invalid_argument("kda_step: a head of 32 to 256 values, in 32s");
    int lanes = WARP;   // threads a column, while a block has them
    while (head_dim * lanes > 1024) lanes /= 2;
    // a few tokens, or a head that is not kda_state_kernel's size: one launch
    if (rows < 32 || snap || head_dim != KDA_TERMS * lanes) {
        kda_step_kernel<<<n_head, head_dim * lanes, 0, cs(stream)>>>(qkv, g1, a, beta, s, g2, o_norm, out, head_dim,
                                                                    n_head, rows, gate_lower_bound, norm_eps, snap,
                                                                    stride, snap ? n_snap : 0);
        return;
    }
    const int group = WARP / lanes;
    kda_gates_kernel<<<rows * n_head, head_dim, 0, cs(stream)>>>(qkv, g1, a, n_head, rows, gate_lower_bound, group);
    kda_state_kernel<<<n_head * (head_dim / KDA_COLUMNS), KDA_COLUMNS * lanes, 0, cs(stream)>>>(qkv, g1, beta, s, out,
                                                                                               n_head, rows);
    kda_out_kernel<<<rows * n_head, head_dim, 0, cs(stream)>>>(out, g2, o_norm, norm_eps, group);
}

void store_f16(const float* x, uint16_t* cache, int n, int64_t pos, int rows, void* stream, uint16_t* host,
               int64_t split) {
    // with a host part, the rows below `split` and the rows from it are two stores
    const int below = host ? (int) std::clamp<int64_t>(split - pos, 0, rows) : rows;
    if (below > 0) {
        const size_t all = size_t(n) * below;
        store_f16_kernel<<<grid(all, 256), 256, 0, cs(stream)>>>(x, (half*) cache + size_t(pos) * n, int(all));
    }
    if (below < rows) {
        const size_t all = size_t(n) * (rows - below);
        store_f16_kernel<<<grid(all, 256), 256, 0, cs(stream)>>>(
            x + size_t(below) * n, (half*) host + size_t(pos + below - split) * n, int(all));
    }
}

void store_f16_at(const float* x, uint16_t* cache, int n, const long long* at, int rows, int ring, void* stream,
                  uint16_t* host, int64_t split) {
    if (host && ring > 0) throw std::invalid_argument("store_f16_at: a ring has no host part");
    if (host)
        store_f16_at_kernel<true><<<grid(size_t(n) * rows, 256), 256, 0, cs(stream)>>>(
            x, (half*) cache, n, rows, at, ring, (half*) host, (long long) split);
    else
        store_f16_at_kernel<false><<<grid(size_t(n) * rows, 256), 256, 0, cs(stream)>>>(x, (half*) cache, n, rows, at,
                                                                                        ring, nullptr, 0);
}

void dsa_attend(void* blas, const float* q, const uint16_t* cache, const int32_t* sel, int sel_stride, int top_rows,
                int kpool, int64_t pos, int rows, int dense_rows, float* keys, float* score, int chunk, float* out,
                int rank, int n_head, float scale, void* stream) {
    const cublasHandle_t h = (cublasHandle_t) blas;
    const float one = 1.0f, zero = 0.0f;
    const long long head = (long long) n_head * rank;
    for (int first = 0; first < rows; first += chunk) {
        const int c = std::min(chunk, rows - first);
        int n_keys = 0;
        for (int t = first; t < first + c; ++t)
            n_keys = std::max(n_keys, dsa_keys(t, (long long) pos, dense_rows, top_rows, kpool));
        const long long key_n = (long long) n_keys * rank, score_n = (long long) n_head * n_keys;
        dsa_keys_kernel<<<dim3(grid(n_keys, KEY_BLOCK), c), 256, 0, cs(stream)>>>(
            (const half*) cache, sel, sel_stride, top_rows, kpool, (long long) pos, first, dense_rows, keys, rank,
            n_keys);
        // score[head][key] = scale * (q[head] . keys[key]).  cuBLAS reads a buffer of rows as the transpose.
        STRATA_BLAS_BEFORE("attention scores");
        blas_ck(cublasSgemmStridedBatched(h, CUBLAS_OP_T, CUBLAS_OP_N, n_keys, n_head, rank, &scale, keys, rank,
                                          key_n, q + size_t(first) * head, rank, head, &zero, score, n_keys,
                                          score_n, c),
                "attention scores");
        dsa_softmax_kernel<<<dim3(n_head, c), 256, 0, cs(stream)>>>(score, n_keys, top_rows, kpool, (long long) pos,
                                                                   first, dense_rows);
        // out[head] = sum over the keys of score[head][key] * keys[key]
        STRATA_BLAS_BEFORE("attention values");
        blas_ck(cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, rank, n_head, n_keys, &one, keys, rank, key_n,
                                          score, n_keys, score_n, &zero, out + size_t(first) * head, rank, head, c),
                "attention values");
    }
}

void idx_store(const float* key, const float* gate, float* cache, int dim, int64_t pos, int rows, void* stream) {
    idx_store_kernel<<<rows, 256, 0, cs(stream)>>>(key, gate, cache + size_t(pos) * 3 * dim, dim);
}

void idx_pool(float* cache, const float* ape, int dim, int kpool, int64_t pos, int rows, void* stream) {
    idx_pool_kernel<<<rows, 256, 0, cs(stream)>>>(cache, ape, dim, kpool, (long long) pos);
}

void idx_score(const float* q, const float* w, const float* cache, float* score, int score_stride, int dim,
               int n_head, int kpool, int64_t pos, int rows, void* stream) {
    const int n_pool = int((pos + rows) / kpool);   // the last token sees the most
    idx_score_kernel<<<dim3(n_pool, rows), WARP, 0, cs(stream)>>>(q, w, cache, score, score_stride, dim, n_head,
                                                                 kpool, (long long) pos,
                                                                 1.0f / std::sqrt(float(dim) * float(n_head)));
}

void idx_select(const float* score, int score_stride, int32_t* sel, int sel_stride, int n_top, int kpool,
                int64_t pos, int rows, void* stream) {
    idx_select_kernel<<<rows, 256, 2 * size_t(n_top) * sizeof(uint32_t), cs(stream)>>>(
        score, score_stride, sel, sel_stride, n_top, kpool, (long long) pos);
}

void idx_score_batch(void* blas, const float* q, const float* w, const float* cache, float* score, int score_stride,
                     int dim, int n_head, int kpool, int64_t pos, int rows, float* work, size_t work_floats,
                     void* stream) {
    const float one = 1.0f, zero = 0.0f;
    const int n_pool = int((pos + rows) / kpool);   // the last token sees the most
    if (n_pool == 0) return;
    const int chunk = (int) std::min<size_t>(65535, work_floats / (size_t(n_pool) * n_head));
    if (chunk < 1) throw std::invalid_argument("idx_score_batch: no room for a token's products");
    // Pool p's key is the third part of cache row kpool * p + kpool - 1: to cuBLAS the pools' keys are a matrix
    // whose columns lie kpool rows apart.
    const float* keys = cache + (size_t(kpool) * 3 - 1) * dim;
    for (int first = 0; first < rows; first += chunk) {
        const int c = std::min(chunk, rows - first);
        STRATA_BLAS_BEFORE("indexer scores");
        blas_ck(cublasSgemm((cublasHandle_t) blas, CUBLAS_OP_T, CUBLAS_OP_N, n_pool, c * n_head, dim, &one, keys,
                            kpool * 3 * dim, q + size_t(first) * n_head * dim, dim, &zero, work, n_pool),
                "indexer scores");
        idx_weigh_kernel<<<dim3(grid(n_pool, 256), c), 256, 0, cs(stream)>>>(
            work, w + size_t(first) * n_head, score + size_t(first) * score_stride, score_stride, n_pool, n_head,
            kpool, (long long) pos + first, 1.0f / std::sqrt(float(dim) * float(n_head)));
    }
}

void swiglu_clamp(const float* gate, const float* up, float* out, int n, float limit, void* stream) {
    swiglu_clamp_kernel<<<grid(n, 256), 256, 0, cs(stream)>>>(gate, up, out, n, limit);
}

void axpy(const float* x, float a, float* acc, int n, bool add, void* stream) {
    axpy_kernel<<<grid(n, 256), 256, 0, cs(stream)>>>(x, a, acc, n, add);
}

void add_scaled(const ScaledRows& rows, int count, int n, void* stream) {
    if (count < 1 || count > 32) throw std::invalid_argument("add_scaled: 1 to 32 vectors");
    add_scaled_kernel<<<grid(n, 256), 256, 0, cs(stream)>>>(rows, count, n);
}

void swiglu_clamp_rows(const float* gate_up, float* out, int n, int rows, float limit, void* stream) {
    swiglu_clamp_rows_kernel<<<dim3(grid(n, 256), rows), 256, 0, cs(stream)>>>(gate_up, out, n, limit);
}

void geglu(const float* gate, const float* up, float* out, int n, void* stream) {
    geglu_kernel<<<grid(n, 256), 256, 0, cs(stream)>>>(gate, up, out, n);
}

void geglu_rows(const float* gate_up, float* out, int n, int rows, void* stream) {
    geglu_rows_kernel<<<dim3(grid(n, 256), rows), 256, 0, cs(stream)>>>(gate_up, out, n);
}

void moe_sum(const float* x, const float* w, float* acc, int n, int n_used, int rows, void* stream) {
    moe_sum_kernel<<<dim3(grid(n, 256), rows), 256, 0, cs(stream)>>>(x, w, acc, n, n_used);
}

void add_bias(const float* b, float* acc, int n, int rows, void* stream) {
    add_bias_kernel<<<dim3(grid(n, 256), rows), 256, 0, cs(stream)>>>(b, acc, n);
}

void head_norm_rope(const float* x, int x_stride, const float* w, float* y, int dim, int heads, int n_rot,
                    float theta_scale, int64_t pos, int rows, float eps, void* stream, const float* freq,
                    const int32_t* axes, int sections, const long long* at) {
    head_norm_rope_kernel<<<rows * heads, dim >= 256 ? 256 : 128, 0, cs(stream)>>>(
        x, x_stride, w, y, dim, heads, n_rot, theta_scale, freq, (long long) pos, eps, axes, sections, at);
}

namespace {
// The tokens from `first` on that one launch of the rows kernels takes (attend(), attend_q8()), or 0 for the general
// kernels.  A launch of c tokens over a span of `keys` rows (`room` when recorded, else up to its last token's row)
// puts c * n_head * keys scores in the scratch and its parts' sums behind them, parts * c * n_head * dim floats.
// Without score_floats the scratch is counted as `chunk` tokens' scores of this span, as it always was: the launch
// takes min(chunk, rest) tokens or none, so that a profile with a long context (a small chunk) sends a window of 3
// tokens to the general kernels at any position.  With score_floats, the floats the caller's scratch really has
// (chunk tokens' scores of its whole context), the launch takes as many of the rest as fit there, so few tokens near
// the start of a long context take one launch and many near its end take several; a token's sums do not depend on
// the tokens beside it, so the split changes no bits.
int rows_tokens(long long pos, int first, int rest, int chunk, size_t score_floats, int n_head, int dim, int room) {
    const auto need = [&](int c) {
        const size_t keys = room > 0 ? size_t(room) : size_t(pos + first + c);
        const size_t parts = (keys + VALUE_PART - 1) / VALUE_PART;
        return size_t(c) * n_head * keys + parts * c * n_head * dim;
    };
    if (score_floats == 0) {
        const int c = std::min(chunk, rest);
        const size_t keys = room > 0 ? size_t(room) : size_t(pos + first + c);
        return need(c) <= size_t(chunk) * n_head * keys ? c : 0;
    }
    for (int c = rest; c > 0; --c)
        if (need(c) <= score_floats) return c;
    return 0;
}
}  // namespace

void attend(const float* q, const uint16_t* kcache, const uint16_t* vcache, int64_t pos, int rows, float* score,
            int chunk, float* out, int dim, int n_head, int kv_heads, float scale, void* stream, int window,
            int ring, bool whole, const long long* at, int room, const KvSpill* spill, size_t score_floats) {
    if (at && (whole || room < 1)) throw std::invalid_argument("attend: a position on the card needs its room");
    if (spill && ring > 0) throw std::invalid_argument("attend: a ring has no host part");
    if (dim % SCORE_VALUES || uintptr_t(q) % (SCORE_VALUES * sizeof(float)) || uintptr_t(kcache) % (SCORE_VALUES * 2))
        throw std::invalid_argument("attend: queries and keys are read four values at a time");
    const long long last = whole ? pos + rows - 1 : -1;
    const int per = n_head / kv_heads;
    // STRATA_POLY_ATTN=0: the kernels of every shape for Qwen3.5/3.6's too (attn_score_rows_kernel)
    static const bool rows_kernels = [] {
        const char* v = std::getenv("STRATA_POLY_ATTN");
        return !(v != nullptr && v[0] == '0');
    }();
    const bool qwen = rows_kernels && dim == 256 && per == 8 && window == 0 && ring == 0 && !whole && !spill;
    for (int first = 0, c = 0; first < rows; first += c) {
        const int fast = qwen ? rows_tokens(pos, first, rows - first, chunk, score_floats, n_head, dim, at ? room : 0) : 0;
        c = fast > 0 ? fast : std::min(chunk, rows - first);
        // the launch's span: from the first row its first token attends to the last its last token does
        const long long p = pos + first, lo = window > 0 && p >= window ? p - window + 1 : 0;
        const int keys = at ? room : int((whole ? last : p + c - 1) - lo) + 1;
        const dim3 sg(grid(size_t(keys) * kv_heads, SCORE_WARPS));
        const dim3 vg(kv_heads * grid(dim, VALUE_DIMS), grid(c * per, VALUE_QUERIES));
        // the parts' sums lie behind the scores, in the scratch (rows_tokens())
        const int parts = (keys + VALUE_PART - 1) / VALUE_PART;
        const size_t scores = size_t(c) * n_head * keys;
        if (fast > 0) {
            float* part_out = score + scores;
            attn_score_rows_kernel<<<grid(size_t(grid(keys, SCORE_ROWS)) * kv_heads, SCORE_WARPS),
                                     SCORE_WARPS * WARP, 0, cs(stream)>>>(q, (const half*) kcache, score, keys, n_head,
                                                                          kv_heads, (long long) pos, first, c, scale, at);
            attn_softmax_kernel<<<dim3(n_head, c), 256, 0, cs(stream)>>>(score, keys, (long long) pos, first, c,
                                                                        window, last, at);
            const half* vc = (const half*) vcache;
            if (c * per <= 8)
                attn_value_part_kernel<8><<<dim3(kv_heads, parts, 1), VALUE_WARPS * WARP, 0, cs(stream)>>>(
                    score, vc, part_out, keys, dim, n_head, kv_heads, (long long) pos, first, c, at);
            else
                attn_value_part_kernel<16><<<dim3(kv_heads, parts, grid(c * per, 16)), VALUE_WARPS * WARP, 0,
                                             cs(stream)>>>(score, vc, part_out, keys, dim, n_head, kv_heads,
                                                           (long long) pos, first, c, at);
            attn_value_sum_kernel<<<grid(size_t(c) * n_head * dim, 256), 256, 0, cs(stream)>>>(
                part_out, out, dim, n_head, kv_heads, (long long) pos, first, c, at);
            continue;
        }
        if (spill)
            attn_score_kernel<true><<<sg, SCORE_WARPS * WARP, 0, cs(stream)>>>(
                q, (const half*) kcache, score, keys, dim, n_head, kv_heads, (long long) pos, first, c, scale, window,
                ring, last, at, (const half*) spill->khost, spill->split);
        else
            attn_score_kernel<false><<<sg, SCORE_WARPS * WARP, 0, cs(stream)>>>(
                q, (const half*) kcache, score, keys, dim, n_head, kv_heads, (long long) pos, first, c, scale, window,
                ring, last, at, nullptr, 0);
        attn_softmax_kernel<<<dim3(n_head, c), 256, 0, cs(stream)>>>(score, keys, (long long) pos, first, c, window,
                                                                    last, at);
        if (spill)
            attn_value_kernel<true><<<vg, VALUE_DIMS * VALUE_SHARES, 0, cs(stream)>>>(
                score, (const half*) vcache, out, keys, dim, n_head, kv_heads, (long long) pos, first, c, window,
                ring, last, at, (const half*) spill->vhost, spill->split);
        else
            attn_value_kernel<false><<<vg, VALUE_DIMS * VALUE_SHARES, 0, cs(stream)>>>(
                score, (const half*) vcache, out, keys, dim, n_head, kv_heads, (long long) pos, first, c, window,
                ring, last, at, nullptr, 0);
    }
}

int attend_batch_block(size_t score_floats, int rows, int n_head) {
    // as attend_batch() sizes its first chunk's blocks: no block of a call has more rows
    const size_t cells = score_floats / 3 * 2;
    const int c = (int) std::min({size_t(rows), size_t(65535 / n_head), cells / n_head});
    const int fit = (int) (cells / (size_t(c) * n_head));
    return fit > 16 ? fit & ~15 : fit;
}

void attend_batch(void* blas, const float* q, const uint16_t* kcache, const uint16_t* vcache, int64_t pos, int rows,
                  float* score, size_t score_floats, float* work, float* out, int dim, int n_head, int kv_heads,
                  float scale, void* stream, int window, int ring, bool whole, const KvSpill* spill) {
    if (spill && ring > 0) throw std::invalid_argument("attend_batch: a ring has no host part");
    if (spill && (kv_heads * dim) % 8) throw std::invalid_argument("attend_batch: a row is copied 16 bytes at a time");
    const cublasHandle_t h = (cublasHandle_t) blas;
    const float one = 1.0f, zero = 0.0f;
    const int per = n_head / kv_heads, width = kv_heads * dim;
    const long long last = whole ? pos + rows - 1 : -1;
    // The products take halves and sum floats, on the tensor cores: the cache's rows as they lie there, the
    // queries and the softmax's weights rounded.  Two thirds of the scratch are for a block's scores, the rest
    // for their weights.
    const size_t cells = score_floats / 3 * 2;
    half* weight = (half*) (score + cells);
    for (int first = 0, c = 0; first < rows; first += c) {
        // every token left, unless a grid's 65535 rows or the scratch for one row's scores is less
        c = (int) std::min({size_t(rows - first), size_t(65535 / n_head), cells / n_head});
        const long long p0 = pos + first;   // the chunk's first position, and the first row its tokens read
        const long long lo0 = window > 0 && p0 >= window ? p0 - window + 1 : 0;
        const int n_keys = int((whole ? last : p0 + c - 1) - lo0) + 1;
        const size_t heads = size_t(c) * n_head;
        half* qg = (half*) work;                      // the queries by group
        float* og = work + (heads * dim + 1) / 2;     // the results by group
        float* top = og + heads * dim;       // a token and head's largest score so far, the sum of its weights
        float* sum = top + heads;            // against it, and what a block of rows makes the sums before it worth
        float* shrink = sum + heads;
        attn_group_kernel<<<dim3(grid(dim, 256), c * n_head), 256, 0, cs(stream)>>>(q + size_t(first) * n_head * dim,
                                                                                   qg, dim, n_head, kv_heads, c);
        if (prefill_mark_fine) prefill_mark_fine(stream, kPpAttnGroup);
        // The rows go a block at a time, as many as have their scores in the scratch, so that a product has the
        // chunk's every token for its columns however long the context is: cuBLAS is several times slower on a
        // few columns.  A token's softmax is kept from block to block (top, sum), and its values so far shrink
        // to the new top.
        const long long cols = (long long) c * per;
        const int fit = (int) (cells / heads);
        const int block = fit > 16 ? fit & ~15 : fit;   // the tensor cores are half as fast on a ragged block
        for (int k0 = 0, m = 0; k0 < n_keys; k0 += m) {
            // the block's rows where the cache has them: a block ends where a ring starts over
            const long long row = ring > 0 ? (lo0 + k0) % ring : lo0 + k0;
            m = (int) std::min<long long>(std::min(block, n_keys - k0), ring > 0 ? ring - row : n_keys);
            // A block that reaches the host part is copied whole to the stage first: the same rows, so the same
            // products, as from a cache wholly in VRAM.
            const half* kb = (const half*) kcache + size_t(row) * width;
            const half* vb = (const half*) vcache + size_t(row) * width;
            if (spill && row + m > spill->split) {
                if (m > spill->stage_rows) throw std::invalid_argument("attend_batch: the stage holds fewer rows");
                const unsigned g = grid(size_t(m) * width / 8, 256);
                kv_gather_kernel<<<g, 256, 0, cs(stream)>>>((const half*) kcache, (const half*) spill->khost,
                                                            spill->split, row, m, width, (half*) spill->kstage);
                kv_gather_kernel<<<g, 256, 0, cs(stream)>>>((const half*) vcache, (const half*) spill->vhost,
                                                            spill->split, row, m, width, (half*) spill->vstage);
                kb = (const half*) spill->kstage;
                vb = (const half*) spill->vstage;
            }
            // A product a group: its heads' queries of every token of the chunk against the group's keys.  cuBLAS
            // reads the cache's rows as the transpose, and the groups' keys lie `dim` apart in a row.
            STRATA_BLAS_BEFORE("attention scores");
            blas_ck(cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, m, (int) cols, dim, &scale,
                                               kb, CUDA_R_16F, width, dim, qg,
                                               CUDA_R_16F, dim, cols * dim, &zero, score, CUDA_R_32F, m, cols * m,
                                               kv_heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                    "attention scores");
            if (prefill_mark_fine) prefill_mark_fine(stream, kPpAttnScore);
            attn_softmax_block_kernel<<<c * n_head, 128, 0, cs(stream)>>>(score, weight, m, lo0 + k0, (long long) pos,
                                                                         first, window, c, per, last, top, sum,
                                                                         shrink, k0 == 0);
            if (prefill_mark_fine) prefill_mark_fine(stream, kPpAttnSoftmax);
            if (k0 > 0) {
                attn_scale_kernel<<<dim3(grid(dim, 256), c * n_head), 256, 0, cs(stream)>>>(og, shrink, dim, false);
                if (prefill_mark_fine) prefill_mark_fine(stream, kPpAttnScale);
            }
            STRATA_BLAS_BEFORE("attention values");
            blas_ck(cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, dim, (int) cols, m, &one,
                                               vb, CUDA_R_16F, width, dim,
                                               weight, CUDA_R_16F, m, cols * m, k0 > 0 ? &one : &zero, og, CUDA_R_32F,
                                               dim, cols * dim, kv_heads, CUBLAS_COMPUTE_32F,
                                               CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                    "attention values");
            if (prefill_mark_fine) prefill_mark_fine(stream, kPpAttnValue);
        }
        attn_scale_kernel<<<dim3(grid(dim, 256), c * n_head), 256, 0, cs(stream)>>>(og, sum, dim, true);
        attn_ungroup_kernel<<<dim3(grid(dim, 256), c * n_head), 256, 0, cs(stream)>>>(
            og, out + size_t(first) * n_head * dim, dim, n_head, kv_heads, c);
        if (prefill_mark_fine) prefill_mark_fine(stream, kPpAttnFinish);
    }
}

void softcap(float* x, size_t n, float cap, void* stream) {
    softcap_kernel<<<grid(n, 256), 256, 0, cs(stream)>>>(x, n, cap);
}

void likeliest(const float* logits, int n, Likeliest* out, void* stream) {
    likeliest_kernel<<<1, 1024, 0, cs(stream)>>>(logits, n, out);
}

void mul_sigmoid(float* acc, const float* g, int n, int rows, int g_stride, bool one_gate, void* stream) {
    for (int r = 0; r < rows; r += 65535)   // a grid has at most 65535 rows, and 4,096 tokens have 16 heads each
        mul_sigmoid_kernel<<<dim3(grid(n, 256), std::min(65535, rows - r)), 256, 0, cs(stream)>>>(
            acc + size_t(r) * n, g + size_t(r) * g_stride, n, g_stride, one_gate);
}

void conv4_silu(const float* x, float* state, const float* w, float* out, int channels, int rows, void* stream,
                float* snap, size_t stride, unsigned snaps) {
    conv4_silu_kernel<<<grid(channels, 256), 256, 0, cs(stream)>>>(x, state, w, out, channels, rows, snap, stride,
                                                                  snap ? snaps : 0u);
}

// STRATA_POLY_GDN_STATE: 0 the prompt batches' state update with gdn_state_kernel, 1, 2 or 4 (the default) with
// gdn_state_reg_kernel and that many columns a warp.  gdn_state_way() sets it instead (poly_kernel_bench).
int g_gdn_way = -1;
int gdn_way() {
    if (g_gdn_way < 0) {
        const char* v = std::getenv("STRATA_POLY_GDN_STATE");
        const int w = v && *v ? std::atoi(v) : 4;
        g_gdn_way = w == 1 || w == 2 || w == 4 ? w : 0;
    }
    return g_gdn_way;
}

void gdn_state_way(int way) { g_gdn_way = way; }

void gdn_step(float* qkv, float* alpha, float* beta, const float* dt, const float* a, float* s, const float* z,
              const float* o_norm, float* out, int dim, int n_head, int k_heads, int rows, float norm_eps, void* stream,
              float* snap, size_t stride, unsigned snaps, void* out_q8, bool gates_done) {
    if (dim % WARP != 0 || dim > 256) throw std::invalid_argument("gdn_step: a head of 32 to 256 values, in 32s");
    const size_t row_n = (2 * size_t(k_heads) + n_head) * dim;
    const int gates = rows * n_head, columns = std::min(dim, 1024 / WARP);   // a block's columns: a warp each
    gdn_norm_kernel<<<rows * k_heads, dim, 0, cs(stream)>>>(qkv, dim, k_heads, row_n, norm_eps);
    if (prefill_mark) prefill_mark(stream, kPpGdnNorm);
    if (!gates_done) gdn_gates_kernel<<<grid(gates, 256), 256, 0, cs(stream)>>>(alpha, beta, dt, a, n_head, gates);
    if (prefill_mark) prefill_mark(stream, kPpGdnGates);
    // a prompt batch (32 tokens or more, no snapshots) of heads of 128: the state in registers
    const int way = !snap && rows >= 32 && dim == 128 ? gdn_way() : 0;
    const int warps = way ? n_head * (dim / way) : 0, blocks = (warps + 7) / 8;
    if (way == 4)
        gdn_state_reg_kernel<128, 4><<<blocks, 256, 0, cs(stream)>>>(qkv, alpha, beta, s, out, n_head, k_heads, rows);
    else if (way == 2)
        gdn_state_reg_kernel<128, 2><<<blocks, 256, 0, cs(stream)>>>(qkv, alpha, beta, s, out, n_head, k_heads, rows);
    else if (way == 1)
        gdn_state_reg_kernel<128, 1><<<blocks, 256, 0, cs(stream)>>>(qkv, alpha, beta, s, out, n_head, k_heads, rows);
    else
        gdn_state_kernel<<<n_head * ((dim + columns - 1) / columns), columns * WARP, 0, cs(stream)>>>(
            qkv, alpha, beta, s, out, dim, n_head, k_heads, rows, snap, stride, snap ? snaps : 0u);
    if (prefill_mark) prefill_mark(stream, kPpGdnState);
    if (out_q8)
        gdn_out_q8_kernel<<<rows * n_head, dim, 0, cs(stream)>>>(out, z, o_norm, (Q81*) out_q8, dim, norm_eps);
    else
        gdn_out_kernel<<<rows * n_head, dim, 0, cs(stream)>>>(out, z, o_norm, dim, norm_eps);
    if (prefill_mark) prefill_mark(stream, kPpGdnOut);
}

// ---- STRATA_POLY_FUSE (kernels.hpp)
bool rms_norm_q8(float* x, const float* add, const float* w, float* y, void* yq, int n, int rows, float eps,
                 void* stream) {
    const int threads = sum_threads(n);
    if (n % threads != 0 || rows < 1) return false;   // every warp of every step has 32 values of the row
    rms_norm_q8_kernel<<<rows, threads, 0, cs(stream)>>>(x, add, 1.f, w, y, (Q81*) yq, n, eps);
    return true;
}

bool swiglu_clamp_q8(const float* gate, const float* up, const float* gate_up, float* out, void* q, int n, int rows,
                     float limit, void* stream) {
    if (n % WARP != 0 || rows < 1 || rows > 65535) return false;
    swiglu_clamp_q8_kernel<<<dim3(grid(n, 256), rows), 256, 0, cs(stream)>>>(gate, up, gate_up, out, (Q81*) q, n,
                                                                            limit);
    return true;
}

bool mul_sigmoid_q8(float* acc, const float* g, void* q, int n, int rows, int g_stride, void* stream) {
    if (n % WARP != 0 || rows < 1 || rows > 65535) return false;
    mul_sigmoid_q8_kernel<<<dim3(grid(n, 256), rows), 256, 0, cs(stream)>>>(acc, g, (Q81*) q, n, g_stride);
    return true;
}

void gdn_ab(const float* wa, const float* wb, const float* x, float* alpha, float* beta, const float* dt,
            const float* a, int n_in, int n_head, int rows, void* stream) {
    const int threads = n_in >= 128 ? sum_threads(n_in) : 32;   // gemv()'s
    gdn_ab_kernel<<<dim3(2 * n_head, rows), threads, 0, cs(stream)>>>(wa, wb, x, alpha, beta, dt, a, n_in, n_head);
}

// ---- the attention over a q8_0 cache (docs/AMD_HIP_KV_Q8.md, kernels.hpp).  The fp16 kernels above are not touched:
// these are their copies with the reads of the cache changed, so that the fp16 path stays as it was to the bit.
namespace {

// q8_head_bytes() of kernels.hpp, for the card too (check_q8() holds them to the same)
__host__ __device__ __forceinline__ size_t q8_hb(int dim) {
    return (size_t(dim) + size_t(dim / kQ8Block) * 2 + 15) & ~size_t(15);
}

// a head of a q8_0 row: its dim codes, then its dim / 32 fp16 scales
__device__ __forceinline__ float q8_scale(const uint8_t* head, int dim, int block) {
    return __half2float(reinterpret_cast<const half*>(head + dim)[block]);
}

// 8 codes (8 bytes, one read) times their scale: the values exactly, a code of 8 bits times a scale of 11 fitting a
// float's 24
__device__ __forceinline__ void q8_eight(const uint8_t* codes, float s, float* x) {
    const uint2 raw = *reinterpret_cast<const uint2*>(codes);
    const int8_t* c = reinterpret_cast<const int8_t*>(&raw);
#pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = float(c[j]) * s;
}

// A warp quantizes a block of 32 values, a lane each, as ggml's quantize_row_q8_0 does: d = max|x| / 127, a code is
// x * (1 / d) rounded half away from zero (0 where d is), and d is kept as fp16.  The divisions and the product are
// rounded as the CPU's are, whatever the compiler's flags.
__device__ __forceinline__ void q8_block(float x, uint8_t* head, int dim, int b, int lane) {
    float a = fabsf(x);
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    const float d = __fdiv_rn(a, 127.0f);
    const float id = d != 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
    head[b * kQ8Block + lane] = (uint8_t) (int8_t) roundf(__fmul_rn(x, id));
    if (lane == 0) reinterpret_cast<half*>(head + dim)[b] = __float2half(d);
}

// store_q8 / store_q8_at: a warp a block of 32 values of a token's head.  at: the first token's row, read on the card.
__global__ void store_q8_kernel(const float* __restrict__ x, uint8_t* __restrict__ cache, int dim, int kv_heads,
                                long long pos, int rows, const long long* __restrict__ at) {
    const int blocks = dim / kQ8Block, per_row = kv_heads * blocks;
    const int unit = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP, lane = threadIdx.x % WARP;
    if (unit >= rows * per_row) return;   // the whole warp
    const int t = unit / per_row, h = unit % per_row / blocks, b = unit % blocks;
    const size_t hb = q8_hb(dim);
    uint8_t* head = cache + size_t((at ? *at : pos) + t) * kv_heads * hb + size_t(h) * hb;
    q8_block(x[(size_t(t) * kv_heads + h) * dim + size_t(b) * kQ8Block + lane], head, dim, b, lane);
}

// kv_dequant_q8: a thread 8 values, written as one 16-byte store of halves
__global__ void kv_dequant_q8_kernel(const uint8_t* __restrict__ cache, long long row0, int m, int dim, int kv_heads,
                                     half* __restrict__ out) {
    const int per = dim / 8;
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(m) * kv_heads * per) return;
    const int c = int(i % per), h = int(i / per % kv_heads);
    const size_t r = i / (size_t(per) * kv_heads), hb = q8_hb(dim);
    const uint8_t* head = cache + size_t(row0 + (long long) r) * kv_heads * hb + size_t(h) * hb;
    float x[8];
    q8_eight(head + 8 * c, q8_scale(head, dim, 8 * c / kQ8Block), x);
    uint4 o;
    half2* o2 = reinterpret_cast<half2*>(&o);
#pragma unroll
    for (int j = 0; j < 4; ++j) o2[j] = __floats2half2_rn(x[2 * j], x[2 * j + 1]);
    *reinterpret_cast<uint4*>(out + (r * kv_heads + h) * size_t(dim) + 8 * c) = o;
}

// attn_scores() over a q8_0 key: a lane's four values a step are four codes in one read, times the block's scale.
// k: the key's head.
template <int N>
__device__ __forceinline__ void attn_scores_q8(const float* __restrict__ q, const uint8_t* __restrict__ k,
                                               float* __restrict__ score, int n_keys, int dim, float scale) {
    const int lane = threadIdx.x % WARP;
    const float* qi[N];
    float acc[N];
#pragma unroll
    for (int i = 0; i < N; ++i) {
        qi[i] = q + size_t(i) * dim;
        acc[i] = 0.0f;
    }
    for (int d = SCORE_VALUES * lane; d < dim; d += SCORE_VALUES * WARP) {
        const uint32_t bits = *reinterpret_cast<const uint32_t*>(k + d);
        const float s = q8_scale(k, dim, d / kQ8Block);
        float x[SCORE_VALUES];
#pragma unroll
        for (int j = 0; j < SCORE_VALUES; ++j) x[j] = float((int8_t) (bits >> 8 * j)) * s;
#pragma unroll
        for (int i = 0; i < N; ++i) {
            const float4 y = *reinterpret_cast<const float4*>(qi[i] + d);
            acc[i] += y.x * x[0];
            acc[i] += y.y * x[1];
            acc[i] += y.z * x[2];
            acc[i] += y.w * x[3];
        }
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
        for (int off = WARP / 2; off > 0; off >>= 1) acc[i] += __shfl_down_sync(0xffffffffu, acc[i], off);
        if (lane == 0) score[size_t(i) * n_keys] = acc[i] * scale;
    }
}

template <int N>
__device__ __forceinline__ void attn_scores_by_q8(int n, const float* __restrict__ q, const uint8_t* __restrict__ k,
                                                  float* __restrict__ score, int n_keys, int dim, float scale) {
    for (; n >= N; n -= N, q += size_t(N) * dim, score += size_t(N) * n_keys)
        attn_scores_q8<N>(q, k, score, n_keys, dim, scale);
    if constexpr (N > 1) attn_scores_by_q8<N / 2>(n, q, k, score, n_keys, dim, scale);
}

// attn_score_kernel over a q8_0 cache (no host part)
__global__ void attn_score_q8_kernel(const float* __restrict__ q, const uint8_t* __restrict__ kcache,
                                     float* __restrict__ score, int n_keys, int dim, int n_head, int kv_heads,
                                     long long pos, int first, int tokens, float scale, int window, int ring,
                                     long long last, const long long* __restrict__ at) {
    const int unit = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;   // a row and a group
    const int key = unit / kv_heads, g = unit % kv_heads, per = n_head / kv_heads;
    long long lo;
    if (key >= attn_span((at ? *at : pos) + first, tokens, window, last, &lo)) return;
    const long long row = ring > 0 ? (lo + key) % ring : lo + key;
    const size_t hb = q8_hb(dim);
    const uint8_t* k = kcache + (size_t(row) * kv_heads + g) * hb;
    q += (size_t(first) * n_head + g * per) * dim;
    score += size_t(g) * per * n_keys + key;
    for (int t = 0; t < tokens; ++t, q += size_t(n_head) * dim, score += size_t(n_head) * n_keys)
        attn_scores_by_q8<SCORE_QUERIES>(per, q, k, score, n_keys, dim, scale);
}

// attn_values() over a q8_0 cache: the thread's value d of its head in each of its rows, a code and its scale.
// v: the head in row 0.
template <int N>
__device__ __forceinline__ void attn_values_q8(const float* __restrict__ w, const uint8_t* __restrict__ v, size_t row_b,
                                               int d, int dim, long long row, int ring, int span, float* sums) {
    float acc[N];
#pragma unroll
    for (int i = 0; i < N; ++i) acc[i] = 0.0f;
    v += size_t(row) * row_b;
    for (int j = threadIdx.x / VALUE_DIMS; j < span; j += VALUE_SHARES) {
        const float x = float((int8_t) v[d]) * q8_scale(v, dim, d / kQ8Block), weight = w[j];
#pragma unroll
        for (int i = 0; i < N; ++i) acc[i] += __shfl_sync(0xffffffffu, weight, i) * x;
        v += VALUE_SHARES * row_b;
        row += VALUE_SHARES;
        if (ring > 0 && row >= ring) row -= ring, v -= size_t(ring) * row_b;
    }
#pragma unroll
    for (int i = 0; i < N; ++i) sums[i] = acc[i];
}

template <int N>
__device__ __forceinline__ void attn_values_for_q8(int n, const float* __restrict__ w, const uint8_t* __restrict__ v,
                                                   size_t row_b, int d, int dim, long long row, int ring, int span,
                                                   float* sums) {
    if constexpr (N > 1)
        if (n <= N / 2) return attn_values_for_q8<N / 2>(n, w, v, row_b, d, dim, row, ring, span, sums);
    attn_values_q8<N>(w, v, row_b, d, dim, row, ring, span, sums);
}

// attn_value_kernel over a q8_0 cache (no host part)
__global__ void attn_value_q8_kernel(const float* __restrict__ score, const uint8_t* __restrict__ vcache,
                                     float* __restrict__ out, int n_keys, int dim, int n_head, int kv_heads,
                                     long long pos, int first, int tokens, int window, int ring, long long last,
                                     const long long* __restrict__ at) {
    __shared__ float share[VALUE_QUERIES][VALUE_SHARES][VALUE_DIMS];
    const int per = n_head / kv_heads, dims = (dim + VALUE_DIMS - 1) / VALUE_DIMS;
    const int g = blockIdx.x / dims, q0 = blockIdx.y * VALUE_QUERIES;
    const int live = min(VALUE_QUERIES, tokens * per - q0);
    const int lane = threadIdx.x % VALUE_DIMS, s = threadIdx.x / VALUE_DIMS;
    const int d = (blockIdx.x % dims) * VALUE_DIMS + lane;
    const long long p = (at ? *at : pos) + first;
    long long lo;
    const int span = attn_span(p, tokens, window, last, &lo);
    const int mine = q0 + min(lane % VALUE_QUERIES, live - 1);
    float sums[VALUE_QUERIES];
    const size_t hb = q8_hb(dim);
    attn_values_for_q8<VALUE_QUERIES>(live, score + (size_t(mine / per) * n_head + g * per + mine % per) * n_keys,
                                      vcache + size_t(g) * hb, size_t(kv_heads) * hb, min(d, dim - 1), dim,
                                      ring > 0 ? (lo + s) % ring : lo + s, ring, span, sums);
    for (int i = 0; i < live; ++i) {
        long long own;
        attn_rows(p + (q0 + i) / per, window, last, &own);
        share[i][(s + VALUE_SHARES - int((own - lo) % VALUE_SHARES)) % VALUE_SHARES][lane] = sums[i];
    }
    __syncthreads();
    if (s != 0 || d >= dim) return;
    for (int i = 0; i < live; ++i) {
        float sum = share[i][0][lane];
        for (int k = 1; k < VALUE_SHARES; ++k) sum += share[i][k][lane];
        const int query = q0 + i;
        out[((size_t(first) + query / per) * n_head + g * per + query % per) * dim + d] = sum;
    }
}

// attn_score_rows_kernel over a q8_0 cache.  A lane takes 8 values of a row's head, a block's quarter, in one read of
// 8 codes and one of their scale, where the fp16 kernel reads twice 4 halves; its queries are those 8 values of each
// of the group's heads.
__global__ void __launch_bounds__(SCORE_WARPS * WARP)
    attn_score_rows_q8_kernel(const float* __restrict__ q, const uint8_t* __restrict__ kcache,
                              float* __restrict__ score, int n_keys, int n_head, int kv_heads, long long pos,
                              int first, int tokens, float scale, const long long* __restrict__ at) {
    constexpr int dim = 256, per = 8;
    const size_t hb = q8_hb(dim), row_b = size_t(kv_heads) * hb;
    const int unit = blockIdx.x * SCORE_WARPS + threadIdx.x / WARP;
    const int g = unit % kv_heads, key0 = unit / kv_heads * SCORE_ROWS;
    long long lo;
    const int span = attn_span((at ? *at : pos) + first, tokens, 0, -1, &lo);
    if (key0 >= span) return;
    const int keys = min(SCORE_ROWS, span - key0), lane = threadIdx.x % WARP;
    const int mine = ((lane >> 4) & 1) * 4 + ((lane >> 3) & 1) * 2 + ((lane >> 2) & 1);
    for (int t = 0; t < tokens; ++t) {
        const float* qt = q + ((size_t(first) + t) * n_head + size_t(g) * per) * dim + 8 * lane;
        float4 y[per][2];
#pragma unroll
        for (int h = 0; h < per; ++h) {
            y[h][0] = *reinterpret_cast<const float4*>(qt + size_t(h) * dim);
            y[h][1] = *reinterpret_cast<const float4*>(qt + size_t(h) * dim + 4);
        }
        float* sc = score + (size_t(t) * n_head + size_t(g) * per + mine) * n_keys + key0;
        for (int r = 0; r < keys; ++r) {
            const uint8_t* k = kcache + (size_t(lo) + key0 + r) * row_b + size_t(g) * hb;
            float x[8];
            q8_eight(k + 8 * lane, q8_scale(k, dim, lane / 4), x);
            float acc[per];
#pragma unroll
            for (int h = 0; h < per; ++h) {
                acc[h] = y[h][0].x * x[0];
                acc[h] += y[h][0].y * x[1];
                acc[h] += y[h][0].z * x[2];
                acc[h] += y[h][0].w * x[3];
                acc[h] += y[h][1].x * x[4];
                acc[h] += y[h][1].y * x[5];
                acc[h] += y[h][1].z * x[6];
                acc[h] += y[h][1].w * x[7];
            }
            const float sum = heads_sum(acc, lane);
            if ((lane & 3) == 0) sc[r] = sum * scale;
        }
    }
}

// attn_value_part_kernel over a q8_0 cache: a lane's 8 values of a row are 8 codes in one read and their scale
template <int Q>
__global__ void __launch_bounds__(VALUE_WARPS * WARP)
    attn_value_part_q8_kernel(const float* __restrict__ score, const uint8_t* __restrict__ vcache,
                              float* __restrict__ part_out, int n_keys, int dim, int n_head, int kv_heads,
                              long long pos, int first, int tokens, const long long* __restrict__ at) {
    static_assert(VALUE_SUB % VALUE_WARPS == 0 && VALUE_PART % VALUE_SUB == 0, "whole rounds of the shares");
    static_assert(Q % 4 == 0 && Q % VALUE_ROUND == 0, "a row's weights are read four at a time");
    constexpr int D = 256, E = D / WARP;
    constexpr int QP = Q + 4;
    constexpr int words = VALUE_SUB * QP > VALUE_ROUND * VALUE_WARPS * D ? VALUE_SUB * QP : VALUE_ROUND * VALUE_WARPS * D;
    __shared__ __align__(16) float mem[words];
    float(*wt)[QP] = reinterpret_cast<float(*)[QP]>(mem);
    float(*share)[VALUE_WARPS][D] = reinterpret_cast<float(*)[VALUE_WARPS][D]>(mem);
    const int per = n_head / kv_heads, all = tokens * per;
    const int g = blockIdx.x, q0 = blockIdx.z * Q, live = min(Q, all - q0);
    const int lane = threadIdx.x % WARP, s = threadIdx.x / WARP, tid = threadIdx.x;
    long long lo;
    const int span = attn_span((at ? *at : pos) + first, tokens, 0, -1, &lo);
    const int part0 = blockIdx.y * VALUE_PART, rows = max(0, min(span - part0, VALUE_PART));
    const size_t hb = q8_hb(D), row_b = size_t(kv_heads) * hb;
    const uint8_t* v = vcache + (size_t(lo) + part0) * row_b + size_t(g) * hb;
    float acc[Q][E];
#pragma unroll
    for (int i = 0; i < Q; ++i)
#pragma unroll
        for (int e = 0; e < E; ++e) acc[i][e] = 0.0f;
    for (int sub = 0; sub < rows; sub += VALUE_SUB) {
        const int n = min(VALUE_SUB, rows - sub);
        __syncthreads();
        for (int i = tid; i < Q * VALUE_SUB; i += VALUE_WARPS * WARP) {
            const int j = i / VALUE_SUB, r = i % VALUE_SUB, qi = q0 + min(j, live - 1);
            wt[r][j] = r < n ? score[(size_t(qi / per) * n_head + size_t(g) * per + qi % per) * n_keys + part0 + sub + r]
                             : 0.0f;
        }
        __syncthreads();
        for (int r = s; r < n; r += VALUE_WARPS) {
            const uint8_t* vr = v + size_t(sub + r) * row_b;
            float x[E];
            q8_eight(vr + E * lane, q8_scale(vr, D, lane / 4), x);
#pragma unroll
            for (int c = 0; c < Q / 4; ++c) {
                const float4 w = *reinterpret_cast<const float4*>(&wt[r][4 * c]);
#pragma unroll
                for (int e = 0; e < E; ++e) {
                    acc[4 * c][e] += w.x * x[e];
                    acc[4 * c + 1][e] += w.y * x[e];
                    acc[4 * c + 2][e] += w.z * x[e];
                    acc[4 * c + 3][e] += w.w * x[e];
                }
            }
        }
    }
    float* dst = part_out + ((size_t(blockIdx.y) * kv_heads + g) * all + q0) * dim;
#pragma unroll
    for (int c = 0; c < Q; c += VALUE_ROUND) {
        if (c >= live) break;
        __syncthreads();
#pragma unroll
        for (int i = 0; i < VALUE_ROUND; ++i)
#pragma unroll
            for (int e = 0; e < E; ++e) share[i][s][E * lane + e] = acc[c + i][e];
        __syncthreads();
        for (int i = tid / D; i < VALUE_ROUND; i += VALUE_WARPS * WARP / D) {
            if (c + i >= live) break;
            const int d = tid % D;
            float sum = share[i][0][d];
#pragma unroll
            for (int k = 1; k < VALUE_WARPS; ++k) sum += share[i][k][d];
            dst[size_t(c + i) * dim + d] = sum;
        }
    }
}

void check_q8(int dim, int kv_heads, const char* what) {
    if (dim < kQ8Block || dim % kQ8Block || kv_heads < 1)
        throw std::invalid_argument(std::string(what) + ": a head is whole blocks of 32 values");
    if (q8_hb(dim) != q8_head_bytes(dim)) throw std::logic_error(std::string(what) + ": two sizes of a head");
}

}  // namespace

void store_q8(const float* x, uint8_t* cache, int dim, int kv_heads, int64_t pos, int rows, void* stream) {
    check_q8(dim, kv_heads, "store_q8");
    const size_t warps = size_t(rows) * kv_heads * (dim / kQ8Block);
    store_q8_kernel<<<grid(warps, 256 / WARP), 256, 0, cs(stream)>>>(x, cache, dim, kv_heads, (long long) pos, rows,
                                                                    nullptr);
}

void store_q8_at(const float* x, uint8_t* cache, int dim, int kv_heads, const long long* at, int rows, void* stream) {
    check_q8(dim, kv_heads, "store_q8_at");
    const size_t warps = size_t(rows) * kv_heads * (dim / kQ8Block);
    store_q8_kernel<<<grid(warps, 256 / WARP), 256, 0, cs(stream)>>>(x, cache, dim, kv_heads, 0, rows, at);
}

void kv_dequant_q8(const uint8_t* cache, long long row0, int m, int dim, int kv_heads, uint16_t* out, void* stream) {
    check_q8(dim, kv_heads, "kv_dequant_q8");
    if (m < 1) return;
    kv_dequant_q8_kernel<<<grid(size_t(m) * kv_heads * (dim / 8), 256), 256, 0, cs(stream)>>>(cache, row0, m, dim,
                                                                                              kv_heads, (half*) out);
}

void attend_kv(const float* q, const void* kcache, bool k_q8, const void* vcache, bool v_q8, int64_t pos, int rows,
               float* score, int chunk, float* out, int dim, int n_head, int kv_heads, float scale, void* stream,
               int window, int ring, bool whole, const long long* at, int room, size_t score_floats) {
    if (k_q8 || v_q8) check_q8(dim, kv_heads, "attend_kv");
    if (at && (whole || room < 1)) throw std::invalid_argument("attend_kv: a position on the card needs its room");
    if (dim % SCORE_VALUES || uintptr_t(q) % 16 || uintptr_t(kcache) % 16 || uintptr_t(vcache) % 16)
        throw std::invalid_argument("attend_kv: queries and caches lie at multiples of 16 bytes");
    const uint8_t* kq = (const uint8_t*) kcache;
    const uint8_t* vq = (const uint8_t*) vcache;
    const half* kh = (const half*) kcache;
    const half* vh = (const half*) vcache;
    const long long last = whole ? pos + rows - 1 : -1;
    const int per = n_head / kv_heads;
    static const bool rows_kernels = [] {   // STRATA_POLY_ATTN=0, as attend()
        const char* v = std::getenv("STRATA_POLY_ATTN");
        return !(v != nullptr && v[0] == '0');
    }();
    const bool qwen = rows_kernels && dim == 256 && per == 8 && window == 0 && ring == 0 && !whole;
    for (int first = 0, c = 0; first < rows; first += c) {
        // the same choice as attend()'s, so that the same calls take the same kernels' shapes
        const int fast = qwen ? rows_tokens(pos, first, rows - first, chunk, score_floats, n_head, dim, at ? room : 0) : 0;
        c = fast > 0 ? fast : std::min(chunk, rows - first);
        const long long p = pos + first, lo = window > 0 && p >= window ? p - window + 1 : 0;
        const int keys = at ? room : int((whole ? last : p + c - 1) - lo) + 1;
        const dim3 sg(grid(size_t(keys) * kv_heads, SCORE_WARPS));
        const dim3 vg(kv_heads * grid(dim, VALUE_DIMS), grid(c * per, VALUE_QUERIES));
        const int parts = (keys + VALUE_PART - 1) / VALUE_PART;
        const size_t scores = size_t(c) * n_head * keys;
        if (fast > 0) {
            // the scores read the keys and the values' parts the values, each cache in its own format
            float* part_out = score + scores;
            const unsigned rg = grid(size_t(grid(keys, SCORE_ROWS)) * kv_heads, SCORE_WARPS);
            if (k_q8)
                attn_score_rows_q8_kernel<<<rg, SCORE_WARPS * WARP, 0, cs(stream)>>>(q, kq, score, keys, n_head, kv_heads,
                                                                                    (long long) pos, first, c, scale, at);
            else
                attn_score_rows_kernel<<<rg, SCORE_WARPS * WARP, 0, cs(stream)>>>(q, kh, score, keys, n_head, kv_heads,
                                                                                 (long long) pos, first, c, scale, at);
            attn_softmax_kernel<<<dim3(n_head, c), 256, 0, cs(stream)>>>(score, keys, (long long) pos, first, c,
                                                                        window, last, at);
            const dim3 v8(kv_heads, parts, 1), v16(kv_heads, parts, grid(c * per, 16));
            if (c * per <= 8 && v_q8)
                attn_value_part_q8_kernel<8><<<v8, VALUE_WARPS * WARP, 0, cs(stream)>>>(
                    score, vq, part_out, keys, dim, n_head, kv_heads, (long long) pos, first, c, at);
            else if (c * per <= 8)
                attn_value_part_kernel<8><<<v8, VALUE_WARPS * WARP, 0, cs(stream)>>>(
                    score, vh, part_out, keys, dim, n_head, kv_heads, (long long) pos, first, c, at);
            else if (v_q8)
                attn_value_part_q8_kernel<16><<<v16, VALUE_WARPS * WARP, 0, cs(stream)>>>(
                    score, vq, part_out, keys, dim, n_head, kv_heads, (long long) pos, first, c, at);
            else
                attn_value_part_kernel<16><<<v16, VALUE_WARPS * WARP, 0, cs(stream)>>>(
                    score, vh, part_out, keys, dim, n_head, kv_heads, (long long) pos, first, c, at);
            attn_value_sum_kernel<<<grid(size_t(c) * n_head * dim, 256), 256, 0, cs(stream)>>>(
                part_out, out, dim, n_head, kv_heads, (long long) pos, first, c, at);
            continue;
        }
        if (k_q8)
            attn_score_q8_kernel<<<sg, SCORE_WARPS * WARP, 0, cs(stream)>>>(q, kq, score, keys, dim, n_head, kv_heads,
                                                                           (long long) pos, first, c, scale, window, ring,
                                                                           last, at);
        else
            attn_score_kernel<false><<<sg, SCORE_WARPS * WARP, 0, cs(stream)>>>(
                q, kh, score, keys, dim, n_head, kv_heads, (long long) pos, first, c, scale, window, ring, last, at,
                nullptr, 0);
        attn_softmax_kernel<<<dim3(n_head, c), 256, 0, cs(stream)>>>(score, keys, (long long) pos, first, c, window,
                                                                    last, at);
        if (v_q8)
            attn_value_q8_kernel<<<vg, VALUE_DIMS * VALUE_SHARES, 0, cs(stream)>>>(score, vq, out, keys, dim, n_head,
                                                                                  kv_heads, (long long) pos, first, c,
                                                                                  window, ring, last, at);
        else
            attn_value_kernel<false><<<vg, VALUE_DIMS * VALUE_SHARES, 0, cs(stream)>>>(
                score, vh, out, keys, dim, n_head, kv_heads, (long long) pos, first, c, window, ring, last, at,
                nullptr, 0);
    }
}

void attend_q8(const float* q, const uint8_t* kcache, const uint8_t* vcache, int64_t pos, int rows, float* score,
               int chunk, float* out, int dim, int n_head, int kv_heads, float scale, void* stream, int window,
               int ring, bool whole, const long long* at, int room, size_t score_floats) {
    check_q8(dim, kv_heads, "attend_q8");
    attend_kv(q, kcache, true, vcache, true, pos, rows, score, chunk, out, dim, n_head, kv_heads, scale, stream, window,
              ring, whole, at, room, score_floats);
}

void attend_batch_kv(void* blas, const float* q, const void* kcache, bool k_q8, const void* vcache, bool v_q8,
                     int64_t pos, int rows, float* score, size_t score_floats, float* work, float* out, int dim,
                     int n_head, int kv_heads, float scale, void* stream, uint16_t* kstage, uint16_t* vstage,
                     int stage_rows, int window, int ring, bool whole) {
    if (k_q8 || v_q8) check_q8(dim, kv_heads, "attend_batch_kv");
    if ((k_q8 && !kstage) || (v_q8 && !vstage) || ((k_q8 || v_q8) && stage_rows < 1))
        throw std::invalid_argument("attend_batch_kv: no stage for the blocks");
    const cublasHandle_t h = (cublasHandle_t) blas;
    const float one = 1.0f, zero = 0.0f;
    const int per = n_head / kv_heads, width = kv_heads * dim;
    const long long last = whole ? pos + rows - 1 : -1;
    const size_t cells = score_floats / 3 * 2;
    half* weight = (half*) (score + cells);
    // attend_batch()'s blocks, but no larger than the stage when a cache is q8_0
    const int cap = !(k_q8 || v_q8) ? std::numeric_limits<int>::max() : stage_rows > 16 ? stage_rows & ~15 : stage_rows;
    for (int first = 0, c = 0; first < rows; first += c) {
        c = (int) std::min({size_t(rows - first), size_t(65535 / n_head), cells / n_head});
        const long long p0 = pos + first;
        const long long lo0 = window > 0 && p0 >= window ? p0 - window + 1 : 0;
        const int n_keys = int((whole ? last : p0 + c - 1) - lo0) + 1;
        const size_t heads = size_t(c) * n_head;
        half* qg = (half*) work;
        float* og = work + (heads * dim + 1) / 2;
        float* top = og + heads * dim;
        float* sum = top + heads;
        float* shrink = sum + heads;
        attn_group_kernel<<<dim3(grid(dim, 256), c * n_head), 256, 0, cs(stream)>>>(q + size_t(first) * n_head * dim,
                                                                                   qg, dim, n_head, kv_heads, c);
        if (prefill_mark_fine) prefill_mark_fine(stream, kPpAttnGroup);
        const long long cols = (long long) c * per;
        const int fit = (int) (cells / heads);
        const int block = std::min(fit > 16 ? fit & ~15 : fit, cap);
        for (int k0 = 0, m = 0; k0 < n_keys; k0 += m) {
            const long long row = ring > 0 ? (lo0 + k0) % ring : lo0 + k0;
            m = (int) std::min<long long>(std::min(block, n_keys - k0), ring > 0 ? ring - row : n_keys);
            // a q8_0 cache's block of rows as halves in its stage, where the products read them; an fp16 cache's
            // rows where they lie
            const unsigned g = grid(size_t(m) * kv_heads * (dim / 8), 256);
            const half* kb = (const half*) kcache + size_t(row) * width;
            const half* vb = (const half*) vcache + size_t(row) * width;
            if (k_q8) {
                kv_dequant_q8_kernel<<<g, 256, 0, cs(stream)>>>((const uint8_t*) kcache, row, m, dim, kv_heads,
                                                                (half*) kstage);
                kb = (const half*) kstage;
            }
            if (v_q8) {
                kv_dequant_q8_kernel<<<g, 256, 0, cs(stream)>>>((const uint8_t*) vcache, row, m, dim, kv_heads,
                                                                (half*) vstage);
                vb = (const half*) vstage;
            }
            STRATA_BLAS_BEFORE("attention scores");
            blas_ck(cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, m, (int) cols, dim, &scale,
                                               kb, CUDA_R_16F, width, dim, qg,
                                               CUDA_R_16F, dim, cols * dim, &zero, score, CUDA_R_32F, m, cols * m,
                                               kv_heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                    "attention scores");
            if (prefill_mark_fine) prefill_mark_fine(stream, kPpAttnScore);
            attn_softmax_block_kernel<<<c * n_head, 128, 0, cs(stream)>>>(score, weight, m, lo0 + k0, (long long) pos,
                                                                         first, window, c, per, last, top, sum,
                                                                         shrink, k0 == 0);
            if (prefill_mark_fine) prefill_mark_fine(stream, kPpAttnSoftmax);
            if (k0 > 0) {
                attn_scale_kernel<<<dim3(grid(dim, 256), c * n_head), 256, 0, cs(stream)>>>(og, shrink, dim, false);
                if (prefill_mark_fine) prefill_mark_fine(stream, kPpAttnScale);
            }
            STRATA_BLAS_BEFORE("attention values");
            blas_ck(cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, dim, (int) cols, m, &one,
                                               vb, CUDA_R_16F, width, dim,
                                               weight, CUDA_R_16F, m, cols * m, k0 > 0 ? &one : &zero, og, CUDA_R_32F,
                                               dim, cols * dim, kv_heads, CUBLAS_COMPUTE_32F,
                                               CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                    "attention values");
            if (prefill_mark_fine) prefill_mark_fine(stream, kPpAttnValue);
        }
        attn_scale_kernel<<<dim3(grid(dim, 256), c * n_head), 256, 0, cs(stream)>>>(og, sum, dim, true);
        attn_ungroup_kernel<<<dim3(grid(dim, 256), c * n_head), 256, 0, cs(stream)>>>(
            og, out + size_t(first) * n_head * dim, dim, n_head, kv_heads, c);
        if (prefill_mark_fine) prefill_mark_fine(stream, kPpAttnFinish);
    }
}

void attend_batch_q8(void* blas, const float* q, const uint8_t* kcache, const uint8_t* vcache, int64_t pos, int rows,
                     float* score, size_t score_floats, float* work, float* out, int dim, int n_head, int kv_heads,
                     float scale, void* stream, uint16_t* kstage, uint16_t* vstage, int stage_rows, int window,
                     int ring, bool whole) {
    check_q8(dim, kv_heads, "attend_batch_q8");
    if (!kstage || !vstage || stage_rows < 1) throw std::invalid_argument("attend_batch_q8: no stage for the blocks");
    attend_batch_kv(blas, q, kcache, true, vcache, true, pos, rows, score, score_floats, work, out, dim, n_head,
                    kv_heads, scale, stream, kstage, vstage, stage_rows, window, ring, whole);
}

void (*prefill_mark)(void* stream, int part) = nullptr;
void (*prefill_mark_fine)(void* stream, int part) = nullptr;

}  // namespace strata::poly
