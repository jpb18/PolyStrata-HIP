// src/glm/kernels.cu - see the header.  One token per call; the reference for every formula is llama.cpp's
// glm5-next graph and the ggml ops it builds.
#include "strata/glm/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

namespace strata::glm {
namespace {

constexpr int WARP = 32;
constexpr int HC = 4;   // hyper-connection streams; the model's hc_mult, checked at load

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffff, x, o, WARP);
    return x;
}

// One value per thread summed over the block.  blockDim.x is a multiple of 32 and at most 256.
__device__ __forceinline__ float block_sum(float x, float* part) {
    x = warp_sum(x);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = x;
    __syncthreads();
    float s = 0.0f;
    for (int k = 0; k < int(blockDim.x) / WARP; ++k) s += part[k];
    __syncthreads();   // part is free again
    return s;
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
                            int rows_per_x) {
    __shared__ float part[8];
    const int o = blockIdx.x;
    const W* wr = w + size_t(o) * n_in;
    const float* xr = x + size_t(o / rows_per_x) * n_in;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n_in; i += blockDim.x) acc += ldw(wr, i) * xr[i];
    acc = block_sum(acc, part);
    if (threadIdx.x == 0) y[o] = acc;
}

template<typename W>
void gemv(const W* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, void* stream) {
    const int threads = n_in >= 1024 ? 256 : n_in >= 128 ? 128 : 32;
    gemv_kernel<W><<<n_out, threads, 0, (cudaStream_t) stream>>>(w, x, y, n_in, rows_per_x);
}

__global__ void rms_norm_kernel(const float* __restrict__ x, const float* __restrict__ w, float* y, int n,
                                float eps) {
    __shared__ float part[8];
    const float* xr = x + size_t(blockIdx.x) * n;
    float* yr = y + size_t(blockIdx.x) * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, part);
    const float scale = 1.0f / sqrtf(ss / float(n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = w ? xr[i] * scale * w[i] : xr[i] * scale;
}

__global__ void layer_norm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                  const float* __restrict__ b, float* y, int n, float eps) {
    __shared__ float part[8];
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += x[i];
    const float mean = block_sum(s, part) / float(n);
    float v = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float d = x[i] - mean;
        v += d * d;
    }
    const float scale = 1.0f / sqrtf(block_sum(v, part) / float(n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = (x[i] - mean) * scale * w[i] + b[i];
}

// mix[o] = fn row o . r for o < n_mix; mix[n_mix] = sum of r^2 (the flat RMS norm's input)
__global__ void hc_mix_kernel(const float* __restrict__ r, const float* __restrict__ fn, float* __restrict__ mix,
                              int n, int n_mix) {
    __shared__ float part[8];
    const int o = blockIdx.x;
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
    float acc = r[i];
    for (int s = 1; s < HC; ++s) acc += r[size_t(s) * n_embd + i];
    y[i] = acc * (1.0f / HC);
}

// d_conv is 4: three past inputs per channel, then the new one
__global__ void kda_conv_kernel(const float* __restrict__ proj, float* __restrict__ state,
                                const float* __restrict__ wq, const float* __restrict__ wk,
                                const float* __restrict__ wv, float* __restrict__ out, int d_inner) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= 3 * d_inner) return;
    const int m = t / d_inner, c = t % d_inner;
    const float* w = (m == 0 ? wq : m == 1 ? wk : wv) + 4 * size_t(c);
    float* st = state + size_t(t) * 3;
    const float x = proj[t];
    const float y = st[0] * w[0] + st[1] * w[1] + st[2] * w[2] + x * w[3];
    st[0] = st[1];
    st[1] = st[2];
    st[2] = x;
    out[t] = y * sigmoid(y);
}

// One block per head, one thread per value dimension j.  s[i + hd * j] holds key dimension i of column j.
__global__ void kda_step_kernel(const float* __restrict__ qkv, const float* __restrict__ g1,
                                const float* __restrict__ a, const float* __restrict__ beta, float* __restrict__ s,
                                const float* __restrict__ g2, const float* __restrict__ o_norm,
                                float* __restrict__ out, int hd, int n_head, float lower_bound, float norm_eps) {
    __shared__ float part[8];
    __shared__ float qn[256], kn[256], decay[256];
    const int h = blockIdx.x, j = threadIdx.x;
    const size_t at = size_t(h) * hd + j, d_inner = size_t(hd) * n_head;
    const float q = qkv[at], k = qkv[d_inner + at], v = qkv[2 * d_inner + at];
    const float qs = block_sum(q * q, part);
    const float ks = block_sum(k * k, part);
    qn[j] = q / sqrtf(qs + 1e-6f) * (1.0f / sqrtf(float(hd)));
    kn[j] = k / sqrtf(ks + 1e-6f);
    // the decay gate of key dimension j: exp(lower_bound * sigmoid(-(g1 * a)))
    decay[j] = expf(lower_bound / (1.0f + expf(g1[at] * a[h])));
    __syncthreads();
    float* col = s + at * hd;
    float sk = 0.0f;
    for (int i = 0; i < hd; ++i) sk += col[i] * decay[i] * kn[i];
    const float d = (v - sk) * sigmoid(beta[h]);
    float o = 0.0f;
    for (int i = 0; i < hd; ++i) {
        const float n = col[i] * decay[i] + kn[i] * d;
        col[i] = n;
        o += n * qn[i];
    }
    const float os = block_sum(o * o, part);
    out[at] = o / sqrtf(os / float(hd) + norm_eps) * o_norm[j] * sigmoid(g2[at]);
}

__global__ void store_f16_kernel(const float* __restrict__ x, half* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(x[i]);
}

// One block per head.  Shared: n_sel scores, then 8 partials.
__global__ void dsa_attend_kernel(const float* __restrict__ q, const half* __restrict__ cache,
                                  const int32_t* __restrict__ sel, int n_sel, float* __restrict__ out, int rank,
                                  float scale) {
    extern __shared__ float sm[];
    float* sc = sm;
    float* part = sm + n_sel;
    const float* qh = q + size_t(blockIdx.x) * rank;
    float mx = -INFINITY;
    for (int t = threadIdx.x; t < n_sel; t += blockDim.x) {
        const half* row = cache + size_t(sel ? sel[t] : t) * rank;
        float acc = 0.0f;
        for (int r = 0; r < rank; ++r) acc += __half2float(row[r]) * qh[r];
        sc[t] = acc * scale;
        mx = fmaxf(mx, sc[t]);
    }
    mx = block_max(mx, part);
    float sum = 0.0f;
    for (int t = threadIdx.x; t < n_sel; t += blockDim.x) {
        sc[t] = expf(sc[t] - mx);
        sum += sc[t];
    }
    const float inv = 1.0f / block_sum(sum, part);
    float* oh = out + size_t(blockIdx.x) * rank;
    for (int r = threadIdx.x; r < rank; r += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < n_sel; ++t) acc += sc[t] * __half2float(cache[size_t(sel ? sel[t] : t) * rank + r]);
        oh[r] = acc * inv;
    }
}

__global__ void idx_store_kernel(const float* __restrict__ key, const float* __restrict__ gate,
                                 float* __restrict__ row, int dim) {
    const int d = threadIdx.x;
    if (d >= dim) return;
    row[d] = key[d];
    row[dim + d] = gate[d];
}

__global__ void idx_pool_kernel(float* cache, const float* __restrict__ ape, int dim, int kpool, long long last) {
    const int d = threadIdx.x;
    if (d >= dim) return;
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

// One block per pool, 32 threads over the indexer heads.
__global__ void idx_score_kernel(const float* __restrict__ q, const float* __restrict__ w,
                                 const float* __restrict__ cache, float* __restrict__ score, int dim, int n_head,
                                 int kpool, float wscale) {
    const float* key = cache + (size_t(blockIdx.x) * kpool + kpool - 1) * 3 * dim + 2 * dim;
    float acc = 0.0f;
    for (int h = threadIdx.x; h < n_head; h += WARP) {
        const float* qh = q + size_t(h) * dim;
        float dot = 0.0f;
        for (int d = 0; d < dim; ++d) dot += qh[d] * key[d];
        acc += fmaxf(dot, 0.0f) * (w[h] * wscale);
    }
    acc = warp_sum(acc);
    if (threadIdx.x == 0) score[blockIdx.x] = acc;
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

inline cudaStream_t cs(void* stream) { return (cudaStream_t) stream; }
inline int grid(int n, int block) { return (n + block - 1) / block; }

}  // namespace

void gemv_f32(const float* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, void* stream) {
    gemv(w, x, y, n_in, n_out, rows_per_x, stream);
}

void gemv_bf16(const uint16_t* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, void* stream) {
    gemv(w, x, y, n_in, n_out, rows_per_x, stream);
}

void rms_norm(const float* x, const float* w, float* y, int n, int rows, float eps, void* stream) {
    rms_norm_kernel<<<rows, n >= 1024 ? 256 : 128, 0, cs(stream)>>>(x, w, y, n, eps);
}

void layer_norm(const float* x, const float* w, const float* b, float* y, int n, float eps, void* stream) {
    layer_norm_kernel<<<1, 128, 0, cs(stream)>>>(x, w, b, y, n, eps);
}

void hc_pre(const float* r, const float* fn, const float* scale, const float* base, float* mix, float* hcw,
            float* cur, int n_embd, float norm_eps, float hc_eps, int sinkhorn_iters, void* stream) {
    const int n_mix = HC * (2 + HC);
    hc_mix_kernel<<<n_mix + 1, 256, 0, cs(stream)>>>(r, fn, mix, HC * n_embd, n_mix);
    hc_read_kernel<<<grid(n_embd, 256), 256, 0, cs(stream)>>>(r, mix, scale, base, hcw, cur, n_embd, norm_eps,
                                                             hc_eps, sinkhorn_iters);
}

void hc_post(const float* x, const float* hcw, float* r, int n_embd, void* stream) {
    hc_post_kernel<<<grid(n_embd, 256), 256, 0, cs(stream)>>>(x, hcw, r, n_embd);
}

void hc_mean(const float* r, float* y, int n_embd, void* stream) {
    hc_mean_kernel<<<grid(n_embd, 256), 256, 0, cs(stream)>>>(r, y, n_embd);
}

void kda_conv(const float* proj, float* state, const float* wq, const float* wk, const float* wv, float* out,
              int d_inner, void* stream) {
    kda_conv_kernel<<<grid(3 * d_inner, 256), 256, 0, cs(stream)>>>(proj, state, wq, wk, wv, out, d_inner);
}

void kda_step(const float* qkv, const float* g1, const float* a, const float* beta, float* s, const float* g2,
              const float* o_norm, float* out, int head_dim, int n_head, float gate_lower_bound, float norm_eps,
              void* stream) {
    kda_step_kernel<<<n_head, head_dim, 0, cs(stream)>>>(qkv, g1, a, beta, s, g2, o_norm, out, head_dim, n_head,
                                                        gate_lower_bound, norm_eps);
}

void store_f16(const float* x, uint16_t* cache, int n, int64_t pos, void* stream) {
    store_f16_kernel<<<grid(n, 256), 256, 0, cs(stream)>>>(x, (half*) cache + size_t(pos) * n, n);
}

void dsa_attend(const float* q, const uint16_t* cache, const int32_t* sel, int n_sel, float* out, int rank,
                int n_head, float scale, void* stream) {
    dsa_attend_kernel<<<n_head, 256, (size_t(n_sel) + 8) * sizeof(float), cs(stream)>>>(
        q, (const half*) cache, sel, n_sel, out, rank, scale);
}

void idx_store(const float* key, const float* gate, float* cache, int dim, int64_t pos, void* stream) {
    idx_store_kernel<<<1, 256, 0, cs(stream)>>>(key, gate, cache + size_t(pos) * 3 * dim, dim);
}

void idx_pool(float* cache, const float* ape, int dim, int kpool, int64_t last, void* stream) {
    idx_pool_kernel<<<1, 256, 0, cs(stream)>>>(cache, ape, dim, kpool, (long long) last);
}

void idx_score(const float* q, const float* w, const float* cache, float* score, int dim, int n_head, int kpool,
               int n_pool, void* stream) {
    idx_score_kernel<<<n_pool, WARP, 0, cs(stream)>>>(q, w, cache, score, dim, n_head, kpool,
                                                      1.0f / std::sqrt(float(dim) * float(n_head)));
}

void swiglu_clamp(const float* gate, const float* up, float* out, int n, float limit, void* stream) {
    swiglu_clamp_kernel<<<grid(n, 256), 256, 0, cs(stream)>>>(gate, up, out, n, limit);
}

void axpy(const float* x, float a, float* acc, int n, bool add, void* stream) {
    axpy_kernel<<<grid(n, 256), 256, 0, cs(stream)>>>(x, a, acc, n, add);
}

}  // namespace strata::glm
