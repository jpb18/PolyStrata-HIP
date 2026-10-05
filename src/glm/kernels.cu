// src/glm/kernels.cu - see the header.  The reference for every formula is llama.cpp's glm5-next graph and the
// ggml ops it builds.  Where a kernel takes several tokens, blockIdx.y is the token unless it says otherwise.
#include "strata/glm/kernels.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace strata::glm {
namespace {

constexpr int WARP = 32;
constexpr int HC = 4;   // hyper-connection streams; the model's hc_mult, checked at load
constexpr int KEY_BLOCK = 16;   // cache rows one block of dsa_keys_kernel copies

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
                            int n_out, int rows_per_x) {
    __shared__ float part[8];
    const int o = blockIdx.x;
    const size_t tok = blockIdx.y;
    const W* wr = w + size_t(o) * n_in;
    const float* xr = x + (tok * size_t(n_out / rows_per_x) + size_t(o / rows_per_x)) * n_in;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n_in; i += blockDim.x) acc += ldw(wr, i) * xr[i];
    acc = block_sum(acc, part);
    if (threadIdx.x == 0) y[tok * n_out + o] = acc;
}

template<typename W>
void gemv(const W* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, int rows, void* stream) {
    const int threads = n_in >= 1024 ? 256 : n_in >= 128 ? 128 : 32;
    gemv_kernel<W><<<dim3(n_out, rows), threads, 0, (cudaStream_t) stream>>>(w, x, y, n_in, n_out, rows_per_x);
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
    __shared__ float part[8];
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
    float acc = r[i];
    for (int s = 1; s < HC; ++s) acc += r[size_t(s) * n_embd + i];
    y[i] = acc * (1.0f / HC);
}

// d_conv is 4: three past inputs per channel, then the new one.  A thread is a channel and walks the tokens.
__global__ void kda_conv_kernel(const float* __restrict__ proj, float* __restrict__ state,
                                const float* __restrict__ wq, const float* __restrict__ wk,
                                const float* __restrict__ wv, float* __restrict__ out, int d_inner, int rows) {
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
    }
    st[0] = s0;
    st[1] = s1;
    st[2] = s2;
}

// One block per head, one thread per value dimension j; the block walks the tokens.  s[i + hd * j] holds key
// dimension i of column j.
__global__ void kda_step_kernel(const float* __restrict__ qkv, const float* __restrict__ g1,
                                const float* __restrict__ a, const float* __restrict__ beta, float* __restrict__ s,
                                const float* __restrict__ g2, const float* __restrict__ o_norm,
                                float* __restrict__ out, int hd, int n_head, int rows, float lower_bound,
                                float norm_eps) {
    __shared__ float part[8];
    __shared__ float qn[256], kn[256], decay[256];
    const int h = blockIdx.x, j = threadIdx.x;
    const size_t d_inner = size_t(hd) * n_head, plane = d_inner * rows;
    float* col = s + (size_t(h) * hd + j) * hd;
    for (int row = 0; row < rows; ++row) {
        const size_t at = size_t(row) * d_inner + size_t(h) * hd + j;
        const float q = qkv[at], k = qkv[plane + at], v = qkv[2 * plane + at];
        const float qs = block_sum(q * q, part);
        const float ks = block_sum(k * k, part);
        qn[j] = q / sqrtf(qs + 1e-6f) * (1.0f / sqrtf(float(hd)));
        kn[j] = k / sqrtf(ks + 1e-6f);
        // the decay gate of key dimension j: exp(lower_bound * sigmoid(-(g1 * a)))
        decay[j] = expf(lower_bound / (1.0f + expf(g1[at] * a[h])));
        __syncthreads();
        float sk = 0.0f;
        for (int i = 0; i < hd; ++i) sk += col[i] * decay[i] * kn[i];
        const float d = (v - sk) * sigmoid(beta[size_t(row) * n_head + h]);
        float o = 0.0f;
        for (int i = 0; i < hd; ++i) {
            const float n = col[i] * decay[i] + kn[i] * d;
            col[i] = n;
            o += n * qn[i];
        }
        // the sum's last barrier also ends this token's reads of qn, kn and decay
        const float os = block_sum(o * o, part);
        out[at] = o / sqrtf(os / float(hd) + norm_eps) * o_norm[j] * sigmoid(g2[at]);
    }
}

__global__ void store_f16_kernel(const float* __restrict__ x, half* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(x[i]);
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

__global__ void swiglu_clamp_rows_kernel(const float* __restrict__ gate_up, float* __restrict__ out, int n,
                                         float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* row = gate_up + size_t(blockIdx.y) * 2 * n;
    const float g = fminf(row[i], limit);
    const float u = fminf(fmaxf(row[n + i], -limit), limit);
    out[size_t(blockIdx.y) * n + i] = g / (1.0f + expf(-g)) * u;
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

inline cudaStream_t cs(void* stream) { return (cudaStream_t) stream; }
inline int grid(size_t n, int block) { return int((n + block - 1) / block); }

void blas_ck(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(what) + ": cuBLAS status " + std::to_string((int) s));
}

}  // namespace

void* blas_create(void* stream) {
    cublasHandle_t h = nullptr;
    blas_ck(cublasCreate(&h), "cublasCreate");
    if (const cublasStatus_t s = cublasSetStream(h, cs(stream)); s != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(h);
        blas_ck(s, "cublasSetStream");
    }
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

void rms_norm(const float* x, const float* w, float* y, int n, int rows, float eps, void* stream) {
    rms_norm_kernel<<<rows, n >= 1024 ? 256 : 128, 0, cs(stream)>>>(x, w, y, n, eps);
}

void layer_norm(const float* x, const float* w, const float* b, float* y, int n, int rows, float eps, void* stream) {
    layer_norm_kernel<<<rows, 128, 0, cs(stream)>>>(x, w, b, y, n, eps);
}

void hc_pre(const float* r, const float* fn, const float* scale, const float* base, float* mix, float* hcw,
            float* cur, int n_embd, int rows, float norm_eps, float hc_eps, int sinkhorn_iters, void* stream) {
    const int n_mix = HC * (2 + HC);
    hc_mix_kernel<<<dim3(n_mix + 1, rows), 256, 0, cs(stream)>>>(r, fn, mix, HC * n_embd, n_mix);
    hc_read_kernel<<<dim3(grid(n_embd, 256), rows), 256, 0, cs(stream)>>>(r, mix, scale, base, hcw, cur, n_embd,
                                                                         norm_eps, hc_eps, sinkhorn_iters);
}

void hc_post(const float* x, const float* hcw, float* r, int n_embd, int rows, void* stream) {
    hc_post_kernel<<<dim3(grid(n_embd, 256), rows), 256, 0, cs(stream)>>>(x, hcw, r, n_embd);
}

void hc_mean(const float* r, float* y, int n_embd, void* stream) {
    hc_mean_kernel<<<grid(n_embd, 256), 256, 0, cs(stream)>>>(r, y, n_embd);
}

void kda_conv(const float* proj, float* state, const float* wq, const float* wk, const float* wv, float* out,
              int d_inner, int rows, void* stream) {
    kda_conv_kernel<<<grid(3 * d_inner, 256), 256, 0, cs(stream)>>>(proj, state, wq, wk, wv, out, d_inner, rows);
}

void kda_step(const float* qkv, const float* g1, const float* a, const float* beta, float* s, const float* g2,
              const float* o_norm, float* out, int head_dim, int n_head, int rows, float gate_lower_bound,
              float norm_eps, void* stream) {
    kda_step_kernel<<<n_head, head_dim, 0, cs(stream)>>>(qkv, g1, a, beta, s, g2, o_norm, out, head_dim, n_head,
                                                        rows, gate_lower_bound, norm_eps);
}

void store_f16(const float* x, uint16_t* cache, int n, int64_t pos, int rows, void* stream) {
    const size_t all = size_t(n) * rows;
    store_f16_kernel<<<grid(all, 256), 256, 0, cs(stream)>>>(x, (half*) cache + size_t(pos) * n, int(all));
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
        blas_ck(cublasSgemmStridedBatched(h, CUBLAS_OP_T, CUBLAS_OP_N, n_keys, n_head, rank, &scale, keys, rank,
                                          key_n, q + size_t(first) * head, rank, head, &zero, score, n_keys,
                                          score_n, c),
                "attention scores");
        dsa_softmax_kernel<<<dim3(n_head, c), 256, 0, cs(stream)>>>(score, n_keys, top_rows, kpool, (long long) pos,
                                                                   first, dense_rows);
        // out[head] = sum over the keys of score[head][key] * keys[key]
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

void swiglu_clamp(const float* gate, const float* up, float* out, int n, float limit, void* stream) {
    swiglu_clamp_kernel<<<grid(n, 256), 256, 0, cs(stream)>>>(gate, up, out, n, limit);
}

void axpy(const float* x, float a, float* acc, int n, bool add, void* stream) {
    axpy_kernel<<<grid(n, 256), 256, 0, cs(stream)>>>(x, a, acc, n, add);
}

void swiglu_clamp_rows(const float* gate_up, float* out, int n, int rows, float limit, void* stream) {
    swiglu_clamp_rows_kernel<<<dim3(grid(n, 256), rows), 256, 0, cs(stream)>>>(gate_up, out, n, limit);
}

void moe_sum(const float* x, const float* w, float* acc, int n, int n_used, int rows, void* stream) {
    moe_sum_kernel<<<dim3(grid(n, 256), rows), 256, 0, cs(stream)>>>(x, w, acc, n, n_used);
}

void add_bias(const float* b, float* acc, int n, int rows, void* stream) {
    add_bias_kernel<<<dim3(grid(n, 256), rows), 256, 0, cs(stream)>>>(b, acc, n);
}

}  // namespace strata::glm
