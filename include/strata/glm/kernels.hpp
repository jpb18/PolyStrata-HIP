// include/strata/glm/kernels.hpp - the float kernels of the GLM 5.3 Flash decode step (one token).
//
// Every call enqueues on the given CUDA stream and returns; nothing here allocates or waits.  Pointers are device
// pointers.  The math follows llama.cpp's glm5-next graph (src/models/glm5-next.cpp), which is the reference the
// engine is checked against.
#pragma once

#include <cstdint>

namespace strata::glm {

/// y[o] = sum_i w[o * n_in + i] * x[(o / rows_per_x) * n_in + i], o in [0, n_out).  With rows_per_x == n_out it is
/// a plain matrix-vector product; a smaller value gives each group of rows its own input (the per-head MLA
/// projections).
void gemv_f32(const float* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, void* stream);
void gemv_bf16(const uint16_t* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, void* stream);

/// RMS norm over each of `rows` groups of n values: y = x / sqrt(mean(x^2) + eps) * w.  `w` (n values, shared by the
/// rows) may be null.
void rms_norm(const float* x, const float* w, float* y, int n, int rows, float eps, void* stream);
/// y = (x - mean) / sqrt(var + eps) * w + b.
void layer_norm(const float* x, const float* w, const float* b, float* y, int n, float eps, void* stream);

/// The hyper-connection read.  r is the hc * n_embd residual streams.  Writes cur (n_embd) and hcw: post[hc] then
/// comb[hc * hc] (comb[dst + hc * src]), which hc_post reads.  `mix` is hc*(2+hc)+1 floats of scratch.
void hc_pre(const float* r, const float* fn, const float* scale, const float* base, float* mix, float* hcw,
            float* cur, int n_embd, float norm_eps, float hc_eps, int sinkhorn_iters, void* stream);
/// The hyper-connection write, in place on r: r[dst] = x * post[dst] + sum_src r[src] * comb[dst, src].
void hc_post(const float* x, const float* hcw, float* r, int n_embd, void* stream);
/// y = mean over the hc streams.
void hc_mean(const float* r, float* y, int n_embd, void* stream);

/// The three causal convolutions of a KDA layer and their SiLU.  proj: q|k|v projections (3 * d_inner).  state:
/// 3 * d_inner * (d_conv - 1) past inputs, oldest first per channel, advanced here.  w*: (d_conv, d_inner).
void kda_conv(const float* proj, float* state, const float* wq, const float* wk, const float* wv, float* out,
              int d_inner, void* stream);
/// The delta-rule step of every head and the gated output.  qkv: the conv outputs.  g1: f_b(f_a(x)) + dt bias
/// (d_inner); it gates the state's decay per key dimension, as in ggml's gated_delta_net op.  a: per head.
/// beta: per head, before the sigmoid.  s: n_head * head_dim * head_dim, updated.  g2: g_b(g_a(x)) (d_inner).
/// out = rms_norm(o) * o_norm * sigmoid(g2), d_inner values.
void kda_step(const float* qkv, const float* g1, const float* a, const float* beta, float* s, const float* g2,
              const float* o_norm, float* out, int head_dim, int n_head, float gate_lower_bound, float norm_eps,
              void* stream);

/// Stores a token's MLA latent (n values) as fp16 at row `pos` of the layer's cache.
void store_f16(const float* x, uint16_t* cache, int n, int64_t pos, void* stream);
/// Sparse attention over cached latents.  q: n_head * rank absorbed queries.  sel: n_sel cache rows, or null for
/// rows 0..n_sel-1.  out: n_head * rank, the attention-weighted latents per head.
void dsa_attend(const float* q, const uint16_t* cache, const int32_t* sel, int n_sel, float* out, int rank,
                int n_head, float scale, void* stream);

/// Indexer cache row: key (dim) | gate (dim) | pooled (dim), fp32.  Stores key and gate at row `pos`.
void idx_store(const float* key, const float* gate, float* cache, int dim, int64_t pos, void* stream);
/// Pools the kpool rows ending at `last` into the pooled part of row `last`.  ape: (dim, kpool).
void idx_pool(float* cache, const float* ape, int dim, int kpool, int64_t last, void* stream);
/// score[p] = sum_h relu(q[h] . pooled(p)) * w[h] for pools p in [0, n_pool); pool p's key is in cache row
/// kpool * p + kpool - 1.  w is the raw indexer_proj output; the 1/sqrt(dim * n_head) factor is applied here.
void idx_score(const float* q, const float* w, const float* cache, float* score, int dim, int n_head, int kpool,
               int n_pool, void* stream);

/// out = silu(min(gate, limit)) * clamp(up, -limit, limit).
void swiglu_clamp(const float* gate, const float* up, float* out, int n, float limit, void* stream);
/// acc = x * a, or acc += x * a when `add`.
void axpy(const float* x, float a, float* acc, int n, bool add, void* stream);

}  // namespace strata::glm
