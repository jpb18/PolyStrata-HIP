// include/strata/glm/kernels.hpp - the float kernels of the forward passes.
//
// Every call enqueues on the given CUDA stream and returns; nothing here but blas_create allocates or waits.
// Pointers are device pointers.  The math follows llama.cpp's graphs (src/models/glm5-next.cpp and qwen35moe.cpp)
// and the ggml ops they build, which are the reference the engine is checked against.
//
// A call works on `rows` tokens at consecutive positions.  A buffer of per-token values holds them row after row,
// each row as long as the one-token case describes.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::glm {

/// y[o] = sum_i w[o * n_in + i] * x[(o / rows_per_x) * n_in + i], o in [0, n_out).  With rows_per_x == n_out it is
/// a plain matrix-vector product; a smaller value gives each group of rows its own input (the per-head MLA
/// projections), so a token's x is n_in * n_out / rows_per_x long.
void gemv_f32(const float* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, int rows, void* stream);
void gemv_bf16(const uint16_t* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, int rows,
               void* stream);

/// RMS norm over each of `rows` groups of n values: y = x / sqrt(mean(x^2) + eps) * w.  `w` (n values, shared by the
/// rows) may be null.  A row of y starts y_stride values after the one before it (0: n), so two calls can fill
/// the two halves of rows twice as long.
void rms_norm(const float* x, const float* w, float* y, int n, int rows, float eps, void* stream, int y_stride = 0);
/// y = (x - mean) / sqrt(var + eps) * w + b.
void layer_norm(const float* x, const float* w, const float* b, float* y, int n, int rows, float eps, void* stream);

/// The hyper-connection read.  r is a token's hc * n_embd residual streams.  Writes cur (n_embd) and hcw: post[hc]
/// then comb[hc * hc] (comb[dst + hc * src]), which hc_post reads.  `mix` is hc*(2+hc)+1 floats of scratch a token.
void hc_pre(const float* r, const float* fn, const float* scale, const float* base, float* mix, float* hcw,
            float* cur, int n_embd, int rows, float norm_eps, float hc_eps, int sinkhorn_iters, void* stream);
/// The hyper-connection write, in place on r: r[dst] = x * post[dst] + sum_src r[src] * comb[dst, src].
void hc_post(const float* x, const float* hcw, float* r, int n_embd, int rows, void* stream);
/// y = mean over the hc streams, token by token.
void hc_mean(const float* r, float* y, int n_embd, int rows, void* stream);

/// The three causal convolutions of a KDA layer and their SiLU.  proj: the q projections of every row, then the k,
/// then the v (3 * rows * d_inner); out has the same layout.  state: 3 * d_inner * (d_conv - 1) past inputs, oldest
/// first per channel, advanced here by `rows` tokens.  w*: (d_conv, d_inner).
/// With snap, the state as it is after each of the first n_snap tokens is also written there: the one after
/// token i starts i * stride floats in.  That is what a window of guessed tokens goes back to (kda_step too).
void kda_conv(const float* proj, float* state, const float* wq, const float* wk, const float* wv, float* out,
              int d_inner, int rows, void* stream, float* snap = nullptr, size_t stride = 0, int n_snap = 0);
/// The delta-rule step of every head and the gated output, token after token.  qkv: the conv outputs.  g1:
/// f_b(f_a(x)) + dt bias (d_inner); it gates the state's decay per key dimension, as in ggml's gated_delta_net op.
/// a: per head.  beta: per head, before the sigmoid.  s: n_head * head_dim * head_dim, updated.  g2: g_b(g_a(x))
/// (d_inner).  out = rms_norm(o) * o_norm * sigmoid(g2), d_inner values.
void kda_step(const float* qkv, const float* g1, const float* a, const float* beta, float* s, const float* g2,
              const float* o_norm, float* out, int head_dim, int n_head, int rows, float gate_lower_bound,
              float norm_eps, void* stream, float* snap = nullptr, size_t stride = 0, int n_snap = 0);

/// Stores the tokens' MLA latents (n values each) as fp16 from row `pos` of the layer's cache.
void store_f16(const float* x, uint16_t* cache, int n, int64_t pos, int rows, void* stream);
/// Attention over cached latents for the tokens from position `pos`.  q: n_head * rank absorbed queries a token.
/// The first dense_rows tokens attend to every cache row up to their own.  The others each have a list in sel
/// (sel_stride apart): top_rows cache rows, then the rows of the pool their own position is still filling.
/// out: n_head * rank a token, the attention-weighted latents per head.
/// It takes `chunk` tokens at a time: their keys are copied to `keys` as floats (rank a key), and two cuBLAS
/// products per chunk do the rest, with `score` (n_head a key) between them.  Both buffers hold a chunk of tokens
/// with the most keys a token has: top_rows + kpool - 1, or the context if that is less.  Throws
/// std::runtime_error.
void dsa_attend(void* blas, const float* q, const uint16_t* cache, const int32_t* sel, int sel_stride, int top_rows,
                int kpool, int64_t pos, int rows, int dense_rows, float* keys, float* score, int chunk, float* out,
                int rank, int n_head, float scale, void* stream);
/// A cuBLAS handle that enqueues on `stream`, for dsa_attend.  Throws std::runtime_error.
void* blas_create(void* stream);
void blas_destroy(void* blas);

/// Indexer cache row: key (dim) | gate (dim) | pooled (dim), fp32.  Stores the tokens' keys and gates from row
/// `pos`.
void idx_store(const float* key, const float* gate, float* cache, int dim, int64_t pos, int rows, void* stream);
/// For each of the rows from `pos` that ends a pool: pools the kpool rows ending there into the pooled part of
/// that row.  ape: (dim, kpool).
void idx_pool(float* cache, const float* ape, int dim, int kpool, int64_t pos, int rows, void* stream);
/// For the tokens from position `pos`: score[p] = sum_h relu(q[h] . pooled(p)) * w[h] for the pools p that are
/// complete at the token's position; pool p's key is in cache row kpool * p + kpool - 1.  A token's scores start
/// score_stride apart.  w is the raw indexer_proj output; the 1/sqrt(dim * n_head) factor is applied here.
void idx_score(const float* q, const float* w, const float* cache, float* score, int score_stride, int dim,
               int n_head, int kpool, int64_t pos, int rows, void* stream);

/// out = silu(min(gate, limit)) * clamp(up, -limit, limit).
void swiglu_clamp(const float* gate, const float* up, float* out, int n, float limit, void* stream);
/// swiglu_clamp for rows that each hold n gate values and then n up values; out has n values a row.
void swiglu_clamp_rows(const float* gate_up, float* out, int n, int rows, float limit, void* stream);
/// acc[t] += sum over k < n_used of w[t * n_used + k] * x[t * n_used + k], for rows of n values: a token's
/// weighted experts added to its row.
void moe_sum(const float* x, const float* w, float* acc, int n, int n_used, int rows, void* stream);
/// acc = x * a, or acc += x * a when `add`.
void axpy(const float* x, float a, float* acc, int n, bool add, void* stream);
/// acc[row] += b for each of the rows of n values.
void add_bias(const float* b, float* acc, int n, int rows, void* stream);

// ---- attention over a key and value cache, and the gated delta net with one decay a head (qwen35moe)

/// For `rows` tokens of `heads` heads each: a head's RMS norm with the weights w (dim values, the same for every
/// head), then ggml's rotary embedding on its first n_rot values in the NeoX pairing: value i goes with value
/// i + n_rot / 2, and the pair turns by position * theta_scale^i.  A head's input starts x_stride floats after
/// the one before it; y has dim values a head.
void head_norm_rope(const float* x, int x_stride, const float* w, float* y, int dim, int heads, int n_rot,
                    float theta_scale, int64_t pos, int rows, float eps, void* stream);
/// Causal attention for the tokens from position `pos`: token t attends the cache rows 0 to pos + t.  q: n_head *
/// dim a token.  kcache and vcache: fp16, kv_heads * dim a row; the heads read a row's keys and values in turn,
/// n_head / kv_heads heads each.  out: n_head * dim a token.  It takes `chunk` tokens at a time, and score is
/// scratch of chunk * n_head * (pos + rows) floats.  A token comes out the same whatever tokens the call has
/// beside it.
void attend(const float* q, const uint16_t* kcache, const uint16_t* vcache, int64_t pos, int rows, float* score,
            int chunk, float* out, int dim, int n_head, int kv_heads, float scale, void* stream);
/// acc[r][i] *= sigmoid(g[r][i]) for `rows` groups of n values; group r's gates start r * g_stride floats into g.
/// With one_gate a group has one gate for all its values.
void mul_sigmoid(float* acc, const float* g, int n, int rows, int g_stride, bool one_gate, void* stream);
/// A causal convolution of four taps over each of `channels` channels, and its SiLU.  x and out: `channels` values
/// a token.  state: 3 past inputs a channel, oldest first, advanced here by `rows` tokens.  w: (4, channels).
/// snap, stride and n_snap as in kda_conv.
void conv4_silu(const float* x, float* state, const float* w, float* out, int channels, int rows, void* stream,
                float* snap = nullptr, size_t stride = 0, int n_snap = 0);
/// The delta-rule step of every value head and its gated output, token after token: ggml's gated_delta_net op
/// with one decay a head.  qkv: a token's conv outputs, the k_heads key heads' q, their k, then the n_head value
/// heads' v, dim values each; value head h reads key head h % k_heads.  alpha and beta: n_head a token, as their
/// projections give them.  dt, a: n_head.  The state decays by exp(softplus(alpha + dt) * a) and takes
/// sigmoid(beta) of the error.  s: n_head * dim * dim, updated.  z: n_head * dim a token.
/// out = rms_norm(o) * o_norm * silu(z), n_head * dim values a token.
void gdn_step(const float* qkv, const float* alpha, const float* beta, const float* dt, const float* a, float* s,
              const float* z, const float* o_norm, float* out, int dim, int n_head, int k_heads, int rows,
              float norm_eps, void* stream, float* snap = nullptr, size_t stride = 0, int n_snap = 0);

}  // namespace strata::glm
