// include/strata/poly/kernels.hpp - the float kernels of the forward passes.
//
// Every call enqueues on the given CUDA stream and returns; nothing here but blas_create allocates or waits.
// Pointers are device pointers.  The math follows llama.cpp's graphs (src/models/glm5-next.cpp, qwen35moe.cpp and
// gemma4.cpp) and the ggml ops they build, which are the reference the engine is checked against.
//
// A call works on `rows` tokens at consecutive positions.  A buffer of per-token values holds them row after row,
// each row as long as the one-token case describes.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::poly {

/// y[o] = sum_i w[o * n_in + i] * x[(o / rows_per_x) * n_in + i], o in [0, n_out).  With rows_per_x == n_out it is
/// a plain matrix-vector product; a smaller value gives each group of rows its own input (the per-head MLA
/// projections), so a token's x is n_in * n_out / rows_per_x long.
void gemv_f32(const float* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, int rows, void* stream);
void gemv_bf16(const uint16_t* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, int rows,
               void* stream);
/// The same for the many rows of a prompt batch, as a cuBLAS product on the stream of `blas` (blas_create), one
/// for each group of rows_per_x outputs.  It sums in another order than gemv_f32, so a row does not come out
/// to the last bit as it does there.  Throws std::runtime_error.
void gemm_f32(void* blas, const float* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, int rows);
/// n BF16 values as floats, for gemm_f32.
void bf16_to_f32(const uint16_t* w, float* out, size_t n, void* stream);

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
/// (d_inner).  out = rms_norm(o) * o_norm * sigmoid(g2), d_inner values.  A call of 32 rows or more without snap
/// takes three launches for the same bits, and leaves its work in qkv and g1.
void kda_step(float* qkv, float* g1, const float* a, const float* beta, float* s, const float* g2,
              const float* o_norm, float* out, int head_dim, int n_head, int rows, float gate_lower_bound,
              float norm_eps, void* stream, float* snap = nullptr, size_t stride = 0, int n_snap = 0);

/// Stores the tokens' MLA latents (n values each) as fp16 from row `pos` of the layer's cache.  With `host`, the
/// rows from `split` on go there instead (KvSpill), row `split` first.
void store_f16(const float* x, uint16_t* cache, int n, int64_t pos, int rows, void* stream, uint16_t* host = nullptr,
               int64_t split = 0);
/// The same from the row the card holds at `at` when the launch runs; with `ring`, a row is its number % ring.
/// `host` and `split` as for store_f16(), not with `ring`.
void store_f16_at(const float* x, uint16_t* cache, int n, const long long* at, int rows, int ring, void* stream,
                  uint16_t* host = nullptr, int64_t split = 0);
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
/// A cuBLAS handle that enqueues on `stream`, for dsa_attend and gemm_f32.  Throws std::runtime_error.
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
/// idx_score() for a prompt batch: the products of the queries and the pools' keys through cuBLAS, as many tokens
/// at a time as have theirs in the work_floats floats of `work`, n_head for each pool of the call's last token.
/// A score has other last bits than idx_score()'s.  Throws std::runtime_error, and std::invalid_argument when
/// work has no room for a token.
void idx_score_batch(void* blas, const float* q, const float* w, const float* cache, float* score, int score_stride,
                     int dim, int n_head, int kpool, int64_t pos, int rows, float* work, size_t work_floats,
                     void* stream);
/// For each of the tokens from position `pos`, from its scores: the n_top pools with the highest, the lower pool
/// first among equals, as their kpool rows each in that order, then the rows that are in no whole pool yet.  A
/// token's list starts sel_stride apart and takes n_top * kpool + kpool - 1 at most.  Every token has n_top
/// whole pools or more.
void idx_select(const float* score, int score_stride, int32_t* sel, int sel_stride, int n_top, int kpool,
                int64_t pos, int rows, void* stream);

/// out = silu(min(gate, limit)) * clamp(up, -limit, limit).
void swiglu_clamp(const float* gate, const float* up, float* out, int n, float limit, void* stream);
/// swiglu_clamp for rows that each hold n gate values and then n up values; out has n values a row.
void swiglu_clamp_rows(const float* gate_up, float* out, int n, int rows, float limit, void* stream);
/// out = gelu(gate) * up, with the tanh form of GELU that ggml's CUDA backend computes.
void geglu(const float* gate, const float* up, float* out, int n, void* stream);
/// geglu for rows that each hold n gate values and then n up values; out has n values a row.
void geglu_rows(const float* gate_up, float* out, int n, int rows, void* stream);
/// acc[t] += sum over k < n_used of w[t * n_used + k] * x[t * n_used + k], for rows of n values: a token's
/// weighted experts added to its row.
void moe_sum(const float* x, const float* w, float* acc, int n, int n_used, int rows, void* stream);
/// acc = x * a, or acc += x * a when `add`.
void axpy(const float* x, float a, float* acc, int n, bool add, void* stream);
/// Up to 32 vectors of n values, each times its weight into its own sum, one after the other: acc[k] += x[k] * w[k]
/// for k = 0, 1, ...  Two of them may have the same sum.
struct ScaledRows {
    const float* x[32];
    float* acc[32];
    float w[32];
};
void add_scaled(const ScaledRows& rows, int count, int n, void* stream);
/// acc[row] += b for each of the rows of n values.
void add_bias(const float* b, float* acc, int n, int rows, void* stream);

// ---- attention over a key and value cache (qwen35moe, gemma4), and the gated delta net with one decay a head

/// For `rows` tokens of `heads` heads each: a head's RMS norm with the weights w (dim values, the same for every
/// head; null: none), then ggml's rotary embedding on its first n_rot values in the NeoX pairing: value i goes
/// with value i + n_rot / 2, and the pair turns by position * theta_scale^i, divided by freq[i] when freq is
/// given.  A head's input starts x_stride floats after the one before it; y has dim values a head.
/// The position is the token's place in the call counted from `pos`.  With `axes` (three numbers a token) it is
/// the token's position on one of three axes instead: pair i takes axis (i % sections) % 3, ggml's interleaved
/// rope_multi, which is how a picture's rows get a line and a column.
/// With `at`, the first token's position is not `pos` but what the card holds at `at` when the launch runs: a
/// launch recorded once can then be replayed for every position (attend() and store_f16_at() take it too).
void head_norm_rope(const float* x, int x_stride, const float* w, float* y, int dim, int heads, int n_rot,
                    float theta_scale, int64_t pos, int rows, float eps, void* stream, const float* freq = nullptr,
                    const int32_t* axes = nullptr, int sections = 1, const long long* at = nullptr);
/// Causal attention for the tokens from position `pos`: token t attends the rows 0 to pos + t, or with `window`
/// only the last `window` of them.  q: n_head * dim a token.  kcache and vcache: fp16, kv_heads * dim a row; the
/// heads read a row's keys and values in turn, n_head / kv_heads heads each.  With `ring`, the sequence's row p
/// is cache row p % ring.  out: n_head * dim a token.  It takes `chunk` tokens a launch, which read the caches
/// once between them, and score is scratch of chunk * n_head floats for each row a launch's tokens attend
/// between them: pos + rows, or window - 1 + chunk if that is less.  A token comes out the same whatever tokens
/// the call has beside it.  Queries and keys are read four values at a time: dim is a multiple of 4, q lies at
/// a multiple of 16 bytes and kcache of 8, or it throws std::invalid_argument.
/// With `whole` the call's tokens see each other: each attends up to the call's last row, so a token can attend
/// window - 1 + rows of them.
/// With `at`, the tokens start at the position the card holds there; the launches are sized for `room` rows,
/// which must be no fewer than a launch's tokens attend between them, and score has room for them.
/// score_floats (STRATA_POLY_ATTN_FIT): the floats `score` really holds.  At Qwen3.5/3.6's shape a launch of the
/// rows kernels then takes as many tokens as its scores and parts' sums fit there, instead of `chunk` tokens when
/// they fit in chunk tokens' scores of the launch's own span; 0, the default, keeps that.
///
/// KV streaming (AMD_KV_STREAMING.md): a key and value cache whose sequence rows from `split` on lie in pinned host
/// memory, row `split` first, read through its device pointer (zero-copy, over PCIe); the rows below `split` are
/// in the VRAM caches the call is given.  The kernels read the same values in the same order as from caches wholly
/// in VRAM, so a token comes out with the same bits.  attend_batch() copies a block of rows that reaches past
/// `split` to kstage / vstage (VRAM, stage_rows rows each) before its products read it.  Not with `ring`.
struct KvSpill {
    const uint16_t* khost = nullptr;
    const uint16_t* vhost = nullptr;
    long long split = 0;
    uint16_t* kstage = nullptr;
    uint16_t* vstage = nullptr;
    int stage_rows = 0;
};
/// The rows of a block attend_batch() takes at most, for a call of `rows` tokens and score_floats floats of scores:
/// what KvSpill::stage_rows must hold for that call.
int attend_batch_block(size_t score_floats, int rows, int n_head);
void attend(const float* q, const uint16_t* kcache, const uint16_t* vcache, int64_t pos, int rows, float* score,
            int chunk, float* out, int dim, int n_head, int kv_heads, float scale, void* stream, int window = 0,
            int ring = 0, bool whole = false, const long long* at = nullptr, int room = 0,
            const KvSpill* spill = nullptr, size_t score_floats = 0);
/// attend() for a prompt batch: the same attention with cuBLAS products on the tensor cores, several times
/// faster on many tokens.  The products read the cache's halves where they lie, take the queries and the
/// softmax's weights rounded to halves, and sum floats.  A token's result depends on the tokens beside it, so
/// the windows of guessed tokens use attend().
/// It takes the call's tokens together and the rows they attend a block at a time: as many rows as have their
/// scores and weights, 6 bytes for each of a token's n_head heads, in the score_floats floats of `score`, which
/// must hold 2 * n_head floats.  work: rows * n_head * (dim + dim / 2 + 3) + 1 floats.  Throws
/// std::runtime_error.
void attend_batch(void* blas, const float* q, const uint16_t* kcache, const uint16_t* vcache, int64_t pos, int rows,
                  float* score, size_t score_floats, float* work, float* out, int dim, int n_head, int kv_heads,
                  float scale, void* stream, int window = 0, int ring = 0, bool whole = false,
                  const KvSpill* spill = nullptr);
// ---- the same attention over a Q8_0 key and value cache (docs/AMD_HIP_KV_Q8.md); the fp16 functions above are
// untouched by it
//
// A row (one position of one layer's keys, or of its values) holds, for each of the kv_heads heads, its dim values
// as int8 codes and then their dim / 32 scales as fp16, one for each block of 32 values, the head padded to a multiple
// of 16 bytes: kv_heads * q8_head_bytes(dim), 544 bytes at 2 heads of 256 against 1,024 in fp16.  A value is code *
// scale.  The codes are ggml's quantize_row_q8_0's: the block's largest |x| over 127 is its scale d, a code is x
// times 1/d rounded half away from zero (every code 0 when the block is), and d is kept rounded to fp16.  A value so
// lies within d / 2 of x, plus 127 times the scale's rounding: 0.562 d at most.  dim is a multiple of 32.
constexpr int kQ8Block = 32;
inline size_t q8_head_bytes(int dim) { return (size_t(dim) + size_t(dim / kQ8Block) * 2 + 15) & ~size_t(15); }
inline size_t q8_row_bytes(int dim, int kv_heads) { return size_t(kv_heads) * q8_head_bytes(dim); }
/// store_f16() for a q8_0 cache: the tokens' rows of kv_heads * dim floats each, quantized into the cache's rows from
/// row `pos` on.
void store_q8(const float* x, uint8_t* cache, int dim, int kv_heads, int64_t pos, int rows, void* stream);
/// The same from the row the card holds at `at` when the launch runs, for a launch that is recorded once and replayed.
void store_q8_at(const float* x, uint8_t* cache, int dim, int kv_heads, const long long* at, int rows, void* stream);
/// Rows [row0, row0 + m) of a q8_0 cache as halves, one row after the other into `out` (kv_heads * dim halves a row):
/// each value rounded to fp16, as attend_batch_q8() hands its products.
void kv_dequant_q8(const uint8_t* cache, long long row0, int m, int dim, int kv_heads, uint16_t* out, void* stream);
/// attend() over q8_0 caches: the same tokens, positions, scratch and results, each key and value read as its code
/// times its scale where attend() reads a half.  The same choice of kernels: at Qwen3.5/3.6's shape (dim 256, 8 heads
/// a group, no window, ring or `whole`) the rows kernels when the scratch has room for the parts' sums (counted as
/// attend() counts it, with or without score_floats), unless STRATA_POLY_ATTN=0; else the general ones.  A token
/// comes out the same whatever tokens the call has beside it.
/// No KV streaming.  q lies at a multiple of 16 bytes, and so do the caches.
void attend_q8(const float* q, const uint8_t* kcache, const uint8_t* vcache, int64_t pos, int rows, float* score,
               int chunk, float* out, int dim, int n_head, int kv_heads, float scale, void* stream, int window = 0,
               int ring = 0, bool whole = false, const long long* at = nullptr, int room = 0,
               size_t score_floats = 0);
/// attend_batch() over q8_0 caches: each block of rows is turned into halves in kstage and vstage first
/// (kv_dequant_q8), stage_rows rows each, which is the most a block takes, and the same products read them there.
/// Where the blocks are attend_batch()'s (stage_rows no fewer than its block), the result is attend_batch()'s over a
/// cache of those halves, bit for bit.  No KV streaming.
void attend_batch_q8(void* blas, const float* q, const uint8_t* kcache, const uint8_t* vcache, int64_t pos, int rows,
                     float* score, size_t score_floats, float* work, float* out, int dim, int n_head, int kv_heads,
                     float scale, void* stream, uint16_t* kstage, uint16_t* vstage, int stage_rows, int window = 0,
                     int ring = 0, bool whole = false);
/// attend_q8() and attend_batch_q8() with the keys and the values each in its own format: k_q8 / v_q8, a q8_0 cache,
/// else an fp16 one (Options::kv_k / kv_v, docs/AMD_HIP_KV_Q8.md 3.3).  The scores read only the keys and the sums
/// only the values, so a cache in q8_0 is read by the q8_0 kernels and one in fp16 by attend()'s, in the same launches
/// and order: with both true they are attend_q8() and attend_batch_q8().  The fp16 cache of a prompt batch is read
/// where it lies; only a q8_0 one needs its stage.  For tests of the two halves of the error apart, and the engine's
/// q8_0 cache.  No KV streaming.
void attend_kv(const float* q, const void* kcache, bool k_q8, const void* vcache, bool v_q8, int64_t pos, int rows,
               float* score, int chunk, float* out, int dim, int n_head, int kv_heads, float scale, void* stream,
               int window = 0, int ring = 0, bool whole = false, const long long* at = nullptr, int room = 0,
               size_t score_floats = 0);
void attend_batch_kv(void* blas, const float* q, const void* kcache, bool k_q8, const void* vcache, bool v_q8,
                     int64_t pos, int rows, float* score, size_t score_floats, float* work, float* out, int dim,
                     int n_head, int kv_heads, float scale, void* stream, uint16_t* kstage, uint16_t* vstage,
                     int stage_rows, int window = 0, int ring = 0, bool whole = false);
/// The rows of the stage attend_batch_kv() dequantizes a q8_0 cache's blocks into (kv_heads * dim halves each): 16
/// MiB for the two at Qwen3.6's shape.  Its blocks are no longer, so a call of a few hundred tokens or fewer takes more.
constexpr int kQ8StageRows = 8192;

/// x = cap * tanh(x / cap).
void softcap(float* x, size_t n, float cap, void* stream);
/// The first of the largest of n logits and the softmax's probability of it, summed without the logits more than
/// 20 below it.
struct Likeliest {
    int32_t token;
    float prob;
};
void likeliest(const float* logits, int n, Likeliest* out, void* stream);
/// acc[r][i] *= sigmoid(g[r][i]) for `rows` groups of n values; group r's gates start r * g_stride floats into g.
/// With one_gate a group has one gate for all its values.
void mul_sigmoid(float* acc, const float* g, int n, int rows, int g_stride, bool one_gate, void* stream);
/// A causal convolution of four taps over each of `channels` channels, and its SiLU.  x and out: `channels` values
/// a token.  state: 3 past inputs a channel, oldest first, advanced here by `rows` tokens.  w: (4, channels).
/// With snap, the state as it is after some of the call's tokens is also written there, for a window of guessed
/// tokens to go back to (gdn_step too): `snaps` has a bit for each token that leaves it, and the states lie in
/// the tokens' order, `stride` floats apart.
void conv4_silu(const float* x, float* state, const float* w, float* out, int channels, int rows, void* stream,
                float* snap = nullptr, size_t stride = 0, unsigned snaps = 0);
/// The delta-rule step of every value head and its gated output, token after token: ggml's gated_delta_net op
/// with one decay a head.  qkv: a token's conv outputs, the k_heads key heads' q, their k, then the n_head value
/// heads' v, dim values each; value head h reads key head h % k_heads.  alpha and beta: n_head a token, as their
/// projections give them.  dt, a: n_head.  The state decays by exp(softplus(alpha + dt) * a) and takes
/// sigmoid(beta) of the error.  s: n_head * dim * dim, updated.  z: n_head * dim a token.
/// out = rms_norm(o) * o_norm * silu(z), n_head * dim values a token.  qkv, alpha and beta are overwritten.
void gdn_step(float* qkv, float* alpha, float* beta, const float* dt, const float* a, float* s,
              const float* z, const float* o_norm, float* out, int dim, int n_head, int k_heads, int rows,
              float norm_eps, void* stream, float* snap = nullptr, size_t stride = 0, unsigned snaps = 0);
/// The kernel of a prompt batch's state update in gdn_step() (32 tokens or more, no snapshots, heads of 128): 0
/// the one of a few tokens, 1, 2 or 4 the one that keeps the state in registers, with that many columns a warp; the
/// same bits.  STRATA_POLY_GDN_STATE sets it when this is not called; 4 when neither is.
void gdn_state_way(int way);

#if defined(STRATA_USE_HIP)
/// n words of 4 bytes from src to dst, bit for bit, by a kernel.  Either side may be pinned host memory
/// (cudaMallocHost), which the card reads and writes where it lies.  On HIP under Windows a small cudaMemcpyAsync
/// between two kernels costs some 100 microseconds (the copy engine's turn; poly_kernel_bench), and this costs what
/// a small kernel does.  What it writes into host memory is the host's to read once an event recorded after it
/// has completed.  Both pointers are 4-byte aligned.
void copy_words(const void* src, void* dst, size_t n, void* stream);

/// add_scaled() with the weights read from device memory, w[k] for vector k.
void add_scaled_dev(const ScaledRows& rows, const float* w, int count, int n, void* stream);

/// Where route_topk_softmax() leaves a token's routing for the host (pinned, coherent memory) besides the card.
struct CardRoute {
    int32_t* ids = nullptr;         ///< the picked experts
    float* w = nullptr;             ///< their weights
    uint8_t* cached = nullptr;      ///< whether each is in VRAM
    const float* x = nullptr;       ///< the experts' input on the card, n_x values ...
    float* x_host = nullptr;        ///< ... copied here, for the CPU's experts
    int n_x = 0;
    int* need = nullptr;            ///< on the card: 1 when a pick is not cached (wait_add() waits for the host)
    uint32_t* ready = nullptr;      ///< set to *seq once all of the above is written (null: not set)
    const uint32_t* seq = nullptr;  ///< on the card: the number of the token under way
    uint32_t* blocks = nullptr;     ///< on the card, 0 between calls: counts the tokens routed, for `ready`
};

/// A token's routing on the card, as Qwen35::route() does it on the host: the softmax of n_expert scores, the
/// n_used likeliest (the lower number first among equals), their probabilities over the sum of them (at least
/// min_sum).  table holds the layer's experts' slots in VRAM (null: not cached).  For pick k: gate_up->w[k] and
/// down->w[k] (at down_off into the slot) get the slot or null, w_dev[k] its weight (0 if not cached); `out` gets
/// the rest.  n_expert at most 1024, n_used at most 8.  rows tokens (up to 8), a block each, each the same as
/// alone: token r's scores at scores + r * n_expert, its slots gate_up[r] and down[r], its weights at w_dev + 8 r,
/// and in `out` ids, w and cached at 8 r, x and x_host at r * n_x, need at r; `ready` is set once, when all are.
void route_topk_softmax(const float* scores, int n_expert, int n_used, float min_sum, const uint8_t* const* table,
                        size_t down_off, void* gate_up, void* down, float* w_dev, const CardRoute& out, int rows,
                        void* stream);

/// When need[r] for any of rows tokens: waits for *go == *seq (the host's word that its experts' sums are in cpu),
/// then fo's row r += cpu's row r (n values each) for each r with need[r].  A wait of more than `ticks`
/// (wall_clock64()) sets *failed and adds nothing.  `waited`, when given, gets the ticks waited added to it.
void wait_add(const int* need, const uint32_t* go, const uint32_t* seq, const float* cpu, float* fo, int n, int rows,
              long long ticks, uint32_t* failed, void* stream, unsigned long long* waited = nullptr);

/// *p = v, in stream order.
void set_word(uint32_t* p, uint32_t v, void* stream);

/// table[i] = slot, in stream order (the expert cache's table on the card follows the host's).
void set_slot(const uint8_t** table, size_t i, const uint8_t* slot, void* stream);
#endif

/// STRATA_POLY_PREFILL_PROF, a diagnostic (engine.cpp): the parts of a prompt batch whose card time is measured,
/// from an event recorded on the stream after each.  A part is the time from the event before it to its own.
enum PrefillPart {
    kPpStart = -1,   // a measured stretch starts: the time before it belongs to no part
    kPpEmbed, kPpNormIn, kPpGdnProj, kPpGdnConv, kPpGdnNorm, kPpGdnGates, kPpGdnState, kPpGdnOut, kPpGdnWo,
    kPpAttnQ, kPpAttnKv, kPpAttnGroup, kPpAttnScore, kPpAttnSoftmax, kPpAttnScale, kPpAttnValue, kPpAttnFinish,
    kPpAttnCore, kPpAttnOut, kPpResidNorm, kPpRouter, kPpShared, kPpHostGap, kPpMoeTables, kPpCachedGu,
    kPpCachedAct, kPpCachedDown, kPpStageWait, kPpStagedMmq, kPpMoeSum, kPpTail, kPpCkpt, kPpOut, kPpParts
};
/// Set by the engine while a prompt batch is measured, null otherwise: gdn_step() calls it between its launches,
/// and attend_batch() calls prefill_mark_fine between its products (STRATA_POLY_PREFILL_PROF=2).
extern void (*prefill_mark)(void* stream, int part);
extern void (*prefill_mark_fine)(void* stream, int part);

}  // namespace strata::poly
