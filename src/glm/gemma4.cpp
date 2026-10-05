// src/glm/gemma4.cpp - Gemma 4's layers.  The reference for the math is llama.cpp's gemma4 graph.
#include "gemma4.hpp"

#include "runtime.hpp"
#include "strata/glm/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace strata::glm {
namespace {

constexpr float kMinWeightSum = 6.103515625e-5f;   // the floor ggml's MoE graph puts under the gate sum
constexpr size_t kScoreFloats = size_t(16) << 20;  // the attention's scores of the tokens it takes at a time

// A value a file gives once for the model or once a layer.
int64_t of_layer(const MetaValue& v, int il) {
    if (v.type != MetaType::ARRAY) return (int64_t) v.u;
    if ((size_t) il >= v.items.size()) throw std::runtime_error("a per-layer value of the GGUF ends before its layers");
    return (int64_t) v.items[(size_t) il].u;
}

}  // namespace

void Gemma4::begin_pass() {
    blocks_.assign(layers_.size(), Block{});
    freq_ = Dev{};
}

void Gemma4::load_geometry() {
    load_shape();
    auto find = [&](const char* key) { return meta_in(*gg_, key); };
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported model: ") + what);
    };
    auto must = [&](const char* key) -> const MetaValue& {
        const MetaValue* v = find(key);
        need(v != nullptr, key);
        return *v;
    };
    gelu_ = true;
    window_ = shape_i("attention.sliding_window");
    dim_full_ = shape_i("attention.key_length");
    dim_swa_ = shape_i("attention.key_length_swa");
    const MetaValue* rot = find("rope.dimension_count");
    const MetaValue* rot_swa = find("rope.dimension_count_swa");
    rot_full_ = rot ? (int) rot->u : dim_full_;
    rot_swa_ = rot_swa ? (int) rot_swa->u : dim_swa_;
    base_full_ = shape_f("rope.freq_base");
    const MetaValue* base_swa = find("rope.freq_base_swa");
    base_swa_ = base_swa ? (float) base_swa->f : base_full_;
    const MetaValue* cap = find("final_logit_softcapping");
    softcap_ = cap ? (float) cap->f : 0.f;

    const MetaValue& pattern = must("attention.sliding_window_pattern");
    const MetaValue& heads = must("attention.head_count");
    const MetaValue& kv = must("attention.head_count_kv");
    swa_.resize((size_t) n_layer_);
    kv_.resize((size_t) n_layer_);
    n_head_ = (int) of_layer(heads, 0);
    for (int il = 0; il < n_layer_; ++il) {
        swa_[(size_t) il] = of_layer(pattern, il) != 0;
        kv_[(size_t) il] = (int) of_layer(kv, il);
        need(of_layer(heads, il) == n_head_, "layers with different numbers of heads");
        need(kv_[(size_t) il] > 0 && n_head_ % kv_[(size_t) il] == 0, "bad attention.head_count_kv");
    }
    need(n_head_ > 0 && n_head_ <= 256, "bad attention.head_count");
    need(window_ > 0 && dim_full_ > 0 && dim_swa_ > 0, "bad attention.sliding_window or key lengths");
    need(shape_i("attention.value_length") == dim_full_ && shape_i("attention.value_length_swa") == dim_swa_,
         "the attention's keys and values differ in length");
    need(rot_full_ % 2 == 0 && rot_full_ <= dim_full_ && rot_swa_ % 2 == 0 && rot_swa_ <= dim_swa_,
         "bad rope.dimension_count");
    const MetaValue* shared = find("attention.shared_kv_layers");
    need(!shared || shared->u == 0, "layers that read another layer's keys and values");
    const MetaValue* per_layer = find("embedding_length_per_layer_input");
    need(!per_layer || per_layer->u == 0, "token embeddings of a layer's own");

    // the dense MLP's width is its matrix's: a file may give it once a layer
    const TensorInfo* gate = tensor(0, "ffn_gate.weight");
    need(gate && gate->shape.size() == 2, "no blk.0.ffn_gate.weight");
    n_ff_ = (int) gate->shape[1];

    // GELU has no limits to clamp to
    clamp_exp_.assign((size_t) n_layer_, INFINITY);
    clamp_shexp_ = clamp_exp_;
}

void Gemma4::load_layer(int il) {
    Layer& L = layers_[(size_t) il];
    Block& B = blocks_[(size_t) il];
    auto f32 = [&](const char* n, int64_t a, int64_t b = 1) { return upload(tensor(il, n), blk(il, n), a, b, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(il, n), a, b, kAnyMat); };
    auto has = [&](const char* n) { return tensor(il, n) != nullptr; };

    B.swa = swa_[(size_t) il] != 0;
    B.dim = B.swa ? dim_swa_ : dim_full_;
    B.n_kv = kv_[(size_t) il];
    B.n_rot = B.swa ? rot_swa_ : rot_full_;
    B.theta_scale = std::pow(B.swa ? base_swa_ : base_full_, -2.f / (float) B.n_rot);
    L.fixed = false;   // the attention's launches take the position

    L.attn_norm = f32("attn_norm.weight", n_embd_);
    B.wq = mat("attn_q.weight", n_embd_, (int64_t) n_head_ * B.dim);
    B.wk = mat("attn_k.weight", n_embd_, (int64_t) B.n_kv * B.dim);
    if (has("attn_v.weight")) B.wv = mat("attn_v.weight", n_embd_, (int64_t) B.n_kv * B.dim);
    B.wo = mat("attn_output.weight", (int64_t) n_head_ * B.dim, n_embd_);
    B.q_norm = f32("attn_q_norm.weight", B.dim);
    B.k_norm = f32("attn_k_norm.weight", B.dim);
    B.post_attn = f32("post_attention_norm.weight", n_embd_);
    // the full layers' rotary pairs turn by frequencies the file scales; the tensor is the model's, not a layer's
    if (!B.swa && !freq_.p)
        freq_ = upload(gg_->find("rope_freqs.weight"), "rope_freqs.weight", B.n_rot / 2, 1, kF32);

    L.ffn_norm = f32("ffn_norm.weight", n_embd_);
    B.ffn_gate = mat("ffn_gate.weight", n_embd_, n_ff_);
    B.ffn_up = mat("ffn_up.weight", n_embd_, n_ff_);
    B.ffn_down = mat("ffn_down.weight", n_ff_, n_embd_);
    B.post_ffn = f32("post_ffw_norm.weight", n_embd_);
    if (const TensorInfo* s = tensor(il, "layer_output_scale.weight")) {
        if (s->type != kTypeF32 || s->elements() != 1 || home(*s).payload(*s) != sizeof(float))
            throw std::runtime_error(s->name + " is " + shape_of(*s) + ", expected F32 [1]");
        std::memcpy(&B.out_scale, home(*s).data(*s), sizeof(float));
    }
    if (!has("ffn_gate_inp.weight")) return;   // a layer with the MLP alone

    B.rscale = f32("ffn_gate_inp.scale", n_embd_);
    B.post_mlp = f32("post_ffw_norm_1.weight", n_embd_);
    B.pre_moe = f32("pre_ffw_norm_2.weight", n_embd_);
    B.post_moe = f32("post_ffw_norm_2.weight", n_embd_);
    load_moe(il, false);
    B.escale.assign((size_t) n_expert_, 1.f);
    if (const TensorInfo* s = tensor(il, "ffn_down_exps.scale")) {
        if (s->type != kTypeF32 || (int64_t) s->elements() != n_expert_ ||
            home(*s).payload(*s) != (size_t) n_expert_ * sizeof(float))
            throw std::runtime_error(s->name + " is " + shape_of(*s) + ", expected F32 [" +
                                     std::to_string(n_expert_) + "]");
        std::memcpy(B.escale.data(), home(*s).data(*s), (size_t) n_expert_ * sizeof(float));
    }
    for (const char* other : {"ffn_gate_exps.scale", "ffn_up_exps.scale", "ffn_gate_up_exps.scale"})
        if (has(other)) throw std::runtime_error(blk(il, other) + ": a scale this engine does not apply");
}

int Gemma4::widest() const { return n_head_ * std::max(dim_full_, dim_swa_); }

// The caches: a full layer keeps a row a position, a window layer a ring of the rows a call can read.
void Gemma4::alloc_caches() {
    ring_ = std::min(opt_.max_context, window_ + rows_);
    keep_ = std::min(window_, ring_);
    size_t all = 0;
    for (const Block& B : blocks_) all += 2 * (size_t) (B.swa ? ring_ : opt_.max_context) * B.n_kv * B.dim;
    uint16_t* kv = (uint16_t*) dmalloc(all * sizeof(uint16_t), "the attention cache");
    for (Block& B : blocks_) {
        const size_t rows_n = (size_t) (B.swa ? ring_ : opt_.max_context) * B.n_kv * B.dim;
        B.keys = kv;
        kv += rows_n;
        B.vals = kv;
        kv += rows_n;
    }
    chunk_ = (int) std::clamp<size_t>(kScoreFloats / ((size_t) n_head_ * opt_.max_context), 1, (size_t) rows_);
    probs_.resize((size_t) n_expert_);
}

// Per-token buffers hold rows_ tokens, row after row.
void Gemma4::assign_scratch() {
    const size_t n = (size_t) rows_, heads = n * widest();
    size_t kv = 0;
    for (const Block& B : blocks_) kv = std::max(kv, n * B.n_kv * B.dim);
    qraw_ = fbuf(heads);
    q_ = fbuf(heads);
    kraw_ = fbuf(kv);
    k_ = fbuf(kv);
    vraw_ = fbuf(kv);
    v_ = fbuf(kv);
    att_ = fbuf(heads);
    score_ = fbuf((size_t) chunk_ * n_head_ * opt_.max_context);
    mlp_ = fbuf(n * n_embd_);
    rin_ = fbuf(n * n_embd_);
}

// A sequence's rows are the full layers': a window layer's are part of its state (save_extra()).
size_t Gemma4::row_bytes() const {
    size_t b = 0;
    for (const Block& B : blocks_)
        if (!B.swa) b += 2 * (size_t) B.n_kv * B.dim * sizeof(uint16_t);
    return b;
}

void Gemma4::save_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context)
        throw std::logic_error("save_rows: the rows are outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    for (const Block& B : blocks_) {
        const size_t row = (size_t) B.n_kv * B.dim * sizeof(uint16_t), bytes = (size_t) n * row;
        if (B.swa || bytes == 0) continue;
        for (const uint16_t* cache : {B.keys, B.vals}) {
            ck(cudaMemcpy(dst, (const uint8_t*) cache + (size_t) first * row, bytes, cudaMemcpyDeviceToHost),
               "save rows");
            dst += bytes;
        }
    }
}

void Gemma4::load_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: the rows are outside what was saved");
    ck(cudaStreamSynchronize(cs(st_)), "load rows");
    for (const Block& B : blocks_) {
        const size_t row = (size_t) B.n_kv * B.dim * sizeof(uint16_t), bytes = (size_t) n * row;
        if (B.swa || bytes == 0) continue;
        for (uint16_t* cache : {B.keys, B.vals}) {
            ck(cudaMemcpy((uint8_t*) cache + (size_t) first * row, src + (size_t) (first - from) * row, bytes,
                          cudaMemcpyHostToDevice),
               "load rows");
            src += (size_t) count * row;
        }
    }
}

// The last keep_ positions before `pos` as runs of a ring's rows: f(the run's place among the keep_, its first
// ring row, its rows).  A position before the sequence's start has no run.
template <class F>
void Gemma4::window_runs(int64_t pos, F&& f) const {
    for (int64_t p = std::max<int64_t>(0, pos - keep_); p < pos;) {
        const int64_t at = p % ring_, m = std::min<int64_t>(pos - p, ring_ - at);
        f((size_t) (p - (pos - keep_)), (size_t) at, (size_t) m);
        p += m;
    }
}

// What a window layer has of the sequence: the rows the next token attends.
size_t Gemma4::extra_bytes() const {
    size_t b = 0;
    for (const Block& B : blocks_)
        if (B.swa) b += 2 * (size_t) keep_ * B.n_kv * B.dim * sizeof(uint16_t);
    return b;
}

void Gemma4::save_extra(uint8_t* dst) {
    std::memset(dst, 0, extra_bytes());   // the places of positions before the sequence's start
    for (const Block& B : blocks_) {
        if (!B.swa) continue;
        const size_t row = (size_t) B.n_kv * B.dim * sizeof(uint16_t);
        for (const uint16_t* cache : {B.keys, B.vals}) {
            window_runs(pos_, [&](size_t slot, size_t at, size_t m) {
                ck(cudaMemcpy(dst + slot * row, (const uint8_t*) cache + at * row, m * row, cudaMemcpyDeviceToHost),
                   "save state");
            });
            dst += (size_t) keep_ * row;
        }
    }
}

void Gemma4::load_extra(const uint8_t* src, int64_t pos) {
    for (const Block& B : blocks_) {
        if (!B.swa) continue;
        const size_t row = (size_t) B.n_kv * B.dim * sizeof(uint16_t);
        for (uint16_t* cache : {B.keys, B.vals}) {
            window_runs(pos, [&](size_t slot, size_t at, size_t m) {
                ck(cudaMemcpy((uint8_t*) cache + at * row, src + slot * row, m * row, cudaMemcpyHostToDevice),
                   "load state");
            });
            src += (size_t) keep_ * row;
        }
    }
}

// n tokens' rows from position `pos` into a layer's cache.  A ring's rows may end and go on from its start.
void Gemma4::store(const float* x, uint16_t* cache, const Block& B, int64_t pos, int n) {
    const int width = B.n_kv * B.dim;
    if (!B.swa) return store_f16(x, cache, width, pos, n, st_);
    const int at = (int) (pos % ring_), head = std::min(n, ring_ - at);
    store_f16(x, cache, width, at, head, st_);
    if (head < n) store_f16(x + (size_t) head * width, cache, width, 0, n - head, st_);
}

// Queries and keys are normed a head and turned by the position; values are normed without weights.  The scores
// are not scaled: the norms set their size.
void Gemma4::attn(Block& B, int64_t pos, int n) {
    const float* freq = B.swa ? nullptr : fp(freq_);
    lin(B.wq, xn_, qraw_, n, xq_);
    head_norm_rope(qraw_, B.dim, fp(B.q_norm), q_, B.dim, n_head_, B.n_rot, B.theta_scale, pos, n, rms_eps_, st_,
                   freq);
    lin(B.wk, xn_, kraw_, n, xq_);
    head_norm_rope(kraw_, B.dim, fp(B.k_norm), k_, B.dim, B.n_kv, B.n_rot, B.theta_scale, pos, n, rms_eps_, st_,
                   freq);
    const float* values = kraw_;   // a layer without a value projection takes the keys'
    if (B.wv.p) {
        lin(B.wv, xn_, vraw_, n, xq_);
        values = vraw_;
    }
    head_norm_rope(values, B.dim, nullptr, v_, B.dim, B.n_kv, 0, 1.f, pos, n, rms_eps_, st_);
    store(k_, B.keys, B, pos, n);
    store(v_, B.vals, B, pos, n);
    attend(q_, B.keys, B.vals, pos, n, score_, chunk_, att_, B.dim, n_head_, B.n_kv, 1.f, st_, B.swa ? window_ : 0,
           B.swa ? ring_ : 0);
    lin(B.wo, att_, ao_, n);
}

// A layer from its input to its router's copy (a layer without routed experts: to its end).  r_ holds the
// residual with the attention added from the middle on, which the MLP, the router and the experts each norm
// their own way.  A prompt batch's MoE (moe_rows) does its own routing.
void Gemma4::layer_head(Layer& L, int il, int n) {
    Block& B = blocks_[(size_t) il];
    const int all = n * n_embd_;
    rms_norm(r_, fp(L.attn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, all, 1, st_);
    dbg("attn_in", il, xn_, n_embd_);
    attn(B, pos_, n);
    rms_norm(ao_, fp(B.post_attn), cur_, n_embd_, n, rms_eps_, st_);
    dbg("attn", il, cur_, n_embd_);
    axpy(cur_, 1.f, r_, all, true, st_);

    rms_norm(r_, fp(L.ffn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, all, 1, st_);
    lin(B.ffn_gate, xn_, g_, n, xq_);
    lin(B.ffn_up, xn_, u_, n, xq_);
    geglu(g_, u_, h_, n * n_ff_, st_);
    lin(B.ffn_down, h_, cur_, n);
    if (!L.moe) {
        rms_norm(cur_, fp(B.post_ffn), fo_, n_embd_, n, rms_eps_, st_);
        dbg("ffn", il, fo_, n_embd_);
        axpy(fo_, 1.f, r_, all, true, st_);
        if (B.out_scale != 1.f) axpy(r_, B.out_scale, r_, all, false, st_);
        return;
    }
    rms_norm(cur_, fp(B.post_mlp), mlp_, n_embd_, n, rms_eps_, st_);
    dbg("mlp", il, mlp_, n_embd_);

    rms_norm(r_, fp(B.rscale), rin_, n_embd_, n, rms_eps_, st_);
    rms_norm(r_, fp(B.pre_moe), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, all, 1, st_);
    dbg("ffn_in", il, xn_, n_embd_);
    if (!(mmq_ && n >= kStageRows)) moe_router(L, n);
}

// The router reads the residual, normed and scaled a value at a time (rin_).
void Gemma4::router_scores(const Layer& L, int n) { lin(L.gate_inp, rin_, rl_, n); }

// No expert is shared: the routed experts' sum starts from nothing.
void Gemma4::moe_shared(const Layer&, int, int n) {
    ck(cudaMemsetAsync(fo_, 0, (size_t) n * n_embd_ * sizeof(float), cs(st_)), "experts");
}

// The routed experts' sum is normed, the MLP's output is added, their sum is normed again and goes into the
// residual, which the layer then scales.
void Gemma4::layer_tail(Layer&, int il, int n) {
    const Block& B = blocks_[(size_t) il];
    const int all = n * n_embd_;
    rms_norm(fo_, fp(B.post_moe), cur_, n_embd_, n, rms_eps_, st_);
    axpy(mlp_, 1.f, cur_, all, true, st_);
    rms_norm(cur_, fp(B.post_ffn), fo_, n_embd_, n, rms_eps_, st_);
    axpy(fo_, 1.f, r_, all, true, st_);
    if (B.out_scale != 1.f) axpy(r_, B.out_scale, r_, all, false, st_);
}

void Gemma4::output_rows(int first, int rows, float* h) {
    rms_norm(r_ + (size_t) first * n_embd_, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
}

void Gemma4::finish_logits(float* logits, int rows) {
    if (softcap_ > 0.f) softcap(logits, (size_t) rows * n_vocab_, softcap_, st_);
}

float Gemma4::embed_scale() const { return std::sqrt((float) n_embd_); }

// What ggml's MoE graph does for this model: the router's scores over the root of the width, a softmax over every
// expert, the n_used most likely with their weights made to sum to one, and each expert's own scale on its output.
void Gemma4::route(int il, const float* logits, int* ids, float* w) {
    const std::vector<float>& escale = blocks_[(size_t) il].escale;
    const float s = 1.f / std::sqrt((float) n_embd_);
    float mx = logits[0] * s;
    for (int e = 1; e < n_expert_; ++e) mx = std::max(mx, logits[e] * s);
    double all = 0.0;
    for (int e = 0; e < n_expert_; ++e) all += probs_[(size_t) e] = std::exp(logits[e] * s - mx);
    const float inv = (float) (1.0 / all);
    float sum = 0.f;
    for (int k = 0; k < n_used_; ++k) {
        int best = 0;
        for (int e = 1; e < n_expert_; ++e)
            if (probs_[(size_t) e] > probs_[(size_t) best]) best = e;
        ids[k] = best;
        w[k] = probs_[(size_t) best] * inv;
        sum += w[k];
        probs_[(size_t) best] = -1.f;
    }
    const float denom = std::max(sum, kMinWeightSum);
    for (int k = 0; k < n_used_; ++k) w[k] = w[k] / denom * escale[(size_t) ids[k]];
}

}  // namespace strata::glm
