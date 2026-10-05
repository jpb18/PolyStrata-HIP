// src/glm/glm5next.cpp - GLM 5.3 Flash's layers.  The reference for the math is llama.cpp's glm5-next graph.
#include "glm5next.hpp"

#include "mmq.hpp"
#include "runtime.hpp"
#include "strata/glm/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace strata::glm {
namespace {

constexpr int kHc = 4, kConv = 4;   // what kernels.cu is written for; checked at load
constexpr int kAttnRows = 32;       // tokens one pass of the sparse attention takes; their keys are 4 MiB a token
constexpr float kMinWeightSum = 6.103515625e-5f;   // the floor ggml's MoE graph puts under the gate sum

// The model's GGUFs come out of more than one converter.  `name` is what the layout of neurall's llama.cpp fork
// calls a thing, and what this file calls it.  `other` is what the same thing is called in the files made for
// patrickbdevaney's fork (the first, where it differs) and in antirez's (the second).
constexpr Names kTensorNames[] = {
    {"hc_attn_fn.weight", {"hc_attn_fn"}},
    {"hc_attn_scale.weight", {"hc_attn_scale"}},
    {"hc_attn_base.weight", {"hc_attn_base"}},
    {"hc_ffn_fn.weight", {"hc_ffn_fn"}},
    {"hc_ffn_scale.weight", {"hc_ffn_scale"}},
    {"hc_ffn_base.weight", {"hc_ffn_base"}},
    {"indexer_compressor_gate.weight", {"indexer.kpool_gate", "indexer.pool_gate.weight"}},
    {"indexer_compressor_ape.weight", {"indexer.kpool_ape", "indexer.pool_ape.weight"}},
    {"attn_q.weight", {nullptr, "kda_q.weight"}},
    {"attn_k.weight", {nullptr, "kda_k.weight"}},
    {"attn_v.weight", {nullptr, "kda_v.weight"}},
    {"attn_output.weight", {nullptr, "kda_output.weight"}},
    {"ssm_conv1d_q.weight", {nullptr, "kda_q_conv.weight"}},
    {"ssm_conv1d_k.weight", {nullptr, "kda_k_conv.weight"}},
    {"ssm_conv1d_v.weight", {nullptr, "kda_v_conv.weight"}},
    {"ssm_f_a.weight", {nullptr, "kda_f_a.weight"}},
    {"ssm_f_b.weight", {nullptr, "kda_f_b.weight"}},
    {"ssm_g_a.weight", {nullptr, "kda_g_a.weight"}},
    {"ssm_g_b.weight", {nullptr, "kda_g_b.weight"}},
    {"ssm_dt.bias", {nullptr, "kda_dt_bias.weight"}},
    {"ssm_beta.weight", {nullptr, "kda_beta.weight"}},
    {"ssm_a", {nullptr, "kda_a_log.weight"}},
    {"ssm_norm.weight", {nullptr, "kda_o_norm.weight"}},
};
constexpr Names kMetaNames[] = {
    {"kda.gate_lower_bound", {"ssm.gate_lower_bound", "linear_attention.gate_lower_bound"}},
    {"kda.head_dim", {nullptr, "linear_attention.head_dimension"}},
    {"ssm.conv_kernel", {nullptr, "linear_attention.conv_kernel"}},
    {"attention.key_length_mla", {nullptr, "attention.key_length"}},
    {"attention.value_length_mla", {nullptr, "attention.value_length"}},
    {"attention.indexer.kpool", {nullptr, "attention.indexer.pool_size"}},
    {"hyper_connection.count", {"attention.hc.mult"}},
    {"hyper_connection.sinkhorn_iterations", {"attention.hc.sinkhorn_iters"}},
    {"hyper_connection.epsilon", {"attention.hc.eps"}},
    {"swiglu_clamp_exp", {"swiglu_limit"}},   // one limit for every layer, and for the shared expert too
    {"swiglu_clamp_shexp", {"swiglu_limit"}},
};

}  // namespace

const Names* Glm5Next::tensor_alias(const char* name) const { return names_of(kTensorNames, name); }
const Names* Glm5Next::meta_alias(const char* key) const { return names_of(kMetaNames, key); }

void Glm5Next::begin_pass() {
    n_dsa_ = n_kda_ = 0;
    gate_rank_ = 0;
}

void Glm5Next::check_draft_block() {
    const Layer& D = layers_[(size_t) n_layer_];
    if (!D.dsa || !D.moe)
        throw std::runtime_error("the block that guesses ahead is not one this engine runs; --no-draft loads "
                                 "the model without it");
}

int Glm5Next::streams() const { return kHc; }

// The routed experts' sum goes back into the residual streams.
void Glm5Next::layer_tail(Layer&, int, int n) { hc_post(fo_, hcw_, r_, n_embd_, n, st_); }

// The model's output for `rows` tokens from row `first` of the call: the streams' mean, normed.
void Glm5Next::output_rows(int first, int rows, float* h) {
    hc_mean(r_ + (size_t) first * kHc * n_embd_, cur_, n_embd_, rows, st_);
    rms_norm(cur_, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
}

void Glm5Next::load_geometry() {
    const Shards& g = *gg_;
    const MetaValue* arch = g.get("general.architecture");
    // Unsloth's converter writes the name without its hyphen
    if (!arch || (arch->s != "glm5-next" && arch->s != "glm5next"))
        throw std::runtime_error("general.architecture is '" + (arch ? arch->s : std::string()) +
                                 "', this engine runs 'glm5-next'");
    const std::string pre = arch->s + ".";
    auto find = [&](const char* key) { return meta_in(g, key); };
    auto meta = [&](const char* key) -> const MetaValue& {
        const MetaValue* v = find(key);
        if (!v) throw std::runtime_error("the GGUF has no " + pre + key);
        return *v;
    };
    auto meta_i = [&](const char* key) { return (int) meta(key).u; };
    auto meta_f = [&](const char* key) { return (float) meta(key).f; };

    // The block that guesses a second token after the model's own is one more block after the model's.  It is the
    // model's and not a file's: a file that leaves it out can take it from another file of the model (draft_model).
    // One converter gives the block a token table or an output matrix of its own, which this engine does not run.
    const MetaValue* nextn = find("nextn_predict_layers");
    n_layer_ = meta_i("block_count") - (nextn ? (int) nextn->u : 0);
    if (opt_.draft && opt_.window > 1) {
        const std::string probe = blk(n_layer_, "nextn.eh_proj.weight");
        if (nextn && nextn->u == 1 && gg_->find(probe)) block_ = gg_.get();
        else if (dg_ && dg_->find(probe)) block_ = dg_.get();
        else if (dg_)
            throw std::runtime_error(opt_.draft_model + " has no block that guesses ahead for this model: no tensor " +
                                     probe);
    }
    draft_ = block_ && !tensor(n_layer_, "nextn.embed_tokens.weight") &&
             !tensor(n_layer_, "nextn.shared_head_head.weight");
    if (!draft_) block_ = nullptr;
    n_embd_ = meta_i("embedding_length");
    n_ff_dense_ = meta_i("feed_forward_length");
    n_head_ = meta_i("attention.head_count");
    n_vocab_ = meta_i("vocab_size");
    rms_eps_ = meta_f("attention.layer_norm_rms_epsilon");
    // the indexer's LayerNorm; a converter that leaves the key out means the model's own 1e-6
    const MetaValue* norm_eps = find("attention.layer_norm_epsilon");
    norm_eps_ = norm_eps ? (float) norm_eps->f : 1e-6f;
    n_expert_ = meta_i("expert_count");
    n_used_ = meta_i("expert_used_count");
    n_ff_exp_ = meta_i("expert_feed_forward_length");
    w_scale_ = meta_f("expert_weights_scale");
    w_norm_ = meta("expert_weights_norm").u != 0;
    kda_dim_ = meta_i("kda.head_dim");
    kda_lb_ = meta_f("kda.gate_lower_bound");
    q_rank_ = meta_i("attention.q_lora_rank");
    kv_rank_ = meta_i("attention.kv_lora_rank");
    mla_k_ = meta_i("attention.key_length_mla");
    mla_v_ = meta_i("attention.value_length_mla");
    idx_heads_ = meta_i("attention.indexer.head_count");
    idx_dim_ = meta_i("attention.indexer.key_length");
    idx_top_k_ = meta_i("attention.indexer.top_k");
    // tokens to a pool: a key, or else the second axis of a layer's pooling table
    if (const MetaValue* kpool = find("attention.indexer.kpool")) kpool_ = (int) kpool->u;
    for (int il = 0; il < n_layer_ && kpool_ == 0; ++il)
        if (const TensorInfo* ape = tensor(il, "indexer_compressor_ape.weight"))
            kpool_ = ape->shape.size() == 2 ? (int) ape->shape[1] : 0;
    sinkhorn_ = meta_i("hyper_connection.sinkhorn_iterations");
    hc_eps_ = meta_f("hyper_connection.epsilon");
    d_inner_ = n_head_ * kda_dim_;

    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported model: ") + what);
    };
    need(meta_i("hyper_connection.count") == kHc, "hyper_connection.count is not 4");
    need(meta_i("ssm.conv_kernel") == kConv, "ssm.conv_kernel is not 4");
    const MetaValue* gating = find("expert_gating_func");   // left out by one converter: the model's sigmoid
    need(!gating || gating->u == 2, "the expert gate is not a sigmoid");
    need(meta_i("expert_shared_count") == 1, "expert_shared_count is not 1");
    need(n_layer_ > 0 && n_embd_ > 0 && n_embd_ % 32 == 0 && n_vocab_ > 0, "bad block_count or embedding_length");
    need(kda_dim_ > 0 && kda_dim_ % 32 == 0 && kda_dim_ <= 256, "kda.head_dim is not a multiple of 32 up to 256");
    need(idx_dim_ > 0 && idx_dim_ <= 256, "the indexer key is longer than 256");
    need(n_used_ >= 1 && n_used_ <= kMaxUsed && n_used_ <= n_expert_, "bad expert_used_count");
    need(kpool_ > 1 && idx_top_k_ > 0 && idx_top_k_ % kpool_ == 0, "indexer.top_k is not whole pools");

    // a block from another file must be this model's: a pruned model's has fewer experts to choose among
    if (block_ && block_ != gg_.get()) {
        const TensorInfo* gate = tensor(n_layer_, "ffn_gate_inp.weight");
        if (!gate || gate->shape.size() != 2 || (int64_t) gate->shape[0] != n_embd_ ||
            (int64_t) gate->shape[1] != n_expert_)
            throw std::runtime_error(opt_.draft_model + " is not a file of this model: its block's router is " +
                                     (gate ? shape_of(*gate) : std::string("missing")) + ", the model's layers have [" +
                                     std::to_string(n_embd_) + ", " + std::to_string(n_expert_) + "]");
    }
    clamp_exp_ = clamp_limits(find("swiglu_clamp_exp"), n_layer_);
    clamp_shexp_ = find("swiglu_clamp_shexp") ? clamp_limits(find("swiglu_clamp_shexp"), n_layer_) : clamp_exp_;
    if (draft_) {   // the draft block's limits are in the file the block is in: the model's own may end before it
        const MetaValue* routed = meta_in(*block_, "swiglu_clamp_exp");
        const MetaValue* shared = meta_in(*block_, "swiglu_clamp_shexp");
        clamp_exp_.push_back(clamp_limits(routed, n_layer_ + 1).back());
        clamp_shexp_.push_back(shared ? clamp_limits(shared, n_layer_ + 1).back() : clamp_exp_.back());
    }
}

void Glm5Next::load_layer(int il) {
    Layer& L = layers_[(size_t) il];
    auto f32 = [&](const char* n, int64_t a, int64_t b = 1) { return upload(tensor(il, n), blk(il, n), a, b, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b, Want w = kAnyMat) {
        return upload(tensor(il, n), blk(il, n), a, b, w);
    };
    auto has = [&](const char* n) { return tensor(il, n) != nullptr; };
    // the low rank of a gate's two-matrix projection is the first matrix's row count
    auto rank = [&](const char* n) {
        const TensorInfo* t = tensor(il, n);
        if (!t || t->shape.size() < 2) throw std::runtime_error("the GGUF has no matrix " + blk(il, n));
        return (int64_t) t->shape[1];
    };
    const int n_mix = kHc * (2 + kHc);

    L.attn_norm = f32("attn_norm.weight", n_embd_);
    L.ffn_norm = f32("ffn_norm.weight", n_embd_);
    if (il < n_layer_) {   // the draft block has one residual stream and adds to it
        L.hc_attn_fn = f32("hc_attn_fn.weight", kHc * n_embd_, n_mix);
        L.hc_attn_scale = f32("hc_attn_scale.weight", 3);
        L.hc_attn_base = f32("hc_attn_base.weight", n_mix);
        L.hc_ffn_fn = f32("hc_ffn_fn.weight", kHc * n_embd_, n_mix);
        L.hc_ffn_scale = f32("hc_ffn_scale.weight", 3);
        L.hc_ffn_base = f32("hc_ffn_base.weight", n_mix);
    }

    L.dsa = has("attn_q_a.weight");
    L.fixed = !L.dsa;   // the sparse attention's launches take the position
    if (L.dsa) {
        L.q_a = mat("attn_q_a.weight", n_embd_, q_rank_);
        L.q_a_norm = f32("attn_q_a_norm.weight", q_rank_);
        L.q_b = mat("attn_q_b.weight", q_rank_, (int64_t) n_head_ * mla_k_);
        L.kv_a = mat("attn_kv_a_mqa.weight", n_embd_, kv_rank_);
        L.kv_a_norm = f32("attn_kv_a_norm.weight", kv_rank_);
        L.k_b = mat("attn_k_b.weight", mla_k_, (int64_t) kv_rank_ * n_head_, kFloatMat);
        L.v_b = mat("attn_v_b.weight", kv_rank_, (int64_t) mla_v_ * n_head_, kFloatMat);
        L.wo = mat("attn_output.weight", (int64_t) n_head_ * mla_v_, n_embd_);
        L.idx_k = mat("indexer.attn_k.weight", n_embd_, idx_dim_);
        L.idx_k_norm = f32("indexer.k_norm.weight", idx_dim_);
        L.idx_k_norm_b = f32("indexer.k_norm.bias", idx_dim_);
        L.idx_q_b = mat("indexer.attn_q_b.weight", q_rank_, (int64_t) idx_dim_ * idx_heads_);
        L.idx_proj = mat("indexer.proj.weight", n_embd_, idx_heads_);
        L.idx_gate = mat("indexer_compressor_gate.weight", n_embd_, idx_dim_);
        L.idx_ape = f32("indexer_compressor_ape.weight", idx_dim_, kpool_);
        ++n_dsa_;
    } else {
        const int64_t fr = rank("ssm_f_a.weight"), gr = rank("ssm_g_a.weight");
        gate_rank_ = std::max<int64_t>({gate_rank_, fr, gr});
        L.wq = mat("attn_q.weight", n_embd_, d_inner_);
        L.wk = mat("attn_k.weight", n_embd_, d_inner_);
        L.wv = mat("attn_v.weight", n_embd_, d_inner_);
        L.wo = mat("attn_output.weight", d_inner_, n_embd_);
        L.conv_q = f32("ssm_conv1d_q.weight", kConv, d_inner_);
        L.conv_k = f32("ssm_conv1d_k.weight", kConv, d_inner_);
        L.conv_v = f32("ssm_conv1d_v.weight", kConv, d_inner_);
        L.f_a = mat("ssm_f_a.weight", n_embd_, fr);
        L.f_b = mat("ssm_f_b.weight", fr, d_inner_);
        L.dt_b = f32("ssm_dt.bias", d_inner_);
        L.beta = mat("ssm_beta.weight", n_embd_, n_head_);
        // kda_step() takes -exp(A_log), which is what one converter stores.  Another stores exp(A_log), and the
        // sign tells those two apart.  The third stores A_log, under a name that says so.
        const TensorInfo* a = tensor(il, "ssm_a");
        float first = -1.f;
        if (a && a->type == kTypeF32 && home(*a).payload(*a) >= sizeof(first))
            std::memcpy(&first, home(*a).data(*a), sizeof(first));
        float (*fix)(float) = nullptr;
        if (a && a->name.find("a_log") != std::string::npos) fix = [](float v) { return -std::exp(v); };
        else if (first > 0.f) fix = [](float v) { return -v; };
        L.a = upload(a, blk(il, "ssm_a"), n_head_, 1, kF32, fix);
        L.g_a = mat("ssm_g_a.weight", n_embd_, gr);
        L.g_b = mat("ssm_g_b.weight", gr, d_inner_);
        L.o_norm = f32("ssm_norm.weight", kda_dim_);
        ++n_kda_;
    }

    L.moe = has("ffn_gate_inp.weight");
    if (!L.moe) {
        L.ffn_gate = mat("ffn_gate.weight", n_embd_, n_ff_dense_);
        L.ffn_up = mat("ffn_up.weight", n_embd_, n_ff_dense_);
        L.ffn_down = mat("ffn_down.weight", n_ff_dense_, n_embd_);
        return;
    }
    L.gate_inp = mat("ffn_gate_inp.weight", n_embd_, n_expert_);
    L.sh_gate = mat("ffn_gate_shexp.weight", n_embd_, n_ff_exp_);
    L.sh_up = mat("ffn_up_shexp.weight", n_embd_, n_ff_exp_);
    L.sh_down = mat("ffn_down_shexp.weight", n_ff_exp_, n_embd_);

    L.probs_b.assign((size_t) n_expert_, 0.f);
    if (const TensorInfo* b = tensor(il, "exp_probs_b.bias")) {
        if (b->type != kTypeF32 || (int64_t) b->elements() != n_expert_)
            throw std::runtime_error(b->name + " is " + shape_of(*b) + ", expected F32 [" +
                                     std::to_string(n_expert_) + "]");
        home(*b).payload(*b);
        std::memcpy(L.probs_b.data(), home(*b).data(*b), (size_t) n_expert_ * sizeof(float));
    }

    // the routed experts stay in the file; fill_expert_cache() copies the ones VRAM has room for
    auto exps = [&](const char* n, int64_t n_in, int64_t rows) -> const TensorInfo& {
        const TensorInfo* t = tensor(il, n);
        if (!t) throw std::runtime_error("the GGUF has no tensor " + blk(il, n));
        if (t->shape.size() != 3 || (int64_t) t->shape[0] != n_in || (int64_t) t->shape[1] != rows ||
            (int64_t) t->shape[2] != n_expert_)
            throw std::runtime_error(t->name + " is " + shape_of(*t) + ", expected [" + std::to_string(n_in) + ", " +
                                     std::to_string(rows) + ", " + std::to_string(n_expert_) + "]");
        return *t;
    };
    const TensorInfo& tg = exps("ffn_gate_exps.weight", n_embd_, n_ff_exp_);
    const TensorInfo& tu = exps("ffn_up_exps.weight", n_embd_, n_ff_exp_);
    const TensorInfo& td = exps("ffn_down_exps.weight", n_ff_exp_, n_embd_);
    if (tg.type != tu.type)
        throw std::runtime_error(tg.name + " and " + tu.name + " have different types, which this engine does not run");
    L.gu_type = tg.type;
    L.d_type = td.type;
    L.gu_bytes = home(tg).payload(tg) / (size_t) n_expert_;
    L.d_bytes = home(td).payload(td) / (size_t) n_expert_;
    home(tu).payload(tu);
    L.exp_gate = home(tg).data(tg);
    L.exp_up = home(tu).data(tu);
    L.exp_down = home(td).data(td);
    std::string err;
    if (!cpu_expert_types_ok((int) L.gu_type, (int) L.d_type, n_embd_, n_ff_exp_, err))
        throw std::runtime_error(blk(il, "ffn_*_exps.weight: ") + err);
    L.slot.assign((size_t) n_expert_, nullptr);
    ++n_moe_;
}

// Per-token buffers hold rows_ tokens, row after row.
void Glm5Next::assign_scratch() {
    const size_t n = (size_t) rows_, e = n * n_embd_, d = n * d_inner_;
    const size_t ff = n * std::max(n_ff_dense_, 2 * n_ff_exp_);
    r_ = fbuf(kHc * e);
    cur_ = fbuf(e);
    xn_ = fbuf(e);
    ao_ = fbuf(e);
    fo_ = fbuf(e);
    mix_ = fbuf(n * (kHc * (2 + kHc) + 1));
    hcw_ = fbuf(n * (kHc + kHc * kHc));

    proj_ = fbuf(3 * d);
    qkv_ = fbuf(3 * d);
    fa_ = fbuf(n * gate_rank_);
    ga_ = fbuf(n * gate_rank_);
    g1_ = fbuf(d);
    g2_ = fbuf(d);
    beta_ = fbuf(n * n_head_);
    gated_ = fbuf(d);

    t1_ = fbuf(n * q_rank_);
    qr_ = fbuf(n * q_rank_);
    q_ = fbuf(n * n_head_ * mla_k_);
    t2_ = fbuf(n * kv_rank_);
    kv_ = fbuf(n * kv_rank_);
    qabs_ = fbuf(n * n_head_ * kv_rank_);
    kqv_ = fbuf(n * n_head_ * kv_rank_);
    vout_ = fbuf(n * n_head_ * mla_v_);
    ikr_ = fbuf(n * idx_dim_);
    ik_ = fbuf(n * idx_dim_);
    ig_ = fbuf(n * idx_dim_);
    iq_ = fbuf(n * idx_heads_ * idx_dim_);
    iw_ = fbuf(n * idx_heads_);
    score_ = fbuf(n * score_stride_);
    const size_t keys = (size_t) std::min(opt_.max_context, idx_top_k_ + kpool_ - 1);
    akeys_ = fbuf(std::min(n, (size_t) kAttnRows) * keys * kv_rank_);
    ascore_ = fbuf(std::min(n, (size_t) kAttnRows) * keys * n_head_);

    g_ = fbuf(ff);   // a routed expert's gate and up rows come out of one launch, back to back
    u_ = fbuf(ff);
    h_ = fbuf(ff);
    d_ = fbuf((size_t) n_embd_);   // one routed expert of one token
    cpu_ = fbuf(e);
    rl_ = fbuf(n * n_expert_);
    logits_ = fbuf((size_t) n_vocab_);

    if (mmq_) {
        const size_t rr = n * n_used_;
        gu_ = fbuf(rr * 2 * n_ff_exp_);
        hx_ = fbuf(rr * n_ff_exp_);
        dx_ = fbuf(rr * n_embd_);
    }
}

void Glm5Next::alloc_state() {
    // --dump-layers prints one token's stages
    rows_ = opt_.dump ? 1 : std::min(opt_.prompt_batch, opt_.max_context);
    const size_t conv_n = size_t(3) * d_inner_ * (kConv - 1), s_n = size_t(n_head_) * kda_dim_ * kda_dim_;
    kda_n_ = size_t(n_kda_) * (conv_n + s_n);
    state_n_ = kda_n_ + (draft_ ? (size_t) n_embd_ : 0);
    state_ = (float*) dmalloc((state_n_ + (draft_ ? (size_t) rows_ * n_embd_ : 0)) * sizeof(float),
                              "the recurrent state");
    if (draft_) {
        hn_ = state_ + kda_n_;
        h_dlogits_ = (float*) hmalloc((size_t) n_vocab_ * sizeof(float), "pinned memory");
    }
    // window(): the state after each of its tokens but the last, for keep(), and every token's logits
    if (opt_.window > 1)
        snap_ = (float*) dmalloc(size_t(opt_.window - 1) * kda_n_ * sizeof(float), "the recurrent state");
    wlogits_ = (float*) dmalloc(size_t(opt_.window) * n_vocab_ * sizeof(float), "scratch");
    h_wlogits_ = (float*) hmalloc(size_t(opt_.window) * n_vocab_ * sizeof(float), "pinned memory");

    const size_t ctx = (size_t) opt_.max_context, lat_n = ctx * kv_rank_, idx_n = ctx * 3 * idx_dim_;
    uint16_t* lat = (uint16_t*) dmalloc(n_dsa_ * lat_n * sizeof(uint16_t), "the attention cache");
    float* idx = (float*) dmalloc(n_dsa_ * idx_n * sizeof(float), "the indexer cache");
    float* st = state_;
    for (Layer& L : layers_) {
        if (L.dsa) {
            L.lat = lat;
            lat += lat_n;
            L.idx = idx;
            idx += idx_n;
        } else {
            L.conv = st;
            st += conv_n;
            L.s = st;
            st += s_n;
        }
    }

    score_stride_ = opt_.max_context / kpool_ + 1;
    sel_stride_ = idx_top_k_ + kpool_;
    const size_t n = (size_t) rows_;

    // A prompt batch runs its routed experts through MMQ when the card has a tile for each of their types.
    // The model's layers: the draft block's experts never run a batch at a time.
    bool tiles = opt_.gpu_experts && rows_ >= kStageRows && n_moe_ > 0;
    size_t one_expert = 0;
    for (int il = 0; il < n_layer_; ++il) {
        const Layer& L = layers_[(size_t) il];
        if (!L.moe) continue;
        tiles = tiles && Mmq::fits((int) L.gu_type, 2 * n_ff_exp_, n_embd_) &&
                Mmq::fits((int) L.d_type, n_embd_, n_ff_exp_);
        one_expert = std::max(one_expert, 2 * L.gu_bytes + L.d_bytes);
    }
    if (tiles) mmq_ = std::make_unique<Mmq>();

    assign_scratch();   // the first pass only counts
    scratch_ = (float*) dmalloc(scratch_at_ * sizeof(float), "scratch");
    scratch_at_ = 0;
    assign_scratch();

    const int q_max = std::max({n_embd_, d_inner_, n_head_ * mla_v_, n_head_ * mla_k_, q_rank_, kv_rank_,
                                n_ff_dense_, n_ff_exp_, (int) gate_rank_, 256});
    xq_ = dmalloc(n * kernels::native_q8_1_bytes(n_embd_), "scratch");
    q8_ = dmalloc(n * kernels::native_q8_1_bytes((q_max + 255) / 256 * 256), "scratch");
    sel_ = (int32_t*) dmalloc(n * sel_stride_ * sizeof(int32_t), "scratch");

    h_r_ = (float*) hmalloc(n * kHc * n_embd_ * sizeof(float), "pinned memory");
    h_rl_ = (float*) hmalloc(n * n_expert_ * sizeof(float), "pinned memory");
    h_x_ = (float*) hmalloc(n * n_embd_ * sizeof(float), "pinned memory");
    h_cpu_ = (float*) hmalloc(n * n_embd_ * sizeof(float), "pinned memory");
    h_logits_ = (float*) hmalloc((size_t) n_vocab_ * sizeof(float), "pinned memory");
    h_score_ = (float*) hmalloc(n * score_stride_ * sizeof(float), "pinned memory");
    h_sel_ = (int32_t*) hmalloc(n * sel_stride_ * sizeof(int32_t), "pinned memory");

    if (mmq_) {
        const size_t rr = n * n_used_, tab = (3 * rr + 3 * (size_t) n_expert_ + 3) * sizeof(int32_t);
        mq_ = dmalloc(Mmq::q8_bytes((int64_t) rr, n_embd_), "scratch");
        mq2_ = dmalloc(Mmq::q8_bytes((int64_t) rr, n_ff_exp_), "scratch");
        mqx_ = dmalloc(Mmq::q8_bytes(rows_, q_max), "scratch");
        tab_ = (int32_t*) dmalloc(tab, "scratch");
        h_tab_ = (int32_t*) hmalloc(tab, "pinned memory");
        iota_ = (int32_t*) dmalloc(rr * sizeof(int32_t), "scratch");
        std::vector<int32_t> ramp(rr);
        std::iota(ramp.begin(), ramp.end(), 0);
        ck(cudaMemcpy(iota_, ramp.data(), rr * sizeof(int32_t), cudaMemcpyHostToDevice), "scratch");

        stage_bytes_ = std::max((size_t) opt_.stage_mib << 20, one_expert);
        chunk_bytes_ = std::max(kChunkBytes, one_expert);
        stage_ = (uint8_t*) dmalloc(stage_bytes_ + 256, "the expert staging buffer");
        for (int b = 0; b < 2; ++b) h_stage_[b] = (uint8_t*) hmalloc(chunk_bytes_, "pinned memory");
        for (void** e : {&ev_stage_[0], &ev_stage_[1], &ev_ahead_, &ev_free_}) {
            cudaEvent_t ev = nullptr;
            ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");
            *e = ev;
        }
        cudaStream_t up = nullptr;
        ck(cudaStreamCreateWithFlags(&up, cudaStreamNonBlocking), "cudaStreamCreate");
        up_ = up;
        ids_.resize(rr);
        first_.resize((size_t) n_expert_);
    }

    probs_.resize((size_t) n_expert_);
    pick_.resize((size_t) n_expert_);
    use_.assign(layers_.size() * n_expert_, 0);
}

size_t Glm5Next::row_bytes() const {
    return (size_t) n_dsa_ * ((size_t) kv_rank_ * sizeof(uint16_t) + (size_t) 3 * idx_dim_ * sizeof(float));
}

void Glm5Next::save_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context)
        throw std::logic_error("save_rows: the rows are outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    const size_t lat = (size_t) kv_rank_ * sizeof(uint16_t), idx = (size_t) 3 * idx_dim_ * sizeof(float);
    const size_t at = (size_t) first, rows = (size_t) n;
    for (const Layer& L : layers_) {
        if (!L.dsa || rows == 0) continue;
        ck(cudaMemcpy(dst, (const uint8_t*) L.lat + at * lat, rows * lat, cudaMemcpyDeviceToHost), "save rows");
        dst += rows * lat;
        ck(cudaMemcpy(dst, (const uint8_t*) L.idx + at * idx, rows * idx, cudaMemcpyDeviceToHost), "save rows");
        dst += rows * idx;
    }
}

void Glm5Next::load_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: the rows are outside what was saved");
    ck(cudaStreamSynchronize(cs(st_)), "load rows");
    const size_t lat = (size_t) kv_rank_ * sizeof(uint16_t), idx = (size_t) 3 * idx_dim_ * sizeof(float);
    const size_t at = (size_t) first, rows = (size_t) n, skip = (size_t) (first - from), all = (size_t) count;
    for (const Layer& L : layers_) {
        if (!L.dsa || rows == 0) continue;
        ck(cudaMemcpy((uint8_t*) L.lat + at * lat, src + skip * lat, rows * lat, cudaMemcpyHostToDevice), "load rows");
        src += all * lat;
        ck(cudaMemcpy((uint8_t*) L.idx + at * idx, src + skip * idx, rows * idx, cudaMemcpyHostToDevice), "load rows");
        src += all * idx;
    }
}

void Glm5Next::kda(Layer& L, int n) {
    const size_t d = (size_t) n * d_inner_;
    lin(L.wq, xn_, proj_, n, xq_);
    lin(L.wk, xn_, proj_ + d, n, xq_);
    lin(L.wv, xn_, proj_ + 2 * d, n, xq_);
    kda_conv(proj_, L.conv, fp(L.conv_q), fp(L.conv_k), fp(L.conv_v), qkv_, d_inner_, n, st_,
             n_snap_ > 0 ? snap_ + (L.conv - state_) : nullptr, kda_n_, n_snap_);

    lin(L.f_a, xn_, fa_, n, xq_);
    lin(L.f_b, fa_, g1_, n);
    add_bias(fp(L.dt_b), g1_, d_inner_, n, st_);
    lin(L.beta, xn_, beta_, n, xq_);
    lin(L.g_a, xn_, ga_, n, xq_);
    lin(L.g_b, ga_, g2_, n);
    kda_step(qkv_, g1_, fp(L.a), beta_, L.s, g2_, fp(L.o_norm), gated_, kda_dim_, n_head_, n, kda_lb_, rms_eps_,
             st_, n_snap_ > 0 ? snap_ + (L.s - state_) : nullptr, kda_n_, n_snap_);
    lin(L.wo, gated_, ao_, n);
}

// For the n tokens from row `first` of a call whose first token is at `at`: the n_top pools with the highest
// indexer score, expanded to their rows, then the tail that is in no pool yet.  The scores come to the host for
// the selection, which costs a wait per layer; it only runs past top_k tokens.
void Glm5Next::dsa_select(Layer& L, int64_t at, int first, int n) {
    const int64_t pos = at + first;
    const int n_top = idx_top_k_ / kpool_;
    lin(L.idx_q_b, qr_ + (size_t) first * q_rank_, iq_, n);
    lin(L.idx_proj, xn_ + (size_t) first * n_embd_, iw_, n,
        (const uint8_t*) xq_ + (size_t) first * kernels::native_q8_1_bytes(n_embd_));
    idx_score(iq_, iw_, L.idx, score_, score_stride_, idx_dim_, idx_heads_, kpool_, pos, n, st_);
    ck(cudaMemcpyAsync(h_score_, score_, (size_t) n * score_stride_ * sizeof(float), cudaMemcpyDeviceToHost,
                       cs(st_)),
       "indexer scores");
    ck(cudaStreamSynchronize(cs(st_)), "indexer scores");

    workers_->run(n, [&](int t) {
        static thread_local std::vector<int32_t> order;
        const int64_t at = pos + t, n_pool = (at + 1) / kpool_;
        order.resize((size_t) n_pool);
        std::iota(order.begin(), order.end(), 0);
        const float* sc = h_score_ + (size_t) t * score_stride_;
        std::partial_sort(order.begin(), order.begin() + n_top, order.end(),
                          [sc](int32_t a, int32_t b) { return sc[a] != sc[b] ? sc[a] > sc[b] : a < b; });
        int32_t* sel = h_sel_ + (size_t) t * sel_stride_;
        for (int i = 0; i < n_top; ++i)
            for (int k = 0; k < kpool_; ++k) *sel++ = order[(size_t) i] * kpool_ + k;
        for (int64_t p = n_pool * kpool_; p <= at; ++p) *sel++ = (int32_t) p;
    });
    ck(cudaMemcpyAsync(sel_, h_sel_, (size_t) n * sel_stride_ * sizeof(int32_t), cudaMemcpyHostToDevice, cs(st_)),
       "indexer selection");
}

// The rows n tokens from position `pos` leave in a sparse-attention layer's caches: the latent the attention
// reads, and the indexer's key and gate.  xn_ and xq_ hold the layer's input.
void Glm5Next::dsa_store(Layer& L, int64_t pos, int n) {
    lin(L.kv_a, xn_, t2_, n, xq_);
    rms_norm(t2_, fp(L.kv_a_norm), kv_, kv_rank_, n, rms_eps_, st_);
    store_f16(kv_, L.lat, kv_rank_, pos, n, st_);

    lin(L.idx_k, xn_, ikr_, n, xq_);
    layer_norm(ikr_, fp(L.idx_k_norm), fp(L.idx_k_norm_b), ik_, idx_dim_, n, norm_eps_, st_);
    lin(L.idx_gate, xn_, ig_, n, xq_);
    idx_store(ik_, ig_, L.idx, idx_dim_, pos, n, st_);
    idx_pool(L.idx, fp(L.idx_ape), idx_dim_, kpool_, pos, n, st_);
}

void Glm5Next::dsa(Layer& L, int64_t pos, int n) {
    lin(L.q_a, xn_, t1_, n, xq_);
    rms_norm(t1_, fp(L.q_a_norm), qr_, q_rank_, n, rms_eps_, st_);
    lin(L.q_b, qr_, q_, n);
    lin(L.k_b, q_, qabs_, n, nullptr, kv_rank_);   // the key projection, absorbed into the query: per head

    dsa_store(L, pos, n);

    // While a token's pools all fit in top_k, every one is selected and its attention is plain causal attention.
    // That holds up to a position, so the tokens of a call are dense first and sparse after.
    const int64_t last_dense = (int64_t) idx_top_k_ + kpool_ - 2;
    const int dense = (int) std::clamp<int64_t>(last_dense - pos + 1, 0, n);
    if (dense < n) dsa_select(L, pos, dense, n - dense);
    // A prompt batch's tokens share their products, padded to the token with the most keys.  A few tokens get a
    // product each: a token of a window must come out as it would alone, to the last bit, and cuBLAS rounds a
    // product of another size another way.
    const int together = n < kStageRows ? 1 : std::min(rows_, kAttnRows);
    dsa_attend(blas_, qabs_, L.lat, sel_, sel_stride_, idx_top_k_, kpool_, pos, n, dense, akeys_, ascore_,
               together, kqv_, kv_rank_, n_head_, 1.f / std::sqrt((float) mla_k_), st_);
    lin(L.v_b, kqv_, vout_, n, nullptr, mla_v_);
    lin(L.wo, vout_, ao_, n);
}

void Glm5Next::dense_ffn(Layer& L, int il, int n) {
    lin(L.ffn_gate, xn_, g_, n, xq_);
    lin(L.ffn_up, xn_, u_, n, xq_);
    swiglu_clamp(g_, u_, h_, n * n_ff_dense_, clamp_shexp_[(size_t) il], st_);
    lin(L.ffn_down, h_, fo_, n);
}

// What ggml's MoE graph does for this model: sigmoid gates, the n_used largest of gate + bias, and weights from
// the bare gates.
void Glm5Next::route(const Layer& L, const float* logits, int* ids, float* w) {
    for (int e = 0; e < n_expert_; ++e) {
        probs_[(size_t) e] = 1.f / (1.f + std::exp(-logits[e]));
        pick_[(size_t) e] = probs_[(size_t) e] + L.probs_b[(size_t) e];
    }
    float sum = 0.f;
    for (int k = 0; k < n_used_; ++k) {
        int best = 0;
        for (int e = 1; e < n_expert_; ++e)
            if (pick_[(size_t) e] > pick_[(size_t) best]) best = e;
        pick_[(size_t) best] = -INFINITY;
        ids[k] = best;
        w[k] = probs_[(size_t) best];
        sum += w[k];
    }
    const float denom = w_norm_ ? std::max(sum, kMinWeightSum) : 1.f;
    for (int k = 0; k < n_used_; ++k) {
        w[k] /= denom;
        if (w_scale_ != 0.f) w[k] *= w_scale_;
    }
}

// The shared expert.  It runs on the GPU while the host routes.
void Glm5Next::moe_shared(const Layer& L, int il, int n) {
    lin(L.sh_gate, xn_, g_, n, xq_);
    lin(L.sh_up, xn_, u_, n, xq_);
    swiglu_clamp(g_, u_, h_, n * n_ff_exp_, clamp_shexp_[(size_t) il], st_);
    lin(L.sh_down, h_, fo_, n);
}

// A layer from its input to its router's copy (a dense layer: to its end).  A prompt batch's MoE (moe_rows) does
// its own routing.
void Glm5Next::layer_head(Layer& L, int il, int n) {
    hc_pre(r_, fp(L.hc_attn_fn), fp(L.hc_attn_scale), fp(L.hc_attn_base), mix_, hcw_, cur_, n_embd_, n, rms_eps_,
           hc_eps_, sinkhorn_, st_);
    rms_norm(cur_, fp(L.attn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, n_embd_ * n, 1, st_);
    dbg("attn_in", il, xn_, n_embd_);
    if (L.dsa) dsa(L, pos_, n);
    else kda(L, n);
    dbg("attn", il, ao_, n_embd_);
    hc_post(ao_, hcw_, r_, n_embd_, n, st_);

    hc_pre(r_, fp(L.hc_ffn_fn), fp(L.hc_ffn_scale), fp(L.hc_ffn_base), mix_, hcw_, cur_, n_embd_, n, rms_eps_,
           hc_eps_, sinkhorn_, st_);
    rms_norm(cur_, fp(L.ffn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, n_embd_ * n, 1, st_);
    dbg("ffn_in", il, xn_, n_embd_);
    if (!L.moe) {
        dense_ffn(L, il, n);
        dbg("ffn", il, fo_, n_embd_);
        hc_post(fo_, hcw_, r_, n_embd_, n, st_);
    } else if (!(mmq_ && n >= kStageRows)) {
        moe_router(L, n);
    }
}

// The draft block's input for n tokens: each token's embedding and the model's output for the token before it
// (h, a row a token), each normed, the two side by side through the block's projection.  Returns the block's
// residual, and leaves its normed attention input in xn_ and xq_.  r_ is free between the model's calls, and the
// block works in it: n rows of the two halves, then the residual.
//
// h_r_ is free to write: every call that copied out of it has waited for the card since.
float* Glm5Next::draft_input(const int32_t* tokens, const float* h, int n) {
    const size_t e = (size_t) n_embd_, rows = (size_t) n;
    for (int t = 0; t < n; ++t) embed(tokens[t], h_r_ + (size_t) t * e);
    ck(cudaMemcpyAsync(cur_, h_r_, rows * e * sizeof(float), cudaMemcpyHostToDevice, cs(st_)), "embed");
    rms_norm(cur_, fp(d_enorm_), r_, n_embd_, n, rms_eps_, st_, 2 * n_embd_);
    rms_norm(h, fp(d_hnorm_), r_ + e, n_embd_, n, rms_eps_, st_, 2 * n_embd_);
    float* x = r_ + 2 * rows * e;
    lin(d_proj_, r_, x, n);
    rms_norm(x, fp(layers_[(size_t) n_layer_].attn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, n_embd_ * n, 1, st_);
    return x;
}

// The rows the draft block's attention keeps for tokens [first, n) of a call that has run: token i is at pos() + i,
// and hn_'s row i is the model's output for the token before it.  A row is all a later guess reads of a token, so
// the rest of the block is not computed for it.  Then the call's last output is the one the next token pairs with.
void Glm5Next::draft_rows(const int32_t* tokens, int first, int n) {
    const size_t e = (size_t) n_embd_;
    if (first < n) {
        draft_input(tokens + first, hn_ + (size_t) first * e, n - first);
        dsa_store(layers_[(size_t) n_layer_], pos_ + first, n - first);
    }
    ck(cudaMemcpyAsync(hn_, hn_ + (size_t) n * e, e * sizeof(float), cudaMemcpyDeviceToDevice, cs(st_)), "draft");
    ahead_pos_ = -1;
}

// One token through the draft block, at pos() + i.  The first token's row is then what draft_rows() would leave
// for it, and eval() and keep() skip it; the rows of the guesses after it are written again when the positions get
// their real tokens.
int32_t Glm5Next::draft(int32_t token, int i, float* prob) {
    if (!draft_) throw std::logic_error("draft: this model has no block that guesses ahead");
    if (token < 0 || token >= n_vocab_) throw std::runtime_error("a token id is outside the vocabulary");
    if (i < 0 || pos_ + i >= opt_.max_context) throw std::logic_error("draft: the position is outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "draft");
    Layer& L = layers_[(size_t) n_layer_];
    float* own = hn_ + n_embd_;   // the block's output, which stands in for the model's from the second guess on
    float* x = draft_input(&token, i == 0 ? hn_ : own, 1);
    dsa(L, pos_ + i, 1);
    axpy(ao_, 1.f, x, n_embd_, true, st_);
    rms_norm(x, fp(L.ffn_norm), xn_, n_embd_, 1, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, n_embd_, 1, st_);
    moe_router(L, 1);
    ck(cudaEventRecord((cudaEvent_t) ev_, cs(st_)), "router");
    moe_shared(L, n_layer_, 1);
    moe(L, n_layer_, 1);
    axpy(fo_, 1.f, x, n_embd_, true, st_);
    rms_norm(x, fp(d_head_), own, n_embd_, 1, rms_eps_, st_);
    lin(output_, own, logits_, 1);
    ck(cudaMemcpyAsync(h_dlogits_, logits_, (size_t) n_vocab_ * sizeof(float), cudaMemcpyDeviceToHost, cs(st_)),
       "draft");
    ck(cudaStreamSynchronize(cs(st_)), "draft");
    if (i == 0) {
        ahead_pos_ = pos_;
        ahead_tok_ = token;
    }

    const float* l = h_dlogits_;
    int best = 0;
    for (int v = 1; v < n_vocab_; ++v)
        if (l[v] > l[best]) best = v;
    // the softmax's sum without the tokens too unlikely to change it
    double sum = 0.0;
    for (int v = 0; v < n_vocab_; ++v)
        if (l[v] > l[best] - 20.f) sum += std::exp((double) (l[v] - l[best]));
    *prob = (float) (1.0 / sum);
    return best;
}

}  // namespace strata::glm
