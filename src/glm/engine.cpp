// src/glm/engine.cpp - see the header.  The reference for the math is llama.cpp's glm5-next graph.
#include "engine.hpp"

#include "mmq.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/glm/kernels.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <thread>

#ifdef __linux__
#include <sched.h>
#endif

namespace strata::glm {
namespace {

constexpr uint32_t kTypeF32 = 0, kTypeQ8_0 = 8, kTypeBF16 = 30;
constexpr int kHc = 4, kConv = 4;   // what kernels.cu is written for; checked at load
constexpr int kMaxUsed = 32;
constexpr int kRows = 512;          // prompt tokens one eval() takes; the scratch is about 1 MiB of VRAM a token
// From this many tokens, a call runs its routed experts on the GPU alone and copies the ones that are not cached
// through VRAM.  Below it, that copy costs more than the CPU takes to compute the few experts.
constexpr int kStageRows = 32;
// From this many tokens, a call routes to most of a layer's experts, and the ones that are not cached start their
// copy before the layer has routed.  The staging buffer holds about what a layer of Q2_K experts leaves out of
// the cache; the VRAM it takes is cache that decoding does without.
constexpr int kAheadRows = 256;
constexpr size_t kStageBytes = size_t(2) << 30;
constexpr size_t kChunkBytes = size_t(512) << 20;   // one upload, and each of the two pinned buffers
constexpr float kMinWeightSum = 6.103515625e-5f;   // the floor ggml's MoE graph puts under the gate sum

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

size_t pad(size_t n) { return (n + 255) & ~size_t(255); }

bool ends_with(const std::string& s, const char* tail) {
    const size_t n = std::strlen(tail);
    return s.size() >= n && s.compare(s.size() - n, n, tail) == 0;
}

std::string blk(int il, const char* name) { return "blk." + std::to_string(il) + "." + name; }

const float* fp(const Dev& d) { return (const float*) d.p; }
cudaStream_t cs(void* s) { return (cudaStream_t) s; }

std::string shape_of(const TensorInfo& t) {
    std::string s = "[";
    for (size_t d = 0; d < t.shape.size(); ++d) s += (d ? ", " : "") + std::to_string(t.shape[d]);
    return s + "] " + t.type_name();
}

// A tensor's byte count, after checking that the file holds that much.
size_t payload(const GgufFile& g, const TensorInfo& t) {
    const uint64_t bytes = tensor_payload_bytes(t), room = g.file_size() - g.data_start();
    if (bytes == 0 || t.offset > room || bytes > room - t.offset)
        throw std::runtime_error("tensor " + t.name + " " + shape_of(t) + " does not fit its file");
    return (size_t) bytes;
}

const MetaValue& meta(const GgufFile& g, const char* key) {
    const MetaValue* v = g.get(std::string("glm5-next.") + key);
    if (!v) throw std::runtime_error(std::string("the GGUF has no glm5-next.") + key);
    return *v;
}

int meta_i(const GgufFile& g, const char* key) { return (int) meta(g, key).u; }
float meta_f(const GgufFile& g, const char* key) { return (float) meta(g, key).f; }

// A SwiGLU clamp limit per layer, stored as one value or one per layer.  Absent or zero means no clamp.
std::vector<float> clamp_limits(const GgufFile& g, const char* key, int n_layer) {
    std::vector<float> out((size_t) n_layer, INFINITY);
    const MetaValue* v = g.get(std::string("glm5-next.") + key);
    if (!v) return out;
    for (int il = 0; il < n_layer; ++il) {
        double x = v->f;
        if (v->type == MetaType::ARRAY) {
            if ((size_t) il >= v->items.size())
                throw std::runtime_error(std::string("glm5-next.") + key + " has no value for layer " +
                                         std::to_string(il));
            x = v->items[(size_t) il].f;
        }
        if (x > 1e-6) out[(size_t) il] = (float) x;
    }
    return out;
}

// A container's CPU quota is not in the affinity mask: a 25-CPU pod on a 208-thread host shows 208.
int usable_cpus() {
    long long n = (long long) std::thread::hardware_concurrency();
#ifdef __linux__
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) == 0) n = CPU_COUNT(&set);
    long long quota = -1, period = 0;
    if (FILE* f = std::fopen("/sys/fs/cgroup/cpu.max", "r")) {   // "max 100000" when there is no quota
        if (std::fscanf(f, "%lld %lld", &quota, &period) != 2) quota = -1;
        std::fclose(f);
    } else if (FILE* q = std::fopen("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r")) {
        if (std::fscanf(q, "%lld", &quota) != 1) quota = -1;
        std::fclose(q);
        if (FILE* p = std::fopen("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r")) {
            if (std::fscanf(p, "%lld", &period) != 1) period = 0;
            std::fclose(p);
        }
    }
    if (quota > 0 && period > 0) n = std::min(n, std::max(1LL, quota / period));
#endif
    return (int) std::max(n, 1LL);
}

// Lines of "layer expert count".  A missing file is not an error: the first run writes it.
void read_profile(const std::string& path, int n_layer, int n_expert, std::vector<int64_t>& out) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return;
    out.assign((size_t) n_layer * n_expert, 0);
    int il = 0, e = 0;
    long long n = 0;
    while (std::fscanf(f, "%d %d %lld", &il, &e, &n) == 3)
        if (il >= 0 && il < n_layer && e >= 0 && e < n_expert && n > 0) out[(size_t) il * n_expert + e] = n;
    std::fclose(f);
}

}  // namespace

Engine::Engine() = default;

Engine::~Engine() {
    if (up_) cudaStreamSynchronize(cs(up_));
    if (st_) cudaStreamSynchronize(cs(st_));
    workers_.reset();
    mmq_.reset();
    for (void* p : dev_allocs_) cudaFree(p);
    for (void* p : host_allocs_) cudaFreeHost(p);
    for (void* e : {ev_stage_[0], ev_stage_[1], ev_ahead_, ev_free_, ev_})
        if (e) cudaEventDestroy((cudaEvent_t) e);
    if (up_) cudaStreamDestroy(cs(up_));
    if (st_) cudaStreamDestroy(cs(st_));
}

void* Engine::dmalloc(size_t bytes, const char* what) {
    void* p = nullptr;
    const cudaError_t e = cudaMalloc(&p, std::max<size_t>(bytes, 256));
    if (e != cudaSuccess)
        throw std::runtime_error("no VRAM for " + std::string(what) + " (" + std::to_string(bytes >> 20) +
                                 " MiB): " + cudaGetErrorString(e));
    dev_allocs_.push_back(p);
    return p;
}

void* Engine::hmalloc(size_t bytes, const char* what) {
    void* p = nullptr;
    ck(cudaMallocHost(&p, std::max<size_t>(bytes, 256)), what);
    host_allocs_.push_back(p);
    return p;
}

float* Engine::fbuf(size_t n) {
    float* p = scratch_ ? scratch_ + scratch_at_ : nullptr;
    scratch_at_ += (n + 63) & ~size_t(63);
    return p;
}

int64_t Engine::meta_u(const char* key, int64_t fallback) const {
    const MetaValue* v = gg_ ? gg_->get(key) : nullptr;
    return v && v->is_num() ? (int64_t) v->u : fallback;
}

void Engine::load_geometry() {
    const GgufFile& g = *gg_;
    const MetaValue* arch = g.get("general.architecture");
    if (!arch || arch->s != "glm5-next")
        throw std::runtime_error("general.architecture is '" + (arch ? arch->s : std::string()) +
                                 "', this engine runs 'glm5-next'");
    n_layer_ = meta_i(g, "block_count");
    n_embd_ = meta_i(g, "embedding_length");
    n_ff_dense_ = meta_i(g, "feed_forward_length");
    n_head_ = meta_i(g, "attention.head_count");
    n_vocab_ = meta_i(g, "vocab_size");
    rms_eps_ = meta_f(g, "attention.layer_norm_rms_epsilon");
    norm_eps_ = meta_f(g, "attention.layer_norm_epsilon");
    n_expert_ = meta_i(g, "expert_count");
    n_used_ = meta_i(g, "expert_used_count");
    n_ff_exp_ = meta_i(g, "expert_feed_forward_length");
    w_scale_ = meta_f(g, "expert_weights_scale");
    w_norm_ = meta(g, "expert_weights_norm").u != 0;
    kda_dim_ = meta_i(g, "kda.head_dim");
    kda_lb_ = meta_f(g, "kda.gate_lower_bound");
    q_rank_ = meta_i(g, "attention.q_lora_rank");
    kv_rank_ = meta_i(g, "attention.kv_lora_rank");
    mla_k_ = meta_i(g, "attention.key_length_mla");
    mla_v_ = meta_i(g, "attention.value_length_mla");
    idx_heads_ = meta_i(g, "attention.indexer.head_count");
    idx_dim_ = meta_i(g, "attention.indexer.key_length");
    idx_top_k_ = meta_i(g, "attention.indexer.top_k");
    kpool_ = meta_i(g, "attention.indexer.kpool");
    sinkhorn_ = meta_i(g, "hyper_connection.sinkhorn_iterations");
    hc_eps_ = meta_f(g, "hyper_connection.epsilon");
    d_inner_ = n_head_ * kda_dim_;

    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported model: ") + what);
    };
    need(meta_i(g, "hyper_connection.count") == kHc, "hyper_connection.count is not 4");
    need(meta_i(g, "ssm.conv_kernel") == kConv, "ssm.conv_kernel is not 4");
    need(meta_i(g, "expert_gating_func") == 2, "the expert gate is not a sigmoid");
    need(meta_i(g, "expert_shared_count") == 1, "expert_shared_count is not 1");
    need(n_layer_ > 0 && n_embd_ > 0 && n_embd_ % 32 == 0 && n_vocab_ > 0, "bad block_count or embedding_length");
    need(kda_dim_ > 0 && kda_dim_ % 32 == 0 && kda_dim_ <= 256, "kda.head_dim is not a multiple of 32 up to 256");
    need(idx_dim_ > 0 && idx_dim_ <= 256, "the indexer key is longer than 256");
    need(n_used_ >= 1 && n_used_ <= kMaxUsed && n_used_ <= n_expert_, "bad expert_used_count");
    need(kpool_ > 1 && idx_top_k_ > 0 && idx_top_k_ % kpool_ == 0, "indexer.top_k is not whole pools");

    clamp_exp_ = clamp_limits(g, "swiglu_clamp_exp", n_layer_);
    clamp_shexp_ = g.get("glm5-next.swiglu_clamp_shexp") ? clamp_limits(g, "swiglu_clamp_shexp", n_layer_) : clamp_exp_;
}

Dev Engine::upload(const std::string& name, int64_t n_in, int64_t n_out, Want want) {
    const TensorInfo* t = gg_->find(name);
    if (!t) throw std::runtime_error("the GGUF has no tensor " + name);
    const uint64_t rest = t->elements() / std::max<uint64_t>(t->shape[0], 1);
    const bool type_ok = t->type == kTypeF32 || (want != kF32 && t->type == kTypeBF16) ||
                         (want == kAnyMat && kernels::native_mmvq_supported((int) t->type));
    if ((int64_t) t->shape[0] != n_in || (int64_t) rest != n_out || !type_ok)
        throw std::runtime_error(name + " is " + shape_of(*t) + ", expected [" + std::to_string(n_in) + ", " +
                                 std::to_string(n_out) + "]" + (type_ok ? "" : " in a type this engine computes"));
    const size_t bytes = payload(*gg_, *t);
    if (dense_at_ + pad(bytes) > dense_bytes_) throw std::logic_error("the dense arena is too small for " + name);
    uint8_t* p = dense_ + dense_at_;
    dense_at_ += pad(bytes);
    ck(cudaMemcpy(p, gg_->tensor_data(*t), bytes, cudaMemcpyHostToDevice), "weight upload");
    return Dev{p, t->type, (int) n_in, (int) n_out};
}

void Engine::load_layer(int il) {
    Layer& L = layers_[(size_t) il];
    auto f32 = [&](const char* n, int64_t a, int64_t b = 1) { return upload(blk(il, n), a, b, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b, Want w = kAnyMat) { return upload(blk(il, n), a, b, w); };
    auto has = [&](const char* n) { return gg_->find(blk(il, n)) != nullptr; };
    // the low rank of a gate's two-matrix projection is the first matrix's row count
    auto rank = [&](const char* n) {
        const TensorInfo* t = gg_->find(blk(il, n));
        if (!t || t->shape.size() < 2) throw std::runtime_error("the GGUF has no matrix " + blk(il, n));
        return (int64_t) t->shape[1];
    };
    const int n_mix = kHc * (2 + kHc);

    L.attn_norm = f32("attn_norm.weight", n_embd_);
    L.ffn_norm = f32("ffn_norm.weight", n_embd_);
    L.hc_attn_fn = f32("hc_attn_fn.weight", kHc * n_embd_, n_mix);
    L.hc_attn_scale = f32("hc_attn_scale.weight", 3);
    L.hc_attn_base = f32("hc_attn_base.weight", n_mix);
    L.hc_ffn_fn = f32("hc_ffn_fn.weight", kHc * n_embd_, n_mix);
    L.hc_ffn_scale = f32("hc_ffn_scale.weight", 3);
    L.hc_ffn_base = f32("hc_ffn_base.weight", n_mix);

    L.dsa = has("attn_q_a.weight");
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
        L.a = f32("ssm_a", n_head_);
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
    if (const TensorInfo* b = gg_->find(blk(il, "exp_probs_b.bias"))) {
        if (b->type != kTypeF32 || (int64_t) b->elements() != n_expert_)
            throw std::runtime_error(b->name + " is " + shape_of(*b) + ", expected F32 [" +
                                     std::to_string(n_expert_) + "]");
        payload(*gg_, *b);
        std::memcpy(L.probs_b.data(), gg_->tensor_data(*b), (size_t) n_expert_ * sizeof(float));
    }

    // the routed experts stay in the file; fill_expert_cache() copies the ones VRAM has room for
    auto exps = [&](const char* n, int64_t n_in, int64_t rows) -> const TensorInfo& {
        const TensorInfo* t = gg_->find(blk(il, n));
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
    L.gu_bytes = payload(*gg_, tg) / (size_t) n_expert_;
    L.d_bytes = payload(*gg_, td) / (size_t) n_expert_;
    payload(*gg_, tu);
    L.exp_gate = gg_->tensor_data(tg);
    L.exp_up = gg_->tensor_data(tu);
    L.exp_down = gg_->tensor_data(td);
    std::string err;
    if (!cpu_expert_types_ok((int) L.gu_type, (int) L.d_type, n_embd_, n_ff_exp_, err))
        throw std::runtime_error(blk(il, "ffn_*_exps.weight: ") + err);
    L.slot.assign((size_t) n_expert_, nullptr);
    ++n_moe_;
}

// Per-token buffers hold rows_ tokens, row after row.
void Engine::assign_scratch() {
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

void Engine::alloc_state() {
    const size_t conv_n = size_t(3) * d_inner_ * (kConv - 1), s_n = size_t(n_head_) * kda_dim_ * kda_dim_;
    state_n_ = size_t(n_kda_) * (conv_n + s_n);
    state_ = (float*) dmalloc(state_n_ * sizeof(float), "the recurrent state");
    state_ck_ = (float*) dmalloc(state_n_ * sizeof(float), "the recurrent state's checkpoint");

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

    // --dump-layers prints one token's stages
    rows_ = opt_.dump ? 1 : std::min(kRows, opt_.max_context);
    score_stride_ = opt_.max_context / kpool_ + 1;
    sel_stride_ = idx_top_k_ + kpool_;
    const size_t n = (size_t) rows_;

    // A prompt batch runs its routed experts through MMQ when the card has a tile for each of their types.
    bool tiles = opt_.gpu_experts && rows_ >= kStageRows && n_moe_ > 0;
    size_t one_expert = 0;
    for (const Layer& L : layers_) {
        if (!L.moe) continue;
        tiles = tiles && Mmq::fits((int) L.gu_type, 2 * n_ff_exp_) && Mmq::fits((int) L.d_type, n_embd_);
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

        stage_bytes_ = std::max(kStageBytes, one_expert);
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
    use_.assign((size_t) n_layer_ * n_expert_, 0);
}

// Experts go to VRAM by use per byte when a profile has counts, and otherwise the same number from every layer.
void Engine::fill_expert_cache() {
    if (!opt_.gpu_experts || n_moe_ == 0) return;
    size_t free_b = 0, total_b = 0;
    ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
    const size_t reserve = size_t(std::max(opt_.vram_reserve_mib, 0)) << 20;
    if (free_b <= reserve) return;

    struct Cand {
        int il, e;
        size_t bytes;
        double value;
    };
    std::vector<Cand> cands;
    size_t all = 0;
    for (int il = 0; il < n_layer_; ++il) {
        const Layer& L = layers_[(size_t) il];
        if (!L.moe || !kernels::native_mmvq_supported((int) L.gu_type) ||
            !kernels::native_mmvq_supported((int) L.d_type))
            continue;
        const size_t bytes = pad(2 * L.gu_bytes + L.d_bytes);
        for (int e = 0; e < n_expert_; ++e) {
            const double used = prior_.empty() ? 0.0 : (double) prior_[(size_t) il * n_expert_ + e];
            cands.push_back({il, e, bytes, used / (double) bytes});
            all += bytes;
        }
    }
    std::stable_sort(cands.begin(), cands.end(),
                     [](const Cand& a, const Cand& b) { return a.value != b.value ? a.value > b.value : a.e < b.e; });

    const size_t arena = std::min(all, free_b - reserve);
    if (arena == 0) return;
    exp_ = (uint8_t*) dmalloc(arena + 256, "the expert cache");
    for (const Cand& c : cands) {
        if (exp_at_ + c.bytes > arena) continue;   // a smaller expert of another layer may still fit
        Layer& L = layers_[(size_t) c.il];
        uint8_t* s = exp_ + exp_at_;
        exp_at_ += c.bytes;
        const size_t e = (size_t) c.e;
        ck(cudaMemcpy(s, L.exp_gate + e * L.gu_bytes, L.gu_bytes, cudaMemcpyHostToDevice), "expert upload");
        ck(cudaMemcpy(s + L.gu_bytes, L.exp_up + e * L.gu_bytes, L.gu_bytes, cudaMemcpyHostToDevice),
           "expert upload");
        ck(cudaMemcpy(s + 2 * L.gu_bytes, L.exp_down + e * L.d_bytes, L.d_bytes, cudaMemcpyHostToDevice),
           "expert upload");
        L.slot[e] = s;
        ++cached_;
    }
}

// What a layer copies to the staging buffer ahead of its routing: the experts the cache left out, the most used
// first, as many as the buffer holds.
void Engine::plan_staging() {
    if (!mmq_) return;
    for (int il = 0; il < n_layer_; ++il) {
        Layer& L = layers_[(size_t) il];
        if (!L.moe) continue;
        for (int e = 0; e < n_expert_; ++e)
            if (!L.slot[(size_t) e]) L.ahead.push_back(e);
        if (!prior_.empty()) {
            const int64_t* used = prior_.data() + (size_t) il * n_expert_;
            std::stable_sort(L.ahead.begin(), L.ahead.end(), [used](int32_t a, int32_t b) { return used[a] > used[b]; });
        }
        L.ahead.resize(std::min(L.ahead.size(), stage_bytes_ / (2 * L.gu_bytes + L.d_bytes)));
    }
}

void Engine::load(const Options& o) {
    opt_ = o;
    if (opt_.max_context < 16) throw std::runtime_error("the context must be at least 16 tokens");
    gg_ = std::make_unique<GgufFile>(opt_.model);
    load_geometry();

    cudaStream_t st = nullptr;
    ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "cudaStreamCreate");
    st_ = st;
    cudaEvent_t ev = nullptr;
    ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");
    ev_ = ev;

    for (const TensorInfo& t : gg_->tensors())
        if (!ends_with(t.name, "_exps.weight") && t.name != "token_embd.weight")
            dense_bytes_ += pad(payload(*gg_, t));
    dense_ = (uint8_t*) dmalloc(dense_bytes_ + 256, "the dense weights");

    const TensorInfo* te = gg_->find("token_embd.weight");
    if (!te || te->type != kTypeQ8_0 || te->shape.size() != 2 || (int64_t) te->shape[0] != n_embd_ ||
        (int64_t) te->shape[1] != n_vocab_)
        throw std::runtime_error("token_embd.weight must be Q8_0 [" + std::to_string(n_embd_) + ", " +
                                 std::to_string(n_vocab_) + "], it is " + (te ? shape_of(*te) : "missing"));
    payload(*gg_, *te);
    tok_embd_ = gg_->tensor_data(*te);

    layers_.resize((size_t) n_layer_);
    for (int il = 0; il < n_layer_; ++il) load_layer(il);
    out_norm_ = upload("output_norm.weight", n_embd_, 1, kF32);
    output_ = upload("output.weight", n_embd_, n_vocab_, kAnyMat);

    alloc_state();
    if (!opt_.profile.empty()) read_profile(opt_.profile, n_layer_, n_expert_, prior_);
    fill_expert_cache();
    plan_staging();
    workers_ =std::make_unique<Workers>(opt_.threads > 0 ? opt_.threads : usable_cpus());
    reset();
}

void Engine::save_profile() const {
    if (opt_.profile.empty()) return;
    const std::string tmp = opt_.profile + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "w");
    if (!f) throw std::runtime_error("cannot write " + tmp);
    for (size_t i = 0; i < use_.size(); ++i) {
        const int64_t n = use_[i] + (prior_.empty() ? 0 : prior_[i]);
        if (n > 0)
            std::fprintf(f, "%d %d %lld\n", (int) (i / (size_t) n_expert_), (int) (i % (size_t) n_expert_),
                         (long long) n);
    }
    if (std::fclose(f) != 0 || std::rename(tmp.c_str(), opt_.profile.c_str()) != 0)
        throw std::runtime_error("cannot write " + opt_.profile);
}

void Engine::reset() {
    ck(cudaMemsetAsync(state_, 0, state_n_ * sizeof(float), cs(st_)), "reset");
    pos_ = 0;
    ck_pos_ = -1;
}

void Engine::checkpoint() {
    ck(cudaMemcpyAsync(state_ck_, state_, state_n_ * sizeof(float), cudaMemcpyDeviceToDevice, cs(st_)),
       "checkpoint");
    ck_pos_ = pos_;
}

bool Engine::restore() {
    if (ck_pos_ < 0) return false;
    ck(cudaMemcpyAsync(state_, state_ck_, state_n_ * sizeof(float), cudaMemcpyDeviceToDevice, cs(st_)), "restore");
    pos_ = ck_pos_;
    return true;
}

// A Q8_0 row: per 32 values, an fp16 scale and 32 int8.
void Engine::embed(int32_t token, float* dst) const {
    const uint8_t* b = tok_embd_ + (size_t) token * (size_t) (n_embd_ / 32) * 34;
    for (int i = 0; i < n_embd_; i += 32, b += 34) {
        uint16_t d;
        std::memcpy(&d, b, sizeof(d));
        const float scale = kernels::f32_from_f16(d);
        for (int k = 0; k < 32; ++k) dst[i + k] = scale * (float) (int8_t) b[2 + k];
    }
}

void Engine::dbg(const char* name, int il, const float* dev, int n) {
    if (!opt_.dump) return;
    std::vector<float> h((size_t) n);
    ck(cudaStreamSynchronize(cs(st_)), "dump");
    ck(cudaMemcpy(h.data(), dev, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost), "dump");
    double sum = 0, abs = 0;
    for (float v : h) {
        sum += v;
        abs += std::fabs(v);
    }
    std::fprintf(stderr, "dump pos=%lld blk=%d %-8s n=%d sum=%.6f abs=%.6f first=%.6f %.6f %.6f\n", (long long) pos_,
                 il, name, n, sum, abs, h[0], h[1], h[2]);
}

// y = w x for n tokens.  xq is x already in Q8_1, when the caller has it.  rows_per_x below n_out gives each group
// of rows its own input (see gemv_f32); only the float types take it.
void Engine::lin(const Dev& w, const float* x, float* y, int n, const void* xq, int rows_per_x) {
    if (w.type == kTypeF32)
        return gemv_f32(fp(w), x, y, w.n_in, w.n_out, rows_per_x ? rows_per_x : w.n_out, n, st_);
    if (w.type == kTypeBF16)
        return gemv_bf16((const uint16_t*) w.p, x, y, w.n_in, w.n_out, rows_per_x ? rows_per_x : w.n_out, n, st_);
    // a prompt batch: llama.cpp's matrix-matrix kernels read the weights once for all the tokens
    if (mmq_ && n >= kStageRows && Mmq::fits((int) w.type, w.n_out)) {
        Mmq::quantize(x, nullptr, mqx_, (int) w.type, w.n_in, w.n_in, n, st_);
        return mmq_->mul(w.p, (int) w.type, w.n_out, w.n_in, 0, 1, mqx_, nullptr, nullptr, n, n, y, w.n_out, st_);
    }
    if (!xq) {
        // the tokens' blocks are back to back, so they quantize as one long vector
        kernels::native_quantize_q8_1(x, q8_, w.n_in * n, 1, st_);
        xq = q8_;
    }
    // a launch takes up to 8 tokens, and each comes out as it would alone
    const size_t token = kernels::native_q8_1_bytes(w.n_in);
    for (int at = 0; at < n; at += 8)
        kernels::native_mmvq((int) w.type, w.p, (const uint8_t*) xq + (size_t) at * token,
                             y + (size_t) at * w.n_out, w.n_in, w.n_out, std::min(8, n - at), st_);
}

void Engine::kda(Layer& L, int n) {
    const size_t d = (size_t) n * d_inner_;
    lin(L.wq, xn_, proj_, n, xq_);
    lin(L.wk, xn_, proj_ + d, n, xq_);
    lin(L.wv, xn_, proj_ + 2 * d, n, xq_);
    kda_conv(proj_, L.conv, fp(L.conv_q), fp(L.conv_k), fp(L.conv_v), qkv_, d_inner_, n, st_);

    lin(L.f_a, xn_, fa_, n, xq_);
    lin(L.f_b, fa_, g1_, n);
    add_bias(fp(L.dt_b), g1_, d_inner_, n, st_);
    lin(L.beta, xn_, beta_, n, xq_);
    lin(L.g_a, xn_, ga_, n, xq_);
    lin(L.g_b, ga_, g2_, n);
    kda_step(qkv_, g1_, fp(L.a), beta_, L.s, g2_, fp(L.o_norm), gated_, kda_dim_, n_head_, n, kda_lb_, rms_eps_,
             st_);
    lin(L.wo, gated_, ao_, n);
}

// For the n tokens from row `first` of the call: the n_top pools with the highest indexer score, expanded to
// their rows, then the tail that is in no pool yet.  The scores come to the host for the selection, which costs
// a wait per layer; it only runs past top_k tokens.
void Engine::dsa_select(Layer& L, int first, int n) {
    const int64_t pos = pos_ + first;
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

void Engine::dsa(Layer& L, int n) {
    lin(L.q_a, xn_, t1_, n, xq_);
    rms_norm(t1_, fp(L.q_a_norm), qr_, q_rank_, n, rms_eps_, st_);
    lin(L.q_b, qr_, q_, n);
    lin(L.k_b, q_, qabs_, n, nullptr, kv_rank_);   // the key projection, absorbed into the query: per head

    lin(L.kv_a, xn_, t2_, n, xq_);
    rms_norm(t2_, fp(L.kv_a_norm), kv_, kv_rank_, n, rms_eps_, st_);
    store_f16(kv_, L.lat, kv_rank_, pos_, n, st_);

    lin(L.idx_k, xn_, ikr_, n, xq_);
    layer_norm(ikr_, fp(L.idx_k_norm), fp(L.idx_k_norm_b), ik_, idx_dim_, n, norm_eps_, st_);
    lin(L.idx_gate, xn_, ig_, n, xq_);
    idx_store(ik_, ig_, L.idx, idx_dim_, pos_, n, st_);
    idx_pool(L.idx, fp(L.idx_ape), idx_dim_, kpool_, pos_, n, st_);

    // While a token's pools all fit in top_k, every one is selected and its attention is plain causal attention.
    // That holds up to a position, so the tokens of a call are dense first and sparse after.
    const int64_t last_dense = (int64_t) idx_top_k_ + kpool_ - 2;
    const int dense = (int) std::clamp<int64_t>(last_dense - pos_ + 1, 0, n);
    if (dense < n) dsa_select(L, dense, n - dense);
    dsa_attend(qabs_, L.lat, sel_, sel_stride_, idx_top_k_, kpool_, pos_, n, dense, kqv_, kv_rank_, n_head_,
               1.f / std::sqrt((float) mla_k_), st_);
    lin(L.v_b, kqv_, vout_, n, nullptr, mla_v_);
    lin(L.wo, vout_, ao_, n);
}

void Engine::dense_ffn(Layer& L, int il, int n) {
    lin(L.ffn_gate, xn_, g_, n, xq_);
    lin(L.ffn_up, xn_, u_, n, xq_);
    swiglu_clamp(g_, u_, h_, n * n_ff_dense_, clamp_shexp_[(size_t) il], st_);
    lin(L.ffn_down, h_, fo_, n);
}

// What ggml's MoE graph does for this model: sigmoid gates, the n_used largest of gate + bias, and weights from
// the bare gates.
void Engine::route(const Layer& L, const float* logits, int* ids, float* w) {
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

void Engine::moe(Layer& L, int il, int n) {
    const size_t e_row = (size_t) n_embd_, rows = (size_t) n;
    lin(L.gate_inp, xn_, rl_, n, xq_);
    ck(cudaMemcpyAsync(h_rl_, rl_, rows * n_expert_ * sizeof(float), cudaMemcpyDeviceToHost, cs(st_)), "router");
    ck(cudaMemcpyAsync(h_x_, xn_, rows * e_row * sizeof(float), cudaMemcpyDeviceToHost, cs(st_)), "router");
    ck(cudaEventRecord((cudaEvent_t) ev_, cs(st_)), "router");

    // the shared expert runs on the GPU while the host routes
    lin(L.sh_gate, xn_, g_, n, xq_);
    lin(L.sh_up, xn_, u_, n, xq_);
    swiglu_clamp(g_, u_, h_, n * n_ff_exp_, clamp_shexp_[(size_t) il], st_);
    lin(L.sh_down, h_, fo_, n);

    Clock::time_point t0 = Clock::now();
    ck(cudaEventSynchronize((cudaEvent_t) ev_), "router");
    stats_.route_wait_ms += ms_since(t0);

    const float limit = clamp_exp_[(size_t) il];
    const size_t xq_row = kernels::native_q8_1_bytes(n_embd_);
    bool any_host = false;
    for (int t = 0; t < n; ++t) {
        int ids[kMaxUsed];
        float w[kMaxUsed];
        route(L, h_rl_ + (size_t) t * n_expert_, ids, w);

        CpuExpert host[kMaxUsed];
        int n_host = 0;
        const void* xq = (const uint8_t*) xq_ + (size_t) t * xq_row;
        float* out = fo_ + (size_t) t * e_row;
        for (int k = 0; k < n_used_; ++k) {
            const size_t e = (size_t) ids[k];
            ++use_[(size_t) il * n_expert_ + e];
            if (const uint8_t* s = L.slot[e]) {
                kernels::native_mmvq((int) L.gu_type, s, xq, g_, n_embd_, 2 * n_ff_exp_, 1, st_);
                swiglu_clamp(g_, g_ + n_ff_exp_, h_, n_ff_exp_, limit, st_);
                kernels::native_quantize_q8_1(h_, q8_, n_ff_exp_, 1, st_);
                kernels::native_mmvq((int) L.d_type, s + 2 * L.gu_bytes, q8_, d_, n_ff_exp_, n_embd_, 1, st_);
                axpy(d_, w[k], out, n_embd_, true, st_);
            } else {
                host[n_host++] = {L.exp_gate + e * L.gu_bytes, L.exp_up + e * L.gu_bytes,
                                  L.exp_down + e * L.d_bytes, w[k]};
            }
        }
        stats_.gpu_experts += n_used_ - n_host;
        stats_.cpu_experts += n_host;

        // The next layer's router wait is what makes h_cpu_ free to write again.
        float* cpu = h_cpu_ + (size_t) t * e_row;
        if (n_host == 0) {
            std::fill(cpu, cpu + e_row, 0.f);
            continue;
        }
        any_host = true;
        t0 = Clock::now();
        cpu_experts(*workers_, host, n_host, (int) L.gu_type, (int) L.d_type, n_embd_, n_ff_exp_,
                    h_x_ + (size_t) t * e_row, limit, cpu);
        stats_.cpu_ms += ms_since(t0);
    }
    if (!any_host) return;
    ck(cudaMemcpyAsync(cpu_, h_cpu_, rows * e_row * sizeof(float), cudaMemcpyHostToDevice, cs(st_)),
       "host experts");
    axpy(cpu_, 1.f, fo_, n * n_embd_, true, st_);
}

// m experts from the mapped file into the staging buffer: gate and up of each back to back from the start, the
// downs after them.  The two pinned buffers take turns, so the workers fill one while the other uploads.
void Engine::stage(const Layer& L, const int32_t* experts, int m, void* stream) {
    const size_t gu2 = 2 * L.gu_bytes, down_at = pad((size_t) m * gu2);
    const int per = (int) (chunk_bytes_ / (gu2 + L.d_bytes));
    for (int j0 = 0; j0 < m; j0 += per) {
        const size_t c = (size_t) std::min(per, m - j0);
        stage_turn_ ^= 1;
        uint8_t* h = h_stage_[stage_turn_];
        const cudaEvent_t left = (cudaEvent_t) ev_stage_[stage_turn_];
        ck(cudaEventSynchronize(left), "expert staging");
        workers_->run((int) (3 * c), [&](int i) {
            const size_t j = (size_t) i / 3, e = (size_t) experts[(size_t) j0 + j];
            if (i % 3 == 0) std::memcpy(h + j * gu2, L.exp_gate + e * L.gu_bytes, L.gu_bytes);
            else if (i % 3 == 1) std::memcpy(h + j * gu2 + L.gu_bytes, L.exp_up + e * L.gu_bytes, L.gu_bytes);
            else std::memcpy(h + c * gu2 + j * L.d_bytes, L.exp_down + e * L.d_bytes, L.d_bytes);
        });
        ck(cudaMemcpyAsync(stage_ + (size_t) j0 * gu2, h, c * gu2, cudaMemcpyHostToDevice, cs(stream)),
           "expert staging");
        ck(cudaMemcpyAsync(stage_ + down_at + (size_t) j0 * L.d_bytes, h + c * gu2, c * L.d_bytes,
                           cudaMemcpyHostToDevice, cs(stream)),
           "expert staging");
        ck(cudaEventRecord(left, cs(stream)), "expert staging");
    }
}

// A MoE layer for a prompt batch: every token's experts on the GPU, by llama.cpp's MMQ.  A row is a token and one
// of the experts it routes to.  The rows are sorted by expert, the experts of the VRAM cache first; the other
// experts pass through the staging buffer, as many at a time as it holds.  A full batch starts the copy of
// L.ahead before it knows the routing, on a stream of its own, so the upload runs under the layer's attention.
void Engine::moe_rows(Layer& L, int il, int n) {
    const int rr = n * n_used_;
    const size_t cap = (size_t) rows_ * n_used_;
    lin(L.gate_inp, xn_, rl_, n, xq_);
    ck(cudaMemcpyAsync(h_rl_, rl_, (size_t) n * n_expert_ * sizeof(float), cudaMemcpyDeviceToHost, cs(st_)),
       "router");
    ck(cudaEventRecord((cudaEvent_t) ev_, cs(st_)), "router");

    lin(L.sh_gate, xn_, g_, n, xq_);
    lin(L.sh_up, xn_, u_, n, xq_);
    swiglu_clamp(g_, u_, h_, n * n_ff_exp_, clamp_shexp_[(size_t) il], st_);
    lin(L.sh_down, h_, fo_, n);

    const int n_ahead = n >= kAheadRows ? (int) L.ahead.size() : 0;
    if (n_ahead > 0) {
        // the buffer is free once the layer before has read it
        ck(cudaStreamWaitEvent(cs(up_), (cudaEvent_t) ev_free_, 0), "expert staging");
        stage(L, L.ahead.data(), n_ahead, up_);
        ck(cudaEventRecord((cudaEvent_t) ev_ahead_, cs(up_)), "expert staging");
    }

    const Clock::time_point t0 = Clock::now();
    ck(cudaEventSynchronize((cudaEvent_t) ev_), "router");
    stats_.route_wait_ms += ms_since(t0);

    // The layer's tables.  tok and place: a row's token, and its place among that token's experts.  bounds: each
    // expert's first row.  rel: the same for the experts of a staging round, counted from the round's first row.
    // w: the weight of each place.
    int32_t* tok = h_tab_;
    int32_t* place = tok + cap;
    int32_t* bounds = place + cap;
    int32_t* rel = bounds + n_expert_ + 1;
    float* w = (float*) (rel + 2 * n_expert_ + 2);
    const int32_t* d_tok = tab_;
    const int32_t* d_place = d_tok + cap;
    const int32_t* d_bounds = d_place + cap;
    const int32_t* d_rel = d_bounds + n_expert_ + 1;
    const float* d_w = (const float*) (d_rel + 2 * n_expert_ + 2);

    count_.assign((size_t) n_expert_, 0);
    for (int t = 0; t < n; ++t) {
        int32_t* ids = ids_.data() + (size_t) t * n_used_;
        route(L, h_rl_ + (size_t) t * n_expert_, ids, w + (size_t) t * n_used_);
        for (int k = 0; k < n_used_; ++k) {
            ++count_[(size_t) ids[k]];
            ++use_[(size_t) il * n_expert_ + (size_t) ids[k]];
        }
    }
    // the cached experts that have rows, the experts already on their way, then the others that have rows
    group_.clear();
    for (int e = 0; e < n_expert_; ++e)
        if (count_[(size_t) e] > 0 && L.slot[(size_t) e]) group_.push_back(e);
    const int n_cached = (int) group_.size();
    group_.insert(group_.end(), L.ahead.begin(), L.ahead.begin() + n_ahead);
    sent_.assign((size_t) n_expert_, 0);
    for (int j = 0; j < n_ahead; ++j) sent_[(size_t) L.ahead[(size_t) j]] = 1;
    for (int e = 0; e < n_expert_; ++e)
        if (count_[(size_t) e] > 0 && !L.slot[(size_t) e] && !sent_[(size_t) e]) group_.push_back(e);
    const int n_group = (int) group_.size();
    auto rows_of = [&](int g) { return count_[(size_t) group_[(size_t) g]]; };

    int at = 0;
    for (int g = 0; g < n_group; ++g) {
        bounds[g] = at;
        first_[(size_t) group_[(size_t) g]] = at;
        at += rows_of(g);
    }
    bounds[n_group] = at;
    for (int i = 0; i < rr; ++i) {
        const int row = first_[(size_t) ids_[(size_t) i]]++;
        tok[row] = i / n_used_;
        place[row] = i;
    }
    const size_t gu2 = 2 * L.gu_bytes;
    // a round is what the staging buffer holds at a time; round_ has the first group of each, and the end
    const int per_round = (int) (stage_bytes_ / (gu2 + L.d_bytes));
    round_.assign(1, n_cached);
    if (n_ahead > 0) round_.push_back(n_cached + n_ahead);
    while (round_.back() < n_group) round_.push_back(std::min(round_.back() + per_round, n_group));
    const int n_round = (int) round_.size() - 1;
    for (int r = 0; r < n_round; ++r)
        for (int g = round_[(size_t) r]; g <= round_[(size_t) r + 1]; ++g)
            rel[g - n_cached + r] = bounds[g] - bounds[round_[(size_t) r]];
    ck(cudaMemcpyAsync(tab_, h_tab_, (3 * cap + 3 * (size_t) n_expert_ + 3) * sizeof(int32_t),
                       cudaMemcpyHostToDevice, cs(st_)),
       "expert rows");

    const int rc = bounds[n_cached];   // the rows of the cached experts
    stats_.gpu_experts += rc;
    stats_.staged_experts += rr - rc;

    const float limit = clamp_exp_[(size_t) il];
    const int gt = (int) L.gu_type, dt = (int) L.d_type;
    const int64_t ff = n_ff_exp_, ff2 = 2 * ff;
    Mmq::quantize(xn_, d_tok, mq_, gt, n_embd_, n_embd_, rr, st_);

    for (int g = 0; g < n_cached; ++g)
        mmq_->mul(L.slot[(size_t) group_[(size_t) g]], gt, ff2, n_embd_, gu2, 1, mq_, d_bounds + g, iota_, rr,
                  rows_of(g), gu_, ff2, st_);
    if (rc > 0) {
        swiglu_clamp_rows(gu_, hx_, n_ff_exp_, rc, limit, st_);
        Mmq::quantize(hx_, nullptr, mq2_, dt, ff, ff, rc, st_);
    }
    for (int g = 0; g < n_cached; ++g)
        mmq_->mul(L.slot[(size_t) group_[(size_t) g]] + gu2, dt, n_embd_, ff, L.d_bytes, 1, mq2_, d_bounds + g,
                  d_place, rc, rows_of(g), dx_, n_embd_, st_);

    for (int r = 0; r < n_round; ++r) {
        const int g0 = round_[(size_t) r], m = round_[(size_t) r + 1] - g0;
        const size_t down_at = pad((size_t) m * gu2);
        if (r == 0 && n_ahead > 0) ck(cudaStreamWaitEvent(cs(st_), (cudaEvent_t) ev_ahead_, 0), "expert staging");
        else stage(L, group_.data() + g0, m, st_);

        const int r0 = bounds[g0], rows = bounds[g0 + m] - r0;
        if (rows == 0) continue;   // nothing routed to the experts that went ahead
        int most = 0;
        for (int g = g0; g < g0 + m; ++g) most = std::max(most, rows_of(g));
        mmq_->mul(stage_, gt, ff2, n_embd_, gu2, m, mq_, d_bounds + g0, iota_, rr, most, gu_, ff2, st_);
        swiglu_clamp_rows(gu_ + (size_t) r0 * ff2, hx_ + (size_t) r0 * ff, n_ff_exp_, rows, limit, st_);
        Mmq::quantize(hx_ + (size_t) r0 * ff, nullptr, mq2_, dt, ff, ff, rows, st_);
        mmq_->mul(stage_ + down_at, dt, n_embd_, ff, L.d_bytes, m, mq2_, d_rel + (g0 - n_cached + r), d_place + r0,
                  rows, most, dx_, n_embd_, st_);
    }
    ck(cudaEventRecord((cudaEvent_t) ev_free_, cs(st_)), "expert staging");
    moe_sum(dx_, d_w, fo_, n_embd_, n_used_, n, st_);
}

void Engine::eval(const int32_t* tokens, int n, bool want_logits) {
    if (n < 1 || n > rows_) throw std::logic_error("eval takes 1 to max_rows() tokens");
    for (int t = 0; t < n; ++t)
        if (tokens[t] < 0 || tokens[t] >= n_vocab_) throw std::runtime_error("a token id is outside the vocabulary");
    if (pos_ + n > opt_.max_context)
        throw std::runtime_error("the context is full (" + std::to_string(opt_.max_context) + " tokens)");

    // h_r_ and h_cpu_ may still be the source of the previous call's copies
    ck(cudaStreamSynchronize(cs(st_)), "eval");
    const size_t e_row = (size_t) n_embd_, r_row = kHc * e_row;
    for (int t = 0; t < n; ++t) {
        float* r = h_r_ + (size_t) t * r_row;
        embed(tokens[t], r);
        for (int h = 1; h < kHc; ++h) std::memcpy(r + (size_t) h * e_row, r, e_row * sizeof(float));
    }
    ck(cudaMemcpyAsync(r_, h_r_, (size_t) n * r_row * sizeof(float), cudaMemcpyHostToDevice, cs(st_)), "embed");

    for (int il = 0; il < n_layer_; ++il) {
        Layer& L = layers_[(size_t) il];
        hc_pre(r_, fp(L.hc_attn_fn), fp(L.hc_attn_scale), fp(L.hc_attn_base), mix_, hcw_, cur_, n_embd_, n,
               rms_eps_, hc_eps_, sinkhorn_, st_);
        rms_norm(cur_, fp(L.attn_norm), xn_, n_embd_, n, rms_eps_, st_);
        kernels::native_quantize_q8_1(xn_, xq_, n_embd_ * n, 1, st_);
        dbg("attn_in", il, xn_, n_embd_);
        if (L.dsa) dsa(L, n);
        else kda(L, n);
        dbg("attn", il, ao_, n_embd_);
        hc_post(ao_, hcw_, r_, n_embd_, n, st_);

        hc_pre(r_, fp(L.hc_ffn_fn), fp(L.hc_ffn_scale), fp(L.hc_ffn_base), mix_, hcw_, cur_, n_embd_, n, rms_eps_,
               hc_eps_, sinkhorn_, st_);
        rms_norm(cur_, fp(L.ffn_norm), xn_, n_embd_, n, rms_eps_, st_);
        kernels::native_quantize_q8_1(xn_, xq_, n_embd_ * n, 1, st_);
        dbg("ffn_in", il, xn_, n_embd_);
        if (!L.moe) dense_ffn(L, il, n);
        else if (mmq_ && n >= kStageRows) moe_rows(L, il, n);
        else moe(L, il, n);
        dbg("ffn", il, fo_, n_embd_);
        hc_post(fo_, hcw_, r_, n_embd_, n, st_);
    }

    if (want_logits) {
        hc_mean(r_ + (size_t) (n - 1) * r_row, cur_, n_embd_, st_);
        rms_norm(cur_, fp(out_norm_), xn_, n_embd_, 1, rms_eps_, st_);
        dbg("out_norm", n_layer_, xn_, n_embd_);
        lin(output_, xn_, logits_, 1);
        ck(cudaMemcpyAsync(h_logits_, logits_, (size_t) n_vocab_ * sizeof(float), cudaMemcpyDeviceToHost, cs(st_)),
           "logits");
        ck(cudaStreamSynchronize(cs(st_)), "eval");
    }
    pos_ += n;
    stats_.tokens += n;
}

}  // namespace strata::glm
