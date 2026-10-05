// src/glm/engine.hpp - GLM 5.3 Flash (llama.cpp architecture "glm5-next").
//
// The engine reads the GGUF in place.  Everything but the routed experts goes to VRAM; as many routed experts as
// fit go there too, and the rest are computed on the CPU from the mapped file while the GPU does its share.
#pragma once

#include "cpu_experts.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata {
class GgufFile;
}

namespace strata::glm {

struct Options {
    std::string model;
    int max_context = 8192;
    int threads = 0;              ///< 0: the CPUs this process may use
    int vram_reserve_mib = 1024;  ///< left free after the expert cache is filled
    std::string profile;          ///< expert use counts; orders the expert cache, and save_profile() adds to it
    bool gpu_experts = true;
    bool dump = false;            ///< print a checksum of every stage, for comparing against llama.cpp
};

/// A tensor in VRAM in its GGUF form.  n_in is the contiguous axis.
struct Dev {
    void* p = nullptr;
    uint32_t type = 0;
    int n_in = 0, n_out = 0;
};

struct Layer {
    bool dsa = false, moe = false;
    Dev attn_norm, ffn_norm;
    Dev hc_attn_fn, hc_attn_scale, hc_attn_base, hc_ffn_fn, hc_ffn_scale, hc_ffn_base;
    Dev wo;
    // KDA
    Dev wq, wk, wv, conv_q, conv_k, conv_v, f_a, f_b, dt_b, beta, a, g_a, g_b, o_norm;
    float* conv = nullptr;  ///< 3 * d_inner * (d_conv - 1)
    float* s = nullptr;     ///< n_head * head_dim * head_dim
    // DSA
    Dev q_a, q_a_norm, q_b, kv_a, kv_a_norm, k_b, v_b;
    Dev idx_k, idx_k_norm, idx_k_norm_b, idx_q_b, idx_proj, idx_gate, idx_ape;
    uint16_t* lat = nullptr;  ///< max_context * kv_rank, fp16
    float* idx = nullptr;     ///< max_context * 3 * idx_dim
    // dense FFN
    Dev ffn_gate, ffn_up, ffn_down;
    // MoE
    Dev gate_inp, sh_gate, sh_up, sh_down;
    std::vector<float> probs_b;
    const uint8_t* exp_gate = nullptr;  ///< n_expert experts back to back, in the mapped file
    const uint8_t* exp_up = nullptr;
    const uint8_t* exp_down = nullptr;
    uint32_t gu_type = 0, d_type = 0;
    size_t gu_bytes = 0, d_bytes = 0;  ///< one expert's gate (or up), and its down
    std::vector<uint8_t*> slot;        ///< per expert: gate | up | down in VRAM, or null
};

/// A routed expert is run from the VRAM cache, on the CPU, or (in a prompt batch) from the staging buffer.
struct Stats {
    int64_t tokens = 0, gpu_experts = 0, cpu_experts = 0, staged_experts = 0;
    double cpu_ms = 0, route_wait_ms = 0;
};

class Mmq;

class Engine {
public:
    Engine();
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // load, eval and the other calls that touch the device throw std::runtime_error.
    void load(const Options& o);

    int n_vocab() const { return n_vocab_; }
    int max_context() const { return opt_.max_context; }
    int threads() const { return workers_ ? workers_->threads() : 0; }
    int64_t pos() const { return pos_; }
    int64_t meta_u(const char* key, int64_t fallback) const;

    /// Forget the sequence.
    void reset();
    /// The most tokens one eval() takes.
    int max_rows() const { return rows_; }
    /// Feed n tokens from position pos().  With want_logits, logits() then holds the scores of the token after the
    /// last one.
    void eval(const int32_t* tokens, int n, bool want_logits);
    const float* logits() const { return h_logits_; }

    /// One saved point of the sequence to come back to.  The attention caches are indexed by position and only
    /// ever written at the current one, so the recurrent state is all there is to save.
    void checkpoint();
    bool restore();
    int64_t checkpoint_pos() const { return ck_pos_; }

    const Stats& stats() const { return stats_; }
    int cached_experts() const { return cached_; }
    int total_experts() const { return n_moe_ * n_expert_; }
    double cached_gib() const { return double(exp_at_) / double(1ull << 30); }
    void save_profile() const;

private:
    enum Want { kF32, kFloatMat, kAnyMat };
    Dev upload(const std::string& name, int64_t n_in, int64_t n_out, Want want);
    void lin(const Dev& w, const float* x, float* y, int n, const void* xq = nullptr, int rows_per_x = 0);
    void load_geometry();
    void load_layer(int il);
    void alloc_state();
    void assign_scratch();
    void fill_expert_cache();
    void kda(Layer& L, int n);
    void dsa(Layer& L, int n);
    void dsa_select(Layer& L, int first, int n);
    void dense_ffn(Layer& L, int il, int n);
    void moe(Layer& L, int il, int n);
    void moe_rows(Layer& L, int il, int n);
    void route(const Layer& L, const float* logits, int* ids, float* w);
    void embed(int32_t token, float* dst) const;
    void dbg(const char* name, int il, const float* dev, int n);
    float* fbuf(size_t n);
    void* dmalloc(size_t bytes, const char* what);
    void* hmalloc(size_t bytes, const char* what);

    Options opt_;
    std::unique_ptr<GgufFile> gg_;
    std::unique_ptr<Workers> workers_;
    void* st_ = nullptr;   ///< the one CUDA stream
    void* ev_ = nullptr;   ///< marks the router's copy to the host
    std::vector<void*> dev_allocs_, host_allocs_;

    // geometry
    int n_embd_ = 0, n_layer_ = 0, n_head_ = 0, kda_dim_ = 0, d_inner_ = 0, n_vocab_ = 0;
    int n_expert_ = 0, n_used_ = 0, n_ff_exp_ = 0, n_ff_dense_ = 0, n_moe_ = 0, n_kda_ = 0, n_dsa_ = 0;
    int q_rank_ = 0, kv_rank_ = 0, mla_k_ = 0, mla_v_ = 0;
    int idx_heads_ = 0, idx_dim_ = 0, idx_top_k_ = 0, kpool_ = 0;
    int sinkhorn_ = 0;
    int rows_ = 1;            ///< tokens the scratch is sized for
    int score_stride_ = 0, sel_stride_ = 0;   ///< one token's indexer scores, and its selected cache rows
    int64_t gate_rank_ = 0;   ///< the widest low-rank gate projection of the KDA layers
    float hc_eps_ = 0, rms_eps_ = 0, norm_eps_ = 0, kda_lb_ = 0, w_scale_ = 1;
    bool w_norm_ = false;
    std::vector<float> clamp_exp_, clamp_shexp_;

    std::vector<Layer> layers_;
    Dev out_norm_, output_;
    const uint8_t* tok_embd_ = nullptr;

    uint8_t* dense_ = nullptr;   ///< the dense weights' arena
    size_t dense_bytes_ = 0, dense_at_ = 0;
    uint8_t* exp_ = nullptr;     ///< the routed experts' arena
    size_t exp_at_ = 0;
    float* scratch_ = nullptr;
    size_t scratch_at_ = 0;
    float* state_ = nullptr;     ///< every KDA layer's conv and delta-rule state
    float* state_ck_ = nullptr;
    size_t state_n_ = 0;
    int cached_ = 0;

    // device scratch
    float *r_ = nullptr, *cur_ = nullptr, *xn_ = nullptr, *ao_ = nullptr, *fo_ = nullptr, *mix_ = nullptr,
          *hcw_ = nullptr;
    float *proj_ = nullptr, *qkv_ = nullptr, *fa_ = nullptr, *g1_ = nullptr, *ga_ = nullptr, *g2_ = nullptr,
          *beta_ = nullptr, *gated_ = nullptr;
    float *t1_ = nullptr, *qr_ = nullptr, *q_ = nullptr, *t2_ = nullptr, *kv_ = nullptr, *qabs_ = nullptr,
          *kqv_ = nullptr, *vout_ = nullptr, *ikr_ = nullptr, *ik_ = nullptr, *ig_ = nullptr, *iq_ = nullptr,
          *iw_ = nullptr, *score_ = nullptr;
    float *g_ = nullptr, *u_ = nullptr, *h_ = nullptr, *d_ = nullptr, *cpu_ = nullptr, *rl_ = nullptr,
          *logits_ = nullptr;
    void *xq_ = nullptr, *q8_ = nullptr;
    int32_t* sel_ = nullptr;

    // pinned host buffers
    float *h_r_ = nullptr, *h_rl_ = nullptr, *h_x_ = nullptr, *h_cpu_ = nullptr, *h_logits_ = nullptr,
          *h_score_ = nullptr;
    int32_t* h_sel_ = nullptr;

    // A prompt batch's MoE (moe_rows).  A row is a token and one of the experts it routes to.
    std::unique_ptr<Mmq> mmq_;   ///< null when the card has no MMQ tile for the experts' types
    float *gu_ = nullptr, *hx_ = nullptr, *dx_ = nullptr;   ///< the rows' gate|up, hidden and down values
    void *mq_ = nullptr, *mq2_ = nullptr;                   ///< the rows quantized for the two products
    void* mqx_ = nullptr;                                   ///< lin()'s tokens quantized for a product
    int32_t *tab_ = nullptr, *h_tab_ = nullptr;   ///< a layer's row tables, in VRAM and pinned
    int32_t* iota_ = nullptr;                     ///< 0, 1, 2, ...
    uint8_t* stage_ = nullptr;                    ///< experts that are not cached, on their way through VRAM
    uint8_t* h_stage_[2] = {};                    ///< pinned; filled in turn
    void* ev_stage_[2] = {};                      ///< marks the upload out of each
    size_t stage_bytes_ = 0;
    int stage_turn_ = 0;
    std::vector<int32_t> ids_, count_, first_, group_;

    int64_t pos_ = 0, ck_pos_ = -1;
    std::vector<float> probs_, pick_;
    std::vector<int64_t> use_, prior_;   ///< routed count per (layer, expert): this run, and the profile's
    Stats stats_;
};

}  // namespace strata::glm
