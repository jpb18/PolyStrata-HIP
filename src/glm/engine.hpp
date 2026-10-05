// src/glm/engine.hpp - GLM 5.3 Flash (llama.cpp architecture "glm5-next").
//
// The engine reads the GGUF in place, in one file or in the numbered files of a split one.  Everything but the
// routed experts goes to VRAM; as many routed experts as fit go there too, and the rest are computed on the CPU
// from the mapped file while the GPU does its share.
#pragma once

#include "cpu_experts.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata {
struct TensorInfo;
}

namespace strata::glm {

class Shards;

struct Options {
    std::string model;
    int max_context = 8192;
    int threads = 0;              ///< 0: the CPUs this process may use
    int vram_reserve_mib = 1024;  ///< left free after the expert cache is filled
    std::string profile;          ///< expert use counts; orders the expert cache, and save_profile() adds to it
    bool gpu_experts = true;
    bool dump = false;            ///< print a checksum of every stage, for comparing against llama.cpp
    // A prompt batch.  More tokens a call and a larger staging buffer read a prompt faster, and both take VRAM
    // that the expert cache, and so decoding, does without.
    int prompt_batch = 1024;      ///< prompt tokens one eval() takes; the scratch is about 1.4 MiB of VRAM a token
    int stage_mib = 2048;         ///< the buffer the experts that are not cached pass through: by default about
                                  ///< what a layer of Q2_K experts leaves out of the cache
    int stage_ahead = 256;        ///< from this many tokens, a call routes to most of a layer's experts, and the
                                  ///< ones that are not cached start their copy before the layer has routed; 0: never
    int expert_swaps = 16;        ///< experts on their way into the cache at a time while an answer is written, each
                                  ///< to the slot of one the answer uses less; 0: the cache stays as it was loaded
    bool stay_near = true;        ///< with several NUMA nodes, keep the CPU threads on the one that holds the experts
    bool graphs = true;           ///< a token's fixed launches are recorded once, a CUDA graph a layer, and replayed
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
    std::vector<int32_t> ahead;        ///< experts a full prompt batch stages before it routes
    int trade = -1;                    ///< layers with the same number can take each other's slots: same size
    void* head = nullptr;              ///< one token's launches up to the router, recorded (eval())
    void* shared = nullptr;            ///< and the shared expert's
};

/// A routed expert is run from the VRAM cache or on the CPU when tokens come a few at a time, as an answer's do
/// (gpu_experts, cpu_experts), and in a prompt batch from the cache or the staging buffer (batch_*).
struct Stats {
    int64_t tokens = 0, gpu_experts = 0, cpu_experts = 0, batch_cached = 0, batch_staged = 0;
    int64_t swapped_experts = 0;   ///< experts that took a cached one's slot (follow())
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
    /// The NUMA node the CPU threads were kept on, and the share of the experts' pages it holds; -1: no node chosen.
    int numa_node() const { return node_; }
    double numa_share() const { return node_share_; }
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

    // A sequence's state, to leave the sequence and come back to it.  The recurrent layers have a state that
    // every token changes.  Each token also leaves a row in the attention caches, which stays as it was written.
    size_t state_bytes() const { return state_n_ * sizeof(float); }
    size_t row_bytes() const;
    /// The recurrent state at pos(), state_bytes() long.
    void save_state(uint8_t* dst);
    /// Back to position `pos` with the recurrent state it had there.  The rows before pos must be the sequence's.
    void load_state(const uint8_t* src, int64_t pos);
    /// The rows of positions [first, first + n), n * row_bytes() long: layer after layer, a layer's rows together.
    void save_rows(int64_t first, int64_t n, uint8_t* dst);
    /// Rows [first, first + n) out of what save_rows wrote for [from, from + count).
    void load_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n);

    const Stats& stats() const { return stats_; }
    int cached_experts() const { return cached_; }
    int total_experts() const { return n_moe_ * n_expert_; }
    double cached_gib() const { return double(exp_at_) / double(1ull << 30); }
    void save_profile() const;

private:
    enum Want { kF32, kFloatMat, kAnyMat };
    Dev upload(const TensorInfo* t, const std::string& name, int64_t n_in, int64_t n_out, Want want,
               float (*fix)(float) = nullptr);
    const TensorInfo* tensor(int il, const char* name) const;
    void lin(const Dev& w, const float* x, float* y, int n, const void* xq = nullptr, int rows_per_x = 0);
    void load_geometry();
    void load_layer(int il);
    void alloc_state();
    void assign_scratch();
    void fill_expert_cache();
    void plan_ahead(Layer& L, int il);
    void kda(Layer& L, int n);
    void dsa(Layer& L, int n);
    void dsa_select(Layer& L, int first, int n);
    void dense_ffn(Layer& L, int il, int n);
    void layer_head(Layer& L, int il, int n);
    void moe_router(const Layer& L, int n);
    void moe_shared(const Layer& L, int il, int n);
    void moe(Layer& L, int il, int n);
    template <class F>
    void* record(F&& work);
    void moe_rows(Layer& L, int il, int n);
    void stage(const Layer& L, const int32_t* experts, int m, void* stream);
    void follow(int written);
    void follow_loop();
    void route(const Layer& L, const float* logits, int* ids, float* w);
    void embed(int32_t token, float* dst) const;
    void dbg(const char* name, int il, const float* dev, int n);
    float* fbuf(size_t n);
    void* dmalloc(size_t bytes, const char* what);
    void* hmalloc(size_t bytes, const char* what);

    Options opt_;
    std::unique_ptr<Shards> gg_;
    std::unique_ptr<Workers> workers_;
    int node_ = -1;
    double node_share_ = 0;
    void* st_ = nullptr;   ///< the stream everything but the staging ahead runs on
    void* blas_ = nullptr; ///< cuBLAS on st_, for the sparse attention
    void* ev_ = nullptr;   ///< marks the router's copy to the host
    std::vector<void*> dev_allocs_, host_allocs_;
    std::vector<void*> graphs_;   ///< every recorded graph, to destroy
    void* tail_ = nullptr;        ///< one token's launches from the last layer to the logits, recorded

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
    const uint8_t* tok_embd_ = nullptr;   ///< the token table in the mapped file, tok_row_ bytes a token
    size_t tok_row_ = 0;
    uint32_t tok_type_ = 0;

    uint8_t* dense_ = nullptr;   ///< the dense weights' arena
    size_t dense_bytes_ = 0, dense_at_ = 0;
    uint8_t* exp_ = nullptr;     ///< the routed experts' arena
    size_t exp_at_ = 0;
    float* scratch_ = nullptr;
    size_t scratch_at_ = 0;
    float* state_ = nullptr;     ///< every KDA layer's conv and delta-rule state
    size_t state_n_ = 0;
    int cached_ = 0;

    // device scratch
    float *r_ = nullptr, *cur_ = nullptr, *xn_ = nullptr, *ao_ = nullptr, *fo_ = nullptr, *mix_ = nullptr,
          *hcw_ = nullptr;
    float *proj_ = nullptr, *qkv_ = nullptr, *fa_ = nullptr, *g1_ = nullptr, *ga_ = nullptr, *g2_ = nullptr,
          *beta_ = nullptr, *gated_ = nullptr;
    float *t1_ = nullptr, *qr_ = nullptr, *q_ = nullptr, *t2_ = nullptr, *kv_ = nullptr, *qabs_ = nullptr,
          *kqv_ = nullptr, *vout_ = nullptr, *ikr_ = nullptr, *ik_ = nullptr, *ig_ = nullptr, *iq_ = nullptr,
          *iw_ = nullptr, *score_ = nullptr, *akeys_ = nullptr, *ascore_ = nullptr;
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
    void* up_ = nullptr;                          ///< the stream of the uploads that start before the routing
    void* ev_ahead_ = nullptr;                    ///< marks the last of them
    void* ev_free_ = nullptr;                     ///< marks a layer's last read of the staging buffer
    size_t stage_bytes_ = 0, chunk_bytes_ = 0;
    int stage_turn_ = 0;
    std::vector<int32_t> ids_, count_, first_, group_, round_;
    std::vector<uint8_t> sent_;

    // The cache follows what is written (follow()).  It is filled for the average answer; while one is written,
    // the experts it keeps routing to and the cache lacks are copied over the cached ones it uses least, by a
    // thread and a stream of their own.
    struct Swap;
    struct Pick {
        int il = -1, e = 0;
        float recent = 0;
        int64_t prior = 0;
    };
    std::unique_ptr<Swap[]> swap_;   ///< the experts on their way, opt_.expert_swaps at most
    int n_swap_ = 0, n_trade_ = 0;
    size_t slot_max_ = 0;            ///< the largest expert of the cache
    void* swap_st_ = nullptr;        ///< the stream of their uploads
    std::thread follower_;
    std::mutex swap_m_;
    std::condition_variable swap_cv_;
    bool swap_quit_ = false;
    std::string swap_error_;         ///< why the thread stopped; follow() throws it
    std::vector<float> recent_;      ///< per (layer, expert): its uses by written tokens, halved every kSwapHalf
    std::vector<uint8_t> coming_;    ///< per (layer, expert): on its way into the cache
    std::vector<Pick> in_, out_;     ///< per trade number: the expert to bring in, and the one to give up

    int64_t pos_ = 0;
    std::vector<float> probs_, pick_;
    std::vector<int64_t> use_, prior_;   ///< routed count per (layer, expert): this run, and the profile's
    Stats stats_;
};

}  // namespace strata::glm
