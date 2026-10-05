// src/glm/glm5next.hpp - GLM 5.3 Flash (llama.cpp architecture "glm5-next").
//
// 45 layers with four residual streams joined by hyper-connections.  A layer's attention is a recurrent one (KDA)
// or a sparse one over cached latents with an indexer (DSA); its FFN is dense in the first layers and routed
// experts with a shared one after them.  One more block after the model's guesses a token ahead.
#pragma once

#include "engine.hpp"

namespace strata::glm {

class Glm5Next final : public Engine {
public:
    size_t row_bytes() const override;
    void save_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;
    int32_t draft(int32_t token, int i, float* prob) override;

protected:
    void load_geometry() override;
    void load_layer(int il) override;
    void begin_pass() override;
    void check_draft_block() override;
    void alloc_state() override;
    void assign_scratch() override;
    int streams() const override;
    void layer_head(Layer& L, int il, int n) override;
    void layer_tail(Layer& L, int il, int n) override;
    void output_rows(int first, int rows, float* h) override;
    void route(const Layer& L, const float* logits, int* ids, float* w) override;
    void moe_shared(const Layer& L, int il, int n) override;
    void draft_rows(const int32_t* tokens, int first, int n) override;
    const Names* tensor_alias(const char* name) const override;
    const Names* meta_alias(const char* key) const override;

private:
    void kda(Layer& L, int n);
    void dsa(Layer& L, int64_t pos, int n);
    void dsa_store(Layer& L, int64_t pos, int n);
    void dsa_select(Layer& L, int64_t pos, int first, int n);
    void dense_ffn(Layer& L, int il, int n);
    float* draft_input(const int32_t* tokens, const float* h, int n);

    int n_head_ = 0, kda_dim_ = 0, d_inner_ = 0, n_ff_dense_ = 0, n_kda_ = 0, n_dsa_ = 0;
    int q_rank_ = 0, kv_rank_ = 0, mla_k_ = 0, mla_v_ = 0;
    int idx_heads_ = 0, idx_dim_ = 0, idx_top_k_ = 0, kpool_ = 0;
    int sinkhorn_ = 0;
    int score_stride_ = 0, sel_stride_ = 0;   ///< one token's indexer scores, and its selected cache rows
    int64_t gate_rank_ = 0;   ///< the widest low-rank gate projection of the KDA layers
    float hc_eps_ = 0, norm_eps_ = 0, kda_lb_ = 0, w_scale_ = 1;
    bool w_norm_ = false;

    // device scratch
    float *mix_ = nullptr, *hcw_ = nullptr;
    float *proj_ = nullptr, *qkv_ = nullptr, *fa_ = nullptr, *g1_ = nullptr, *ga_ = nullptr, *g2_ = nullptr,
          *beta_ = nullptr, *gated_ = nullptr;
    float *t1_ = nullptr, *qr_ = nullptr, *q_ = nullptr, *t2_ = nullptr, *kv_ = nullptr, *qabs_ = nullptr,
          *kqv_ = nullptr, *vout_ = nullptr, *ikr_ = nullptr, *ik_ = nullptr, *ig_ = nullptr, *iq_ = nullptr,
          *iw_ = nullptr, *score_ = nullptr, *akeys_ = nullptr, *ascore_ = nullptr;
    int32_t* sel_ = nullptr;

    // pinned host buffers
    float* h_score_ = nullptr;
    int32_t* h_sel_ = nullptr;

    std::vector<float> probs_, pick_;   ///< route()'s scratch
};

}  // namespace strata::glm
