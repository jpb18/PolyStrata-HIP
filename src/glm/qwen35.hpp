// src/glm/qwen35.hpp - Qwen3.5 and Qwen3.6 with routed experts (llama.cpp architecture "qwen35moe"), and the models
// trained from them.
//
// One residual stream.  Three layers in four have a recurrent attention, the gated delta net; the fourth has
// attention over the keys and values of every token so far, with a rotary embedding and a gate on its output.
// Every layer's FFN is routed experts with a gated shared one.  One more block after the model's, with the plain
// attention, guesses a token ahead.
#pragma once

#include "engine.hpp"

namespace strata::glm {

class Qwen35 final : public Engine {
public:
    size_t row_bytes() const override;
    void save_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;

protected:
    void load_geometry() override;
    void load_layer(int il) override;
    void begin_pass() override;
    void check_draft_block() override;
    size_t recurrent_floats() const override;
    void alloc_caches() override;
    void assign_scratch() override;
    int widest() const override;
    int streams() const override { return 1; }
    void layer_head(Layer& L, int il, int n) override;
    void layer_tail(Layer& L, int il, int n) override;
    void output_rows(int first, int rows, float* h) override;
    void route(int il, const float* logits, int* ids, float* w) override;
    void moe_shared(const Layer& L, int il, int n) override;
    void block_attend(int64_t pos, int n) override;
    void block_rows(int64_t pos, int n) override;

private:
    /// A layer's weights beside the ones every architecture has (Layer), and its places in the state and caches.
    struct Block {
        bool gdn = false;   ///< the gated delta net; else attention over the cache
        Dev wo;
        // gated delta net
        Dev wqkv, wz, conv, dt, a, alpha, beta, o_norm;
        float* past = nullptr;   ///< the convolution's: 3 * conv_dim
        float* s = nullptr;      ///< v_heads * s_dim * s_dim
        // attention
        Dev wq, wk, wv, q_norm, k_norm;
        uint16_t* keys = nullptr;   ///< max_context * n_kv * head_dim, fp16
        uint16_t* vals = nullptr;
        Dev sh_gate_inp;   ///< the shared expert's gate
    };

    void gdn(Block& B, int n);
    void attn(Block& B, int64_t pos, int n);
    void attn_store(Block& B, int64_t pos, int n);

    std::vector<Block> blocks_;   ///< one a layer of layers_
    int n_head_ = 0, n_kv_ = 0, head_dim_ = 0, n_rot_ = 0;
    float theta_scale_ = 0;   ///< a rotary pair turns by position * theta_scale^pair
    int s_dim_ = 0, k_heads_ = 0, v_heads_ = 0, d_inner_ = 0, conv_dim_ = 0;
    int n_gdn_ = 0, n_attn_ = 0;
    int chunk_ = 1;   ///< tokens the attention takes at a time

    // device scratch
    float *proj_ = nullptr, *qkv_ = nullptr, *z_ = nullptr, *alpha_ = nullptr, *beta_ = nullptr, *gated_ = nullptr;
    float *qg_ = nullptr, *q_ = nullptr, *kraw_ = nullptr, *k_ = nullptr, *v_ = nullptr, *att_ = nullptr,
          *score_ = nullptr, *sg_ = nullptr;

    std::vector<float> probs_;   ///< route()'s scratch
};

}  // namespace strata::glm
