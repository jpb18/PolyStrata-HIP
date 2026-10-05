// src/glm/gemma4.hpp - Gemma 4 with routed experts (llama.cpp architecture "gemma4"), as 26B-A4B is.
//
// One residual stream and attention in every layer.  Five layers in six attend only the last tokens (a sliding
// window) and keep their keys and values in a ring; the sixth attends every token so far, with wider heads.  A
// layer's FFN is a dense GELU MLP and routed GELU experts side by side, each normed before the two are added,
// and the router reads the layer's residual, not the experts' input.  The output matrix is the token table, and
// the logits are capped by a tanh.
#pragma once

#include "engine.hpp"

namespace strata::glm {

class Gemma4 final : public Engine {
public:
    size_t row_bytes() const override;
    void save_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;

protected:
    void load_geometry() override;
    void load_layer(int il) override;
    void begin_pass() override;
    size_t recurrent_floats() const override { return 0; }
    void alloc_caches() override;
    void assign_scratch() override;
    int widest() const override;
    int streams() const override { return 1; }
    void layer_head(Layer& L, int il, int n) override;
    void layer_tail(Layer& L, int il, int n) override;
    void output_rows(int first, int rows, float* h) override;
    void route(int il, const float* logits, int* ids, float* w) override;
    void moe_shared(const Layer& L, int il, int n) override;
    void router_scores(const Layer& L, int n) override;
    void finish_logits(float* logits, int rows) override;
    float embed_scale() const override;
    size_t extra_bytes() const override;
    void save_extra(uint8_t* dst) override;
    void load_extra(const uint8_t* src, int64_t pos) override;

private:
    /// A layer's weights beside the ones every architecture has (Layer), and its caches.
    struct Block {
        bool swa = false;   ///< attends a sliding window; its caches are rings
        int dim = 0, n_kv = 0, n_rot = 0;   ///< a head's width, the heads that have keys of their own
        float theta_scale = 0;
        Dev wq, wk, wv, wo, q_norm, k_norm, post_attn;   ///< wv is absent where the values are the keys' projection
        uint16_t* keys = nullptr;   ///< rows of n_kv * dim, fp16: max_context of them, or ring_ in a window layer
        uint16_t* vals = nullptr;
        Dev ffn_gate, ffn_up, ffn_down;   ///< the dense MLP
        Dev post_mlp, pre_moe, post_moe, post_ffn, rscale;
        std::vector<float> escale;   ///< what each expert's output is multiplied by
        float out_scale = 1;
    };

    void attn(Block& B, int64_t pos, int n);
    void store(const float* x, uint16_t* cache, const Block& B, int64_t pos, int n);
    template <class F>
    void window_runs(int64_t pos, F&& f) const;

    std::vector<Block> blocks_;   ///< one a layer of layers_
    std::vector<uint8_t> swa_;    ///< per layer: a window layer
    std::vector<int> kv_;         ///< per layer: the heads with keys of their own
    int n_head_ = 0, dim_full_ = 0, dim_swa_ = 0, rot_full_ = 0, rot_swa_ = 0;
    float base_full_ = 0, base_swa_ = 0;
    int window_ = 0;   ///< the tokens a window layer attends, the token itself among them
    int ring_ = 0;     ///< rows of a window layer's caches: the window and a call's tokens
    int keep_ = 0;     ///< rows of a window layer a sequence's state keeps
    float softcap_ = 0;
    Dev freq_;         ///< the full layers' rotary frequency factors
    int chunk_ = 1;    ///< tokens the attention takes at a time

    // device scratch
    float *qraw_ = nullptr, *q_ = nullptr, *kraw_ = nullptr, *k_ = nullptr, *vraw_ = nullptr, *v_ = nullptr,
          *att_ = nullptr, *score_ = nullptr, *mlp_ = nullptr, *rin_ = nullptr;

    std::vector<float> probs_;   ///< route()'s scratch
};

}  // namespace strata::glm
