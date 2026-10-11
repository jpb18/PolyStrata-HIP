// src/poly/qwen35.cpp - the layers of Qwen3.5 and Qwen3.6 with routed experts.  The reference for the math is
// llama.cpp's qwen35moe graph.
#include "qwen35.hpp"

#include "runtime.hpp"
#include "strata/core/conversation_memory.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace strata::poly {
namespace {

constexpr int kConv = 4;   // what conv4_silu is written for; checked at load
constexpr float kMinWeightSum = 6.103515625e-5f;   // the floor ggml's MoE graph puts under the gate sum
constexpr size_t kScoreFloats = size_t(16) << 20;  // the attention's scores of the tokens it takes at a time

}  // namespace

void Qwen35::begin_pass() {
    blocks_.assign(layers_.size(), Block{});
    n_gdn_ = n_attn_ = 0;
}

void Qwen35::check_draft_block() {
    if (blocks_[(size_t) n_layer_].gdn)
        throw std::runtime_error("the block that guesses ahead is not one this engine runs; --no-draft loads "
                                 "the model without it");
}

void Qwen35::load_geometry() {
    load_shape();
    auto find = [&](const char* key) { return meta_in(*gg_, key); };
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported model: ") + what);
    };
    n_head_ = shape_i("attention.head_count");
    n_kv_ = shape_i("attention.head_count_kv");
    head_dim_ = shape_i("attention.key_length");
    const MetaValue* rot = find("rope.dimension_count");
    n_rot_ = rot ? (int) rot->u : head_dim_;
    s_dim_ = shape_i("ssm.state_size");
    k_heads_ = shape_i("ssm.group_count");
    v_heads_ = shape_i("ssm.time_step_rank");
    d_inner_ = shape_i("ssm.inner_size");
    conv_dim_ = 2 * k_heads_ * s_dim_ + d_inner_;

    need(n_kv_ > 0 && n_head_ % n_kv_ == 0 && n_head_ <= 256, "bad attention.head_count");
    need(head_dim_ > 0 && shape_i("attention.value_length") == head_dim_,
         "the attention's keys and values differ in length");
    need(n_rot_ > 0 && n_rot_ % 2 == 0 && n_rot_ <= head_dim_, "bad rope.dimension_count");
    need(shape_i("ssm.conv_kernel") == kConv, "ssm.conv_kernel is not 4");
    need(s_dim_ > 0 && s_dim_ % 32 == 0 && s_dim_ <= 256, "ssm.state_size is not a multiple of 32 up to 256");
    need(k_heads_ > 0 && v_heads_ % k_heads_ == 0 && d_inner_ == v_heads_ * s_dim_, "bad ssm head counts");

    // The rotary embedding has an axis for time, two for an image and a spare one, and its sections say which
    // pairs turn by which.  A token of text has its position on the first three and none on the spare one, so
    // the kernel turns every pair by the position: that is wrong for a pair the sections leave to the spare axis.
    const MetaValue* sec = find("rope.dimension_sections");
    need(sec && sec->type == MetaType::ARRAY && sec->items.size() >= 3, "no rope.dimension_sections");
    int sections[4] = {}, covered = 0;
    for (size_t i = 0; i < 4 && i < sec->items.size(); ++i) covered += sections[i] = (int) sec->items[i].u;
    need(covered > 0, "empty rope.dimension_sections");
    for (int pair = 0; pair < n_rot_ / 2; ++pair) {
        const int s = pair % covered;
        need(s < 3 * sections[s % 3], "the rotary embedding leaves pairs to an axis text has no position on");
    }
    sections_ = covered;
    theta_scale_ = std::pow(shape_f("rope.freq_base"), -2.f / (float) n_rot_);

    // this architecture puts no limit on a SwiGLU's inputs
    clamp_exp_.assign((size_t) n_layer_ + 1, INFINITY);
    clamp_shexp_ = clamp_exp_;
}

void Qwen35::load_layer(int il) {
    Layer& L = layers_[(size_t) il];
    Block& B = blocks_[(size_t) il];
    auto f32 = [&](const char* n, int64_t a, int64_t b = 1) { return upload(tensor(il, n), blk(il, n), a, b, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(il, n), a, b, kAnyMat); };

    L.attn_norm = f32("attn_norm.weight", n_embd_);
    L.ffn_norm = f32("post_attention_norm.weight", n_embd_);
    B.gdn = tensor(il, "attn_qkv.weight") != nullptr;
    L.fixed = true;   // the attention's launches read the position on the card when they are recorded
    if (B.gdn) {
        B.wqkv = mat("attn_qkv.weight", n_embd_, conv_dim_);
        B.wz = mat("attn_gate.weight", n_embd_, d_inner_);
        B.conv = f32("ssm_conv1d.weight", kConv, conv_dim_);
        B.dt = f32("ssm_dt.bias", v_heads_);
        B.a = f32("ssm_a", v_heads_);
        B.alpha = mat("ssm_alpha.weight", n_embd_, v_heads_);
        B.beta = mat("ssm_beta.weight", n_embd_, v_heads_);
        B.o_norm = f32("ssm_norm.weight", s_dim_);
        B.wo = mat("ssm_out.weight", d_inner_, n_embd_);
        ++n_gdn_;
    } else {
        // the query projection gives every head its query and then the gate of its output
        B.wq = mat("attn_q.weight", n_embd_, (int64_t) 2 * n_head_ * head_dim_);
        B.wk = mat("attn_k.weight", n_embd_, (int64_t) n_kv_ * head_dim_);
        B.wv = mat("attn_v.weight", n_embd_, (int64_t) n_kv_ * head_dim_);
        B.q_norm = f32("attn_q_norm.weight", head_dim_);
        B.k_norm = f32("attn_k_norm.weight", head_dim_);
        B.wo = mat("attn_output.weight", (int64_t) n_head_ * head_dim_, n_embd_);
        ++n_attn_;
    }
    load_moe(il);
    B.sh_gate_inp = f32("ffn_gate_inp_shexp.weight", n_embd_);
}

size_t Qwen35::recurrent_floats() const {
    return size_t(n_gdn_) * (size_t(3) * conv_dim_ + size_t(v_heads_) * s_dim_ * s_dim_);
}

int Qwen35::widest() const { return std::max(d_inner_, n_head_ * head_dim_); }

// KV streaming (AMD_KV_STREAMING.md): the positions whose rows stay in VRAM.  Every token's attention reads every
// row, so a row in RAM costs its 2 KiB a layer over PCIe for each token (12.6 GB/s against VRAM's 300 on an RX 9060
// XT), and a row in VRAM is read more often than any expert the cache gives up for it.  So `auto` keeps every row
// in VRAM when they fit beside what the engine needs besides the expert cache, and otherwise as many as fit.
int64_t Qwen35::plan_kv() const {
    const int64_t ctx = opt_.max_context;
    if (opt_.kv_resident > 0) return std::min<int64_t>(opt_.kv_resident, ctx);
    if (opt_.kv_resident == 0 || n_attn_ == 0) return ctx;
    size_t free_b = 0, total_b = 0;
    ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
    const size_t reserve = opt_.vram_reserve_mib >= 0 ? size_t(opt_.vram_reserve_mib) << 20
                                                      : std::clamp<size_t>(total_b >> 24, 512, 1024) << 20;
    // what is still to be allocated: the scratch (about 1.4 MiB a prompt token), the attention's scores (64 MiB),
    // the stage of a split cache (under 64 MiB) and the recorded launches and libraries' work (some 256 MiB)
    const size_t later = (size_t) rows_ * (size_t(3) << 19) + (size_t(384) << 20);
    const size_t per = (size_t) n_attn_ * 2 * n_kv_ * head_dim_ * sizeof(uint16_t);
    const size_t room = free_b > reserve + later ? free_b - reserve - later : 0;
    if ((size_t) ctx * per <= room) return ctx;
    return std::min<int64_t>(ctx, (int64_t) (room / per) / 256 * 256);
}

// The rows of a block from kv_split_ on, for the kernels (null: every row is in VRAM).
const KvSpill* Qwen35::spill(const Block& B) {
    if (!B.keys_h) return nullptr;
    spill_ = KvSpill{B.keys_h, B.vals_h, (long long) kv_split_, kstage_, vstage_, stage_rows_};
    return &spill_;
}

void Qwen35::kv_budget(Budget& b) const {
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t), caches = size_t(n_attn_) * 2;
    b.kv_cells_vram = kv_split_;
    b.kv_cells_host = opt_.max_context - kv_split_;
    b.kv_vram = caches * (size_t) kv_split_ * row;
    b.kv_host = caches * (size_t) b.kv_cells_host * row;
    b.kv_stage = kstage_ ? 2 * (size_t) stage_rows_ * row : 0;
}

// The attention layers' caches, and each recurrent layer's convolution and delta-rule state in state_.
void Qwen35::alloc_caches() {
    const size_t past_n = size_t(3) * conv_dim_, s_n = size_t(v_heads_) * s_dim_ * s_dim_;
    const size_t width = (size_t) n_kv_ * head_dim_;
    kv_split_ = plan_kv();
    const size_t rows_n = (size_t) kv_split_ * width, host_n = (size_t) (opt_.max_context - kv_split_) * width;
    uint16_t* kv = (uint16_t*) dmalloc(size_t(n_attn_) * 2 * rows_n * sizeof(uint16_t), "the attention cache");
    uint16_t *hkv = nullptr, *hkv_dev = nullptr;
    if (host_n > 0 && n_attn_ > 0) {
        const size_t bytes = size_t(n_attn_) * 2 * host_n * sizeof(uint16_t);
        // Pinned memory is never paged out, and the experts outside VRAM are read from the file's pages: the rows
        // in RAM leave kv_ram_floor_mib of it available, or stay under kv_ram_mib.
        const std::optional<uint64_t> avail = strata::core::conversation_available_memory();
        const uint64_t floor = (uint64_t) std::max<int64_t>(opt_.kv_ram_floor_mib, 0) << 20;
        const uint64_t cap = opt_.kv_ram_mib >= 0 ? (uint64_t) opt_.kv_ram_mib << 20
                             : avail ? (*avail > floor ? *avail - floor : 0) : UINT64_MAX;
        if (bytes > cap) {
            const size_t per = size_t(n_attn_) * 2 * width * sizeof(uint16_t);
            throw std::runtime_error(
                "KV streaming: the " + std::to_string(opt_.max_context - kv_split_) + " positions outside VRAM need " +
                std::to_string(bytes >> 20) + " MiB of pinned RAM and " +
                (opt_.kv_ram_mib >= 0 ? "--kv-ram-mib allows " : "what leaves the RAM floor available is ") +
                std::to_string(cap >> 20) + " MiB: a context of " + std::to_string(kv_split_ + (int64_t) (cap / per)) +
                " fits with these settings");
        }
        hkv = (uint16_t*) hmalloc_mapped(bytes, "the attention cache's rows in RAM", (void**) &hkv_dev);
    }
    float* st = state_;
    for (Block& B : blocks_) {
        if (B.gdn) {
            B.past = st;
            st += past_n;
            B.s = st;
            st += s_n;
        } else {
            B.keys = kv;
            kv += rows_n;
            B.vals = kv;
            kv += rows_n;
            if (hkv) {
                B.keys_hp = hkv;
                B.keys_h = hkv_dev;
                hkv += host_n;
                hkv_dev += host_n;
                B.vals_hp = hkv;
                B.vals_h = hkv_dev;
                hkv += host_n;
                hkv_dev += host_n;
            }
        }
    }
    chunk_ = (int) std::clamp<size_t>(kScoreFloats / ((size_t) n_head_ * opt_.max_context), 1, (size_t) rows_);
    if (host_n > 0 && n_attn_ > 0) {
        // a prompt batch copies the blocks of rows that reach RAM here: as many rows as its widest block
        if (rows_ >= kStageRows) {
            stage_rows_ = (int) std::min<int64_t>(
                opt_.max_context,
                attend_batch_block((size_t) chunk_ * n_head_ * opt_.max_context, kStageRows, n_head_));
            kstage_ = (uint16_t*) dmalloc((size_t) stage_rows_ * width * sizeof(uint16_t), "the attention's stage");
            vstage_ = (uint16_t*) dmalloc((size_t) stage_rows_ * width * sizeof(uint16_t), "the attention's stage");
        }
        std::fprintf(stderr, "strata-poly: KV streaming: positions 0 to %lld of the attention caches in VRAM (%.2f GiB), "
                             "%lld to %d in pinned RAM (%.2f GiB)\n",
                     (long long) kv_split_ - 1, double(size_t(n_attn_) * 2 * rows_n * 2) / 1073741824.0,
                     (long long) kv_split_, opt_.max_context - 1,
                     double(size_t(n_attn_) * 2 * host_n * 2) / 1073741824.0);
    }
    axes_ = (int32_t*) dmalloc((size_t) rows_ * 3 * sizeof(int32_t), "a call's rotary positions");
    h_axes_.resize((size_t) rows_ * 3);
}

// Per-token buffers hold rows_ tokens, row after row.
void Qwen35::assign_scratch() {
    const size_t n = (size_t) rows_, heads = n * n_head_ * head_dim_, kv = n * n_kv_ * head_dim_;
    proj_ = fbuf(n * conv_dim_);
    qkv_ = fbuf(n * conv_dim_);
    z_ = fbuf(n * d_inner_);
    alpha_ = fbuf(n * v_heads_);
    beta_ = fbuf(n * v_heads_);
    gated_ = fbuf(n * d_inner_);

    qg_ = fbuf(2 * heads);
    q_ = fbuf(heads);
    kraw_ = fbuf(kv);
    k_ = fbuf(kv);
    v_ = fbuf(kv);
    att_ = fbuf(heads);
    score_ = fbuf((size_t) chunk_ * n_head_ * opt_.max_context);
    sg_ = fbuf(n);
    if (rows_ >= kStageRows) awork_ = fbuf(heads + heads / 2 + 1 + 3 * n * n_head_);   // a prompt batch's attention
}

// A recurrent layer's convolution and delta-rule state lie one behind the other in state_ (alloc_caches()).
void Qwen35::save_layer(int il, int64_t, uint8_t* dst) {
    const Block& B = blocks_[(size_t) il];
    if (!B.past) return;
    const size_t n = size_t(3) * conv_dim_ + size_t(v_heads_) * s_dim_ * s_dim_;
    ck(cudaMemcpy(dst + (size_t) (B.past - state_) * sizeof(float), B.past, n * sizeof(float),
                  cudaMemcpyDeviceToHost),
       "save state");
}

size_t Qwen35::row_bytes() const { return (size_t) n_attn_ * 2 * n_kv_ * head_dim_ * sizeof(uint16_t); }

void Qwen35::save_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context)
        throw std::logic_error("save_rows: the rows are outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t), bytes = (size_t) n * row;
    // the rows below kv_split_ from VRAM, the others from pinned RAM
    const int64_t below = std::clamp<int64_t>(kv_split_ - first, 0, n);
    for (const Block& B : blocks_) {
        if (B.gdn || bytes == 0) continue;
        for (int c = 0; c < 2; ++c) {
            const uint16_t* cache = c ? B.vals : B.keys;
            if (below > 0)
                ck(cudaMemcpy(dst, (const uint8_t*) cache + (size_t) first * row, (size_t) below * row,
                              cudaMemcpyDeviceToHost),
                   "save rows");
            if (below < n)
                std::memcpy(dst + (size_t) below * row,
                            (const uint8_t*) (c ? B.vals_hp : B.keys_hp) + (size_t) (first + below - kv_split_) * row,
                            (size_t) (n - below) * row);
            dst += bytes;
        }
    }
}

void Qwen35::load_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: the rows are outside what was saved");
    ck(cudaStreamSynchronize(cs(st_)), "load rows");
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t), bytes = (size_t) n * row;
    const size_t skip = (size_t) (first - from) * row, all = (size_t) count * row;
    const int64_t below = std::clamp<int64_t>(kv_split_ - first, 0, n);
    for (const Block& B : blocks_) {
        if (B.gdn || bytes == 0) continue;
        for (int c = 0; c < 2; ++c) {
            uint16_t* cache = c ? B.vals : B.keys;
            if (below > 0)
                ck(cudaMemcpy((uint8_t*) cache + (size_t) first * row, src + skip, (size_t) below * row,
                              cudaMemcpyHostToDevice),
                   "load rows");
            if (below < n)
                std::memcpy((uint8_t*) (c ? B.vals_hp : B.keys_hp) + (size_t) (first + below - kv_split_) * row,
                            src + skip + (size_t) below * row, (size_t) (n - below) * row);
            src += all;
        }
    }
}

// The gated delta net.  One projection gives the queries, keys and values, which a convolution over the last
// four tokens mixes; another gives the gate of the output.
void Qwen35::gdn(Block& B, int n) {
    lin(B.wqkv, xn_, proj_, n, xq_);
    lin(B.wz, xn_, z_, n, xq_);
    lin(B.alpha, xn_, alpha_, n, xq_);
    lin(B.beta, xn_, beta_, n, xq_);
    conv4_silu(proj_, B.past, fp(B.conv), qkv_, conv_dim_, n, st_,
               snap_bits_ ? snap_at_ + (B.past - state_) : nullptr, rec_n_, snap_bits_);
    gdn_step(qkv_, alpha_, beta_, fp(B.dt), fp(B.a), B.s, z_, fp(B.o_norm), gated_, s_dim_, v_heads_, k_heads_, n,
             rms_eps_, st_, snap_bits_ ? snap_at_ + (B.s - state_) : nullptr, rec_n_, snap_bits_);
    lin(B.wo, gated_, ao_, n);
}

// A row's position on the rotary embedding's three axes, as llama.cpp's mtmd counts them.  Text has the same on
// all three and counts up.  A picture's rows share the position the picture starts at on the first axis, and add
// their line to it on the second and their column on the third; the text after a picture goes on as many
// positions later as the picture has lines or columns, whichever is more.
void Qwen35::turn_of(int64_t row, int32_t* thw) const {
    int64_t skipped = 0;   // the rows before this one that took no position of their own
    for (const Image& im : images_) {
        if (row < im.at) break;
        if (row < im.at + im.n) {
            const int64_t p = im.at - skipped, j = row - im.at;
            thw[0] = (int32_t) p;
            thw[1] = (int32_t) (p + j / im.nx);
            thw[2] = (int32_t) (p + j % im.nx);
            return;
        }
        skipped += im.n - std::max(im.nx, im.n / im.nx);
    }
    thw[0] = thw[1] = thw[2] = (int32_t) (row - skipped);
}

// The rotary positions of n rows from row `pos`.  Rows of text alone count up from *first, and the result is
// null.  With a picture's rows among them the result is the positions of each row on the three axes, on the card.
const int32_t* Qwen35::turns(int64_t pos, int n, int64_t* first) {
    int32_t thw[3];
    turn_of(pos, thw);
    *first = thw[0];
    const bool pictures = std::any_of(images_.begin(), images_.end(), [&](const Image& im) {
        return im.at < pos + n && im.at + im.n > pos;
    });
    if (!pictures) return nullptr;
    if (axes_pos_ != pos || axes_n_ != n || axes_set_ != images_set_) {
        for (int t = 0; t < n; ++t) turn_of(pos + t, h_axes_.data() + (size_t) t * 3);
        // the launches that read the positions of the call before this one have to be through
        ck(cudaStreamSynchronize(cs(st_)), "rotary positions");
        ck(cudaMemcpy(axes_, h_axes_.data(), (size_t) n * 3 * sizeof(int32_t), cudaMemcpyHostToDevice),
           "rotary positions");
        axes_pos_ = pos;
        axes_n_ = n;
        axes_set_ = images_set_;
    }
    return axes_;
}

// The rows n tokens from position `pos` leave in an attention layer's caches: each head group's key, normed and
// turned by the position, and its value.  xn_ and xq_ hold the layer's input.
void Qwen35::attn_store(Block& B, int64_t pos, int n) {
    int64_t first;
    const int32_t* axes = turns(pos, n, &first);
    const long long* at = recording_ ? at_ : nullptr;   // a recorded launch: the tokens from what at_ holds
    lin(B.wk, xn_, kraw_, n, xq_);
    head_norm_rope(kraw_, head_dim_, fp(B.k_norm), k_, head_dim_, n_kv_, n_rot_, theta_scale_, first, n, rms_eps_, st_,
                   nullptr, axes, sections_, at ? at + 1 : nullptr);
    lin(B.wv, xn_, v_, n, xq_);
    if (at) {
        store_f16_at(k_, B.keys, n_kv_ * head_dim_, at, n, 0, st_, B.keys_h, kv_split_);
        store_f16_at(v_, B.vals, n_kv_ * head_dim_, at, n, 0, st_, B.vals_h, kv_split_);
    } else {
        store_f16(k_, B.keys, n_kv_ * head_dim_, pos, n, st_, B.keys_h, kv_split_);
        store_f16(v_, B.vals, n_kv_ * head_dim_, pos, n, st_, B.vals_h, kv_split_);
    }
}

// The recordings of an attention layer's head are numbered by the rows they are sized for.
int Qwen35::head_key(int il, int n) const { return blocks_[(size_t) il].gdn ? 0 : room_key(n); }

int64_t Qwen35::turn(int64_t row) const {
    int32_t thw[3];
    turn_of(row, thw);
    return thw[0];
}

void Qwen35::attn(Block& B, int64_t pos, int n) {
    int64_t first;
    const int32_t* axes = turns(pos, n, &first);
    const long long* at = recording_ ? at_ : nullptr;
    lin(B.wq, xn_, qg_, n, xq_);
    head_norm_rope(qg_, 2 * head_dim_, fp(B.q_norm), q_, head_dim_, n_head_, n_rot_, theta_scale_, first, n, rms_eps_,
                   st_, nullptr, axes, sections_, at ? at + 1 : nullptr);
    attn_store(B, pos, n);
    const float scale = 1.f / std::sqrt((float) head_dim_);
    // a prompt batch through cuBLAS; a few tokens with the kernels that give a token the same bits in any company
    if (n >= kStageRows)
        attend_batch(blas_, q_, B.keys, B.vals, pos, n, score_, (size_t) chunk_ * n_head_ * opt_.max_context, awork_,
                     att_, head_dim_, n_head_, n_kv_, scale, st_, 0, 0, false, spill(B));
    else
        attend(q_, B.keys, B.vals, pos, n, score_, chunk_, att_, head_dim_, n_head_, n_kv_, scale, st_, 0, 0, false,
               at, at ? room(n) : 0, spill(B));
    mul_sigmoid(att_, qg_ + head_dim_, head_dim_, n * n_head_, 2 * head_dim_, false, st_);
    lin(B.wo, att_, ao_, n);
}

// A layer from its input to its router's copy.  A prompt batch's MoE (moe_rows) does its own routing.
void Qwen35::layer_head(Layer& L, int il, int n) {
    Block& B = blocks_[(size_t) il];
    rms_norm(r_, fp(L.attn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, n_embd_ * n, 1, st_);
    dbg("attn_in", il, xn_, n_embd_);
    if (B.gdn) gdn(B, n);
    else attn(B, pos_, n);
    dbg("attn", il, ao_, n_embd_);
    axpy(ao_, 1.f, r_, n * n_embd_, true, st_);

    rms_norm(r_, fp(L.ffn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, n_embd_ * n, 1, st_);
    dbg("ffn_in", il, xn_, n_embd_);
    if (!(mmq_ && n >= kStageRows)) moe_router(L, n);
}

void Qwen35::layer_tail(Layer&, int, int n) { axpy(fo_, 1.f, r_, n * n_embd_, true, st_); }

void Qwen35::output_rows(int first, int rows, float* h) {
    rms_norm(r_ + (size_t) first * n_embd_, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
}

// What ggml's MoE graph does for this model: a softmax over every expert's score, the n_used most likely, and
// their weights made to sum to one.
void Qwen35::route(int, const float* logits, int* ids, float* w, float* probs) const {
    float mx = logits[0];
    for (int e = 1; e < n_expert_; ++e) mx = std::max(mx, logits[e]);
    double all = 0.0;
    for (int e = 0; e < n_expert_; ++e) all += probs[e] = std::exp(logits[e] - mx);
    const float inv = (float) (1.0 / all);
    float sum = 0.f;
    for (int k = 0; k < n_used_; ++k) {
        int best = 0;
        for (int e = 1; e < n_expert_; ++e)
            if (probs[e] > probs[best]) best = e;
        ids[k] = best;
        w[k] = probs[best] * inv;
        sum += w[k];
        probs[best] = -1.f;
    }
    const float denom = std::max(sum, kMinWeightSum);
    for (int k = 0; k < n_used_; ++k) w[k] /= denom;
}

#if defined(STRATA_USE_HIP)
// route() above is kernels::route_topk_softmax()'s routing (Engine::card_moe()).
float Qwen35::card_route_floor() const { return kMinWeightSum; }
#endif

// The shared expert, times the sigmoid of its own gate.  It runs on the GPU while the host routes.
void Qwen35::moe_shared(const Layer& L, int il, int n) {
    lin(L.sh_gate, xn_, g_, n, xq_);
    lin(L.sh_up, xn_, u_, n, xq_);
    swiglu_clamp(g_, u_, h_, n * n_ff_shexp_, clamp_shexp_[(size_t) il], st_);
    lin(L.sh_down, h_, fo_, n);
    lin(blocks_[(size_t) il].sh_gate_inp, xn_, sg_, n);
    mul_sigmoid(fo_, sg_, n_embd_, n, 1, true, st_);
}

// The draft block is one more attention layer.
void Qwen35::block_attend(int64_t pos, int n) { attn(blocks_[(size_t) n_layer_], pos, n); }
void Qwen35::block_rows(int64_t pos, int n) { attn_store(blocks_[(size_t) n_layer_], pos, n); }

}  // namespace strata::poly
