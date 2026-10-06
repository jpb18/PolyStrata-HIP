// src/poly/engine.hpp - the engine that runs a model from its GGUF.
//
// The engine reads the GGUF in place, in one file or in the numbered files of a split one.  Everything but the
// routed experts goes to VRAM; as many routed experts as fit go there too, and the rest are computed on the CPU
// from the mapped file while the GPU does its share.
//
// Engine is what every architecture shares.  An architecture is a class derived from it (glm5next.hpp, qwen35.hpp,
// gemma4.hpp) that reads the model's shape, loads a layer's weights and composes a layer from the kernels;
// Engine::open() picks it by the file's general.architecture.
#pragma once

#include "cpu_experts.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace strata {
struct TensorInfo;
struct MetaValue;
}

namespace strata::poly {

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
    int window = 4;               ///< the most tokens window() takes: the known one and guesses after it; 1: none
    bool draft = true;            ///< run the block that guesses ahead, when there is one and window is above 1
    std::string draft_model;      ///< a file of the same model to take that block from, when the model's has none;
                                  ///< for an architecture that guesses with a model of its own, that model's file
};

/// A thing a model's file may call by other names than the engine does: converters differ.
struct Names {
    const char* name;
    const char* other[2];
};

/// A tensor in VRAM in its GGUF form.  n_in is the contiguous axis.
struct Dev {
    void* p = nullptr;
    uint32_t type = 0;
    int n_in = 0, n_out = 0;
};

/// What a layer has in every architecture.  Its attention's weights, and a dense FFN's, are the architecture's.
struct Layer {
    bool moe = false;
    bool fixed = false;   ///< one token's launches can be recorded and replayed for the next ones (eval()): they
                          ///< are the same for every token, or read the position on the card (Engine::at_)
    Dev attn_norm, ffn_norm;   ///< the norms ahead of the attention and of the FFN
    // MoE
    Dev gate_inp, sh_gate, sh_up, sh_down;
    const uint8_t* exp_gate = nullptr;  ///< n_expert experts back to back, in the mapped file
    const uint8_t* exp_up = nullptr;
    const uint8_t* exp_down = nullptr;
    uint32_t gu_type = 0, d_type = 0;
    size_t gu_bytes = 0, d_bytes = 0;  ///< one expert's gate (or up), and its down
    size_t gu_stride = 0;              ///< from an expert's gate to the next one's: twice gu_bytes in a file that
                                       ///< keeps an expert's gate and up in one tensor
    std::vector<uint8_t*> slot;        ///< per expert: gate | up | down in VRAM, or null
    size_t slot_bytes = 0;             ///< from a slot of the layer to the one behind it: whole blocks of both types
    std::vector<int32_t> ahead;        ///< experts a full prompt batch stages before it routes
    int trade = -1;                    ///< layers with the same number can take each other's slots: same size
    // What a call of a few tokens hands the card, recorded (Engine::run()): by the call's tokens, and the head
    // also by Engine::head_key()
    std::vector<void*> heads;          ///< up to the router
    std::vector<void*> shared;         ///< the shared expert
    std::vector<void*> tail;           ///< after the experts
};

/// A routed expert is run from the VRAM cache or on the CPU when tokens come a few at a time, as an answer's do
/// (gpu_experts, cpu_experts), and in a prompt batch from the cache or the staging buffer (batch_*).
struct Stats {
    int64_t tokens = 0, gpu_experts = 0, cpu_experts = 0, batch_cached = 0, batch_staged = 0;
    int64_t swapped_experts = 0;   ///< experts that took a cached one's slot (follow())
    double cpu_ms = 0, route_wait_ms = 0;
};

class Mmq;

/// A picture in a sequence: n rows that stand where n tokens would, each the vision encoder's output for a patch,
/// as wide as a token's embedding.  The patches come line after line, nx to a line.
struct Image {
    int64_t at = 0;   ///< the sequence's row its first one takes
    int n = 0, nx = 0;
    const float* rows = nullptr;
};

class Engine {
public:
    Engine();
    virtual ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    /// The engine of the architecture the model file names, loaded.  Throws std::runtime_error, as do eval and
    /// the other calls that touch the device.
    static std::unique_ptr<Engine> open(const Options& o);
    void load(const Options& o);

    int n_vocab() const { return n_vocab_; }
    int n_embd() const { return n_embd_; }
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

    /// Whether the architecture reads pictures.
    virtual bool sees() const { return false; }
    /// The pictures of the sequence, in its order, in place of the ones set before.  eval() reads a row that lies
    /// in one from the picture and not from its token, so their rows must stay where they are while tokens are
    /// read.  A sequence that is come back to must get the pictures it was read with.
    void images(const Image* im, int n);
    /// Of m rows from row `at`, how many the next eval() takes.  An architecture whose pictures' rows see each
    /// other takes a picture in a call of its own, which can be more than m.
    virtual int64_t whole(int64_t at, int64_t m) const { (void) at; return m; }

    /// Guessed tokens.  window() feeds t tokens from pos(): the first is the one that is known, each of the others a
    /// guess at what follows the one before it.  logits_of(i) then holds the scores of the token after the i-th, so
    /// the guesses can be checked, and keep(n) makes the first n tokens part of the sequence and forgets the rest.
    /// What a kept token computed is what eval() would have computed for it.
    int max_window() const { return opt_.window; }
    void window(const int32_t* tokens, int t);
    const float* logits_of(int i) const { return h_wlogits_ + (size_t) i * n_vocab_; }
    void keep(int n);

    /// Some files carry a block that guesses ahead: one layer that takes a token and the model's output for the
    /// token before it, and gives what the model would likely say after the token.  drafts(): it is there and runs.
    /// draft(token, 0, &p) takes the token at pos(), picked and not fed yet, and returns the guess at the one after
    /// it; draft(guess, i, &p) for i = 1, 2, ... then guesses on from the i-th guess, with the block's own output
    /// in the model's place.  p is the probability the block gives what it returns.  pos() + i must be inside the
    /// context.  logits() is left alone.  An architecture may guess with a small model of its own instead of a
    /// block (draft_step()); the calls are the same.
    bool drafts() const { return draft_; }
    int32_t draft(int32_t token, int i, float* prob);
    /// Whether a call since the last time this was asked recorded launches for replaying, which takes far longer
    /// than the call does afterwards: its time says nothing of the calls to come.
    bool took_record() { return std::exchange(recorded_, false); }

    // A sequence's state, to leave the sequence and come back to it.  The recurrent layers have a state that
    // every token changes.  Each token also leaves a row in the attention caches, which stays as it was written.
    size_t state_bytes() const { return state_n_ * sizeof(float) + extra_bytes(); }
    virtual size_t row_bytes() const = 0;
    /// The recurrent state at pos(), state_bytes() long.
    void save_state(uint8_t* dst);
    /// Back to position `pos` with the recurrent state it had there.  The rows before pos must be the sequence's.
    void load_state(const uint8_t* src, int64_t pos);
    /// The rows of positions [first, first + n), n * row_bytes() long: layer after layer, a layer's rows together.
    virtual void save_rows(int64_t first, int64_t n, uint8_t* dst) = 0;
    /// Rows [first, first + n) out of what save_rows wrote for [from, from + count).
    virtual void load_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) = 0;

    const Stats& stats() const { return stats_; }
    int cached_experts() const { return cached_; }
    int total_experts() const { return n_moe_ * n_expert_; }
    double cached_gib() const { return double(exp_at_) / double(1ull << 30); }
    void save_profile() const;

protected:
    // ---- what an architecture supplies ----
    /// The model's shape from its metadata: load_shape() for what every architecture has, then its own.
    virtual void load_geometry() = 0;
    /// A layer's weights, to VRAM (upload()) or as places in the mapped file (load_moe()).  load() runs the layers
    /// through twice: begin_pass() is called before each.
    virtual void load_layer(int il) = 0;
    virtual void begin_pass() {}
    /// Throws when the draft block is not one the architecture runs.
    virtual void check_draft_block() {}
    /// The weights of what guesses ahead, to VRAM.  The engine's own: the draft block's parts beside its layer.
    virtual void load_drafter();
    /// One guess (draft()): the token through what guesses ahead, its scores into logits_.  hn_'s first row is the
    /// model's output for the token before pos(); the second is this step's to leave and the next step's to read.
    /// The engine's own runs the draft block.
    virtual void draft_step(int32_t token, int i);
    /// The floats of every recurrent layer's state together.  They are state_'s first.
    virtual size_t recurrent_floats() const = 0;
    /// The layers' places in state_, and the attention's caches: a row a position.
    virtual void alloc_caches() = 0;
    /// The scratch of the architecture's own kernels, a fbuf() each.  Called twice: the first time only counts.
    virtual void assign_scratch() = 0;
    /// The widest input lin() gets, beside the model's width and the FFNs'.
    virtual int widest() const = 0;
    /// The copies of a token's embedding a layer's input starts from.
    virtual int streams() const = 0;
    /// A layer from its input to its router's copy (a layer without routed experts: to its end).
    virtual void layer_head(Layer& L, int il, int n) = 0;
    /// The routed experts' sum (fo_) into the layer's output.
    virtual void layer_tail(Layer& L, int il, int n) = 0;
    /// The model's output for `rows` tokens from row `first` of the call, normed: what the output matrix takes.
    virtual void output_rows(int first, int rows, float* h) = 0;
    /// A token's experts and their weights from its scores at layer il's router.  tmp: 2 * n_expert_ floats of
    /// scratch.  A prompt batch's tokens are routed by several threads at once: it changes nothing of the engine's.
    virtual void route(int il, const float* logits, int* ids, float* w, float* tmp) const = 0;
    /// The shared expert, into fo_.  It runs on the GPU while the host routes.
    virtual void moe_shared(const Layer& L, int il, int n) = 0;
    /// The scores of a layer's router for n tokens, into rl_.  The engine's own: the router's matrix times the
    /// experts' input (xn_).
    virtual void router_scores(const Layer& L, int n);
    /// What is done to `rows` rows of logits on the card before they are copied out.
    virtual void finish_logits(float* logits, int rows) { (void) logits, (void) rows; }
    /// What a token's embedding is multiplied by on its way in.
    virtual float embed_scale() const { return 1.f; }
    /// What a sequence has beside its recurrent state and its rows, in bytes: it is saved and brought back with
    /// the state (save_state(), load_state()).  load_extra() gets the position the state is for.
    virtual size_t extra_bytes() const { return 0; }
    virtual void save_extra(uint8_t* dst) { (void) dst; }
    virtual void load_extra(const uint8_t* src, int64_t pos) { (void) src, (void) pos; }
    /// The draft block's attention for n tokens from position `pos`.  xn_ and xq_ hold its normed input; ao_ gets
    /// its output, and the tokens leave their rows in its caches.  block_rows() only leaves the rows.
    virtual void block_attend(int64_t pos, int n) { (void) pos, (void) n; }
    virtual void block_rows(int64_t pos, int n) { (void) pos, (void) n; }
    /// Which recording of a layer's head fits n tokens from pos().  Launches that are sized by the rows they read
    /// are recorded once for each size (room()), and the architecture numbers the sizes from 0.
    virtual int head_key(int il, int n) const { (void) il, (void) n; return 0; }
    /// The rotary position of a row of text: its place in the sequence, unless pictures before it took fewer
    /// positions than rows.
    virtual int64_t turn(int64_t row) const { return row; }
    /// The buffers that hold a row a token, and pos(), moved `by` rows on (back, when negative): a call for one
    /// token then works in the place of another row of a window (run()).  An architecture adds its own buffers
    /// that a layer's head leaves for the layer's tail.
    virtual void shift_rows(ptrdiff_t by);
    /// The other names of a tensor or a metadata key in the files of this architecture, or null.
    virtual const Names* tensor_alias(const char* name) const { (void) name; return nullptr; }
    virtual const Names* meta_alias(const char* key) const { (void) key; return nullptr; }

    // ---- what it builds on ----
    enum Want { kF32, kFloatMat, kAnyMat };
    Dev upload(const TensorInfo* t, const std::string& name, int64_t n_in, int64_t n_out, Want want,
               float (*fix)(float) = nullptr);
    const TensorInfo* tensor(int il, const char* name) const;
    const Shards& home(const TensorInfo& t) const;
    const MetaValue* meta_in(const Shards& g, const char* key) const;
    /// A number of the model's shape, which its file must have.
    int shape_i(const char* key) const;
    float shape_f(const char* key) const;
    void load_shape();
    void load_moe(int il, bool shared = true);
    void alloc();
    void shared_scratch();
    float* draft_input(const int32_t* tokens, int64_t row, const float* h, int n);
    void draft_rows(const int32_t* tokens, int first, int n);
    const Image* image_at(int64_t row) const;
    bool input(int32_t token, int64_t row, float* dst) const;
    void lin(const Dev& w, const float* x, float* y, int n, const void* xq = nullptr, int rows_per_x = 0);
    void fill_expert_cache();
    void plan_ahead(Layer& L, int il);
    void run(const int32_t* tokens, int n);
    void moe_router(const Layer& L, int n);
    bool moe(Layer& L, int il, int n, const std::function<void()>* first = nullptr);
    void moe_add(int n);
    template <class F>
    void* record(F&& work);
    void moe_rows(Layer& L, int il, int n);
    void stage(const Layer& L, const int32_t* experts, int m, void* stream);
    void follow(int written);
    void follow_loop();
    void embed(int32_t token, float* dst) const;
    void dbg(const char* name, int il, const float* dev, int n);
    float* fbuf(size_t n);
    int room(int n) const;
    int room_key(int n) const;
    void* dmalloc(size_t bytes, const char* what);
    void* hmalloc(size_t bytes, const char* what);

    Options opt_;
    std::unique_ptr<Shards> gg_;
    std::unique_ptr<Shards> dg_;      ///< Options::draft_model, while the draft block is read from it
    const Shards* block_ = nullptr;   ///< the file the draft block is in: gg_ or dg_
    std::unique_ptr<Workers> workers_;
    int node_ = -1;
    double node_share_ = 0;
    std::thread mapper_;                  ///< reads the experts' pages once after load, so they are mapped
    std::atomic<bool> map_quit_{false};
    void* st_ = nullptr;   ///< the stream everything but the staging ahead runs on
    void* blas_ = nullptr; ///< cuBLAS on st_
    void* ev_ = nullptr;   ///< marks the router's copy to the host: ev_row_'s first, or a window's row's own
    void* ev_row_[8] = {}; ///< one for each token a window can have
    std::vector<void*> dev_allocs_, host_allocs_;
    std::vector<void*> graphs_;   ///< every recorded graph, to destroy
    void* tail_ = nullptr;        ///< one token's launches from the last layer to the logits, recorded
    // A recorded launch cannot be given the position as an argument: it reads it here when it runs.  eval() writes
    // the token's place in the sequence and its rotary position before it replays the layers.
    long long* at_ = nullptr;     ///< on the card: {pos(), turn(pos())}, and the same for a window's later tokens
    long long* h_at_ = nullptr;   ///< pinned
    bool recording_ = false;      ///< a layer's launches are being recorded: positions are read from at_
    bool recorded_ = false;       ///< took_record()

    // geometry
    int n_embd_ = 0, n_layer_ = 0, n_vocab_ = 0;
    int n_ff_ = 0;            ///< a dense FFN's width, in the architectures that have one
    int n_expert_ = 0, n_used_ = 0, n_ff_exp_ = 0, n_ff_shexp_ = 0, n_moe_ = 0;
    int rows_ = 1;            ///< tokens the scratch is sized for
    float rms_eps_ = 0;
    bool gelu_ = false;       ///< the routed experts' activation: GELU of the gate, else SiLU within the layer's limits
    std::vector<float> clamp_exp_, clamp_shexp_;   ///< a layer's SwiGLU limits: the routed experts', the shared one's

    std::vector<Layer> layers_;   ///< the model's, and after them the draft block
    Dev out_norm_, output_;
    bool draft_ = false;
    bool block_layer_ = false;    ///< what guesses ahead is a block after the model's layers (layers_' last)
    Dev d_proj_, d_enorm_, d_hnorm_, d_head_;   ///< the draft block's input projection and norms, and its output norm
    const uint8_t* tok_embd_ = nullptr;   ///< the token table in the mapped file, tok_row_ bytes a token
    size_t tok_row_ = 0;
    uint32_t tok_type_ = 0;

    uint8_t* dense_ = nullptr;   ///< the dense weights' arena
    size_t dense_bytes_ = 0, dense_at_ = 0;
    uint8_t* exp_ = nullptr;     ///< the routed experts' arena
    size_t exp_at_ = 0;
    float* scratch_ = nullptr;
    size_t scratch_at_ = 0;
    float* state_ = nullptr;     ///< every recurrent layer's state; with a draft block, hn_'s first row
    size_t state_n_ = 0, rec_n_ = 0;   ///< all of it, and the recurrent layers' part
    // The draft block reads the model's output for the token before each of its tokens.  Row 0 is the output of
    // the last token of the sequence, and the outputs of the tokens of the call under way follow it.
    float* hn_ = nullptr;
    float* h_dlogits_ = nullptr;        ///< a guess's logits, pinned
    int32_t win_tok_[8] = {};           ///< the window's tokens, for keep()
    int64_t ahead_pos_ = -1;            ///< draft() has left the block's row of this position as its token leaves it,
    int32_t ahead_tok_ = 0;             ///< which is this one
    float* snap_ = nullptr;      ///< window(): the recurrent state after each token but the last, rec_n_ apart
    int n_snap_ = 0;             ///< how many of those the eval under way leaves
    int win_ = 0;                ///< the tokens of the window that keep() has not settled yet
    float *wlogits_ = nullptr, *h_wlogits_ = nullptr;   ///< a window's logits, token after token: VRAM and pinned
    int cached_ = 0;

    // device scratch: a layer's input streams, one stream, the normed input, the attention's and the FFN's output
    float *r_ = nullptr, *cur_ = nullptr, *xn_ = nullptr, *ao_ = nullptr, *fo_ = nullptr;
    float *g_ = nullptr, *u_ = nullptr, *h_ = nullptr, *d_ = nullptr, *cpu_ = nullptr, *rl_ = nullptr,
          *logits_ = nullptr;
    void *xq_ = nullptr, *q8_ = nullptr;

    // pinned host buffers
    float *h_r_ = nullptr, *h_rl_ = nullptr, *h_x_ = nullptr, *h_cpu_ = nullptr, *h_logits_ = nullptr;

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
    std::vector<float> route_tmp_;   ///< route()'s scratch, a share for each host thread
    struct Run {
        int c, j, m, most;   ///< the rows' class, its first cached expert and how many, the most rows one has
    };
    std::vector<Run> runs_;        ///< a layer's launches over cached experts whose slots lie one behind the other
    std::vector<int32_t> chan_;    ///< a layer's cached experts in the order of their slots
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
    std::vector<Pick> in_, out_;     ///< per trade number: the experts to bring in, and the ones to give up
    std::vector<size_t> trade_at_;   ///< per trade number: how many of its pairs follow() has taken

    int64_t pos_ = 0;
    std::vector<Image> images_;          ///< the sequence's pictures, by row
    uint64_t images_set_ = 0;            ///< counts images(): what was derived from an earlier list is stale
    std::vector<int64_t> use_, prior_;   ///< routed count per (layer, expert): this run, and the profile's
    Stats stats_;
};

}  // namespace strata::poly
