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

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace strata {
struct TensorInfo;
struct MetaValue;
}

namespace strata::poly {

class Shards;
struct Likeliest;

/// The format of the attention caches' keys or values (docs/AMD_HIP_KV_Q8.md): fp16, or ggml's Q8_0 (blocks of 32
/// int8 values and an fp16 scale), which only Qwen3.5/3.6 will have and which is not in the engine yet.
enum class KvType : uint8_t { kF16, kQ8_0 };
/// "fp16" or "q8_0", as --kv takes it and INFO kv= reports it.
const char* kv_name(KvType t);
/// The format --kv names; false for any other name.
bool kv_parse(const std::string& name, KvType* t);

struct Options {
    std::string model;
    int max_context = 8192;
    int threads = 0;              ///< 0: the CPUs this process may use
    int vram_reserve_mib = -1;    ///< left free after the expert cache is filled; below 0: by the card's size
    std::string profile;          ///< expert use counts; orders the expert cache, and save_profile() adds to it
    bool gpu_experts = true;
    bool dump = false;            ///< print a checksum of every stage, for comparing against llama.cpp
    // A prompt batch.  More tokens a call and a larger staging buffer read a prompt faster, and both take VRAM
    // that the expert cache, and so decoding, does without.
    int prompt_batch = 1024;      ///< prompt tokens one eval() takes; the scratch is about 1.4 MiB of VRAM a token
    int stage_mib = 2048;         ///< the buffer the experts that are not cached pass through: by default about
                                  ///< what a layer of Q2_K experts leaves out of the cache, and never more than
                                  ///< a layer's experts take
    int stage_ahead = 256;        ///< from this many tokens, a call routes to most of a layer's experts, and the
                                  ///< ones that are not cached start their copy before the layer has routed; in a
                                  ///< read of several calls every call does; 0: never
    int expert_swaps = 16;        ///< experts on their way into the cache at a time while an answer is written, each
                                  ///< to the slot of one the answer uses less; 0: the cache stays as it was loaded
    bool stay_near = true;        ///< with several NUMA nodes, keep the CPU threads on the one that holds the experts
    bool graphs = true;           ///< a token's fixed launches are recorded once, a CUDA graph a layer, and replayed
    int window = 0;               ///< the most tokens window() takes: the known one and guesses after it; 1: none;
                                  ///< 0: 8, or 4 where every guess keeps a copy of the recurrent state
    bool draft = true;            ///< run the block that guesses ahead, when there is one and window is above 1
    std::string draft_model;      ///< a file of the same model to take that block from, when the model's has none;
                                  ///< for an architecture that guesses with a model of its own, that model's file
    // KV streaming (AMD_KV_STREAMING.md; Qwen3.5 and Qwen3.6).  An attention layer's keys and values for the first
    // kv_resident positions stay in VRAM and the rest lie in pinned RAM, which the attention reads over PCIe for
    // every token.  The attention is the same to the bit; the time it takes for the rows in RAM is ~24 times what
    // it takes for rows in VRAM (12.6 against 300 GB/s on an RX 9060 XT).
    int kv_resident = 0;          ///< positions whose rows stay in VRAM; 0: every position (no streaming); -1: as
                                  ///< many as fit in VRAM beside the dense weights, the scratch and the reserve
                                  ///< (every position when they all fit: the expert cache takes what is left)
    int64_t kv_ram_mib = -1;      ///< the most RAM the rows outside VRAM may pin; -1: what leaves kv_ram_floor_mib
                                  ///< of physical RAM available when they are allocated
    int64_t kv_ram_floor_mib = 4096;
    // The attention caches' format (--kv, or --kv-k / --kv-v for one of the two, a test's way to tell their errors
    // apart).  Checked before anything is allocated: a format the model's architecture or this build does not have
    // is an error, never fp16 in its place.
    KvType kv_k = KvType::kF16, kv_v = KvType::kF16;
    // Tests (kv_q8_compare.py): the expert cache holds at most this many experts (0: as many as fit), so that two
    // runs whose free VRAM differs by a few MiB cache the same experts and so compute the same.
    int expert_cache_max = 0;
    // The fewest experts the expert cache may be cut to (0: none, as many as the reserve allows).  When the reserve
    // cannot be had with that many experts in VRAM the floor wins: the cache keeps them and the reserve stays short
    // by what the engine reports (reserve_short_mib(), the INFO line, POST /v1/vram).  AMD_HIP_VRAM_RESIDENCY.md: on
    // a card whose Windows budget drops once the process holds some 10 GiB, a cache that gave VRAM back was filled
    // again before every request, 4 s each.
    int expert_cache_min = 0;
};

/// Where the engine's memory went (Engine::budget()), in bytes.
struct Budget {
    size_t dense = 0;          ///< the dense weights in VRAM
    size_t experts = 0;        ///< the expert cache in VRAM
    size_t kv_vram = 0;        ///< the attention caches' rows in VRAM
    size_t kv_host = 0;        ///< ... in pinned RAM (KV streaming)
    size_t kv_stage = 0;       ///< VRAM a prompt batch copies the blocks of rows in RAM to
    size_t state = 0;          ///< the recurrent state and its copies for a window's guesses
    size_t vram_other = 0;     ///< the rest the engine allocated in VRAM: scratch, staging, tables
    size_t pinned_other = 0;   ///< the rest of its pinned RAM: staging, uploads, copies to the host
    int64_t kv_cells_vram = 0, kv_cells_host = 0;   ///< positions of a layer in each
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
    bool card = false;                 ///< HIP: a token on its own is routed on the card (Engine::card_moe())
    // What a call of a few tokens hands the card, recorded (Engine::run()): by the call's tokens, and the head
    // also by Engine::head_key()
    std::vector<void*> heads;          ///< up to the router
#if defined(STRATA_USE_HIP)
    std::vector<void*> card_heads;     ///< up to the router, routed on the card (its router's launches are others)
#endif
    std::vector<void*> shared;         ///< the shared expert
    std::vector<void*> tail;           ///< after the experts
};

/// A routed expert is run from the VRAM cache or on the CPU when tokens come a few at a time, as an answer's do
/// (gpu_experts, cpu_experts), and in a prompt batch from the cache or the staging buffer (batch_*).
struct Stats {
    int64_t tokens = 0, gpu_experts = 0, cpu_experts = 0, batch_cached = 0, batch_staged = 0;
    int64_t swapped_experts = 0;   ///< experts that took a cached one's slot (follow())
    int64_t staged_experts = 0, staged_bytes = 0;   ///< a prompt batch's experts through the staging buffer (stage())
    double cpu_ms = 0, route_wait_ms = 0;
    double stage_ms = 0;           ///< the host's time in stage(): its copies to pinned memory and waits for a buffer
};

/// The most likely token of n logits: the first of the largest.
int most_likely(const float* logits, int n);

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
    /// Feed n tokens of a prompt, none of them a picture's row, as eval() without logits would in calls of
    /// max_rows(): the same tokens a call.  With experts outside the cache it is faster than those calls, the
    /// more the more of them n is.  The result is the calls' own bit for bit, but for a call of fewer than
    /// stage_ahead tokens beside others: its experts' sums are split as a larger call's, not as on its own.
    ///
    /// A mark is a place inside the read where the caller would have stopped between two calls: `at` of the
    /// read's tokens come before it.  The calls end there as they would have, and `state`, when it is not null,
    /// gets what save_state() gives at that place.  The marks rise, each above 0 and below n.
    struct Mark {
        int at;
        uint8_t* state;
    };
    int max_read() const { return read_rows_; }
    /// `turn`, when given, is asked each time the read's tokens are through one more layer (or, where the read
    /// is its calls one by one, through one more call), with how many of how many that is.  False gives the read
    /// up: read() returns false, and what the engine holds from pos() on is spoiled, its recurrent state and a
    /// window layer's rows among it.  The caller goes back to a state saved before (load_state()) or resets.
    bool read(const int32_t* tokens, int n, const Mark* marks = nullptr, int n_marks = 0,
              const std::function<bool(int, int)>* turn = nullptr);

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
    /// STRATA_POLY_TRACE=2: prints the card's time for each part of the calls since the last time, and forgets it.
    void print_trace() const;
    /// STRATA_POLY_PREFILL_PROF=1 or 2 (HIP): prints the card's time for each part of the prompt batches since the
    /// last time, and forgets it.  Nothing without the variable.
    void print_prefill_prof() const;

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
    /// The cache as load() filled it, which is as large as resize_cache() makes it again.
    int full_experts() const { return full_cached_; }
    double full_gib() const { return double(full_bytes_) / double(1ull << 30); }
    /// Options::expert_cache_min as the last fill used it: the fewest experts the cache may be cut to.
    int cache_floor_experts() const { return floor_experts_; }
    /// Of the reserve the last fill was sized to leave, the MiB it could not have with the cache cut to its floor: 0
    /// without a floor, when the cache was left above the floor, and when the reserve was there.
    int reserve_short_mib() const { return reserve_short_mib_; }
    /// The VRAM the cache was last sized to leave free, and what is free now, in MiB.
    int reserve_mib() const { return reserve_mib_; }
    int64_t vram_free_mib() const;
    /// The expert cache sized again, to leave `mib` of VRAM free (below 0: what load() left), and filled with the
    /// experts used most so far.  It gets no larger than load() made it.  Between calls that feed tokens: the
    /// experts it gives up are computed on the CPU from then on, as any expert outside it.
    void resize_cache(int mib);
    /// Between requests: sizes the cache again when what the engine allocated since has left less than a quarter
    /// of the reserve free.  True when it did.
    bool mind_vram();
    void save_profile() const;
    /// Where the engine's memory went.
    Budget budget() const;
    /// The attention caches' format, as --kv names it: "fp16", "q8_0", or "k=<format>,v=<format>" when they differ.
    std::string kv_format() const;
    /// A hash of which experts the cache holds (STRATA_POLY_CACHE_HASH): two runs with the same one route the same
    /// experts to the same side, so their sums are rounded alike.
    uint64_t expert_set_hash() const;
    /// Tests (--dump-kv): while `to` is set, each attention layer's queries of a call are appended to it as the
    /// attention gets them, n_head * head_dim floats a token.  Needs launches that are not replayed (--no-graphs).
    void capture_queries(std::vector<float>* to) { q_capture_ = to; }

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
    /// Throws when the attention caches cannot have the format Options asks for (kv_k, kv_v).  Called once the
    /// model's shape is read and before anything is allocated.  The engine's own: fp16 only.
    virtual void check_kv_format() const;
    /// The weights of what guesses ahead, to VRAM.  The engine's own: the draft block's parts beside its layer.
    virtual void load_drafter();
    /// One guess (draft()): the token through what guesses ahead, its scores into logits_.  hn_'s first row is the
    /// model's output for the token before pos(); the second is this step's to leave and the next step's to read.
    /// The engine's own runs the draft block.
    virtual void draft_step(int32_t token, int i);
    /// The floats of every recurrent layer's state together.  They are state_'s first.
    virtual size_t recurrent_floats() const = 0;
    /// Whether the recurrent layers leave their state after any tokens of a call, and not only after its first
    /// ones.  A window then takes more tokens than it has places for their states (snap_tokens()).
    virtual bool snaps_spread() const { return false; }
    /// STRATA_POLY_TRACE=2 counts a layer's head by its kind: 0, or 1 for an attention layer among other kinds.
    virtual int layer_kind(int il) const { (void) il; return 0; }
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
#if defined(STRATA_USE_HIP)
    /// For a model whose route() is kernels::route_topk_softmax()'s (a softmax, the likeliest, their weights over
    /// their sum): the floor under that sum, which lets a token be routed on the card (card_moe()).  0: it is not.
    virtual float card_route_floor() const { return 0.f; }
#endif
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
    virtual void load_extra(const uint8_t* src, int64_t pos) { (void) src, (void) pos; }
    /// Layer il's part of what save_state() writes, to its place in dst: the layer's recurrent state, and what
    /// extra_bytes() counts for it, as they are once the layer has run every token before `pos`.  The card is
    /// done with the layer when this is called.  save_state() is this for every layer; read() calls it for a
    /// layer when the layer gets to a checkpoint, which is before the layers after it do.
    virtual void save_layer(int il, int64_t pos, uint8_t* dst) = 0;
    /// The draft block's attention for n tokens from position `pos`.  xn_ and xq_ hold its normed input; ao_ gets
    /// its output, and the tokens leave their rows in its caches.  block_rows() only leaves the rows.
    virtual void block_attend(int64_t pos, int n) { (void) pos, (void) n; }
    virtual void block_rows(int64_t pos, int n) { (void) pos, (void) n; }
    /// The attention caches in budget(): their VRAM, pinned RAM and stage bytes and their positions in each.
    virtual void kv_budget(Budget& b) const { (void) b; }
    /// Around a layer's head in run(), outside any recording: what the architecture hands other streams before
    /// and after the head's launches (KV streaming's copies ahead, Qwen35).
    virtual void before_head(int il, int n) { (void) il, (void) n; }
    virtual void after_head(int il, int n) { (void) il, (void) n; }
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
    /// The tokens of a window of t that leave the recurrent state as it is after them, a bit a token.
    unsigned snap_tokens(int t) const;
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
    void start_swaps();
    void plan_ahead(Layer& L, int il);
    void run(const int32_t* tokens, int n);
    bool pick_way(int t);
    void took_way(int t, double ms);
    void layer(int il, int n);
    void save_last(uint8_t* dst);
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
    // STRATA_POLY_PREFILL_PROF: an event on the stream after a part of a prompt batch (kernels.hpp's PrefillPart),
    // while pp_begin() has one measured; nothing otherwise.
    void pmark(int part);
    bool pp_begin(int n);
    void pp_end(bool began);
    float* fbuf(size_t n);
    int room(int n) const;
    int room_key(int n) const;
    void* dmalloc(size_t bytes, const char* what);
    void* hmalloc(size_t bytes, const char* what);
    /// Pinned RAM the card reads and writes where it lies (zero-copy): its device pointer in *dev.  Throws
    /// std::runtime_error when the system does not pin that much.
    void* hmalloc_mapped(size_t bytes, const char* what, void** dev);

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
    std::unordered_map<void*, size_t> alloc_bytes_;   ///< what each of them took, for budget()
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
    int read_rows_ = 1;       ///< the most tokens read() takes
    // read()'s calls between two layers: r_ holds two calls' rows, and with more calls than that the others'
    // wait in host memory, copied out and in beside the card's work on the call between
    struct Call {
        int at, m, mark;      ///< its first token, how many, and the mark it ends at or -1
    };
    std::vector<Call> calls_;
    float* h_res_ = nullptr;  ///< h_res_rows_ tokens' rows, pinned: as many as the longest read so far that needed it
    int h_res_rows_ = 0;
    void* rs_ = nullptr;      ///< the stream of those copies
    void* ev_in_[2] = {};     ///< marks a call's rows in each half of r_
    void* ev_out_[2] = {};    ///< marks the card done with a call in each half
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
    int asked_reserve_mib_ = -1; ///< Options::vram_reserve_mib as load() got it
    int reserve_mib_ = 0;        ///< what the cache was sized to leave free
    int full_cached_ = 0;        ///< the cache as load() filled it: its experts and their bytes
    size_t full_bytes_ = 0;
    int floor_experts_ = 0;      ///< the floor: the first Options::expert_cache_min experts by use
    size_t floor_bytes_ = 0;     ///< ... and the bytes they take; the cache is never cut below this
    int reserve_short_mib_ = 0;  ///< what the reserve lacked with the cache at the floor (0 without a floor)
    float* scratch_ = nullptr;
    size_t scratch_at_ = 0;
    float* state_ = nullptr;     ///< every recurrent layer's state; with a draft block, hn_'s first row
    size_t state_n_ = 0, rec_n_ = 0;   ///< all of it, and the recurrent layers' part
    // The draft block reads the model's output for the token before each of its tokens.  Row 0 is the output of
    // the last token of the sequence, and the outputs of the tokens of the call under way follow it.
    float* hn_ = nullptr;
    Likeliest *guess_ = nullptr, *h_guess_ = nullptr;   ///< a guess and its probability: VRAM and pinned
    int draft_vocab_ = 0;        ///< the tokens a guess of the draft block is chosen among (STRATA_POLY_DRAFT_VOCAB)
    int32_t win_tok_[8] = {};           ///< the window's tokens, for keep()
    int64_t ahead_pos_ = -1;            ///< draft() has left the block's row of this position as its token leaves it,
    int32_t ahead_tok_ = 0;             ///< which is this one
    float* snap_ = nullptr;      ///< window(): the recurrent state after tokens of it (snap_tokens()), rec_n_ apart
    int snap_places_ = 0;        ///< how many states it holds
    float* snap_at_ = nullptr;   ///< where the call under way leaves the first of its states
    unsigned snap_bits_ = 0;     ///< the tokens of the call under way that leave their state, a bit a token
    bool windowed_ = false;      ///< the call under way is a window's
    int win_ = 0;                ///< the tokens of the window that keep() has not settled yet
    // A window with experts on the CPU goes through the layers whole, or in two halves one behind the other
    // (run()).  Which takes less time depends on the card, on the CPU and on how many of the experts the cache
    // has, so each way's time is kept for each window size (pick_way(), took_way()).
    struct Way {
        double ms = 0;           ///< what a window took this way, the later ones counting for more
        int n = 0;               ///< how many were timed
    };
    Way way_[2][9];              ///< [in halves][the window's tokens]
    int way_calls_[9] = {};      ///< windows of each size so far
    bool halves_ = false;        ///< the way of the window under way
    float *wlogits_ = nullptr, *h_wlogits_ = nullptr;   ///< a window's logits, token after token: VRAM and pinned
    int cached_ = 0;

    // device scratch: a layer's input streams, one stream, the normed input, the attention's and the FFN's output
    float *r_ = nullptr, *cur_ = nullptr, *xn_ = nullptr, *ao_ = nullptr, *fo_ = nullptr;
    float* wf_ = nullptr;      ///< the largest BF16 matrix as floats: lin() multiplies a prompt batch by cuBLAS
    size_t bf16_max_ = 0;      ///< that matrix's values
    float *g_ = nullptr, *u_ = nullptr, *h_ = nullptr, *d_ = nullptr, *cpu_ = nullptr, *rl_ = nullptr,
          *logits_ = nullptr;
    void *xq_ = nullptr, *q8_ = nullptr;

    // pinned host buffers
    float *h_r_ = nullptr, *h_rl_ = nullptr, *h_x_ = nullptr, *h_cpu_ = nullptr, *h_logits_ = nullptr;

#if defined(STRATA_USE_HIP)
    // A token on its own routed on the card (STRATA_POLY_CARD_ROUTE, card_route_setup()).  run() hands the card
    // the whole token: each MoE layer's routing is a kernel behind its router, the cached experts follow at once,
    // and a layer that picked an expert outside VRAM waits on the card for the host's sum of it (wait_add()).  The
    // host goes through the layers behind the card (card_host()).
    bool card_route_ = false;                ///< every MoE layer of the model can be routed on the card
    bool card_window_ = false;               ///< ... and so can a call of up to kCardRows tokens (a window's)
    bool card_draft_ = false;                ///< ... and the draft block's experts (STRATA_POLY_CARD_DRAFT)
    bool drafted_on_card_ = false;           ///< draft_step() routed the block on the card: draft() follows it
    bool card_pipe_ = false;                 ///< run() is handing a token to the card that way
    std::atomic<bool> warming_{false};       ///< map_pages() is still reading the experts outside VRAM
    const uint8_t** dslot_ = nullptr;        ///< per (layer, expert): Layer::slot, on the card
    void *dgu_ = nullptr, *ddn_ = nullptr;   ///< per token: its gate|up and down launches (kernels::MmvqSlots)
    float* dw_ = nullptr;                    ///< their weights (0 for an expert on the CPU), kSlots a token
    float* dcard_ = nullptr;                 ///< the experts' down products, kSlots rows a token
    int* dneed_ = nullptr;                   ///< per (layer, token): a pick is outside VRAM
    uint32_t* dblocks_ = nullptr;            ///< the routing's count of the tokens it has routed (CardRoute::blocks)
    uint32_t* dseq_ = nullptr;               ///< the number of the token under way, on the card ...
    uint32_t seq_ = 0;                       ///< ... and on the host
    long long wait_ticks_ = 0;               ///< how long wait_add() waits for the host: 1 s of wall_clock64()
    unsigned long long* dwaited_ = nullptr;  ///< STRATA_POLY_TRACE=2: wall_clock64() ticks wait_add() waited, in all
    // pinned, coherent, per layer and token (kCardRows of them): the routing, the experts' input, the CPU's sum;
    // and per layer the two sides' words
    int32_t* h_ids_ = nullptr;
    float* h_w_ = nullptr;
    uint8_t* h_cached_ = nullptr;
    float *h_xl_ = nullptr, *h_cpul_ = nullptr, *h_rlc_ = nullptr;
    uint32_t *h_ready_ = nullptr, *h_go_ = nullptr, *h_failed_ = nullptr;
    // STRATA_POLY_CARD_ROUTE_CHECK=1: each routing on the card is done again by route() on the host, and compared
    bool card_check_ = false;
    long long card_checked_ = 0, card_ids_differ_ = 0, card_w_bits_ = 0;
    double card_w_diff_ = 0;
    void card_route_setup();
    void upload_slots();
    void slot_changed(int il, int e);
    bool card_takes(int n) const;
    void card_launch(Layer& L, int il, int n);
    void card_host(int n);
    bool card_host_layer(int il, int n, std::chrono::steady_clock::time_point start);
    void card_gave_up();
#endif

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
    int staged_il_ = -1;                          ///< the layer whose experts the buffer holds for its next call, or -1
    std::vector<int32_t> staged_;                 ///< those experts, in the buffer's order: one round of the layer's
    double ahead_ms_ = 0, route_ms_ = 0;          ///< what that layer's copy ahead took, and the card to its routing
    bool shared_ahead_ = false;                   ///< read() runs several calls: each copies ahead, whatever its size
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
    std::vector<float>* q_capture_ = nullptr;   ///< capture_queries()
    std::vector<Image> images_;          ///< the sequence's pictures, by row
    uint64_t images_set_ = 0;            ///< counts images(): what was derived from an earlier list is stale
    std::vector<int64_t> use_, prior_;   ///< routed count per (layer, expert): this run, and the profile's
    Stats stats_;
};

}  // namespace strata::poly
