// src/glm/engine.cpp - see the header.  What every architecture shares: the files, the weights' way to VRAM, the
// routed experts (the cache, the CPU, a prompt batch), the sequence's state, and the calls that feed tokens.
#include "engine.hpp"

#include "glm5next.hpp"
#include "mmq.hpp"
#include "qwen35.hpp"
#include "runtime.hpp"
#include "strata/glm/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif

namespace strata::glm {
namespace {

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

// On a machine with several NUMA nodes the file cache's pages are on the node whose CPU read them first, and the
// CPU reads the experts at the speed of memory: from the other node that is slower (29.4 against 22.5 ms of CPU
// experts a token on a two-socket Xeon).  `spans` are the experts in the mapped files.  When most of their pages
// are on one node, and the process may use `threads` CPUs of it, this thread is kept on that node's CPUs, and so
// are the threads it starts from then on.  Returns the node, or -1 when nothing was changed.
int stay_near(const std::vector<std::pair<const uint8_t*, size_t>>& spans, const std::vector<std::string>& files,
              int threads, double& share) {
#ifdef __linux__
    if (access("/sys/devices/system/node/node1", F_OK) != 0) return -1;   // one node
    // A page shows in numa_maps once it is mapped: touch a few thousand, spread over the experts.
    size_t total = 0;
    for (const auto& s : spans) total += s.second;
    const size_t step = std::max<size_t>(total / 4096, 4096);
    volatile uint8_t seen = 0;
    for (const auto& s : spans)
        for (size_t at = 0; at < s.second; at += step) seen = (uint8_t) (seen + s.first[at]);

    std::vector<std::string> real;
    for (const std::string& f : files)
        if (char* r = realpath(f.c_str(), nullptr)) {
            real.emplace_back(r);
            std::free(r);
        }
    std::vector<long long> pages;   // per node, of the model's files
    if (FILE* f = std::fopen("/proc/self/numa_maps", "r")) {
        char* line = nullptr;
        size_t cap = 0;
        while (getline(&line, &cap, f) > 0) {
            const char* file = std::strstr(line, " file=");
            if (!file) continue;
            file += 6;
            const size_t len = std::strcspn(file, " \n");
            bool ours = false;
            for (const std::string& r : real) ours = ours || (r.size() == len && r.compare(0, len, file, len) == 0);
            for (const char* p = file + len; ours && (p = std::strstr(p, " N")) != nullptr; ++p) {
                int node = 0;
                long long n = 0;
                if (std::sscanf(p, " N%d=%lld", &node, &n) != 2 || node < 0 || node > 1023) continue;
                if ((size_t) node >= pages.size()) pages.resize((size_t) node + 1, 0);
                pages[(size_t) node] += n;
            }
        }
        std::free(line);
        std::fclose(f);
    }
    long long all = 0;
    for (long long n : pages) all += n;
    if (all == 0) return -1;
    const int best = (int) (std::max_element(pages.begin(), pages.end()) - pages.begin());
    share = (double) pages[(size_t) best] / (double) all;
    if (share < 0.6) return -1;   // split between the nodes: one is as good as another

    cpu_set_t may, node;
    CPU_ZERO(&node);
    const std::string list = "/sys/devices/system/node/node" + std::to_string(best) + "/cpulist";
    if (FILE* f = std::fopen(list.c_str(), "r")) {   // "52-103,156-207"
        int a = 0, b = 0;
        while (std::fscanf(f, "%d", &a) == 1) {
            b = a;
            int c = std::fgetc(f);
            if (c == '-' && std::fscanf(f, "%d", &b) == 1) c = std::fgetc(f);
            for (int i = a; i <= b && i < CPU_SETSIZE; ++i) CPU_SET(i, &node);
            if (c != ',') break;
        }
        std::fclose(f);
    }
    if (sched_getaffinity(0, sizeof(may), &may) != 0) return -1;
    CPU_AND(&node, &node, &may);
    if (CPU_COUNT(&node) < threads || sched_setaffinity(0, sizeof(node), &node) != 0) return -1;
    return best;
#else
    (void) spans, (void) files, (void) threads, (void) share;
    return -1;
#endif
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

// One expert on its way into the cache.  The engine's thread asks, the follower copies and sends, and the engine's
// thread gives it the slot once the upload has arrived.
struct Engine::Swap {
    enum { kFree, kAsked, kSent };
    std::atomic<int> at{kFree};
    int il = 0, e = 0;
    uint8_t* slot = nullptr;       // where it goes in VRAM: the slot of the expert it replaces
    uint8_t* pinned = nullptr;
    cudaEvent_t done = nullptr;    // marks its upload
};

Engine::Engine() = default;

Engine::~Engine() {
    if (follower_.joinable()) {
        {
            std::lock_guard<std::mutex> l(swap_m_);
            swap_quit_ = true;
        }
        swap_cv_.notify_all();
        follower_.join();
    }
    if (swap_st_) cudaStreamSynchronize(cs(swap_st_));
    if (up_) cudaStreamSynchronize(cs(up_));
    if (st_) cudaStreamSynchronize(cs(st_));
    workers_.reset();
    mmq_.reset();
    for (void* p : dev_allocs_) cudaFree(p);
    for (void* p : host_allocs_) cudaFreeHost(p);
    for (void* e : {ev_stage_[0], ev_stage_[1], ev_ahead_, ev_free_, ev_})
        if (e) cudaEventDestroy((cudaEvent_t) e);
    for (int j = 0; j < n_swap_; ++j)
        if (swap_[(size_t) j].done) cudaEventDestroy(swap_[(size_t) j].done);
    for (void* g : graphs_) cudaGraphExecDestroy((cudaGraphExec_t) g);
    blas_destroy(blas_);
    if (swap_st_) cudaStreamDestroy(cs(swap_st_));
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

// A layer's tensor, under any of its names.  The draft block's are in the file it comes from.
const TensorInfo* Engine::tensor(int il, const char* name) const {
    const Shards& g = block_ && il >= n_layer_ ? *block_ : *gg_;
    if (const TensorInfo* t = g.find(blk(il, name))) return t;
    if (const Names* n = tensor_alias(name))
        for (const char* other : n->other)
            if (const TensorInfo* t = other ? g.find(blk(il, other)) : nullptr) return t;
    return nullptr;
}

// A value of the metadata of the model that `g` is a file of, under any of the names converters give it.
const MetaValue* Engine::meta_in(const Shards& g, const char* key) const {
    const MetaValue* arch = g.get("general.architecture");
    if (!arch) return nullptr;
    const std::string pre = arch->s + ".";
    if (const MetaValue* v = g.get(pre + key)) return v;
    if (const Names* n = meta_alias(key))
        for (const char* other : n->other)
            if (const MetaValue* v = other ? g.get(pre + other) : nullptr) return v;
    return nullptr;
}

int Engine::shape_i(const char* key) const {
    const MetaValue* v = meta_in(*gg_, key);
    if (!v) throw std::runtime_error("the GGUF has no " + gg_->get("general.architecture")->s + "." + key);
    return (int) v->u;
}

float Engine::shape_f(const char* key) const {
    const MetaValue* v = meta_in(*gg_, key);
    if (!v) throw std::runtime_error("the GGUF has no " + gg_->get("general.architecture")->s + "." + key);
    return (float) v->f;
}

// What every architecture's file says of the model: its layers, its width, its vocabulary and its experts.
//
// The block that guesses a second token after the model's own is one more block after the model's.  It is the
// model's and not a file's: a file that leaves it out can take it from another file of the model (draft_model).
// One converter gives the block a token table or an output matrix of its own, which this engine does not run.
void Engine::load_shape() {
    const MetaValue* nextn = meta_in(*gg_, "nextn_predict_layers");
    n_layer_ = shape_i("block_count") - (nextn ? (int) nextn->u : 0);
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
    n_embd_ = shape_i("embedding_length");
    // a file without the key has as many tokens as its token table
    const MetaValue* vocab = meta_in(*gg_, "vocab_size");
    const TensorInfo* te = gg_->find("token_embd.weight");
    n_vocab_ = vocab ? (int) vocab->u : te && te->shape.size() == 2 ? (int) te->shape[1] : 0;
    rms_eps_ = shape_f("attention.layer_norm_rms_epsilon");
    n_expert_ = shape_i("expert_count");
    n_used_ = shape_i("expert_used_count");
    n_ff_exp_ = shape_i("expert_feed_forward_length");
    const MetaValue* shexp = meta_in(*gg_, "expert_shared_feed_forward_length");
    n_ff_shexp_ = shexp ? (int) shexp->u : n_ff_exp_;

    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported model: ") + what);
    };
    need(n_layer_ > 0 && n_embd_ > 0 && n_embd_ % 32 == 0 && n_vocab_ > 0, "bad block_count or embedding_length");
    need(n_used_ >= 1 && n_used_ <= kMaxUsed && n_used_ <= n_expert_, "bad expert_used_count");

    // a block from another file must be this model's: a pruned model's has fewer experts to choose among
    if (block_ && block_ != gg_.get()) {
        const TensorInfo* gate = tensor(n_layer_, "ffn_gate_inp.weight");
        if (!gate || gate->shape.size() != 2 || (int64_t) gate->shape[0] != n_embd_ ||
            (int64_t) gate->shape[1] != n_expert_)
            throw std::runtime_error(opt_.draft_model + " is not a file of this model: its block's router is " +
                                     (gate ? shape_of(*gate) : std::string("missing")) + ", the model's layers have [" +
                                     std::to_string(n_embd_) + ", " + std::to_string(n_expert_) + "]");
    }
}

// A MoE layer: its router and shared expert to VRAM, and its routed experts' places in the file, where they
// stay; fill_expert_cache() copies the ones VRAM has room for.
void Engine::load_moe(int il) {
    Layer& L = layers_[(size_t) il];
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(il, n), a, b, kAnyMat); };
    L.moe = true;
    L.gate_inp = mat("ffn_gate_inp.weight", n_embd_, n_expert_);
    L.sh_gate = mat("ffn_gate_shexp.weight", n_embd_, n_ff_shexp_);
    L.sh_up = mat("ffn_up_shexp.weight", n_embd_, n_ff_shexp_);
    L.sh_down = mat("ffn_down_shexp.weight", n_ff_shexp_, n_embd_);

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

// The engine for a model file: its general.architecture says which.
std::unique_ptr<Engine> Engine::open(const Options& o) {
    const Shards file(o.model);
    const MetaValue* arch = file.get("general.architecture");
    const std::string name = arch ? arch->s : std::string();
    std::unique_ptr<Engine> engine;
    if (name == "glm5-next" || name == "glm5next") engine = std::make_unique<Glm5Next>();   // Unsloth: no hyphen
    else if (name == "qwen35moe") engine = std::make_unique<Qwen35>();
    else throw std::runtime_error(o.model + " is a '" + name + "' model, which this engine does not run");
    engine->load(o);
    return engine;
}

// The file a tensor is in.
const Shards& Engine::home(const TensorInfo& t) const {
    return block_ && gg_->find(t.name) != &t ? *block_ : *gg_;
}

// A tensor to VRAM.  It goes as the file has it when the kernels compute with that type.  Otherwise it is
// converted on the way: to F32 where a kernel reads floats, to BF16 for a matrix.  `fix` is applied to every
// value, which also takes the float path.  While dense_ is null the call only adds up the arena (load() runs the
// model's tensors through twice).
Dev Engine::upload(const TensorInfo* t, const std::string& name, int64_t n_in, int64_t n_out, Want want,
                   float (*fix)(float)) {
    if (!t) throw std::runtime_error("the GGUF has no tensor " + name);
    const uint64_t rest = t->elements() / std::max<uint64_t>(t->shape[0], 1);
    if ((int64_t) t->shape[0] != n_in || (int64_t) rest != n_out)
        throw std::runtime_error(t->name + " is " + shape_of(*t) + ", expected [" + std::to_string(n_in) + ", " +
                                 std::to_string(n_out) + "]");
    const size_t bytes = home(*t).payload(*t), count = (size_t) t->elements();
    const bool as_is = !fix && (t->type == kTypeF32 || (want != kF32 && t->type == kTypeBF16) ||
                                (want == kAnyMat && kernels::native_mmvq_supported((int) t->type)));
    const uint32_t type = as_is ? t->type : want == kF32 ? kTypeF32 : kTypeBF16;
    const size_t out = as_is ? bytes : count * (type == kTypeF32 ? sizeof(float) : sizeof(uint16_t));
    const Dev d{dense_ ? dense_ + dense_at_ : nullptr, type, (int) n_in, (int) n_out};
    dense_at_ += pad(out);
    if (!dense_) return d;
    if (dense_at_ > dense_bytes_) throw std::logic_error("the dense arena is too small for " + t->name);
    if (as_is) {
        ck(cudaMemcpy(d.p, home(*t).data(*t), bytes, cudaMemcpyHostToDevice), "weight upload");
        return d;
    }

    std::vector<float> f(count);
    const ggml_type_traits* traits = ggml_get_type_traits((ggml_type) t->type);
    if (t->type == kTypeF32) std::memcpy(f.data(), home(*t).data(*t), bytes);
    else if (traits && traits->to_float) traits->to_float(home(*t).data(*t), f.data(), (int64_t) count);
    else throw std::runtime_error(t->name + " is " + shape_of(*t) + ", a type this engine cannot read");
    if (fix)
        for (float& v : f) v = fix(v);
    if (type == kTypeF32) {
        ck(cudaMemcpy(d.p, f.data(), out, cudaMemcpyHostToDevice), "weight upload");
        return d;
    }
    std::vector<ggml_bf16_t> h(count);
    ggml_fp32_to_bf16_row(f.data(), h.data(), (int64_t) count);
    ck(cudaMemcpy(d.p, h.data(), out, cudaMemcpyHostToDevice), "weight upload");
    return d;
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
    std::vector<size_t> sizes;   // the slot sizes there are; a layer's place among them is its trade number
    size_t all = 0;
    for (int il = 0; il < (int) layers_.size(); ++il) {
        Layer& L = layers_[(size_t) il];
        if (!L.moe || !kernels::native_mmvq_supported((int) L.gu_type) ||
            !kernels::native_mmvq_supported((int) L.d_type))
            continue;
        const size_t bytes = pad(2 * L.gu_bytes + L.d_bytes);
        if (std::find(sizes.begin(), sizes.end(), bytes) == sizes.end()) sizes.push_back(bytes);
        L.trade = (int) (std::find(sizes.begin(), sizes.end(), bytes) - sizes.begin());
        slot_max_ = std::max(slot_max_, bytes);
        for (int e = 0; e < n_expert_; ++e) {
            const double used = prior_.empty() ? 0.0 : (double) prior_[(size_t) il * n_expert_ + e];
            cands.push_back({il, e, bytes, used / (double) bytes});
            all += bytes;
        }
    }
    std::stable_sort(cands.begin(), cands.end(),
                     [](const Cand& a, const Cand& b) { return a.value != b.value ? a.value > b.value : a.e < b.e; });

    n_trade_ = (int) sizes.size();
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

// What a layer copies to the staging buffer ahead of its routing: the experts outside the cache, the most used
// first, as many as the buffer holds.  Worked out when the layer runs: the cache changes (follow()).
void Engine::plan_ahead(Layer& L, int il) {
    L.ahead.clear();
    for (int e = 0; e < n_expert_; ++e)
        if (!L.slot[(size_t) e]) L.ahead.push_back(e);
    if (!prior_.empty()) {
        const int64_t* used = prior_.data() + (size_t) il * n_expert_;
        std::stable_sort(L.ahead.begin(), L.ahead.end(), [used](int32_t a, int32_t b) { return used[a] > used[b]; });
    }
    L.ahead.resize(std::min(L.ahead.size(), stage_bytes_ / (2 * L.gu_bytes + L.d_bytes)));
}

// The follower: an asked-for expert goes from the mapped file to its pinned buffer, and from there to its slot.
// The file may have to be read for it, which is why this is not the engine's thread.
void Engine::follow_loop() {
    std::unique_lock<std::mutex> lk(swap_m_);
    for (;;) {
        Swap* s = nullptr;
        swap_cv_.wait(lk, [&] {
            for (int j = 0; j < n_swap_ && !s; ++j)
                if (swap_[(size_t) j].at.load() == Swap::kAsked) s = &swap_[(size_t) j];
            return swap_quit_ || s;
        });
        if (swap_quit_) return;
        lk.unlock();
        const Layer& L = layers_[(size_t) s->il];
        const size_t e = (size_t) s->e, gu = L.gu_bytes;
        std::memcpy(s->pinned, L.exp_gate + e * gu, gu);
        std::memcpy(s->pinned + gu, L.exp_up + e * gu, gu);
        std::memcpy(s->pinned + 2 * gu, L.exp_down + e * L.d_bytes, L.d_bytes);
        cudaError_t err = cudaMemcpyAsync(s->slot, s->pinned, 2 * gu + L.d_bytes, cudaMemcpyHostToDevice, cs(swap_st_));
        if (err == cudaSuccess) err = cudaEventRecord(s->done, cs(swap_st_));
        lk.lock();
        if (err != cudaSuccess) {
            swap_error_ = cudaGetErrorString(err);
            return;
        }
        s->at.store(Swap::kSent);
    }
}

// The cache follows what is written.  Called at the start of an eval, where nothing queued reads the cache, with
// the number of tokens the eval writes (0 for a prompt batch).  The experts whose upload has arrived get their
// slots.  Then, for written tokens, the experts used lately that the cache lacks start on their way, each to the
// slot of the cached expert of its size that was used least (among equals, the one the profile ranks last).
//
// A prompt batch neither counts nor moves anything.  What a prompt routes to says no more about the answer than
// the profile does, and moving the cache while the prompt was read cost 3% of the reading speed (measured).
void Engine::follow(int written) {
    {
        std::lock_guard<std::mutex> l(swap_m_);
        if (!swap_error_.empty()) throw std::runtime_error("expert upload: " + swap_error_);
    }
    const size_t ne = (size_t) n_expert_;
    int n_free = 0;
    for (int j = 0; j < n_swap_; ++j) {
        Swap& s = swap_[(size_t) j];
        if (s.at.load() == Swap::kSent) {
            const cudaError_t arrived = cudaEventQuery(s.done);
            if (arrived == cudaErrorNotReady) continue;
            ck(arrived, "expert upload");
            layers_[(size_t) s.il].slot[(size_t) s.e] = s.slot;
            coming_[(size_t) s.il * ne + (size_t) s.e] = 0;
            s.at.store(Swap::kFree);
            ++stats_.swapped_experts;
        }
        n_free += s.at.load() == Swap::kFree;
    }
    if (written <= 0) return;
    const float keep = std::exp2(-(float) written / (float) kSwapHalf);
    for (float& r : recent_) r *= keep;

    bool asked = false;
    for (; n_free > 0; --n_free) {
        in_.assign((size_t) n_trade_, Pick{});
        out_.assign((size_t) n_trade_, Pick{});
        for (int il = 0; il < (int) layers_.size(); ++il) {
            const Layer& L = layers_[(size_t) il];
            if (L.trade < 0) continue;
            Pick& in = in_[(size_t) L.trade];
            Pick& out = out_[(size_t) L.trade];
            for (int e = 0; e < n_expert_; ++e) {
                const size_t i = (size_t) il * ne + (size_t) e;
                const float r = recent_[i];
                const int64_t p = prior_.empty() ? 0 : prior_[i];
                if (L.slot[(size_t) e]) {
                    if (out.il < 0 || r < out.recent || (r == out.recent && p < out.prior)) out = {il, e, r, p};
                } else if (r >= kSwapMin && !coming_[i] &&
                           (in.il < 0 || r > in.recent || (r == in.recent && p > in.prior))) {
                    in = {il, e, r, p};
                }
            }
        }
        int best = -1;
        for (int c = 0; c < n_trade_; ++c) {
            const Pick& in = in_[(size_t) c];
            const Pick& out = out_[(size_t) c];
            if (in.il < 0 || out.il < 0 || in.recent < 2.f * out.recent + kSwapMin) continue;
            if (best < 0 || in.recent - out.recent > in_[(size_t) best].recent - out_[(size_t) best].recent) best = c;
        }
        if (best < 0) break;

        const Pick in = in_[(size_t) best], out = out_[(size_t) best];
        Swap* s = nullptr;
        for (int j = 0; j < n_swap_ && !s; ++j)
            if (swap_[(size_t) j].at.load() == Swap::kFree) s = &swap_[(size_t) j];
        uint8_t*& given = layers_[(size_t) out.il].slot[(size_t) out.e];
        s->il = in.il;
        s->e = in.e;
        s->slot = given;
        given = nullptr;
        coming_[(size_t) in.il * ne + (size_t) in.e] = 1;
        s->at.store(Swap::kAsked);
        asked = true;
    }
    if (asked) {
        { std::lock_guard<std::mutex> l(swap_m_); }
        swap_cv_.notify_one();
    }
}

void Engine::load(const Options& o) {
    opt_ = o;
    if (opt_.max_context < 16) throw std::runtime_error("the context must be at least 16 tokens");
    if (opt_.prompt_batch < 1 || opt_.prompt_batch > 4096)
        throw std::runtime_error("a prompt batch is 1 to 4096 tokens");
    if (opt_.stage_mib < 1) throw std::runtime_error("the staging buffer must be at least 1 MiB");
    if (opt_.stage_ahead < 0) throw std::runtime_error("the batch size to stage ahead from cannot be negative");
    if (opt_.expert_swaps < 0 || opt_.expert_swaps > 64)
        throw std::runtime_error("0 to 64 experts can be on their way into the cache at a time");
    if (opt_.window < 1 || opt_.window > 8) throw std::runtime_error("a window is 1 to 8 tokens");
    opt_.window = std::min(opt_.window, opt_.dump ? 1 : std::min(opt_.prompt_batch, opt_.max_context));
    gg_ = std::make_unique<Shards>(opt_.model);
    if (!opt_.draft_model.empty() && opt_.draft && opt_.window > 1) dg_ = std::make_unique<Shards>(opt_.draft_model);
    load_geometry();
    if (block_ != dg_.get()) dg_.reset();

    cudaStream_t st = nullptr;
    ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "cudaStreamCreate");
    st_ = st;
    cudaEvent_t ev = nullptr;
    ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");
    ev_ = ev;
    blas_ = blas_create(st_);

    // the token table stays in the file, a row a token, and embed() reads a row in whatever type it has
    const TensorInfo* te = gg_->find("token_embd.weight");
    const ggml_type_traits* tt = te ? ggml_get_type_traits((ggml_type) te->type) : nullptr;
    if (!te || te->shape.size() != 2 || (int64_t) te->shape[0] != n_embd_ || (int64_t) te->shape[1] != n_vocab_ ||
        !tt || tt->blck_size <= 0 || n_embd_ % tt->blck_size != 0 || (te->type != kTypeF32 && !tt->to_float))
        throw std::runtime_error("token_embd.weight must be [" + std::to_string(n_embd_) + ", " +
                                 std::to_string(n_vocab_) + "] in a type ggml converts to floats, it is " +
                                 (te ? shape_of(*te) : "missing"));
    tok_row_ = gg_->payload(*te) / (size_t) n_vocab_;
    tok_type_ = te->type;
    tok_embd_ = gg_->data(*te);

    // twice: the first pass adds up the arena, the second fills it
    for (int pass = 0; pass < 2; ++pass) {
        const int n_all = n_layer_ + (draft_ ? 1 : 0);
        layers_.assign((size_t) n_all, Layer{});
        n_moe_ = 0;
        dense_at_ = 0;
        begin_pass();
        for (int il = 0; il < n_all; ++il) load_layer(il);
        out_norm_ = upload(gg_->find("output_norm.weight"), "output_norm.weight", n_embd_, 1, kF32);
        output_ = upload(gg_->find("output.weight"), "output.weight", n_embd_, n_vocab_, kAnyMat);
        if (draft_) {
            check_draft_block();
            auto part = [&](const char* n, int64_t a, int64_t b, Want w) {
                return upload(tensor(n_layer_, n), blk(n_layer_, n), a, b, w);
            };
            d_proj_ = part("nextn.eh_proj.weight", 2 * n_embd_, n_embd_, kAnyMat);
            d_enorm_ = part("nextn.enorm.weight", n_embd_, 1, kF32);
            d_hnorm_ = part("nextn.hnorm.weight", n_embd_, 1, kF32);
            // without a norm of its own, the block's output goes through the model's
            d_head_ = tensor(n_layer_, "nextn.shared_head_norm.weight")
                          ? part("nextn.shared_head_norm.weight", n_embd_, 1, kF32)
                          : out_norm_;
        }
        if (pass == 0) {
            dense_bytes_ = dense_at_;
            dense_ = (uint8_t*) dmalloc(dense_bytes_ + 256, "the dense weights");
        }
    }

    alloc();
    if (!opt_.profile.empty()) read_profile(opt_.profile, (int) layers_.size(), n_expert_, prior_);
    fill_expert_cache();
    const int threads = opt_.threads > 0 ? opt_.threads : usable_cpus();
    if (opt_.stay_near) {   // before any thread that reads the experts is started
        std::vector<std::pair<const uint8_t*, size_t>> spans;
        for (const Layer& L : layers_) {
            if (!L.moe) continue;
            spans.emplace_back(L.exp_gate, (size_t) n_expert_ * L.gu_bytes);
            spans.emplace_back(L.exp_up, (size_t) n_expert_ * L.gu_bytes);
            spans.emplace_back(L.exp_down, (size_t) n_expert_ * L.d_bytes);
        }
        std::vector<std::string> files = gg_->paths();
        if (dg_) files.insert(files.end(), dg_->paths().begin(), dg_->paths().end());
        node_ = stay_near(spans, files, threads, node_share_);
    }
    workers_ = std::make_unique<Workers>(threads);
    if (cached_ > 0 && opt_.expert_swaps > 0) {
        swap_ = std::make_unique<Swap[]>((size_t) opt_.expert_swaps);
        cudaStream_t up = nullptr;
        ck(cudaStreamCreateWithFlags(&up, cudaStreamNonBlocking), "cudaStreamCreate");
        swap_st_ = up;
        for (; n_swap_ < opt_.expert_swaps; ++n_swap_) {
            Swap& s = swap_[(size_t) n_swap_];
            s.pinned = (uint8_t*) hmalloc(slot_max_, "pinned memory");
            ck(cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming), "cudaEventCreate");
        }
        recent_.assign(use_.size(), 0.f);
        coming_.assign(use_.size(), 0);
        follower_ = std::thread([this] { follow_loop(); });
    }
    reset();
}

// Per-token buffers hold rows_ tokens, row after row.
void Engine::shared_scratch() {
    const size_t n = (size_t) rows_, e = n * n_embd_;
    const size_t ff = n * std::max({n_ff_, n_ff_shexp_, 2 * n_ff_exp_});
    r_ = fbuf(std::max(streams(), draft_ ? 3 : 1) * e);   // the draft block works in it too (draft_input())
    cur_ = fbuf(e);
    xn_ = fbuf(e);
    ao_ = fbuf(e);
    fo_ = fbuf(e);
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

// The sequence's state, the caches and every buffer, once the weights are known.
void Engine::alloc() {
    // --dump-layers prints one token's stages
    rows_ = opt_.dump ? 1 : std::min(opt_.prompt_batch, opt_.max_context);
    rec_n_ = recurrent_floats();
    state_n_ = rec_n_ + (draft_ ? (size_t) n_embd_ : 0);
    state_ = (float*) dmalloc((state_n_ + (draft_ ? (size_t) rows_ * n_embd_ : 0)) * sizeof(float),
                              "the recurrent state");
    if (draft_) {
        hn_ = state_ + rec_n_;
        h_dlogits_ = (float*) hmalloc((size_t) n_vocab_ * sizeof(float), "pinned memory");
    }
    // window(): the state after each of its tokens but the last, for keep(), and every token's logits
    if (opt_.window > 1)
        snap_ = (float*) dmalloc(size_t(opt_.window - 1) * rec_n_ * sizeof(float), "the recurrent state");
    wlogits_ = (float*) dmalloc(size_t(opt_.window) * n_vocab_ * sizeof(float), "scratch");
    h_wlogits_ = (float*) hmalloc(size_t(opt_.window) * n_vocab_ * sizeof(float), "pinned memory");
    alloc_caches();

    // A prompt batch runs its routed experts through MMQ when the card has a tile for each of their types.
    // The model's layers: the draft block's experts never run a batch at a time.
    const size_t n = (size_t) rows_;
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

    shared_scratch();   // the first pass only counts
    assign_scratch();
    scratch_ = (float*) dmalloc(scratch_at_ * sizeof(float), "scratch");
    scratch_at_ = 0;
    shared_scratch();
    assign_scratch();

    const int q_max = std::max({n_embd_, n_ff_, n_ff_exp_, n_ff_shexp_, widest(), 256});
    xq_ = dmalloc(n * kernels::native_q8_1_bytes(n_embd_), "scratch");
    q8_ = dmalloc(n * kernels::native_q8_1_bytes((q_max + 255) / 256 * 256), "scratch");

    h_r_ = (float*) hmalloc(n * streams() * n_embd_ * sizeof(float), "pinned memory");
    h_rl_ = (float*) hmalloc(n * n_expert_ * sizeof(float), "pinned memory");
    h_x_ = (float*) hmalloc(n * n_embd_ * sizeof(float), "pinned memory");
    h_cpu_ = (float*) hmalloc(n * n_embd_ * sizeof(float), "pinned memory");
    h_logits_ = (float*) hmalloc((size_t) n_vocab_ * sizeof(float), "pinned memory");

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
    use_.assign(layers_.size() * n_expert_, 0);
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
    ahead_pos_ = -1;
}

void Engine::save_state(uint8_t* dst) {
    ck(cudaStreamSynchronize(cs(st_)), "save state");
    ck(cudaMemcpy(dst, state_, state_bytes(), cudaMemcpyDeviceToHost), "save state");
}

void Engine::load_state(const uint8_t* src, int64_t pos) {
    if (pos < 0 || pos > opt_.max_context) throw std::logic_error("load_state: the position is outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "load state");
    ck(cudaMemcpy(state_, src, state_bytes(), cudaMemcpyHostToDevice), "load state");
    pos_ = pos;
    ahead_pos_ = -1;
}

void Engine::embed(int32_t token, float* dst) const {
    const uint8_t* row = tok_embd_ + (size_t) token * tok_row_;
    if (tok_type_ == kTypeF32) std::memcpy(dst, row, (size_t) n_embd_ * sizeof(float));
    else ggml_get_type_traits((ggml_type) tok_type_)->to_float(row, dst, n_embd_);
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
    if (mmq_ && n >= kStageRows && Mmq::fits((int) w.type, w.n_out, w.n_in)) {
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

// The router's scores, and the copies of them and of the layer's input to the host.
void Engine::moe_router(const Layer& L, int n) {
    const size_t rows = (size_t) n;
    lin(L.gate_inp, xn_, rl_, n, xq_);
    ck(cudaMemcpyAsync(h_rl_, rl_, rows * n_expert_ * sizeof(float), cudaMemcpyDeviceToHost, cs(st_)), "router");
    ck(cudaMemcpyAsync(h_x_, xn_, rows * n_embd_ * sizeof(float), cudaMemcpyDeviceToHost, cs(st_)), "router");
}

// The routed experts of tokens that come a few at a time.  moe_router() is on its way, with ev_ behind it.
void Engine::moe(Layer& L, int il, int n) {
    const size_t e_row = (size_t) n_embd_, rows = (size_t) n;
    Clock::time_point t0 = Clock::now();
    ck(cudaEventSynchronize((cudaEvent_t) ev_), "router");
    stats_.route_wait_ms += ms_since(t0);

    const float limit = clamp_exp_[(size_t) il];
    const size_t xq_row = kernels::native_q8_1_bytes(n_embd_);
    bool any_host = false;
    // A window's tokens together: the CPU reads an expert that several of them chose once.
    for (int at = 0; at < n; at += kCpuRows) {
        const int m = std::min(kCpuRows, n - at);
        struct Cached {
            const uint8_t* slot;
            float w;
            int row;
        };
        CpuExpert host[kCpuRows * kMaxUsed];
        Cached cached[kCpuRows * kMaxUsed];
        int n_host = 0, n_cached = 0;
        for (int t = 0; t < m; ++t) {
            int ids[kMaxUsed];
            float w[kMaxUsed];
            route(il, h_rl_ + (size_t) (at + t) * n_expert_, ids, w);
            for (int k = 0; k < n_used_; ++k) {
                const size_t e = (size_t) ids[k];
                ++use_[(size_t) il * n_expert_ + e];
                if (n_swap_ > 0) recent_[(size_t) il * n_expert_ + e] += 1.f;
                if (const uint8_t* s = L.slot[e])
                    cached[n_cached++] = {s, w[k], at + t};
                else
                    host[n_host++] = {L.exp_gate + e * L.gu_bytes, L.exp_up + e * L.gu_bytes,
                                      L.exp_down + e * L.d_bytes, w[k], t};
            }
        }
        stats_.gpu_experts += n_cached;
        stats_.cpu_experts += n_host;
        // Five launches a cached expert.  Nothing else goes to the stream until the CPU experts are done, so one
        // of their threads hands these to the card meanwhile: on this thread they held the CPU experts up.
        const std::function<void()> launch = [&] {
            for (int i = 0; i < n_cached; ++i) {
                const Cached& c = cached[i];
                const void* xq = (const uint8_t*) xq_ + (size_t) c.row * xq_row;
                kernels::native_mmvq((int) L.gu_type, c.slot, xq, g_, n_embd_, 2 * n_ff_exp_, 1, st_);
                swiglu_clamp(g_, g_ + n_ff_exp_, h_, n_ff_exp_, limit, st_);
                kernels::native_quantize_q8_1(h_, q8_, n_ff_exp_, 1, st_);
                kernels::native_mmvq((int) L.d_type, c.slot + 2 * L.gu_bytes, q8_, d_, n_ff_exp_, n_embd_, 1, st_);
                axpy(d_, c.w, fo_ + (size_t) c.row * e_row, n_embd_, true, st_);
            }
        };

        // The next layer's router wait is what makes h_cpu_ free to write again.
        float* cpu = h_cpu_ + (size_t) at * e_row;
        if (n_host == 0) {
            launch();
            std::fill(cpu, cpu + (size_t) m * e_row, 0.f);
            continue;
        }
        any_host = true;
        t0 = Clock::now();
        cpu_experts(*workers_, host, n_host, m, (int) L.gu_type, (int) L.d_type, n_embd_, n_ff_exp_,
                    h_x_ + (size_t) at * e_row, limit, cpu, n_cached > 0 ? &launch : nullptr);
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

    moe_shared(L, il, n);

    const bool ahead = opt_.stage_ahead > 0 && n >= opt_.stage_ahead;
    if (ahead) plan_ahead(L, il);
    const int n_ahead = ahead ? (int) L.ahead.size() : 0;
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
        route(il, h_rl_ + (size_t) t * n_expert_, ids, w + (size_t) t * n_used_);
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
    stats_.batch_cached += rc;
    stats_.batch_staged += rr - rc;

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

// Records what `work` hands to the stream, as a graph that is launched with one call from then on.
template <class F>
void* Engine::record(F&& work) {
    cudaGraph_t g = nullptr;
    ck(cudaStreamBeginCapture(cs(st_), cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture");
    try {
        work();
    } catch (...) {
        if (cudaStreamEndCapture(cs(st_), &g) == cudaSuccess && g) cudaGraphDestroy(g);
        throw;
    }
    ck(cudaStreamEndCapture(cs(st_), &g), "cudaStreamEndCapture");
    cudaGraphExec_t exec = nullptr;
    const cudaError_t made = cudaGraphInstantiateWithFlags(&exec, g, 0);
    cudaGraphDestroy(g);
    ck(made, "cudaGraphInstantiate");
    graphs_.push_back(exec);
    return exec;
}

// n tokens from pos() through every layer.  The sequence is not longer for it: the caller says by how much.
void Engine::run(const int32_t* tokens, int n) {
    if (n < 1 || n > rows_) throw std::logic_error("eval takes 1 to max_rows() tokens");
    for (int t = 0; t < n; ++t)
        if (tokens[t] < 0 || tokens[t] >= n_vocab_) throw std::runtime_error("a token id is outside the vocabulary");
    if (pos_ + n > opt_.max_context)
        throw std::runtime_error("the context is full (" + std::to_string(opt_.max_context) + " tokens)");

    // h_r_ and h_cpu_ may still be the source of the previous call's copies
    ck(cudaStreamSynchronize(cs(st_)), "eval");
    if (n_swap_ > 0) follow(n < kStageRows ? n : 0);
    const size_t e_row = (size_t) n_embd_, r_row = (size_t) streams() * e_row;
    for (int t = 0; t < n; ++t) {
        float* r = h_r_ + (size_t) t * r_row;
        embed(tokens[t], r);
        for (int h = 1; h < streams(); ++h) std::memcpy(r + (size_t) h * e_row, r, e_row * sizeof(float));
    }
    ck(cudaMemcpyAsync(r_, h_r_, (size_t) n * r_row * sizeof(float), cudaMemcpyHostToDevice, cs(st_)), "embed");

    // A token alone hands the card the same launches every time, some thirty a layer up to the router, and they
    // cost this thread more than the card.  They are recorded once, a graph a layer, and replayed with one call.
    // Only for the layers whose launches are the same at every position (Layer::fixed).
    const bool replay = opt_.graphs && n == 1 && !opt_.dump;
    for (int il = 0; il < n_layer_; ++il) {
        Layer& L = layers_[(size_t) il];
        if (replay && L.fixed) {
            if (!L.head) L.head = record([&] { layer_head(L, il, 1); });
            ck(cudaGraphLaunch((cudaGraphExec_t) L.head, cs(st_)), "layer graph");
        } else {
            layer_head(L, il, n);
        }
        if (!L.moe) continue;
        if (mmq_ && n >= kStageRows) {
            moe_rows(L, il, n);
        } else {
            ck(cudaEventRecord((cudaEvent_t) ev_, cs(st_)), "router");
            if (replay) {
                if (!L.shared) L.shared = record([&] { moe_shared(L, il, 1); });
                ck(cudaGraphLaunch((cudaGraphExec_t) L.shared, cs(st_)), "layer graph");
            } else {
                moe_shared(L, il, n);
            }
            moe(L, il, n);
        }
        dbg("ffn", il, fo_, n_embd_);
        layer_tail(L, il, n);
    }
}

void Engine::eval(const int32_t* tokens, int n, bool want_logits) {
    run(tokens, n);
    // The model's output: the last token's for the logits, and every token's for the draft block.
    if (want_logits || draft_) {
        const bool replay = opt_.graphs && n == 1 && want_logits && !opt_.dump;
        const int m = draft_ ? n : 1;
        float* h = draft_ ? hn_ + n_embd_ : xn_;
        const auto tail = [&] {
            output_rows(n - m, m, h);
            dbg("out_norm", n_layer_, h, n_embd_);
            if (!want_logits) return;
            lin(output_, h + (size_t) (m - 1) * n_embd_, logits_, 1);
            ck(cudaMemcpyAsync(h_logits_, logits_, (size_t) n_vocab_ * sizeof(float), cudaMemcpyDeviceToHost,
                               cs(st_)),
               "logits");
        };
        if (replay) {
            if (!tail_) tail_ = record(tail);
            ck(cudaGraphLaunch((cudaGraphExec_t) tail_, cs(st_)), "logits graph");
        } else {
            tail();
        }
        if (want_logits) ck(cudaStreamSynchronize(cs(st_)), "eval");
    }
    // behind the logits: the caller has them while the card does this
    if (draft_) draft_rows(tokens, ahead_pos_ == pos_ && tokens[0] == ahead_tok_ ? 1 : 0, n);
    pos_ += n;
    stats_.tokens += n;
}

// The recurrent layers are the only ones a wrong guess would spoil: every other layer's row of a position is
// written again when the position gets its real token.  So they leave the state as it was after each token but
// the last (snap_), and keep() puts back the one after the last token that was right.
void Engine::window(const int32_t* tokens, int t) {
    if (t < 1 || t > opt_.window) throw std::logic_error("a window takes 1 to max_window() tokens");
    n_snap_ = t - 1;
    try {
        run(tokens, t);
    } catch (...) {
        n_snap_ = 0;
        throw;
    }
    n_snap_ = 0;
    float* h = draft_ ? hn_ + n_embd_ : xn_;
    output_rows(0, t, h);
    lin(output_, h, wlogits_, t);
    ck(cudaMemcpyAsync(h_wlogits_, wlogits_, (size_t) t * n_vocab_ * sizeof(float), cudaMemcpyDeviceToHost, cs(st_)),
       "logits");
    ck(cudaStreamSynchronize(cs(st_)), "window");
    std::copy(tokens, tokens + t, win_tok_);
    win_ = t;
}

void Engine::keep(int n) {
    if (n < 1 || n > win_) throw std::logic_error("keep takes 1 to the window's tokens");
    if (n < win_)
        ck(cudaMemcpyAsync(state_, snap_ + size_t(n - 1) * rec_n_, rec_n_ * sizeof(float), cudaMemcpyDeviceToDevice,
                           cs(st_)),
           "keep");
    if (draft_) draft_rows(win_tok_, ahead_pos_ == pos_ && win_tok_[0] == ahead_tok_ ? 1 : 0, n);
    pos_ += n;
    stats_.tokens += n;
    win_ = 0;
}

// The draft block's input for n tokens: each token's embedding and the model's output for the token before it
// (h, a row a token), each normed, the two side by side through the block's projection.  Returns the block's
// residual, and leaves its normed attention input in xn_ and xq_.  r_ is free between the model's calls, and the
// block works in it: n rows of the two halves, then the residual.
//
// h_r_ is free to write: every call that copied out of it has waited for the card since.
float* Engine::draft_input(const int32_t* tokens, const float* h, int n) {
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
void Engine::draft_rows(const int32_t* tokens, int first, int n) {
    const size_t e = (size_t) n_embd_;
    if (first < n) {
        draft_input(tokens + first, hn_ + (size_t) first * e, n - first);
        block_rows(pos_ + first, n - first);
    }
    ck(cudaMemcpyAsync(hn_, hn_ + (size_t) n * e, e * sizeof(float), cudaMemcpyDeviceToDevice, cs(st_)), "draft");
    ahead_pos_ = -1;
}

// One token through the draft block, at pos() + i.  The first token's row is then what draft_rows() would leave
// for it, and eval() and keep() skip it; the rows of the guesses after it are written again when the positions get
// their real tokens.
int32_t Engine::draft(int32_t token, int i, float* prob) {
    if (!draft_) throw std::logic_error("draft: this model has no block that guesses ahead");
    if (token < 0 || token >= n_vocab_) throw std::runtime_error("a token id is outside the vocabulary");
    if (i < 0 || pos_ + i >= opt_.max_context) throw std::logic_error("draft: the position is outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "draft");
    Layer& L = layers_[(size_t) n_layer_];
    float* own = hn_ + n_embd_;   // the block's output, which stands in for the model's from the second guess on
    float* x = draft_input(&token, i == 0 ? hn_ : own, 1);
    block_attend(pos_ + i, 1);
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
