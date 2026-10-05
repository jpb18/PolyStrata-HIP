// src/glm/engine.cpp - see the header.  The reference for the math is llama.cpp's glm5-next graph.
#include "engine.hpp"

#include "mmq.hpp"
#include "strata/artifact/gguf_reader.hpp"
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

constexpr uint32_t kTypeF32 = 0, kTypeBF16 = 30;
constexpr int kHc = 4, kConv = 4;   // what kernels.cu is written for; checked at load
constexpr int kMaxUsed = 32;
constexpr int kAttnRows = 32;       // tokens one pass of the sparse attention takes; their keys are 4 MiB a token
// From this many tokens, a call runs its routed experts on the GPU alone and copies the ones that are not cached
// through VRAM.  Below it, that copy costs more than the CPU takes to compute the few experts.
constexpr int kStageRows = 32;
constexpr size_t kChunkBytes = size_t(512) << 20;   // one upload, and each of the two pinned buffers
constexpr float kMinWeightSum = 6.103515625e-5f;   // the floor ggml's MoE graph puts under the gate sum
// follow(): an expert's recent use is the count of its uses, halved every kSwapHalf tokens.  An expert takes a
// cached one's slot when it was used twice lately and more than twice as much as that one.  Counts this small are
// noisy, and an upload gets in the way of the work: a slot goes to an expert that will clearly be used more.
constexpr float kSwapMin = 2.f;
constexpr int kSwapHalf = 128;

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

size_t pad(size_t n) { return (n + 255) & ~size_t(255); }

std::string blk(int il, const char* name) { return "blk." + std::to_string(il) + "." + name; }

const float* fp(const Dev& d) { return (const float*) d.p; }
cudaStream_t cs(void* s) { return (cudaStream_t) s; }

std::string shape_of(const TensorInfo& t) {
    std::string s = "[";
    for (size_t d = 0; d < t.shape.size(); ++d) s += (d ? ", " : "") + std::to_string(t.shape[d]);
    return s + "] " + t.type_name();
}

// The model's GGUFs come out of more than one converter.  `name` is what the layout of neurall's llama.cpp fork
// calls a thing, and what this file calls it.  `other` is what the same thing is called in the files made for
// patrickbdevaney's fork (the first, where it differs) and in antirez's (the second).
struct Names {
    const char* name;
    const char* other[2];
};
constexpr Names kTensorNames[] = {
    {"hc_attn_fn.weight", {"hc_attn_fn"}},
    {"hc_attn_scale.weight", {"hc_attn_scale"}},
    {"hc_attn_base.weight", {"hc_attn_base"}},
    {"hc_ffn_fn.weight", {"hc_ffn_fn"}},
    {"hc_ffn_scale.weight", {"hc_ffn_scale"}},
    {"hc_ffn_base.weight", {"hc_ffn_base"}},
    {"indexer_compressor_gate.weight", {"indexer.kpool_gate", "indexer.pool_gate.weight"}},
    {"indexer_compressor_ape.weight", {"indexer.kpool_ape", "indexer.pool_ape.weight"}},
    {"attn_q.weight", {nullptr, "kda_q.weight"}},
    {"attn_k.weight", {nullptr, "kda_k.weight"}},
    {"attn_v.weight", {nullptr, "kda_v.weight"}},
    {"attn_output.weight", {nullptr, "kda_output.weight"}},
    {"ssm_conv1d_q.weight", {nullptr, "kda_q_conv.weight"}},
    {"ssm_conv1d_k.weight", {nullptr, "kda_k_conv.weight"}},
    {"ssm_conv1d_v.weight", {nullptr, "kda_v_conv.weight"}},
    {"ssm_f_a.weight", {nullptr, "kda_f_a.weight"}},
    {"ssm_f_b.weight", {nullptr, "kda_f_b.weight"}},
    {"ssm_g_a.weight", {nullptr, "kda_g_a.weight"}},
    {"ssm_g_b.weight", {nullptr, "kda_g_b.weight"}},
    {"ssm_dt.bias", {nullptr, "kda_dt_bias.weight"}},
    {"ssm_beta.weight", {nullptr, "kda_beta.weight"}},
    {"ssm_a", {nullptr, "kda_a_log.weight"}},
    {"ssm_norm.weight", {nullptr, "kda_o_norm.weight"}},
};
constexpr Names kMetaNames[] = {
    {"kda.gate_lower_bound", {"ssm.gate_lower_bound", "linear_attention.gate_lower_bound"}},
    {"kda.head_dim", {nullptr, "linear_attention.head_dimension"}},
    {"ssm.conv_kernel", {nullptr, "linear_attention.conv_kernel"}},
    {"attention.key_length_mla", {nullptr, "attention.key_length"}},
    {"attention.value_length_mla", {nullptr, "attention.value_length"}},
    {"attention.indexer.kpool", {nullptr, "attention.indexer.pool_size"}},
    {"hyper_connection.count", {"attention.hc.mult"}},
    {"hyper_connection.sinkhorn_iterations", {"attention.hc.sinkhorn_iters"}},
    {"hyper_connection.epsilon", {"attention.hc.eps"}},
    {"swiglu_clamp_exp", {"swiglu_limit"}},   // one limit for every layer, and for the shared expert too
    {"swiglu_clamp_shexp", {"swiglu_limit"}},
};

// table's entry for `name`, or null
template <size_t N> const Names* names_of(const Names (&table)[N], const char* name) {
    for (const Names& n : table)
        if (std::strcmp(n.name, name) == 0) return &n;
    return nullptr;
}

// A SwiGLU clamp limit per layer, stored as one value or one per layer.  Absent or zero means no clamp.
std::vector<float> clamp_limits(const MetaValue* v, int n_layer) {
    std::vector<float> out((size_t) n_layer, INFINITY);
    if (!v) return out;
    for (int il = 0; il < n_layer; ++il) {
        double x = v->f;
        if (v->type == MetaType::ARRAY) {
            if ((size_t) il >= v->items.size())
                throw std::runtime_error("the GGUF's SwiGLU clamp has no value for layer " + std::to_string(il));
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

// The model's GGUF: one file, or the numbered files llama.cpp splits a large one into.  The first has the
// metadata, and a tensor is in the file that lists it.
class Shards {
public:
    explicit Shards(const std::string& path) {
        files_.push_back(std::make_unique<GgufFile>(path));
        paths_.push_back(path);
        const MetaValue* no = files_[0]->get("split.no");
        const MetaValue* count = files_[0]->get("split.count");
        if (no && no->is_num() && no->u != 0)
            throw std::runtime_error(path + " is not the first file of its model: give the one whose name ends in "
                                            "-00001-of-...");
        const int n = count && count->is_num() ? (int) count->u : 1;
        if (n > 1) {
            char tail[40];
            std::snprintf(tail, sizeof(tail), "-00001-of-%05d.gguf", n);
            const size_t len = std::strlen(tail);
            if (path.size() < len || path.compare(path.size() - len, len, tail) != 0)
                throw std::runtime_error(path + " is the first of " + std::to_string(n) +
                                         " files, and the others are found by its name: it must end in " + tail);
            const std::string base = path.substr(0, path.size() - len);
            for (int i = 2; i <= n; ++i) {
                std::snprintf(tail, sizeof(tail), "-%05d-of-%05d.gguf", i, n);
                files_.push_back(std::make_unique<GgufFile>(base + tail));
                paths_.push_back(base + tail);
            }
        }
        for (const auto& f : files_)
            for (const TensorInfo& t : f->tensors())
                if (!where_.emplace(t.name, Where{&t, f.get()}).second)
                    throw std::runtime_error("tensor " + t.name + " is in two of the model's files");
    }

    const MetaValue* get(const std::string& key) const { return files_[0]->get(key); }
    const std::vector<std::string>& paths() const { return paths_; }
    const TensorInfo* find(const std::string& name) const {
        const auto it = where_.find(name);
        return it == where_.end() ? nullptr : it->second.t;
    }
    const uint8_t* data(const TensorInfo& t) const { return where_.at(t.name).file->tensor_data(t); }
    /// A tensor's byte count, after checking that its file holds that much.
    size_t payload(const TensorInfo& t) const {
        const GgufFile& g = *where_.at(t.name).file;
        const uint64_t bytes = tensor_payload_bytes(t), room = g.file_size() - g.data_start();
        if (bytes == 0 || t.offset > room || bytes > room - t.offset)
            throw std::runtime_error("tensor " + t.name + " " + shape_of(t) + " does not fit its file");
        return (size_t) bytes;
    }

private:
    struct Where {
        const TensorInfo* t;
        const GgufFile* file;
    };
    std::vector<std::unique_ptr<GgufFile>> files_;
    std::vector<std::string> paths_;
    std::unordered_map<std::string, Where> where_;
};

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

// A layer's tensor, under any of its names.
const TensorInfo* Engine::tensor(int il, const char* name) const {
    if (const TensorInfo* t = gg_->find(blk(il, name))) return t;
    if (const Names* n = names_of(kTensorNames, name))
        for (const char* other : n->other)
            if (const TensorInfo* t = other ? gg_->find(blk(il, other)) : nullptr) return t;
    return nullptr;
}

void Engine::load_geometry() {
    const Shards& g = *gg_;
    const MetaValue* arch = g.get("general.architecture");
    // Unsloth's converter writes the name without its hyphen
    if (!arch || (arch->s != "glm5-next" && arch->s != "glm5next"))
        throw std::runtime_error("general.architecture is '" + (arch ? arch->s : std::string()) +
                                 "', this engine runs 'glm5-next'");
    const std::string pre = arch->s + ".";
    auto find = [&](const char* key) -> const MetaValue* {
        if (const MetaValue* v = g.get(pre + key)) return v;
        if (const Names* n = names_of(kMetaNames, key))
            for (const char* other : n->other)
                if (const MetaValue* v = other ? g.get(pre + other) : nullptr) return v;
        return nullptr;
    };
    auto meta = [&](const char* key) -> const MetaValue& {
        const MetaValue* v = find(key);
        if (!v) throw std::runtime_error("the GGUF has no " + pre + key);
        return *v;
    };
    auto meta_i = [&](const char* key) { return (int) meta(key).u; };
    auto meta_f = [&](const char* key) { return (float) meta(key).f; };

    // a file can carry the block that predicts a second token after the model's own; it is not run here
    const MetaValue* nextn = find("nextn_predict_layers");
    n_layer_ = meta_i("block_count") - (nextn ? (int) nextn->u : 0);
    n_embd_ = meta_i("embedding_length");
    n_ff_dense_ = meta_i("feed_forward_length");
    n_head_ = meta_i("attention.head_count");
    n_vocab_ = meta_i("vocab_size");
    rms_eps_ = meta_f("attention.layer_norm_rms_epsilon");
    // the indexer's LayerNorm; a converter that leaves the key out means the model's own 1e-6
    const MetaValue* norm_eps = find("attention.layer_norm_epsilon");
    norm_eps_ = norm_eps ? (float) norm_eps->f : 1e-6f;
    n_expert_ = meta_i("expert_count");
    n_used_ = meta_i("expert_used_count");
    n_ff_exp_ = meta_i("expert_feed_forward_length");
    w_scale_ = meta_f("expert_weights_scale");
    w_norm_ = meta("expert_weights_norm").u != 0;
    kda_dim_ = meta_i("kda.head_dim");
    kda_lb_ = meta_f("kda.gate_lower_bound");
    q_rank_ = meta_i("attention.q_lora_rank");
    kv_rank_ = meta_i("attention.kv_lora_rank");
    mla_k_ = meta_i("attention.key_length_mla");
    mla_v_ = meta_i("attention.value_length_mla");
    idx_heads_ = meta_i("attention.indexer.head_count");
    idx_dim_ = meta_i("attention.indexer.key_length");
    idx_top_k_ = meta_i("attention.indexer.top_k");
    // tokens to a pool: a key, or else the second axis of a layer's pooling table
    if (const MetaValue* kpool = find("attention.indexer.kpool")) kpool_ = (int) kpool->u;
    for (int il = 0; il < n_layer_ && kpool_ == 0; ++il)
        if (const TensorInfo* ape = tensor(il, "indexer_compressor_ape.weight"))
            kpool_ = ape->shape.size() == 2 ? (int) ape->shape[1] : 0;
    sinkhorn_ = meta_i("hyper_connection.sinkhorn_iterations");
    hc_eps_ = meta_f("hyper_connection.epsilon");
    d_inner_ = n_head_ * kda_dim_;

    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported model: ") + what);
    };
    need(meta_i("hyper_connection.count") == kHc, "hyper_connection.count is not 4");
    need(meta_i("ssm.conv_kernel") == kConv, "ssm.conv_kernel is not 4");
    const MetaValue* gating = find("expert_gating_func");   // left out by one converter: the model's sigmoid
    need(!gating || gating->u == 2, "the expert gate is not a sigmoid");
    need(meta_i("expert_shared_count") == 1, "expert_shared_count is not 1");
    need(n_layer_ > 0 && n_embd_ > 0 && n_embd_ % 32 == 0 && n_vocab_ > 0, "bad block_count or embedding_length");
    need(kda_dim_ > 0 && kda_dim_ % 32 == 0 && kda_dim_ <= 256, "kda.head_dim is not a multiple of 32 up to 256");
    need(idx_dim_ > 0 && idx_dim_ <= 256, "the indexer key is longer than 256");
    need(n_used_ >= 1 && n_used_ <= kMaxUsed && n_used_ <= n_expert_, "bad expert_used_count");
    need(kpool_ > 1 && idx_top_k_ > 0 && idx_top_k_ % kpool_ == 0, "indexer.top_k is not whole pools");

    clamp_exp_ = clamp_limits(find("swiglu_clamp_exp"), n_layer_);
    clamp_shexp_ = find("swiglu_clamp_shexp") ? clamp_limits(find("swiglu_clamp_shexp"), n_layer_) : clamp_exp_;
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
    const size_t bytes = gg_->payload(*t), count = (size_t) t->elements();
    const bool as_is = !fix && (t->type == kTypeF32 || (want != kF32 && t->type == kTypeBF16) ||
                                (want == kAnyMat && kernels::native_mmvq_supported((int) t->type)));
    const uint32_t type = as_is ? t->type : want == kF32 ? kTypeF32 : kTypeBF16;
    const size_t out = as_is ? bytes : count * (type == kTypeF32 ? sizeof(float) : sizeof(uint16_t));
    const Dev d{dense_ ? dense_ + dense_at_ : nullptr, type, (int) n_in, (int) n_out};
    dense_at_ += pad(out);
    if (!dense_) return d;
    if (dense_at_ > dense_bytes_) throw std::logic_error("the dense arena is too small for " + t->name);
    if (as_is) {
        ck(cudaMemcpy(d.p, gg_->data(*t), bytes, cudaMemcpyHostToDevice), "weight upload");
        return d;
    }

    std::vector<float> f(count);
    const ggml_type_traits* traits = ggml_get_type_traits((ggml_type) t->type);
    if (t->type == kTypeF32) std::memcpy(f.data(), gg_->data(*t), bytes);
    else if (traits && traits->to_float) traits->to_float(gg_->data(*t), f.data(), (int64_t) count);
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

void Engine::load_layer(int il) {
    Layer& L = layers_[(size_t) il];
    auto f32 = [&](const char* n, int64_t a, int64_t b = 1) { return upload(tensor(il, n), blk(il, n), a, b, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b, Want w = kAnyMat) {
        return upload(tensor(il, n), blk(il, n), a, b, w);
    };
    auto has = [&](const char* n) { return tensor(il, n) != nullptr; };
    // the low rank of a gate's two-matrix projection is the first matrix's row count
    auto rank = [&](const char* n) {
        const TensorInfo* t = tensor(il, n);
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
        // kda_step() takes -exp(A_log), which is what one converter stores.  Another stores exp(A_log), and the
        // sign tells those two apart.  The third stores A_log, under a name that says so.
        const TensorInfo* a = tensor(il, "ssm_a");
        float first = -1.f;
        if (a && a->type == kTypeF32 && gg_->payload(*a) >= sizeof(first))
            std::memcpy(&first, gg_->data(*a), sizeof(first));
        float (*fix)(float) = nullptr;
        if (a && a->name.find("a_log") != std::string::npos) fix = [](float v) { return -std::exp(v); };
        else if (first > 0.f) fix = [](float v) { return -v; };
        L.a = upload(a, blk(il, "ssm_a"), n_head_, 1, kF32, fix);
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
    if (const TensorInfo* b = tensor(il, "exp_probs_b.bias")) {
        if (b->type != kTypeF32 || (int64_t) b->elements() != n_expert_)
            throw std::runtime_error(b->name + " is " + shape_of(*b) + ", expected F32 [" +
                                     std::to_string(n_expert_) + "]");
        gg_->payload(*b);
        std::memcpy(L.probs_b.data(), gg_->data(*b), (size_t) n_expert_ * sizeof(float));
    }

    // the routed experts stay in the file; fill_expert_cache() copies the ones VRAM has room for
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
    L.gu_bytes = gg_->payload(tg) / (size_t) n_expert_;
    L.d_bytes = gg_->payload(td) / (size_t) n_expert_;
    gg_->payload(tu);
    L.exp_gate = gg_->data(tg);
    L.exp_up = gg_->data(tu);
    L.exp_down = gg_->data(td);
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
    const size_t keys = (size_t) std::min(opt_.max_context, idx_top_k_ + kpool_ - 1);
    akeys_ = fbuf(std::min(n, (size_t) kAttnRows) * keys * kv_rank_);
    ascore_ = fbuf(std::min(n, (size_t) kAttnRows) * keys * n_head_);

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
    rows_ = opt_.dump ? 1 : std::min(opt_.prompt_batch, opt_.max_context);
    score_stride_ = opt_.max_context / kpool_ + 1;
    sel_stride_ = idx_top_k_ + kpool_;
    const size_t n = (size_t) rows_;

    // A prompt batch runs its routed experts through MMQ when the card has a tile for each of their types.
    bool tiles = opt_.gpu_experts && rows_ >= kStageRows && n_moe_ > 0;
    size_t one_expert = 0;
    for (const Layer& L : layers_) {
        if (!L.moe) continue;
        tiles = tiles && Mmq::fits((int) L.gu_type, 2 * n_ff_exp_, n_embd_) &&
                Mmq::fits((int) L.d_type, n_embd_, n_ff_exp_);
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
    std::vector<size_t> sizes;   // the slot sizes there are; a layer's place among them is its trade number
    size_t all = 0;
    for (int il = 0; il < n_layer_; ++il) {
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
        for (int il = 0; il < n_layer_; ++il) {
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
    gg_ = std::make_unique<Shards>(opt_.model);
    load_geometry();

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
        layers_.assign((size_t) n_layer_, Layer{});
        n_dsa_ = n_kda_ = n_moe_ = 0;
        gate_rank_ = 0;
        dense_at_ = 0;
        for (int il = 0; il < n_layer_; ++il) load_layer(il);
        out_norm_ = upload(gg_->find("output_norm.weight"), "output_norm.weight", n_embd_, 1, kF32);
        output_ = upload(gg_->find("output.weight"), "output.weight", n_embd_, n_vocab_, kAnyMat);
        if (pass == 0) {
            dense_bytes_ = dense_at_;
            dense_ = (uint8_t*) dmalloc(dense_bytes_ + 256, "the dense weights");
        }
    }

    alloc_state();
    if (!opt_.profile.empty()) read_profile(opt_.profile, n_layer_, n_expert_, prior_);
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
        node_ = stay_near(spans, gg_->paths(), threads, node_share_);
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
}

size_t Engine::row_bytes() const {
    return (size_t) n_dsa_ * ((size_t) kv_rank_ * sizeof(uint16_t) + (size_t) 3 * idx_dim_ * sizeof(float));
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
}

void Engine::save_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context)
        throw std::logic_error("save_rows: the rows are outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    const size_t lat = (size_t) kv_rank_ * sizeof(uint16_t), idx = (size_t) 3 * idx_dim_ * sizeof(float);
    const size_t at = (size_t) first, rows = (size_t) n;
    for (const Layer& L : layers_) {
        if (!L.dsa || rows == 0) continue;
        ck(cudaMemcpy(dst, (const uint8_t*) L.lat + at * lat, rows * lat, cudaMemcpyDeviceToHost), "save rows");
        dst += rows * lat;
        ck(cudaMemcpy(dst, (const uint8_t*) L.idx + at * idx, rows * idx, cudaMemcpyDeviceToHost), "save rows");
        dst += rows * idx;
    }
}

void Engine::load_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: the rows are outside what was saved");
    ck(cudaStreamSynchronize(cs(st_)), "load rows");
    const size_t lat = (size_t) kv_rank_ * sizeof(uint16_t), idx = (size_t) 3 * idx_dim_ * sizeof(float);
    const size_t at = (size_t) first, rows = (size_t) n, skip = (size_t) (first - from), all = (size_t) count;
    for (const Layer& L : layers_) {
        if (!L.dsa || rows == 0) continue;
        ck(cudaMemcpy((uint8_t*) L.lat + at * lat, src + skip * lat, rows * lat, cudaMemcpyHostToDevice), "load rows");
        src += all * lat;
        ck(cudaMemcpy((uint8_t*) L.idx + at * idx, src + skip * idx, rows * idx, cudaMemcpyHostToDevice), "load rows");
        src += all * idx;
    }
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
    dsa_attend(blas_, qabs_, L.lat, sel_, sel_stride_, idx_top_k_, kpool_, pos_, n, dense, akeys_, ascore_,
               std::min(rows_, kAttnRows), kqv_, kv_rank_, n_head_, 1.f / std::sqrt((float) mla_k_), st_);
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

// The router's scores, and the copies of them and of the layer's input to the host.
void Engine::moe_router(const Layer& L, int n) {
    const size_t rows = (size_t) n;
    lin(L.gate_inp, xn_, rl_, n, xq_);
    ck(cudaMemcpyAsync(h_rl_, rl_, rows * n_expert_ * sizeof(float), cudaMemcpyDeviceToHost, cs(st_)), "router");
    ck(cudaMemcpyAsync(h_x_, xn_, rows * n_embd_ * sizeof(float), cudaMemcpyDeviceToHost, cs(st_)), "router");
}

// The shared expert.  It runs on the GPU while the host routes.
void Engine::moe_shared(const Layer& L, int il, int n) {
    lin(L.sh_gate, xn_, g_, n, xq_);
    lin(L.sh_up, xn_, u_, n, xq_);
    swiglu_clamp(g_, u_, h_, n * n_ff_exp_, clamp_shexp_[(size_t) il], st_);
    lin(L.sh_down, h_, fo_, n);
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
    for (int t = 0; t < n; ++t) {
        int ids[kMaxUsed];
        float w[kMaxUsed];
        route(L, h_rl_ + (size_t) t * n_expert_, ids, w);

        CpuExpert host[kMaxUsed];
        const uint8_t* slot[kMaxUsed];
        float slot_w[kMaxUsed];
        int n_host = 0, n_slot = 0;
        const void* xq = (const uint8_t*) xq_ + (size_t) t * xq_row;
        float* out = fo_ + (size_t) t * e_row;
        for (int k = 0; k < n_used_; ++k) {
            const size_t e = (size_t) ids[k];
            ++use_[(size_t) il * n_expert_ + e];
            if (n_swap_ > 0) recent_[(size_t) il * n_expert_ + e] += 1.f;
            if (const uint8_t* s = L.slot[e]) {
                slot[n_slot] = s;
                slot_w[n_slot++] = w[k];
            } else {
                host[n_host++] = {L.exp_gate + e * L.gu_bytes, L.exp_up + e * L.gu_bytes,
                                  L.exp_down + e * L.d_bytes, w[k]};
            }
        }
        stats_.gpu_experts += n_slot;
        stats_.cpu_experts += n_host;
        // Five launches a cached expert.  Nothing else goes to the stream until the CPU experts are done, so one
        // of their threads hands these to the card meanwhile: on this thread they held the CPU experts up.
        const std::function<void()> cached = [&] {
            for (int i = 0; i < n_slot; ++i) {
                kernels::native_mmvq((int) L.gu_type, slot[i], xq, g_, n_embd_, 2 * n_ff_exp_, 1, st_);
                swiglu_clamp(g_, g_ + n_ff_exp_, h_, n_ff_exp_, limit, st_);
                kernels::native_quantize_q8_1(h_, q8_, n_ff_exp_, 1, st_);
                kernels::native_mmvq((int) L.d_type, slot[i] + 2 * L.gu_bytes, q8_, d_, n_ff_exp_, n_embd_, 1, st_);
                axpy(d_, slot_w[i], out, n_embd_, true, st_);
            }
        };

        // The next layer's router wait is what makes h_cpu_ free to write again.
        float* cpu = h_cpu_ + (size_t) t * e_row;
        if (n_host == 0) {
            cached();
            std::fill(cpu, cpu + e_row, 0.f);
            continue;
        }
        any_host = true;
        t0 = Clock::now();
        cpu_experts(*workers_, host, n_host, (int) L.gu_type, (int) L.d_type, n_embd_, n_ff_exp_,
                    h_x_ + (size_t) t * e_row, limit, cpu, n_slot > 0 ? &cached : nullptr);
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

// A layer from its input to its router's copy (a dense layer: to its end).  A prompt batch's MoE (moe_rows) does
// its own routing.
void Engine::layer_head(Layer& L, int il, int n) {
    hc_pre(r_, fp(L.hc_attn_fn), fp(L.hc_attn_scale), fp(L.hc_attn_base), mix_, hcw_, cur_, n_embd_, n, rms_eps_,
           hc_eps_, sinkhorn_, st_);
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
    if (!L.moe) {
        dense_ffn(L, il, n);
        dbg("ffn", il, fo_, n_embd_);
        hc_post(fo_, hcw_, r_, n_embd_, n, st_);
    } else if (!(mmq_ && n >= kStageRows)) {
        moe_router(L, n);
    }
}

void Engine::eval(const int32_t* tokens, int n, bool want_logits) {
    if (n < 1 || n > rows_) throw std::logic_error("eval takes 1 to max_rows() tokens");
    for (int t = 0; t < n; ++t)
        if (tokens[t] < 0 || tokens[t] >= n_vocab_) throw std::runtime_error("a token id is outside the vocabulary");
    if (pos_ + n > opt_.max_context)
        throw std::runtime_error("the context is full (" + std::to_string(opt_.max_context) + " tokens)");

    // h_r_ and h_cpu_ may still be the source of the previous call's copies
    ck(cudaStreamSynchronize(cs(st_)), "eval");
    if (n_swap_ > 0) follow(n < kStageRows ? n : 0);
    const size_t e_row = (size_t) n_embd_, r_row = kHc * e_row;
    for (int t = 0; t < n; ++t) {
        float* r = h_r_ + (size_t) t * r_row;
        embed(tokens[t], r);
        for (int h = 1; h < kHc; ++h) std::memcpy(r + (size_t) h * e_row, r, e_row * sizeof(float));
    }
    ck(cudaMemcpyAsync(r_, h_r_, (size_t) n * r_row * sizeof(float), cudaMemcpyHostToDevice, cs(st_)), "embed");

    // A token alone hands the card the same launches every time, some thirty a layer up to the router, and they
    // cost this thread more than the card.  They are recorded once, a graph a layer, and replayed with one call.
    // Not the sparse attention's: its launches take the position.
    const bool replay = opt_.graphs && n == 1 && !opt_.dump;
    for (int il = 0; il < n_layer_; ++il) {
        Layer& L = layers_[(size_t) il];
        if (replay && !L.dsa) {
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
        hc_post(fo_, hcw_, r_, n_embd_, n, st_);
    }

    if (want_logits) {
        const auto logits = [&] {
            hc_mean(r_ + (size_t) (n - 1) * r_row, cur_, n_embd_, st_);
            rms_norm(cur_, fp(out_norm_), xn_, n_embd_, 1, rms_eps_, st_);
            dbg("out_norm", n_layer_, xn_, n_embd_);
            lin(output_, xn_, logits_, 1);
            ck(cudaMemcpyAsync(h_logits_, logits_, (size_t) n_vocab_ * sizeof(float), cudaMemcpyDeviceToHost,
                               cs(st_)),
               "logits");
        };
        if (replay) {
            if (!tail_) tail_ = record(logits);
            ck(cudaGraphLaunch((cudaGraphExec_t) tail_, cs(st_)), "logits graph");
        } else {
            logits();
        }
        ck(cudaStreamSynchronize(cs(st_)), "eval");
    }
    pos_ += n;
    stats_.tokens += n;
}

}  // namespace strata::glm
