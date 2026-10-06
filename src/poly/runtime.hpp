// src/poly/runtime.hpp - what the engine's source files share and its users do not see.
#pragma once

#include "engine.hpp"

#include "strata/artifact/gguf_reader.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::poly {

constexpr uint32_t kTypeF32 = 0, kTypeBF16 = 30;
constexpr int kMaxUsed = 32;
constexpr int kSumRows = 32;        // routed experts one launch adds into their tokens' rows (ScaledRows)
constexpr int kCpuRows = 8;         // tokens whose CPU experts one call computes; a window has no more
// From this many tokens, a call runs its routed experts on the GPU alone and copies the ones that are not cached
// through VRAM.  Below it, that copy costs more than the CPU takes to compute the few experts.
constexpr int kStageRows = 32;
constexpr size_t kChunkBytes = size_t(512) << 20;   // one upload, and each of the two pinned buffers
// Zero bytes behind every matrix the batched kernels may multiply (Mmq::fits): they read up to a step of 256
// columns past a row that ends inside one, which is at most 7 blocks of 34 bytes.
constexpr size_t kTail = 256;
// follow(): an expert's recent use is the count of its uses, halved every kSwapHalf tokens.  An expert takes a
// cached one's slot when it was used twice lately and more than twice as much as that one.  Counts this small are
// noisy, and an upload gets in the way of the work: a slot goes to an expert that will clearly be used more.
constexpr float kSwapMin = 2.f;
constexpr int kSwapHalf = 128;

using Clock = std::chrono::steady_clock;

inline double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

inline void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

inline size_t pad(size_t n) { return (n + 255) & ~size_t(255); }

inline std::string blk(int il, const char* name) { return "blk." + std::to_string(il) + "." + name; }

inline const float* fp(const Dev& d) { return (const float*) d.p; }
inline cudaStream_t cs(void* s) { return (cudaStream_t) s; }

inline std::string shape_of(const TensorInfo& t) {
    std::string s = "[";
    for (size_t d = 0; d < t.shape.size(); ++d) s += (d ? ", " : "") + std::to_string(t.shape[d]);
    return s + "] " + t.type_name();
}

// table's entry for `name`, or null
template <size_t N> const Names* names_of(const Names (&table)[N], const char* name) {
    for (const Names& n : table)
        if (std::strcmp(n.name, name) == 0) return &n;
    return nullptr;
}

// A SwiGLU clamp limit per layer, stored as one value or one per layer.  Absent or zero means no clamp.
inline std::vector<float> clamp_limits(const MetaValue* v, int n_layer) {
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

}  // namespace strata::poly
