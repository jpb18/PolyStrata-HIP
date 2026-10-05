// src/glm/cpu_experts.hpp - routed experts computed on the CPU from their GGUF bytes.
//
// The dot products and the activation quantizers are ggml-cpu's (ggml_get_type_traits_cpu), so an expert here
// computes what llama.cpp's CPU backend computes for the same tensor.  The workers are this file's own: the pool
// in strata/kernels/cpu/pool.hpp is built around the other model's expert blob.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::glm {

/// Runs fn(0..n-1) on the workers and the calling thread, and returns when all of them are done.
class Workers {
public:
    explicit Workers(int threads);
    ~Workers();
    Workers(const Workers&) = delete;
    Workers& operator=(const Workers&) = delete;

    void run(int n, const std::function<void(int)>& fn);
    int threads() const { return (int) th_.size() + 1; }

private:
    struct Batch {
        const std::function<void(int)>* fn;
        int n;
        std::atomic<int> next{0}, left{0};
    };
    void loop();
    static void drain(Batch& b);

    std::vector<std::thread> th_;
    std::atomic<Batch*> cur_{nullptr};
    std::atomic<int> users_{0};      ///< workers that may hold a Batch pointer
    std::atomic<int> sleepers_{0};
    std::atomic<bool> quit_{false};
    std::mutex m_;
    std::condition_variable cv_;
};

/// One routed expert of a layer: its three matrices in the GGUF, and its router weight.
struct CpuExpert {
    const uint8_t* gate;
    const uint8_t* up;
    const uint8_t* down;
    float weight;
};

/// Checks that ggml-cpu has a dot product for both types.  Call once per layer format at load.
bool cpu_expert_types_ok(int gu_type, int d_type, int n_embd, int n_ff, std::string& err);

/// out[n_embd] = sum over the experts of weight * down(silu(min(gate x, limit)) * clamp(up x, -limit, limit)).
void cpu_experts(Workers& workers, const CpuExpert* experts, int n, int gu_type, int d_type, int n_embd, int n_ff,
                 const float* x, float limit, float* out);

}  // namespace strata::glm
