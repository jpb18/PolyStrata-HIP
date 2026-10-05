// src/glm/cpu_experts.cpp - see the header.
#include "cpu_experts.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <immintrin.h>

namespace strata::glm {
namespace {

constexpr int kRowsPerTask = 64;
constexpr int kSpinsBeforeSleep = 20000;   // about a millisecond of pause instructions

const ggml_type_traits_cpu* traits(int type) { return ggml_get_type_traits_cpu((ggml_type) type); }

void init_once() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_cpu_init(); });
}

}  // namespace

Workers::Workers(int threads) {
    for (int i = 1; i < threads; ++i) th_.emplace_back([this] { loop(); });
}

Workers::~Workers() {
    quit_.store(true);
    { std::lock_guard<std::mutex> l(m_); }
    cv_.notify_all();
    for (auto& t : th_) t.join();
}

void Workers::drain(Batch& b) {
    for (;;) {
        const int i = b.next.fetch_add(1, std::memory_order_relaxed);
        if (i >= b.n) return;
        (*b.fn)(i);
        b.left.fetch_sub(1, std::memory_order_release);
    }
}

// A worker with nothing to do only reads cur_: a write there by each of them, every turn of the loop, kept run()
// from ever seeing users_ at zero for long.  When there is a batch, the worker counts itself in users_ and reads
// cur_ again before it uses it, so run() can tell when nobody holds its Batch any more.
void Workers::loop() {
    int idle = 0;
    while (!quit_.load(std::memory_order_relaxed)) {
        Batch* b = cur_.load(std::memory_order_acquire);
        if (b) {
            users_.fetch_add(1);
            b = cur_.load();
            if (b) drain(*b);
            users_.fetch_sub(1);
        }
        if (b) {
            idle = 0;
            continue;
        }
        if (++idle < kSpinsBeforeSleep) {
            _mm_pause();
            continue;
        }
        std::unique_lock<std::mutex> l(m_);
        sleepers_.fetch_add(1);
        cv_.wait(l, [this] { return cur_.load() != nullptr || quit_.load(); });
        sleepers_.fetch_sub(1);
        idle = 0;
    }
}

void Workers::run(int n, const std::function<void(int)>& fn) {
    if (n <= 0) return;
    if (n == 1) return fn(0);   // nothing to share out
    Batch b;
    b.fn = &fn;
    b.n = n;
    b.left.store(n);
    cur_.store(&b);
    if (sleepers_.load() > 0) {
        { std::lock_guard<std::mutex> l(m_); }
        cv_.notify_all();
    }
    drain(b);
    cur_.store(nullptr);   // every task is taken: a worker that comes now has nothing to find
    while (b.left.load(std::memory_order_acquire) != 0) _mm_pause();
    while (users_.load() != 0) _mm_pause();
}

bool cpu_expert_types_ok(int gu_type, int d_type, int n_embd, int n_ff, std::string& err) {
    init_once();
    for (const int type : {gu_type, d_type}) {
        const ggml_type_traits_cpu* t = traits(type);
        const ggml_type_traits_cpu* a = t && t->vec_dot ? traits(t->vec_dot_type) : nullptr;
        if (a == nullptr || a->from_float == nullptr) {
            err = "ggml-cpu has no dot product for tensor type " + std::to_string(type);
            return false;
        }
        const int n = type == gu_type ? n_embd : n_ff;
        if (n % ggml_blck_size((ggml_type) type) || n % ggml_blck_size(t->vec_dot_type)) {
            err = "a row of " + std::to_string(n) + " values is not whole blocks of tensor type " + std::to_string(type);
            return false;
        }
    }
    return true;
}

void cpu_experts(Workers& workers, const CpuExpert* experts, int n, int gu_type, int d_type, int n_embd, int n_ff,
                 const float* x, float limit, float* out, const std::function<void()>* meanwhile) {
    init_once();
    const ggml_type_traits_cpu* tg = traits(gu_type);
    const ggml_type_traits_cpu* td = traits(d_type);
    const size_t gu_row = ggml_row_size((ggml_type) gu_type, n_embd);
    const size_t d_row = ggml_row_size((ggml_type) d_type, n_ff);
    const size_t h_bytes = ggml_row_size(td->vec_dot_type, n_ff);

    static thread_local std::vector<uint8_t> act, hq;
    static thread_local std::vector<float> h;
    act.resize(ggml_row_size(tg->vec_dot_type, n_embd));
    hq.resize((size_t) n * h_bytes);
    h.resize((size_t) n * n_ff);
    traits(tg->vec_dot_type)->from_float(x, act.data(), n_embd);

    const uint8_t* actp = act.data();
    float* hp = h.data();
    const int gu_tasks = (n_ff + kRowsPerTask - 1) / kRowsPerTask;
    const int first = meanwhile ? 1 : 0;   // task 0, so it is taken at once
    workers.run(first + n * gu_tasks, [&](int t) {
        if (t < first) return (*meanwhile)();
        t -= first;
        const CpuExpert& e = experts[t / gu_tasks];
        float* he = hp + (size_t) (t / gu_tasks) * n_ff;
        const int r0 = (t % gu_tasks) * kRowsPerTask, r1 = std::min(n_ff, r0 + kRowsPerTask);
        for (int r = r0; r < r1; ++r) {
            float g = 0.f, u = 0.f;
            tg->vec_dot(n_embd, &g, 0, e.gate + (size_t) r * gu_row, 0, actp, 0, 1);
            tg->vec_dot(n_embd, &u, 0, e.up + (size_t) r * gu_row, 0, actp, 0, 1);
            g = std::min(g, limit);
            u = std::clamp(u, -limit, limit);
            he[r] = g / (1.f + std::exp(-g)) * u;
        }
    });
    for (int e = 0; e < n; ++e)
        traits(td->vec_dot_type)->from_float(hp + (size_t) e * n_ff, hq.data() + (size_t) e * h_bytes, n_ff);

    const uint8_t* hqp = hq.data();
    const int d_tasks = (n_embd + kRowsPerTask - 1) / kRowsPerTask;
    workers.run(d_tasks, [&](int t) {
        const int r0 = t * kRowsPerTask, r1 = std::min(n_embd, r0 + kRowsPerTask);
        for (int r = r0; r < r1; ++r) {
            float acc = 0.f;
            for (int e = 0; e < n; ++e) {
                float s = 0.f;
                td->vec_dot(n_ff, &s, 0, experts[e].down + (size_t) r * d_row, 0, hqp + (size_t) e * h_bytes, 0, 1);
                acc += experts[e].weight * s;
            }
            out[r] = acc;
        }
    });
}

}  // namespace strata::glm
