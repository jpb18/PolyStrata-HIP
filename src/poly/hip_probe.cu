// src/poly/hip_probe.cu - what strata-poly's way of driving the card costs on this PC, without a model
// (STRATA_POLY_HIP only; AMD_PORT_IMPLEMENTATION.md).
//
//   hip_poly_probe
//
// An answer's token goes through every MoE layer with a round trip: the router's scores to the host, a wait for
// them, the experts' launches back (Engine::moe()).  The engine records a layer's fixed launches as a graph and
// replays it.  It keeps 1 to 2 GiB of pinned host memory, and copies experts through it.  Each of those is timed
// here, with the host's clock (HIP's events do not bracket copies on Windows, docs/AMD_HIP.md #380).  It also says
// whether a query that finds work unfinished leaves hipErrorNotReady as the thread's last error, which a launch
// check after it would take for a failure.  Exit 0 when every step ran, 1 when one failed.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double us_since(Clock::time_point t) {
    return std::chrono::duration<double, std::micro>(Clock::now() - t).count();
}

bool ok(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("FAIL %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

__global__ void tiny(float* p) {
    if (threadIdx.x == 0 && blockIdx.x == 0) p[0] += 1.0f;
}

// about as much work as it is given: a chain of dependent multiply-adds a thread
__global__ void spin(float* p, int iters) {
    float x = p[threadIdx.x];
    for (int i = 0; i < iters; ++i) x = x * 1.0000001f + 1e-7f;
    p[threadIdx.x] = x;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main() {
    bool good = true;
    int dev = 0;
    cudaDeviceProp prop{};
    if (!ok(cudaGetDevice(&dev), "cudaGetDevice") || !ok(cudaGetDeviceProperties(&prop, dev), "properties")) return 1;
#if defined(STRATA_USE_HIP)
    std::printf("device %d: %s (%s), %d CUs, %.1f GiB\n", dev, prop.name, prop.gcnArchName, prop.multiProcessorCount,
                double(prop.totalGlobalMem) / double(1ull << 30));
#else
    std::printf("device %d: %s, %d SMs\n", dev, prop.name, prop.multiProcessorCount);
#endif
    cudaStream_t st = nullptr;
    if (!ok(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream")) return 1;
    float* d = nullptr;
    if (!ok(cudaMalloc(&d, 1 << 20), "cudaMalloc") || !ok(cudaMemset(d, 0, 1 << 20), "cudaMemset")) return 1;
    float* h = nullptr;
    if (!ok(cudaMallocHost(&h, 1 << 20), "cudaMallocHost")) return 1;
    cudaEvent_t ev = nullptr;
    if (!ok(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "event")) return 1;

    // 1. a query of unfinished work, and the thread's last error after it
    {
        spin<<<1, 256, 0, st>>>(d, 1 << 24);
        cudaEventRecord(ev, st);
        const cudaError_t q = cudaEventQuery(ev);
        const cudaError_t last = cudaPeekAtLastError();
        std::printf("query of unfinished work: %s; last error after it: %s\n", cudaGetErrorString(q),
                    cudaGetErrorString(last));
        if (q == cudaErrorNotReady && last == cudaErrorNotReady)
            std::printf("  -> NotReady is left as the last error: a launch check after a query must clear it\n");
        good = ok(cudaStreamSynchronize(st), "sync after query") && good;
        (void) cudaGetLastError();
    }

    // 2. what a launch costs the host, and the card's time for a tiny kernel back to back
    {
        constexpr int n = 20000;
        for (int i = 0; i < 100; ++i) tiny<<<1, 32, 0, st>>>(d);
        cudaStreamSynchronize(st);
        const Clock::time_point t0 = Clock::now();
        for (int i = 0; i < n; ++i) tiny<<<1, 32, 0, st>>>(d);
        const double submit = us_since(t0);
        good = ok(cudaStreamSynchronize(st), "launches") && good;
        const double all = us_since(t0);
        std::printf("launch: %.2f us a launch on the host, %.2f us each until the last has run (%d tiny kernels)\n",
                    submit / n, all / n, n);
        good = ok(cudaGetLastError(), "launch errors") && good;
    }

    // 3. the router's round trip: a launch, 1 KiB to pinned memory, an event, the host waits for it
    {
        std::vector<double> t;
        for (int i = 0; i < 2100; ++i) {
            const Clock::time_point t0 = Clock::now();
            tiny<<<1, 32, 0, st>>>(d);
            cudaMemcpyAsync(h, d, 1024, cudaMemcpyDeviceToHost, st);
            cudaEventRecord(ev, st);
            cudaEventSynchronize(ev);
            if (i >= 100) t.push_back(us_since(t0));
        }
        good = ok(cudaGetLastError(), "round trips") && good;
        std::printf("round trip (launch + 1 KiB D2H + event wait): median %.1f us, min %.1f, max %.1f\n", median(t),
                    *std::min_element(t.begin(), t.end()), *std::max_element(t.begin(), t.end()));
    }

    // 4. 35 small launches a layer: one after the other, and recorded once and replayed
    {
        constexpr int k = 35, reps = 1000;
        std::vector<double> direct, replay;
        for (int r = 0; r < reps + 50; ++r) {
            const Clock::time_point t0 = Clock::now();
            for (int i = 0; i < k; ++i) tiny<<<1, 32, 0, st>>>(d);
            cudaStreamSynchronize(st);
            if (r >= 50) direct.push_back(us_since(t0));
        }
        cudaGraph_t g = nullptr;
        cudaGraphExec_t ge = nullptr;
        bool rec = ok(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "begin capture");
        if (rec) {
            for (int i = 0; i < k; ++i) tiny<<<1, 32, 0, st>>>(d);
            rec = ok(cudaStreamEndCapture(st, &g), "end capture") &&
                  ok(cudaGraphInstantiateWithFlags(&ge, g, 0), "instantiate");
        }
        if (rec) {
            for (int r = 0; r < reps + 50; ++r) {
                const Clock::time_point t0 = Clock::now();
                cudaGraphLaunch(ge, st);
                cudaStreamSynchronize(st);
                if (r >= 50) replay.push_back(us_since(t0));
            }
            rec = ok(cudaGetLastError(), "graph launches");
            std::printf("35 launches + wait: median %.1f us one by one, %.1f us as a replayed graph\n", median(direct),
                        median(replay));
        }
        good = rec && good;
        if (ge) cudaGraphExecDestroy(ge);
        if (g) cudaGraphDestroy(g);
    }

    // 5. the engine's pinned memory: two staging buffers and the rows of a read, 512 MiB each
    std::vector<void*> big;
    {
        const Clock::time_point t0 = Clock::now();
        for (int i = 0; i < 3; ++i) {
            void* p = nullptr;
            if (!ok(cudaMallocHost(&p, size_t(512) << 20), "512 MiB of pinned memory")) {
                good = false;
                break;
            }
            big.push_back(p);
        }
        std::printf("pinned memory: %zu of 3 buffers of 512 MiB in %.0f ms\n", big.size(), us_since(t0) / 1000.0);
    }

    // 6. host to device from pinned memory, as the staging buffer copies experts
    if (!big.empty()) {
        void* dbig = nullptr;
        if (ok(cudaMalloc(&dbig, size_t(512) << 20), "512 MiB of VRAM")) {
            std::memset(big[0], 1, size_t(512) << 20);
            std::vector<double> t;
            for (int i = 0; i < 6; ++i) {
                const Clock::time_point t0 = Clock::now();
                cudaMemcpyAsync(dbig, big[0], size_t(512) << 20, cudaMemcpyHostToDevice, st);
                cudaStreamSynchronize(st);
                if (i > 0) t.push_back(us_since(t0));
            }
            good = ok(cudaGetLastError(), "copies") && good;
            std::printf("host to device, 512 MiB from pinned memory: %.1f GB/s (median of 5)\n",
                        double(size_t(512) << 20) / (median(t) * 1e3));
            cudaFree(dbig);
        } else {
            good = false;
        }
    }
    for (void* p : big) cudaFreeHost(p);

    size_t free_b = 0, total_b = 0;
    if (ok(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo"))
        std::printf("VRAM: %.0f MiB free of %.0f (Windows: within the budget it gives this process)\n",
                    double(free_b) / 1048576.0, double(total_b) / 1048576.0);
    cudaEventDestroy(ev);
    cudaFreeHost(h);
    cudaFree(d);
    cudaStreamDestroy(st);
    std::printf("%s\n", good ? "probe: every step ran" : "probe: a step failed");
    return good ? 0 : 1;
}
