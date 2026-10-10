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

// the router's copy as a kernel: n words to pinned memory, then (flag) a number the host waits for
__global__ void copy_out(const float* src, float* dst, int n, unsigned* flag, unsigned seq) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i] + float(seq);
    if (!flag) return;
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        *(volatile unsigned*) flag = seq;
    }
}

__global__ void set_flag(unsigned* flag, unsigned seq) {
    __threadfence_system();
    *(volatile unsigned*) flag = seq;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

#if defined(STRATA_USE_HIP) && defined(_WIN32)
// hip_poly_probe vram: VRAM taken 512 MiB at a time, and after each step what the HIP runtime says is free and what
// the engine counts free within Windows' budget for the process (cudaMemGetInfo, src/core/device.cu): the gap is
// what the budget withholds, and whether it moves as the process takes more.
int vram_steps() {
    std::vector<void*> got;
    for (int step = 0; step < 40; ++step) {
        size_t raw_free = 0, total = 0, free_b = 0, total_b = 0;
        hipMemGetInfo(&raw_free, &total);
        cudaMemGetInfo(&free_b, &total_b);
        std::printf("taken %6zu MiB: HIP free %6zu MiB, free within the budget %6zu MiB, withheld %6zu MiB\n",
                    got.size() * 512, raw_free >> 20, free_b >> 20, (raw_free - free_b) >> 20);
        void* p = nullptr;
        if (cudaMalloc(&p, size_t(512) << 20) != cudaSuccess) {
            (void) cudaGetLastError();
            std::printf("no more VRAM\n");
            break;
        }
        cudaMemset(p, 1, size_t(512) << 20);
        cudaDeviceSynchronize();
        got.push_back(p);
    }
    for (void* p : got) cudaFree(p);
    return 0;
}
#endif

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // a step that hangs leaves the ones before it on the screen
#if defined(STRATA_USE_HIP) && defined(_WIN32)
    if (argc > 1 && std::strcmp(argv[1], "vram") == 0) return vram_steps();
#else
    (void) argc, (void) argv;
#endif
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

#if defined(STRATA_USE_HIP)
    // 3b. the same trip other ways, as a decode's layer has it: the kernel that makes the scores, their copy out
    // (9 KiB: the scores and the experts' input), the host's wait, and the next launch, which the card waits for.
    // 0: memcpy + event wait (the engine before the port); 1: a copy kernel + event wait (the engine now on
    // Windows); 2: 1 with the event polled; 3: the copy kernel also writes a number to coherent pinned memory after
    // the data, and the host polls that; 4: 3 with one hipStreamQuery after the launches, which hands them to the
    // card (the runtime may hold launches back until something waits); 5: 4 with the data to ordinary pinned memory
    // and the number by a kernel of its own behind the copy's.  A poll gives up after 1 s (the trip counts as timed
    // out, and the stream is waited for).  The data is checked each time.
    {
        constexpr int words = 2304, kWays = 6;
        unsigned* hflag = nullptr;
        float* hc = nullptr;
        bool can = ok(hipHostMalloc((void**) &hflag, 64, hipHostMallocCoherent | hipHostMallocMapped),
                      "coherent pinned memory") &&
                   ok(hipHostMalloc((void**) &hc, words * sizeof(float), hipHostMallocCoherent | hipHostMallocMapped),
                      "coherent pinned memory");
        float* hn = nullptr;
        can = can && ok(cudaMallocHost(&hn, words * sizeof(float)), "pinned memory");
        if (can) {
            std::vector<float> ramp(words);
            for (int i = 0; i < words; ++i) ramp[i] = float(i);
            cudaMemcpy(d + 4096, ramp.data(), words * sizeof(float), cudaMemcpyHostToDevice);
            *hflag = 0;
            const char* names[kWays] = {"memcpy + event wait", "copy kernel + event wait", "copy kernel + event polled",
                                        "copy kernel + flag polled", "same, launches handed over",
                                        "same, flag kernel"};
            unsigned seq = 0;
            for (int way = 0; way < kWays; ++way) {
                std::vector<double> t;
                int bad = 0, late = 0;
                float* dst = way == 3 || way == 4 ? hc : hn;
                for (int i = 0; i < 2100 && late < 3; ++i) {
                    ++seq;
                    const Clock::time_point t0 = Clock::now();
                    tiny<<<1, 32, 0, st>>>(d);
                    if (way == 0) {
                        copy_out<<<1, 256, 0, st>>>(d + 4096, d + 8192, words, nullptr, seq);
                        cudaMemcpyAsync(dst, d + 8192, words * sizeof(float), cudaMemcpyDeviceToHost, st);
                    } else {
                        copy_out<<<1, 256, 0, st>>>(d + 4096, dst, words, way == 3 || way == 4 ? hflag : nullptr,
                                                    seq);
                        if (way == 5) set_flag<<<1, 1, 0, st>>>(hflag, seq);
                    }
                    if (way < 3) cudaEventRecord(ev, st);
                    if (way < 2) {
                        cudaEventSynchronize(ev);
                    } else if (way == 2) {
                        while (cudaEventQuery(ev) == cudaErrorNotReady) {
                        }
                    } else {
                        if (way >= 4) (void) cudaStreamQuery(st);
                        while (*(volatile unsigned*) hflag != seq) {
                            if (us_since(t0) > 1e6) {
                                ++late;
                                cudaStreamSynchronize(st);
                                break;
                            }
                        }
                    }
                    (void) cudaGetLastError();   // a query's NotReady
                    if (dst[words - 1] != float(words - 1) + float(seq) || dst[7] != 7.0f + float(seq)) ++bad;
                    tiny<<<1, 32, 0, st>>>(d);   // the experts' launch the card waits for
                    if (i >= 100) t.push_back(us_since(t0));
                }
                cudaStreamSynchronize(st);
                good = ok(cudaGetLastError(), names[way]) && good;
                if (late > 0) {
                    std::printf("layer trip, %-27s: the host's poll timed out (%d times): the card had not run it\n",
                                names[way], late);
                    continue;
                }
                std::printf("layer trip, %-27s: median %.1f us, min %.1f, max %.1f; stale data %d of %zu\n",
                            names[way], median(t), *std::min_element(t.begin(), t.end()),
                            *std::max_element(t.begin(), t.end()), bad, t.size() + 100);
                good = good && bad == 0;
            }
        } else {
            good = false;
        }
        if (hn) cudaFreeHost(hn);
        if (hc) hipHostFree(hc);
        if (hflag) hipHostFree(hflag);
    }
#endif

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
