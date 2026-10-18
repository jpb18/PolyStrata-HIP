// A HIP engine is a disposable process. Never unwind through GPU cleanup after a
// device failure: a synchronize, free or stream destroy may hang on that context.
#pragma once

#if defined(STRATA_USE_HIP) || defined(GGML_USE_HIP)
#include <hip/hip_runtime_api.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace strata::poly {

inline std::atomic_flag& gpu_failure_flag() {
    static std::atomic_flag failed = ATOMIC_FLAG_INIT;
    return failed;
}

inline void gpu_failure_guard() {
    // Inference may share the device with the expert upload thread. Stop publishing
    // tokens as soon as that thread latches a failure, while it emits the diagnostic.
    while (gpu_failure_flag().test()) std::this_thread::yield();
}

[[noreturn]] inline void gpu_fatal(const char* operation, const char* detail,
                                  int code = -1, const char* name = "inference_state_unknown") {
    if (!gpu_failure_flag().test_and_set()) {
        // One diagnostic, and one machine-readable protocol line. No further HIP
        // calls, cache persistence, destructors or atexit handlers after detection.
        std::fprintf(stderr, "strata-poly: GPU_FATAL backend=hip operation=%s code=%d name=%s: %s; "
                             "inference state discarded; recovery requires a new process\n",
                     operation, code, name, detail);
        std::fflush(stderr);
        std::printf("ERR GPU_FATAL backend=hip operation=%s code=%d name=%s: %s; new process required\n",
                    operation, code, name, detail);
        std::fflush(stdout);
        std::_Exit(70);
    }
    // A second failing thread must not end the process before the first emits its
    // diagnostic. It also must not return to inference while that happens.
    for (;;) std::this_thread::yield();
}

inline void hip_check(hipError_t error, const char* operation) {
    if (error != hipSuccess)
        gpu_fatal(operation, hipGetErrorString(error), (int) error, hipGetErrorName(error));
}

inline void hip_launch_check(const char* operation) {
    const hipError_t error = hipGetLastError();
    // Older runtimes also leave the result of a nonblocking query here.
    if (error != hipErrorNotReady) hip_check(error, operation);
}

inline void hip_query_check(hipError_t error, const char* operation) {
    if (error != hipErrorNotReady) hip_check(error, operation);
    // Do not erase an unrelated fatal error while clearing a benign NotReady.
    hip_launch_check(operation);
}

inline void hip_clear_expected(hipError_t expected, const char* operation) {
    const hipError_t error = hipGetLastError();
    if (error != expected && error != hipErrorNotReady) hip_check(error, operation);
}

inline void hip_sync(hipStream_t stream, const char* operation) {
    hip_launch_check(operation);  // launch errors need not be returned by synchronize
    hip_check(hipStreamSynchronize(stream), operation);  // asynchronous execution errors
}

}  // namespace strata::poly
#endif
