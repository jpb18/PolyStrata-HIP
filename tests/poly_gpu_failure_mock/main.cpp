#define STRATA_USE_HIP 1
#include "strata/poly/gpu_failure.hpp"
#include <cstring>

struct Cleanup {
    ~Cleanup() { std::puts("UNSAFE_GPU_CLEANUP"); }
};

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    Cleanup cleanup;
    const auto error = (hipError_t) std::atoi(argv[2]);
    last_error = (hipError_t) std::atoi(argv[3]);
    using namespace strata::poly;
    try {
        if (std::strcmp(argv[1], "query") == 0) hip_query_check(error, "routing query");
        else if (std::strcmp(argv[1], "expected") == 0) hip_clear_expected(error, "optional allocation");
        else if (std::strcmp(argv[1], "sync") == 0) {
            sync_error = error;
            hip_sync(nullptr, "logits boundary");
        } else if (std::strcmp(argv[1], "launch") == 0) hip_launch_check("kernel launch");
        else if (std::strcmp(argv[1], "state") == 0) gpu_fatal("generate", "KV state unknown");
        else if (std::strcmp(argv[1], "worker") == 0) {
            std::thread worker([&] { hip_check(error, "expert upload worker"); });
            worker.join();
        } else hip_check(error, "copy or event");
    } catch (...) {
        std::puts("UNSAFE_RECOVERY");
    }
    std::printf("CONTINUED sync=%d\n", synchronized);
}
