// src/core/device.cu - P2.S1: the CUDA side of the runtime core.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/core/device.hpp"

#include <cstdio>
#include <cstring>

namespace strata::core {

namespace {

void check(dpct::err0 e, const char *what) {
    /*
    DPCT1000:1125: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (e != 0) {
        /*
        DPCT1009:1126: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1001:1124: The statement could not be removed.
        */
        throw CudaError(
            std::string(what) + ": " + dpct::get_error_string_dummy(e), (int)e);
    }
}

// A NaN pattern, not zero.  Zeros read from uninitialised memory are indistinguishable from real zeros in a
// dequantized weight or a masked attention score, which is exactly the kind of wrong-but-plausible value the
// Phase 1 harnesses kept catching.
__dpct_inline__ void poison_kernel(float *p, uint64_t n_floats) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const uint64_t i =
        (uint64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < n_floats) p[i] = sycl::bit_cast<float>(0x7fc00000);
}

}  // namespace

DeviceInfo device_info(int ordinal) {
    int count = 0;
    check(DPCT_CHECK_ERROR(count = dpct::device_count()), "cudaGetDeviceCount");
    if (count == 0) {
        throw CudaError("no CUDA device is present; Strata needs an NVIDIA GPU of compute capability 8.0 or newer", -1);
    }
    if (ordinal < 0 || ordinal >= count) {
        throw CudaError("device ordinal " + std::to_string(ordinal) + " is out of range (have " +
                            std::to_string(count) + ")",
                        -1);
    }
    DeviceInfo d;
    d.ordinal = ordinal;
    /*
    DPCT1093:1127: The "ordinal" device may be not the one intended for use.
    Adjust the selected device if needed.
    */
    check(DPCT_CHECK_ERROR(dpct::select_device(ordinal)), "cudaSetDevice");

    dpct::device_info p{};
    check(DPCT_CHECK_ERROR(dpct::get_device(ordinal).get_device_info(p)),
          "cudaGetDeviceProperties");
    d.name = p.get_name();
    /*
    DPCT1005:1128: The SYCL device version is different from CUDA Compute
    Compatibility. You may need to rewrite this code.
    */
    d.cc_major = p.get_major_version();
    /*
    DPCT1005:1129: The SYCL device version is different from CUDA Compute
    Compatibility. You may need to rewrite this code.
    */
    d.cc_minor = p.get_minor_version();
    d.multi_processor_count = p.get_max_compute_units();

    size_t free_b = 0, total_b = 0;
    /*
    DPCT1106:1130: 'cudaMemGetInfo' was migrated with the Intel extensions for
    device information which may not be supported by all compilers or runtimes.
    You may need to adjust the code.
    */
    check(DPCT_CHECK_ERROR(
              dpct::get_current_device().get_memory_info(free_b, total_b)),
          "cudaMemGetInfo");
    d.free_bytes = free_b;
    d.total_bytes = total_b;

    /*
    DPCT1043:1131: The version-related API is different in SYCL. An initial code
    was generated, but you need to adjust it.
    */
    check(DPCT_CHECK_ERROR(d.driver_version = dpct::get_major_version(
                               dpct::get_current_device())),
          "cudaDriverGetVersion");
    /*
    DPCT1043:1132: The version-related API is different in SYCL. An initial code
    was generated, but you need to adjust it.
    */
    check(DPCT_CHECK_ERROR(d.runtime_version = dpct::get_major_version(
                               dpct::get_current_device())),
          "cudaRuntimeGetVersion");

    // The kernels need sm_80 or newer (tf32 mma in the attention scorer, bf16 math) - the same floor the
    // arch guard in CMakeLists enforces at build time (RTX 30 / 40 / 50).  Anything older is caught here,
    // because a binary can be carried to a machine with an older card and would otherwise silently take
    // whatever path the driver chose.  Pre-Blackwell is untested by the author; trust, then verify.
    if (d.cc_major < 8) {
        throw CudaError("device " + d.name + " reports compute capability " + std::to_string(d.cc_major) +
                            "." + std::to_string(d.cc_minor) +
                            "; Strata needs an NVIDIA GPU of compute capability 8.0 or newer (RTX 30 / 40 / 50)",
                        -1);
    }
    return d;
}

DeviceArena::DeviceArena(uint64_t bytes, int ordinal, bool poison)
    : capacity_(bytes), ordinal_(ordinal), poison_(poison) {
    if (bytes == 0) throw CudaError("DeviceArena of 0 bytes", -1);
    /*
    DPCT1093:1133: The "ordinal" device may be not the one intended for use.
    Adjust the selected device if needed.
    */
    check(DPCT_CHECK_ERROR(dpct::select_device(ordinal)), "cudaSetDevice");
    // One allocation for the whole region.  cudaMalloc of a large block is the thing that can fail late, so it
    // happens once, here, before anything depends on it.
    check(DPCT_CHECK_ERROR(base_ = (void *)sycl::malloc_device(
                               (size_t)bytes, dpct::get_in_order_queue())),
          "cudaMalloc");
    if (poison_) {
        const int threads = 256;
        const uint64_t n = bytes / sizeof(float);
        const uint64_t blocks = (n + threads - 1) / threads;
        // gridDim.x is 32-bit, so a large region needs a loop.  12 GB of floats is 3e9 elements = 1.2e7
        // blocks, which fits, but the loop keeps it correct for any size rather than for today's sizes.
        const uint64_t max_blocks = 0x7FFFFFFFull;
        for (uint64_t b = 0; b < blocks; b += max_blocks) {
            const uint64_t chunk = (blocks - b < max_blocks) ? (blocks - b) : max_blocks;
            {
                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};

                dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
                    auto float_base__b_threads_ct0 =
                        (float *)base_ + b * threads;
                    auto n_b_threads_ct1 = n - b * threads;

                    cgh.parallel_for<
                        dpct_kernel_name<class poison_kernel_cca729>>(
                        sycl::nd_range<3>(sycl::range(1, 1, (unsigned)chunk) *
                                              sycl::range(1, 1, threads),
                                          sycl::range(1, 1, threads)),
                        exp_props, [=](sycl::nd_item<3> item_ct1) {
                            poison_kernel(float_base__b_threads_ct0,
                                          n_b_threads_ct1);
                        });
                });
            }
            /*
            DPCT1010:1134: SYCL uses exceptions to report errors and does not
            use the error codes. The cudaGetLastError function call was replaced
            with 0. You need to rewrite this code.
            */
            check(0, "poison_kernel");
        }
        check(DPCT_CHECK_ERROR(
                  dpct::get_current_device().queues_wait_and_throw()),
              "poison sync");
    }
}

DeviceArena::~DeviceArena() {
    if (base_) sycl::free(
        base_,
        dpct::get_in_order_queue()); // best effort: a destructor must not throw
}

void* DeviceArena::alloc(uint64_t bytes, uint64_t align) {
    if (bytes == 0) return nullptr;
    if (align == 0 || (align & (align - 1)) != 0) {
        throw CudaError("DeviceArena::alloc alignment must be a power of two", -1);
    }
    const uint64_t start = (used_ + align - 1) & ~(align - 1);
    if (start + bytes > capacity_) {
        char msg[256];
        std::snprintf(msg, sizeof(msg),
                      "DeviceArena out of memory: asked for %llu B at offset %llu (align %llu) in a %llu B "
                      "region - the plan from P1.S9 did not close",
                      (unsigned long long) bytes, (unsigned long long) start, (unsigned long long) align,
                      (unsigned long long) capacity_);
        throw CudaError(msg, -1);
    }
    used_ = start + bytes;
    return (char*) base_ + start;
}

}  // namespace strata::core
