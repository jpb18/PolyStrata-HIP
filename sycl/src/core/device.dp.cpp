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
    DPCT1000: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (e != 0) {
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1001: The statement could not be removed.
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

#if defined(STRATA_USE_HIP)
#if !defined(STRATA_HIP_ARCHS)
#error "STRATA_HIP_ARCHS (the compiled HIP architectures) is set by cmake/hip_backend.cmake"
#endif
// "gfx1201:sramecc-:xnack-" -> "gfx1201"
std::string base_arch(const char* gcn_arch_name) {
    std::string arch(gcn_arch_name);
    const size_t colon = arch.find(':');
    if (colon != std::string::npos) arch.resize(colon);
    return arch;
}

bool compiled_for(const std::string& arch) {
    const std::string list = STRATA_HIP_ARCHS;
    size_t a = 0;
    while (a <= list.size()) {
        size_t b = list.find(',', a);
        if (b == std::string::npos) b = list.size();
        if (!arch.empty() && list.compare(a, b - a, arch) == 0 && b - a == arch.size()) return true;
        a = b + 1;
    }
    return false;
}

std::string arch_problem(const cudaDeviceProp& p, int ordinal) {
    const std::string arch = base_arch(p.gcnArchName);
    const std::string card = "GPU " + std::to_string(ordinal) + " (" + p.name + ", " + arch + ")";
    if (!compiled_for(arch)) {
        return card + " is not an architecture this Strata engine was compiled for (" + STRATA_HIP_ARCHS +
               "); compile it for this card (./setup.sh --backend hip, or -DCMAKE_HIP_ARCHITECTURES=" + arch +
               ", docs/AMD_HIP.md) or choose another GPU with HIP_VISIBLE_DEVICES";
    }
    if (p.warpSize != 32) {
        return card + " runs wave" + std::to_string(p.warpSize) + "; Strata's HIP kernels need wave32";
    }
    return "";
}
#endif

}  // namespace

const char* compiled_gpu_archs() {
#if defined(STRATA_USE_HIP)
    return STRATA_HIP_ARCHS;
#else
    return "";
#endif
}

std::string gpu_arch_problem(int ordinal) {
#if defined(STRATA_USE_HIP)
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || ordinal < 0 || ordinal >= count) {
        cudaGetLastError();
        return "";
    }
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, ordinal) != cudaSuccess) {
        cudaGetLastError();
        return "";
    }
    return arch_problem(p, ordinal);
#else
    (void) ordinal;
    return "";
#endif
}

DeviceInfo device_info(int ordinal) {
    int count = 0;
    check(DPCT_CHECK_ERROR(count = dpct::device_count()), "cudaGetDeviceCount");
    if (count == 0) {
#if defined(STRATA_USE_HIP)
        throw CudaError(std::string("no HIP device is present; this engine was compiled for ") + STRATA_HIP_ARCHS, -1);
#else
        throw CudaError("no CUDA device is present; Strata needs an NVIDIA GPU (RTX 20 series or newer)", -1);
#endif
    }
    if (ordinal < 0 || ordinal >= count) {
        throw CudaError("device ordinal " + std::to_string(ordinal) + " is out of range (have " +
                            std::to_string(count) + ")",
                        -1);
    }
    DeviceInfo d;
    d.ordinal = ordinal;
    /*
    DPCT1093: The "ordinal" device may be not the one intended for use.
    Adjust the selected device if needed.
    */
    check(DPCT_CHECK_ERROR(dpct::select_device(ordinal)), "cudaSetDevice");

    dpct::device_info p{};
    check(DPCT_CHECK_ERROR(dpct::get_device(ordinal).get_device_info(p)),
          "cudaGetDeviceProperties");
    d.name = p.get_name();
    /*
    DPCT1005: The SYCL device version is different from CUDA Compute
    Compatibility. You may need to rewrite this code.
    */
    d.cc_major = p.get_major_version();
    /*
    DPCT1005: The SYCL device version is different from CUDA Compute
    Compatibility. You may need to rewrite this code.
    */
    d.cc_minor = p.get_minor_version();
    d.multi_processor_count = p.get_max_compute_units();

    size_t free_b = 0, total_b = 0;
    /*
    DPCT1106: 'cudaMemGetInfo' was migrated with the Intel extensions for
    device information which may not be supported by all compilers or runtimes.
    You may need to adjust the code.
    */
    check(DPCT_CHECK_ERROR(
              dpct::get_current_device().get_memory_info(free_b, total_b)),
          "cudaMemGetInfo");
    d.free_bytes = free_b;
    d.total_bytes = total_b;

    /*
    DPCT1043: The version-related API is different in SYCL. An initial code
    was generated, but you need to adjust it.
    */
    check(DPCT_CHECK_ERROR(d.driver_version = dpct::get_major_version(
                               dpct::get_current_device())),
          "cudaDriverGetVersion");
    /*
    DPCT1043: The version-related API is different in SYCL. An initial code
    was generated, but you need to adjust it.
    */
    check(DPCT_CHECK_ERROR(d.runtime_version = dpct::get_major_version(
                               dpct::get_current_device())),
          "cudaRuntimeGetVersion");

    // The engine supports compute capability 7.5 and newer (Turing: the QSA scorer's tf32 mma has a portable
    // fp32-FMA fallback below sm_80, the tensor-core prompt kernels refuse and fall back).  Compiling for a
    // supported arch is enforced by CMake; RUNNING on an older card is caught here, because a binary can be carried
    // to a machine with an older card and would otherwise silently take whatever path the driver chose.  The HIP
    // backend checks the card against the architectures the binary was compiled for (and wave32).
#if defined(STRATA_USE_HIP)
    d.arch = base_arch(p.gcnArchName);
    if (const std::string why = arch_problem(p, ordinal); !why.empty()) throw CudaError(why, -1);
#else
    // #236: the experimental build (-DSTRATA_EXPERIMENTAL_SM60=ON: Pascal sm_60, Volta sm_70) runs on the cards it
    // was built for - refusing them below 7.5 there made the flag useless; the release engine keeps 7.5
#if defined(STRATA_EXPERIMENTAL_SM60)
    constexpr int kMinCc = 60;
    const char* const kNeed = "6.0 or newer (this is the experimental Pascal / Volta build)";
#else
    constexpr int kMinCc = 75;
    const char* const kNeed = "7.5 or newer (RTX 20 / 30 / 40 / 50 series)";
#endif
    if (d.cc_major * 10 + d.cc_minor < kMinCc) {
        throw CudaError("device " + d.name + " reports compute capability " + std::to_string(d.cc_major) +
                            "." + std::to_string(d.cc_minor) + "; Strata needs compute capability " + kNeed,
                        -1);
    }
#endif
    return d;
}

DeviceArena::DeviceArena(uint64_t bytes, int ordinal, bool poison)
    : capacity_(bytes), ordinal_(ordinal), poison_(poison) {
    if (bytes == 0) throw CudaError("DeviceArena of 0 bytes", -1);
    /*
    DPCT1093: The "ordinal" device may be not the one intended for use.
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
                        dpct_kernel_name<class poison_kernel_58fc0a>>(
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
            DPCT1010: SYCL uses exceptions to report errors and does not
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
