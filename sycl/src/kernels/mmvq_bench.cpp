// sycl/src/kernels/mmvq_bench.cpp - SYCL port: time the dense native_mmvq kernels on random weights.
//   mmvq_bench [n_in] [n_out] [ncols]      (weights are random bytes: dequant cost is data-independent)
#include "strata/kernels/native_mmvq.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc, char** argv) {
    const int n_in = argc > 1 ? std::atoi(argv[1]) : 2560, n_out = argc > 2 ? std::atoi(argv[2]) : 2560, ncols = argc > 3 ? std::atoi(argv[3]) : 4;
    sycl::queue* s = &dpct::get_in_order_queue();
    // Q6_K: 210 bytes per 256 weights
    const size_t wbytes = (size_t) n_out * (n_in / 256) * 210;
    std::vector<uint8_t> hw(wbytes);
    for (size_t i = 0; i < wbytes; ++i) hw[i] = (uint8_t) (i * 2654435761u >> 13);
    std::vector<float> hx((size_t) ncols * n_in);
    for (size_t i = 0; i < hx.size(); ++i) hx[i] = (float) ((i * 7919) % 1000) / 500.f - 1.f;
    void* w = sycl::malloc_device(wbytes, *s);
    float* x = sycl::malloc_device<float>(hx.size(), *s);
    void* xq = sycl::malloc_device((size_t) ncols * (n_in / 32) * 36, *s);
    float* y = sycl::malloc_device<float>((size_t) ncols * n_out, *s);
    s->memcpy(w, hw.data(), wbytes).wait(); s->memcpy(x, hx.data(), hx.size() * 4).wait();
    strata::kernels::native_quantize_q8_1(x, xq, n_in, ncols, s);
    {   // warm up until the clocks are up (~300 ms of work), then time
        const auto w0 = std::chrono::steady_clock::now();
        while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count() < 300) {
            for (int i = 0; i < 20; ++i) strata::kernels::native_q6_k_mmvq(w, xq, y, n_in, n_out, ncols, s);
            s->wait();
        }
    }
    const int it = 400;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < it; ++i) strata::kernels::native_q6_k_mmvq(w, xq, y, n_in, n_out, ncols, s);
    s->wait();
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / it;
    std::printf("q6_k mmvq %d x %d, %d cols: %.1f us  (%.1f GB/s of weights)\n", n_in, n_out, ncols, us, wbytes / us / 1e3);
    return 0;
}
