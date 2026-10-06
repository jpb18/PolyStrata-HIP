// include/strata/sycl_queue.hpp - the SYCL port's one addition to the engine's API surface.
//
// Every launcher takes `void* stream`, a cudaStream_t where null means the default stream. dpct migrates the
// cast to `(dpct::queue_ptr) stream` and dereferences it, so a null stream is a null sycl::queue* and a crash.
// q_of() is that cast with CUDA's null-stream meaning restored: the default in-order queue.
#pragma once
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

namespace strata {
inline sycl::queue* q_of(const void* stream) {
    return stream ? (sycl::queue*) stream : &dpct::get_in_order_queue();
}
}  // namespace strata

namespace strata {
// A large device fill as compute kernels of at most `chunk` bytes, each waited for.  On the Arc Pro B70 (xe driver) one
// queue.memset of many GiB runs on the blitter engine and times out ("Engine memory CAT error", GT reset); kernels do not.
// Small fills (<= chunk) and non-multiple-of-8 tails fall back to memset.
inline void big_fill_zero(sycl::queue& q, void* p, size_t bytes, size_t chunk = (size_t)256 << 20) {
    uint8_t* b = (uint8_t*)p;
    size_t off = 0;
    while (off < bytes) {
        size_t n = bytes - off < chunk ? bytes - off : chunk;
        if (n >= (1u << 20) && ((uintptr_t)(b + off) & 7) == 0) {
            size_t w = n / 8;
            uint64_t* d = (uint64_t*)(b + off);
            q.parallel_for(sycl::range<1>(w), [=](sycl::id<1> i) { d[i] = 0; });
            if (n & 7) q.memset(b + off + w * 8, 0, n & 7);
        } else {
            q.memset(b + off, 0, n);
        }
        q.wait();
        off += n;
    }
}
}  // namespace strata
