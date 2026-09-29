// src/kernels/cuda/router_top10.cu - P2.S2: the MoE router.
//
// P2.S2's spec: "BF16 GEMV, softmax / top-k / renormalize per docs/semantics.md; emits (expert_id, weight) x 10
// per token".  This is the softmax/top-k/renormalize half; the BF16 GEMV that produces the logits is a
// separate kernel and is NOT here.
//
// The semantics are transcribed from `ref/moe.py::router`, which is itself transcribed from llama-graph.cpp:
//
//     p   = softmax(logits)                       over ALL experts, not over the selected subset
//     ids = stable argsort(-p)[:k]                ties break by INDEX, ascending
//     w   = gather(p, ids)
//     s   = max(sum(w), 2**-14)                   ggml_clamp(..., 2**-14, INF)
//     return ids, w / s
//
// The two-step structure matters and is not cosmetic: computing softmax over the selected subset instead is
// mathematically identical (softmax is shift-invariant) but it moves the renormalisation, and the CLAMP is
// part of the renormalisation.  Reproducing the gather form is what makes the clamp land in the same place.
//
// WHY THIS IS NO LONGER "NAIVE BY DESIGN", AND WHAT THE OLD REASONING GOT WRONG.
//
// The first version was ONE THREAD PER TOKEN: k passes over the experts, each pass recomputing
// `exp((double) l[e] - mx)` for every expert.  Its comment justified that with
//
//     "5,120 operations per token against 2.36e9 weights of expert matvec - three orders of magnitude smaller
//      than the thing it feeds, so the simplest correct version is also fast enough"
//
// **That is a statement about OPERATION COUNT and it says nothing about TIME, because all 5,120 operations were
// on ONE THREAD while the matvec has thousands.**  Measured (round 218, stage-by-stage inside one block):
//
//     moe_layer   4.077 ms      router_top10   3.387 ms      bf16 gemv   0.270 ms
//
// 3.39 ms per layer x 48 layers = 163 ms of a 289 ms token - **56% of the whole forward pass, in a kernel that
// reads 2.6 MB.**  It was invisible in `router_top10_parity` because that test runs the kernel ONCE and checks
// the right ids, and invisible in every per-layer test for the same reason.
//
// WHAT IS PARALLELISED AND WHAT IS DELIBERATELY NOT:
//
//   * the MAX is a tree reduction of `fmaxf`, which is exact and order-independent - bit-identical to the
//     serial scan it replaces;
//   * the 512 EXPONENTIALS are computed ONCE each, in parallel.  The old kernel computed them ELEVEN TIMES
//     (once for the softmax, then again inside every one of the k passes).  This is where the 3.4 ms was;
//   * the DOUBLE SUM is still accumulated on ONE thread in ASCENDING order.  512 double adds is about a
//     microsecond and it keeps the accumulation order - and therefore the last bits - identical to the
//     reference-faithful serial version.  Parallelising it would be a free speedup and a silent change to the
//     values, and the exp was the cost, not this;
//   * the SELECTION is by RANK rather than by k repeated maxima:
//
//         rank(e) = #{ f : p[f] > p[e] }  +  #{ f < e : p[f] == p[e] }
//
//     which IS "stable descending argsort, ties by ascending index" - the same order the repeated-maximum loop
//     produced - computed in ONE parallel pass over the experts instead of k serial ones.  It is O(n^2)
//     comparisons and that is the right trade here: n = 512, every comparison is independent, and the old
//     version was O(k*n) with a serial `exp` inside.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/router_top10.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int RT_MAX_THREADS = 512;

/// One BLOCK per token, so the reductions have somewhere to happen.  `n_tokens` is 1 in decode; the grid keeps
/// the batch case working without a second code path.
/*
DPCT1110:101: The total declared local variable size in device function
router_top10_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void router_top10_kernel(const float *__restrict__ logits,
                                         int n_tokens, int n_expert, int k,
                                         int *__restrict__ ids,
                                         float *__restrict__ weights,
                                         uint8_t *dpct_local) {
    // **`s_taken` IS FIRST SO THE OTHER TWO KEEP THEIR ALIGNMENT WITHOUT AN OFFSET PARAMETER**, and `s_p`'s
    // existing `s_ex + n_expert` stays correct because it is relative to `s_ex`.  One byte per expert.
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto s_raw = (unsigned char *)dpct_local;
    const size_t taken_bytes = ((size_t) n_expert + 15u) & ~(size_t) 15u;
    unsigned char* s_taken = s_raw;
    double* s_ex = (double*) (s_raw + taken_bytes);
    auto &s_red = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[RT_MAX_THREADS / 32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_rid = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        int[RT_MAX_THREADS / 32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_sum = *sycl::ext::oneapi::group_local_memory_for_overwrite<double>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());

    const int t = item_ct1.get_group(2);
    if (t >= n_tokens) return;
    const int tid = item_ct1.get_local_id(2);
    const int nt = item_ct1.get_local_range(2);
    const float* l = logits + (size_t) t * n_expert;

    // ---- softmax over ALL experts, for stability: the max.  A tree of `fmaxf` is EXACT and order-independent,
    // so this is bit-identical to the serial scan.
    float mx = -INFINITY;
#pragma unroll
    for (int e = tid; e < n_expert; e += nt) mx = sycl::fmax(mx, l[e]);
    /*
DPCT1108:102: '__shfl_down_sync' was migrated with the experimental feature
masked sub_group function which may not be supported by all compilers or
runtimes. You may need to adjust the code.
*/
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) mx = sycl::fmax(
        mx, dpct::experimental::shift_sub_group_left(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                mx, off));
    if ((tid & 31) == 0) s_red[tid >> 5] = mx;
    /*
    DPCT1065:526: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (tid < 32) {
        const int nw = (nt + 31) >> 5;
        float v = (tid < nw) ? s_red[tid] : -INFINITY;
        /*
DPCT1108:103: '__shfl_down_sync' was migrated with the experimental feature
masked sub_group function which may not be supported by all compilers or
runtimes. You may need to adjust the code.
*/
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v = sycl::fmax(
            v, dpct::experimental::shift_sub_group_left(
                   0xffffffffu,
                   sycl::ext::oneapi::this_work_item::get_sub_group(), v, off));
        if (tid == 0) s_red[0] = v;
    }
    /*
    DPCT1065:527: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    mx = s_red[0];

    // ---- THE 512 EXPONENTIALS, ONCE EACH AND IN PARALLEL.  `exp` in double is software-emulated on this die
    // and was 5,632 serial calls before; it is 512 parallel ones now.
#pragma unroll
    for (int e = tid; e < n_expert; e += nt)
        s_ex[e] = sycl::exp((double)l[e] - (double)mx);
    /*
    DPCT1065:528: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // ---- the sum, ascending, on one thread: see the note above on why this is NOT parallelised.
    if (tid == 0) {
        double sum = 0.0;
#pragma unroll
        for (int e = 0; e < n_expert; ++e) sum += s_ex[e];
        s_sum = sum;
    }
    /*
    DPCT1065:529: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const float inv = (float) (1.0 / s_sum);

    // ---- the selection, by rank.  `p[e] = (float)(s_ex[e] * inv)` reproduces the old expression exactly:
    // `(float)(exp((double) l[e] - (double) mx) * inv)` with `inv` a FLOAT.
    // ---- p[] ONCE.  The rank loop below compares every expert against every other, so computing
    // `(float) (s_ex[f] * inv)` INSIDE it did **512 x 512 = 262,144** double multiplies and float converts per
    // token per layer instead of 512 - and a consumer die runs FP64 at a small fraction of FP32, which is why
    // this was worth more than the branch.  The shared load stays either way; the ARITHMETIC is what goes.
    //
    // THIS IS BIT-IDENTICAL, not merely equivalent: `p[e]` is the same expression it was, computed once instead
    // of 513 times, and the comparisons then see exactly the same floats.
    float* s_p = (float*) (s_ex + n_expert);
#pragma unroll
    for (int e = tid; e < n_expert; e += nt) s_p[e] = (float)(s_ex[e] * inv);
    /*
    DPCT1065:530: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // ================================ THE SELECTION, IN k PASSES ================================
    //
    // **THE RANK-BY-COUNTING LOOP COMPARED EVERY EXPERT AGAINST EVERY OTHER**: 512 x 512 = 262,144 shared
    // loads and compares per token per layer, to choose ten things out of 512 - measured at 34.35 us, 63% of
    // `moe_route`'s accounted cost and the largest single item in the R3 plan.  The comment above records that
    // the FP64 ARITHMETIC inside that loop was hoisted out once already ("262,144 double multiplies per token
    // per layer instead of 512"); **the O(n^2) STRUCTURE was left, and it is the structure that costs.**
    //
    // `k` passes of a block-wide argmax is 10 x 512 = 5,120 compares - **51x less work** - and it produces the
    // SAME ORDER.  That is the whole correctness argument: scanning `e` ascending with a strict `>` keeps the
    // LOWEST index on a tie, which is exactly the rule the rank loop spelled out as
    // `else if (f < e && pf == pe) ++rank`.  A stable descending top-k, ties by index.
#pragma unroll
    for (int e = tid; e < n_expert; e += nt) s_taken[e] = 0;
    /*
    DPCT1065:531: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    for (int i = 0; i < k; ++i) {
        float bv = -INFINITY;
        int bi = n_expert;               // a sentinel that loses to every real index
        for (int e = tid; e < n_expert; e += nt) {
            if (s_taken[e]) continue;
            const float pe = s_p[e];
            if (pe > bv) { bv = pe; bi = e; }
        }
        for (int off = 16; off > 0; off >>= 1) {
            /*
            DPCT1108:106: '__shfl_down_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const float ov = dpct::experimental::shift_sub_group_left(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                bv, off);
            /*
            DPCT1108:107: '__shfl_down_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const int oi = dpct::experimental::shift_sub_group_left(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if ((tid & 31) == 0) { s_red[tid >> 5] = bv; s_rid[tid >> 5] = bi; }
        /*
        DPCT1118:104: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:533: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (tid < 32) {
            const int nw = (nt + 31) >> 5;
            float v = (tid < nw) ? s_red[tid] : -INFINITY;
            int ix = (tid < nw) ? s_rid[tid] : n_expert;
            for (int off = 16; off > 0; off >>= 1) {
                /*
                DPCT1108:108: '__shfl_down_sync' was migrated with the
                experimental feature masked sub_group function which may not be
                supported by all compilers or runtimes. You may need to adjust
                the code.
                */
                const float ov = dpct::experimental::shift_sub_group_left(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), v, off);
                /*
                DPCT1108:109: '__shfl_down_sync' was migrated with the
                experimental feature masked sub_group function which may not be
                supported by all compilers or runtimes. You may need to adjust
                the code.
                */
                const int oi = dpct::experimental::shift_sub_group_left(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), ix,
                    off);
                if (ov > v || (ov == v && oi < ix)) { v = ov; ix = oi; }
            }
            if (tid == 0 && ix < n_expert) {
                ids[(size_t) t * k + i] = ix;
                weights[(size_t) t * k + i] = v;
                s_taken[ix] = 1;
            }
        }
        /*
        DPCT1118:105: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:534: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
    /*
    DPCT1065:532: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // ---- renormalise, with ggml's lower clamp.  Order preserved.
    if (tid == 0) {
        double s = 0.0;
#pragma unroll
        for (int i = 0; i < k; ++i) s += (double)weights[(size_t)t * k + i];
        const double sc = sycl::fmax(s, 6.103515625e-05); // 2**-14
#pragma unroll
        for (int i = 0; i < k; ++i)
            weights[(size_t) t * k + i] = (float) ((double) weights[(size_t) t * k + i] / sc);
    }
}

}  // namespace

void router_top10(const float *logits, int n_tokens, int n_expert, int k,
                  int *ids, float *weights, void *stream) try {
    if (n_tokens <= 0 || n_expert <= 0 || k <= 0) return;
    if (k > 64) {
        std::fprintf(stderr, "router_top10: k %d exceeds the kernel's 64\n", k);
        std::exit(1);
    }
    if (n_expert > RT_MAX_THREADS * 64) {
        std::fprintf(stderr, "router_top10: n_expert %d is past the kernel's %d\n", n_expert,
                     RT_MAX_THREADS * 64);
        std::exit(1);
    }
    int threads = n_expert < RT_MAX_THREADS ? n_expert : RT_MAX_THREADS;
    threads = (threads + 31) & ~31;                  // at least one full warp, for the reductions
    // the selection's taken-mask, then n_expert doubles for the exponentials, then n_expert floats for the
    // probabilities.  The mask is first so the two aligned arrays need no offset parameter.
    const size_t taken_bytes = ((size_t) n_expert + 15u) & ~(size_t) 15u;
    const size_t smem =
        /*
        DPCT1083:111: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        taken_bytes + (size_t)n_expert * sizeof(double) +
        (size_t)n_expert * sizeof(float);
    /*
    DPCT1049:110: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(
            strata::q_of(stream)->get_device(),
            {sycl::aspect::fp64});

        strata::q_of(stream)
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range(smem), cgh);

                cgh.parallel_for<
                    dpct_kernel_name<class router_top10_kernel_3e0655>>(
                    sycl::nd_range<3>(sycl::range(1, 1, (unsigned)n_tokens) *
                                          sycl::range(1, 1, threads),
                                      sycl::range(1, 1, threads)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        32)]] {
                        router_top10_kernel(
                            logits, n_tokens, n_expert, k, ids, weights,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
    }
    /*
    DPCT1010:535: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;

    if (stream == nullptr) {
        const dpct::err0 s = DPCT_CHECK_ERROR(
            dpct::get_current_device().queues_wait_and_throw());
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

}  // namespace strata::kernels
