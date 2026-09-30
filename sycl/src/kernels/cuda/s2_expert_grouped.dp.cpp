// src/kernels/cuda/s2_expert_grouped.cu - R4's grouped GPU expert.  Read the header first.
//
// THE ARITHMETIC IS `src/kernels/cpu/expert.cpp`'s, over the same bytes, one warp per output row.
//
//     sum_j (c_j - 1) * d_w * xhat_j * d_x   ==   d_w * d_x * ( sum_j c_j*xhat_j  -  sum_j xhat_j )
//
// Both sums are INT8 x INT8, so both are `__dp4a` - four multiply-accumulates per instruction, exact in
// integer and with no dequantization inside the loop.  The weight scale `d_w` is fp16 (one per 64 elements)
// and the activation scale `d_x` is fp16 (one per 32 elements), so the float work is one multiply-add per
// 32-element chunk rather than one per element.
//
// **THE SUMMATION ORDER IS NOT THE CPU's AND CANNOT BE.**  Lane `L` takes chunks `L, L+32, ...` and the
// partials are reduced through shuffle; the CPU walks every chunk in order with its own accumulator shape.
// The two agree to float rounding and not to the bit, which is the same contract `s_gemv_parity` carries for
// the same reason.  `bench/micro/moe_hit_parity.cu` is the check.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_math.hpp"
#include "strata/sycl_queue.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#include "strata/kernels/quantize_act.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {
namespace {

// THE BLOB'S OWN GEOMETRY, from `include/strata/kernels/cpu/expert.hpp`.  Restated as literals because that
// header is the CPU path's and this file must not silently follow it if the two ever disagree: the sizes below
// are what the CPU kernel's indexing computes, and `moe_hit_parity` compares the two end to end.
constexpr int H = 2560;
constexpr int FF = 640;
constexpr int QK = 64;                       // Q2_0's group: one fp16 scale per 64 weights
constexpr int ROW_GU = H / 4;                // 640 B of codes per gate/up row (2 bits per element)
constexpr int ROW_D = FF / 4;                // 160 B per down row
constexpr int SC_GU = H / QK;                // 40 fp16 scales per gate/up row
constexpr int SC_D = FF / QK;                // 10 per down row
constexpr size_t O_D_CODES = (size_t) 2 * FF * ROW_GU;
constexpr size_t O_GU_SCALES = O_D_CODES + (size_t) H * ROW_D;
constexpr size_t O_D_SCALES = O_GU_SCALES + (size_t) 2 * FF * SC_GU * 2;

__dpct_inline__ float f16_at(const uint8_t *p) {
    return sycl::vec<sycl::half, 1>(sycl::bit_cast<sycl::half, unsigned short>(
                                        (uint16_t)(p[0] | (p[1] << 8))))
        .convert<float, sycl::rounding_mode::automatic>()[0];
}

/// **ONE S2 ROW AGAINST A Q8_0 ACTIVATION, WARP-WIDE.**  Every lane accumulates its own float partial over a
/// strided set of 32-element chunks and the caller reduces; `chunk` indices are absolute so the caller can
/// start the lane at any offset.
///
/// Returns the lane's partial.  `codes` is `n_in/4` bytes and `scales` `n_in/QK` fp16, both for THIS row.
///
/// **`x_scales` IS R4.2h AND IT IS OPTIONAL ON PURPOSE.**  When it is non-null the activation's multiplier
/// comes from an fp32 array instead of the block's fp16 `d`.  The CPU pool multiplies by the fp32
/// `ActQ::scale` (`cpu/expert.cpp:92`), and the two disagreed by **4.761e-04 relative on 80 of 80 chunks**
/// (`bench/micro/act_quant_parity.cu`), which is what made a cache hit compute a different expert from a
/// cache miss.  Null keeps the previous fp16 behaviour **exactly**, so `moe_hit_parity` - which passes no
/// scales - still measures the kernel it always measured.
__dpct_inline__ float
row_dot_s2_q8(const uint8_t *__restrict__ codes,
              const uint8_t *__restrict__ scales,
              const uint8_t *__restrict__ x_q8_0, int n_chunks, int lane,
              const float *__restrict__ x_scales = nullptr) {
    float acc = 0.0f;
    for (int c = lane; c < n_chunks; c += 32) {
        const uint8_t* cb = codes + (size_t) c * 8;             // 8 code bytes = 32 elements
        const uint8_t* xb = x_q8_0 + (size_t) c * 34;           // one block_q8_0
        const float dx = x_scales ? x_scales[c] : f16_at(xb);
        const int8_t* xq = (const int8_t*) (xb + 2);

        // ---- THE CODES, EXPANDED TO ONE BYTE PER ELEMENT, THEN FOUR AT A TIME INTO `dp4a`.
        //
        // The packed form is LSB-first: element 4j+k is bits [2k, 2k+2) of code byte j.  `dp4a` needs both
        // operands as packed int8, so each code byte becomes a word whose four bytes are its four 2-bit
        // fields - which is exactly `(c & 3) | ((c>>2)&3)<<8 | ((c>>4)&3)<<16 | ((c>>6)&3)<<24`.
        int s = 0;      // sum of code * x
        int hx = 0;     // sum of x        - the weight-independent term, as ones * x
        const int ones = 0x01010101;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const unsigned cbyte = cb[j];
            const int cw = (int) ((cbyte & 3u) | (((cbyte >> 2) & 3u) << 8) | (((cbyte >> 4) & 3u) << 16) |
                                  (((cbyte >> 6) & 3u) << 24));
            // **`memcpy`, NOT A CAST.**  A `block_q8_0` is 34 BYTES - two of header then 32 of int8 - so the
            // activation data at offset 2 is never 4-byte aligned, and `*(const int*)(xq + 4*j)` faults with
            // "misaligned address".  It faulted exactly that way on this kernel's first run.  `memcpy` of a
            // known 4 bytes compiles to whatever load is legal for the alignment, which is the point of using
            // it rather than reasoning about which cast happens to work.
            int xw;
            memcpy(&xw, xq + 4 * j, 4);
            s = strata::dp4a(cw, xw, s);
            hx = strata::dp4a(ones, xw, hx);
        }
        // One weight scale per 64 elements, so per TWO 32-element chunks.
        const float dw = f16_at(scales + (size_t) (c >> 1) * 2);
        acc += dw * dx * (float) (s - hx);
    }
    return acc;
}

__dpct_inline__ float warp_sum(float v) {
#pragma unroll
    /*
    DPCT1108: '__shfl_down_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    /*
    DPCT1121: Make sure that the "v" which is used in the SYCL group
    function/algorithm is initialized.
    */
    for (int off = 16; off > 0; off >>= 1) v +=
        dpct::experimental::shift_sub_group_left(
            0xFFFFFFFFu, sycl::ext::oneapi::this_work_item::get_sub_group(), v,
            off);
    return v;
}

/// GATE AND UP, ONE WARP PER ROW.
///
/// **THE ROWS ARE INTERLEAVED AND THE FIRST VERSION OF THIS GOT IT WRONG.**  In the blob, row-SLOT `i` of the
/// `2*FF` gate/up rows is gate row `i/2` when `i` is even and up row `(i-1)/2` when it is odd - that is what
/// `O_GU_CODES + (2r)*ROW_GU` and `+ (2r+1)*ROW_GU` mean in `expert.cpp`.  This kernel decoded the correct
/// code row for slot `i` and then wrote it to output slot `i` of a layout whose first `FF` entries are gate and
/// whose last `FF` are up, which pairs `silu(gate[r]) * up[r]` with the WRONG `up` for every `r`.  It produced
/// finite, plausible numbers.  `moe_hit_parity` caught it on the first run, at worst relative error 2.2e+03.
///
/// So slot `i` is DECODED from row-slot `i` and WRITTEN to the output slot its parity says it belongs to.
__dpct_inline__ void
gu_kernel(const uint8_t *__restrict__ blob_base,
          const int32_t *__restrict__ slot_index, long long blob_bytes,
          const uint8_t *__restrict__ x_q8_0,
          const float *__restrict__ x_scales, float *__restrict__ gate_up,
          int n_hits, const int32_t *__restrict__ d_count = nullptr,
          const int32_t *__restrict__ dst_index = nullptr, int tok_div = 0) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int warps_per_block = (int)(item_ct1.get_local_range(2) >> 5);
    const long long slot = (long long)item_ct1.get_group(2) * warps_per_block +
                           (item_ct1.get_local_id(2) >> 5);
    const long long rows_per_hit = 2LL * FF;
    const long long total = (long long) n_hits * rows_per_hit;
    if (slot >= total) return;
    const int h = (int) (slot / rows_per_hit);
    if (d_count != nullptr && h >= *d_count) return;     // token graph: capacity layout, device count
    const int i = (int) (slot % rows_per_hit);
    const int lane = item_ct1.get_local_id(2) & 31;

    const uint8_t* blob = blob_base + (size_t) slot_index[h] * (size_t) blob_bytes;
    if (tok_div > 0) {   // plan v0.3 P6 verify window: each hit reads its own token's activation
        const int tok = dst_index[h] / tok_div;
        x_q8_0 += (size_t) tok * (size_t) (H / 32) * 34;
        if (x_scales != nullptr) x_scales += (size_t) tok * (size_t) (H / 32);
    }
    const float acc = row_dot_s2_q8(blob + (size_t) i * ROW_GU,
                                    blob + O_GU_SCALES + (size_t) i * SC_GU * 2, x_q8_0, H / 32, lane, x_scales);
    const float s = warp_sum(acc);
    if (lane != 0) return;
    // ---- THE OUTPUT LAYOUT IS GATE-MAJOR, AND THAT IS NOT COSMETIC.
    //
    // The first version wrote `gate_up[h*2FF + {0..FF-1}] = gate` and `[h*2FF + FF..] = up`, i.e. a per-hit
    // [gate | up] pair, and then called the shared `swiglu_kernel` and `quantize_q8_0` over the whole buffer.
    // Both of those walk a CONTIGUOUS range, so with more than one hit they read hit 0's UP rows where they
    // wanted hit 1's GATE rows - finite numbers, wrong expert.  With one hit it would have passed.
    //
    // Gate-major removes the mismatch instead of adding a stride to two other kernels: every hit's gate rows
    // are contiguous from 0, every hit's up rows are contiguous from `n_hits * FF`, and the swiglu's output
    // lands in the first `n_hits * FF` floats exactly where `quantize_q8_0` reads it.
    const int r = i >> 1;
    const size_t base = (i & 1) ? ((size_t) n_hits * FF + (size_t) h * FF) : ((size_t) h * FF);
    gate_up[base + (size_t) r] = s;
}

/// `silu(gate) * up`, in place, over a GATE-MAJOR buffer: `[0, n_pairs)` is every hit's gate and
/// `[n_pairs, 2*n_pairs)` is every hit's up, so hit `h`'s row `r` meets itself at `h*FF + r`.
///
/// SiLU on the GATE and multiplied by up - the reading `docs/semantics.md` records, and the one that is wrong
/// the other way round in a way that still produces a finite number.
__dpct_inline__ void swiglu_kernel(float *__restrict__ gate_up,
                                   long long n_pairs) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long i =
        (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n_pairs) return;
    const float g = gate_up[i];
    const float u = gate_up[n_pairs + i];
    gate_up[i] = (g / (1.0f + sycl::native::exp(-g))) * u;
}

/// DOWN, ONE WARP PER ROW, reading the quantized intermediate the caller produced.
///
/// `dst_index[h]` is which row of the shared output buffer hit `h` fills - see the header.  It is the router's
/// slot, not `h`, and the two differ on every layer where some experts are resident and some are not.
__dpct_inline__ void
down_kernel(const uint8_t *__restrict__ blob_base,
            const int32_t *__restrict__ slot_index,
            const int32_t *__restrict__ dst_index, long long blob_bytes,
            const uint8_t *__restrict__ h_q8_0,
            const float *__restrict__ h_scales, float *__restrict__ out,
            int n_hits, const int32_t *__restrict__ d_count = nullptr) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int warps_per_block = (int)(item_ct1.get_local_range(2) >> 5);
    const long long row = (long long)item_ct1.get_group(2) * warps_per_block +
                          (item_ct1.get_local_id(2) >> 5);
    const long long total = (long long) n_hits * H;
    if (row >= total) return;
    const int h = (int) (row / H);
    if (d_count != nullptr && h >= *d_count) return;
    const int r = (int) (row % H);
    const int lane = item_ct1.get_local_id(2) & 31;

    const uint8_t* blob = blob_base + (size_t) slot_index[h] * (size_t) blob_bytes;
    const uint8_t* xb = h_q8_0 + (size_t) h * (size_t) (FF / 32) * 34;
    const float acc = row_dot_s2_q8(blob + O_D_CODES + (size_t) r * ROW_D,
                                    blob + O_D_SCALES + (size_t) r * SC_D * 2, xb, FF / 32, lane,
                                    h_scales ? h_scales + (size_t) h * (size_t) (FF / 32) : nullptr);
    const float s = warp_sum(acc);
    if (lane == 0) out[(size_t) dst_index[h] * H + r] = s;
}

// The CPU subtracts the weight bias after its eight FMA accumulators have been reduced. Moving the
// subtraction into each integer dot, as the legacy kernel does, changes rounding even with equal scales.
__dpct_inline__ void activation_correction_kernel(const uint8_t *q8,
                                                  const float *scales,
                                                  float *hx, int chunks) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c >= chunks) return;
    const int8_t* q = (const int8_t*) (q8 + (size_t) c * 34 + 2);
    int sum = 0;
#pragma unroll
    for (int j = 0; j < 32; ++j) sum += q[j];
    /*
    DPCT1013: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    hx[c] = scales[c] * (float)sum;
}

__dpct_inline__ int dot4(const uint8_t *codes, const int8_t *q) {
    const unsigned c = *codes;
    const int cw = (int) ((c & 3u) | (((c >> 2) & 3u) << 8) |
                          (((c >> 4) & 3u) << 16) | (((c >> 6) & 3u) << 24));
    int xw;
    memcpy(&xw, q, sizeof xw);
    return strata::dp4a(cw, xw, 0);
}

__dpct_inline__ float row_dot_cpu_order(const uint8_t *codes,
                                        const uint8_t *scales,
                                        const uint8_t *xq, const float *xs,
                                        const float *hx, int blocks, int lane) {
    float acc = 0.0f;
    float corr = 0.0f;
    for (int b = 0; b < blocks; ++b) {
        const float d = f16_at(scales + 2 * b);
        const int lo = dot4(codes + b * 16 + lane,
                           (const int8_t*) (xq + (size_t) (2 * b) * 34 + 2) + lane * 4);
        const int hi = dot4(codes + b * 16 + 8 + lane,
                           (const int8_t*) (xq + (size_t) (2 * b + 1) * 34 + 2) + lane * 4);
        /*
        DPCT1013: The rounding mode could not be specified and the
        generated code may have different accuracy than the original code.
        Verify the correctness. SYCL math built-in function rounding mode is
        aligned with OpenCL C 1.2 standard.
        */
        acc = sycl::fma(d * xs[2 * b], (float)lo, acc);
        /*
        DPCT1013: The rounding mode could not be specified and the
        generated code may have different accuracy than the original code.
        Verify the correctness. SYCL math built-in function rounding mode is
        aligned with OpenCL C 1.2 standard.
        */
        acc = sycl::fma(d * xs[2 * b + 1], (float)hi, acc);
        if (lane == 0)
            /*
            DPCT1013: The rounding mode could not be specified and the
            generated code may have different accuracy than the original code.
            Verify the correctness. SYCL math built-in function rounding mode is
            aligned with OpenCL C 1.2 standard.
            */
            corr = corr + d * hx[2 * b] + hx[2 * b + 1];
    }
    // _mm_add_ps(low128, high128), then two _mm_hadd_ps. The pair order is 4, 1, 2;
    // a standard shuffle tree in the order 4, 2, 1 is a different floating-point expression.
    constexpr unsigned mask = 0xffffffffu;
    /*
    DPCT1108: '__shfl_down_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    /*
    DPCT1013: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    acc = acc + dpct::experimental::shift_sub_group_left(
                    mask, sycl::ext::oneapi::this_work_item::get_sub_group(),
                    acc, 4, 8);
    /*
    DPCT1108: '__shfl_down_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    /*
    DPCT1013: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    acc = acc + dpct::experimental::shift_sub_group_left(
                    mask, sycl::ext::oneapi::this_work_item::get_sub_group(),
                    acc, 1, 8);
    /*
    DPCT1108: '__shfl_down_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    /*
    DPCT1013: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    acc = acc + dpct::experimental::shift_sub_group_left(
                    mask, sycl::ext::oneapi::this_work_item::get_sub_group(),
                    acc, 2, 8);
    /*
    DPCT1013: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    return acc - corr; // Only lane zero is consumed.
}

template <bool DOWN>
__dpct_inline__ void
cpu_order_projection_kernel(const uint8_t *blob_base, const int32_t *slots,
                            const int32_t *destinations, long long blob_bytes,
                            const uint8_t *xq, const float *xs, const float *hx,
                            float *out, int n_hits) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int rows_per_hit = DOWN ? H : 2 * FF;
    const int row = item_ct1.get_group(2) * (item_ct1.get_local_range(2) / 8) +
                    item_ct1.get_local_id(2) / 8;
    if (row >= n_hits * rows_per_hit) return;
    const int h = row / rows_per_hit;
    const int r = row % rows_per_hit;
    const int lane = item_ct1.get_local_id(2) & 7;
    const uint8_t* blob = blob_base + (size_t) slots[h] * (size_t) blob_bytes;
    const int chunks_offset = DOWN ? h * (FF / 32) : 0;
    const uint8_t* codes = DOWN ? blob + O_D_CODES + (size_t) r * ROW_D : blob + (size_t) r * ROW_GU;
    const uint8_t* scales = DOWN ? blob + O_D_SCALES + (size_t) r * SC_D * 2
                                 : blob + O_GU_SCALES + (size_t) r * SC_GU * 2;
    const float value = row_dot_cpu_order(codes, scales, xq + (size_t) chunks_offset * 34,
                                           xs + chunks_offset, hx + chunks_offset,
                                           DOWN ? SC_D : SC_GU, lane);
    if (lane != 0) return;
    if (DOWN) out[(size_t) destinations[h] * H + r] = value;
    else out[((r & 1) ? (size_t) n_hits * FF : 0) + (size_t) h * FF + (r >> 1)] = value;
}

__dpct_inline__ void cpu_order_swiglu_kernel(float *gu, int pairs) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= pairs) return;
    const float g = gu[i];
    // Accurate fp32 exponential; __expf's approximation would introduce an additional source of error.
    // CPU/GPU libc last-bit differences are diagnosed separately by the micro, not hidden with FP64 here.
    const float eg = sycl::native::exp(-g);
    /*
    DPCT1013: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    gu[i] = g / (1.0f + eg) * gu[pairs + i];
}

__dpct_inline__ void cpu_order_quantize_kernel(const float *x, uint8_t *blocks,
                                               float *scales, float *hx,
                                               int chunks) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c >= chunks) return;
    const float* xb = x + c * 32;
    uint8_t* out = blocks + (size_t) c * 34;
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 32; ++j) amax = sycl::fmax(amax, sycl::fabs(xb[j]));
    /*
    DPCT1013: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    const float s = amax > 0.0f ? amax / 127.0f : 0.0f;
    /*
    DPCT1013: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    const float inv = s > 0.0f ? 1.0f / s : 0.0f;
    scales[c] = s;
    const uint16_t bits = sycl::bit_cast<unsigned short, sycl::half>(
        sycl::vec<float, 1>(s)
            .convert<sycl::half, sycl::rounding_mode::rte>()[0]);
    out[0] = (uint8_t) bits;
    out[1] = (uint8_t) (bits >> 8);
    int sum = 0;
    for (int j = 0; j < 32; ++j) {
        /*
        DPCT1013: The rounding mode could not be specified and the
        generated code may have different accuracy than the original code.
        Verify the correctness. SYCL math built-in function rounding mode is
        aligned with OpenCL C 1.2 standard.
        */
        const float t = xb[j] * inv;
        /*
        DPCT1013: The rounding mode could not be specified and the
        generated code may have different accuracy than the original code.
        Verify the correctness. SYCL math built-in function rounding mode is
        aligned with OpenCL C 1.2 standard.
        */
        int v = (int)t + t >= 0.0f ? 0.5f : -0.5f;
        v = v < -127 ? -127 : (v > 127 ? 127 : v);
        out[2 + j] = (uint8_t) (int8_t) v;
        sum += v;
    }
    /*
    DPCT1013: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    hx[c] = s * (float)sum;
}

constexpr int THREADS = 256;

void check(const char* who, void* stream) {
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;

    // Deliberately NOT synchronising for a non-null stream: this is called once per layer from a captured
    // graph's worth of work, and `finish()`'s null-stream sync in `s_gemv.cu` is the pattern that made a whole
    // round of measurements the driver's cost instead of the kernel's (Memory/ERRORS.md RC-7).
    (void) stream;
}

}  // namespace

uint64_t moe_hit_grouped_scratch_bytes(int64_t n_hits, int64_t n_embd, int64_t n_ff) {
    if (n_hits <= 0) return 0;
    const uint64_t gu = (uint64_t) n_hits * (uint64_t) (2 * n_ff) * 4;
    const uint64_t q8 = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 34;
    // R4.2h: the fp32 scales for the INTERMEDIATE's own quantization, one per 32-element chunk per hit.
    const uint64_t hs = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 4;
    const uint64_t xh = (uint64_t) (n_embd / 32) * 4;
    return ((gu + 15) & ~15ull) + ((q8 + 15) & ~15ull) + 2 * ((hs + 15) & ~15ull) +
           ((xh + 15) & ~15ull);
}

void moe_hit_grouped_s2(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                        int64_t n_hits, int64_t blob_bytes, const uint8_t* x_q8_0, void* scratch, float* out,
                        void* stream, const float* x_scales) {
    if (n_hits <= 0) return;
    dpct::queue_ptr cs = strata::q_of(stream);
    const int warps = THREADS / 32;

    const uint64_t gu_bytes = ((uint64_t) n_hits * (uint64_t) (2 * FF) * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) n_hits * (uint64_t) (FF / 32) * 34 + 15) & ~15ull;
    float* gate_up = (float*) scratch;
    uint8_t* h_q8_0 = (uint8_t*) scratch + gu_bytes;
    float* h_scales = (float*) ((uint8_t*) scratch + gu_bytes + q8_bytes);

    // 1. gate + up, one launch for every row of every hit.
    {
        const long long rows = n_hits * 2LL * FF;
        const unsigned blocks = (unsigned) ((rows + warps - 1) / warps);
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            cs->parallel_for<dpct_kernel_name<class gu_kernel_feb5f7>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks) *
                                      sycl::range(1, 1, THREADS),
                                  sycl::range(1, 1, THREADS)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gu_kernel(blob_base, slot_index, blob_bytes, x_q8_0,
                                  x_scales, gate_up, (int)n_hits, nullptr,
                                  nullptr, 0);
                    });
        }
        check("moe_hit_grouped_s2/gu", stream);
    }
    // 2. silu(gate) * up.
    {
        const long long pairs = n_hits * (long long) FF;
        const unsigned blocks = (unsigned) ((pairs + THREADS - 1) / THREADS);
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            cs->parallel_for<dpct_kernel_name<class swiglu_kernel_6bd648>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks) *
                                      sycl::range(1, 1, THREADS),
                                  sycl::range(1, 1, THREADS)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    swiglu_kernel(gate_up, pairs);
                });
        }
        check("moe_hit_grouped_s2/swiglu", stream);
    }
    // 3. the intermediate's own contract, which is `ggml_mul_mat`'s rule and NOT a choice: the down weight is
    //    Q2_0, whose `vec_dot_type` is Q8_0.  `gate_up` holds the pairs; the product went into the first half.
    //    **R4.2h: the CPU quantizes this intermediate into `a2` with fp32 scales too** (`expert.cpp:232`), so
    //    when the caller supplies `x_scales` the intermediate gets the CPU's contract as well - otherwise the
    //    down projection would keep the very disagreement the gate/up projection just had removed.
    if (x_scales != nullptr) quantize_q8_0_scaled(gate_up, h_q8_0, h_scales, n_hits * (int64_t) FF, stream);
    else quantize_q8_0(gate_up, h_q8_0, n_hits * (int64_t) FF, stream);
    // 4. down.
    {
        const long long rows = n_hits * (long long) H;
        const unsigned blocks = (unsigned) ((rows + warps - 1) / warps);
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            cs->submit([&](sycl::handler &cgh) {
                const float *x_scales_nullptr_h_scales_nullptr_ct5 =
                    x_scales != nullptr ? h_scales : nullptr;

                cgh.parallel_for<dpct_kernel_name<class down_kernel_6b2ba9>>(
                    sycl::nd_range<3>(sycl::range(1, 1, blocks) *
                                          sycl::range(1, 1, THREADS),
                                      sycl::range(1, 1, THREADS)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            down_kernel(blob_base, slot_index, dst_index,
                                        blob_bytes, h_q8_0,
                                        x_scales_nullptr_h_scales_nullptr_ct5,
                                        out, (int)n_hits, nullptr);
                        });
            });
        }
        check("moe_hit_grouped_s2/down", stream);
    }
}

namespace {
// Plan v0.3 P4 token graph: which of this layer's routed experts are resident, decided ON THE DEVICE from the
// static residency row, so no host step sits between the ring and the hit kernels.  One warp; k <= 32.
__dpct_inline__ void hit_select_kernel(const int32_t *__restrict__ ids,
                                       const int32_t *__restrict__ res_row,
                                       int k, int n_expert,
                                       int32_t *__restrict__ slot,
                                       int32_t *__restrict__ dst,
                                       int32_t *__restrict__ count) {
    const int lane =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);
    int s = -1;
    if (lane < k) {
        const int e = ids[lane];
        if (e >= 0 && e < n_expert) s = res_row[e];
    }
    const unsigned hit = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(),
        (0xffffffffu &
         (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                     .get_local_linear_id())) &&
                s >= 0
            ? (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                          .get_local_linear_id())
            : 0,
        sycl::ext::oneapi::plus<>());
    if (s >= 0) {
        const int at = sycl::popcount(hit & ((1u << lane) - 1u));
        slot[at] = s;
        dst[at] = lane;
    }
    if (lane == 0) *count = sycl::popcount(hit);
}

// Plan v0.3 P6: the same for up to 128 routed entries (a verify window of T tokens x k): four warps, ballots
// compacted in entry order.
__dpct_inline__ void hit_select_multi_kernel(
    const int32_t *__restrict__ ids, const int32_t *__restrict__ res_row, int n,
    int n_expert, int32_t *__restrict__ slot, int32_t *__restrict__ dst,
    int32_t *__restrict__ count) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &warp_count = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[4]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int i = item_ct1.get_local_id(2), lane = i & 31, warp = i >> 5;
    int s = -1;
    if (i < n) {
        const int e = ids[i];
        if (e >= 0 && e < n_expert) s = res_row[e];
    }
    const unsigned hit = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(),
        (0xffffffffu &
         (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                     .get_local_linear_id())) &&
                s >= 0
            ? (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                          .get_local_linear_id())
            : 0,
        sycl::ext::oneapi::plus<>());
    if (lane == 0) warp_count[warp] = sycl::popcount(hit);
    item_ct1.barrier(sycl::access::fence_space::local_space);
    int before = 0;
#pragma unroll
    for (int w = 0; w < warp; ++w) before += warp_count[w];
    if (s >= 0) {
        const int at = before + sycl::popcount(hit & ((1u << lane) - 1u));
        slot[at] = s;
        dst[at] = i;
    }
    if (i == 0) *count = warp_count[0] + warp_count[1] + warp_count[2] + warp_count[3];
}

__dpct_inline__ void add_hits_kernel(float *__restrict__ parts,
                                     const float *__restrict__ hit_out,
                                     const int32_t *__restrict__ dst,
                                     const int32_t *__restrict__ count,
                                     int n_embd) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int h = item_ct1.get_group(1);
    if (h >= *count) return;
    const size_t row = (size_t) dst[h] * (size_t) n_embd;
#pragma unroll
    for (int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
         i < n_embd;
         i += item_ct1.get_group_range(2) * item_ct1.get_local_range(2))
        parts[row + i] += hit_out[row + i];
}
}  // namespace

void moe_hit_select(const int32_t* ids, const int32_t* res_row, int k, int n_expert, int32_t* slot, int32_t* dst,
                    int32_t* count, void* stream) {
    if (k < 1 || k > 32) { std::fprintf(stderr, "moe_hit_select: k must be 1..32\n"); std::exit(1); }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class hit_select_kernel_62d6cd>>(
                sycl::nd_range<3>(sycl::range(1, 1, 32), sycl::range(1, 1, 32)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        hit_select_kernel(ids, res_row, k, n_expert, slot, dst,
                                          count);
                    });
    }
    check("moe_hit_select", stream);
}

void moe_hit_grouped_s2_dev(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                            const int32_t* d_count, int64_t cap, int64_t blob_bytes, const uint8_t* x_q8_0,
                            void* scratch, float* out, void* stream, const float* x_scales) {
    if (cap <= 0) return;
    dpct::queue_ptr cs = strata::q_of(stream);
    const int warps = THREADS / 32;
    const uint64_t gu_bytes = ((uint64_t) cap * (uint64_t) (2 * FF) * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) cap * (uint64_t) (FF / 32) * 34 + 15) & ~15ull;
    float* gate_up = (float*) scratch;
    uint8_t* h_q8_0 = (uint8_t*) scratch + gu_bytes;
    float* h_scales = (float*) ((uint8_t*) scratch + gu_bytes + q8_bytes);
    {
        const long long rows = cap * 2LL * FF;
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            cs->parallel_for<dpct_kernel_name<class gu_kernel_fd0f96>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, (unsigned)((rows + warps - 1) / warps)) *
                        sycl::range(1, 1, THREADS),
                    sycl::range(1, 1, THREADS)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gu_kernel(blob_base, slot_index, blob_bytes, x_q8_0,
                                  x_scales, gate_up, (int)cap, d_count, nullptr,
                                  0);
                    });
        }
        check("moe_hit_grouped_s2_dev/gu", stream);
    }
    {
        const long long pairs = cap * (long long) FF;
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            cs->parallel_for<dpct_kernel_name<class swiglu_kernel_40adf2>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1,
                                (unsigned)((pairs + THREADS - 1) / THREADS)) *
                        sycl::range(1, 1, THREADS),
                    sycl::range(1, 1, THREADS)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    swiglu_kernel(gate_up, pairs);
                });
        }
        check("moe_hit_grouped_s2_dev/swiglu", stream);
    }
    if (x_scales != nullptr) quantize_q8_0_scaled(gate_up, h_q8_0, h_scales, cap * (int64_t) FF, stream);
    else quantize_q8_0(gate_up, h_q8_0, cap * (int64_t) FF, stream);
    {
        const long long rows = cap * (long long) H;
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            cs->submit([&](sycl::handler &cgh) {
                const float *x_scales_nullptr_h_scales_nullptr_ct5 =
                    x_scales != nullptr ? h_scales : nullptr;

                cgh.parallel_for<dpct_kernel_name<class down_kernel_e2a4a9>>(
                    sycl::nd_range<3>(
                        sycl::range(1, 1,
                                    (unsigned)((rows + warps - 1) / warps)) *
                            sycl::range(1, 1, THREADS),
                        sycl::range(1, 1, THREADS)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            down_kernel(blob_base, slot_index, dst_index,
                                        blob_bytes, h_q8_0,
                                        x_scales_nullptr_h_scales_nullptr_ct5,
                                        out, (int)cap, d_count);
                        });
            });
        }
        check("moe_hit_grouped_s2_dev/down", stream);
    }
}

void moe_hit_select_multi(const int32_t* ids, const int32_t* res_row, int n, int n_expert, int32_t* slot, int32_t* dst,
                          int32_t* count, void* stream) {
    if (n < 1 || n > 128) { std::fprintf(stderr, "moe_hit_select_multi: n must be 1..128\n"); std::exit(1); }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class hit_select_multi_kernel_a381b5>>(
                sycl::nd_range<3>(sycl::range(1, 1, 128),
                                  sycl::range(1, 1, 128)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        hit_select_multi_kernel(ids, res_row, n, n_expert, slot,
                                                dst, count);
                    });
    }
    check("moe_hit_select_multi", stream);
}

void moe_hit_grouped_s2_multi(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                              const int32_t* d_count, int64_t cap, int64_t blob_bytes, const uint8_t* x_q8_0,
                              const float* x_scales, int k_per_token, void* scratch, float* out, void* stream) {
    if (cap <= 0) return;
    dpct::queue_ptr cs = strata::q_of(stream);
    const int warps = THREADS / 32;
    const uint64_t gu_bytes = ((uint64_t) cap * (uint64_t) (2 * FF) * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) cap * (uint64_t) (FF / 32) * 34 + 15) & ~15ull;
    float* gate_up = (float*) scratch;
    uint8_t* h_q8_0 = (uint8_t*) scratch + gu_bytes;
    float* h_scales = (float*) ((uint8_t*) scratch + gu_bytes + q8_bytes);
    {
        const long long rows = cap * 2LL * FF;
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            cs->parallel_for<dpct_kernel_name<class gu_kernel_583c8a>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, (unsigned)((rows + warps - 1) / warps)) *
                        sycl::range(1, 1, THREADS),
                    sycl::range(1, 1, THREADS)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gu_kernel(blob_base, slot_index, blob_bytes, x_q8_0,
                                  x_scales, gate_up, (int)cap, d_count,
                                  dst_index, k_per_token);
                    });
        }
        check("moe_hit_grouped_s2_multi/gu", stream);
    }
    {
        const long long pairs = cap * (long long) FF;
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            cs->parallel_for<dpct_kernel_name<class swiglu_kernel_f4e4db>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1,
                                (unsigned)((pairs + THREADS - 1) / THREADS)) *
                        sycl::range(1, 1, THREADS),
                    sycl::range(1, 1, THREADS)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    swiglu_kernel(gate_up, pairs);
                });
        }
        check("moe_hit_grouped_s2_multi/swiglu", stream);
    }
    if (x_scales != nullptr) quantize_q8_0_scaled(gate_up, h_q8_0, h_scales, cap * (int64_t) FF, stream);
    else quantize_q8_0(gate_up, h_q8_0, cap * (int64_t) FF, stream);
    {
        const long long rows = cap * (long long) H;
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            cs->submit([&](sycl::handler &cgh) {
                const float *x_scales_nullptr_h_scales_nullptr_ct5 =
                    x_scales != nullptr ? h_scales : nullptr;

                cgh.parallel_for<dpct_kernel_name<class down_kernel_1ae146>>(
                    sycl::nd_range<3>(
                        sycl::range(1, 1,
                                    (unsigned)((rows + warps - 1) / warps)) *
                            sycl::range(1, 1, THREADS),
                        sycl::range(1, 1, THREADS)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            down_kernel(blob_base, slot_index, dst_index,
                                        blob_bytes, h_q8_0,
                                        x_scales_nullptr_h_scales_nullptr_ct5,
                                        out, (int)cap, d_count);
                        });
            });
        }
        check("moe_hit_grouped_s2_multi/down", stream);
    }
}

namespace {
// Plan v0.3 P6: experts GROUPED - one blob pointer per group (a VRAM slot or a mapped host blob read over PCIe),
// every entry of the group (a token routed to that expert) computed from ONE read of each row.  Per entry the
// arithmetic is `row_dot_s2_q8`'s, chunk by chunk in the same lane order, so every entry is bitwise the per-entry
// hit kernel's.
constexpr int GU_CHUNKS = (H / 32 + 31) / 32;   // 3: chunks of a gate/up row per lane (80 chunks / 32 lanes)
constexpr int GMAX = 8;                          // entries per group (tokens routed to one expert in a window)
constexpr int GU_ROWS = 32;                      // gate/up rows per block: 4 per warp
constexpr int D_ROWS = 64;                       // down rows per block: 8 per warp

// One activation chunk's contribution, `row_dot_s2_q8`'s inner body with the 32 int8 of the chunk already in
// aligned words: same dp4a sequence, same float expression, so the result is bitwise the per-entry kernel's.
__dpct_inline__ float chunk_dot(sycl::uint2 cb, const int *xw, float dw,
                                float dx) {
    const uint8_t* cbytes = (const uint8_t*) &cb;
    int s = 0, hx = 0;
    const int ones = 0x01010101;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const unsigned cbyte = cbytes[j];
        const int cw = (int) ((cbyte & 3u) | (((cbyte >> 2) & 3u) << 8) | (((cbyte >> 4) & 3u) << 16) |
                              (((cbyte >> 6) & 3u) << 24));
        s = strata::dp4a(cw, xw[j], s);
        hx = strata::dp4a(ones, xw[j], hx);
    }
    return dw * dx * (float) (s - hx);
}

// Gate/up: a block = GU_ROWS rows of ONE group.  The group's activations (each entry's token row of x_q8_0 and
// its fp32 scales) are staged once into shared memory as aligned words; each warp then walks its rows, loading
// each lane's code chunks once and dotting them with every entry.
/*
DPCT1110: The total declared local variable size in device function
gu_grouped_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void gu_grouped_kernel(
    const unsigned long long *__restrict__ grp_ptr,
    const int32_t *__restrict__ grp_start, const int32_t *__restrict__ n_groups,
    const int32_t *__restrict__ ent_tok, const uint8_t *__restrict__ x_q8_0,
    const float *__restrict__ x_scales, float *__restrict__ gate_up,
    int cap_entries) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &xs_q =
    *sycl::ext::oneapi::group_local_memory_for_overwrite<int[GMAX][H / 4]>(
        sycl::ext::oneapi::this_work_item::get_work_group<
            3>()); // the entries' int8 activations as words (2560 B each)
    auto &xs_d = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[GMAX][H / 32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int g = item_ct1.get_group(1);
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], ne = sycl::min(grp_start[g + 1] - e0, GMAX);
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    for (int i = t; i < ne * (H / 32); i += item_ct1.get_local_range(2)) {
        const int k = i / (H / 32), c = i - k * (H / 32);
        const uint8_t* xb = x_q8_0 + (size_t) ent_tok[e0 + k] * (size_t) (H / 32) * 34 + (size_t) c * 34;
        xs_d[k][c] = x_scales ? x_scales[(size_t) ent_tok[e0 + k] * (H / 32) + c] : f16_at(xb);
        const int8_t* q = (const int8_t*) (xb + 2);
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            int v;
            memcpy(&v, q + 4 * w, 4);
            xs_q[k][c * 8 + w] = v;
        }
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const int row0 = item_ct1.get_group(2) * GU_ROWS;
    for (int rr = warp; rr < GU_ROWS; rr += 8) {
        const int i = row0 + rr;
        const uint8_t* codes = blob + (size_t) i * ROW_GU;
        const uint8_t* scales = blob + O_GU_SCALES + (size_t) i * SC_GU * 2;
        sycl::uint2 cb[GU_CHUNKS];
        float dw[GU_CHUNKS];
#pragma unroll
        for (int q = 0; q < GU_CHUNKS; ++q) {
            const int c = lane + 32 * q;
            if (c < H / 32) {
                cb[q] = *(const sycl::uint2 *)(codes + (size_t)c * 8);
                dw[q] = f16_at(scales + (size_t) (c >> 1) * 2);
            }
        }
        for (int k = 0; k < ne; ++k) {
            float acc = 0.0f;
#pragma unroll
            for (int q = 0; q < GU_CHUNKS; ++q) {
                const int c = lane + 32 * q;
                if (c >= H / 32) break;
                acc += chunk_dot(cb[q], &xs_q[k][c * 8], dw[q], xs_d[k][c]);
            }
            const float sum = warp_sum(acc);
            if (lane == 0) {
                const int e = e0 + k, r = i >> 1;
                const size_t base = (i & 1) ? ((size_t) cap_entries * FF + (size_t) e * FF) : ((size_t) e * FF);
                gate_up[base + (size_t) r] = sum;
            }
        }
    }
}

// Down: a block = D_ROWS rows of ONE group; the entries' quantized intermediates staged once.  A down row is 20
// chunks, so lanes 0..19 each hold one chunk, as in the per-entry kernel.
__dpct_inline__ void down_grouped_kernel(
    const unsigned long long *__restrict__ grp_ptr,
    const int32_t *__restrict__ grp_start, const int32_t *__restrict__ n_groups,
    const int32_t *__restrict__ ent_dst, const uint8_t *__restrict__ h_q8_0,
    const float *__restrict__ h_scales, float *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &hs_q =
    *sycl::ext::oneapi::group_local_memory_for_overwrite<int[GMAX][FF / 4]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &hs_d = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[GMAX][FF / 32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int g = item_ct1.get_group(1);
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], ne = sycl::min(grp_start[g + 1] - e0, GMAX);
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    for (int i = t; i < ne * (FF / 32); i += item_ct1.get_local_range(2)) {
        const int k = i / (FF / 32), c = i - k * (FF / 32);
        const uint8_t* xb = h_q8_0 + (size_t) (e0 + k) * (size_t) (FF / 32) * 34 + (size_t) c * 34;
        hs_d[k][c] = h_scales ? h_scales[(size_t) (e0 + k) * (FF / 32) + c] : f16_at(xb);
        const int8_t* q = (const int8_t*) (xb + 2);
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            int v;
            memcpy(&v, q + 4 * w, 4);
            hs_q[k][c * 8 + w] = v;
        }
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const int row0 = item_ct1.get_group(2) * D_ROWS;
    for (int rr = warp; rr < D_ROWS; rr += 8) {
        const int r = row0 + rr;
        const uint8_t* codes = blob + O_D_CODES + (size_t) r * ROW_D;
        const uint8_t* scales = blob + O_D_SCALES + (size_t) r * SC_D * 2;
        const int c = lane;
        sycl::uint2 cb = sycl::uint2(0, 0);
        float dw = 0.0f;
        if (c < FF / 32) {
            cb = *(const sycl::uint2 *)(codes + (size_t)c * 8);
            dw = f16_at(scales + (size_t) (c >> 1) * 2);
        }
        for (int k = 0; k < ne; ++k) {
            float acc = 0.0f;
            if (c < FF / 32) acc += chunk_dot(cb, &hs_q[k][c * 8], dw, hs_d[k][c]);
            const float sum = warp_sum(acc);
            if (lane == 0) out[(size_t) ent_dst[e0 + k] * H + r] = sum;
        }
    }
}
}  // namespace

namespace {
// Plan v0.3 P6: groups built on the device when every expert is resident at `base + id * blob` (the MTP layer):
// one block of 128 threads, groups in first-appearance order, entries of a group in routing order.
__dpct_inline__ void group_resident_kernel(
    const int32_t *__restrict__ ids, int n, int k_per_tok, const uint8_t *base,
    long long blob, unsigned long long *__restrict__ grp_ptr,
    int32_t *__restrict__ grp_start, int32_t *__restrict__ counts,
    int32_t *__restrict__ ent_dst, int32_t *__restrict__ ent_tok) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &e_s = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[128]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &first_s =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[128]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &size_s =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[128]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &gidx_s =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[128]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &gstart_s =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[129]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int i = item_ct1.get_local_id(2);
    const int e = i < n ? ids[i] : -1;
    e_s[i] = e;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    int first = i, rank = 0, size = 0;
    if (i < n) {
#pragma unroll
        for (int j = 0; j < i; ++j)
            if (e_s[j] == e) { if (first == i) first = j; ++rank; }
        if (first == i)
#pragma unroll
            for (int j = i; j < n; ++j) size += e_s[j] == e;
    }
    first_s[i] = first;
    size_s[i] = (i < n && first == i) ? size : 0;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (i == 0) {
        int gi = 0, acc = 0;
#pragma unroll
        for (int j = 0; j < n; ++j)
            if (first_s[j] == j) {
                gidx_s[j] = gi;
                gstart_s[gi] = acc;
                grp_ptr[gi] = (unsigned long long) (base + (size_t) e_s[j] * (size_t) blob);
                grp_start[gi] = acc;
                acc += size_s[j];
                ++gi;
            }
        grp_start[gi] = acc;
        counts[0] = gi;
        counts[1] = acc;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (i < n) {
        const int at = gstart_s[gidx_s[first]] + rank;
        ent_dst[at] = i;
        ent_tok[at] = i / k_per_tok;
    }
}
}  // namespace

void moe_group_resident(const int32_t* ids, int n, int k_per_tok, const uint8_t* base, int64_t blob,
                        unsigned long long* grp_ptr, int32_t* grp_start, int32_t* counts, int32_t* ent_dst,
                        int32_t* ent_tok, void* stream) {
    if (n < 1 || n > 128) { std::fprintf(stderr, "moe_group_resident: n must be 1..128\n"); std::exit(1); }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class group_resident_kernel_937b58>>(
                sycl::nd_range<3>(sycl::range(1, 1, 128),
                                  sycl::range(1, 1, 128)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    group_resident_kernel(ids, n, k_per_tok, base,
                                          (long long)blob, grp_ptr, grp_start,
                                          counts, ent_dst, ent_tok);
                });
    }
    check("moe_group_resident", stream);
}

void moe_grouped_s2(const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                    const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries,
                    const uint8_t* x_q8_0, const float* x_scales, void* scratch, float* out, void* stream) {
    if (cap_groups <= 0 || cap_entries <= 0) return;
    dpct::queue_ptr cs = strata::q_of(stream);
    const uint64_t gu_bytes = ((uint64_t) cap_entries * (uint64_t) (2 * FF) * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) cap_entries * (uint64_t) (FF / 32) * 34 + 15) & ~15ull;
    float* gate_up = (float*) scratch;
    uint8_t* h_q8_0 = (uint8_t*) scratch + gu_bytes;
    float* h_scales = (float*) ((uint8_t*) scratch + gu_bytes + q8_bytes);
    {
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            cs->parallel_for<dpct_kernel_name<class gu_grouped_kernel_393d46>>(
                sycl::nd_range<3>(sycl::range(1, (unsigned)cap_groups,
                                              (unsigned)(2 * FF / GU_ROWS)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gu_grouped_kernel(grp_ptr, grp_start, n_groups, ent_tok,
                                          x_q8_0, x_scales, gate_up,
                                          (int)cap_entries);
                    });
        }
        check("moe_grouped_s2/gu", stream);
    }
    {
        const long long pairs = cap_entries * (long long) FF;
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            cs->parallel_for<dpct_kernel_name<class swiglu_kernel_2292bf>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1,
                                (unsigned)((pairs + THREADS - 1) / THREADS)) *
                        sycl::range(1, 1, THREADS),
                    sycl::range(1, 1, THREADS)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    swiglu_kernel(gate_up, pairs);
                });
        }
        check("moe_grouped_s2/swiglu", stream);
    }
    if (x_scales != nullptr) quantize_q8_0_scaled(gate_up, h_q8_0, h_scales, cap_entries * (int64_t) FF, stream);
    else quantize_q8_0(gate_up, h_q8_0, cap_entries * (int64_t) FF, stream);
    {
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            cs->submit([&](sycl::handler &cgh) {
                const float *x_scales_nullptr_h_scales_nullptr_ct5 =
                    x_scales != nullptr ? h_scales : nullptr;

                cgh.parallel_for<
                    dpct_kernel_name<class down_grouped_kernel_f03334>>(
                    sycl::nd_range<3>(sycl::range(1, (unsigned)cap_groups,
                                                  (unsigned)(H / D_ROWS)) *
                                          sycl::range(1, 1, 256),
                                      sycl::range(1, 1, 256)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            down_grouped_kernel(
                                grp_ptr, grp_start, n_groups, ent_dst, h_q8_0,
                                x_scales_nullptr_h_scales_nullptr_ct5, out);
                        });
            });
        }
        check("moe_grouped_s2/down", stream);
    }
}

void moe_hit_add(float* parts, const float* hit_out, const int32_t* dst, const int32_t* count, int64_t cap,
                 int64_t n_embd, void* stream) {
    if (cap <= 0) return;
    const dpct::dim3 grid(
        (unsigned)((n_embd + 255) / 256 < 8 ? (n_embd + 255) / 256 : 8),
        (unsigned)cap);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class add_hits_kernel_9d7642>>(
                sycl::nd_range<3>(grid * sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    add_hits_kernel(parts, hit_out, dst, count, (int)n_embd);
                });
    }
    check("moe_hit_add", stream);
}

void moe_hit_grouped_s2_cpu_order(const uint8_t *blob_base,
                                  const int32_t *slot_index,
                                  const int32_t *dst_index, int64_t n_hits,
                                  int64_t blob_bytes, const uint8_t *x_q8_0,
                                  void *scratch, float *out, void *stream,
                                  const float *x_scales,
                                  float *gate_up_trace) try {
    if (n_hits <= 0) return;
    if (x_scales == nullptr) {
        std::fprintf(stderr, "moe_hit_grouped_s2_cpu_order requires fp32 activation scales\n");
        std::exit(1);
    }
    dpct::queue_ptr cs = strata::q_of(stream);
    const uint64_t gu_bytes = ((uint64_t) n_hits * 2 * FF * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) n_hits * (FF / 32) * 34 + 15) & ~15ull;
    const uint64_t scale_bytes = ((uint64_t) n_hits * (FF / 32) * 4 + 15) & ~15ull;
    float* gu = (float*) scratch;
    uint8_t* hq = (uint8_t*) scratch + gu_bytes;
    float* hs = (float*) (hq + q8_bytes);
    float* hh = (float*) ((uint8_t*) hs + scale_bytes);
    float* xh = (float*) ((uint8_t*) hh + scale_bytes);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs->submit([&](sycl::handler &cgh) {
            auto H_ct3 = H / 32;

            cgh.parallel_for<
                dpct_kernel_name<class activation_correction_kernel_f8a161>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, (H / 32 + THREADS - 1) / THREADS) *
                        sycl::range(1, 1, THREADS),
                    sycl::range(1, 1, THREADS)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    activation_correction_kernel(x_q8_0, x_scales, xh, H_ct3);
                });
        });
    }
    check("cpu_order/input_correction", stream);
    const int rows_per_block = THREADS / 8;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs->parallel_for<
            dpct_kernel_name<class cpu_order_projection_kernel_ca15a9,
                             dpct_kernel_scalar<false>>>(
            sycl::nd_range<3>(
                sycl::range(1, 1,
                            (unsigned)((n_hits * 2 * FF + rows_per_block - 1) /
                                       rows_per_block)) *
                    sycl::range(1, 1, THREADS),
                sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                cpu_order_projection_kernel<false>(
                    blob_base, slot_index, dst_index, blob_bytes, x_q8_0,
                    x_scales, xh, gu, (int)n_hits);
            });
    }
    check("cpu_order/gate_up", stream);
    if (gate_up_trace != nullptr &&
        /*
        DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        DPCT_CHECK_ERROR(cs->memcpy(
            gate_up_trace, gu, (size_t)n_hits * 2 * FF * sizeof(float))) != 0) {
        std::fprintf(stderr, "cpu_order/gate_up_trace copy failed\n");
        std::exit(1);
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs->submit([&](sycl::handler &cgh) {
            auto int_n_hits_FF_ct1 = (int)n_hits * FF;

            cgh.parallel_for<
                dpct_kernel_name<class cpu_order_swiglu_kernel_883f7b>>(
                sycl::nd_range<3>(
                    sycl::range(
                        1, 1,
                        (unsigned)((n_hits * FF + THREADS - 1) / THREADS)) *
                        sycl::range(1, 1, THREADS),
                    sycl::range(1, 1, THREADS)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    cpu_order_swiglu_kernel(gu, int_n_hits_FF_ct1);
                });
        });
    }
    check("cpu_order/swiglu", stream);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs->submit([&](sycl::handler &cgh) {
            auto int_n_hits_FF_ct4 = (int)n_hits * (FF / 32);

            cgh.parallel_for<
                dpct_kernel_name<class cpu_order_quantize_kernel_813a16>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1,
                                (unsigned)((n_hits * (FF / 32) + THREADS - 1) /
                                           THREADS)) *
                        sycl::range(1, 1, THREADS),
                    sycl::range(1, 1, THREADS)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    cpu_order_quantize_kernel(gu, hq, hs, hh,
                                              int_n_hits_FF_ct4);
                });
        });
    }
    check("cpu_order/intermediate_quantize", stream);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs->parallel_for<
            dpct_kernel_name<class cpu_order_projection_kernel_7f4897,
                             dpct_kernel_scalar<true>>>(
            sycl::nd_range<3>(
                sycl::range(1, 1,
                            (unsigned)((n_hits * H + rows_per_block - 1) /
                                       rows_per_block)) *
                    sycl::range(1, 1, THREADS),
                sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                cpu_order_projection_kernel<true>(blob_base, slot_index,
                                                  dst_index, blob_bytes, hq, hs,
                                                  hh, out, (int)n_hits);
            });
    }
    check("cpu_order/down", stream);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

}  // namespace strata::kernels
