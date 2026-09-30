// src/kernels/cuda/iq_kernels.cu - see include/strata/kernels/iq_kernels.hpp.
//
// The dot products (vec_dot_*_q8_1), the dequantizers and the q8_1 quantizer are transcribed from llama.cpp
// (ggml/src/ggml-cuda/vecdotq.cuh, dequantize.cuh, quantize.cu at the commit in third_party/ggml/VERSION.txt;
// MIT license, third_party/ggml/LICENSE).  The block structs and codebook grids come from its ggml-common.h,
// included unchanged.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_math.hpp"
#include "strata/sycl_queue.hpp"
#include "strata/kernels/iq_kernels.hpp"

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
}

// ---------------------------------------------------------------- llama.cpp helpers (vecdotq.cuh)
__dpct_inline__ int get_int_b2(const void *x, const int &i32) {
    const uint16_t* x16 = (const uint16_t*) x;
    int x32 = x16[2 * i32 + 0] << 0;
    x32 |= x16[2 * i32 + 1] << 16;
    return x32;
}
__dpct_inline__ int get_int_b4(const void *x, const int &i32) {
    return ((const int *)x)[i32];
}
__dpct_inline__ uint32_t unpack_ksigns(const uint8_t v) {
    const uint32_t p = sycl::popcount(v) & 1;
    const uint32_t s = v ^ p << 7;
    return s * 0x01010101;
}
__dpct_inline__ sycl::int2 get_int_from_table_16(const int &q4,
                                                 const int8_t *table) {
    const uint32_t* table32 = (const uint32_t*) table;
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low =
            dpct::byte_level_permute(table32[0], table32[1], q4 >> shift);
        const uint32_t high =
            dpct::byte_level_permute(table32[2], table32[3], q4 >> shift);
        tmp[i] = dpct::byte_level_permute(low, high,
                                          low_high_selection_indices >> shift);
    }
    return sycl::int2(dpct::byte_level_permute(tmp[0], tmp[1], 0x6420),
                      dpct::byte_level_permute(tmp[0], tmp[1], 0x7531));
}
#define ggml_cuda_dp4a(a, b, c) strata::dp4a((a), (b), (c))

// ---------------------------------------------------------------- the dot products (vecdotq.cuh)
__dpct_inline__ float vec_dot_q2_0_q8_1(const void *__restrict__ vbq,
                                        const block_q8_1 *__restrict__ bq8_1,
                                        const int &kbx, const int &iqs) {
    const block_q2_0* bq2_0 = (const block_q2_0*) vbq + kbx;
    const float d2 = bq2_0->d;
    const int16_t* qs = (const int16_t*) bq2_0->qs + iqs * 4;
    const block_q8_1* bq8_1_chunk = bq8_1 + iqs;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = qs[j];
        const int u = get_int_b4(bq8_1_chunk->qs, j * 2 + 0);
        const int v = get_int_b4(bq8_1_chunk->qs, j * 2 + 1);
        const int qe = dpct::byte_level_permute(0x020100FF, 0x020100FF, q >> 0);
        const int qo = dpct::byte_level_permute(0x020100FF, 0x020100FF, q >> 2);
        const int qx = dpct::byte_level_permute(qe, qo, 0x5140);
        const int qy = dpct::byte_level_permute(qe, qo, 0x7362);
        sumi = ggml_cuda_dp4a(u, qx, sumi);
        sumi = ggml_cuda_dp4a(v, qy, sumi);
    }
    const float d8 = bq8_1_chunk->ds[0];
    return d2 * d8 * sumi;
}

__dpct_inline__ float vec_dot_iq2_xxs_q8_1(const void *__restrict__ vbq,
                                           const block_q8_1 *__restrict__ bq8_1,
                                           const int &kbx, const int &iqs) {
    const block_iq2_xxs* bq2 = (const block_iq2_xxs*) vbq + kbx;
    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t* aux8 = (const uint8_t*) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const sycl::uint2 grid_pos =
            ((const sycl::uint2 *)iq2xxs_grid)[aux8[k0 / 2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
        const int signs0 = dpct::vectorized_binary<sycl::uchar4>(
            signs & 0x08040201, 0, std::not_equal_to<>());
        const int grid0 = dpct::vectorized_binary<sycl::uchar4>(
            grid_pos.x() ^ signs0, signs0, std::minus<>());
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 0);
        sumi = ggml_cuda_dp4a(grid0, u0, sumi);
        const int signs1 = dpct::vectorized_binary<sycl::uchar4>(
            signs & 0x80402010, 0, std::not_equal_to<>());
        const int grid1 = dpct::vectorized_binary<sycl::uchar4>(
            grid_pos.y() ^ signs1, signs1, std::minus<>());
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 1);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }
    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    const float d = sycl::vec<sycl::half, 1>(bq2->d)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    bq8_1[iqs / 2].ds[0];
    return d * sumi;
}

__dpct_inline__ float vec_dot_iq2_xs_q8_1(const void *__restrict__ vbq,
                                          const block_q8_1 *__restrict__ bq8_1,
                                          const int &kbx, const int &iqs) {
    const block_iq2_xs* bq2 = (const block_iq2_xs*) vbq + kbx;
    const sycl::int2 q2_packed =
        sycl::int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
    const uint16_t* q2 = (const uint16_t*) &q2_packed;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const sycl::uint2 grid_pos =
            ((const sycl::uint2 *)iq2xs_grid)[q2[l0 / 2] & 0x1FF];
        const uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
        const int signs0 = dpct::vectorized_binary<sycl::uchar4>(
            signs & 0x08040201, 0, std::not_equal_to<>());
        const int grid_l = dpct::vectorized_binary<sycl::uchar4>(
            grid_pos.x() ^ signs0, signs0, std::minus<>());
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = dpct::vectorized_binary<sycl::uchar4>(
            signs & 0x80402010, 0, std::not_equal_to<>());
        const int grid_h = dpct::vectorized_binary<sycl::uchar4>(
            grid_pos.y() ^ signs1, signs1, std::minus<>());
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = sycl::vec<sycl::half, 1>(bq2->d)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    bq8_1[iqs / 2].ds[0];
    return d * sumi;
}

__dpct_inline__ float vec_dot_iq2_s_q8_1(const void *__restrict__ vbq,
                                         const block_q8_1 *__restrict__ bq8_1,
                                         const int &kbx, const int &iqs) {
    const block_iq2_s* bq2 = (const block_iq2_s*) vbq + kbx;
    const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq2->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int* grid_pos = (const int*) (iq2s_grid + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
        const int signs0 = dpct::vectorized_binary<sycl::uchar4>(
            ((signs_packed_8[l0 / 2] & 0x03) << 7) |
                ((signs_packed_8[l0 / 2] & 0x0C) << 21),
            0x00000000, std::not_equal_to<>());
        const int signs1 = dpct::vectorized_binary<sycl::uchar4>(
            ((signs_packed_8[l0 / 2] & 0x30) << 3) |
                ((signs_packed_8[l0 / 2] & 0xC0) << 17),
            0x00000000, std::not_equal_to<>());
        const int grid_l = dpct::vectorized_binary<sycl::uchar4>(
            grid_pos[0] ^ signs0, signs0, std::minus<>());
        const int grid_h = dpct::vectorized_binary<sycl::uchar4>(
            grid_pos[1] ^ signs1, signs1, std::minus<>());
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = sycl::vec<sycl::half, 1>(bq2->d)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    bq8_1[iqs / 2].ds[0];
    return d * sumi;
}

__dpct_inline__ float vec_dot_iq3_xxs_q8_1(const void *__restrict__ vbq,
                                           const block_q8_1 *__restrict__ bq8_1,
                                           const int &kbx, const int &iqs) {
    const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
    const sycl::int2 q3_packed =
        sycl::int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* q3 = (const uint8_t*) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const sycl::int2 grid_pos =
            sycl::int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
        const int signs0 = dpct::vectorized_binary<sycl::uchar4>(
            signs & 0x08040201, 0, std::not_equal_to<>());
        const int grid_l = dpct::vectorized_binary<sycl::uchar4>(
            grid_pos.x() ^ signs0, signs0, std::minus<>());
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = dpct::vectorized_binary<sycl::uchar4>(
            signs & 0x80402010, 0, std::not_equal_to<>());
        const int grid_h = dpct::vectorized_binary<sycl::uchar4>(
            grid_pos.y() ^ signs1, signs1, std::minus<>());
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 2;
    const float d = sycl::vec<sycl::half, 1>(bq3->d)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    bq8_1[iqs / 2].ds[0];
    return d * sumi;
}

__dpct_inline__ float vec_dot_iq3_s_q8_1(const void *__restrict__ vbq,
                                         const block_q8_1 *__restrict__ bq8_1,
                                         const int &kbx, const int &iqs) {
    const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
    const sycl::int2 qs_packed =
        sycl::int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq3->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const sycl::int2 grid_pos =
            sycl::int2(iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                       iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
        const int signs0 = dpct::vectorized_binary<sycl::uchar4>(
            ((signs_packed_8[l0 / 2] & 0x03) << 7) |
                ((signs_packed_8[l0 / 2] & 0x0C) << 21),
            0x00000000, std::not_equal_to<>());
        const int signs1 = dpct::vectorized_binary<sycl::uchar4>(
            ((signs_packed_8[l0 / 2] & 0x30) << 3) |
                ((signs_packed_8[l0 / 2] & 0xC0) << 17),
            0x00000000, std::not_equal_to<>());
        const int grid_l = dpct::vectorized_binary<sycl::uchar4>(
            grid_pos.x() ^ signs0, signs0, std::minus<>());
        const int grid_h = dpct::vectorized_binary<sycl::uchar4>(
            grid_pos.y() ^ signs1, signs1, std::minus<>());
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    sumi *= 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = sycl::vec<sycl::half, 1>(bq3->d)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    bq8_1[iqs / 2].ds[0];
    return d * sumi;
}

__dpct_inline__ float vec_dot_iq1_m_q8_1(const void *__restrict__ vbq,
                                         const block_q8_1 *__restrict__ bq8_1,
                                         const int &kbx, const int &iqs) {
    const block_iq1_m* bq1 = (const block_iq1_m*) vbq + kbx;
    const int qs_packed = get_int_b4(bq1->qs, iqs);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    int sumi[2] = {0, 0};
    float sumf[2] = {0.0f, 0.0f};
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qhl = bq1->qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
        const int grid = iq1s_grid_gpu[qs[l0 / 2] | ((qhl & 0x07) << 8)];
        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;
        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
        sumi[l0 / 4] = ggml_cuda_dp4a(grid0, u0, sumi[l0 / 4]);
        sumi[l0 / 4] = ggml_cuda_dp4a(grid1, u1, sumi[l0 / 4]);
        const float delta = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f * IQ1M_DELTA / 0x08);
        int sumy = 0;
        sumy = ggml_cuda_dp4a(u0, 0x01010101, sumy);
        sumy = ggml_cuda_dp4a(u1, 0x01010101, sumy);
        sumf[l0 / 4] += delta * sumy;
    }
    const uint16_t* sc = (const uint16_t*) bq1->scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    const float d = sycl::vec<sycl::half, 1>(scale.f16)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    bq8_1[iqs].ds[0];
    const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
    const int sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
    const int sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
    return d * ((sumi[0] + sumf[0]) * sc0 + (sumi[1] + sumf[1]) * sc1);
}

__dpct_inline__ float vec_dot_iq4_nl_q8_1(const void *__restrict__ vbq,
                                          const block_q8_1 *__restrict__ bq8_1,
                                          const int &kbx, const int &iqs) {
    const block_iq4_nl* bq4 = (const block_iq4_nl*) vbq + kbx;
    const int* q8 = (const int*) bq8_1->qs + iqs;
    int sumi = 0;
#pragma unroll
    for (int l = 0; l < 2; ++l) {
        const int aux_q4 = get_int_b2(bq4->qs, iqs + l);
        const sycl::int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        sumi = ggml_cuda_dp4a(v.x(), q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y(), q8[l + 4], sumi);
    }
    const float d = sycl::vec<sycl::half, 1>(bq4->d)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    bq8_1->ds[0];
    return d * sumi;
}

// IQ4_XS: 256 values as 8 sub-blocks of 32 (6-bit scale each); one call covers one sub-block (iqs = 4 * sub-block),
// and `bq8_1` is the super-block's first q8_1 block, so the call's activation is bq8_1[iqs / 4].  The GSQ-RCO IQ3_S
// file keeps one layer's routed gate/up experts in this format.
__dpct_inline__ float vec_dot_iq4_xs_q8_1(const void *__restrict__ vbq,
                                          const block_q8_1 *__restrict__ bq8_1,
                                          const int &kbx, const int &iqs) {
    const block_iq4_xs* bq4 = (const block_iq4_xs*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        const sycl::int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        const int u0 = get_int_b4(bq8_1[iqs / 4].qs, j + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 4].qs, j + 4);
        sumi = ggml_cuda_dp4a(v.x(), u0, sumi);
        sumi = ggml_cuda_dp4a(v.y(), u1, sumi);
    }
    const int ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    const float d = sycl::vec<sycl::half, 1>(bq4->d)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    bq8_1[iqs / 4].ds[0];
    return d * sumi;
}

// ---------------------------------------------------------------- the formats
// qk = values per block, ipb = dot calls per block (qi / vdr), step = the iqs stride between calls.
template<int TY> struct Fmt;
template<> struct Fmt<16> { static constexpr int qk = 256, ipb = 8, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_xxs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<17> { static constexpr int qk = 256, ipb = 8, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_xs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<18> { static constexpr int qk = 256, ipb = 8, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq3_xxs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<20> { static constexpr int qk = 32, ipb = 2, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq4_nl_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<21> { static constexpr int qk = 256, ipb = 8, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq3_s_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<23> { static constexpr int qk = 256, ipb = 8, step = 4;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq4_xs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<22> { static constexpr int qk = 256, ipb = 8, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_s_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<29> { static constexpr int qk = 256, ipb = 8, step = 1;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq1_m_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<42> { static constexpr int qk = 64, ipb = 2, step = 1;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q2_0_q8_1(v, y, kbx, iqs); } };

__dpct_inline__ float warp_sum(float v) {
#pragma unroll
    /*
    DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = 16; o > 0; o >>= 1) v +=
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), v,
            o);
    return v;
}

// SYCL port: lanes per row for the grouped expert kernels. A 2560-wide row is 80 (block, part) calls; over 32
// lanes that is 2.5 calls each and a 5-step reduction - the work per sub-group is too small for the fixed cost.
// STRATA_EXPERT_LANES (compile time) lanes share a row; 256 / that many rows per work-group.
#ifndef STRATA_EXPERT_LANES
#define STRATA_EXPERT_LANES 8
#endif
constexpr int kExpertLanes = STRATA_EXPERT_LANES;
constexpr int kExpertRows = 256 / kExpertLanes;
template <int LANES>
__dpct_inline__ float lanes_sum(float v) {
#pragma unroll
    for (int o = LANES / 2; o > 0; o >>= 1)
        v += dpct::experimental::permute_sub_group_by_xor(0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), v, o);
    return v;
}
template <int TY, int LANES>
__dpct_inline__ float row_dot_lanes(const uint8_t *row, const block_q8_1 *x, int nb, int sub) {
    using F = Fmt<TY>;
    float s = 0.0f;
    for (int k = sub; k < nb * F::ipb; k += LANES) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        s += F::dot(row, x + kbx * (F::qk / 32), kbx, iqs);
    }
    return lanes_sum<LANES>(s);
}

// One row against one q8_1 activation, the whole warp: call k = (block, part) is lane-strided.
template <int TY>
__dpct_inline__ float row_dot(const uint8_t *row, const block_q8_1 *x, int nb,
                              int lane) {
    using F = Fmt<TY>;
    float s = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        /*
        DPCT1084: The function call "Fmt::dot" has multiple migration
        results in different template instantiations that could not be unified.
        You may need to adjust the code.
        */
        s += F::dot(row, x + kbx * (F::qk / 32), kbx, iqs);
    }
    return warp_sum(s);
}

template <int TY>
__dpct_inline__ void
mmvq_kernel(const uint8_t *__restrict__ w, size_t row_bytes,
            const block_q8_1 *__restrict__ x, float *__restrict__ y, int n_in,
            int n_out, int ncols) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = item_ct1.get_group(2) * 4 + item_ct1.get_local_id(1);
    if (row >= n_out) return;
    const int lane = item_ct1.get_local_id(2);
    const int nb = n_in / Fmt<TY>::qk;
    const uint8_t* wr = w + (size_t) row * row_bytes;
    for (int c = 0; c < ncols; ++c) {
        const float s =
            row_dot<TY>(wr, x + (size_t)c * (n_in / 32), nb, lane);
        if (lane == 0) y[(size_t) c * n_out + row] = s;
    }
}

// ---------------------------------------------------------------- grouped native experts

// ---------------------------------------------------------------- SYCL port: multi-entry dots
//
// A window routes the same expert for several tokens (up to 6). The single-entry dot above recomputes the
// dequantised weights - grid lookups, sign unpacking, the byte-wise compare/xor/subtract emulations - once per
// token; only the dp4a against the activation and the scale differ. Multi<TY> splits each format into `prep`
// (weights, once) and `acts` + `finish` (per token). NW packed int32s of weights per call.
struct MultiW { int w[8]; int a = 0, b = 0; float d = 0.f; };

template <int TY> struct Multi { static constexpr bool has = false; static constexpr int NW = 0; };

template <> struct Multi<18> {   // iq3_xxs
    static constexpr bool has = true; static constexpr int NW = 8;
    __dpct_inline__ static void prep(const void* vbq, int kbx, int iqs, MultiW& m) {
        const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
        const sycl::int2 q3_packed = sycl::int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
        const uint8_t* q3 = (const uint8_t*) &q3_packed;
        const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const sycl::int2 grid_pos = sycl::int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
            const int signs0 = dpct::vectorized_binary<sycl::uchar4>(signs & 0x08040201, 0, std::not_equal_to<>());
            const int signs1 = dpct::vectorized_binary<sycl::uchar4>(signs & 0x80402010, 0, std::not_equal_to<>());
            m.w[l0 + 0] = dpct::vectorized_binary<sycl::uchar4>(grid_pos.x() ^ signs0, signs0, std::minus<>());
            m.w[l0 + 1] = dpct::vectorized_binary<sycl::uchar4>(grid_pos.y() ^ signs1, signs1, std::minus<>());
        }
        m.a = (int) (aux32 >> 28);
        m.d = sycl::vec<sycl::half, 1>(bq3->d).convert<float, sycl::rounding_mode::automatic>()[0];
    }
    __dpct_inline__ static void acts(const block_q8_1* y, int iqs, int* u, float& ds) {
#pragma unroll
        for (int j = 0; j < 8; ++j) u[j] = get_int_b4(y[iqs / 2].qs, j);
        ds = y[iqs / 2].ds[0];
    }
    __dpct_inline__ static float finish(int s0, int s1, const MultiW& m, float ds) {
        const int sumi = s0 + s1;
        return m.d * ds * (float) ((m.a * sumi + sumi / 2) / 2);
    }
};

template <> struct Multi<22> {   // iq2_s
    static constexpr bool has = true; static constexpr int NW = 8;
    __dpct_inline__ static void prep(const void* vbq, int kbx, int iqs, MultiW& m) {
        const block_iq2_s* bq2 = (const block_iq2_s*) vbq + kbx;
        const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq2->qh[iqs / 2];
        const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
        const uint8_t* sp = (const uint8_t*) &signs_packed_32;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int* grid_pos = (const int*) (iq2s_grid + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
            const int signs0 = dpct::vectorized_binary<sycl::uchar4>(((sp[l0 / 2] & 0x03) << 7) | ((sp[l0 / 2] & 0x0C) << 21), 0, std::not_equal_to<>());
            const int signs1 = dpct::vectorized_binary<sycl::uchar4>(((sp[l0 / 2] & 0x30) << 3) | ((sp[l0 / 2] & 0xC0) << 17), 0, std::not_equal_to<>());
            m.w[l0 + 0] = dpct::vectorized_binary<sycl::uchar4>(grid_pos[0] ^ signs0, signs0, std::minus<>());
            m.w[l0 + 1] = dpct::vectorized_binary<sycl::uchar4>(grid_pos[1] ^ signs1, signs1, std::minus<>());
        }
        m.a = bq2->scales[iqs / 2] & 0x0F;
        m.b = bq2->scales[iqs / 2] >> 4;
        m.d = sycl::vec<sycl::half, 1>(bq2->d).convert<float, sycl::rounding_mode::automatic>()[0];
    }
    __dpct_inline__ static void acts(const block_q8_1* y, int iqs, int* u, float& ds) {
#pragma unroll
        for (int j = 0; j < 8; ++j) u[j] = get_int_b4(y[iqs / 2].qs, j);
        ds = y[iqs / 2].ds[0];
    }
    __dpct_inline__ static float finish(int s0, int s1, const MultiW& m, float ds) {
        return m.d * ds * (float) ((s0 * m.a + s1 * m.b + (s0 + s1) / 2) / 4);
    }
};

template <> struct Multi<21> {   // iq3_s
    static constexpr bool has = true; static constexpr int NW = 8;
    __dpct_inline__ static void prep(const void* vbq, int kbx, int iqs, MultiW& m) {
        const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
        const sycl::int2 qs_packed = sycl::int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq3->qh[iqs / 2];
        const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
        const uint8_t* sp = (const uint8_t*) &signs_packed_32;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const sycl::int2 grid_pos = sycl::int2(iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                                                   iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
            const int signs0 = dpct::vectorized_binary<sycl::uchar4>(((sp[l0 / 2] & 0x03) << 7) | ((sp[l0 / 2] & 0x0C) << 21), 0, std::not_equal_to<>());
            const int signs1 = dpct::vectorized_binary<sycl::uchar4>(((sp[l0 / 2] & 0x30) << 3) | ((sp[l0 / 2] & 0xC0) << 17), 0, std::not_equal_to<>());
            m.w[l0 + 0] = dpct::vectorized_binary<sycl::uchar4>(grid_pos.x() ^ signs0, signs0, std::minus<>());
            m.w[l0 + 1] = dpct::vectorized_binary<sycl::uchar4>(grid_pos.y() ^ signs1, signs1, std::minus<>());
        }
        m.a = 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
        m.d = sycl::vec<sycl::half, 1>(bq3->d).convert<float, sycl::rounding_mode::automatic>()[0];
    }
    __dpct_inline__ static void acts(const block_q8_1* y, int iqs, int* u, float& ds) {
#pragma unroll
        for (int j = 0; j < 8; ++j) u[j] = get_int_b4(y[iqs / 2].qs, j);
        ds = y[iqs / 2].ds[0];
    }
    __dpct_inline__ static float finish(int s0, int s1, const MultiW& m, float ds) {
        return m.d * ds * (float) ((s0 + s1) * m.a);
    }
};

template <> struct Multi<20> {   // iq4_nl (down)
    static constexpr bool has = true; static constexpr int NW = 4;
    __dpct_inline__ static void prep(const void* vbq, int kbx, int iqs, MultiW& m) {
        const block_iq4_nl* bq4 = (const block_iq4_nl*) vbq + kbx;
#pragma unroll
        for (int l = 0; l < 2; ++l) {
            const int aux_q4 = get_int_b2(bq4->qs, iqs + l);
            const sycl::int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
            m.w[2 * l + 0] = v.x();
            m.w[2 * l + 1] = v.y();
        }
        m.d = sycl::vec<sycl::half, 1>(bq4->d).convert<float, sycl::rounding_mode::automatic>()[0];
    }
    __dpct_inline__ static void acts(const block_q8_1* y, int iqs, int* u, float& ds) {
        const int* q8 = (const int*) y->qs + iqs;
#pragma unroll
        for (int l = 0; l < 2; ++l) { u[2 * l + 0] = q8[l + 0]; u[2 * l + 1] = q8[l + 4]; }
        ds = y->ds[0];
    }
    __dpct_inline__ static float finish(int s0, int s1, const MultiW& m, float ds) { return m.d * ds * (float) (s0 + s1); }
};

template <> struct Multi<42> {   // q2_0 (down)
    static constexpr bool has = true; static constexpr int NW = 8;
    __dpct_inline__ static void prep(const void* vbq, int kbx, int iqs, MultiW& m) {
        const block_q2_0* bq2_0 = (const block_q2_0*) vbq + kbx;
        const int16_t* qs = (const int16_t*) bq2_0->qs + iqs * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = qs[j];
            const int qe = dpct::byte_level_permute(0x020100FF, 0x020100FF, q >> 0);
            const int qo = dpct::byte_level_permute(0x020100FF, 0x020100FF, q >> 2);
            m.w[2 * j + 0] = dpct::byte_level_permute(qe, qo, 0x5140);
            m.w[2 * j + 1] = dpct::byte_level_permute(qe, qo, 0x7362);
        }
        m.d = bq2_0->d;
    }
    __dpct_inline__ static void acts(const block_q8_1* y, int iqs, int* u, float& ds) {
        const block_q8_1* chunk = y + iqs;
#pragma unroll
        for (int j = 0; j < 4; ++j) { u[2 * j + 0] = get_int_b4(chunk->qs, j * 2 + 0); u[2 * j + 1] = get_int_b4(chunk->qs, j * 2 + 1); }
        ds = chunk->ds[0];
    }
    __dpct_inline__ static float finish(int s0, int s1, const MultiW& m, float ds) { return m.d * ds * (float) (s0 + s1); }
};

// One row against E activations at once, LANES lanes per row.
template <int TY, int LANES, int E>
__dpct_inline__ void row_dot_multi(const uint8_t* row, const block_q8_1* const* xs, int nb, int sub, float* out) {
    using F = Fmt<TY>;
    using M = Multi<TY>;
    float s[E];
#pragma unroll
    for (int e = 0; e < E; ++e) s[e] = 0.f;
    for (int k = sub; k < nb * F::ipb; k += LANES) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        MultiW m;
        M::prep(row, kbx, iqs, m);
#pragma unroll
        for (int e = 0; e < E; ++e) {
            int u[8];
            float ds;
            M::acts(xs[e] + kbx * (F::qk / 32), iqs, u, ds);
            int s0 = 0, s1 = 0;
#pragma unroll
            for (int j = 0; j < M::NW / 2; ++j) s0 = ggml_cuda_dp4a(m.w[j], u[j], s0);
#pragma unroll
            for (int j = M::NW / 2; j < M::NW; ++j) s1 = ggml_cuda_dp4a(m.w[j], u[j], s1);
            s[e] += M::finish(s0, s1, m, ds);
        }
    }
#pragma unroll
    for (int e = 0; e < E; ++e) out[e] = lanes_sum<LANES>(s[e]);
}

// The entries e0..e1 of one row: chunks of 4 through the multi-entry dot, the rest one at a time.
template <int TY, int LANES>
__dpct_inline__ void row_entries(const uint8_t* wr, const block_q8_1* x, int x_stride, const int32_t* ent_idx,
                                 int e0, int e1, int nb, int sub, float* dst, size_t dst_stride) {
    int e = e0;
    if constexpr (Multi<TY>::has) {
        for (; e + 4 <= e1; e += 4) {
            const block_q8_1* xs[4] = {x + (size_t) ent_idx[e] * x_stride, x + (size_t) ent_idx[e + 1] * x_stride,
                                       x + (size_t) ent_idx[e + 2] * x_stride, x + (size_t) ent_idx[e + 3] * x_stride};
            float o[4];
            row_dot_multi<TY, LANES, 4>(wr, xs, nb, sub, o);
            if (sub == 0)
#pragma unroll
                for (int i = 0; i < 4; ++i) dst[(size_t) (e + i) * dst_stride] = o[i];
        }
        if (e + 2 <= e1) {
            const block_q8_1* xs[2] = {x + (size_t) ent_idx[e] * x_stride, x + (size_t) ent_idx[e + 1] * x_stride};
            float o[2];
            row_dot_multi<TY, LANES, 2>(wr, xs, nb, sub, o);
            if (sub == 0) { dst[(size_t) e * dst_stride] = o[0]; dst[(size_t) (e + 1) * dst_stride] = o[1]; }
            e += 2;
        }
    }
    for (; e < e1; ++e) {
        const float v = row_dot_lanes<TY, LANES>(wr, x + (size_t) ent_idx[e] * x_stride, nb, sub);
        if (sub == 0) dst[(size_t) e * dst_stride] = v;
    }
}

constexpr int GU_ROWS = 8;     // rows per block (one warp each)

template <int TG>
__dpct_inline__ void native_gu_kernel(
    const unsigned long long *__restrict__ grp_ptr,
    const int32_t *__restrict__ grp_start, const int32_t *__restrict__ n_groups,
    const int32_t *__restrict__ ent_tok, const block_q8_1 *__restrict__ xq,
    NativeExpertLayout L, float *__restrict__ gate, float *__restrict__ up) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int g = item_ct1.get_group(1);
    if (g >= *n_groups) return;
    const int rib = item_ct1.get_local_id(2) / kExpertLanes,
              sub = item_ct1.get_local_id(2) % kExpertLanes;
    const int row = item_ct1.get_group(2) * kExpertRows + rib; // 0 .. 2*n_ff
    if (row >= 2 * L.n_ff) return;
    const bool is_up = row >= L.n_ff;
    const int r = is_up ? row - (int) L.n_ff : row;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint8_t* wr = blob + (is_up ? L.up_off : 0) + (size_t) r * L.gu_row;
    const int nb = (int) (L.n_embd / Fmt<TG>::qk), xb = (int) (L.n_embd / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    row_entries<TG, kExpertLanes>(wr, xq, xb, ent_tok, e0, e1, nb, sub, (is_up ? up : gate) + r, (size_t) L.n_ff);
}

__dpct_inline__ void swiglu_entries_kernel(const float *__restrict__ gate,
                                           const float *__restrict__ up,
                                           float *__restrict__ h, long long n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long i =
        (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n) return;
    const float g = gate[i];
    h[i] = (g / (1.0f + sycl::native::exp(-g))) * up[i];
}

template <int TD>
__dpct_inline__ void native_down_kernel(
    const unsigned long long *__restrict__ grp_ptr,
    const int32_t *__restrict__ grp_start, const int32_t *__restrict__ n_groups,
    const int32_t *__restrict__ ent_dst, const block_q8_1 *__restrict__ hq,
    NativeExpertLayout L, float *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int g = item_ct1.get_group(1);
    if (g >= *n_groups) return;
    const int rib = item_ct1.get_local_id(2) / kExpertLanes,
              sub = item_ct1.get_local_id(2) % kExpertLanes;
    const int r = item_ct1.get_group(2) * kExpertRows + rib;
    if (r >= L.n_embd) return;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint8_t* wr = blob + L.down_off + (size_t) r * L.d_row;
    const int nb = (int) (L.n_ff / Fmt<TD>::qk), hb = (int) (L.n_ff / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    // down: entry e's activation is row e of hq (identity index), its output row is ent_dst[e]
    {
        int e = e0;
        if constexpr (Multi<TD>::has) {
            for (; e + 4 <= e1; e += 4) {
                const block_q8_1* xs[4] = {hq + (size_t) e * hb, hq + (size_t) (e + 1) * hb, hq + (size_t) (e + 2) * hb, hq + (size_t) (e + 3) * hb};
                float o[4];
                row_dot_multi<TD, kExpertLanes, 4>(wr, xs, nb, sub, o);
                if (sub == 0)
#pragma unroll
                    for (int i = 0; i < 4; ++i) out[(size_t) ent_dst[e + i] * L.n_embd + r] = o[i];
            }
            if (e + 2 <= e1) {
                const block_q8_1* xs[2] = {hq + (size_t) e * hb, hq + (size_t) (e + 1) * hb};
                float o[2];
                row_dot_multi<TD, kExpertLanes, 2>(wr, xs, nb, sub, o);
                if (sub == 0) { out[(size_t) ent_dst[e] * L.n_embd + r] = o[0]; out[(size_t) ent_dst[e + 1] * L.n_embd + r] = o[1]; }
                e += 2;
            }
        }
        for (; e < e1; ++e) {
            const float v = row_dot_lanes<TD, kExpertLanes>(wr, hq + (size_t) e * hb, nb, sub);
            if (sub == 0) out[(size_t) ent_dst[e] * L.n_embd + r] = v;
        }
    }
}

// ---------------------------------------------------------------- q8_1 (quantize.cu)
__dpct_inline__ void quantize_q8_1_kernel(const float *__restrict__ x,
                                          block_q8_1 *__restrict__ y,
                                          long long n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long i =
        (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n) return;
    const float xi = x[i];
    float amax = sycl::fabs(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        amax = sycl::fmax(
            amax,
            dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                amax, o));
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        sum += dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            sum, o);
    }
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : sycl::round(xi / d);
    const long long ib = i / 32, iqs = i % 32;
    y[ib].qs[iqs] = q;
    if (iqs == 0) y[ib].ds = sycl::half2(d, sum);
}

// ---------------------------------------------------------------- dequant (dequantize.cuh)
template <typename dst_t> __dpct_inline__ dst_t cvt(float v);
template <> __dpct_inline__ float cvt<float>(float v) { return v; }
template <>
__dpct_inline__ sycl::half cvt<sycl::half>(float v) {
    return sycl::vec<float, 1>(v)
        .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

template <typename dst_t>
inline void dq_iq2_xxs(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_iq2_xxs* x = (const block_iq2_xxs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint8_t* aux8 = (const uint8_t*) q2;
    const uint8_t* grid = (const uint8_t*) (iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
#pragma unroll
    for (int j = 0; j < 8; ++j)
        y[j] = cvt<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}
template <typename dst_t>
inline void dq_iq2_xs(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_iq2_xs* x = (const block_iq2_xs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint8_t* grid = (const uint8_t*) (iq2xs_grid + (q2[il] & 511));
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
#pragma unroll
    for (int j = 0; j < 8; ++j)
        y[j] = cvt<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}
template <typename dst_t>
inline void dq_iq2_s(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_iq2_s* x = (const block_iq2_s*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* grid = (const uint8_t*) (iq2s_grid + (x[ibs].qs[4 * ib + il] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 0x300)));
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[ibs].qs[QK_K / 8 + 4 * ib + il];
#pragma unroll
    for (int j = 0; j < 8; ++j)
        y[j] = cvt<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}
template <typename dst_t>
inline void dq_iq3_xxs(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_iq3_xxs* x = (const block_iq3_xxs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* q3 = x[ibs].qs + 8 * ib;
    const uint16_t* gas = (const uint16_t*) (x[ibs].qs + QK_K / 4) + 2 * ib;
    const uint8_t* grid1 = (const uint8_t*) (iq3xxs_grid + q3[2 * il + 0]);
    const uint8_t* grid2 = (const uint8_t*) (iq3xxs_grid + q3[2 * il + 1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
        y[j + 4] = cvt<dst_t>(d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
    }
}
template <typename dst_t>
inline void dq_iq3_s(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_iq3_s* x = (const block_iq3_s*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* qs = x[ibs].qs + 8 * ib;
    const uint8_t* grid1 = (const uint8_t*) (iq3s_grid + (qs[2 * il + 0] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 256)));
    const uint8_t* grid2 = (const uint8_t*) (iq3s_grid + (qs[2 * il + 1] | ((x[ibs].qh[ib] << (7 - 2 * il)) & 256)));
    const float d = (float) x[ibs].d * (1 + 2 * ((x[ibs].scales[ib / 2] >> 4 * (ib % 2)) & 0xf));
    const uint8_t signs = x[ibs].signs[4 * ib + il];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
        y[j + 4] = cvt<dst_t>(d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
    }
}
template <typename dst_t>
inline void dq_iq1_m(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_iq1_m* x = (const block_iq1_m*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* sc = (const uint16_t*) x[ibs].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int64_t ib16 = 2 * ib + il / 2;
    const float d = (float) scale.f16 * (2 * ((sc[ib16 / 4] >> 3 * (ib16 % 4)) & 0x7) + 1);
    const float delta = x[ibs].qh[2 * ib + il / 2] & (0x08 << 4 * (il % 2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2];
    const int8_t* q = (const int8_t*) grid32;
    grid32[0] = iq1s_grid_gpu[x[ibs].qs[4 * ib + il] | (((x[ibs].qh[2 * ib + il / 2] >> 4 * (il % 2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * (q[j] + delta));
}
template <typename dst_t>
inline void dq_iq4_nl(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_iq4_nl* x = (const block_iq4_nl*) vx + ibs * (QK_K / QK4_NL);
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 4 * il;
    const uint8_t* q4 = x[ib].qs + 4 * il;
    const float d = (float) x[ib].d;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j + 16] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] >> 4]);
    }
}
// Q3_K (the Q2_0 file's token_embd): llama.cpp's dequantize_block_q3_K, its 64 threads folded onto 32
template <typename dst_t>
inline void dq_q3_k(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_q3_K* x = (const block_q3_K*) vx + ibs;
    for (int tt = tid; tt < 64; tt += 32) {
        const int r = tt / 4, t2 = r / 2, is0 = r % 2;
        const int l0 = 16 * is0 + 4 * (tt % 4);
        const int n = t2 / 4, j = t2 - 4 * n;
        const uint8_t m = (uint8_t) (1 << (4 * n + j));
        const int is = 8 * n + 2 * j + is0;
        const int shift = 2 * j;
        const int8_t us = is < 4  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 8] >> 0) & 3) << 4)) :
                          is < 8  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 4] >> 2) & 3) << 4)) :
                          is < 12 ? (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is + 0] >> 4) & 3) << 4)) :
                                    (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is - 4] >> 6) & 3) << 4));
        const float dl = (float) x->d * (us - 32);
        dst_t* y = yy + 128 * n + 32 * j;
        const uint8_t* q = x->qs + 32 * n;
        const uint8_t* hm = x->hmask;
#pragma unroll
        for (int l = l0; l < l0 + 4; ++l) y[l] = cvt<dst_t>(
            dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4)));
    }
}
template <typename dst_t>
inline void dq_iq4_xs(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_iq4_xs* x = (const block_iq4_xs*) vx + ibs;
    const int il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 4 * il;
    const uint8_t* q4 = x->qs + 16 * ib + 4 * il;
    const float d = (float) x->d * ((((x->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x->scales_h >> 2 * ib) & 3) << 4)) - 32);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j + 16] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] >> 4]);
    }
}
template <typename dst_t>
inline void dq_q2_0(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    // one "superblock" = 256 values = 4 blocks of 64; thread tid writes 8 values
    const block_q2_0* x = (const block_q2_0*) vx + ibs * 4;
    const int b = tid / 8, part = tid % 8;          // block 0..3, 8 values each
    const float d = (float) x[b].d;
    for (int j = 0; j < 8; ++j) {
        const int i = part * 8 + j;
        const int code = (x[b].qs[i / 4] >> ((i % 4) * 2)) & 3;
        yy[b * 64 + i] = cvt<dst_t>(d * (float) (code - 1));
    }
}

template <typename dst_t>
__dpct_inline__ void
dq_dispatch(int ty, const void *vx, int64_t ibs, dst_t *y, int tid) {
    switch (ty) {
        case 16: dq_iq2_xxs(vx, ibs, y, tid);
            break;
        case 17: dq_iq2_xs(vx, ibs, y, tid);
            break;
        case 18: dq_iq3_xxs(vx, ibs, y, tid);
            break;
        case 20: dq_iq4_nl(vx, ibs, y, tid); break;
        case 21: dq_iq3_s(vx, ibs, y, tid); break;
        case 22: dq_iq2_s(vx, ibs, y, tid); break;
        case 29: dq_iq1_m(vx, ibs, y, tid); break;
        case 23: dq_iq4_xs(vx, ibs, y, tid); break;
        case 11: dq_q3_k(vx, ibs, y, tid); break;
        case 42: dq_q2_0(vx, ibs, y, tid); break;
        default: break;
    }
}

// flat: superblock i -> y + 256 i
template <typename dst_t>
__dpct_inline__ void dequant_flat_kernel(
    int ty, const void *__restrict__ vx, dst_t *__restrict__ y) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i = item_ct1.get_group(2);
    dq_dispatch<dst_t>(ty, vx, i, y + i * QK_K, item_ct1.get_local_id(2));
}
// gate/up: superblock i of a role matrix (n_embd/256 per row) -> interleaved row 2r + parity
__dpct_inline__ void dequant_gu_kernel(
    int ty, const void *__restrict__ gate, const void *__restrict__ up,
    int64_t per_row, sycl::half *__restrict__ y) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i = item_ct1.get_group(2);
    const int parity = item_ct1.get_group(1);
    const int64_t r = i / per_row, c = i % per_row;
    dq_dispatch<sycl::half>(ty, parity ? up : gate, i,
                            y + ((2 * r + parity) * per_row + c) * QK_K,
                            item_ct1.get_local_id(2));
}

bool is_iq(int t) { return t == 16 || t == 17 || t == 18 || t == 20 || t == 21 || t == 22 || t == 23 || t == 29 || t == 42 || t == 11; }

}  // namespace

bool iq_supported(int t) noexcept { return is_iq(t); }

size_t iq_row_bytes(int t, int64_t n) noexcept {
    switch (t) {
        case 16: return (size_t) (n / 256) * sizeof(block_iq2_xxs);
        case 17: return (size_t) (n / 256) * sizeof(block_iq2_xs);
        case 18: return (size_t) (n / 256) * sizeof(block_iq3_xxs);
        case 20: return (size_t) (n / 32) * sizeof(block_iq4_nl);
        case 21: return (size_t) (n / 256) * sizeof(block_iq3_s);
        case 22: return (size_t) (n / 256) * sizeof(block_iq2_s);
        case 29: return (size_t) (n / 256) * sizeof(block_iq1_m);
        case 23: return (size_t) (n / 256) * sizeof(block_iq4_xs);
        case 11: return (size_t) (n / 256) * sizeof(block_q3_K);
        case 42: return (size_t) (n / 64) * sizeof(block_q2_0);
        default: return 0;
    }
}

void quantize_q8_1_rows(const float* x, int64_t n_rows, int64_t n_cols, void* y, void* stream) {
    const long long n = (long long) n_rows * n_cols;
    if (n <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class quantize_q8_1_kernel_a566ca>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, (unsigned)((n + 255) / 256)) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        quantize_q8_1_kernel(x, (block_q8_1 *)y, n);
                    });
    }
    check("quantize_q8_1_rows");
}

void iq_mmvq(int t, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    const dpct::dim3 grid((unsigned)((n_out + 3) / 4)), block(32, 4);
    const size_t rb = iq_row_bytes(t, n_in);
    dpct::queue_ptr s = strata::q_of(stream);
    const auto* W = (const uint8_t*) w;
    const auto* X = (const block_q8_1*) x_q8_1;
    switch (t) {
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    case 16: {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<class mmvq_kernel_2769e1,
                                              dpct_kernel_scalar<16>>>(
                sycl::nd_range<3>(grid * block, block), exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        mmvq_kernel<16>(W, rb, X, y, n_in, n_out, ncols);
                    });
        });
    } break;
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    case 17: {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<class mmvq_kernel_4ee329,
                                              dpct_kernel_scalar<17>>>(
                sycl::nd_range<3>(grid * block, block), exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        mmvq_kernel<17>(W, rb, X, y, n_in, n_out, ncols);
                    });
        });
    } break;
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    case 18: {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<class mmvq_kernel_34dcfe,
                                              dpct_kernel_scalar<18>>>(
                sycl::nd_range<3>(grid * block, block), exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        mmvq_kernel<18>(W, rb, X, y, n_in, n_out, ncols);
                    });
        });
    } break;
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    case 20: {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<class mmvq_kernel_dfe57c,
                                              dpct_kernel_scalar<20>>>(
                sycl::nd_range<3>(grid * block, block), exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        mmvq_kernel<20>(W, rb, X, y, n_in, n_out, ncols);
                    });
        });
    } break;
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    case 21: {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<class mmvq_kernel_9d8f61,
                                              dpct_kernel_scalar<21>>>(
                sycl::nd_range<3>(grid * block, block), exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        mmvq_kernel<21>(W, rb, X, y, n_in, n_out, ncols);
                    });
        });
    } break;
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    case 22: {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<class mmvq_kernel_979367,
                                              dpct_kernel_scalar<22>>>(
                sycl::nd_range<3>(grid * block, block), exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        mmvq_kernel<22>(W, rb, X, y, n_in, n_out, ncols);
                    });
        });
    } break;
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    case 23: {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<class mmvq_kernel_15e579,
                                              dpct_kernel_scalar<23>>>(
                sycl::nd_range<3>(grid * block, block), exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        mmvq_kernel<23>(W, rb, X, y, n_in, n_out, ncols);
                    });
        });
    } break;
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    case 29: {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<class mmvq_kernel_bc4076,
                                              dpct_kernel_scalar<29>>>(
                sycl::nd_range<3>(grid * block, block), exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        mmvq_kernel<29>(W, rb, X, y, n_in, n_out, ncols);
                    });
        });
    } break;
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    case 42: {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<class mmvq_kernel_2c3d4c,
                                              dpct_kernel_scalar<42>>>(
                sycl::nd_range<3>(grid * block, block), exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        mmvq_kernel<42>(W, rb, X, y, n_in, n_out, ncols);
                    });
        });
    } break;
        default: std::fprintf(stderr, "iq_mmvq: type %d is not supported\n", t); std::exit(1);
    }
    check("iq_mmvq");
}

void iq_dequant_f16(int t, const void* src, int64_t n, uint16_t* dst, void* stream) {
    if (n % 256 != 0 || !is_iq(t)) { std::fprintf(stderr, "iq_dequant_f16: bad arguments\n"); std::exit(1); }
    {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(
            strata::q_of(stream)->get_device(),
            {sycl::aspect::fp16});

        strata::q_of(stream)
            ->submit([&](sycl::handler &cgh) {

                cgh.parallel_for<dpct_kernel_name<
                    class dequant_flat_kernel_d891c8, sycl::half>>(
                    sycl::nd_range<3>(sycl::range(1, 1, (unsigned)(n / 256)) *
                                          sycl::range(1, 1, 32),
                                      sycl::range(1, 1, 32)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        dequant_flat_kernel<sycl::half>(
                            t, src, (sycl::half *)dst);
                    });
            });
    }
    check("iq_dequant_f16");
}

namespace {
__dpct_inline__ void embed_rows_kernel(
    int ty, const uint8_t *__restrict__ table, size_t row_bytes,
    const int32_t *__restrict__ tokens, int64_t n_embd, float *__restrict__ y) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1);
    const int64_t b = item_ct1.get_group(2);
    const uint8_t* row = table + (size_t) tokens[t] * row_bytes;
    dq_dispatch<float>(ty, row, b, y + (size_t)t * n_embd + b * QK_K,
                       item_ct1.get_local_id(2));
}
}  // namespace

void iq_embed_rows(int t, const void* table, size_t row_bytes, const int32_t* tokens, int64_t n_tok, int64_t n_embd,
                   float* out, void* stream) {
    if (n_tok <= 0) return;
    if (n_embd % 256 != 0 || !is_iq(t)) { std::fprintf(stderr, "iq_embed_rows: bad arguments\n"); std::exit(1); }
    {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(
            strata::q_of(stream)->get_device(),
            {sycl::aspect::fp16});

        strata::q_of(stream)
            ->submit([&](sycl::handler &cgh) {

                cgh.parallel_for<
                    dpct_kernel_name<class embed_rows_kernel_284a43>>(
                    sycl::nd_range<3>(sycl::range(1, (unsigned)n_tok,
                                                  (unsigned)(n_embd / 256)) *
                                          sycl::range(1, 1, 32),
                                      sycl::range(1, 1, 32)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        embed_rows_kernel(
                            t, (const uint8_t *)table, row_bytes, tokens,
                            n_embd, out);
                    });
            });
    }
    check("iq_embed_rows");
}

void iq_dequant_f32(int t, const void* src, int64_t n, float* dst, void* stream) {
    if (n % 256 != 0 || !is_iq(t)) { std::fprintf(stderr, "iq_dequant_f32: bad arguments\n"); std::exit(1); }
    {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(
            strata::q_of(stream)->get_device(),
            {sycl::aspect::fp16});

        strata::q_of(stream)
            ->submit([&](sycl::handler &cgh) {

                cgh.parallel_for<
                    dpct_kernel_name<class dequant_flat_kernel_d7b5ca, float>>(
                    sycl::nd_range<3>(sycl::range(1, 1, (unsigned)(n / 256)) *
                                          sycl::range(1, 1, 32),
                                      sycl::range(1, 1, 32)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        dequant_flat_kernel<float>(
                            t, src, dst);
                    });
            });
    }
    check("iq_dequant_f32");
}

void iq_dequant_gu_f16(int t, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* dst, void* stream) {
    const int64_t per_row = n_embd / 256;
    {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(
            strata::q_of(stream)->get_device(),
            {sycl::aspect::fp16});

        strata::q_of(stream)
            ->submit([&](sycl::handler &cgh) {

                cgh.parallel_for<
                    dpct_kernel_name<class dequant_gu_kernel_2397ca>>(
                    sycl::nd_range<3>(
                        sycl::range(1, 2, (unsigned)(n_ff * per_row)) *
                            sycl::range(1, 1, 32),
                        sycl::range(1, 1, 32)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        dequant_gu_kernel(
                            t, gate, up, per_row, (sycl::half *)dst);
                    });
            });
    }
    check("iq_dequant_gu_f16");
}

NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) {
    NativeExpertLayout L;
    L.gu_type = gu_type;
    L.d_type = d_type;
    L.n_embd = n_embd;
    L.n_ff = n_ff;
    L.gu_row = iq_row_bytes(gu_type, n_embd);
    L.d_row = iq_row_bytes(d_type, n_ff);
    L.up_off = (size_t) n_ff * L.gu_row;
    L.down_off = 2 * L.up_off;
    L.bytes = L.down_off + (size_t) n_embd * L.d_row;
    return L;
}

size_t native_expert_scratch_bytes(int64_t cap, int64_t n_ff) {
    const size_t f = (size_t) cap * (size_t) n_ff * sizeof(float);
    return 3 * ((f + 255) & ~(size_t) 255) + (((size_t) cap * (size_t) (n_ff / 32) * sizeof(block_q8_1) + 255) & ~(size_t) 255);
}

void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                           const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                           int64_t cap_entries, const void* x_q8_1, void* scratch, float* out, void* stream) {
    if (cap_groups <= 0 || cap_entries <= 0) return;
    dpct::queue_ptr s = strata::q_of(stream);
    const size_t f = (size_t) cap_entries * (size_t) L.n_ff * sizeof(float), fa = (f + 255) & ~(size_t) 255;
    float* gate = (float*) scratch;
    float* up = (float*) ((uint8_t*) scratch + fa);
    float* h = (float*) ((uint8_t*) scratch + 2 * fa);
    block_q8_1* hq = (block_q8_1*) ((uint8_t*) scratch + 3 * fa);
    const auto* X = (const block_q8_1*) x_q8_1;
    const dpct::dim3 ggu((unsigned)((2 * L.n_ff + kExpertRows - 1) / kExpertRows),
                         (unsigned)cap_groups);
    switch (L.gu_type) {
    case 16: {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<class native_gu_kernel_472f9a,
                                              dpct_kernel_scalar<16>>>(
                sycl::nd_range<3>(ggu * sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_gu_kernel<16>(grp_ptr, grp_start, n_groups,
                                             ent_tok, X, L, gate, up);
                    });
        });
    } break;
    case 17: {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<class native_gu_kernel_3af217,
                                              dpct_kernel_scalar<17>>>(
                sycl::nd_range<3>(ggu * sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_gu_kernel<17>(grp_ptr, grp_start, n_groups,
                                             ent_tok, X, L, gate, up);
                    });
        });
    } break;
    case 18: {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<class native_gu_kernel_17c573,
                                              dpct_kernel_scalar<18>>>(
                sycl::nd_range<3>(ggu * sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_gu_kernel<18>(grp_ptr, grp_start, n_groups,
                                             ent_tok, X, L, gate, up);
                    });
        });
    } break;
    case 21: {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<class native_gu_kernel_1f1e65,
                                              dpct_kernel_scalar<21>>>(
                sycl::nd_range<3>(ggu * sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_gu_kernel<21>(grp_ptr, grp_start, n_groups,
                                             ent_tok, X, L, gate, up);
                    });
        });
    } break;
    case 22: {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<class native_gu_kernel_5b900c,
                                              dpct_kernel_scalar<22>>>(
                sycl::nd_range<3>(ggu * sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_gu_kernel<22>(grp_ptr, grp_start, n_groups,
                                             ent_tok, X, L, gate, up);
                    });
        });
    } break;
    case 23: {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<class native_gu_kernel_ca9850,
                                              dpct_kernel_scalar<23>>>(
                sycl::nd_range<3>(ggu * sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_gu_kernel<23>(grp_ptr, grp_start, n_groups,
                                             ent_tok, X, L, gate, up);
                    });
        });
    } break;
    case 29: {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<class native_gu_kernel_5ad3a4,
                                              dpct_kernel_scalar<29>>>(
                sycl::nd_range<3>(ggu * sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_gu_kernel<29>(grp_ptr, grp_start, n_groups,
                                             ent_tok, X, L, gate, up);
                    });
        });
    } break;
    case 42: {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<class native_gu_kernel_331601,
                                              dpct_kernel_scalar<42>>>(
                sycl::nd_range<3>(ggu * sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_gu_kernel<42>(grp_ptr, grp_start, n_groups,
                                             ent_tok, X, L, gate, up);
                    });
        });
    } break;
        default: std::fprintf(stderr, "native_expert_grouped: gate/up type %d\n", L.gu_type); std::exit(1);
    }
    check("native_expert_grouped/gu");
    const long long nh = (long long) cap_entries * L.n_ff;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class swiglu_entries_kernel_a4959c>>(
            sycl::nd_range<3>(sycl::range(1, 1, (unsigned)((nh + 255) / 256)) *
                                  sycl::range(1, 1, 256),
                              sycl::range(1, 1, 256)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                swiglu_entries_kernel(gate, up, h, nh);
            });
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class quantize_q8_1_kernel_3fc248>>(
            sycl::nd_range<3>(sycl::range(1, 1, (unsigned)((nh + 255) / 256)) *
                                  sycl::range(1, 1, 256),
                              sycl::range(1, 1, 256)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                quantize_q8_1_kernel(h, hq, nh);
            });
    }
    const dpct::dim3 gd((unsigned)((L.n_embd + kExpertRows - 1) / kExpertRows), (unsigned)cap_groups);
    switch (L.d_type) {
    case 20: {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<class native_down_kernel_f2a0ef,
                                              dpct_kernel_scalar<20>>>(
                sycl::nd_range<3>(gd * sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_down_kernel<20>(grp_ptr, grp_start, n_groups,
                                               ent_dst, hq, L, out);
                    });
        });
    } break;
    case 23: {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<class native_down_kernel_cf19c8,
                                              dpct_kernel_scalar<23>>>(
                sycl::nd_range<3>(gd * sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_down_kernel<23>(grp_ptr, grp_start, n_groups,
                                               ent_dst, hq, L, out);
                    });
        });
    } break;
    case 42: {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<class native_down_kernel_57a5b9,
                                              dpct_kernel_scalar<42>>>(
                sycl::nd_range<3>(gd * sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_down_kernel<42>(grp_ptr, grp_start, n_groups,
                                               ent_dst, hq, L, out);
                    });
        });
    } break;
        default: std::fprintf(stderr, "native_expert_grouped: down type %d\n", L.d_type); std::exit(1);
    }
    check("native_expert_grouped/down");
}

}  // namespace strata::kernels
