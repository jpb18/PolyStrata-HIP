// src/glm/mmq.cu - see the header.  The kernels are compiled from the llama.cpp checkout the build already takes
// ggml from; src/prefill/ggml_cuda_host.cu supplies the host symbols of ggml-cuda.cu they use.
#include "mmq.hpp"

#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"

#include <stdexcept>
#include <string>

namespace strata::glm {
namespace {

int64_t pad512(int64_t n) { return (n + 511) / 512 * 512; }

void ck(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

}  // namespace

Mmq::Mmq() { ctx_ = new ggml_backend_cuda_context(ggml_cuda_get_device()); }

Mmq::~Mmq() { delete (ggml_backend_cuda_context*) ctx_; }

// mul_mat_q_case's own choice: a tile size whose configuration exists for this card and fits its shared memory.
// It aborts the process when there is none.
bool Mmq::fits(int type, int64_t w_rows) {
    const ggml_type t = (ggml_type) type;
    if (t != GGML_TYPE_Q2_K && t != GGML_TYPE_Q3_K && t != GGML_TYPE_Q4_K) return false;
    const bool fallback = w_rows % 128 != 0;
    const auto& dev = ggml_cuda_info().devices[ggml_cuda_get_device()];
    for (int J = 8; J <= 128; J += 8) {
        const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(t, J, fallback, dev.cc);
        if (c.type != GGML_TYPE_COUNT && mmq_get_nbytes_shared(c, dev.cc) <= dev.smpbo) return true;
    }
    return false;
}

// the kernels read whole tiles, so the last rows need room behind them
size_t Mmq::q8_bytes(int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) (pad512(cols) / QK8_1_MMQ) * sizeof(block_q8_1_mmq) + 128 * sizeof(block_q8_1_mmq);
}

void Mmq::quantize(const float* x, const int32_t* ids, void* xq, int type, int64_t cols, int64_t ld, int64_t rows,
                   void* stream) {
    if (rows <= 0) return;
    quantize_mmq_q8_1_cuda(x, ids, xq, (ggml_type) type, cols, ld, rows * ld, rows * ld, pad512(cols), rows, 1, 1,
                           (cudaStream_t) stream);
    ck("mmq quantize");
}

void Mmq::mul(const void* w, int type, int64_t w_rows, int64_t w_cols, size_t stride, int n, const void* xq,
              const int32_t* bounds, const int32_t* ids, int64_t total_rows, int64_t max_rows, float* dst,
              int64_t ld_dst, void* stream) {
    if (n <= 0 || max_rows <= 0) return;
    const ggml_type t = (ggml_type) type;
    // the matrices are the channels of x; the tokens' rows are the columns of y and of dst
    mmq_args a = {};
    a.x = (const char*) w;
    a.type_x = t;
    a.y = (const int*) xq;
    a.ids_dst = ids;
    a.expert_bounds = bounds;
    a.dst = dst;
    a.ncols_x = w_cols;
    a.nrows_x = w_rows;
    a.stride_row_x = w_cols / ggml_blck_size(t);
    a.ncols_y = total_rows;
    a.ncols_dst = total_rows;
    a.nrows_dst = ld_dst;
    a.nchannels_x = n;
    a.nchannels_y = n;
    a.stride_channel_x = (int64_t) (stride / ggml_type_size(t));
    a.nsamples_x = 1;
    a.nsamples_y = 1;
    a.ncols_max = max_rows;
    a.ncols_opt = max_rows;

    auto& ctx = *(ggml_backend_cuda_context*) ctx_;
    const cudaStream_t s = (cudaStream_t) stream;
    switch (t) {
        case GGML_TYPE_Q2_K: mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, a, s); break;
        case GGML_TYPE_Q3_K: mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, a, s); break;
        case GGML_TYPE_Q4_K: mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, a, s); break;
        default: throw std::runtime_error("mmq: tensor type " + std::to_string(type) + " is not built");
    }
    ck("mmq product");
}

}  // namespace strata::glm
