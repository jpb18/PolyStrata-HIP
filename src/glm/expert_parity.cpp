// src/glm/expert_parity.cpp - a routed expert computed three ways, on the model's own weights.
//
//   glm_expert_parity MODEL.gguf
//
// For a few experts of every tensor type the file uses, the GPU path (native_mmvq over the GGUF blocks) and the
// CPU path (cpu_experts) are compared with a double-precision product over ggml's dequantized weights.  Both
// quantize their input to 8 bits, so neither is exact; the test fails when one of them is far from the reference.
#include "cpu_experts.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/glm/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using namespace strata;

void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

// rows x n_in weights in the GGUF's blocks, as floats
std::vector<float> dequant(uint32_t type, const uint8_t* w, int n_in, int rows) {
    std::vector<float> out((size_t) rows * n_in);
    ggml_get_type_traits((ggml_type) type)->to_float(w, out.data(), (int64_t) rows * n_in);
    return out;
}

std::vector<double> matvec(const std::vector<float>& w, const std::vector<double>& x, int n_in, int rows) {
    std::vector<double> y((size_t) rows);
    for (int r = 0; r < rows; ++r) {
        double acc = 0.0;
        for (int i = 0; i < n_in; ++i) acc += (double) w[(size_t) r * n_in + i] * x[(size_t) i];
        y[(size_t) r] = acc;
    }
    return y;
}

double rel_error(const float* got, const std::vector<double>& want) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
        num += ((double) got[i] - want[i]) * ((double) got[i] - want[i]);
        den += want[i] * want[i];
    }
    return std::sqrt(num / den);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: glm_expert_parity MODEL.gguf\n");
        return 2;
    }
    try {
        GgufFile g(argv[1]);
        const auto meta = [&](const char* k) {
            const MetaValue* v = g.get(std::string("glm5-next.") + k);
            if (!v) throw std::runtime_error(std::string("the GGUF has no glm5-next.") + k);
            return (int) v->u;
        };
        const int n_layer = meta("block_count"), n_embd = meta("embedding_length"),
                  n_ff = meta("expert_feed_forward_length"), n_expert = meta("expert_count");
        const float limit = 10.f;

        // the first layer of every (gate/up type, down type) pair
        std::map<std::pair<uint32_t, uint32_t>, int> layers;
        for (int il = 0; il < n_layer; ++il) {
            const TensorInfo* tg = g.find("blk." + std::to_string(il) + ".ffn_gate_exps.weight");
            const TensorInfo* td = g.find("blk." + std::to_string(il) + ".ffn_down_exps.weight");
            if (tg && td) layers.emplace(std::make_pair(tg->type, td->type), il);
        }
        if (layers.empty()) throw std::runtime_error("the GGUF has no routed experts");

        // a residual-like input: mostly small, a few large values
        std::mt19937 rng(1234);
        std::normal_distribution<float> small(0.f, 0.05f);
        std::vector<float> x((size_t) n_embd);
        for (float& v : x) v = small(rng);
        for (int i = 0; i < n_embd; i += 97) x[(size_t) i] *= 20.f;
        const std::vector<double> xd(x.begin(), x.end());

        cudaStream_t st = nullptr;
        ck(cudaStreamCreate(&st));
        float *d_x, *d_gu, *d_h, *d_out;
        void *d_xq, *d_hq;
        ck(cudaMalloc(&d_x, (size_t) n_embd * 4));
        ck(cudaMalloc(&d_gu, (size_t) 2 * n_ff * 4));
        ck(cudaMalloc(&d_h, (size_t) n_ff * 4));
        ck(cudaMalloc(&d_out, (size_t) n_embd * 4));
        ck(cudaMalloc(&d_xq, kernels::native_q8_1_bytes(n_embd)));
        ck(cudaMalloc(&d_hq, kernels::native_q8_1_bytes(n_ff)));
        ck(cudaMemcpy(d_x, x.data(), (size_t) n_embd * 4, cudaMemcpyHostToDevice));
        kernels::native_quantize_q8_1(d_x, d_xq, n_embd, 1, st);

        glm::Workers workers(8);
        bool ok = true;
        for (const auto& [types, il] : layers) {
            const std::string b = "blk." + std::to_string(il) + ".";
            const TensorInfo& tg = *g.find(b + "ffn_gate_exps.weight");
            const TensorInfo& tu = *g.find(b + "ffn_up_exps.weight");
            const TensorInfo& td = *g.find(b + "ffn_down_exps.weight");
            const size_t gu_bytes = tensor_payload_bytes(tg) / (size_t) n_expert,
                         d_bytes = tensor_payload_bytes(td) / (size_t) n_expert;
            std::string err;
            if (!glm::cpu_expert_types_ok((int) tg.type, (int) td.type, n_embd, n_ff, err)) throw std::runtime_error(err);
            const bool gpu = kernels::native_mmvq_supported((int) tg.type) && kernels::native_mmvq_supported((int) td.type);
            uint8_t* d_w = nullptr;
            ck(cudaMalloc(&d_w, 2 * gu_bytes + d_bytes + 256));

            for (int e : {0, n_expert / 2, n_expert - 1}) {
                const uint8_t* gate = g.tensor_data(tg) + (size_t) e * gu_bytes;
                const uint8_t* up = g.tensor_data(tu) + (size_t) e * gu_bytes;
                const uint8_t* down = g.tensor_data(td) + (size_t) e * d_bytes;

                const std::vector<double> gv = matvec(dequant(tg.type, gate, n_embd, n_ff), xd, n_embd, n_ff);
                const std::vector<double> uv = matvec(dequant(tu.type, up, n_embd, n_ff), xd, n_embd, n_ff);
                std::vector<double> h((size_t) n_ff);
                for (int i = 0; i < n_ff; ++i) {
                    const double gg = std::min(gv[(size_t) i], (double) limit);
                    h[(size_t) i] = gg / (1.0 + std::exp(-gg)) * std::clamp(uv[(size_t) i], (double) -limit, (double) limit);
                }
                const std::vector<double> want = matvec(dequant(td.type, down, n_ff, n_embd), h, n_ff, n_embd);

                std::vector<float> cpu((size_t) n_embd), dev((size_t) n_embd);
                const glm::CpuExpert ce{gate, up, down, 1.f};
                glm::cpu_experts(workers, &ce, 1, (int) tg.type, (int) td.type, n_embd, n_ff, x.data(), limit, cpu.data());
                const double e_cpu = rel_error(cpu.data(), want);
                double e_gpu = 0.0;
                if (gpu) {
                    ck(cudaMemcpy(d_w, gate, gu_bytes, cudaMemcpyHostToDevice));
                    ck(cudaMemcpy(d_w + gu_bytes, up, gu_bytes, cudaMemcpyHostToDevice));
                    ck(cudaMemcpy(d_w + 2 * gu_bytes, down, d_bytes, cudaMemcpyHostToDevice));
                    kernels::native_mmvq((int) tg.type, d_w, d_xq, d_gu, n_embd, 2 * n_ff, 1, st);
                    glm::swiglu_clamp(d_gu, d_gu + n_ff, d_h, n_ff, limit, st);
                    kernels::native_quantize_q8_1(d_h, d_hq, n_ff, 1, st);
                    kernels::native_mmvq((int) td.type, d_w + 2 * gu_bytes, d_hq, d_out, n_ff, n_embd, 1, st);
                    ck(cudaMemcpyAsync(dev.data(), d_out, (size_t) n_embd * 4, cudaMemcpyDeviceToHost, st));
                    ck(cudaStreamSynchronize(st));
                    e_gpu = rel_error(dev.data(), want);
                }
                const bool pass = e_cpu < 0.05 && e_gpu < 0.05;
                ok = ok && pass;
                std::printf("blk.%d expert %3d  %s/%s  relative error: CPU %.4f, GPU %s%s\n", il, e, tg.type_name(),
                            td.type_name(), e_cpu, gpu ? std::to_string(e_gpu).c_str() : "(no kernel)",
                            pass ? "" : "  FAIL");
            }
            ck(cudaFree(d_w));
        }
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "glm_expert_parity: %s\n", e.what());
        return 1;
    }
}
