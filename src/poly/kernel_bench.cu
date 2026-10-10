// src/poly/kernel_bench.cu - the time of each kernel a token of Qwen3.5/3.6-35B-A3B runs through, at that model's
// shapes, on random data and without a model (STRATA_POLY_HIP only; AMD_PORT_IMPLEMENTATION.md).
//
//   poly_kernel_bench [reps]
//
// Each kernel is launched `reps` times back to back (default 400) and the card waited for once: the figure is the
// host time a call, which is what a layer's chain of them costs, launch gaps included.  For a matrix it also says
// the bytes of the weights read a second.  The values are random: the times, not the results, are the point.
#include "strata/kernels/native_mmvq.hpp"
#include "strata/poly/kernels.hpp"
#include "mmq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using namespace strata;

int g_reps = 400;
cudaStream_t g_st = nullptr;

void* dev_random(size_t bytes, bool floats) {
    std::vector<uint8_t> h(bytes);
    std::mt19937 rng(1234);
    if (floats) {
        std::uniform_real_distribution<float> d(-1.f, 1.f);
        for (size_t i = 0; i + 4 <= bytes; i += 4) {
            const float v = d(rng) * 0.1f;
            std::memcpy(&h[i], &v, 4);
        }
    } else {
        for (auto& b : h) b = (uint8_t) rng();
    }
    void* p = nullptr;
    if (cudaMalloc(&p, bytes + 4096) != cudaSuccess) {
        std::printf("no VRAM for %zu bytes\n", bytes);
        std::exit(1);
    }
    cudaMemset(p, 0, bytes + 4096);
    cudaMemcpy(p, h.data(), bytes, cudaMemcpyHostToDevice);
    return p;
}

// fp16 block scales of random bytes can be infinities or NaNs; a small scale in every block keeps the sums finite
void tame_scales(void* w, size_t bytes, int block_bytes, int scale_at) {
    std::vector<uint8_t> h(bytes);
    cudaMemcpy(h.data(), w, bytes, cudaMemcpyDeviceToHost);
    for (size_t b = 0; b + block_bytes <= bytes; b += (size_t) block_bytes) {
        h[b + scale_at] = 0x00;
        h[b + scale_at + 1] = 0x1c;   // fp16 0.0039
    }
    cudaMemcpy(w, h.data(), bytes, cudaMemcpyHostToDevice);
}

double time_us(const std::function<void()>& f) {
    for (int i = 0; i < 20; ++i) f();
    cudaStreamSynchronize(g_st);
    const Clock::time_point t0 = Clock::now();
    for (int i = 0; i < g_reps; ++i) f();
    cudaStreamSynchronize(g_st);
    const double us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / g_reps;
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::printf("launch failed: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return us;
}

void line(const char* what, double us, double bytes = 0) {
    if (bytes > 0)
        std::printf("  %-52s %8.2f us  %6.1f GB/s\n", what, us, bytes / (us * 1e3));
    else
        std::printf("  %-52s %8.2f us\n", what, us);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1) g_reps = std::max(10, std::atoi(argv[1]));
    cudaStreamCreateWithFlags(&g_st, cudaStreamNonBlocking);
    void* st = g_st;
    const int E = 2048;   // n_embd
    float* x = (float*) dev_random(16384 * 4, true);
    float* y = (float*) dev_random(262144 * 4, true);
    void* xq = dev_random(strata::kernels::native_q8_1_bytes(8192) * 8, false);
    strata::kernels::native_quantize_q8_1(x, xq, 8192, 1, st);

    {   // a process's first call of each library path a prompt batch takes, and the second: what the first costs once
        const auto once = [&](const char* what, const std::function<void()>& f) {
            Clock::time_point t0 = Clock::now();
            f();
            cudaStreamSynchronize(g_st);
            const double first = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            t0 = Clock::now();
            f();
            cudaStreamSynchronize(g_st);
            const double second = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            std::printf("  first call %-46s %9.1f ms, then %7.2f ms\n", what, first, second);
        };
        std::printf("a process's first calls\n");
        void* blas = poly::blas_create(st);
        float* w = (float*) dev_random((size_t) E * 256 * 4, true);
        float* xs = (float*) dev_random((size_t) E * 1024 * 4, true);
        float* ys = (float*) dev_random((size_t) 256 * 1024 * 4, true);
        once("hipBLAS SGEMM 2048 x 256, 1024 rows", [&] { poly::gemm_f32(blas, w, xs, ys, E, 256, 256, 1024); });
        poly::Mmq mmq;
        void* xq8 = dev_random(poly::Mmq::q8_bytes(1024, 4096), false);
        const struct {
            int type, block_elems, block_bytes;
            const char* name;
        } mm[] = {{8, 32, 34, "MMQ Q8_0 2048 -> 8192, 1024 rows"}, {18, 256, 98, "MMQ IQ3_XXS 2048 -> 1024"},
                  {23, 256, 136, "MMQ IQ4_XS 512 -> 2048"}, {14, 256, 210, "MMQ Q6_K 2048 -> 2048"}};
        for (const auto& m : mm) {
            const int n_in = m.type == 23 ? 512 : 2048, n_out = m.type == 8 ? 8192 : m.type == 23 ? 2048 :
                                                                   m.type == 14 ? 2048 : 1024;
            const size_t wb = (size_t) n_out * (n_in / m.block_elems) * m.block_bytes;
            void* wq = dev_random(wb + 4096, false);
            tame_scales(wq, wb, m.block_bytes, m.type == 14 ? 208 : 0);
            float* out = (float*) dev_random((size_t) n_out * 1024 * 4, true);
            once(m.name, [&] {
                poly::Mmq::quantize(xs, nullptr, xq8, m.type, n_in, n_in, 1024, st);
                mmq.mul(wq, m.type, n_out, n_in, 0, 1, xq8, nullptr, nullptr, 1024, 1024, out, n_out, st);
            });
            cudaFree(wq);
            cudaFree(out);
        }
        poly::blas_destroy(blas);
    }
    std::printf("one token of Qwen3.5/3.6-35B-A3B, %d launches each\n", g_reps);
    line("quantize 2048 floats to q8_1", time_us([&] { kernels::native_quantize_q8_1(x, xq, E, 1, st); }));
    float* nw = (float*) dev_random(E * 4, true);
    line("rms_norm 2048", time_us([&] { poly::rms_norm(x, nw, y, E, 1, 1e-6f, st); }));
    line("axpy 2048", time_us([&] { poly::axpy(x, 1.f, y, E, true, st); }));

    // Q8_0 matrices of a layer (34 bytes a block of 32)
    struct Shape {
        const char* name;
        int n_in, n_out;
    } q8[] = {{"Q8_0 attn_qkv 2048 -> 8192", 2048, 8192}, {"Q8_0 attn_gate 2048 -> 4096", 2048, 4096},
              {"Q8_0 ssm_out / attn_output 4096 -> 2048", 4096, 2048}, {"Q8_0 attn_k/v, shexp up 2048 -> 512", 2048, 512},
              {"Q8_0 ffn_down_shexp 512 -> 2048", 512, 2048}, {"Q8_0 ssm_alpha/beta 2048 -> 32", 2048, 32}};
    for (const Shape& s : q8) {
        const size_t bytes = (size_t) s.n_out * (s.n_in / 32) * 34;
        void* w = dev_random(bytes, false);
        tame_scales(w, bytes, 34, 0);
        line(s.name, time_us([&] { kernels::native_mmvq(8, w, xq, y, s.n_in, s.n_out, 1, st); }), (double) bytes);
        cudaFree(w);
    }
    {   // the router (F32) and the shared expert's gate (F32, one output)
        float* w = (float*) dev_random((size_t) E * 256 * 4, true);
        line("F32 router 2048 -> 256 (gemv_f32)", time_us([&] { poly::gemv_f32(w, x, y, E, 256, 256, 1, st); }),
             (double) E * 256 * 4);
        line("F32 shared gate 2048 -> 1 (gemv_f32)", time_us([&] { poly::gemv_f32(w, x, y, E, 1, 1, 1, st); }),
             (double) E * 4);
        cudaFree(w);
    }
    {   // the gated delta net: conv of 8192 channels, 32 value heads of 128, 16 key heads
        const int dim = 128, heads = 32, kh = 16, conv = 2 * kh * dim + heads * dim;
        float* state = (float*) dev_random((size_t) conv * 3 * 4, true);
        float* cw = (float*) dev_random((size_t) conv * 4 * 4, true);
        float* qkv = (float*) dev_random((size_t) conv * 4, true);
        float* proj = (float*) dev_random((size_t) conv * 4, true);
        line("conv4_silu 8192 channels", time_us([&] { poly::conv4_silu(proj, state, cw, qkv, conv, 1, st); }));
        float* s = (float*) dev_random((size_t) heads * dim * dim * 4, true);
        float* ab = (float*) dev_random(256 * 4, true);
        float* z = (float*) dev_random((size_t) heads * dim * 4, true);
        float* on = (float*) dev_random(dim * 4, true);
        float* out = (float*) dev_random((size_t) heads * dim * 4, true);
        line("gdn_step 32 heads x 128 (state 2 MiB)",
             time_us([&] {
                 poly::gdn_step(qkv, ab, ab + 64, ab + 128, ab + 192, s, z, on, out, dim, heads, kh, 1, 1e-6f, st);
             }),
             2.0 * heads * dim * dim * 4);
    }
    line("swiglu_clamp 512", time_us([&] { poly::swiglu_clamp(x, x + 512, y, 512, 1e30f, st); }));
    line("mul_sigmoid 2048, one gate", time_us([&] { poly::mul_sigmoid(y, x, E, 1, 1, true, st); }));

    {   // a layer's cached experts, eight a launch: IQ3_XXS gate|up (2048 -> 1024) and IQ4_XS down (512 -> 2048)
        const size_t gu = (size_t) 1024 * (2048 / 256) * 98, dn = (size_t) 2048 * (512 / 256) * 136;
        void* wgu = dev_random(gu * 8, false);
        void* wdn = dev_random(dn * 8, false);
        tame_scales(wgu, gu * 8, 98, 0);
        tame_scales(wdn, dn * 8, 136, 0);
        float* g = (float*) dev_random((size_t) 8 * 1024 * 4, true);
        void* hq = dev_random(kernels::native_q8_1_bytes(512) * 8, false);
        kernels::MmvqSlots a{}, b{};
        for (int j = 0; j < 8; ++j) {
            a.w[j] = (const uint8_t*) wgu + j * gu;
            a.x[j] = xq;
            a.y[j] = g + j * 1024;
            b.w[j] = (const uint8_t*) wdn + j * dn;
            b.x[j] = (const uint8_t*) hq + j * kernels::native_q8_1_bytes(512);
            b.y[j] = y + j * 2048;
        }
        line("8 experts IQ3_XXS gate|up 2048 -> 1024 (slots)",
             time_us([&] { kernels::native_mmvq_slots(18, a, 8, 2048, 1024, st); }), 8.0 * gu);
        line("8 experts IQ4_XS down 512 -> 2048 (slots)",
             time_us([&] { kernels::native_mmvq_slots(23, b, 8, 512, 2048, st); }), 8.0 * dn);
        line("1 expert IQ3_XXS gate|up (native_mmvq)",
             time_us([&] { kernels::native_mmvq(18, wgu, xq, g, 2048, 1024, 1, st); }), (double) gu);
        line("1 expert IQ4_XS down (native_mmvq)",
             time_us([&] { kernels::native_mmvq(23, wdn, b.x[0], y, 512, 2048, 1, st); }), (double) dn);
        line("8 experts IQ4_XS down, one launch each (native_mmvq)", time_us([&] {
                 for (int j = 0; j < 8; ++j) kernels::native_mmvq(23, b.w[j], b.x[j], b.y[j], 512, 2048, 1, st);
             }), 8.0 * dn);
    }
    {   // several matrices a launch give each the bits of a launch of its own, for every format and the short
        // rows of an expert's down projection (n_in 256 and 512), on random bytes: the results are compared, not judged
        // where each block keeps its fp16 scales (d, and dmin where there is one): random bytes there would be
        // infinities and NaNs, whose bits two correct sums need not share
        const struct {
            int type, block_elems, block_bytes, d_at, dmin_at;
            const char* name;
        } fmts[] = {{2, 32, 18, 0, -1, "Q4_0"},     {6, 32, 22, 0, -1, "Q5_0"},     {8, 32, 34, 0, -1, "Q8_0"},
                    {20, 32, 18, 0, -1, "IQ4_NL"},  {10, 256, 84, 80, 82, "Q2_K"},  {11, 256, 110, 108, -1, "Q3_K"},
                    {12, 256, 144, 0, 2, "Q4_K"},   {13, 256, 176, 0, 2, "Q5_K"},   {14, 256, 210, 208, -1, "Q6_K"},
                    {23, 256, 136, 0, -1, "IQ4_XS"}};
        int bad = 0, checked = 0;
        for (const auto& f : fmts) {
            for (int n_in : {256, 512}) {
                const int n_out = 2048;
                const size_t wb = (size_t) n_out * (n_in / f.block_elems) * f.block_bytes;
                void* w = dev_random(2 * wb, false);
                tame_scales(w, 2 * wb, f.block_bytes, f.d_at);
                if (f.dmin_at >= 0) tame_scales(w, 2 * wb, f.block_bytes, f.dmin_at);
                void* q = dev_random(kernels::native_q8_1_bytes(n_in) * 2, false);
                kernels::native_quantize_q8_1(x, q, n_in, 1, st);
                kernels::native_quantize_q8_1(x + 4096, (uint8_t*) q + kernels::native_q8_1_bytes(n_in), n_in, 1, st);
                float* o = (float*) dev_random((size_t) 4 * n_out * 4, true);
                kernels::MmvqSlots m{};
                for (int j = 0; j < 2; ++j) {
                    m.w[j] = (const uint8_t*) w + j * wb;
                    m.x[j] = (const uint8_t*) q + j * kernels::native_q8_1_bytes(n_in);
                    m.y[j] = o + j * n_out;
                    kernels::native_mmvq(f.type, m.w[j], m.x[j], o + (2 + j) * n_out, n_in, n_out, 1, st);
                }
                if (!kernels::native_mmvq_slots(f.type, m, 2, n_in, n_out, st)) continue;
                std::vector<uint32_t> h((size_t) 4 * n_out);
                cudaStreamSynchronize(g_st);   // the stream does not wait for the default one, nor it for the stream
                cudaMemcpy(h.data(), o, h.size() * 4, cudaMemcpyDeviceToHost);
                ++checked;
                if (std::memcmp(h.data(), h.data() + 2 * n_out, (size_t) 2 * n_out * 4) != 0) {
                    ++bad;
                    std::printf("  slots and one launch each DIFFER: %s, n_in %d\n", f.name, n_in);
                }
                cudaFree(w);
                cudaFree(q);
                cudaFree(o);
            }
        }
        std::printf("  slots against one launch each, bit for bit: %d of %d shapes the same\n", checked - bad, checked);
        if (bad) return 1;
    }
    {   // the small copies of a layer: the router's scores and the experts' input out, the CPU's sum back in
        float* h = nullptr;
        cudaMallocHost(&h, 1 << 20);
        line("D2H 1 KiB to pinned, in the stream",
             time_us([&] { cudaMemcpyAsync(h, y, 1024, cudaMemcpyDeviceToHost, g_st); }));
        line("D2H 8 KiB to pinned, in the stream",
             time_us([&] { cudaMemcpyAsync(h, y, 8192, cudaMemcpyDeviceToHost, g_st); }));
        line("H2D 8 KiB from pinned, in the stream",
             time_us([&] { cudaMemcpyAsync(y, h, 8192, cudaMemcpyHostToDevice, g_st); }));
        line("kernel + D2H 1 KiB, alternating", time_us([&] {
                 poly::axpy(x, 1.f, y, E, true, st);
                 cudaMemcpyAsync(h, y, 1024, cudaMemcpyDeviceToHost, g_st);
             }));
        // the same through the kernels: pinned memory is the card's to read and write where it lies
        line("kernel + kernel writing 1 KiB into pinned memory", time_us([&] {
                 poly::axpy(x, 1.f, y, E, true, st);
                 poly::copy_words(y, h, 256, st);
             }));
        line("kernel + kernel reading 8 KiB from pinned memory", time_us([&] {
                 poly::axpy(x, 1.f, y, E, true, st);
                 poly::copy_words(h, y + 4096, 2048, st);
             }));
        {   // a token's logits (248,320 floats) out, and a prompt batch's 1,024 embeddings (8 MiB) in, both ways
            float* hb = nullptr;
            float* db = nullptr;
            cudaMallocHost(&hb, size_t(8) << 20);
            cudaMalloc(&db, size_t(8) << 20);
            const size_t lg = 248320, emb = size_t(2) << 20;
            line("kernel + D2H logits 0.95 MiB (copy engine)", time_us([&] {
                     poly::axpy(x, 1.f, y, E, true, st);
                     cudaMemcpyAsync(hb, db, lg * 4, cudaMemcpyDeviceToHost, g_st);
                 }));
            line("kernel + D2H logits 0.95 MiB (kernel)", time_us([&] {
                     poly::axpy(x, 1.f, y, E, true, st);
                     poly::copy_words(db, hb, lg, st);
                 }));
            line("kernel + H2D 8 MiB (copy engine)", time_us([&] {
                     poly::axpy(x, 1.f, y, E, true, st);
                     cudaMemcpyAsync(db, hb, emb * 4, cudaMemcpyHostToDevice, g_st);
                 }));
            line("kernel + H2D 8 MiB (kernel)", time_us([&] {
                     poly::axpy(x, 1.f, y, E, true, st);
                     poly::copy_words(hb, db, emb, st);
                 }));
            cudaFreeHost(hb);
            cudaFree(db);
        }
        {   // what a kernel writes into pinned memory is what the host reads after the stream's event
            std::vector<float> want(256);
            cudaMemcpy(want.data(), y, 1024, cudaMemcpyDeviceToHost);
            std::memset(h, 0, 1024);
            poly::copy_words(y, h, 256, st);
            cudaEvent_t ev = nullptr;
            cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
            cudaEventRecord(ev, g_st);
            cudaEventSynchronize(ev);
            cudaEventDestroy(ev);
            std::printf("  kernel's write into pinned memory, read by the host after an event: %s\n",
                        std::memcmp(want.data(), h, 1024) == 0 ? "the same bytes" : "DIFFERENT BYTES");
        }
        cudaFreeHost(h);
    }
    {   // bandwidth of a plain read, for scale
        const size_t n = size_t(256) << 20;
        void* a = dev_random(1024, false);
        void* b = nullptr;
        void* c = nullptr;
        cudaMalloc(&b, n);
        cudaMalloc(&c, n);
        const double us = time_us([&] { cudaMemcpyAsync(c, b, n, cudaMemcpyDeviceToDevice, g_st); });
        line("device copy 256 MiB (read + write)", us, 2.0 * n);
        cudaFree(a);
        cudaFree(b);
        cudaFree(c);
    }
    std::printf("done\n");
    return 0;
}
