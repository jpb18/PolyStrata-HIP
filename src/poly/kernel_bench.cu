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

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using namespace strata;

int g_reps = 400;
int g_warm = 20;   // launches before the timed ones
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
    for (int i = 0; i < g_warm; ++i) f();
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

// poly_kernel_bench attn [reps]: a decode token's attention (attend()) and a prompt batch's (attend_batch(), 1,024
// tokens) at Qwen3.5-35B-A3B's shapes (16 heads, 2 key-value heads of 256) over caches of 8K to 262K positions:
// wholly in VRAM, and with KV streaming (AMD_KV_STREAMING.md) from position 0 and from 49,152 on in pinned RAM.
// Each split's results against the cache wholly in VRAM, bit for bit.
int attn_bench() {
    const int dim = 256, n_head = 16, kv = 2, width = kv * dim, batch = 1024;
    const float scale = 1.f / 16.f;
    const int64_t most = 262144;
    const size_t cache_bytes = (size_t) most * width * 2;
    uint16_t* k = (uint16_t*) dev_random(cache_bytes, false);
    uint16_t* v = (uint16_t*) dev_random(cache_bytes, false);
    {   // random halves can be infinities: values in [-1, 1)
        std::vector<uint16_t> h((size_t) most * width);
        std::mt19937 rng(7);
        for (auto& x : h) x = (uint16_t) (0x3000 + (rng() & 0x0bff)) | (uint16_t) ((rng() & 1) << 15);
        cudaMemcpy(k, h.data(), cache_bytes, cudaMemcpyHostToDevice);
        for (auto& x : h) x = (uint16_t) (0x3000 + (rng() & 0x0bff)) | (uint16_t) ((rng() & 1) << 15);
        cudaMemcpy(v, h.data(), cache_bytes, cudaMemcpyHostToDevice);
    }
    uint16_t *hk = nullptr, *hv = nullptr, *hk_dev = nullptr, *hv_dev = nullptr;
    if (cudaHostAlloc((void**) &hk, cache_bytes, cudaHostAllocMapped) != cudaSuccess ||
        cudaHostAlloc((void**) &hv, cache_bytes, cudaHostAllocMapped) != cudaSuccess) {
        std::printf("no pinned memory for the host part\n");
        return 1;
    }
    cudaHostGetDevicePointer((void**) &hk_dev, hk, 0);
    cudaHostGetDevicePointer((void**) &hv_dev, hv, 0);
    float* q = (float*) dev_random((size_t) batch * n_head * dim * 4, true);
    const size_t score_floats = size_t(16) << 20;
    float* score = (float*) dev_random(score_floats * 4, true);
    float* work = (float*) dev_random((size_t) batch * n_head * (dim + dim / 2 + 3) * 4 + 64, true);
    float* out = (float*) dev_random((size_t) (batch + 1) * n_head * dim * 4, true);   // the token's, the batch's
    float* ref = (float*) dev_random((size_t) (batch + 1) * n_head * dim * 4, true);
    void* blas = poly::blas_create(g_st);
    const int stage_rows = poly::attend_batch_block(score_floats, 32, n_head);
    uint16_t* ks = (uint16_t*) dev_random((size_t) stage_rows * width * 2, false);
    uint16_t* vs = (uint16_t*) dev_random((size_t) stage_rows * width * 2, false);
    int bad = 0;
    for (int64_t ctx : {8192, 32768, 49152, 65536, 131072, 262144}) {
        const int chunk = (int) std::max<int64_t>(1, (int64_t) score_floats / (n_head * ctx));
        const int64_t pos = ctx - 1;   // a decode token at the end of the cache
        std::printf("context %lld\n", (long long) ctx);
        for (int64_t split : {ctx, std::min<int64_t>(49152, ctx), int64_t(0)}) {
            poly::KvSpill sp{hk_dev, hv_dev, (long long) split, ks, vs, stage_rows};
            if (split < ctx) {   // the rows from split on, as the cache in VRAM has them
                cudaMemcpy(hk, k + (size_t) split * width, (size_t) (ctx - split) * width * 2, cudaMemcpyDeviceToHost);
                cudaMemcpy(hv, v + (size_t) split * width, (size_t) (ctx - split) * width * 2, cudaMemcpyDeviceToHost);
            }
            const poly::KvSpill* s = split < ctx ? &sp : nullptr;
            const double host_mb = (double) (ctx - split) * width * 4 / 1e6;   // K and V rows read from RAM
            const int reps = g_reps;
            g_reps = std::max(5, (int) (g_reps * 8192 / ctx));
            const double us = time_us([&] {
                poly::attend(q, k, v, pos, 1, score, chunk, out, dim, n_head, kv, scale, g_st, 0, 0, false, nullptr, 0, s);
            });
            g_reps = std::max(3, reps / 64);
            const int64_t bpos = ctx - batch;   // a prompt batch that ends the cache
            const double bus = time_us([&] {
                poly::attend_batch(blas, q, k, v, bpos, batch, score, score_floats, work, out + (size_t) n_head * dim, dim,
                                   n_head, kv, scale, g_st, 0, 0, false, s);
            });
            if (split == 0) {   // which reads cost: the keys' from RAM and the values' from VRAM, and the other way
                poly::KvSpill only_k = sp, only_v = sp;
                only_k.vhost = v;
                only_v.khost = k;
                g_reps = std::max(5, (int) (reps * 8192 / ctx));
                const double ku = time_us([&] {
                    poly::attend(q, k, v, pos, 1, score, chunk, out, dim, n_head, kv, scale, g_st, 0, 0, false, nullptr,
                                 0, &only_k);
                });
                const double vu = time_us([&] {
                    poly::attend(q, k, v, pos, 1, score, chunk, out, dim, n_head, kv, scale, g_st, 0, 0, false, nullptr,
                                 0, &only_v);
                });
                std::printf("  keys from RAM only: %.1f us, values from RAM only: %.1f us\n", ku, vu);
            }
            g_reps = reps;
            char name[96];
            std::snprintf(name, sizeof name, "%s: decode token", split == ctx ? "all in VRAM" :
                          split == 0 ? "all in RAM" : "from 49,152 in RAM");
            std::printf("  %-36s %9.1f us (%7.1f MB from RAM, %5.1f GB/s overall); batch of 1,024 %9.1f ms\n", name,
                        us, host_mb, host_mb > 0 ? host_mb * 1e3 / us : 0.0, bus / 1e3);
            // bit for bit against the cache wholly in VRAM
            std::vector<uint32_t> a((size_t) (batch + 1) * n_head * dim), b(a.size());
            cudaStreamSynchronize(g_st);
            cudaMemcpy(split == ctx ? a.data() : b.data(), out, a.size() * 4, cudaMemcpyDeviceToHost);
            if (split == ctx) cudaMemcpy(ref, out, a.size() * 4, cudaMemcpyDeviceToDevice);
            else {
                cudaMemcpy(a.data(), ref, a.size() * 4, cudaMemcpyDeviceToHost);
                const bool same = std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
                std::printf("    the same bits as all in VRAM: %s\n", same ? "yes" : "NO");
                bad += !same;
            }
        }
    }
    poly::blas_destroy(blas);
    cudaFreeHost(hk);
    cudaFreeHost(hv);
    return bad ? 1 : 0;
}

// poly_kernel_bench attn-flash [reps]: a prompt batch's attention, flash attention (attend_flash(), what the engine's
// attend_batch() and attend_batch_kv() take on gfx12) against their products with STRATA_POLY_FLASH=0, over fp16 and
// q8_0 caches of 8K to 131K positions at Qwen3.6-35B-A3B's shapes (16 heads, 2 key-value heads of 256): the times, and
// each result's distance to a double-precision attention of the same half-rounded inputs on a few tokens.  Then
// batches of other sizes and positions (a first chunk, a position that is no multiple of the step, a batch of 33
// tokens) against the double-precision attention only.
int attn_flash_bench() {
    const int dim = 256, n_head = 16, kv = 2, width = kv * dim, batch = 1024;
    const float scale = 1.f / 16.f;
    const int64_t most = 131072;
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::vector<float> hx((size_t) most * width);
    float* xf = nullptr;
    cudaMalloc((void**) &xf, hx.size() * 4);
    uint16_t* k16 = (uint16_t*) dev_random((size_t) most * width * 2, false);
    uint16_t* v16 = (uint16_t*) dev_random((size_t) most * width * 2, false);
    uint8_t* kq = nullptr;
    uint8_t* vq = nullptr;
    const size_t q8_bytes = (size_t) most * poly::q8_row_bytes(dim, kv);
    cudaMalloc((void**) &kq, q8_bytes);
    cudaMalloc((void**) &vq, q8_bytes);
    for (int which = 0; which < 2; ++which) {
        for (auto& x : hx) x = u(rng);
        cudaMemcpy(xf, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice);
        poly::store_f16(xf, which ? v16 : k16, width, 0, (int) most, g_st);
        poly::store_q8(xf, which ? vq : kq, dim, kv, 0, (int) most, g_st);
    }
    cudaStreamSynchronize(g_st);
    std::vector<float> hq((size_t) batch * n_head * dim);
    for (auto& x : hq) x = u(rng) * 8.f;   // scores with a spread of a few: a peaked softmax
    float* q = nullptr;
    cudaMalloc((void**) &q, hq.size() * 4);
    cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice);
    const size_t score_floats = size_t(16) << 20;
    float* score = (float*) dev_random(score_floats * 4, true);
    float* work = (float*) dev_random((size_t) batch * n_head * (dim + dim / 2 + 3) * 4 + 64, true);
    float* out_old = nullptr;
    float* out_new = nullptr;
    cudaMalloc((void**) &out_old, (size_t) batch * n_head * dim * 4);
    cudaMalloc((void**) &out_new, (size_t) batch * n_head * dim * 4);
    const int stage_rows = poly::kQ8StageRows;
    uint16_t* ks = (uint16_t*) dev_random((size_t) stage_rows * width * 2, false);
    uint16_t* vs = (uint16_t*) dev_random((size_t) stage_rows * width * 2, false);
    void* blas = poly::blas_create(g_st);
    uint16_t* dq = nullptr;   // a q8_0 cache as the halves the products read, for the reference
    uint16_t* dq2 = nullptr;
    float* out_chk = nullptr;
    cudaMalloc((void**) &dq, (size_t) most * width * 2);
    cudaMalloc((void**) &dq2, (size_t) most * width * 2);
    cudaMalloc((void**) &out_chk, (size_t) batch * n_head * dim * 4);
    std::vector<uint16_t> hk((size_t) most * width), hv((size_t) most * width);
    const auto half_to_double = [](uint16_t h) {
        const int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
        const double v = e == 0 ? std::ldexp((double) m, -24) : std::ldexp((double) (m + 1024), e - 25);
        return s ? -v : v;
    };
    int bad = 0;
    // one batch of `n` tokens from position bpos over a cache of q8 or fp16 rows: the times when `timed`, and the
    // distance of each result to the double-precision attention
    const auto compare = [&](int64_t bpos, int n, int q8, bool timed) {
        const int64_t ctx = bpos + n;
        const auto run = [&](float* o) {
            if (q8)
                poly::attend_batch_kv(blas, q, kq, true, vq, true, bpos, n, score, score_floats, work, o, dim, n_head, kv,
                                      scale, g_st, ks, vs, stage_rows);
            else
                poly::attend_batch(blas, q, k16, v16, bpos, n, score, score_floats, work, o, dim, n_head, kv, scale,
                                   g_st);
        };
        double old_us = 0, new_us = 0;
        _putenv("STRATA_POLY_FLASH=0");
        if (timed) old_us = time_us([&] { run(out_old); });
        else run(out_old);
        _putenv("STRATA_POLY_FLASH=1");
        if (timed) new_us = time_us([&] { run(out_new); });
        else run(out_new);
        cudaStreamSynchronize(g_st);
        // the reference: the halves the products read (a q8_0 cache's, dequantized)
        if (q8) {
            poly::kv_dequant_q8(kq, 0, (int) ctx, dim, kv, dq, g_st);
            poly::kv_dequant_q8(vq, 0, (int) ctx, dim, kv, dq2, g_st);
            cudaStreamSynchronize(g_st);
            cudaMemcpy(hk.data(), dq, (size_t) ctx * width * 2, cudaMemcpyDeviceToHost);
            cudaMemcpy(hv.data(), dq2, (size_t) ctx * width * 2, cudaMemcpyDeviceToHost);
            // flash over the q8_0 rows against flash over the halves they dequantize to: the same bits when the
            // kernel's own dequantization is kv_dequant_q8()'s
            poly::attend_batch(blas, q, dq, dq2, bpos, n, score, score_floats, work, out_chk, dim, n_head, kv, scale,
                               g_st);
            cudaStreamSynchronize(g_st);
            std::vector<uint32_t> x((size_t) n * n_head * dim), y(x.size());
            cudaMemcpy(x.data(), out_new, x.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(y.data(), out_chk, y.size() * 4, cudaMemcpyDeviceToHost);
            const bool same = std::memcmp(x.data(), y.data(), x.size() * 4) == 0;
            if (!same) std::printf("  flash over q8_0 rows differs from flash over their dequantized halves\n");
            bad += !same;
        } else {
            cudaMemcpy(hk.data(), k16, (size_t) ctx * width * 2, cudaMemcpyDeviceToHost);
            cudaMemcpy(hv.data(), v16, (size_t) ctx * width * 2, cudaMemcpyDeviceToHost);
        }
        std::vector<float> a((size_t) n * n_head * dim), b(a.size());
        cudaMemcpy(a.data(), out_old, a.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(), out_new, b.size() * 4, cudaMemcpyDeviceToHost);
        double err_old = 0, err_new = 0, err_between = 0, mag = 0;
        const int probes[4] = {0, std::min(1, n - 1), n / 2 + 1 < n ? n / 2 + 1 : n - 1, n - 1};
        for (int t : probes) {
            const int64_t last = bpos + t;   // attends keys 0..last
            for (int h = 0; h < n_head; h += 5) {
                const int g = h / (n_head / kv);
                const float* qh = &hq[((size_t) t * n_head + h) * dim];
                std::vector<double> w((size_t) last + 1);
                double mx = -1e300;
                for (int64_t j = 0; j <= last; ++j) {
                    double s = 0;
                    for (int d = 0; d < dim; ++d) {
                        const _Float16 qhv = (_Float16) qh[d];   // the half the products take for the query
                        s += (double) (float) qhv * half_to_double(hk[(size_t) j * width + g * dim + d]);
                    }
                    w[(size_t) j] = s * scale;
                    mx = std::max(mx, w[(size_t) j]);
                }
                double sum = 0;
                for (auto& x : w) {
                    x = std::exp(x - mx);
                    sum += x;
                }
                for (int d = 0; d < dim; ++d) {
                    double o = 0;
                    for (int64_t j = 0; j <= last; ++j)
                        o += w[(size_t) j] * half_to_double(hv[(size_t) j * width + g * dim + d]);
                    o /= sum;
                    const size_t at = ((size_t) t * n_head + h) * dim + d;
                    err_old = std::max(err_old, std::fabs(a[at] - o));
                    err_new = std::max(err_new, std::fabs(b[at] - o));
                    err_between = std::max(err_between, (double) std::fabs(a[at] - b[at]));
                    mag = std::max(mag, std::fabs(o));
                }
            }
        }
        if (timed)
            std::printf("ctx %6lld %s batch %d: products %9.2f ms | flash %9.2f ms (%.2fx) | max |error| vs double: "
                        "products %.2e flash %.2e (values up to %.2f), flash vs products %.2e\n",
                        (long long) ctx, q8 ? "q8_0" : "fp16", n, old_us / 1e3, new_us / 1e3, old_us / new_us, err_old,
                        err_new, mag, err_between);
        else
            std::printf("from %6lld, %4d tokens %s: max |error| vs double: products %.2e flash %.2e (values up to %.2f)\n",
                        (long long) bpos, n, q8 ? "q8_0" : "fp16", err_old, err_new, mag);
        bad += !(err_new < 8 * err_old + 1e-5);
    };
    for (int64_t ctx : {8192, 32768, 65536, 131072})
        for (int q8 = 0; q8 < 2; ++q8) {
            g_reps = std::max(3, 200 * 8192 / (int) ctx / 8);
            compare(ctx - batch, batch, q8, true);
        }
    struct Edge {
        int64_t pos;
        int n;
    };
    for (const Edge e : {Edge{0, 32}, Edge{0, 64}, Edge{0, 200}, Edge{5, 33}, Edge{31, 97}, Edge{1000, 100},
                         Edge{8000, 129}, Edge{50000, 777}})
        for (int q8 = 0; q8 < 2; ++q8) compare(e.pos, e.n, q8, false);
    poly::blas_destroy(blas);
    return bad ? 1 : 0;
}

// poly_kernel_bench attn-decode [reps]: the attention of a call of 1 to 4 tokens (a decode token, a window of
// guesses) at Qwen3.6-35B-A3B's shapes over caches of 2K to 45K positions in VRAM, as the engine launches it from a
// recording (the rows sized by room(), the positions read on the card).  Beside each time, what reading the keys
// and values once takes at the card's copy rate: the least the attention can take.  Each call's tokens are
// checked against the same tokens one at a time, bit for bit.
int attn_decode_bench() {
    const int dim = 256, n_head = 16, kv = 2, width = kv * dim, most_rows = 8;
    const float scale = 1.f / 16.f;
    const int64_t most = 49152;
    const size_t cache_bytes = (size_t) most * width * 2;
    uint16_t* k = (uint16_t*) dev_random(cache_bytes, false);
    uint16_t* v = (uint16_t*) dev_random(cache_bytes, false);
    {   // random halves can be infinities: values in [-1, 1)
        std::vector<uint16_t> h((size_t) most * width);
        std::mt19937 rng(7);
        for (auto& x : h) x = (uint16_t) (0x3000 + (rng() & 0x0bff)) | (uint16_t) ((rng() & 1) << 15);
        cudaMemcpy(k, h.data(), cache_bytes, cudaMemcpyHostToDevice);
        for (auto& x : h) x = (uint16_t) (0x3000 + (rng() & 0x0bff)) | (uint16_t) ((rng() & 1) << 15);
        cudaMemcpy(v, h.data(), cache_bytes, cudaMemcpyHostToDevice);
    }
    float* q = (float*) dev_random((size_t) most_rows * n_head * dim * 4, true);
    const size_t score_floats = size_t(16) << 20;
    float* score = (float*) dev_random(score_floats * 4, true);
    float* out = (float*) dev_random((size_t) most_rows * n_head * dim * 4, true);
    float* one = (float*) dev_random((size_t) most_rows * n_head * dim * 4, true);
    long long* at = (long long*) dev_random(most_rows * 2 * sizeof(long long), false);
    // the copy rate: one cache read and written
    const double copy_us = time_us([&] { cudaMemcpyAsync(score, k, cache_bytes / 2, cudaMemcpyDeviceToDevice, g_st); });
    const double rate = (double) cache_bytes / (copy_us * 1e3);   // GB/s of reads and writes
    std::printf("copy rate %.1f GB/s (read + write)\n", rate);
    int bad = 0;
    for (int64_t ctx : {2048, 8192, 31296, 45355}) {
        const int chunk = (int) std::max<int64_t>(1, (int64_t) score_floats / (n_head * most));
        const int64_t pos = ctx;   // the call's first token comes after ctx rows
        for (int rows : {1, 2, 3, 4, 8}) {
            int room = 1024;   // Engine::room()
            while (room < pos + rows) room *= 2;
            room = (int) std::min<int64_t>(room, most);
            std::vector<long long> h_at((size_t) rows * 2);
            for (int t = 0; t < rows; ++t) h_at[(size_t) 2 * t] = h_at[(size_t) 2 * t + 1] = pos + t;
            cudaStreamSynchronize(g_st);
            cudaMemcpy(at, h_at.data(), h_at.size() * sizeof(long long), cudaMemcpyHostToDevice);
            const int reps = g_reps;
            g_reps = std::max(10, (int) (g_reps * 8192 / ctx));
            const double us = time_us([&] {
                poly::attend(q, k, v, pos, rows, score, chunk, out, dim, n_head, kv, scale, g_st, 0, 0, false, at, room);
            });
            g_reps = reps;
            const double least = (double) (pos + rows) * width * 2 * 2 / (rate * 1e3);   // K and V, read once
            std::printf("  context %6lld, %d tokens: %8.1f us   (K and V once: %6.1f us; %.2f x)\n", (long long) ctx, rows,
                        us, least, us / least);
            if (rows == 1) {   // the token's values, to set against another build's (STRATA_POLY_ATTN=0)
                std::vector<float> o((size_t) n_head * dim);
                cudaStreamSynchronize(g_st);
                cudaMemcpy(o.data(), out, o.size() * 4, cudaMemcpyDeviceToHost);
                char path[64];
                std::snprintf(path, sizeof path, "attn-decode-%lld.bin", (long long) ctx);
                if (FILE* f = std::fopen(path, "rb")) {   // a run before this one left its values: compare
                    std::vector<float> r(o.size());
                    const size_t got = std::fread(r.data(), 4, r.size(), f);
                    std::fclose(f);
                    double most = 0, scale_of = 0;
                    size_t differ = 0;
                    for (size_t i = 0; i < o.size() && got == o.size(); ++i) {
                        differ += std::memcmp(&o[i], &r[i], 4) != 0;
                        most = std::max(most, (double) std::fabs(o[i] - r[i]));
                        scale_of = std::max(scale_of, (double) std::fabs(r[i]));
                    }
                    std::printf("    against %s: %zu of %zu values differ, largest difference %.3g (values up to %.3g)\n",
                                path, differ, o.size(), most, scale_of);
                } else if (FILE* w = std::fopen(path, "wb")) {
                    std::fwrite(o.data(), 4, o.size(), w);
                    std::fclose(w);
                }
                continue;
            }
            // each token on its own, at its own position
            for (int t = 0; t < rows; ++t) {
                cudaStreamSynchronize(g_st);   // the stream does not wait for this copy, nor it for the stream
                cudaMemcpy(at, h_at.data() + 2 * t, 2 * sizeof(long long), cudaMemcpyHostToDevice);
                poly::attend(q + (size_t) t * n_head * dim, k, v, pos + t, 1, score, chunk,
                             one + (size_t) t * n_head * dim, dim, n_head, kv, scale, g_st, 0, 0, false, at, room);
            }
            cudaStreamSynchronize(g_st);
            std::vector<uint32_t> a((size_t) rows * n_head * dim), b(a.size());
            cudaMemcpy(a.data(), out, a.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(b.data(), one, b.size() * 4, cudaMemcpyDeviceToHost);
            if (std::memcmp(a.data(), b.data(), a.size() * 4) != 0) {
                std::printf("    NOT the same bits as one token at a time:");
                for (int t = 0; t < rows; ++t) {
                    size_t differ = 0;
                    float most_off = 0.f;
                    for (size_t i = (size_t) t * n_head * dim; i < (size_t) (t + 1) * n_head * dim; ++i)
                        if (a[i] != b[i]) {
                            float x, y;
                            std::memcpy(&x, &a[i], 4);
                            std::memcpy(&y, &b[i], 4);
                            ++differ;
                            most_off = std::max(most_off, std::fabs(x - y));
                        }
                    std::printf(" token %d: %zu values (%.2g)", t, differ, most_off);
                }
                std::printf("\n");
                ++bad;
            }
        }
    }
    return bad ? 1 : 0;
}

// poly_kernel_bench cols: whether a product of several tokens at once (native_mmvq with 2 to 8 columns, what a
// window's dense matrices get) gives each token the bits it gets alone (1 column), for the formats of
// Qwen3.6-35B-A3B UD-Q3_K_M's dense weights (Q8_0) and output (Q6_K), and the F32 router (gemv_f32).
int cols_check() {
    int bad = 0;
    struct Case {
        int type, block_values, block_bytes, scale_at, n_in, n_out;
        const char* what;
    } cases[] = {{8, 32, 34, 0, 2048, 8192, "Q8_0 2048 -> 8192"},
                 {8, 32, 34, 0, 4096, 2048, "Q8_0 4096 -> 2048"},
                 {14, 256, 210, 208, 2048, 4096, "Q6_K 2048 -> 4096"}};
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    for (const Case& c : cases) {
        const size_t wbytes = (size_t) c.n_out * (c.n_in / c.block_values) * c.block_bytes;
        void* w = dev_random(wbytes, false);
        tame_scales(w, wbytes, c.block_bytes, c.scale_at);
        std::vector<float> hx((size_t) 8 * c.n_in);
        for (auto& v : hx) v = u(rng);
        float* x = (float*) dev_random(hx.size() * 4, true);
        cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice);
        const size_t qrow = strata::kernels::native_q8_1_bytes(c.n_in);
        void* xq = dev_random(qrow * 8, false);
        strata::kernels::native_quantize_q8_1(x, xq, c.n_in * 8, 1, g_st);
        float* y = (float*) dev_random((size_t) 8 * c.n_out * 4, true);
        float* y1 = (float*) dev_random((size_t) 8 * c.n_out * 4, true);
        for (int t = 0; t < 8; ++t)
            strata::kernels::native_mmvq(c.type, w, (const uint8_t*) xq + t * qrow, y1 + (size_t) t * c.n_out, c.n_in,
                                         c.n_out, 1, g_st);
        std::vector<uint32_t> a((size_t) 8 * c.n_out), b(a.size());
        cudaStreamSynchronize(g_st);
        cudaMemcpy(b.data(), y1, b.size() * 4, cudaMemcpyDeviceToHost);
        std::printf("  %-22s", c.what);
        for (int n = 2; n <= 8; ++n) {
            strata::kernels::native_mmvq(c.type, w, xq, y, c.n_in, c.n_out, n, g_st);
            cudaStreamSynchronize(g_st);
            cudaMemcpy(a.data(), y, (size_t) n * c.n_out * 4, cudaMemcpyDeviceToHost);
            size_t differ = 0;
            for (size_t i = 0; i < (size_t) n * c.n_out; ++i) differ += a[i] != b[i];
            std::printf(" n=%d:%s", n, differ ? ("DIFFER(" + std::to_string(differ) + ")").c_str() : "same");
            bad += differ != 0;
        }
        std::printf("\n");
    }
    {   // the delta net's step (gdn_step) and its convolution (conv4_silu), Qwen3.6's shapes, n tokens at once
        // against one at a time: the output rows and the state after them
        const int dim = 128, heads = 32, kh = 16, row_n = (2 * kh + heads) * dim, ch = row_n, inner = heads * dim;
        const size_t s_n = (size_t) heads * dim * dim;
        auto upload = [&](const std::vector<float>& h) {
            float* p = (float*) dev_random(h.size() * 4 + 64, true);
            cudaMemcpy(p, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
            return p;
        };
        auto rnd = [&](size_t n, float scale) {
            std::vector<float> v(n);
            for (auto& x : v) x = u(rng) * scale;
            return v;
        };
        const std::vector<float> qkv0 = rnd((size_t) 8 * row_n, 1.f), al0 = rnd(8 * heads, 1.f), be0 = rnd(8 * heads, 1.f),
                                 dt = rnd(heads, 1.f), a = rnd(heads, 1.f), s0 = rnd(s_n, 0.05f),
                                 z = rnd((size_t) 8 * inner, 1.f), on = rnd(dim, 1.f), conv_w = rnd((size_t) 4 * ch, 0.5f),
                                 past0 = rnd((size_t) 3 * ch, 1.f);
        float *dtd = upload(dt), *ad = upload(a), *zd = upload(z), *ond = upload(on), *cw = upload(conv_w);
        std::printf("  %-22s", "gdn_step + conv4_silu");
        for (int n = 2; n <= 8; ++n) {
            std::vector<float> out_many((size_t) n * inner), out_one(out_many.size()), s_many(s_n), s_one(s_n),
                c_many((size_t) n * ch), c_one(c_many.size()), p_many((size_t) 3 * ch), p_one(p_many.size());
            for (int way = 0; way < 2; ++way) {   // 0: n at once, 1: one at a time
                float *qkv = upload(qkv0), *al = upload(al0), *be = upload(be0), *s = upload(s0), *past = upload(past0);
                float* conv = (float*) dev_random((size_t) 8 * ch * 4, true);
                float* out = (float*) dev_random((size_t) 8 * inner * 4, true);
                for (int t0 = 0; t0 < n; t0 += way ? 1 : n) {
                    const int m = way ? 1 : n;
                    poly::conv4_silu(qkv + (size_t) t0 * row_n, past, cw, conv + (size_t) t0 * ch, ch, m, g_st);
                    poly::gdn_step(conv + (size_t) t0 * ch, al + t0 * heads, be + t0 * heads, dtd, ad, s,
                                   zd + (size_t) t0 * inner, ond, out + (size_t) t0 * inner, dim, heads, kh, m, 1e-6f,
                                   g_st);
                }
                cudaStreamSynchronize(g_st);
                cudaMemcpy((way ? out_one : out_many).data(), out, out_many.size() * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy((way ? s_one : s_many).data(), s, s_n * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy((way ? c_one : c_many).data(), conv, c_many.size() * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy((way ? p_one : p_many).data(), past, p_many.size() * 4, cudaMemcpyDeviceToHost);
            }
            const bool same = std::memcmp(out_many.data(), out_one.data(), out_many.size() * 4) == 0 &&
                              std::memcmp(s_many.data(), s_one.data(), s_n * 4) == 0;
            const bool csame = std::memcmp(c_many.data(), c_one.data(), c_many.size() * 4) == 0 &&
                               std::memcmp(p_many.data(), p_one.data(), p_many.size() * 4) == 0;
            std::printf(" n=%d:%s%s", n, csame ? "" : "CONV-DIFFER,", same ? "same" : "DIFFER");
            bad += !same || !csame;
        }
        std::printf("\n");
    }
    {   // the router: F32 2048 -> 256
        const int n_in = 2048, n_out = 256;
        float* w = (float*) dev_random((size_t) n_in * n_out * 4, true);
        float* x = (float*) dev_random((size_t) 8 * n_in * 4, true);
        float* y = (float*) dev_random((size_t) 8 * n_out * 4, true);
        float* y1 = (float*) dev_random((size_t) 8 * n_out * 4, true);
        for (int t = 0; t < 8; ++t)
            poly::gemv_f32(w, x + (size_t) t * n_in, y1 + (size_t) t * n_out, n_in, n_out, n_out, 1, g_st);
        std::vector<uint32_t> a((size_t) 8 * n_out), b(a.size());
        cudaStreamSynchronize(g_st);
        cudaMemcpy(b.data(), y1, b.size() * 4, cudaMemcpyDeviceToHost);
        std::printf("  %-22s", "F32 router 2048 -> 256");
        for (int n = 2; n <= 8; ++n) {
            poly::gemv_f32(w, x, y, n_in, n_out, n_out, n, g_st);
            cudaStreamSynchronize(g_st);
            cudaMemcpy(a.data(), y, (size_t) n * n_out * 4, cudaMemcpyDeviceToHost);
            size_t differ = 0;
            for (size_t i = 0; i < (size_t) n * n_out; ++i) differ += a[i] != b[i];
            std::printf(" n=%d:%s", n, differ ? ("DIFFER(" + std::to_string(differ) + ")").c_str() : "same");
            bad += differ != 0;
        }
        std::printf("\n");
    }
    return bad ? 1 : 0;
}

// ---- the q8_0 cache (docs/AMD_HIP_KV_Q8.md), at Qwen3.6-35B-A3B's shapes: 16 heads, 2 key-value heads of 256

constexpr int kDim = 256, kHeads = 16, kKv = 2, kWidth = kKv * kDim;
const size_t kQ8Row = poly::q8_row_bytes(kDim, kKv), kHb = poly::q8_head_bytes(kDim);

void* blas_handle() {
    static void* b = poly::blas_create(g_st);
    return b;
}

// a cache's rows as floats, kWidth a row: an fp16 cache's halves, or a q8_0 cache's code * scale
std::vector<float> f16_floats(const std::vector<uint16_t>& h) {
    std::vector<float> f(h.size());
    for (size_t i = 0; i < h.size(); ++i) f[i] = ggml_fp16_to_fp32(h[i]);
    return f;
}

std::vector<float> q8_floats(const std::vector<uint8_t>& c, int64_t rows) {
    std::vector<float> f((size_t) rows * kWidth);
    for (int64_t r = 0; r < rows; ++r)
        for (int h = 0; h < kKv; ++h) {
            const uint8_t* head = c.data() + (size_t) r * kQ8Row + h * kHb;
            for (int d = 0; d < kDim; ++d) {
                uint16_t sb;
                std::memcpy(&sb, head + kDim + 2 * (d / poly::kQ8Block), 2);
                f[(size_t) r * kWidth + h * kDim + d] = (float) (int8_t) head[d] * ggml_fp16_to_fp32(sb);
            }
        }
    return f;
}

// The attention of one token at position p over rows 0..p of K and V (floats, kWidth a row), in doubles: the
// reference.  out: kHeads * kDim; w (when given): each head's weights, p + 1 of them.
void ref_attend(const float* q, const float* k, const float* v, int64_t p, float scale, double* out,
                std::vector<double>* w = nullptr) {
    const int per = kHeads / kKv;
    const int64_t n = p + 1;
    auto head = [&](int hd) {
        const int g = hd / per;
        std::vector<double> s((size_t) n);
        double mx = -1e300;
        for (int64_t j = 0; j < n; ++j) {
            const float* kr = k + (size_t) j * kWidth + g * kDim;
            double acc = 0;
            for (int d = 0; d < kDim; ++d) acc += (double) q[hd * kDim + d] * kr[d];
            s[(size_t) j] = acc * scale;
            mx = std::max(mx, s[(size_t) j]);
        }
        double sum = 0;
        for (double& x : s) sum += x = std::exp(x - mx);
        double o[kDim] = {};
        for (int64_t j = 0; j < n; ++j) {
            const double wj = s[(size_t) j] / sum;
            const float* vr = v + (size_t) j * kWidth + g * kDim;
            for (int d = 0; d < kDim; ++d) o[d] += wj * vr[d];
            if (w) w[hd][(size_t) j] = wj;
        }
        for (int d = 0; d < kDim; ++d) out[hd * kDim + d] = o[d];
    };
    if (w)
        for (int hd = 0; hd < kHeads; ++hd) w[hd].assign((size_t) n, 0.0);
    std::vector<std::thread> pool;
    const int threads = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
    for (int t = 0; t < threads; ++t)
        pool.emplace_back([&, t] {
            for (int hd = t; hd < kHeads; hd += threads) head(hd);
        });
    for (auto& th : pool) th.join();
}

struct Diff {
    double max_abs = 0, max_ref = 0, rms = 0, rms_ref = 0;
    double rel_max() const { return max_ref > 0 ? max_abs / max_ref : 0; }
    double rel_rms() const { return rms_ref > 0 ? std::sqrt(rms / rms_ref) : 0; }
};

template <class A, class B>
Diff diff(const A* a, const B* ref, size_t n) {
    Diff d;
    for (size_t i = 0; i < n; ++i) {
        const double x = (double) a[i] - (double) ref[i];
        d.max_abs = std::max(d.max_abs, std::fabs(x));
        d.max_ref = std::max(d.max_ref, std::fabs((double) ref[i]));
        d.rms += x * x;
        d.rms_ref += (double) ref[i] * ref[i];
    }
    return d;
}

// T1 (poly_kernel_bench q8-quant): store_q8's codes and scales against ggml's own quantize_row_q8_0 (through
// ggml_quantize_chunk), bit for bit, over a million blocks of eight kinds; each value's error against the bound
// d/2 + 127 |d - fp16(d)|; store_q8_at replayed from a recorded graph at three positions; kv_dequant_q8's halves
// against fp16(code * scale) on the host.
int q8_quant_check() {
    const int64_t rows = 65536;   // 2 heads of 8 blocks a row: 1,048,576 blocks
    const size_t n = (size_t) rows * kWidth;
    std::vector<float> x(n);
    std::mt19937 rng(2026);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(-1.f, 1.f), ue(-4.f, 3.f);
    const char* kinds[8] = {"uniform, scale 1e-4 to 1e3", "normal, one value x40", "zeros", "one value",
                            "ties at x/d = k + 1/2", "tiny (scale under fp16)", "huge (scale over fp16 too)",
                            "normal, channel scales"};
    for (size_t b = 0; b < n / 32; ++b) {
        float* v = x.data() + b * 32;
        const int kind = (int) (b % 8);
        const float s = std::pow(10.f, ue(rng));
        for (int j = 0; j < 32; ++j) {
            switch (kind) {
                case 0: v[j] = ud(rng) * s; break;
                case 1: v[j] = nd(rng) * (j == (int) (b / 8 % 32) ? 40.f : 1.f); break;
                case 2: v[j] = 0.f; break;
                case 3: v[j] = j == 7 ? nd(rng) : 0.f; break;
                case 4: v[j] = j == 0 ? 127.f : (float) ((int) (rng() % 252) - 126) + 0.5f; break;
                case 5: v[j] = nd(rng) * 1e-30f; break;
                case 6: v[j] = nd(rng) * (b % 16 < 8 ? 1e6f : 1e7f); break;
                default: v[j] = nd(rng) * (1.f + 7.f * (j % 11 == 3)); break;
            }
        }
    }
    float* dx = (float*) dev_random(n * 4, true);
    cudaMemcpy(dx, x.data(), n * 4, cudaMemcpyHostToDevice);
    uint8_t* cache = (uint8_t*) dev_random((size_t) rows * kQ8Row, false);
    poly::store_q8(dx, cache, kDim, kKv, 0, (int) rows, g_st);
    cudaStreamSynchronize(g_st);
    std::vector<uint8_t> got((size_t) rows * kQ8Row);
    cudaMemcpy(got.data(), cache, got.size(), cudaMemcpyDeviceToHost);
    // ggml: a block is its fp16 d, then 32 codes
    std::vector<uint8_t> ref((size_t) rows * kKv * (kDim / 32) * 34);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, x.data(), ref.data(), 0, rows * kKv, kDim, nullptr);
    size_t bad_blocks[8] = {}, blocks[8] = {};
    double worst = 0;   // the largest error over d, of the blocks whose fp16 scale is a normal number
    size_t over = 0, subnormal = 0;
    for (int64_t r = 0; r < rows; ++r)
        for (int h = 0; h < kKv; ++h)
            for (int b = 0; b < kDim / 32; ++b) {
                const size_t bi = ((size_t) r * kKv + h) * (kDim / 32) + b;
                const uint8_t* rb = ref.data() + bi * 34;
                const uint8_t* head = got.data() + (size_t) r * kQ8Row + h * kHb;
                const int kind = (int) (bi % 8);
                ++blocks[kind];
                if (std::memcmp(head + b * 32, rb + 2, 32) != 0 || std::memcmp(head + kDim + 2 * b, rb, 2) != 0)
                    ++bad_blocks[kind];
                uint16_t sb;
                std::memcpy(&sb, rb, 2);
                const float d16 = ggml_fp16_to_fp32(sb);
                float amax = 0;
                for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[bi * 32 + j]));
                const double d = (double) (amax / 127.f);
                if (d == 0 || !std::isfinite(d16) || d16 == 0) continue;
                const bool normal = d16 >= 6.103515625e-05f;   // 2^-14: fp16's least normal number
                subnormal += !normal;
                const double bound = 0.5 * d + 127 * std::fabs(d - (double) d16) + 1e-6 * d;
                for (int j = 0; j < 32; ++j) {
                    const double e = std::fabs((double) x[bi * 32 + j] - (double) (int8_t) rb[2 + j] * d16);
                    if (normal) worst = std::max(worst, e / d);
                    over += e > bound;
                }
            }
    int bad = 0;
    std::printf("T1 store_q8 against ggml's quantize_row_q8_0 (%lld rows, %lld blocks):\n", (long long) rows,
                (long long) (rows * kKv * kDim / 32));
    for (int k = 0; k < 8; ++k) {
        std::printf("  %-32s %8zu blocks, %zu not the same bits\n", kinds[k], blocks[k], bad_blocks[k]);
        bad += bad_blocks[k] != 0;
    }
    std::printf("  largest error %.4f d where fp16(d) is a normal number (at most 0.562 d there); every value against "
                "0.5 d + 127 |d - fp16(d)|: %zu over it; %zu blocks have a subnormal fp16 scale (max |x| under "
                "0.0078), whose error is only within that bound\n", worst, over, subnormal);
    bad += over != 0 || worst > 0.5625;

    // store_q8_at, recorded once and replayed at three positions: the rows it wrote against store_q8's
    {
        uint8_t* c2 = (uint8_t*) dev_random((size_t) rows * kQ8Row, false);
        long long* at = (long long*) dev_random(16, false);
        cudaGraph_t g = nullptr;
        cudaGraphExec_t ge = nullptr;
        cudaStreamBeginCapture(g_st, cudaStreamCaptureModeThreadLocal);
        poly::store_q8_at(dx, c2, kDim, kKv, at, 16, g_st);
        cudaStreamEndCapture(g_st, &g);
        const bool made = g && cudaGraphInstantiateWithFlags(&ge, g, 0) == cudaSuccess;
        int same = 0;
        for (long long off : {0LL, 1000LL, rows - 16}) {
            cudaMemcpy(at, &off, 8, cudaMemcpyHostToDevice);
            if (made) cudaGraphLaunch(ge, g_st);
            cudaStreamSynchronize(g_st);
            std::vector<uint8_t> w((size_t) 16 * kQ8Row);
            cudaMemcpy(w.data(), c2 + (size_t) off * kQ8Row, w.size(), cudaMemcpyDeviceToHost);
            same += std::memcmp(w.data(), got.data(), w.size()) == 0;
        }
        std::printf("  store_q8_at from a recorded graph, replayed at rows 0, 1000, %lld: %d of 3 the same bits as "
                    "store_q8%s\n", (long long) rows - 16, same, made ? "" : " (the graph was not made)");
        bad += same != 3;
    }
    // kv_dequant_q8: the halves of code * scale
    {
        uint16_t* out = (uint16_t*) dev_random(n * 2, false);
        poly::kv_dequant_q8(cache, 0, (int) rows, kDim, kKv, out, g_st);
        cudaStreamSynchronize(g_st);
        std::vector<uint16_t> h(n);
        cudaMemcpy(h.data(), out, n * 2, cudaMemcpyDeviceToHost);
        const std::vector<float> f = q8_floats(got, rows);
        size_t differ = 0;
        for (size_t i = 0; i < n; ++i) differ += h[i] != ggml_fp32_to_fp16(f[i]);
        std::printf("  kv_dequant_q8: %zu of %zu halves differ from fp16(code * scale)\n", differ, n);
        bad += differ != 0;
        const double us = time_us([&] { poly::kv_dequant_q8(cache, 0, (int) rows, kDim, kKv, out, g_st); });
        std::printf("  kv_dequant_q8 of %lld rows: %.1f us (%.1f GB/s read + written)\n", (long long) rows, us,
                    (double) rows * (kQ8Row + kWidth * 2) / (us * 1e3));
        const double su = time_us([&] { poly::store_q8(dx, cache, kDim, kKv, 0, 1024, g_st); });
        std::printf("  store_q8 of 1,024 tokens (a prompt batch's rows of one cache): %.1f us\n", su);
    }
    std::printf("T1 %s\n", bad ? "FAILED" : "passed");
    return bad ? 1 : 0;
}

// Floats with the spread of a model's keys (a few channels much larger) or values, uploaded `rows` rows at a time
// into an fp16 cache (store_f16) and a q8_0 one (store_q8): the same values in both.
void fill_caches(uint16_t* f16, uint8_t* q8, int64_t rows, bool keys, uint32_t seed) {
    const int64_t step = 16384;
    std::vector<float> h((size_t) step * kWidth);
    float* d = (float*) dev_random(h.size() * 4, true);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> chan(kWidth);
    for (int c = 0; c < kWidth; ++c) chan[(size_t) c] = keys ? (c % 37 == 5 ? 6.f : 1.f) * 0.5f : 0.3f;
    for (int64_t r0 = 0; r0 < rows; r0 += step) {
        const int64_t m = std::min(step, rows - r0);
        for (size_t i = 0; i < (size_t) m * kWidth; ++i) h[i] = nd(rng) * chan[i % kWidth];
        cudaMemcpy(d, h.data(), (size_t) m * kWidth * 4, cudaMemcpyHostToDevice);
        poly::store_f16(d, f16, kWidth, r0, (int) m, g_st);
        poly::store_q8(d, q8, kDim, kKv, r0, (int) m, g_st);
        cudaStreamSynchronize(g_st);
    }
    cudaFree(d);
}

int profile_chunk(int64_t profile) { return (int) std::max<int64_t>(1, (int64_t(16) << 20) / (kHeads * profile)); }

// whether attend() takes the rows kernels for a call (its condition, on the host)
bool rows_kernels(int rows, int chunk, int room) {
    if (const char* v = std::getenv("STRATA_POLY_ATTN"); v != nullptr && v[0] == '0') return false;
    const int c = std::min(chunk, rows), parts = (room + 511) / 512;
    return (size_t) c * kHeads * room + (size_t) parts * c * kHeads * kDim <= (size_t) chunk * kHeads * room;
}

// T2 and the speed (poly_kernel_bench attn-q8 [reps]): attend() over an fp16 cache and attend_q8() over a q8_0 cache of
// the same values, as the engine launches a token or a window (recorded: the rows sized by room(), the positions read
// on the card), with the chunk each profile's context gives (48K: 21, 128K: 8, 262K: 4), at contexts of 2K to 261K;
// each against the least that reading its cache once takes.  Each window's tokens against the same tokens one at a
// time, bit for bit.  Then, for some of them, both against the attention in doubles over the values each reads, and
// against each other; and a prompt batch's attend_batch_q8() against attend_batch() over a cache of the halves it hands
// its products, bit for bit, and its time against attend_batch() over the fp16 cache.
int attn_q8_bench() {
    const int64_t most = 262144;
    const float scale = 1.f / 16.f;
    uint16_t* k16 = (uint16_t*) dev_random((size_t) most * kWidth * 2, false);
    uint16_t* v16 = (uint16_t*) dev_random((size_t) most * kWidth * 2, false);
    uint8_t* kq = (uint8_t*) dev_random((size_t) most * kQ8Row, false);
    uint8_t* vq = (uint8_t*) dev_random((size_t) most * kQ8Row, false);
    fill_caches(k16, kq, most, true, 11);
    fill_caches(v16, vq, most, false, 12);
    const int most_rows = 8, batch = 1024;
    std::vector<float> hq((size_t) batch * kHeads * kDim);
    {
        std::mt19937 rng(13);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& x : hq) x = nd(rng);
    }
    float* q = (float*) dev_random(hq.size() * 4, true);
    cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice);
    const size_t score_floats = size_t(16) << 20;
    float* score = (float*) dev_random(score_floats * 4, true);
    float* out = (float*) dev_random((size_t) batch * kHeads * kDim * 4, true);
    float* out2 = (float*) dev_random((size_t) batch * kHeads * kDim * 4, true);
    float* one = (float*) dev_random((size_t) most_rows * kHeads * kDim * 4, true);
    long long* at = (long long*) dev_random(most_rows * 2 * sizeof(long long), false);
    const double copy_us = time_us([&] {
        cudaMemcpyAsync(score, k16, (size_t) 8192 * kWidth * 2 * 4, cudaMemcpyDeviceToDevice, g_st);
    });
    const double rate = (double) 8192 * kWidth * 2 * 4 * 2 / (copy_us * 1e3);   // GB/s read + written
    std::printf("copy rate %.1f GB/s (read + write); a row of K and V: fp16 %d bytes, q8_0 %zu\n", rate, kWidth * 4,
                2 * kQ8Row);
    int bad = 0;
    struct Case {
        int64_t ctx, profile;
    } cases[] = {{2048, 49152},   {8192, 49152},   {32768, 49152},   {45355, 49152}, {65536, 131072},
                 {131000, 131072}, {32768, 262144}, {131000, 262144}, {261000, 262144}};
    std::printf("decode (attend / attend_q8), us a call; least: K and V read once at the copy rate\n");
    for (const Case& cs : cases) {
        const int chunk = profile_chunk(cs.profile);
        for (int rows : {1, 2, 4, 8}) {
            const int64_t pos = cs.ctx;
            int room = 1024;   // Engine::room()
            while (room < pos + rows) room *= 2;
            room = (int) std::min<int64_t>(room, cs.profile);
            std::vector<long long> h_at((size_t) rows * 2);
            for (int t = 0; t < rows; ++t) h_at[(size_t) 2 * t] = h_at[(size_t) 2 * t + 1] = pos + t;
            cudaStreamSynchronize(g_st);
            cudaMemcpy(at, h_at.data(), h_at.size() * sizeof(long long), cudaMemcpyHostToDevice);
            const int reps = g_reps;
            g_reps = std::max(10, (int) (g_reps * 8192 / cs.ctx));
            const double f16 = time_us([&] {
                poly::attend(q, k16, v16, pos, rows, score, chunk, out, kDim, kHeads, kKv, scale, g_st, 0, 0, false, at,
                             room);
            });
            const double q8 = time_us([&] {
                poly::attend_q8(q, kq, vq, pos, rows, score, chunk, out2, kDim, kHeads, kKv, scale, g_st, 0, 0, false,
                                at, room);
            });
            g_reps = reps;
            const double least16 = (double) (pos + rows) * kWidth * 2 * 2 / (rate * 1e3);   // as attn-decode counts it
            const double least8 = (double) (pos + rows) * kQ8Row * 2 / (rate * 1e3);
            std::printf("  ctx %6lld (profile %6lld, chunk %2d) %d tok, %s kernels: fp16 %8.1f us (%.2f x least) | q8_0 "
                        "%8.1f us (%.2f x least) | q8_0 / fp16 %.3f\n", (long long) cs.ctx, (long long) cs.profile,
                        chunk, rows, rows_kernels(rows, chunk, room) ? "rows" : "general", f16, f16 / least16, q8,
                        q8 / least8, q8 / f16);
            if (rows == 1) continue;
            // each token of the window on its own, at its own position, for both formats.  The kernels' shapes decide:
            // a token alone takes the rows kernels, and a window of `chunk` tokens or more the general ones.
            for (int fmt = 0; fmt < 2; ++fmt) {
                if (fmt == 0)
                    poly::attend(q, k16, v16, pos, rows, score, chunk, out, kDim, kHeads, kKv, scale, g_st, 0, 0, false,
                                 at, room);
                for (int t = 0; t < rows; ++t) {
                    cudaStreamSynchronize(g_st);
                    cudaMemcpy(at, h_at.data() + 2 * t, 2 * sizeof(long long), cudaMemcpyHostToDevice);
                    float* o1 = one + (size_t) t * kHeads * kDim;
                    if (fmt == 0)
                        poly::attend(q + (size_t) t * kHeads * kDim, k16, v16, pos + t, 1, score, chunk, o1, kDim,
                                     kHeads, kKv, scale, g_st, 0, 0, false, at, room);
                    else
                        poly::attend_q8(q + (size_t) t * kHeads * kDim, kq, vq, pos + t, 1, score, chunk, o1, kDim,
                                        kHeads, kKv, scale, g_st, 0, 0, false, at, room);
                }
                cudaStreamSynchronize(g_st);
                cudaMemcpy(at, h_at.data(), h_at.size() * sizeof(long long), cudaMemcpyHostToDevice);
                std::vector<uint32_t> a((size_t) rows * kHeads * kDim), b(a.size());
                cudaMemcpy(a.data(), fmt ? out2 : out, a.size() * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(b.data(), one, b.size() * 4, cudaMemcpyDeviceToHost);
                size_t differ = 0;
                for (size_t i = 0; i < a.size(); ++i) differ += a[i] != b[i];
                std::printf("    %s: the window and its tokens one at a time: %s", fmt ? "q8_0" : "fp16",
                            differ ? "differ" : "the same bits");
                if (differ) std::printf(" (%zu values)", differ);
                std::printf("\n");
                // a failure only where the window and a token alone take the same kernels
                bad += differ != 0 && fmt == 1 && rows_kernels(rows, chunk, room) == rows_kernels(1, chunk, room);
            }
        }
    }

    // T2: the kernels against the attention in doubles over the values each reads
    std::printf("T2 decode: the kernels against doubles (relative to the largest output value)\n");
    std::vector<uint16_t> hk16, hv16;
    std::vector<uint8_t> hkq, hvq;
    for (int64_t ctx : {int64_t(2048), int64_t(32768), int64_t(261000)}) {
        const int64_t n_rows = ctx + 4;
        hk16.resize((size_t) n_rows * kWidth);
        hv16.resize(hk16.size());
        hkq.resize((size_t) n_rows * kQ8Row);
        hvq.resize(hkq.size());
        cudaMemcpy(hk16.data(), k16, hk16.size() * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(hv16.data(), v16, hv16.size() * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(hkq.data(), kq, hkq.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(hvq.data(), vq, hvq.size(), cudaMemcpyDeviceToHost);
        std::vector<float> fk16 = f16_floats(hk16), fv16 = f16_floats(hv16);
        std::vector<float> fkq = q8_floats(hkq, n_rows), fvq = q8_floats(hvq, n_rows);
        const int64_t profile = ctx > 131072 ? 262144 : 49152;
        const int chunk = profile_chunk(profile);
        for (int rows : {1, 4}) {
            int room = 1024;
            while (room < ctx + rows) room *= 2;
            room = (int) std::min<int64_t>(room, profile);
            std::vector<long long> h_at((size_t) rows * 2);
            for (int t = 0; t < rows; ++t) h_at[(size_t) 2 * t] = h_at[(size_t) 2 * t + 1] = ctx + t;
            cudaStreamSynchronize(g_st);
            cudaMemcpy(at, h_at.data(), h_at.size() * sizeof(long long), cudaMemcpyHostToDevice);
            poly::attend(q, k16, v16, ctx, rows, score, chunk, out, kDim, kHeads, kKv, scale, g_st, 0, 0, false, at, room);
            poly::attend_q8(q, kq, vq, ctx, rows, score, chunk, out2, kDim, kHeads, kKv, scale, g_st, 0, 0, false, at,
                            room);
            cudaStreamSynchronize(g_st);
            std::vector<float> o16((size_t) rows * kHeads * kDim), o8(o16.size());
            cudaMemcpy(o16.data(), out, o16.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(o8.data(), out2, o8.size() * 4, cudaMemcpyDeviceToHost);
            Diff k16d, k8d, quant, kern;
            for (int t = 0; t < rows; ++t) {
                std::vector<double> r16((size_t) kHeads * kDim), r8(r16.size());
                const float* qt = hq.data() + (size_t) t * kHeads * kDim;
                ref_attend(qt, fk16.data(), fv16.data(), ctx + t, scale, r16.data());
                ref_attend(qt, fkq.data(), fvq.data(), ctx + t, scale, r8.data());
                const size_t o = (size_t) t * kHeads * kDim, m = (size_t) kHeads * kDim;
                const Diff a = diff(o16.data() + o, r16.data(), m), b = diff(o8.data() + o, r8.data(), m),
                           c = diff(r8.data(), r16.data(), m), e = diff(o8.data() + o, o16.data() + o, m);
                auto add = [](Diff& to, const Diff& x) {
                    to.max_abs = std::max(to.max_abs, x.max_abs);
                    to.max_ref = std::max(to.max_ref, x.max_ref);
                    to.rms += x.rms;
                    to.rms_ref += x.rms_ref;
                };
                add(k16d, a);
                add(k8d, b);
                add(quant, c);
                add(kern, e);
            }
            std::printf("  ctx %6lld, %d tok (%s kernels): fp16 kernel vs doubles max %.2e rms %.2e | q8_0 kernel vs "
                        "doubles max %.2e rms %.2e | q8_0 vs fp16 values (doubles) rms %.2e, kernels rms %.2e\n",
                        (long long) ctx, rows, rows_kernels(rows, chunk, room) ? "rows" : "general", k16d.rel_max(),
                        k16d.rel_rms(), k8d.rel_max(), k8d.rel_rms(), quant.rel_rms(), kern.rel_rms());
            bad += k8d.rel_max() > 1e-4;
        }
    }

    // prompt batches: attend_batch_q8 against attend_batch over the halves it hands its products, and the times
    std::printf("prompt batch of %d tokens ending the cache (attend_batch / attend_batch_q8, stage of 8,192 rows)\n",
                batch);
    const int stage_rows = 8192;
    uint16_t* ks = (uint16_t*) dev_random((size_t) stage_rows * kWidth * 2, false);
    uint16_t* vs = (uint16_t*) dev_random((size_t) stage_rows * kWidth * 2, false);
    uint16_t* kd = (uint16_t*) dev_random((size_t) most * kWidth * 2, false);
    uint16_t* vd = (uint16_t*) dev_random((size_t) most * kWidth * 2, false);
    poly::kv_dequant_q8(kq, 0, (int) most, kDim, kKv, kd, g_st);
    poly::kv_dequant_q8(vq, 0, (int) most, kDim, kKv, vd, g_st);
    float* work = (float*) dev_random((size_t) batch * kHeads * (kDim + kDim / 2 + 3) * 4 + 64, true);
    for (int64_t ctx : {int64_t(8192), int64_t(32768), int64_t(131072), int64_t(262144)}) {
        for (int n : {batch, 32}) {
            const int64_t bpos = ctx - n;
            const int reps = g_reps;
            g_reps = std::max(3, reps / 64);
            const double f16 = time_us([&] {
                poly::attend_batch(blas_handle(), q, k16, v16, bpos, n, score, score_floats, work, out, kDim, kHeads,
                                   kKv, scale, g_st);
            });
            const double q8 = time_us([&] {
                poly::attend_batch_q8(blas_handle(), q, kq, vq, bpos, n, score, score_floats, work, out2, kDim, kHeads,
                                      kKv, scale, g_st, ks, vs, stage_rows);
            });
            g_reps = reps;
            poly::attend_batch(blas_handle(), q, kd, vd, bpos, n, score, score_floats, work, out, kDim, kHeads, kKv,
                               scale, g_st);
            cudaStreamSynchronize(g_st);
            std::vector<float> a((size_t) n * kHeads * kDim), b(a.size());
            cudaMemcpy(a.data(), out, a.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(b.data(), out2, b.size() * 4, cudaMemcpyDeviceToHost);
            const bool same = std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
            const Diff d = diff(b.data(), a.data(), a.size());
            const int block = poly::attend_batch_block(score_floats, n, kHeads);
            std::printf("  ctx %6lld, %4d tokens: fp16 %8.2f ms | q8_0 %8.2f ms (%.3f x) | against attend_batch over its "
                        "halves: %s (block %d rows%s)\n", (long long) ctx, n, f16 / 1e3, q8 / 1e3, q8 / f16,
                        same ? "the same bits" : "differs", block, block > stage_rows ? ", larger than the stage" : "");
            if (!same) std::printf("    largest difference %.2e of %.2e, rms %.2e\n", d.max_abs, d.max_ref, d.rel_rms());
            bad += !same && block <= stage_rows;
        }
    }
    std::printf("attn-q8 %s\n", bad ? "FAILED" : "passed");
    return bad ? 1 : 0;
}

// poly_kernel_bench attn-dq8 [most]: a decode call's attention over q8_0 caches (attend_q8(), the engine's call: the
// position read on the card, the rows sized by room()) for 1, 2, 4 and 8 tokens at contexts up to `most` positions
// (default 100,000: it needs 2 x most x 544 bytes of VRAM and the f16 sources while it fills them, which is why it is
// separate from attn-q8): the time a call with the rows kernels (STRATA_POLY_FLASH=0) and with the flash kernel, the
// bytes of K and V a second the latter reads, and each one's largest error against the attention in doubles over the
// same q8_0 values.  The flash kernel's window against its tokens one at a time, bit for bit.
int attn_dq8_bench(int64_t most) {
    const float scale = 1.f / 16.f;
    uint8_t* kq = (uint8_t*) dev_random((size_t) most * kQ8Row, false);
    uint8_t* vq = (uint8_t*) dev_random((size_t) most * kQ8Row, false);
    uint16_t* k16 = (uint16_t*) dev_random((size_t) most * kWidth * 2, false);
    uint16_t* v16 = (uint16_t*) dev_random((size_t) most * kWidth * 2, false);
    fill_caches(k16, kq, most, true, 11);
    fill_caches(v16, vq, most, false, 12);
    const int batch = 8;
    std::vector<float> hq((size_t) batch * kHeads * kDim);
    {
        std::mt19937 rng(13);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& x : hq) x = nd(rng);
    }
    float* q = (float*) dev_random(hq.size() * 4, true);
    cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice);
    const size_t score_floats = size_t(16) << 20;
    float* score = (float*) dev_random(score_floats * 4, true);
    float* out_old = (float*) dev_random((size_t) batch * kHeads * kDim * 4, true);
    float* out = (float*) dev_random((size_t) batch * kHeads * kDim * 4, true);
    float* one = (float*) dev_random((size_t) batch * kHeads * kDim * 4, true);
    long long* at = (long long*) dev_random(batch * 2 * sizeof(long long), false);
    uint8_t* src = (uint8_t*) dev_random((size_t) 64 << 20, false);
    uint8_t* dst = (uint8_t*) dev_random((size_t) 64 << 20, false);
    const double copy_us = time_us([&] { cudaMemcpyAsync(dst, src, (size_t) 64 << 20, cudaMemcpyDeviceToDevice, g_st); });
    const double rate = 2.0 * (64 << 20) / (copy_us * 1e3);   // GB/s read + written
    std::printf("copy rate %.1f GB/s (read + write); a row of K and V in q8_0: %zu bytes\n", rate, 2 * kQ8Row);
    std::vector<uint8_t> hkq((size_t) most * kQ8Row), hvq(hkq.size());
    cudaMemcpy(hkq.data(), kq, hkq.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(hvq.data(), vq, hvq.size(), cudaMemcpyDeviceToHost);
    const auto q8_value = [&](const std::vector<uint8_t>& c, int64_t r, int g, int d) {
        const uint8_t* head = c.data() + (size_t) r * kQ8Row + (size_t) g * kHb;
        uint16_t sh;
        std::memcpy(&sh, head + kDim + 2 * (d / 32), 2);
        const int s = sh >> 15, e = (sh >> 10) & 31, m = sh & 1023;
        const double sc = e == 0 ? std::ldexp((double) m, -24) : std::ldexp((double) (m + 1024), e - 25);
        return (double) (int8_t) head[d] * (s ? -sc : sc);
    };
    int bad = 0;
    for (int64_t ctx : {int64_t(8192), int64_t(32768), int64_t(65536), most - 16}) {
        if (ctx > most - 16) continue;
        const int64_t profile = 262144;
        const int chunk = profile_chunk(profile);
        for (int rows : {1, 2, 4, 8}) {
            const int64_t pos = ctx;
            int room = 1024;
            while (room < pos + rows) room *= 2;
            room = (int) std::min<int64_t>(room, profile);
            std::vector<long long> h_at((size_t) rows * 2);
            for (int t = 0; t < rows; ++t) h_at[(size_t) 2 * t] = h_at[(size_t) 2 * t + 1] = pos + t;
            cudaStreamSynchronize(g_st);
            cudaMemcpy(at, h_at.data(), h_at.size() * sizeof(long long), cudaMemcpyHostToDevice);
            const int reps = g_reps;
            g_reps = std::max(10, (int) (g_reps * 8192 / ctx));
            const auto call = [&](float* o) {
                poly::attend_q8(q, kq, vq, pos, rows, score, chunk, o, kDim, kHeads, kKv, scale, g_st, 0, 0, false, at,
                                room);
            };
            _putenv("STRATA_POLY_FLASH=0");
            const double old_us = time_us([&] { call(out_old); });
            _putenv("STRATA_POLY_FLASH=1");
            const double us = time_us([&] { call(out); });
            g_reps = reps;
            const double bytes = (double) (pos + rows) * kQ8Row * 2;
            std::printf("  ctx %6lld, %d tok: rows kernels %8.1f us | flash %8.1f us (%.2fx), %6.1f GB/s of K and V (%.0f "
                        "MB), %.2f x the least\n",
                        (long long) ctx, rows, old_us, us, old_us / us, bytes / (us * 1e3), bytes / 1e6,
                        us / (bytes / (rate * 1e3)));
            // against the doubles: a few tokens and heads
            std::vector<float> a((size_t) rows * kHeads * kDim), b(a.size());
            cudaMemcpy(a.data(), out_old, a.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(b.data(), out, b.size() * 4, cudaMemcpyDeviceToHost);
            double err_old = 0, err_new = 0, mag = 0;
            if (rows == 1 || rows == 4)
                for (int t : {0, rows - 1}) {
                    const int64_t last = pos + t;
                    for (int h : {0, 7, 12}) {
                        const int g = h / (kHeads / kKv);
                        const float* qh = &hq[((size_t) t * kHeads + h) * kDim];
                        std::vector<double> w((size_t) last + 1);
                        double mx = -1e300;
                        for (int64_t j = 0; j <= last; ++j) {
                            double s = 0;
                            for (int d = 0; d < kDim; ++d) s += (double) qh[d] * q8_value(hkq, j, g, d);
                            w[(size_t) j] = s * scale;
                            mx = std::max(mx, w[(size_t) j]);
                        }
                        double sum = 0;
                        for (auto& x : w) {
                            x = std::exp(x - mx);
                            sum += x;
                        }
                        for (int d = 0; d < kDim; d += 7) {
                            double o = 0;
                            for (int64_t j = 0; j <= last; ++j) o += w[(size_t) j] * q8_value(hvq, j, g, d);
                            o /= sum;
                            const size_t at_i = ((size_t) t * kHeads + h) * kDim + d;
                            err_old = std::max(err_old, std::fabs(a[at_i] - o));
                            err_new = std::max(err_new, std::fabs(b[at_i] - o));
                            mag = std::max(mag, std::fabs(o));
                        }
                    }
                }
            if (mag > 0)
                std::printf("      max |error| against doubles (values up to %.3f): rows kernels %.2e, flash %.2e\n", mag,
                            err_old, err_new);
            if (rows == 1) continue;
            // the flash window's tokens against each alone
            for (int t = 0; t < rows; ++t) {
                cudaStreamSynchronize(g_st);
                cudaMemcpy(at, h_at.data() + 2 * t, 2 * sizeof(long long), cudaMemcpyHostToDevice);
                poly::attend_q8(q + (size_t) t * kHeads * kDim, kq, vq, pos + t, 1, score, chunk,
                                one + (size_t) t * kHeads * kDim, kDim, kHeads, kKv, scale, g_st, 0, 0, false, at, room);
            }
            cudaStreamSynchronize(g_st);
            cudaMemcpy(at, h_at.data(), h_at.size() * sizeof(long long), cudaMemcpyHostToDevice);
            call(out);
            cudaStreamSynchronize(g_st);
            std::vector<uint32_t> x((size_t) rows * kHeads * kDim), y(x.size());
            cudaMemcpy(x.data(), out, x.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(y.data(), one, y.size() * 4, cudaMemcpyDeviceToHost);
            size_t differ = 0;
            for (size_t i = 0; i < x.size(); ++i) differ += x[i] != y[i];
            std::printf("      the flash window and its tokens one at a time: %s\n",
                        differ ? "DIFFER" : "the same bits");
            bad += differ != 0;
        }
    }
    // fp16 caches (attend()): 1 and 4 tokens, the same
    std::vector<uint16_t> hk16((size_t) most * kWidth), hv16(hk16.size());
    cudaMemcpy(hk16.data(), k16, hk16.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(hv16.data(), v16, hv16.size() * 2, cudaMemcpyDeviceToHost);
    const auto half_value = [](uint16_t h) {
        const int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
        const double v = e == 0 ? std::ldexp((double) m, -24) : std::ldexp((double) (m + 1024), e - 25);
        return s ? -v : v;
    };
    std::printf("fp16 caches (attend())\n");
    for (int64_t ctx : {int64_t(8192), most - 16}) {
        const int chunk = profile_chunk(262144);
        for (int rows : {1, 4}) {
            const int64_t pos = ctx;
            int room = 1024;
            while (room < pos + rows) room *= 2;
            std::vector<long long> h_at((size_t) rows * 2);
            for (int t = 0; t < rows; ++t) h_at[(size_t) 2 * t] = h_at[(size_t) 2 * t + 1] = pos + t;
            cudaStreamSynchronize(g_st);
            cudaMemcpy(at, h_at.data(), h_at.size() * sizeof(long long), cudaMemcpyHostToDevice);
            const int reps = g_reps;
            g_reps = std::max(10, (int) (g_reps * 8192 / ctx));
            const auto call = [&](float* o) {
                poly::attend(q, k16, v16, pos, rows, score, chunk, o, kDim, kHeads, kKv, scale, g_st, 0, 0, false, at,
                             room);
            };
            _putenv("STRATA_POLY_FLASH=0");
            const double old_us = time_us([&] { call(out_old); });
            _putenv("STRATA_POLY_FLASH=1");
            const double us = time_us([&] { call(out); });
            g_reps = reps;
            std::vector<float> a((size_t) rows * kHeads * kDim), b(a.size());
            cudaMemcpy(a.data(), out_old, a.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(b.data(), out, b.size() * 4, cudaMemcpyDeviceToHost);
            double err_old = 0, err_new = 0, mag = 0;
            for (int t : {0, rows - 1}) {
                const int64_t last = pos + t;
                for (int h : {0, 7, 12}) {
                    const int g = h / (kHeads / kKv);
                    const float* qh = &hq[((size_t) t * kHeads + h) * kDim];
                    std::vector<double> w((size_t) last + 1);
                    double mx = -1e300;
                    for (int64_t j = 0; j <= last; ++j) {
                        double s = 0;
                        for (int d = 0; d < kDim; ++d)
                            s += (double) qh[d] * half_value(hk16[(size_t) j * kWidth + g * kDim + d]);
                        w[(size_t) j] = s * scale;
                        mx = std::max(mx, w[(size_t) j]);
                    }
                    double sum = 0;
                    for (auto& x : w) {
                        x = std::exp(x - mx);
                        sum += x;
                    }
                    for (int d = 0; d < kDim; d += 7) {
                        double o = 0;
                        for (int64_t j = 0; j <= last; ++j)
                            o += w[(size_t) j] * half_value(hv16[(size_t) j * kWidth + g * kDim + d]);
                        o /= sum;
                        const size_t at_i = ((size_t) t * kHeads + h) * kDim + d;
                        err_old = std::max(err_old, std::fabs(a[at_i] - o));
                        err_new = std::max(err_new, std::fabs(b[at_i] - o));
                        mag = std::max(mag, std::fabs(o));
                    }
                }
            }
            std::printf("  ctx %6lld, %d tok: rows kernels %8.1f us | flash %8.1f us (%.2fx); max |error| against doubles "
                        "(values up to %.3f): %.2e, flash %.2e\n",
                        (long long) ctx, rows, old_us, us, old_us / us, mag, err_old, err_new);
            if (rows > 1) {
                for (int t = 0; t < rows; ++t) {
                    cudaStreamSynchronize(g_st);
                    cudaMemcpy(at, h_at.data() + 2 * t, 2 * sizeof(long long), cudaMemcpyHostToDevice);
                    poly::attend(q + (size_t) t * kHeads * kDim, k16, v16, pos + t, 1, score, chunk,
                                 one + (size_t) t * kHeads * kDim, kDim, kHeads, kKv, scale, g_st, 0, 0, false, at, room);
                }
                cudaStreamSynchronize(g_st);
                cudaMemcpy(at, h_at.data(), h_at.size() * sizeof(long long), cudaMemcpyHostToDevice);
                call(out);
                cudaStreamSynchronize(g_st);
                std::vector<uint32_t> x((size_t) rows * kHeads * kDim), y(x.size());
                cudaMemcpy(x.data(), out, x.size() * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(y.data(), one, y.size() * 4, cudaMemcpyDeviceToHost);
                size_t differ = 0;
                for (size_t i = 0; i < x.size(); ++i) differ += x[i] != y[i];
                std::printf("      the flash window and its tokens one at a time: %s\n", differ ? "DIFFER" : "the same bits");
                bad += differ != 0;
            }
        }
    }
    return bad ? 1 : 0;
}

// Phase 3 (poly_kernel_bench attn-kv-mix): attend_kv() and attend_batch_kv(), the engine's calls, with the keys and the
// values each in fp16 or q8_0 (--kv, --kv-k, --kv-v).  Both fp16 against attend() and both q8_0 against attend_q8(), bit
// for bit: the same launches.  Each of the four against the attention in doubles over the values it reads; a window
// against its tokens one at a time, bit for bit where they take the same kernels; with and without the scratch count of
// STRATA_POLY_ATTN_FIT.  A prompt batch's attend_batch_kv() against attend_batch() over the halves its products read
// (a q8_0 cache's dequantized, an fp16 one's as it is), bit for bit where the blocks are the same.
int attn_kv_mix_bench() {
    const int64_t most = 131072 + 16;
    const float scale = 1.f / 16.f;
    uint16_t* k16 = (uint16_t*) dev_random((size_t) most * kWidth * 2, false);
    uint16_t* v16 = (uint16_t*) dev_random((size_t) most * kWidth * 2, false);
    uint8_t* kq = (uint8_t*) dev_random((size_t) most * kQ8Row, false);
    uint8_t* vq = (uint8_t*) dev_random((size_t) most * kQ8Row, false);
    fill_caches(k16, kq, most, true, 21);
    fill_caches(v16, vq, most, false, 22);
    const int most_rows = 8, batch = 1024;
    std::vector<float> hq((size_t) batch * kHeads * kDim);
    {
        std::mt19937 rng(23);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& x : hq) x = nd(rng);
    }
    float* q = (float*) dev_random(hq.size() * 4, true);
    cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice);
    const size_t score_floats = size_t(16) << 20;
    float* score = (float*) dev_random(score_floats * 4, true);
    float* out = (float*) dev_random((size_t) batch * kHeads * kDim * 4, true);
    float* out2 = (float*) dev_random((size_t) batch * kHeads * kDim * 4, true);
    float* one = (float*) dev_random((size_t) most_rows * kHeads * kDim * 4, true);
    long long* at = (long long*) dev_random(most_rows * 2 * sizeof(long long), false);
    const auto bits = [](const float* a, const float* b, size_t n) {
        std::vector<uint32_t> x(n), y(n);
        cudaStreamSynchronize(g_st);
        cudaMemcpy(x.data(), a, n * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(y.data(), b, n * 4, cudaMemcpyDeviceToHost);
        size_t differ = 0;
        for (size_t i = 0; i < n; ++i) differ += x[i] != y[i];
        return differ;
    };
    const char* names[4] = {"k fp16 v fp16", "k q8_0 v fp16", "k fp16 v q8_0", "k q8_0 v q8_0"};
    int bad = 0;
    struct Case {
        int64_t ctx, profile;
    } cases[] = {{2048, 49152}, {45355, 49152}, {45355, 262144}, {131000, 131072}, {131000, 262144}};
    std::printf("decode: attend_kv against attend()/attend_q8(), and a window against its tokens one at a time\n");
    for (const Case& cs : cases) {
        const int chunk = profile_chunk(cs.profile);
        const size_t fit_floats = (size_t) chunk * kHeads * cs.profile;
        for (size_t sf : {size_t(0), fit_floats})
            for (int rows : {1, 3, 4, 8}) {
                const int64_t pos = cs.ctx;
                int room = 1024;
                while (room < pos + rows) room *= 2;
                room = (int) std::min<int64_t>(room, cs.profile);
                std::vector<long long> h_at((size_t) rows * 2);
                for (int t = 0; t < rows; ++t) h_at[(size_t) 2 * t] = h_at[(size_t) 2 * t + 1] = pos + t;
                cudaStreamSynchronize(g_st);
                cudaMemcpy(at, h_at.data(), h_at.size() * sizeof(long long), cudaMemcpyHostToDevice);
                std::printf("  ctx %6lld (profile %6lld, chunk %2d, %s) %d tok:", (long long) cs.ctx,
                            (long long) cs.profile, chunk, sf ? "fit" : "old", rows);
                const size_t n = (size_t) rows * kHeads * kDim;
                for (int mix = 0; mix < 4; ++mix) {
                    const bool kq8 = mix & 1, vq8 = mix & 2;
                    const void* kc = kq8 ? (const void*) kq : k16;
                    const void* vc = vq8 ? (const void*) vq : v16;
                    poly::attend_kv(q, kc, kq8, vc, vq8, pos, rows, score, chunk, out, kDim, kHeads, kKv, scale, g_st, 0,
                                    0, false, at, room, sf);
                    // both one format: the function the engine had for it
                    size_t same_fn = 0;
                    if (mix == 0 || mix == 3) {
                        if (mix == 0)
                            poly::attend(q, k16, v16, pos, rows, score, chunk, out2, kDim, kHeads, kKv, scale, g_st, 0, 0,
                                         false, at, room, nullptr, sf);
                        else
                            poly::attend_q8(q, kq, vq, pos, rows, score, chunk, out2, kDim, kHeads, kKv, scale, g_st, 0, 0,
                                            false, at, room, sf);
                        same_fn = bits(out, out2, n);
                        bad += same_fn != 0;
                    }
                    size_t alone = 0;
                    if (rows > 1) {
                        for (int t = 0; t < rows; ++t) {
                            cudaStreamSynchronize(g_st);
                            cudaMemcpy(at, h_at.data() + 2 * t, 2 * sizeof(long long), cudaMemcpyHostToDevice);
                            poly::attend_kv(q + (size_t) t * kHeads * kDim, kc, kq8, vc, vq8, pos + t, 1, score, chunk,
                                            one + (size_t) t * kHeads * kDim, kDim, kHeads, kKv, scale, g_st, 0, 0, false,
                                            at, room, sf);
                        }
                        cudaStreamSynchronize(g_st);
                        cudaMemcpy(at, h_at.data(), h_at.size() * sizeof(long long), cudaMemcpyHostToDevice);
                        alone = bits(out, one, n);
                        // a failure where the window takes the rows kernels as a token alone does (fit), or always
                        // where the old count takes them for both
                        bad += alone != 0 && (sf != 0 || rows_kernels(rows, chunk, room) == rows_kernels(1, chunk, room));
                    }
                    std::printf(" | %s:%s%s", names[mix],
                                mix == 0 || mix == 3 ? (same_fn ? " fn DIFFERS" : " fn same") : "",
                                rows > 1 ? (alone ? " alone differs" : " alone same") : "");
                }
                std::printf("\n");
            }
    }

    // against doubles over the values each format gives
    std::printf("against doubles (relative to the largest output value)\n");
    for (int64_t ctx : {int64_t(2048), int64_t(45355)}) {
        const int64_t n_rows = ctx + 4;
        std::vector<uint16_t> hk16((size_t) n_rows * kWidth), hv16(hk16.size());
        std::vector<uint8_t> hkq((size_t) n_rows * kQ8Row), hvq(hkq.size());
        cudaMemcpy(hk16.data(), k16, hk16.size() * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(hv16.data(), v16, hv16.size() * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(hkq.data(), kq, hkq.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(hvq.data(), vq, hvq.size(), cudaMemcpyDeviceToHost);
        const std::vector<float> fk16 = f16_floats(hk16), fv16 = f16_floats(hv16);
        const std::vector<float> fkq = q8_floats(hkq, n_rows), fvq = q8_floats(hvq, n_rows);
        const int chunk = profile_chunk(49152), rows = 4;
        int room = 1024;
        while (room < ctx + rows) room *= 2;
        std::vector<long long> h_at((size_t) rows * 2);
        for (int t = 0; t < rows; ++t) h_at[(size_t) 2 * t] = h_at[(size_t) 2 * t + 1] = ctx + t;
        cudaStreamSynchronize(g_st);
        cudaMemcpy(at, h_at.data(), h_at.size() * sizeof(long long), cudaMemcpyHostToDevice);
        std::vector<double> ref16((size_t) rows * kHeads * kDim);
        for (int t = 0; t < rows; ++t)
            ref_attend(hq.data() + (size_t) t * kHeads * kDim, fk16.data(), fv16.data(), ctx + t, scale,
                       ref16.data() + (size_t) t * kHeads * kDim);
        for (int mix = 0; mix < 4; ++mix) {
            const bool kq8 = mix & 1, vq8 = mix & 2;
            poly::attend_kv(q, kq8 ? (const void*) kq : k16, kq8, vq8 ? (const void*) vq : v16, vq8, ctx, rows, score,
                            chunk, out, kDim, kHeads, kKv, scale, g_st, 0, 0, false, at, room);
            cudaStreamSynchronize(g_st);
            std::vector<float> o((size_t) rows * kHeads * kDim);
            cudaMemcpy(o.data(), out, o.size() * 4, cudaMemcpyDeviceToHost);
            std::vector<double> ref(o.size());
            for (int t = 0; t < rows; ++t)
                ref_attend(hq.data() + (size_t) t * kHeads * kDim, kq8 ? fkq.data() : fk16.data(),
                           vq8 ? fvq.data() : fv16.data(), ctx + t, scale, ref.data() + (size_t) t * kHeads * kDim);
            const Diff kern = diff(o.data(), ref.data(), o.size()), fmt = diff(ref.data(), ref16.data(), o.size());
            std::printf("  ctx %6lld, %d tok, %s: kernel vs doubles max %.2e rms %.2e | its values vs fp16's (doubles) "
                        "rms %.3e\n", (long long) ctx, rows, names[mix], kern.rel_max(), kern.rel_rms(), fmt.rel_rms());
            bad += kern.rel_max() > 1e-4;
        }
    }

    // prompt batches
    std::printf("prompt batch of %d tokens ending the cache: attend_batch_kv against attend_batch over the halves its "
                "products read\n", batch);
    const int stage_rows = poly::kQ8StageRows;
    uint16_t* ks = (uint16_t*) dev_random((size_t) stage_rows * kWidth * 2, false);
    uint16_t* vs = (uint16_t*) dev_random((size_t) stage_rows * kWidth * 2, false);
    uint16_t* kd = (uint16_t*) dev_random((size_t) most * kWidth * 2, false);
    uint16_t* vd = (uint16_t*) dev_random((size_t) most * kWidth * 2, false);
    poly::kv_dequant_q8(kq, 0, (int) most, kDim, kKv, kd, g_st);
    poly::kv_dequant_q8(vq, 0, (int) most, kDim, kKv, vd, g_st);
    float* work = (float*) dev_random((size_t) batch * kHeads * (kDim + kDim / 2 + 3) * 4 + 64, true);
    for (int64_t ctx : {int64_t(8192), int64_t(32768), int64_t(131072)})
        for (int n : {batch, 32}) {
            const int64_t bpos = ctx - n;
            const int block = poly::attend_batch_block(score_floats, n, kHeads);
            std::printf("  ctx %6lld, %4d tokens (block %d rows%s):", (long long) ctx, n, block,
                        block > stage_rows ? ", larger than the stage" : "");
            for (int mix = 0; mix < 4; ++mix) {
                const bool kq8 = mix & 1, vq8 = mix & 2;
                const int reps = g_reps;
                g_reps = std::max(3, reps / 64);
                const double us = time_us([&] {
                    poly::attend_batch_kv(blas_handle(), q, kq8 ? (const void*) kq : k16, kq8,
                                          vq8 ? (const void*) vq : v16, vq8, bpos, n, score, score_floats, work, out2,
                                          kDim, kHeads, kKv, scale, g_st, kq8 ? ks : nullptr, vq8 ? vs : nullptr,
                                          stage_rows);
                });
                g_reps = reps;
                poly::attend_batch(blas_handle(), q, kq8 ? kd : k16, vq8 ? vd : v16, bpos, n, score, score_floats, work,
                                   out, kDim, kHeads, kKv, scale, g_st);
                const size_t differ = bits(out, out2, (size_t) n * kHeads * kDim);
                std::printf(" | %s %.2f ms %s", names[mix], us / 1e3, differ ? "DIFFERS" : "same bits");
                bad += differ != 0 && (block <= stage_rows || mix == 0);
            }
            std::printf("\n");
        }
    std::printf("attn-kv-mix %s\n", bad ? "FAILED" : "passed");
    return bad ? 1 : 0;
}

// T3 (poly_kernel_bench attn-q8-real FILE): a real model's keys, values and queries (strata-poly --dump-kv), each
// attention layer's last token: attend() over its fp16 cache against attend_q8() over the same values quantized, and
// in doubles the share of the error the keys and the values each bring.
int attn_q8_real(const char* path) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::printf("cannot read %s\n", path);
        return 1;
    }
    char magic[4];
    uint32_t version = 0;
    int64_t head[3];
    if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "PKVD", 4) != 0 || std::fread(&version, 4, 1, f) != 1 ||
        version != 1 || std::fread(head, 8, 3, f) != 3) {
        std::printf("%s is not a --dump-kv file\n", path);
        return 1;
    }
    const int64_t rows = head[0], row_bytes = head[1], qn = head[2];
    const int layers = (int) (row_bytes / (kWidth * 2 * 2)), q_layers = (int) (qn / (kHeads * kDim));
    std::printf("%s: %lld positions, %d attention layers, queries of %d of them (Qwen3.5/3.6 shapes)\n", path,
                (long long) rows, layers, q_layers);
    std::vector<float> qs((size_t) qn);
    if (std::fread(qs.data(), 4, qs.size(), f) != qs.size()) return 1;
    const size_t cache_b = (size_t) rows * kWidth * 2;
    std::vector<uint16_t> hk(cache_b / 2), hv(cache_b / 2);
    uint16_t* k16 = (uint16_t*) dev_random(cache_b, false);
    uint16_t* v16 = (uint16_t*) dev_random(cache_b, false);
    uint8_t* kq = (uint8_t*) dev_random((size_t) rows * kQ8Row, false);
    uint8_t* vq = (uint8_t*) dev_random((size_t) rows * kQ8Row, false);
    float* fl = (float*) dev_random((size_t) rows * kWidth * 4, true);
    float* q = (float*) dev_random((size_t) kHeads * kDim * 4, true);
    const size_t score_floats = size_t(16) << 20;
    float* score = (float*) dev_random(score_floats * 4, true);
    float* out = (float*) dev_random((size_t) kHeads * kDim * 4, true);
    float* out2 = (float*) dev_random((size_t) kHeads * kDim * 4, true);
    const int64_t profile = rows <= 49152 ? 49152 : rows <= 131072 ? 131072 : 262144;
    const int chunk = profile_chunk(profile);
    const float scale = 1.f / 16.f;
    const int64_t p = rows - 1;   // the last token: its own row is the cache's last
    int bad = 0;
    std::printf("layer | q8_0 vs fp16 kernels: rms   max | doubles: K and V q8_0  K only  V only (rms) | attention "
                "weights: KL(fp16 || K q8_0) mean max, largest weight\n");
    for (int l = 0; l < q_layers && l < layers; ++l) {
        if (std::fread(hk.data(), 1, cache_b, f) != cache_b || std::fread(hv.data(), 1, cache_b, f) != cache_b) return 1;
        cudaMemcpy(k16, hk.data(), cache_b, cudaMemcpyHostToDevice);
        cudaMemcpy(v16, hv.data(), cache_b, cudaMemcpyHostToDevice);
        const std::vector<float> fk = f16_floats(hk), fv = f16_floats(hv);
        // the q8_0 cache of the same values (the engine would quantize the floats before fp16 rounds them)
        cudaMemcpy(fl, fk.data(), fk.size() * 4, cudaMemcpyHostToDevice);
        poly::store_q8(fl, kq, kDim, kKv, 0, (int) rows, g_st);
        cudaStreamSynchronize(g_st);
        cudaMemcpy(fl, fv.data(), fv.size() * 4, cudaMemcpyHostToDevice);
        poly::store_q8(fl, vq, kDim, kKv, 0, (int) rows, g_st);
        cudaStreamSynchronize(g_st);
        const float* ql = qs.data() + (size_t) l * kHeads * kDim;
        cudaMemcpy(q, ql, (size_t) kHeads * kDim * 4, cudaMemcpyHostToDevice);
        poly::attend(q, k16, v16, p, 1, score, chunk, out, kDim, kHeads, kKv, scale, g_st);
        poly::attend_q8(q, kq, vq, p, 1, score, chunk, out2, kDim, kHeads, kKv, scale, g_st);
        cudaStreamSynchronize(g_st);
        std::vector<float> o16((size_t) kHeads * kDim), o8(o16.size());
        cudaMemcpy(o16.data(), out, o16.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(o8.data(), out2, o8.size() * 4, cudaMemcpyDeviceToHost);
        std::vector<uint8_t> hkq((size_t) rows * kQ8Row), hvq(hkq.size());
        cudaMemcpy(hkq.data(), kq, hkq.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(hvq.data(), vq, hvq.size(), cudaMemcpyDeviceToHost);
        const std::vector<float> fkq = q8_floats(hkq, rows), fvq = q8_floats(hvq, rows);
        const size_t m = (size_t) kHeads * kDim;
        std::vector<double> r16(m), rq(m), rk(m), rv(m);
        std::vector<double> w16[kHeads], wk[kHeads];
        ref_attend(ql, fk.data(), fv.data(), p, scale, r16.data(), w16);
        ref_attend(ql, fkq.data(), fvq.data(), p, scale, rq.data());
        ref_attend(ql, fkq.data(), fv.data(), p, scale, rk.data(), wk);
        ref_attend(ql, fk.data(), fvq.data(), p, scale, rv.data());
        double kl_mean = 0, kl_max = 0, top = 0;
        for (int hd = 0; hd < kHeads; ++hd) {
            double kl = 0;
            for (size_t j = 0; j < w16[hd].size(); ++j) {
                const double a = w16[hd][j], b = wk[hd][j];
                if (a > 0) kl += a * (std::log(a) - std::log(std::max(b, 1e-300)));
                top = std::max(top, a);
            }
            kl_mean += kl / kHeads;
            kl_max = std::max(kl_max, kl);
        }
        const Diff kern = diff(o8.data(), o16.data(), m), both = diff(rq.data(), r16.data(), m),
                   konly = diff(rk.data(), r16.data(), m), vonly = diff(rv.data(), r16.data(), m),
                   exact = diff(o8.data(), rq.data(), m);
        std::printf("  %2d  | %.2e %.2e | %.2e  %.2e  %.2e | %.2e %.2e, %.3f | q8_0 kernel vs doubles max %.1e\n", l,
                    kern.rel_rms(), kern.rel_max(), both.rel_rms(), konly.rel_rms(), vonly.rel_rms(), kl_mean, kl_max,
                    top, exact.rel_max());
        bad += kern.rel_rms() > 0.01 || exact.rel_max() > 1e-4;
    }
    std::fclose(f);
    std::printf("T3 %s (q8_0 against fp16: rms of a layer's output at most 1%%; the kernel against doubles 1e-4)\n",
                bad ? "FAILED" : "passed");
    return bad ? 1 : 0;
}

// Phase 2.5 (poly_kernel_bench attn-fit [reps], STRATA_POLY_ATTN_FIT): attend() and attend_q8() with the choice of
// kernels they always made (score_floats 0) and with the one that counts the scratch the engine really has
// (score_floats = chunk * 16 * the profile's context), launched as the engine launches a token or a window
// (recorded: the rows sized by room(), the positions read on the card), in the profiles 48K, 128K and 262K at
// contexts of 2K to 261K, for 1, 2, 3, 4 and 8 tokens.  For each: the launches of each choice and the most scratch
// they use, the times, each window against its tokens one at a time (bit for bit: a failure with the new choice),
// the two choices against each other (bit for bit where every token takes the same kernels in both), and at 45K,
// 131K and 261K both against the attention in doubles.  The floats past the profile's scratch must stay as they
// were.  Then a graph recorded with the new choice, replayed at three positions, against direct launches.

// a call's launches as attend() makes them (rows_tokens()), recorded: the span is `room` rows
struct Launch {
    int tokens;
    bool rows;       // the rows kernels, else the general ones
    size_t floats;   // the scratch it uses: scores, and for the rows kernels the parts' sums behind them
};

std::vector<Launch> attend_plan(int rows, int chunk, size_t fit, int room) {
    const bool on = [] {
        const char* v = std::getenv("STRATA_POLY_ATTN");
        return !(v != nullptr && v[0] == '0');
    }();
    const size_t parts = ((size_t) room + 511) / 512;
    const auto need = [&](int c) { return (size_t) c * kHeads * room + parts * c * kHeads * kDim; };
    std::vector<Launch> out;
    for (int first = 0; first < rows;) {
        const int rest = rows - first;
        int fast = 0;
        if (on && fit == 0) {
            const int c = std::min(chunk, rest);
            fast = need(c) <= (size_t) chunk * kHeads * room ? c : 0;
        } else if (on) {
            for (int c = rest; c > 0 && !fast; --c)
                if (need(c) <= fit) fast = c;
        }
        const int c = fast ? fast : std::min(chunk, rest);
        out.push_back({c, fast > 0, fast ? need(c) : (size_t) c * kHeads * room});
        first += c;
    }
    return out;
}

std::string plan_text(const std::vector<Launch>& p) {
    std::string s;
    for (const Launch& l : p) s += (s.empty() ? "" : "+") + std::string(l.rows ? "r" : "g") + std::to_string(l.tokens);
    return s;
}

// each token's kernels: true for the rows kernels
std::vector<bool> plan_tokens(const std::vector<Launch>& p) {
    std::vector<bool> t;
    for (const Launch& l : p) t.insert(t.end(), (size_t) l.tokens, l.rows);
    return t;
}

int engine_room(int64_t pos, int rows, int64_t profile) {
    int64_t room = 1024;   // Engine::room()
    while (room < pos + rows) room *= 2;
    return (int) std::min<int64_t>(room, profile);
}

int attn_fit_bench() {
    const int64_t most = 262144;
    const float scale = 1.f / 16.f;
    uint16_t* k16 = (uint16_t*) dev_random((size_t) most * kWidth * 2, false);
    uint16_t* v16 = (uint16_t*) dev_random((size_t) most * kWidth * 2, false);
    uint8_t* kq = (uint8_t*) dev_random((size_t) most * kQ8Row, false);
    uint8_t* vq = (uint8_t*) dev_random((size_t) most * kQ8Row, false);
    fill_caches(k16, kq, most, true, 11);
    fill_caches(v16, vq, most, false, 12);
    const int most_rows = 8;
    const size_t per_token = (size_t) kHeads * kDim;
    std::vector<float> hq((size_t) most_rows * per_token);
    {
        std::mt19937 rng(13);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& x : hq) x = nd(rng);
    }
    float* q = (float*) dev_random(hq.size() * 4, true);
    cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice);
    // the most scratch a profile has (128K and 262K: chunk * 16 * context is 16 Mi floats; 48K: 21 * 16 * 49,152),
    // twice that for the larger scratches tried at the end, and a guard behind it
    const size_t score_most = size_t(32) << 20, guard = size_t(1) << 20;
    float* score = nullptr;
    if (cudaMalloc(&score, (score_most + guard) * 4) != cudaSuccess) {
        std::printf("no VRAM for the scratch\n");
        return 1;
    }
    float *o_old = nullptr, *o_new = nullptr, *one = nullptr;
    cudaMalloc(&o_old, (size_t) most_rows * per_token * 4);
    cudaMalloc(&o_new, (size_t) most_rows * per_token * 4);
    cudaMalloc(&one, (size_t) most_rows * per_token * 4);
    long long* at = (long long*) dev_random(most_rows * 2 * sizeof(long long), false);
    const auto set_at = [&](int64_t pos, int rows) {
        std::vector<long long> h((size_t) rows * 2);
        for (int t = 0; t < rows; ++t) h[(size_t) 2 * t] = h[(size_t) 2 * t + 1] = pos + t;
        cudaStreamSynchronize(g_st);
        cudaMemcpy(at, h.data(), h.size() * sizeof(long long), cudaMemcpyHostToDevice);
    };
    const auto get = [&](const float* d, int rows) {
        std::vector<float> h((size_t) rows * per_token);
        cudaStreamSynchronize(g_st);
        cudaMemcpy(h.data(), d, h.size() * 4, cudaMemcpyDeviceToHost);
        return h;
    };
    // the two formats behind one call: fmt 0 fp16 (attend), 1 q8_0 (attend_q8)
    const auto call = [&](int fmt, int64_t pos, int rows, int chunk, size_t fit, const long long* a, int room,
                          const float* qq, float* out) {
        if (fmt == 0)
            poly::attend(qq, k16, v16, pos, rows, score, chunk, out, kDim, kHeads, kKv, scale, g_st, 0, 0, false, a,
                         room, nullptr, fit);
        else
            poly::attend_q8(qq, kq, vq, pos, rows, score, chunk, out, kDim, kHeads, kKv, scale, g_st, 0, 0, false, a,
                            room, fit);
    };
    const double copy_us = time_us([&] {
        cudaMemcpyAsync(score, k16, (size_t) 8192 * kWidth * 2 * 4, cudaMemcpyDeviceToDevice, g_st);
    });
    const double rate = (double) 8192 * kWidth * 2 * 4 * 2 / (copy_us * 1e3);
    std::printf("copy rate %.1f GB/s (read + write)\n", rate);

    // the doubles' reference of tokens 0..7 from a context, both formats, computed once a context
    const int64_t ref_rows = 261000 + most_rows;
    std::vector<float> fk16, fv16, fkq, fvq;
    {
        std::vector<uint16_t> h16((size_t) ref_rows * kWidth);
        std::vector<uint8_t> h8((size_t) ref_rows * kQ8Row);
        cudaMemcpy(h16.data(), k16, h16.size() * 2, cudaMemcpyDeviceToHost);
        fk16 = f16_floats(h16);
        cudaMemcpy(h16.data(), v16, h16.size() * 2, cudaMemcpyDeviceToHost);
        fv16 = f16_floats(h16);
        cudaMemcpy(h8.data(), kq, h8.size(), cudaMemcpyDeviceToHost);
        fkq = q8_floats(h8, ref_rows);
        cudaMemcpy(h8.data(), vq, h8.size(), cudaMemcpyDeviceToHost);
        fvq = q8_floats(h8, ref_rows);
    }
    struct Ref {
        int64_t ctx = -1;
        std::vector<double> r[2];   // fp16, q8_0: most_rows tokens
    };
    std::vector<Ref> refs;
    refs.reserve(8);
    const auto ref_of = [&](int64_t ctx) -> const Ref& {
        for (const Ref& r : refs)
            if (r.ctx == ctx) return r;
        Ref r;
        r.ctx = ctx;
        for (int fmt = 0; fmt < 2; ++fmt) {
            r.r[fmt].resize((size_t) most_rows * per_token);
            for (int t = 0; t < most_rows; ++t)
                ref_attend(hq.data() + (size_t) t * per_token, fmt ? fkq.data() : fk16.data(),
                           fmt ? fvq.data() : fv16.data(), ctx + t, scale, r.r[fmt].data() + (size_t) t * per_token);
        }
        refs.push_back(std::move(r));
        return refs.back();
    };

    int bad = 0;
    struct Case {
        int64_t ctx, profile;
        bool ref;
    } cases[] = {{2048, 49152, false},   {8192, 49152, false},    {45355, 49152, true},   {2048, 131072, false},
                 {45355, 131072, true},  {131000, 131072, true},  {2048, 262144, false},  {8192, 262144, false},
                 {45355, 262144, true},  {131000, 262144, true},  {261000, 262144, true}};
    std::printf("decode (attend / attend_q8) with the old choice of kernels and the new (STRATA_POLY_ATTN_FIT), "
                "us a call;\nlaunches: r = rows kernels, g = general, with their tokens; scratch: the most floats a "
                "launch uses, against the profile's\n");
    for (const Case& cs : cases) {
        const int chunk = profile_chunk(cs.profile);
        const size_t fit = (size_t) chunk * kHeads * cs.profile;
        cudaMemset(score + fit, 0xA5, (score_most + guard - fit) * 4);
        std::printf("ctx %lld, profile %lld (chunk %d, scratch %zu floats = %.1f MiB)\n", (long long) cs.ctx,
                    (long long) cs.profile, chunk, fit, fit * 4 / 1048576.0);
        for (int rows : {1, 2, 3, 4, 8}) {
            const int64_t pos = cs.ctx;
            const int room = engine_room(pos, rows, cs.profile);
            const std::vector<Launch> p_old = attend_plan(rows, chunk, 0, room), p_new = attend_plan(rows, chunk, fit, room);
            size_t most_old = 0, most_new = 0;
            for (const Launch& l : p_old) most_old = std::max(most_old, l.floats);
            for (const Launch& l : p_new) most_new = std::max(most_new, l.floats);
            bad += most_new > fit || most_old > fit;
            set_at(pos, rows);
            const int reps = g_reps;
            g_reps = std::max(10, (int) (g_reps * 8192 / cs.ctx));
            double us[2][2];   // [fmt][old, new]
            for (int fmt = 0; fmt < 2; ++fmt)
                for (int nw = 0; nw < 2; ++nw)
                    us[fmt][nw] = time_us([&] { call(fmt, pos, rows, chunk, nw ? fit : 0, at, room, q, nw ? o_new : o_old); });
            g_reps = reps;
            std::printf("  %d tok, room %6d: old %-11s new %-11s | scratch old %5.1f%% new %5.1f%% | fp16 %8.1f -> "
                        "%8.1f us (%.3f) | q8_0 %8.1f -> %8.1f us (%.3f)\n",
                        rows, room, plan_text(p_old).c_str(), plan_text(p_new).c_str(), 100.0 * most_old / fit,
                        100.0 * most_new / fit, us[0][0], us[0][1], us[0][1] / us[0][0], us[1][0], us[1][1],
                        us[1][1] / us[1][0]);
            const std::vector<bool> t_old = plan_tokens(p_old), t_new = plan_tokens(p_new);
            for (int fmt = 0; fmt < 2; ++fmt) {
                set_at(pos, rows);
                call(fmt, pos, rows, chunk, 0, at, room, q, o_old);
                call(fmt, pos, rows, chunk, fit, at, room, q, o_new);
                const std::vector<float> a = get(o_old, rows), b = get(o_new, rows);
                // the window's tokens one at a time, with the new choice
                for (int t = 0; t < rows; ++t) {
                    set_at(pos + t, 1);
                    call(fmt, pos + t, 1, chunk, fit, at, room, q + (size_t) t * per_token, one + (size_t) t * per_token);
                }
                const std::vector<float> c = get(one, rows);
                const bool alone = std::memcmp(b.data(), c.data(), b.size() * 4) == 0;
                bad += !alone;
                const bool same_kernels = t_old == t_new;
                const bool same_bits = std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
                bad += same_kernels && !same_bits;
                const Diff d = diff(b.data(), a.data(), a.size());
                std::printf("    %s: window = tokens alone: %s | new against old: %s", fmt ? "q8_0" : "fp16",
                            alone ? "same bits" : "DIFFER", same_bits ? "same bits" : "differ");
                if (!same_bits)
                    std::printf(" (max %.2e of %.2e = %.2e, rms %.2e; %s)", d.max_abs, d.max_ref, d.rel_max(), d.rel_rms(),
                                same_kernels ? "SAME KERNELS" : "rows kernels against general");
                if (cs.ref) {
                    const Ref& r = ref_of(cs.ctx);
                    const Diff ro = diff(a.data(), r.r[fmt].data(), a.size()), rn = diff(b.data(), r.r[fmt].data(), b.size());
                    std::printf(" | vs doubles: old max %.2e rms %.2e, new max %.2e rms %.2e", ro.rel_max(), ro.rel_rms(),
                                rn.rel_max(), rn.rel_rms());
                    bad += rn.rel_max() > 1e-4;
                }
                std::printf("\n");
            }
        }
        std::vector<uint32_t> g((score_most + guard - fit));
        cudaStreamSynchronize(g_st);
        cudaMemcpy(g.data(), score + fit, g.size() * 4, cudaMemcpyDeviceToHost);
        size_t touched = 0;
        for (uint32_t w : g) touched += w != 0xA5A5A5A5u;
        std::printf("  the %zu floats past the scratch: %s\n", g.size(), touched ? "WRITTEN" : "untouched");
        bad += touched != 0;
    }

    // what a larger scratch than the engine's would give at 262K, the way left aside: the times of the new choice with
    // 1.5 and 2 times the 64 MiB (+32 and +64 MiB of VRAM)
    std::printf("a larger scratch (not what the engine has): the new choice with more floats, profile 262144\n");
    for (int64_t ctx : {int64_t(131000), int64_t(261000)})
        for (int rows : {3, 4, 8}) {
            const int chunk = profile_chunk(262144), room = engine_room(ctx, rows, 262144);
            const size_t base = (size_t) chunk * kHeads * 262144;
            set_at(ctx, rows);
            std::printf("  ctx %lld, %d tok:", (long long) ctx, rows);
            for (size_t fit : {base, base * 3 / 2, base * 2}) {
                const int reps = g_reps;
                g_reps = std::max(10, (int) (g_reps * 8192 / ctx));
                const double f16 = time_us([&] { call(0, ctx, rows, chunk, fit, at, room, q, o_new); });
                const double q8 = time_us([&] { call(1, ctx, rows, chunk, fit, at, room, q, o_new); });
                g_reps = reps;
                std::printf(" | %3.0f MiB %-11s fp16 %8.1f us, q8_0 %8.1f us", fit * 4 / 1048576.0,
                            plan_text(attend_plan(rows, chunk, fit, room)).c_str(), f16, q8);
            }
            std::printf("\n");
        }

    // a graph recorded once with the new choice and replayed at three positions of its room, as the engine does,
    // against the same launches made directly (with the position on the card, and with it given on the host)
    std::printf("graphs: recorded with the new choice, replayed at three positions\n");
    struct GCase {
        int64_t profile, ctx;
        int rows;
    } gcases[] = {{49152, 45355, 8}, {131072, 131000, 8}, {262144, 45355, 4}, {262144, 45355, 8}, {262144, 261000, 4},
                  {262144, 261000, 8}};
    for (const GCase& gc : gcases) {
        const int chunk = profile_chunk(gc.profile);
        const size_t fit = (size_t) chunk * kHeads * gc.profile;
        const int room = engine_room(gc.ctx, gc.rows, gc.profile);
        for (int fmt = 0; fmt < 2; ++fmt) {
            set_at(gc.ctx, gc.rows);
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t exec = nullptr;
            cudaStreamBeginCapture(g_st, cudaStreamCaptureModeThreadLocal);
            call(fmt, gc.ctx, gc.rows, chunk, fit, at, room, q, o_new);
            if (cudaStreamEndCapture(g_st, &graph) != cudaSuccess ||
                cudaGraphInstantiateWithFlags(&exec, graph, 0) != cudaSuccess) {
                std::printf("  could not record the graph\n");
                return 1;
            }
            cudaGraphDestroy(graph);
            std::printf("  profile %lld, %d tok, room %d, %s, launches %s:", (long long) gc.profile, gc.rows, room,
                        fmt ? "q8_0" : "fp16", plan_text(attend_plan(gc.rows, chunk, fit, room)).c_str());
            for (int64_t p : {gc.ctx, gc.ctx - 777, gc.ctx - 3001}) {
                set_at(p, gc.rows);
                cudaGraphLaunch(exec, g_st);
                const std::vector<float> a = get(o_new, gc.rows);
                call(fmt, p, gc.rows, chunk, fit, at, room, q, o_old);
                const std::vector<float> b = get(o_old, gc.rows);
                call(fmt, p, gc.rows, chunk, fit, nullptr, 0, q, o_old);
                const std::vector<float> c = get(o_old, gc.rows);
                const bool same = std::memcmp(a.data(), b.data(), a.size() * 4) == 0 &&
                                  std::memcmp(a.data(), c.data(), a.size() * 4) == 0;
                bad += !same;
                std::printf(" pos %lld %s;", (long long) p, same ? "same bits" : "DIFFER");
            }
            std::printf("\n");
            const double replay = time_us([&] { cudaGraphLaunch(exec, g_st); });
            const double direct = time_us([&] { call(fmt, gc.ctx - 3001, gc.rows, chunk, fit, at, room, q, o_old); });
            std::printf("    replay %.1f us, direct %.1f us\n", replay, direct);
            cudaGraphExecDestroy(exec);
        }
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::printf("launch failed: %s\n", cudaGetErrorString(e));
        bad++;
    }
    std::printf("attn-fit %s\n", bad ? "FAILED" : "passed");
    return bad ? 1 : 0;
}

}  // namespace

// poly_kernel_bench gdn [reps]: a prompt batch's delta-net state update (gdn_step) at Qwen3.6's shapes, 1,024 tokens,
// with each kernel gdn_state_way() offers: the time, and the output and the state after it against way 0's, bit for
// bit, from the same state and inputs.
int gdn_bench() {
    const int dim = 128, heads = 32, kh = 16, row_n = (2 * kh + heads) * dim, inner = heads * dim;
    const size_t s_n = (size_t) heads * dim * dim;
    int bad = 0;
    for (int rows : {32, 100, 1024}) {
        float* qkv0 = (float*) dev_random((size_t) rows * row_n * 4, true);
        float* ab0 = (float*) dev_random((size_t) rows * heads * 2 * 4, true);
        float* dt = (float*) dev_random(heads * 4, true);
        float* a = (float*) dev_random(heads * 4, true);
        float* z = (float*) dev_random((size_t) rows * inner * 4, true);
        float* on = (float*) dev_random(dim * 4, true);
        float* s0 = (float*) dev_random(s_n * 4, true);
        float* qkv = (float*) dev_random((size_t) rows * row_n * 4, true);
        float* ab = (float*) dev_random((size_t) rows * heads * 2 * 4, true);
        float* s = (float*) dev_random(s_n * 4, true);
        float* out = (float*) dev_random((size_t) rows * inner * 4, true);
        std::vector<uint32_t> ref_out((size_t) rows * inner), ref_s(s_n), o2(ref_out.size()), s2(s_n);
        // the inputs as a model's are: q and k normed by gdn_step itself, the decay below 1
        const auto fresh = [&] {
            cudaMemcpyAsync(qkv, qkv0, (size_t) rows * row_n * 4, cudaMemcpyDeviceToDevice, g_st);
            cudaMemcpyAsync(ab, ab0, (size_t) rows * heads * 2 * 4, cudaMemcpyDeviceToDevice, g_st);
            cudaMemcpyAsync(s, s0, s_n * 4, cudaMemcpyDeviceToDevice, g_st);
        };
        const auto step = [&] {
            poly::gdn_step(qkv, ab, ab + (size_t) rows * heads, dt, a, s, z, on, out, dim, heads, kh, rows, 1e-6f, g_st);
        };
        for (int way : {0, 1, 2, 4}) {
            poly::gdn_state_way(way);
            fresh();
            step();
            cudaStreamSynchronize(g_st);
            cudaMemcpy(way ? o2.data() : ref_out.data(), out, o2.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(way ? s2.data() : ref_s.data(), s, s_n * 4, cudaMemcpyDeviceToHost);
            const bool same = way == 0 || (std::memcmp(o2.data(), ref_out.data(), o2.size() * 4) == 0 &&
                                           std::memcmp(s2.data(), ref_s.data(), s_n * 4) == 0);
            bad += !same;
            // the time of the four launches, from the same inputs every time
            const double us = time_us([&] {
                fresh();
                step();
            });
            const double copy = time_us(fresh);
            std::printf("  gdn_step %4d tokens, state kernel %d: %9.1f us (copies of the inputs %7.1f us)%s\n", rows,
                        way, us - copy, copy, way == 0 ? "" : same ? "  same bits as kernel 0" : "  NOT THE SAME BITS");
        }
    }
    poly::gdn_state_way(-1);
    return bad ? 1 : 0;
}

int main(int argc, char** argv) {
    cudaStreamCreateWithFlags(&g_st, cudaStreamNonBlocking);
    if (argc > 1 && std::strcmp(argv[1], "gdn") == 0) {
        g_reps = argc > 2 ? std::max(10, std::atoi(argv[2])) : 50;
        g_warm = 3;
        return gdn_bench();
    }
    if (argc > 1 && std::strcmp(argv[1], "cols") == 0) return cols_check();
    if (argc > 1 && std::strcmp(argv[1], "attn-decode") == 0) {
        g_reps = argc > 2 ? std::max(10, std::atoi(argv[2])) : 200;
        g_warm = 3;
        return attn_decode_bench();
    }
    if (argc > 1 && std::strcmp(argv[1], "q8-quant") == 0) {
        g_reps = 50;
        g_warm = 3;
        return q8_quant_check();
    }
    if (argc > 1 && std::strcmp(argv[1], "attn-q8") == 0) {
        g_reps = argc > 2 ? std::max(10, std::atoi(argv[2])) : 200;
        g_warm = 3;
        return attn_q8_bench();
    }
    if (argc > 2 && std::strcmp(argv[1], "attn-q8-real") == 0) return attn_q8_real(argv[2]);
    if (argc > 1 && std::strcmp(argv[1], "attn-kv-mix") == 0) {
        g_reps = argc > 2 ? std::max(10, std::atoi(argv[2])) : 100;
        g_warm = 3;
        return attn_kv_mix_bench();
    }
    if (argc > 1 && std::strcmp(argv[1], "attn-fit") == 0) {
        g_reps = argc > 2 ? std::max(10, std::atoi(argv[2])) : 200;
        g_warm = 3;
        return attn_fit_bench();
    }
    if (argc > 1 && std::strcmp(argv[1], "attn-dq8") == 0) {
        g_reps = 100;
        g_warm = 3;
        return attn_dq8_bench(argc > 2 ? std::atoll(argv[2]) : 100000);
    }
    if (argc > 1 && std::strcmp(argv[1], "attn-flash") == 0) {
        g_reps = argc > 2 ? std::max(3, std::atoi(argv[2])) : 10;
        g_warm = 1;
        return attn_flash_bench();
    }
    if (argc > 1 && std::strcmp(argv[1], "attn") == 0) {
        g_reps = argc > 2 ? std::max(10, std::atoi(argv[2])) : 200;
        g_warm = 2;
        return attn_bench();
    }
    if (argc > 1) g_reps = std::max(10, std::atoi(argv[1]));
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
    {   // several matrices a launch give each the bits of a launch of its own, for every format, the short rows of an
        // expert's down projection (n_in 256 and 512) and the long ones of its gate and up (2048), with the slots
        // given by value and read from device memory (the routing on the card; a slot of no matrix gives zeros), on
        // random bytes: the results are compared, not judged
        // where each block keeps its fp16 scales (d, and dmin where there is one): random bytes there would be
        // infinities and NaNs, whose bits two correct sums need not share
        const struct {
            int type, block_elems, block_bytes, d_at, dmin_at;
            const char* name;
        } fmts[] = {{2, 32, 18, 0, -1, "Q4_0"},     {6, 32, 22, 0, -1, "Q5_0"},     {8, 32, 34, 0, -1, "Q8_0"},
                    {20, 32, 18, 0, -1, "IQ4_NL"},  {10, 256, 84, 80, 82, "Q2_K"},  {11, 256, 110, 108, -1, "Q3_K"},
                    {12, 256, 144, 0, 2, "Q4_K"},   {13, 256, 176, 0, 2, "Q5_K"},   {14, 256, 210, 208, -1, "Q6_K"},
                    {23, 256, 136, 0, -1, "IQ4_XS"}};
        int bad = 0, checked = 0, dev_bad = 0, dev_checked = 0;
        void* dslots = nullptr;
        cudaMalloc(&dslots, sizeof(kernels::MmvqSlots));
        for (const auto& f : fmts) {
            for (int n_in : {256, 512, 2048}) {
                const int n_out = n_in == 2048 ? 1024 : 2048;
                const size_t wb = (size_t) n_out * (n_in / f.block_elems) * f.block_bytes;
                void* w = dev_random(2 * wb, false);
                tame_scales(w, 2 * wb, f.block_bytes, f.d_at);
                if (f.dmin_at >= 0) tame_scales(w, 2 * wb, f.block_bytes, f.dmin_at);
                void* q = dev_random(kernels::native_q8_1_bytes(n_in) * 2, false);
                kernels::native_quantize_q8_1(x, q, n_in, 1, st);
                kernels::native_quantize_q8_1(x + 4096, (uint8_t*) q + kernels::native_q8_1_bytes(n_in), n_in, 1, st);
                // [0, 2): slots by value, [2, 4): one launch each, [4, 7): slots in device memory, the last of no matrix
                float* o = (float*) dev_random((size_t) 7 * n_out * 4, true);
                kernels::MmvqSlots m{};
                for (int j = 0; j < 2; ++j) {
                    m.w[j] = (const uint8_t*) w + j * wb;
                    m.x[j] = (const uint8_t*) q + j * kernels::native_q8_1_bytes(n_in);
                    m.y[j] = o + j * n_out;
                    kernels::native_mmvq(f.type, m.w[j], m.x[j], o + (2 + j) * n_out, n_in, n_out, 1, st);
                }
                if (!kernels::native_mmvq_slots(f.type, m, 2, n_in, n_out, st)) continue;
                kernels::MmvqSlots md{};
                for (int j = 0; j < 3; ++j) {
                    md.w[j] = j < 2 ? m.w[j] : nullptr;
                    md.x[j] = m.x[j < 2 ? j : 0];
                    md.y[j] = o + (4 + j) * n_out;
                }
                cudaMemcpyAsync(dslots, &md, sizeof md, cudaMemcpyHostToDevice, g_st);
                const bool dev = kernels::native_mmvq_slots_dev(f.type, (const kernels::MmvqSlots*) dslots, 3, n_in,
                                                                n_out, st);
                std::vector<uint32_t> h((size_t) 7 * n_out);
                cudaStreamSynchronize(g_st);   // the stream does not wait for the default one, nor it for the stream
                cudaMemcpy(h.data(), o, h.size() * 4, cudaMemcpyDeviceToHost);
                ++checked;
                if (std::memcmp(h.data(), h.data() + 2 * n_out, (size_t) 2 * n_out * 4) != 0) {
                    ++bad;
                    std::printf("  slots and one launch each DIFFER: %s, n_in %d\n", f.name, n_in);
                }
                if (dev) {
                    ++dev_checked;
                    const bool zeros = std::all_of(h.begin() + 6 * n_out, h.end(), [](uint32_t b) { return b == 0; });
                    if (std::memcmp(h.data() + 4 * n_out, h.data() + 2 * n_out, (size_t) 2 * n_out * 4) != 0 || !zeros) {
                        ++dev_bad;
                        std::printf("  slots in device memory DIFFER: %s, n_in %d%s\n", f.name, n_in,
                                    zeros ? "" : " (the slot of no matrix is not zeros)");
                    }
                } else {
                    std::printf("  slots in device memory: no kernel for %s, n_in %d\n", f.name, n_in);
                }
                cudaFree(w);
                cudaFree(q);
                cudaFree(o);
            }
        }
        cudaFree(dslots);
        std::printf("  slots against one launch each, bit for bit: %d of %d shapes the same\n", checked - bad, checked);
        std::printf("  slots in device memory against one launch each, bit for bit: %d of %d shapes the same\n",
                    dev_checked - dev_bad, dev_checked);
        if (bad || dev_bad) return 1;
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
