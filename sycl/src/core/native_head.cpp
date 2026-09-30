#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/core/native_head.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <climits>
#include <cstdio>
#include <cstring>
#include <exception>

namespace strata::core {

NativeHead::~NativeHead() {
    if (scratch_) sycl::free(scratch_, dpct::get_in_order_queue());
    if (weights_) sycl::free(weights_, dpct::get_in_order_queue());
}

bool NativeHead::load(const std::string &path, int64_t n_in, int64_t n_out,
                      std::string &err) try {
    if (loaded()) { err = "native head is already loaded"; return false; }
    if (n_in <= 0 || n_out <= 0 || n_in > INT_MAX || n_out > INT_MAX || n_in % 256) {
        err = "native head requires positive int32 dimensions and whole 256-value rows";
        return false;
    }
    try {
        strata::GgufFile gguf(path);
        err = strata::check_architecture(gguf);
        if (!err.empty()) return false;
        const strata::TensorInfo* tensor = nullptr;
        for (const auto& candidate : gguf.tensors()) {
            if (candidate.name != "output.weight") continue;
            if (tensor) { err = "native head: duplicate output.weight"; return false; }
            tensor = &candidate;
        }
        if (!tensor || !strata::kernels::native_mmvq_supported((int) tensor->type) || tensor->shape.size() != 2 ||
            tensor->shape[0] != (uint64_t) n_in || tensor->shape[1] != (uint64_t) n_out) {
            err = "native head: expected a natively supported output.weight with the canonical head dimensions";
            return false;
        }
        const uint64_t bytes = strata::kernels::native_mmvq_weight_bytes((int) tensor->type, (int) n_in, (int) n_out);
        const uint64_t payload = gguf.file_size() - gguf.data_start();
        if (tensor->offset > payload || bytes > payload - tensor->offset) {
            err = "native head: truncated output.weight payload";
            return false;
        }
        void* weights = nullptr;
        void* scratch = nullptr;
        dpct::err0 status =
            DPCT_CHECK_ERROR(weights = (void *)sycl::malloc_device(
                                 bytes, dpct::get_in_order_queue()));
        if (status == 0)
            status = DPCT_CHECK_ERROR(
                scratch = (void *)sycl::malloc_device(
                    strata::kernels::native_q8_1_bytes((int)n_in, 1),
                    dpct::get_in_order_queue()));
        if (status == 0)
            status = DPCT_CHECK_ERROR(
                dpct::get_in_order_queue()
                    .memcpy(weights, gguf.tensor_data(*tensor), bytes)
                    .wait());
        /*
        DPCT1000: Error handling if-stmt was detected but could not be
        rewritten.
        */
        if (status != 0) {
            if (scratch) sycl::free(scratch, dpct::get_in_order_queue());
            if (weights) sycl::free(weights, dpct::get_in_order_queue());
            /*
            DPCT1009: SYCL reports errors using exceptions and does not use
            error codes. Please replace the "get_error_string_dummy(...)" with a
            real error-handling function.
            */
            /*
            DPCT1001: The statement could not be removed.
            */
            err = std::string("native head upload: ") +
                  dpct::get_error_string_dummy(status);
            return false;
        }
        weights_ = weights;
        scratch_ = scratch;
        bytes_ = bytes;
        n_in_ = (int) n_in;
        n_out_ = (int) n_out;
        type_ = (int) tensor->type;
        return true;
    } catch (const std::exception& error) {
        err = std::string("native head: ") + error.what();
        return false;
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool NativeHead::run(const float *mixed, float *logits, void *stream,
                     std::string &err) const try {
    if (!loaded() || !mixed || !logits || !stream) {
        err = "native head requires loaded weights, device buffers and an explicit stream";
        return false;
    }
    try {
        if (type_ == 13) {
            strata::kernels::native_q5_k_f32(weights_, mixed, scratch_, logits, n_in_, n_out_, 1, stream);
        } else {
            strata::kernels::native_quantize_q8_1(mixed, scratch_, n_in_, 1, stream);
            strata::kernels::native_mmvq(type_, weights_, scratch_, logits, n_in_, n_out_, 1, stream);
        }
    } catch (const std::exception& error) {
        err = std::string("native head launch: ") + error.what();
        return false;
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaPeekAtLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 status = 0;
    /*
    DPCT1000: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (status != 0) {
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1001: The statement could not be removed.
        */
        err = std::string("native head launch: ") +
              dpct::get_error_string_dummy(status);
        return false;
    }
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// ================================ plan v0.3 P6: THE NATIVE EMBEDDING ================================

namespace {
const NativeEmbed* g_embed = nullptr;
}
void set_native_embed(const NativeEmbed* e) { g_embed = e; }
const NativeEmbed* native_embed() { return g_embed; }

NativeEmbed::~NativeEmbed() {
    if (host_) sycl::free(host_, dpct::get_in_order_queue());
    else if (dev_)
        sycl::free((void *) dev_, dpct::get_in_order_queue()); // the VRAM fallback below
}

bool NativeEmbed::load(const std::string &path, int64_t n_embd, int64_t n_vocab,
                       std::string &err) try {
    try {
        strata::GgufFile gguf(path);
        const strata::TensorInfo* t = nullptr;
        for (const auto& c : gguf.tensors())
            if (c.name == "token_embd.weight") t = &c;
        if (!t || t->shape.size() != 2 || t->shape[0] != (uint64_t) n_embd || t->shape[1] != (uint64_t) n_vocab ||
            !strata::kernels::iq_supported((int) t->type) || n_embd % 256) {
            err = "native embedding: token_embd.weight is absent, of another shape, or of a type without a GPU "
                  "dequantizer";
            return false;
        }
        row_ = strata::kernels::iq_row_bytes((int) t->type, n_embd);
        bytes_ = (uint64_t) row_ * (uint64_t) n_vocab;
        /*
        DPCT1048: The original value cudaHostAllocMapped is not meaningful in
        the migrated code and was removed or replaced with 0. You may need to
        check the migrated code.
        */
        /*
        DPCT1048: The original value cudaHostAllocPortable is not meaningful
        in the migrated code and was removed or replaced with 0. You may need to
        check the migrated code.
        */
        if (DPCT_CHECK_ERROR(host_ = (void *)sycl::malloc_host(
                                 bytes_, dpct::get_in_order_queue())) != 0) {
            // Under WSL2 the driver's pinned/mapped host budget (~1 GiB) can be spent by the GPU contexts
            // themselves (three cards). The table is only gathered from, so keep it in the current device's VRAM
            // instead: it costs its size there and reads faster than over PCIe.
            /*
            DPCT1026: The call to cudaGetLastError was removed because this
            functionality is redundant in SYCL.
            */
            host_ = nullptr;
            void* d = nullptr;
            if (DPCT_CHECK_ERROR(d = (void *)sycl::malloc_device(
                                     bytes_, dpct::get_in_order_queue())) !=
                    0 ||
                /*
                DPCT1114: cudaMemcpy is migrated to asynchronization
                memcpy, assuming in the original code the source host memory is
                pageable memory. If the memory is not pageable, call wait() on
                event return by memcpy API to ensure synchronization behavior.
                */
                DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(
                    d, gguf.tensor_data(*t), bytes_).wait()) != 0) {
                if (d) sycl::free(d, dpct::get_in_order_queue());
                /*
                DPCT1026: The call to cudaGetLastError was removed because
                this functionality is redundant in SYCL.
                */
                err = "native embedding: cannot pin " +
                      std::to_string(bytes_ >> 20) +
                      " MiB, nor place it in VRAM";
                return false;
            }
            std::fprintf(stderr, "strata: native embedding: cannot pin %llu MiB, kept in VRAM instead\n",
                         (unsigned long long) (bytes_ >> 20));
            dev_ = d;
        } else {
            std::memcpy(host_, gguf.tensor_data(*t), bytes_);
            void* d = nullptr;
            if (DPCT_CHECK_ERROR(d = (void *)host_) != 0) {
                err = "native embedding: no device alias for the mapped table";
                return false;
            }
            dev_ = d;
        }
        type_ = (int) t->type;
        n_embd_ = n_embd;
        n_vocab_ = n_vocab;
        return true;
    } catch (const std::exception& e) {
        err = std::string("native embedding: ") + e.what();
        return false;
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void NativeEmbed::gather_dev(const int32_t* tokens, int64_t n_tok, float* out, void* stream) const {
    strata::kernels::iq_embed_rows(type_, dev_, row_, tokens, n_tok, n_embd_, out, stream);
}

void NativeEmbed::gather_one(int64_t token, float* out, void* stream) const {
    strata::kernels::iq_dequant_f32(type_, (const uint8_t*) dev_ + (size_t) token * row_, n_embd_, out, stream);
}

}  // namespace strata::core
