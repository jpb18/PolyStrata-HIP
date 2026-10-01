# Strata on Intel GPUs - the SYCL build of the engine (Phase 4 of the Intel port, docs/INTEL.md).
#
# This is a separate project on purpose. The top-level build describes the CUDA engine; every source it lists is
# taken from ../src unless this directory carries a migrated copy: `src/X.cu` -> `sycl/src/X.dp.cpp`,
# `src/X.cpp` -> `sycl/src/X.cpp` (only host files that called the CUDA runtime were migrated). Headers resolve
# the same way through the include order. Upstream files are never edited, so the port is the content of this
# directory plus the tool that produced it (../../sycl/migrate.sh).
#
#   cmake -S sycl -B build-sycl -G Ninja -DCMAKE_CXX_COMPILER=icpx && cmake --build build-sycl
#
# Device code is JIT-compiled from SPIR-V unless STRATA_SYCL_AOT names a device (bmg-g31 for the Arc Pro B70).
cmake_minimum_required(VERSION 3.24)
project(strata_sycl VERSION 0.1.24 LANGUAGES C CXX)
set(STRATA_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
add_compile_definitions(STRATA_VERSION="${PROJECT_VERSION}-sycl")

set(STRATA_SYCL_AOT "" CACHE STRING "AOT device for the GPU code, e.g. bmg-g31 (empty: SPIR-V, JIT at run time)")
set(_sycl_flags -fsycl)
if(STRATA_SYCL_AOT)
  list(APPEND _sycl_flags -fsycl-targets=spir64_gen)
  # the divide/sqrt rounding option (below) must reach ocloc here: an AOT build has no spir64 backend to take it
  set(_sycl_link -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen
      "-device ${STRATA_SYCL_AOT} -options -cl-fp32-correctly-rounded-divide-sqrt")
else()
  set(_sycl_link -fsycl)
endif()
# The Intel GPU's fp32 divide is not correctly rounded by default (OpenCL allows 2.5 ulp). Strata's quantizers
# are byte-exact against ggml through `amax / 127`, so the device compiler is told to round divides and sqrt
# properly; measured: quantize_act_parity goes from 303k mismatches to byte-exact. (Same as
# SYCL_PROGRAM_COMPILE_OPTIONS=-cl-fp32-correctly-rounded-divide-sqrt at run time.) The AOT build passes it to
# ocloc above; without that the 303k mismatches came back on 2026-09-30, unnoticed since the AOT switch.
if(NOT STRATA_SYCL_AOT)
  list(APPEND _sycl_link "-Xsycl-target-backend=spir64" "-cl-fp32-correctly-rounded-divide-sqrt")
endif()
# Every translation unit gets -fsycl: the migrated host files construct queues and USM, not only the kernels.
# icpx defaults to a fast floating-point model (reciprocal approximations, reassociation); the kernels are
# compared byte-for-byte against ggml's, which nvcc built value-safe, so ask for the same here.
# Every kernel was written for 32-lane warps (lane = tid & 31, warp-synchronous reductions). dpct pins the
# sub-group size only where it saw a shuffle; the rest would get Xe2's default of 16 and race. Force 32 everywhere.
list(APPEND _sycl_flags -fsycl-default-sub-group-size=32)
# One device module per kernel: the first launch of a kernel then loads only its own module. With the default
# split the first launch in the prompt path (the embedding gather) paid 245 ms for the image it shared (2026-09-30).
list(APPEND _sycl_link -fsycl-device-code-split=per_kernel)
list(APPEND _sycl_flags -fsycl-device-code-split=per_kernel)
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${_sycl_flags}>" -fp-model=precise -Wno-unused-parameter
                    -Wno-unused-variable -Wno-deprecated-declarations)
add_link_options(${_sycl_link})

# --- source resolution: the migrated copy when there is one, the original otherwise -------------------------------
function(strata_resolve out)
  set(r "")
  foreach(f IN LISTS ARGN)
    if(f MATCHES "\\.cu$")
      string(REGEX REPLACE "\\.cu$" ".dp.cpp" m "${f}")
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${m})
        list(APPEND r ${CMAKE_CURRENT_SOURCE_DIR}/${m})
      else()
        message(FATAL_ERROR "no migrated copy of ${f} (expected ${m})")
      endif()
    elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${f})
      list(APPEND r ${CMAKE_CURRENT_SOURCE_DIR}/${f})
    elseif(EXISTS ${STRATA_ROOT}/${f})
      list(APPEND r ${STRATA_ROOT}/${f})
    else()
      message(FATAL_ERROR "no such source: ${f}")
    endif()
  endforeach()
  set(${out} ${r} PARENT_SCOPE)
endfunction()

add_library(strata_includes INTERFACE)
target_include_directories(strata_includes INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include      # migrated headers + dpct/ helpers, first
  ${STRATA_ROOT}/include)
target_compile_options(strata_includes INTERFACE -Wall -Wextra)

# --- CPU-side pieces, identical to the top-level build ------------------------------------------------------------
strata_resolve(_spec src/spec/suffix_drafter.cpp src/spec/controller.cpp src/spec/draft_policy.cpp)
add_library(strata_spec STATIC ${_spec})
target_link_libraries(strata_spec PUBLIC strata_includes)

strata_resolve(_cpu src/kernels/cpu/expert.cpp src/kernels/cpu/pool.cpp src/kernels/cpu/expert_layout.cpp
                    src/kernels/cpu/q2_avx2.cpp)
add_library(strata_kernels_cpu STATIC ${_cpu})
set_source_files_properties(${STRATA_ROOT}/src/kernels/cpu/expert.cpp PROPERTIES COMPILE_OPTIONS
  "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
set_source_files_properties(${STRATA_ROOT}/src/kernels/cpu/q2_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
find_package(Threads REQUIRED)
target_link_libraries(strata_kernels_cpu PUBLIC strata_includes Threads::Threads)

option(STRATA_NATIVE_EXPERTS "Native (IQ) expert support through ggml-cpu" ON)
set(STRATA_GGML_DIR "" CACHE PATH "llama.cpp checkout to take ggml from (empty: fetch the pinned commit)")
if(STRATA_NATIVE_EXPERTS)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(GGML_STATIC ON CACHE BOOL "" FORCE)
  set(GGML_CUDA OFF CACHE BOOL "" FORCE)
  set(GGML_SYCL OFF CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
  set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GGML_NATIVE ON CACHE BOOL "" FORCE)
  if(STRATA_GGML_DIR)
    add_subdirectory(${STRATA_GGML_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
  else()
    include(FetchContent)
    FetchContent_Declare(strata_llamacpp
      GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
      GIT_TAG 3cf03257f219afbe7334045ff7c6a06ac68c627d
      GIT_SHALLOW FALSE)
    FetchContent_GetProperties(strata_llamacpp)
    if(NOT strata_llamacpp_POPULATED)
      FetchContent_Populate(strata_llamacpp)
    endif()
    add_subdirectory(${strata_llamacpp_SOURCE_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
  endif()
  strata_resolve(_cpu_native src/kernels/cpu/native_expert.cpp src/kernels/cpu/iq_avx512.cpp src/kernels/cpu/iq_avx2.cpp)
  target_sources(strata_kernels_cpu PRIVATE ${_cpu_native})
  target_include_directories(strata_kernels_cpu PRIVATE ${STRATA_ROOT}/third_party/ggml)
  set_source_files_properties(${STRATA_ROOT}/src/kernels/cpu/iq_avx512.cpp PROPERTIES COMPILE_OPTIONS
    "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
  set_source_files_properties(${STRATA_ROOT}/src/kernels/cpu/iq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
  target_link_libraries(strata_kernels_cpu PUBLIC ggml-cpu ggml-base)
  target_compile_definitions(strata_kernels_cpu PUBLIC STRATA_NATIVE_EXPERTS=1)
endif()

# --- the GPU engine ------------------------------------------------------------------------------------------------
strata_resolve(_core src/core/device.cu src/core/pinned.cu src/platform/memory.cpp src/core/graph.cpp
                     src/core/weights.cpp src/core/layout.cpp)
add_library(strata_core STATIC ${_core})
target_link_libraries(strata_core PUBLIC strata_includes)

set(_kernel_cu dequant_s2 s2_gemv s_gemv s2_gemv_quads s2_gemv_fast router_top10 quantize_act rope sampler s2_gemv_q8
    s2_expert_grouped shared_expert gr gdn qsa elementwise bf16_gemv native_mmvq native_bf16 native_gr_norm native_moe
    native_gdn native_gr_postops native_router native_gdn_preprocess native_qsa native_rope native_ple_postops
    native_flash_attn native_qsa_indexer native_qsa_score ple kv_q8 kv_stream kv_q4 fused_gr cvec qsa_decode_attn
    qsa_prompt_attn fused_gdn dequant_bf16 qsa_select verify_kernels iq_kernels)
set(_kernel_files "")
foreach(k IN LISTS _kernel_cu)
  list(APPEND _kernel_files src/kernels/cuda/${k}.cu)
endforeach()
strata_resolve(_kernels ${_kernel_files} src/kernels/ngram.cpp src/ngram/ple_reader.cpp src/platform/direct_file.cpp)
list(APPEND _kernels ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/qsa_prompt_attn_xmx.dp.cpp)   # port-only: the XMX prompt attention
add_library(strata_kernels STATIC ${_kernels})
target_include_directories(strata_kernels PRIVATE ${STRATA_ROOT}/third_party/ggml)
target_link_libraries(strata_kernels PUBLIC strata_core)

strata_resolve(_engine src/core/layer.cpp src/core/session.cpp src/core/expert_source.cpp src/core/remote_experts.cpp
                       src/core/expert_cache.cpp src/core/native_head.cpp src/core/native_dense.cpp src/core/verify.cpp
                       src/core/mtp.cpp)
add_library(strata_engine STATIC ${_engine} ${CMAKE_CURRENT_SOURCE_DIR}/src/core/gguf_expert_source.cpp)
target_link_libraries(strata_engine PUBLIC strata_core strata_kernels strata_kernels_cpu)

# cuBLAS -> oneMKL through dpct::blas (needs the MKL SYCL BLAS domain)
strata_resolve(_prefill src/prefill/gemm.cu src/prefill/kernels.cu src/prefill/prefill.cpp)
add_library(strata_prefill STATIC ${_prefill})
target_compile_options(strata_prefill PUBLIC -qmkl=sequential)
target_link_options(strata_prefill PUBLIC -qmkl=sequential)
target_link_libraries(strata_prefill PUBLIC strata_engine)

strata_resolve(_main src/program/generate.cpp)
add_executable(strata ${_main})
target_link_libraries(strata PRIVATE strata_engine strata_prefill strata_spec)

# --- the kernel parity self-tests (each runs on the card and compares against a CPU reference) ---------------------
option(STRATA_SYCL_PARITY "Build the kernel parity tests" ON)
if(STRATA_SYCL_PARITY)
  enable_testing()
  foreach(p IN ITEMS dequant_s2 s2_gemv iq shared_expert gr gdn s2_gemv_q8 sampler rope quantize_act router_top10
                     s_gemv elementwise bf16_gemv s_gemv_q8k qsa ple kv_q8 kv_stream kv_q4 cvec)
    strata_resolve(_src src/kernels/${p}_parity.cpp)
    add_executable(${p}_parity ${_src})
    target_include_directories(${p}_parity PRIVATE ${STRATA_ROOT}/src/kernels)   # the oracle .inc fixtures
    target_link_libraries(${p}_parity PRIVATE strata_kernels)
    add_test(NAME ${p}_parity COMMAND ${p}_parity --selftest)
  endforeach()
  # the prompt-attention parity takes a context length, not --selftest, and needs the tensor-core kernel (not
  # ported yet: it reports "refused the pools" here). Built, not registered.
  strata_resolve(_src src/kernels/qsa_prompt_attn_parity.cpp)
  add_executable(qsa_prompt_attn_parity ${_src})
  target_link_libraries(qsa_prompt_attn_parity PRIVATE strata_kernels)
  # the hybrid K8V4 KV test (upstream 0.1.27) passes its appends, gathers and attention here and fails only its
  # last step, the same tensor-core prompt kernel. Built, not registered, until that kernel is ported.
  strata_resolve(_src src/kernels/kv_hybrid_parity.cpp)
  add_executable(kv_hybrid_parity ${_src})
  target_include_directories(kv_hybrid_parity PRIVATE ${STRATA_ROOT}/src/kernels)
  target_link_libraries(kv_hybrid_parity PRIVATE strata_kernels)
  add_executable(mmvq_bench ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/mmvq_bench.cpp)   # dense mmvq timing
  target_link_libraries(mmvq_bench PRIVATE strata_kernels)
  add_executable(mmvq_sg_bench ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/mmvq_sg_bench.cpp)   # SIMD16 vs 32 decode mmvq
  target_link_libraries(mmvq_sg_bench PRIVATE strata_kernels)
  add_executable(xmx_gemm_bench ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/xmx_gemm_bench.cpp)   # XMX quantized GEMM vs dequant+oneMKL
  target_link_libraries(xmx_gemm_bench PRIVATE strata_kernels strata_prefill)   # -qmkl for the oneMKL reference
  # one native expert three ways on real GGUF rows (ggml reference, CPU path, GPU native_expert_grouped):
  #   native_expert_parity <shard1.gguf> [layer ...]
  if(STRATA_NATIVE_EXPERTS)
    add_executable(native_expert_parity ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/native_expert_parity.cpp)
    target_include_directories(native_expert_parity PRIVATE ${STRATA_ROOT}/third_party/ggml)
    target_link_libraries(native_expert_parity PRIVATE strata_kernels strata_kernels_cpu)
  endif()
endif()
