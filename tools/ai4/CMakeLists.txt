# llama-ai4-quantize: AI4 (Ampere INT4, formerly D2) GPU encoder + GGUF writer. CUDA only.
set(TARGET llama-ai4-quantize)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# This target mixes C++ and CUDA sources, so CMake links it with the C++ compiler but also adds the CUDA implicit link
# directories. When nvcc's host compiler is an older gcc than CMAKE_CXX_COMPILER (here nvcc 12.4 -> gcc 13, c++ -> gcc 15),
# one of those is /usr/lib/gcc/<triple>/13, whose libstdc++.so then shadows the C++ compiler's own and the link fails on
# symbols the gcc-15-built libllama-common needs (__cxa_call_terminate@CXXABI_1.3.15). Drop the gcc-private directories:
# the C++ driver adds its own, and the CUDA runtime libraries live in the system library directories.
list(FILTER CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES EXCLUDE REGEX "/lib/gcc/")

add_executable(${TARGET} ai4-quantize.cpp ai4-encode.cu)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET} PRIVATE llama-common llama ggml CUDA::cudart CUDA::cublas CUDA::cusolver ${CMAKE_THREAD_LIBS_INIT})

# llama-ai4-hessian (AIQ, private measurement tool): full input Hessians via cuBLAS SSYRK in the eval callback
add_executable(llama-ai4-hessian ai4-hessian.cpp)
target_link_libraries(llama-ai4-hessian PRIVATE llama ggml CUDA::cudart CUDA::cublas ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(llama-ai4-hessian PRIVATE cxx_std_17)
target_compile_features(${TARGET} PRIVATE cxx_std_17 cuda_std_17)
set_target_properties(${TARGET} PROPERTIES CUDA_ARCHITECTURES "86")
target_compile_options(${TARGET} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v -lineinfo>)

if (LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

# llama-ai4-kern-bench: exactness check + microbench of the AI4 fast-path kernels (ggml-cuda/ai4-kern.cuh), no ggml.
add_executable(llama-ai4-kern-bench ai4-kern-bench.cu)
target_include_directories(llama-ai4-kern-bench PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src ${CMAKE_SOURCE_DIR}/ggml/include)
target_link_libraries(llama-ai4-kern-bench PRIVATE ggml CUDA::cudart)
target_compile_features(llama-ai4-kern-bench PRIVATE cxx_std_17 cuda_std_17)
set_target_properties(llama-ai4-kern-bench PROPERTIES CUDA_ARCHITECTURES "86")
target_compile_options(llama-ai4-kern-bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v -lineinfo>)
# llama-ai4-fullk-test (EXLGAP): synthetic GPU test of full-K LDLQ at rate 4 / 5 (ai4_cuda_encode_fullk_rate)
add_executable(llama-ai4-fullk-test test-aiq-fullk-rate.cpp ai4-encode.cu)
target_include_directories(llama-ai4-fullk-test PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src ${CMAKE_SOURCE_DIR}/ggml/include ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(llama-ai4-fullk-test PRIVATE ggml CUDA::cudart CUDA::cublas CUDA::cusolver ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(llama-ai4-fullk-test PRIVATE cxx_std_17 cuda_std_17)
set_target_properties(llama-ai4-fullk-test PROPERTIES CUDA_ARCHITECTURES "86")

# llama-ai4-errtab (AIQ3, private): per-tensor x per-tier error tables for aiq_alloc.py + CPU reference tier encoder. CPU only.
add_executable(llama-ai4-errtab ai4-errtab.cpp)
target_include_directories(llama-ai4-errtab PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src)
target_link_libraries(llama-ai4-errtab PRIVATE llama-common ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(llama-ai4-errtab PRIVATE cxx_std_17)

# llama-ai4-rate-test (AIBEST, private): synthetic GPU test of the AI+ rates k = 2, 3, 4, 5, 6 + the A/B control constants
add_executable(llama-ai4-rate-test test-aiq-rate.cpp ai4-encode.cu)
target_include_directories(llama-ai4-rate-test PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src ${CMAKE_SOURCE_DIR}/ggml/include ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(llama-ai4-rate-test PRIVATE ggml CUDA::cudart CUDA::cublas CUDA::cusolver ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(llama-ai4-rate-test PRIVATE cxx_std_17 cuda_std_17)
set_target_properties(llama-ai4-rate-test PROPERTIES CUDA_ARCHITECTURES "86")
