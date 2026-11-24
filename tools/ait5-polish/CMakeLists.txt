enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

set(TARGET bench-ait5)

add_executable(${TARGET}
    bench_ait5.cpp
    bwref_flush.cu
)

target_link_libraries(${TARGET} PRIVATE ggml ggml-cuda CUDA::cudart Threads::Threads)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/ggml/src
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
set_target_properties(${TARGET} PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)
