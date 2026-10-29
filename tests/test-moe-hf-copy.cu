#include <cuda_runtime.h>
#include "../ggml/src/ggml-cuda/moe-hf-copy.cuh"

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_OK(call) do { const cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); std::exit(1); } } while (0)

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) { return 77; }
    constexpr int slots = 32;
    unsigned cases = 0;
    for (size_t bytes : {size_t(16), size_t(240), size_t(256), size_t(272), size_t(98320), size_t(786432)}) {
        const size_t stride = (bytes + 255) / 256 * 256 + 256;
        const size_t size = slots * stride + 256;
        unsigned char * host;
        unsigned char * mapped;
        unsigned char * device;
        unsigned char * a;
        unsigned char * b;
        const void ** src;
        int * count;
        CUDA_OK(cudaHostAlloc(&host, size, cudaHostAllocMapped));
        CUDA_OK(cudaHostGetDevicePointer(&mapped, host, 0));
        CUDA_OK(cudaMalloc(&device, size));
        CUDA_OK(cudaMalloc(&a, size));
        CUDA_OK(cudaMalloc(&b, size));
        CUDA_OK(cudaMalloc(&src, slots * sizeof(void *)));
        CUDA_OK(cudaMalloc(&count, sizeof(int)));
        for (size_t i = 0; i < size; ++i) { host[i] = (unsigned char) ((i * 37 + i / 251) & 255); }
        CUDA_OK(cudaMemcpy(device, host, size, cudaMemcpyHostToDevice));
        // Exercise mapped host and device sources, reordered pointers, partial unrolls,
        // full/empty plans, and canaries in every stride gap and after the last slot.
        for (bool on_host : {false, true}) {
            std::vector<const void *> pointers(slots);
            for (int r = 0; r < slots; ++r) {
                pointers[r] = (on_host ? mapped : device) + ((r * 7) % slots) * stride;
            }
            CUDA_OK(cudaMemcpy(src, pointers.data(), slots * sizeof(void *), cudaMemcpyHostToDevice));
            for (int blocks : {1, 3, 24}) {
                for (int n : {0, 1, 2, 7, 32}) {
                    CUDA_OK(cudaMemcpy(count, &n, sizeof(n), cudaMemcpyHostToDevice));
                    CUDA_OK(cudaMemset(a, 0xa5, size));
                    CUDA_OK(cudaMemset(b, 0xa5, size));
                    moe_hf_copy_kernel<false><<<blocks, 256>>>(src, count, (char *) a, stride, bytes);
                    moe_hf_copy_kernel<true><<<blocks, 256>>>(src, count, (char *) b, stride, bytes);
                    CUDA_OK(cudaGetLastError());
                    std::vector<unsigned char> ref(size), got(size), expected(size, 0xa5);
                    CUDA_OK(cudaMemcpy(ref.data(), a, size, cudaMemcpyDeviceToHost));
                    CUDA_OK(cudaMemcpy(got.data(), b, size, cudaMemcpyDeviceToHost));
                    for (int r = 0; r < n; ++r) {
                        for (size_t i = 0; i < bytes; ++i) {
                            expected[r * stride + i] = host[((r * 7) % slots) * stride + i];
                        }
                    }
                    if (ref != expected || got != expected) {
                        fprintf(stderr, "FAIL bytes=%zu blocks=%d experts=%d host=%d\n", bytes, blocks, n, on_host);
                        return 1;
                    }
                    ++cases;
                }
            }
        }
        CUDA_OK(cudaFree(count)); CUDA_OK(cudaFree(src));
        CUDA_OK(cudaFree(a)); CUDA_OK(cudaFree(b)); CUDA_OK(cudaFree(device));
        CUDA_OK(cudaFreeHost(host));
    }
    printf("PASS %u copy cases: original/candidate/expected bytes and canaries match\n", cases);
    return 0;
}
