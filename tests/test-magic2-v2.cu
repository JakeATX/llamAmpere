#include "../ggml/src/ggml-cuda/magic2-v2.cuh"
#include "../ggml/src/ggml-aif25.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

#define CK(call) do { const cudaError_t e = (call); if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); std::exit(1); } } while (0)

static void require(bool value, const char * message) {
    if (!value) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}

// Bit-at-a-time reference is independent of the device window and byte packing.
template<int B, int V, uint32_t C>
static int reference(const uint8_t * tile, int index) {
    constexpr int NB = 128 * B / V;
    const int step = index / V;
    uint32_t state = 0;
    for (int j = 0; j < 16; ++j) {
        const int bit = (B * (step + 1) - 16 + j + NB) % NB;
        const int byte = (bit / 32) * 4 + 3 - (bit % 32) / 8;
        state = (state << 1) | ((tile[byte] >> (7 - bit % 8)) & 1);
    }
    uint32_t z = state * ((V == 2 && (index & 1)) ? uint32_t(uint64_t(C) * C) : C);
    int total = 0;
    for (int j = 0; j < 4; ++j) { total += z & 255; z >>= 8; }
    return ((total + 2) >> 2) - 128;
}

template<int B, int V, uint32_t C>
static __global__ void decode_check(const uint32_t * tile, int8_t * out, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    constexpr int TW = magic2::tile_words<B,V>;
    const int n = index % 128, group = index / 128;
    const uint32_t q = magic2::fragment<B,V,C>(tile + group * TW, n / 8, (n % 8) / 4);
    out[index] = static_cast<int8_t>(q >> (8 * (n % 4)));
}

template<int B, int V, uint32_t C, int XSH = 0>
static void test(std::mt19937 & rng, bool perf) {
    constexpr int UB = magic2::unit_bytes<B,V>, TB = 4 * magic2::tile_words<B,V>;
    if (!perf) {
        // Include tail-biting, every shift alignment, and many independent streams.
        std::vector<uint8_t> tiles(4096 * TB);
        for (uint8_t & v : tiles) v = rng();
        uint32_t * dt; int8_t * dq;
        CK(cudaMalloc(&dt, tiles.size())); CK(cudaMalloc(&dq, 4096 * 128));
        CK(cudaMemcpy(dt, tiles.data(), tiles.size(), cudaMemcpyHostToDevice));
        decode_check<B,V,C><<<4096,128>>>(dt, dq, 4096 * 128);
        CK(cudaGetLastError());
        std::vector<int8_t> q(4096 * 128);
        CK(cudaMemcpy(q.data(), dq, q.size(), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < q.size(); ++i) {
            require(q[i] == reference<B,V,C>(tiles.data() + i / 128 * TB, i % 128), "decoder mismatch");
            if constexpr (B == 5 && V == 2 && C == AIF25_MUL) {
                require(q[i] == aif25_level(tiles.data() + i / 128 * TB, i % 128), "model CPU decoder mismatch");
            }
        }
        CK(cudaFree(dt)); CK(cudaFree(dq));
        std::printf("DECODE PASS B=%d V=%d checks=%zu\n", B, V, q.size());
    }
    const std::pair<int,int> shapes[] = {{32,128}, {640,2560}, {2560,640}};
    for (const auto shape : shapes) {
        const int M = shape.first, K = shape.second, nu = K / 128;
        if (XSH && K % 256) continue;
        std::vector<uint8_t> w(size_t(M / 16) * nu * UB);
        for (uint8_t & v : w) v = rng();
        for (int slab = 0; slab < M / 16; ++slab) for (int u = 0; u < nu; ++u) {
            auto scales = reinterpret_cast<__half *>(w.data() + (slab * nu + u) * UB);
            for (int r = 0; r < 16; ++r) scales[r] = __float2half((int(rng() % 19) - 9) / 128.0f);
        }
        uint8_t * dw; int8_t * dx; float * ds, * dy;
        CK(cudaMalloc(&dw, w.size())); CK(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dx, 9 * K)); CK(cudaMalloc(&ds, 9 * (nu >> XSH) * sizeof(float))); CK(cudaMalloc(&dy, 9 * M * sizeof(float)));
        for (int N : {1,4,5,8,9}) {
            std::vector<int8_t> x(N * K); std::vector<float> s(N * (nu >> XSH)), y(N * M);
            for (int8_t & v : x) v = int(rng() % 255) - 127;
            for (float & v : s) v = (1 + rng() % 13) / 64.0f;
            CK(cudaMemcpy(dx, x.data(), x.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(ds, s.data(), s.size() * sizeof(float), cudaMemcpyHostToDevice));
            const dim3 grid(M / 16, (N + 7) / 8);
            magic2::mmv<B,V,C,XSH><<<grid,128>>>(dw, dx, ds, dy, M, K, N);
            CK(cudaGetLastError()); CK(cudaMemcpy(y.data(), dy, y.size() * sizeof(float), cudaMemcpyDeviceToHost));
            if (!perf) for (int n = 0; n < N; ++n) for (int m = 0; m < M; ++m) {
                double expected = 0, magnitude = 0;
                for (int u = 0; u < nu; ++u) {
                    const uint8_t * unit = w.data() + ((m / 16) * nu + u) * UB;
                    const float scale = __half2float(reinterpret_cast<const __half *>(unit)[m % 16]);
                    int dot = 0;
                    for (int k = 0; k < 128; ++k) dot += reference<B,V,C>(unit + 32 + (k / 8) * TB, (m % 16) * 8 + k % 8) * x[n * K + u * 128 + k];
                    const double value = dot * double(scale) * s[n * (nu >> XSH) + (u >> XSH)];
                    expected += value; magnitude += std::abs(value);
                }
                require(std::isfinite(y[n*M+m]) && std::abs(y[n*M+m] - expected) <= 2e-6 * (1 + magnitude), "matrix mismatch");
            }
            float ms = 0;
            if (perf) {
                // CUDA events time only the already-prepared matrix kernel.
                cudaEvent_t start, stop; CK(cudaEventCreate(&start)); CK(cudaEventCreate(&stop));
                for (int i = 0; i < 20; ++i) magic2::mmv<B,V,C,XSH><<<grid,128>>>(dw, dx, ds, dy, M, K, N);
                CK(cudaEventRecord(start));
                for (int i = 0; i < 1000; ++i) magic2::mmv<B,V,C,XSH><<<grid,128>>>(dw, dx, ds, dy, M, K, N);
                CK(cudaEventRecord(stop)); CK(cudaEventSynchronize(stop));
                CK(cudaEventElapsedTime(&ms, start, stop));
                CK(cudaEventDestroy(start)); CK(cudaEventDestroy(stop));
            }
            std::printf("MM %s B=%d V=%d M=%d K=%d N=%d kernel_us=%.6f bytes=%zu\n", perf ? "PERF" : "PASS", B,V,M,K,N,ms,w.size());
            std::fflush(stdout);
        }
        CK(cudaFree(dw)); CK(cudaFree(dx)); CK(cudaFree(ds)); CK(cudaFree(dy));
    }
}

int main(int argc, char ** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    const bool perf = argc == 3 && std::strcmp(argv[1], "perf") == 0;
    require(argc == 1 || (perf && (std::strcmp(argv[2], "pair") == 0 || std::strcmp(argv[2], "mix") == 0)), "usage: magic2_v2 [perf pair|mix]");
    std::mt19937 rng(20261024);
    if (!perf || std::strcmp(argv[2], "pair") == 0) {
        test<5,2,0x87D8D12Du>(rng, perf);
        test<5,2,0x87D8D12Du,1>(rng, perf);
    }
    if (!perf || std::strcmp(argv[2], "mix") == 0) {
        test<2,1,0xC3CCD12Du>(rng, perf);
        test<3,1,0x87D8D12Du>(rng, perf);
    }
    std::puts(perf ? "PERF COMPLETE prepared-activation kernel only" : "PASS computed-pair and scalar controls");
}
