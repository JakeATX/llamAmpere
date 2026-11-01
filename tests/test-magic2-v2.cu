#include "../ggml/src/ggml-cuda/magic2-v2.cuh"
#include "../ggml/src/ggml-aif25.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
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

static void test_large_row_offsets(std::mt19937 & rng) {
    constexpr int M = 32, K = 6144, N = 349536, NU = K/128, TB = 4*magic2::tile_words<5,2>;
    static_assert(int64_t(N-1)*K > INT32_MAX, "activation offsets cross signed 32-bit range");
    std::vector<uint8_t> w(size_t(M/16)*NU*AIF25_UNIT_BYTES);
    for (uint8_t & v : w) v = rng();
    for (size_t offset = 0; offset < w.size(); offset += AIF25_UNIT_BYTES) {
        for (int m = 0; m < 16; ++m) {
            const __half scale = __float2half(1.0f);
            std::memcpy(w.data()+offset+2*m, &scale, sizeof(scale));
        }
    }
    std::vector<float> scales(size_t(N)*(NU/2), 1.0f);
    uint8_t * dw; int8_t * dx; float * ds, * dy;
    CK(cudaMalloc(&dw, w.size())); CK(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dx, size_t(N)*K)); CK(cudaMemset(dx, 1, size_t(N)*K));
    CK(cudaMalloc(&ds, scales.size()*sizeof(float)));
    CK(cudaMemcpy(ds, scales.data(), scales.size()*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dy, size_t(N)*M*sizeof(float)));
    magic2::mmv<5,2,AIF25_MUL,1><<<dim3(M/16, (N+7)/8),128>>>(dw, dx, ds, dy, M, K, N);
    CK(cudaGetLastError());
    int expected[M] = {};
    for (int m = 0; m < M; ++m) for (int u = 0; u < NU; ++u) for (int k = 0; k < 128; ++k) {
        const uint8_t * unit = w.data()+((m/16)*NU+u)*AIF25_UNIT_BYTES;
        expected[m] += reference<5,2,AIF25_MUL>(unit+32+(k/8)*TB, (m%16)*8+k%8);
    }
    for (int row : {0, INT32_MAX/K-1, INT32_MAX/K, INT32_MAX/K+1, N-1}) {
        float y[M];
        CK(cudaMemcpy(y, dy+int64_t(row)*M, sizeof(y), cudaMemcpyDeviceToHost));
        for (int m = 0; m < M; ++m) {
            require(std::isfinite(y[m]) && y[m] == expected[m], "large activation row mismatch");
        }
    }
    CK(cudaFree(dw)); CK(cudaFree(dx)); CK(cudaFree(ds)); CK(cudaFree(dy));
    std::puts("PASS fractional activation offsets above INT32_MAX");
}

static std::vector<float> backend_product(ggml_backend_t backend, int K, int M, int tokens, int seqs, int groups,
                                         const std::vector<uint8_t> & w, const std::vector<float> & x) {
    ggml_context * ctx = ggml_init({ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true});
    require(ctx != nullptr, "backend context allocation failed");
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_AIF25_U, K, M);
    ggml_tensor * input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, K, tokens, seqs, groups);
    ggml_tensor * out = ggml_mul_mat(ctx, weight, input);
    require(out->ne[0] == M && out->ne[1] == tokens && out->ne[2] == seqs && out->ne[3] == groups, "backend output shape mismatch");
    require(ggml_backend_supports_op(backend, out), "fractional backend dispatch rejected activation shape");
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "backend tensor allocation failed");
    require(ggml_nbytes(weight) == w.size() && ggml_nbytes(input) == x.size()*sizeof(float), "backend input size mismatch");
    ggml_backend_tensor_set(weight, w.data(), 0, w.size());
    ggml_backend_tensor_set(input, x.data(), 0, x.size()*sizeof(float));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "fractional backend graph failed");
    std::vector<float> y(ggml_nelements(out));
    ggml_backend_tensor_get(out, y.data(), 0, y.size()*sizeof(float));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return y;
}

static void test_backend(std::mt19937 & rng) {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    require(backend != nullptr, "CUDA backend initialization failed");
    for (bool wide_k : {false, true}) {
        ggml_context * ctx = ggml_init({ggml_tensor_overhead()*4, nullptr, true});
        require(ctx != nullptr, "shape-only context allocation failed");
        const int64_t K = wide_k ? int64_t(INT32_MAX)+1 : 128;
        const int64_t M = wide_k ? 16 : int64_t(INT32_MAX)+1;
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_AIF25_U, K, M);
        ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
        require(!ggml_backend_supports_op(backend, ggml_mul_mat(ctx, weight, input)), "fractional dispatch narrowed an oversized dimension");
        ggml_free(ctx);
    }
    constexpr int M = 32;
    for (int K : {128, 256, 6144}) {
        std::vector<uint8_t> w(size_t(M/16)*(K/128)*AIF25_UNIT_BYTES);
        for (uint8_t & v : w) v = rng();
        for (size_t offset = 0; offset < w.size(); offset += AIF25_UNIT_BYTES) {
            for (int m = 0; m < 16; ++m) {
                const __half scale = __float2half((int(rng()%19)-9)/128.0f);
                std::memcpy(w.data()+offset+2*m, &scale, sizeof(scale));
            }
        }
        std::vector<float> decoded(size_t(M)*K);
        for (int slab = 0; slab < M/16; ++slab) {
            aif25_dequantize_slab(w.data()+size_t(slab)*(K/128)*AIF25_UNIT_BYTES,
                                  decoded.data()+size_t(slab)*16*K, K);
        }
        for (int tokens : {1, 5}) for (int seqs : {1, 3}) for (int groups : {1, 2}) {
            const int rows = tokens*seqs*groups;
            std::vector<float> x(rows*K);
            for (float & v : x) v = (int(rng()%255)-127)/64.0f;
            const auto y = backend_product(backend, K, M, tokens, seqs, groups, w, x);
            // Compare raw inputs with the CPU bit decoder and inverse rotation.
            double error = 0, norm = 0;
            for (int row = 0; row < rows; ++row) for (int m = 0; m < M; ++m) {
                double expected = 0;
                for (int k = 0; k < K; ++k) expected += double(decoded[m*K+k])*x[row*K+k];
                const double delta = y[row*M+m]-expected;
                error += delta*delta;
                norm += expected*expected;
            }
            const double relative = std::sqrt(error/std::max(norm, 1e-30));
            if (!(relative < .02)) std::fprintf(stderr, "fractional raw-input mismatch K=%d rows=%d relative=%.8g\n", K, rows, relative);
            require(relative < .02, "fractional raw-input CPU reference mismatch");
            // Each row gets its own graph so a dropped or mixed sequence cannot match.
            for (int row = 0; row < rows; ++row) {
                const std::vector<float> xi(x.begin()+row*K, x.begin()+(row+1)*K);
                const auto ref = backend_product(backend, K, M, 1, 1, 1, w, xi);
                for (int m = 0; m < M; ++m) {
                    require(std::isfinite(y[row*M+m]) && y[row*M+m] == ref[m], "backend sequence row mismatch");
                }
            }
            std::printf("BACKEND PASS K=%d tokens=%d seqs=%d groups=%d rows=%d\n", K, tokens, seqs, groups, rows);
        }
    }
    ggml_backend_free(backend);
    std::puts("PASS fractional raw-input CPU reference");
    std::puts("PASS fractional backend sequence dimensions");
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
    if (!perf) {
        test_backend(rng);
        test_large_row_offsets(rng);
    }
    std::puts(perf ? "PERF COMPLETE prepared-activation kernel only" : "PASS computed-pair and scalar controls");
}
