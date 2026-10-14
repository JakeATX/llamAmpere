// RQ4 M1 microbenchmark: W4A8 GEMM (rotated int4 weights x int8 activations on IMMA) at prefill widths,
// against raw cuBLAS fp16/bf16 GEMMs on the same shapes. The IQ4_XS MMQ and dequant+cuBLAS baselines
// come from test-backend-ops (GGML_TEST_RQ4_M1=1), see run.sh.
//
// build: nvcc -O3 -std=c++17 -arch=sm_86 -lineinfo -Xptxas -v tools/rq4-m1/rq4-m1.cu -lcublas -o rq4-m1
// run:   rq4-m1 [reps]           (correctness first; exits 1 if it fails)

#include "../../ggml/src/ggml-cuda/rq4.cuh"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS %d at %s:%d\n", (int) s_, __FILE__, __LINE__); exit(2); } } while (0)

using codec128 = rq4::rq4_codec_nibble<128>;
using codec64  = rq4::rq4_codec_nibble<64>;

// variants: (gs, bn, stages); stages 2 with bn 64 fits two CTAs per SM in shared memory
struct variant { int gs, bn, st; };
static const variant variants[] = { {128, 128, 3}, {128, 64, 3}, {128, 64, 2}, {64, 128, 3}, {64, 64, 3}, {64, 64, 2} };

static void run_gemm(const variant & v, const uint8_t * W, int64_t rb, const int8_t * q, const float * s, float * Y,
                     int M, int N, int K) {
    if (v.gs == 128 && v.bn == 128 && v.st == 3) rq4::launch_gemm<codec128, 128, 3>(W, rb, q, s, Y, M, M, N, K, 0);
    if (v.gs == 128 && v.bn ==  64 && v.st == 3) rq4::launch_gemm<codec128,  64, 3>(W, rb, q, s, Y, M, M, N, K, 0);
    if (v.gs == 128 && v.bn ==  64 && v.st == 2) rq4::launch_gemm<codec128,  64, 2>(W, rb, q, s, Y, M, M, N, K, 0);
    if (v.gs ==  64 && v.bn == 128 && v.st == 3) rq4::launch_gemm<codec64,  128, 3>(W, rb, q, s, Y, M, M, N, K, 0);
    if (v.gs ==  64 && v.bn ==  64 && v.st == 3) rq4::launch_gemm<codec64,   64, 3>(W, rb, q, s, Y, M, M, N, K, 0);
    if (v.gs ==  64 && v.bn ==  64 && v.st == 2) rq4::launch_gemm<codec64,   64, 2>(W, rb, q, s, Y, M, M, N, K, 0);
}

// ------------------------------------------------------------------------------------------------
// correctness

static bool check(const variant & v) {
    const int gs = v.gs;
    const int M = 256, K = 2048, N = 72;
    std::mt19937 rng(1234 + v.gs + v.bn + v.st);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> W((size_t) M*K), X((size_t) N*K);
    for (auto & v : W) v = nd(rng)*0.02f;
    for (auto & v : X) v = nd(rng);
    for (int i = 0; i < 64; ++i) X[rng() % X.size()] *= 30.0f;   // outliers
    const int64_t rb = (int64_t) (K/RQ4_SB)*rq4_sb_bytes(gs);
    std::vector<uint8_t> Wq((size_t) M*rb);
    for (int m = 0; m < M; ++m) rq4_quantize_row(&W[(size_t) m*K], &Wq[(size_t) m*rb], K, gs, nullptr);

    std::vector<int8_t> qc((size_t) N*K);
    std::vector<float>  sc((size_t) N*(K/RQ4_HB));
    for (int n = 0; n < N; ++n) rq4_quantize_act_row(&X[(size_t) n*K], &qc[(size_t) n*K], &sc[(size_t) n*(K/RQ4_HB)], K);

    uint8_t * dW; float * dX; int8_t * dq; float * ds; float * dY;
    CK(cudaMalloc(&dW, Wq.size())); CK(cudaMalloc(&dX, X.size()*4)); CK(cudaMalloc(&dq, qc.size()));
    CK(cudaMalloc(&ds, sc.size()*4)); CK(cudaMalloc(&dY, (size_t) M*N*4));
    CK(cudaMemcpy(dW, Wq.data(), Wq.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dX, X.data(), X.size()*4, cudaMemcpyHostToDevice));
    rq4::launch_quantize_act(dX, K, dq, ds, K, N, 0);
    run_gemm(v, dW, rb, dq, ds, dY, M, N, K);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<int8_t> qg(qc.size()); std::vector<float> sg(sc.size()), Y((size_t) M*N);
    CK(cudaMemcpy(qg.data(), dq, qg.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(sg.data(), ds, sg.size()*4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(Y.data(), dY, Y.size()*4, cudaMemcpyDeviceToHost));

    // 1) activation kernel vs CPU reference (FWHT summation order differs: allow +-1 on a few codes)
    int64_t qdiff = 0, qbad = 0; double smax = 0;
    for (size_t i = 0; i < qc.size(); ++i) { const int d = abs(qg[i] - qc[i]); qdiff += d != 0; qbad += d > 1; }
    for (size_t i = 0; i < sc.size(); ++i) smax = std::max(smax, (double) fabsf(sg[i] - sc[i])/std::max(1e-30f, sc[i]));

    // 2) GEMM vs CPU integer reference on the GPU's own activation codes (should match to fp32 rounding)
    std::vector<float> Wr((size_t) M*K);
    for (int m = 0; m < M; ++m) rq4_dequantize_row_rotated(&Wq[(size_t) m*rb], &Wr[(size_t) m*K], K, gs);
    double e_ref = 0, n_ref = 0, e_fp = 0, n_fp = 0;
    std::vector<float> xr(K);
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            xr[k] = sg[(size_t) n*(K/RQ4_HB) + k/RQ4_HB] *
                    qg[(size_t) n*K + (k/RQ4_CHUNK)*RQ4_CHUNK + rq4_act_byte(k & (RQ4_CHUNK - 1))];
        }
        for (int m = 0; m < M; ++m) {
            double r = 0, f = 0;
            for (int k = 0; k < K; ++k) { r += (double) Wr[(size_t) m*K + k]*xr[k]; f += (double) W[(size_t) m*K + k]*X[(size_t) n*K + k]; }
            const double y = Y[(size_t) n*M + m];
            e_ref += (y - r)*(y - r); n_ref += r*r;
            e_fp  += (y - f)*(y - f); n_fp  += f*f;
        }
    }
    // 3) decode/verify tile kernel (N <= 8) on the first nv tokens, same integer reference
    double e_mmv = 0, n_mmv = 0;
    if (v.bn == 128 && v.st == 3) {
        for (int kern = 0; kern < 2; ++kern)   // 0: mmv (old), 1: mmv_pair (default in ggml)
        for (int nv : {1, 3, 8}) {
            CK(cudaMemset(dY, 0, (size_t) M*N*4));
            if (kern == 0) {
                if (gs == 128) rq4::launch_mmv<codec128>(dW, rb, dq, ds, dY, M, M, nv, K, 0);
                else           rq4::launch_mmv<codec64 >(dW, rb, dq, ds, dY, M, M, nv, K, 0);
            } else {
                if (gs == 128) rq4::launch_mmv_pair<codec128>(dW, rb, dq, ds, dY, M, M, nv, K, 0);
                else           rq4::launch_mmv_pair<codec64 >(dW, rb, dq, ds, dY, M, M, nv, K, 0);
            }
            CK(cudaGetLastError());
            std::vector<float> Yv((size_t) M*N);
            CK(cudaMemcpy(Yv.data(), dY, Yv.size()*4, cudaMemcpyDeviceToHost));
            for (int n = 0; n < N; ++n) {
                for (int k = 0; k < K; ++k) {
                    xr[k] = sg[(size_t) n*(K/RQ4_HB) + k/RQ4_HB] *
                            qg[(size_t) n*K + (k/RQ4_CHUNK)*RQ4_CHUNK + rq4_act_byte(k & (RQ4_CHUNK - 1))];
                }
                for (int m = 0; m < M; ++m) {
                    double r = 0;
                    if (n < nv) for (int k = 0; k < K; ++k) r += (double) Wr[(size_t) m*K + k]*xr[k];
                    const double y = Yv[(size_t) n*M + m];   // n >= nv must stay 0 (untouched)
                    e_mmv += (y - r)*(y - r); n_mmv += r*r;
                }
            }
        }
    }
    const double nmse_mmv = n_mmv > 0 ? e_mmv/n_mmv : 0.0;
    const double nmse_ref = e_ref/n_ref, nmse_fp = e_fp/n_fp;
    const bool ok = qbad == 0 && qdiff*1000 < (int64_t) qc.size() && smax < 1e-5 && nmse_ref < 1e-9 && nmse_fp < 0.05 && nmse_mmv < 1e-9;
    printf("check gs=%d bn=%d st=%d: act codes off-by-1 %lld/%zu (>1: %lld), scale rel %.2e, "
           "NMSE vs int ref %.3e, NMSE vs fp32 matmul %.4f, mmv NMSE vs int ref %.3e -> %s\n",
           v.gs, v.bn, v.st, (long long) qdiff, qc.size(), (long long) qbad, smax, nmse_ref, nmse_fp, nmse_mmv, ok ? "OK" : "FAIL");
    cudaFree(dW); cudaFree(dX); cudaFree(dq); cudaFree(ds); cudaFree(dY);
    return ok;
}

// ------------------------------------------------------------------------------------------------
// timing

template <class F>
static double time_ms(F f, int reps) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int i = 0; i < 3; ++i) f();
    std::vector<float> t;
    for (int i = 0; i < reps; ++i) {
        CK(cudaEventRecord(a)); f(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b)); t.push_back(ms);
    }
    CK(cudaGetLastError());
    cudaEventDestroy(a); cudaEventDestroy(b);
    std::sort(t.begin(), t.end());
    return t[t.size()/2];
}

__global__ void fill_bytes(uint8_t * p, size_t n, uint32_t seed) {
    for (size_t i = blockIdx.x*(size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x*blockDim.x) {
        uint32_t x = (uint32_t) i*2654435761u ^ seed; x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
        p[i] = (uint8_t) x;
    }
}

__global__ void fix_scales(uint8_t * W, int64_t rows, int64_t sbs, int sbb, int scb) {
    // overwrite the scale header of every super block with a sane fp16 (0.002 + small variation)
    for (int64_t i = blockIdx.x*(int64_t) blockDim.x + threadIdx.x; i < rows*sbs; i += (int64_t) gridDim.x*blockDim.x) {
        half * d = (half *) (W + i*sbb);
        for (int g = 0; g < scb/2; ++g) d[g] = __float2half(0.002f + 1e-4f*((i + g) & 7));
    }
}

__global__ void fill_f32(float * p, size_t n) {
    for (size_t i = blockIdx.x*(size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x*blockDim.x) {
        uint32_t x = (uint32_t) i*2246822519u; x ^= x >> 15; x *= 0x27d4eb2du; x ^= x >> 13;
        p[i] = ((x & 0xffff)/32768.0f - 1.0f);
    }
}

__global__ void f32_to_f16(const float * a, half * b, size_t n) {
    for (size_t i = blockIdx.x*(size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x*blockDim.x) b[i] = __float2half(a[i]*0.05f);
}
__global__ void f32_to_bf16(const float * a, __nv_bfloat16 * b, size_t n) {
    for (size_t i = blockIdx.x*(size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x*blockDim.x) b[i] = __float2bfloat16(a[i]*0.05f);
}

int main(int argc, char ** argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 20;
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("device %s sm_%d%d, %d SMs\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    bool ok = true;
    for (const variant & v : variants) ok &= check(v);
    if (!ok) { printf("CORRECTNESS FAILED\n"); return 1; }

    // Qwen3.8-27B projections (K, M), same list as test-backend-ops add_rq4_m1_cases
    const int shapes[][2] = { {5120, 17408}, {17408, 5120}, {5120, 10240}, {5120, 6144}, {6144, 5120}, {5120, 12288} };
    const int Ns[] = { 512, 1024, 2048 };

    cublasHandle_t cb; CB(cublasCreate(&cb));
    CB(cublasSetMathMode(cb, CUBLAS_DEFAULT_MATH));

    printf("\n%-6s %-6s %-5s | %-26s | %s\n", "K", "M", "N", "arm", "median ms   TOPS(2MNK)");
    for (auto & sh : shapes) {
        const int K = sh[0], M = sh[1];
        const int Nmax = 2048;
        float * dX; int8_t * dq; float * ds; float * dY;
        CK(cudaMalloc(&dX, (size_t) Nmax*K*4)); CK(cudaMalloc(&dq, (size_t) Nmax*K));
        CK(cudaMalloc(&ds, (size_t) Nmax*(K/RQ4_HB)*4)); CK(cudaMalloc(&dY, (size_t) Nmax*M*4));
        fill_f32<<<1024, 256>>>(dX, (size_t) Nmax*K);

        uint8_t * dW[2]; int64_t rb[2];
        for (int v = 0; v < 2; ++v) {
            const int gs = v == 0 ? 128 : 64;
            rb[v] = (int64_t) (K/RQ4_SB)*rq4_sb_bytes(gs);
            CK(cudaMalloc(&dW[v], (size_t) M*rb[v]));
            fill_bytes<<<1024, 256>>>(dW[v], (size_t) M*rb[v], 77 + v);
            fix_scales<<<1024, 256>>>(dW[v], M, K/RQ4_SB, rq4_sb_bytes(gs), 2*(RQ4_SB/gs));
        }
        half * hW; half * hX; half * hY; __nv_bfloat16 * bW; __nv_bfloat16 * bX;
        CK(cudaMalloc(&hW, (size_t) M*K*2)); CK(cudaMalloc(&hX, (size_t) Nmax*K*2)); CK(cudaMalloc(&hY, (size_t) Nmax*M*2));
        CK(cudaMalloc(&bW, (size_t) M*K*2)); CK(cudaMalloc(&bX, (size_t) Nmax*K*2));
        {
            float * tmp; CK(cudaMalloc(&tmp, (size_t) M*K*4)); fill_f32<<<1024, 256>>>(tmp, (size_t) M*K);
            f32_to_f16<<<1024, 256>>>(tmp, hW, (size_t) M*K); f32_to_bf16<<<1024, 256>>>(tmp, bW, (size_t) M*K);
            CK(cudaDeviceSynchronize()); cudaFree(tmp);
            f32_to_f16<<<1024, 256>>>(dX, hX, (size_t) Nmax*K); f32_to_bf16<<<1024, 256>>>(dX, bX, (size_t) Nmax*K);
        }
        CK(cudaDeviceSynchronize());

        for (int N : Ns) {
            const double flop = 2.0*M*N*K;
            auto pr = [&](const char * arm, double ms) {
                printf("%-6d %-6d %-5d | %-26s | %9.4f   %6.1f\n", K, M, N, arm, ms, flop/ms*1e-9);
            };
            const double t_act = time_ms([&] { rq4::launch_quantize_act(dX, K, dq, ds, K, N, 0); }, reps);
            pr("rq4_act_quant", t_act);
            for (const variant & var : variants) {
                const int v = var.gs == 128 ? 0 : 1;
                char name[64];
                const double tg = time_ms([&] { run_gemm(var, dW[v], rb[v], dq, ds, dY, M, N, K); }, reps);
                snprintf(name, sizeof(name), "rq4_g%d_bn%d_s%d_gemm", var.gs, var.bn, var.st); pr(name, tg);
                const double tt = time_ms([&] {
                    rq4::launch_quantize_act(dX, K, dq, ds, K, N, 0);
                    run_gemm(var, dW[v], rb[v], dq, ds, dY, M, N, K); }, reps);
                snprintf(name, sizeof(name), "rq4_g%d_bn%d_s%d_total", var.gs, var.bn, var.st); pr(name, tt);
            }
            if (N == 512) {
                for (int nv : {1, 4, 8}) {
                    char name[64];
                    const double flop_v = 2.0*M*nv*K;
                    for (int kv = 0; kv < 4; ++kv) {
                        const int v = kv & 1, pair = kv >> 1;
                        const double tv = time_ms([&] {
                            if (!pair && v == 0) rq4::launch_mmv<codec128>     (dW[v], rb[v], dq, ds, dY, M, M, nv, K, 0);
                            if (!pair && v == 1) rq4::launch_mmv<codec64 >     (dW[v], rb[v], dq, ds, dY, M, M, nv, K, 0);
                            if ( pair && v == 0) rq4::launch_mmv_pair<codec128>(dW[v], rb[v], dq, ds, dY, M, M, nv, K, 0);
                            if ( pair && v == 1) rq4::launch_mmv_pair<codec64 >(dW[v], rb[v], dq, ds, dY, M, M, nv, K, 0); }, reps);
                        snprintf(name, sizeof(name), "rq4_g%d_%s_n%d", v == 0 ? 128 : 64, pair ? "mmvpair" : "mmv", nv);
                        printf("%-6d %-6d %-5d | %-26s | %9.4f   %6.1f   %6.1f GB/s weights\n", K, M, nv, name, tv,
                               flop_v/tv*1e-9, (double) M*rb[v]/tv*1e-6);
                    }
                }
            }
            const float a32 = 1.0f, b32 = 0.0f; const half a16 = __float2half(1.0f), b16 = __float2half(0.0f);
            pr("cublas_f16_acc16", time_ms([&] { CB(cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &a16,
                hW, CUDA_R_16F, K, hX, CUDA_R_16F, K, &b16, hY, CUDA_R_16F, M, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)); }, reps));
            pr("cublas_f16_acc32", time_ms([&] { CB(cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &a32,
                hW, CUDA_R_16F, K, hX, CUDA_R_16F, K, &b32, dY, CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)); }, reps));
            pr("cublas_bf16_acc32", time_ms([&] { CB(cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &a32,
                bW, CUDA_R_16BF, K, bX, CUDA_R_16BF, K, &b32, dY, CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)); }, reps));
        }
        cudaFree(dX); cudaFree(dq); cudaFree(ds); cudaFree(dY); cudaFree(dW[0]); cudaFree(dW[1]);
        cudaFree(hW); cudaFree(hX); cudaFree(hY); cudaFree(bW); cudaFree(bX);
    }
    cublasDestroy(cb);
    printf("DONE\n");
    return 0;
}
