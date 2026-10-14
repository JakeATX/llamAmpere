// AI4 (Ampere INT4, formerly D2) kernel microbench + exactness check (ai4-kern.cuh), standalone (no ggml).
//
// 1) exactness: the kernels against a CPU integer reference built from the same GPU-quantized activations and the
//    CPU decoder (ggml-ai4.h ai4_get_state / ai4_level): per 128-group int32 dot, times d_w * d_x, summed in double.
//    Only fp32 accumulation order differs, so the bound is tight: max |err| / rms(Y) < 1e-5 (K = 17408 at N = 512
//    reaches 2.1e-6 from summing 136 chunk partials in fp32; an indexing or decode bug gives O(1)).
// 2) W4A8 error: the kernels against the reconstruct reference (k_ai4_reconstruct to f32, original basis, f32 x)
//    computed in double on the CPU, as NMSE (the activation int8 rounding; test-backend-ops allows 5e-4).
// 3) timing: Qwen3.8-27B projection shapes, decode/verify widths 1..8 and prefill widths 512..2048, every kernel
//    variant, CUDA events, median of R repetitions (L2 flushed between reps for N <= 8 by rotating over enough
//    weight copies that the working set exceeds L2).
//
// build: part of tools/ai4 (target llama-ai4-kern-bench), or
//        nvcc -O3 -std=c++17 -arch=sm_86 -lineinfo -Xptxas -v -I ggml/src tools/ai4/ai4-kern-bench.cu -o ai4-kern-bench
// run:   llama-ai4-kern-bench [check|bench|all] [reps]   (default all; exit 1 if a check fails)
//        llama-ai4-kern-bench gguf <file.gguf>          real-tensor parity: every AI4 tensor of the first, middle and
//                                                      last block at N = 1, 8, 64, 512 (rows subsampled)

#include "ggml-cuda/ai4-kern.cuh"
#include "ggml-cuda/ai4.cuh"
#include "ggml-ai4.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

// ------------------------------------------------------------------------------------------------
// kernel variants

enum kvar { MMV_XS, MMV_S2, MMV_S3, MMV_S4, GEMM_64, GEMM_128, GEMM_64_NTB2, GEMM_128_NTB2, GEMM_128_NTB8, NVAR };
static const char * kvar_name[NVAR] = { "mmv_xsmem_s3", "mmv_s2", "mmv_s3", "mmv_s4", "gemm_bn64", "gemm_bn128", "gemm_bn64_ntb2",
                                        "gemm_bn128_ntb2", "gemm_bn128_ntb8" };

static void run(kvar v, const uint8_t * W, const int8_t * q, const float * s, float * Y, int M, int N, int K) {
    switch (v) {
        case MMV_XS:        ai4k::launch_mmv<3, 3, false>(W, q, s, Y, M, M, N, K, 0); break;
        case MMV_S2:        ai4k::launch_mmv<2, 6>(W, q, s, Y, M, M, N, K, 0); break;
        case MMV_S3:        ai4k::launch_mmv<3, 6>(W, q, s, Y, M, M, N, K, 0); break;
        case MMV_S4:        ai4k::launch_mmv<4, 6>(W, q, s, Y, M, M, N, K, 0); break;
        case GEMM_64:       ai4k::launch_gemm<64>(W, q, s, Y, M, M, N, K, 0); break;
        case GEMM_128:      ai4k::launch_gemm<128>(W, q, s, Y, M, M, N, K, 0); break;
        case GEMM_64_NTB2:  ai4k::launch_gemm<64, 3, 2>(W, q, s, Y, M, M, N, K, 0); break;
        case GEMM_128_NTB2: ai4k::launch_gemm<128, 3, 2>(W, q, s, Y, M, M, N, K, 0); break;
        case GEMM_128_NTB8: ai4k::launch_gemm<128, 3, 8>(W, q, s, Y, M, M, N, K, 0); break;
        default: break;
    }
}

// ------------------------------------------------------------------------------------------------
// random valid AI4 bytes

static std::vector<uint8_t> random_ai4(int M, int K, std::mt19937 & rng) {
    std::vector<uint8_t> b((size_t) (M/16)*ai4k::slab_bytes(K));
    std::uniform_real_distribution<float> ud(0.002f, 0.02f);
    for (size_t o = 0; o < b.size(); o += ai4k::SB_BYTES) {
        uint16_t * d = (uint16_t *) &b[o];
        for (int i = 0; i < 128; ++i) d[i] = ai4_fp32_to_fp16(ud(rng));
        uint32_t * w = (uint32_t *) &b[o + 256];
        for (int i = 0; i < 128*16; ++i) w[i] = (uint32_t) rng();
    }
    return b;
}

// ------------------------------------------------------------------------------------------------
// exactness

static bool verify(const std::vector<uint8_t> & Wb, int M, int K, int N, int rstep, const char * label, std::mt19937 & rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> X((size_t) N*K);
    for (auto & v : X) v = nd(rng);
    for (int i = 0; i < 16; ++i) X[rng() % X.size()] *= 30.0f;

    uint8_t * dW; float * dX; int8_t * dq; float * ds; float * dY; float * dWr;
    CK(cudaMalloc(&dW, Wb.size())); CK(cudaMalloc(&dX, X.size()*4)); CK(cudaMalloc(&dq, (size_t) N*K));
    CK(cudaMalloc(&ds, (size_t) N*(K/256)*4)); CK(cudaMalloc(&dY, (size_t) M*N*4)); CK(cudaMalloc(&dWr, (size_t) M*K*4));
    CK(cudaMemcpy(dW, Wb.data(), Wb.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dX, X.data(), X.size()*4, cudaMemcpyHostToDevice));
    rq4::launch_quantize_act(dX, K, dq, ds, K, N, 0);
    ai4_reconstruct_launch<float>(dW, dWr, K, M, 0);
    CK(cudaDeviceSynchronize());
    std::vector<int8_t> q((size_t) N*K); std::vector<float> s((size_t) N*(K/256)), Wr((size_t) M*K);
    CK(cudaMemcpy(q.data(), dq, q.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(s.data(), ds, s.size()*4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(Wr.data(), dWr, Wr.size()*4, cudaMemcpyDeviceToHost));

    // natural-order int8 activations: byte t*32 + j*8 + h*4 + b of chunk c = x'[128c + 32j + 16h + 4t + b]
    std::vector<int8_t> xn((size_t) N*K);
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < K/128; ++c)
            for (int t = 0; t < 4; ++t) for (int j = 0; j < 4; ++j) for (int h = 0; h < 2; ++h) for (int b = 0; b < 4; ++b)
                xn[(size_t) n*K + 128*c + 32*j + 16*h + 4*t + b] = q[(size_t) n*K + 128*c + t*32 + j*8 + h*4 + b];
    // levels and scales (rotated basis), CPU decoder
    std::vector<int16_t> L((size_t) M*K); std::vector<float> D((size_t) M*(K/128));
    const int64_t sbytes = ai4k::slab_bytes(K);
    for (int sl = 0; sl < M/16; ++sl)
        for (int sb = 0; sb < K/1024; ++sb) {
            const uint8_t * blk = Wb.data() + sl*sbytes + (int64_t) sb*ai4k::SB_BYTES;
            const uint16_t * dh = (const uint16_t *) blk;
            const uint32_t * wd = (const uint32_t *) (blk + 256);
            for (int r = 0; r < 16; ++r) {
                for (int g = 0; g < 8; ++g) D[(size_t) (sl*16 + r)*(K/128) + sb*8 + g] = ai4_fp16_to_fp32(dh[r*8 + g]);
                for (int k = 0; k < 1024; ++k)
                    L[(size_t) (sl*16 + r)*K + sb*1024 + k] = (int16_t) ai4_level(ai4_get_state(wd + (k/8)*16, r*8 + k%8));
            }
        }
    std::vector<double> Yi((size_t) M*N), Yf((size_t) M*N);
    for (int n = 0; n < N; ++n)
        for (int m = 0; m < M; m += rstep) {
            double a = 0, af = 0;
            for (int c = 0; c < K/128; ++c) {
                int64_t p = 0;
                for (int k = 128*c; k < 128*c + 128; ++k) p += (int64_t) L[(size_t) m*K + k]*xn[(size_t) n*K + k];
                a += (double) p*D[(size_t) m*(K/128) + c]*s[(size_t) n*(K/256) + c/2];
            }
            for (int k = 0; k < K; ++k) af += (double) Wr[(size_t) m*K + k]*X[(size_t) n*K + k];
            Yi[(size_t) n*M + m] = a; Yf[(size_t) n*M + m] = af;
        }
    double rms = 0, rmsf = 0; size_t nchk = 0;
    for (int n = 0; n < N; ++n)
        for (int m = 0; m < M; m += rstep) { const size_t i = (size_t) n*M + m; rms += Yi[i]*Yi[i]; rmsf += Yf[i]*Yf[i]; nchk++; }
    rms = sqrt(rms/nchk);

    bool ok = true;
    std::vector<float> Y((size_t) M*N);
    for (int v = 0; v < NVAR; ++v) {
        const bool is_mmv = v <= MMV_S4;
        if (is_mmv != (N <= 8)) continue;
        CK(cudaMemset(dY, 0xff, (size_t) M*N*4));
        run((kvar) v, dW, dq, ds, dY, M, N, K);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(Y.data(), dY, Y.size()*4, cudaMemcpyDeviceToHost));
        double emax = 0, ef = 0;
        bool nan = false;
        for (size_t i = 0; i < Y.size(); ++i) {
            if (!std::isfinite(Y[i])) nan = true;
            if (((int) (i % M)) % rstep != 0) continue;
            emax = std::max(emax, fabs(Y[i] - Yi[i]));
            ef += (Y[i] - Yf[i])*(Y[i] - Yf[i]);
        }
        const double rel = emax/rms, nmse = ef/rmsf;
        const bool pass = !nan && rel < 1e-5 && nmse < 5e-4;
        ok &= pass;
        printf("%-28s M %5d K %5d N %4d %-16s max|err|/rms(int ref) %.2e  NMSE vs f32 reconstruct %.2e  %s\n",
               label, M, K, N, kvar_name[v], rel, nmse, pass ? "ok" : "FAIL");
    }
    cudaFree(dW); cudaFree(dX); cudaFree(dq); cudaFree(ds); cudaFree(dY); cudaFree(dWr);
    return ok;
}

static bool check(int M, int K, int N, std::mt19937 & rng) {
    return verify(random_ai4(M, K, rng), M, K, N, 1, "random", rng);
}

// real tensors: every AI4 tensor of blocks 0, n/2 and n-1 of a GGUF (gguf.h from ggml-base), rows subsampled for the
// CPU references
static bool check_gguf(const char * path) {
    ggml_context * meta = nullptr;
    gguf_init_params ip = { /*no_alloc =*/ true, /*ctx =*/ &meta };
    gguf_context * g = gguf_init_from_file(path, ip);
    if (!g) { fprintf(stderr, "cannot open %s\n", path); return false; }
    const int64_t nt = gguf_get_n_tensors(g);
    int nblk = 0;
    for (int64_t i = 0; i < nt; ++i) {
        int b = -1;
        if (sscanf(gguf_get_tensor_name(g, i), "blk.%d.", &b) == 1) nblk = std::max(nblk, b + 1);
    }
    FILE * f = fopen(path, "rb");
    std::mt19937 rng(99);
    bool ok = true; int n_checked = 0;
    for (int64_t i = 0; i < nt; ++i) {
        if (gguf_get_tensor_type(g, i) != GGML_TYPE_AI4) continue;
        const char * name = gguf_get_tensor_name(g, i);
        int b = -1;
        if (sscanf(name, "blk.%d.", &b) != 1 || (b != 0 && b != nblk/2 && b != nblk - 1)) continue;
        const size_t sz = gguf_get_tensor_size(g, i);
        const size_t off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, i);
        const ggml_tensor * t = ggml_get_tensor(meta, name);
        const int64_t ne0 = t->ne[0], ne1 = t->ne[1];
        std::vector<uint8_t> Wb(sz);
        fseeko(f, (off_t) off, SEEK_SET);
        if (fread(Wb.data(), 1, sz, f) != sz) { fprintf(stderr, "short read %s\n", name); ok = false; break; }
        const int M = (int) ne1, K = (int) ne0;
        if ((size_t) (M/16)*ai4k::slab_bytes(K) != sz) { fprintf(stderr, "%s: size mismatch\n", name); ok = false; continue; }
        const int rstep = M >= 4096 ? 61 : 7;
        for (int N : {1, 8, 64, 512}) ok &= verify(Wb, M, K, N, rstep, name, rng);
        n_checked++;
    }
    fclose(f);
    gguf_free(g);
    ggml_free(meta);
    printf("gguf check: %d tensors, %s\n", n_checked, ok && n_checked > 0 ? "PASS" : "FAIL");
    return ok && n_checked > 0;
}

// ------------------------------------------------------------------------------------------------
// timing

static float median(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size()/2]; }

static void bench(int reps) {
    struct shape { int M, K; const char * name; };
    const shape shapes[] = { {17408, 5120, "gate/up"}, {5120, 17408, "down"}, {10240, 5120, "10240x5120"},
                             {6144, 5120, "6144x5120"}, {5120, 6144, "5120x6144"}, {12288, 5120, "12288x5120"} };
    const int widths[] = { 1, 2, 4, 5, 6, 7, 8, 512, 1024, 2048 };
    std::mt19937 rng(7);
    int l2 = 0; CK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
    printf("\nbench: median of %d reps, times in us; weights rotated over >= 4x L2 (%d MiB) for N <= 8\n", reps, l2 >> 20);
    printf("%-11s %6s %6s %5s %-16s %9s %9s %9s %9s\n", "shape", "M", "K", "N", "kernel", "act_us", "kern_us", "GB/s", "TOPS");
    for (const shape & sh : shapes) {
        const size_t wb = (size_t) (sh.M/16)*ai4k::slab_bytes(sh.K);
        const int ncopy = (int) std::max<size_t>(1, (4*(size_t) l2 + wb - 1)/wb);
        std::vector<uint8_t> Wb = random_ai4(sh.M, sh.K, rng);
        uint8_t * dW; CK(cudaMalloc(&dW, wb*ncopy));
        for (int c = 0; c < ncopy; ++c) CK(cudaMemcpy(dW + c*wb, Wb.data(), wb, cudaMemcpyHostToDevice));
        const int NMAX = 2048;
        float * dX; int8_t * dq; float * ds; float * dY;
        CK(cudaMalloc(&dX, (size_t) NMAX*sh.K*4)); CK(cudaMalloc(&dq, (size_t) NMAX*sh.K));
        CK(cudaMalloc(&ds, (size_t) NMAX*(sh.K/256)*4)); CK(cudaMalloc(&dY, (size_t) NMAX*sh.M*4));
        {
            std::vector<float> X((size_t) NMAX*sh.K);
            std::normal_distribution<float> nd(0.0f, 1.0f);
            for (auto & v : X) v = nd(rng);
            CK(cudaMemcpy(dX, X.data(), X.size()*4, cudaMemcpyHostToDevice));
        }
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        for (int N : widths) {
            const int r = N <= 8 ? reps : std::max(5, reps/10);
            std::vector<float> ta;
            for (int i = 0; i < r + 2; ++i) {
                CK(cudaEventRecord(e0));
                rq4::launch_quantize_act(dX, sh.K, dq, ds, sh.K, N, 0);
                CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
                float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); if (i >= 2) ta.push_back(ms*1e3f);
            }
            for (int v = 0; v < NVAR; ++v) {
                const bool is_mmv = v <= MMV_S4;
                if (is_mmv != (N <= 8)) continue;
                std::vector<float> tk;
                for (int i = 0; i < r + 2; ++i) {
                    const uint8_t * W = dW + (size_t) (i % ncopy)*wb;
                    CK(cudaEventRecord(e0));
                    run((kvar) v, W, dq, ds, dY, sh.M, N, sh.K);
                    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
                    float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); if (i >= 2) tk.push_back(ms*1e3f);
                }
                CK(cudaGetLastError());
                const float us = median(tk);
                printf("%-11s %6d %6d %5d %-16s %9.2f %9.2f %9.1f %9.1f\n", sh.name, sh.M, sh.K, N, kvar_name[v],
                       median(ta), us, wb/(us*1e3), 2.0*sh.M*sh.K*(double) N/(us*1e6));
            }
        }
        cudaFree(dW); cudaFree(dX); cudaFree(dq); cudaFree(ds); cudaFree(dY);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }
}


// ------------------------------------------------------------------------------------------------
// AI5_U (LAY_U5) / AI4_U (LAY_U) and the span layouts AI5_S (LAY_S5) / AI4_S (LAY_S): exactness of the kernels vs a CPU
// integer reference (aik_get_state), NMSE vs the original-basis f32 reconstruct, and a decode/prefill timing table over
// k = 4 / 5 x unit / span. The span layout drops the 7 redundant 32-byte headers per 1024 K, so its only kernel-visible
// difference is where the fp16 scales live (sc_off): same tiles, same order, one header per 8 chunks.

template <int LAY, int XSH>
static void run_u(int v, const uint8_t * W, const int8_t * q, const float * s, float * Y, int M, int N, int K) {
    switch (v) {
        case 0: ai4k::launch_mmv<2, 6, true, LAY, XSH>(W, q, s, Y, M, M, N, K, 0); break;
        case 1: ai4k::launch_mmv<3, 6, true, LAY, XSH>(W, q, s, Y, M, M, N, K, 0); break;
        case 2: ai4k::launch_gemm<64, 3, 4, LAY, XSH>(W, q, s, Y, M, M, N, K, 0); break;
        case 3: ai4k::launch_gemm<128, 3, 4, LAY, XSH>(W, q, s, Y, M, M, N, K, 0); break;
        default: break;
    }
}
static const char * run_u_name[4] = { "mmv_s2", "mmv_s3", "gemm_bn64", "gemm_bn128" };

static std::vector<uint8_t> random_unit(int kb, int M, int K, std::mt19937 & rng) {
    const int ub = 32 + 256*kb;
    std::vector<uint8_t> b((size_t) (M/16)*(K/128)*ub);
    std::uniform_real_distribution<float> ud(0.002f, 0.02f);
    for (size_t o = 0; o < b.size(); o += ub) {
        uint16_t * d = (uint16_t *) &b[o];
        for (int i = 0; i < 16; ++i) d[i] = ai4_fp32_to_fp16(ud(rng));
        uint32_t * w = (uint32_t *) &b[o + 32];
        for (int i = 0; i < 64*kb; ++i) w[i] = (uint32_t) rng();
    }
    return b;
}

static std::vector<uint8_t> random_span(int kb, int M, int K, std::mt19937 & rng) {
    const int sb = 32 + 2048*kb;
    std::vector<uint8_t> b((size_t) (M/16)*(K/1024)*sb);
    std::uniform_real_distribution<float> ud(0.002f, 0.02f);
    for (size_t o = 0; o < b.size(); o += sb) {
        uint16_t * d = (uint16_t *) &b[o];
        for (int i = 0; i < 16; ++i) d[i] = ai4_fp32_to_fp16(ud(rng));
        uint32_t * w = (uint32_t *) &b[o + 32];
        for (int i = 0; i < 512*kb; ++i) w[i] = (uint32_t) rng();
    }
    return b;
}

template <int KB, bool SPAN = false>
static bool verify_u(const std::vector<uint8_t> & Wb, int M, int K, int N, int rstep, const char * label, std::mt19937 & rng) {
    constexpr int LAY = SPAN ? (KB == 5 ? ai4k::LAY_S5 : ai4k::LAY_S) : (KB == 5 ? ai4k::LAY_U5 : ai4k::LAY_U);
    GGML_ASSERT(!SPAN || K % 1024 == 0);
    const int xsh = K % 256 == 0 ? 1 : 0, ns = K >> (7 + xsh), ub = 32 + 256*KB, sb = 32 + 2048*KB;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> X((size_t) N*K);
    for (auto & v : X) v = nd(rng);
    for (int i = 0; i < 16; ++i) X[rng() % X.size()] *= 30.0f;
    uint8_t * dW; float * dX; int8_t * dq; float * ds; float * dY; float * dWr;
    CK(cudaMalloc(&dW, Wb.size())); CK(cudaMalloc(&dX, X.size()*4)); CK(cudaMalloc(&dq, (size_t) N*K));
    CK(cudaMalloc(&ds, (size_t) N*ns*4)); CK(cudaMalloc(&dY, (size_t) M*N*4)); CK(cudaMalloc(&dWr, (size_t) M*K*4));
    CK(cudaMemcpy(dW, Wb.data(), Wb.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dX, X.data(), X.size()*4, cudaMemcpyHostToDevice));
    if (xsh) rq4::launch_quantize_act(dX, K, dq, ds, K, N, 0); else ai4k::launch_quantize_act128(dX, K, dq, ds, K, N, 0);
    aik_reconstruct_launch<float, KB, SPAN>(dW, dWr, K, M, true, 0);
    CK(cudaDeviceSynchronize());
    std::vector<int8_t> q((size_t) N*K); std::vector<float> s((size_t) N*ns), Wr((size_t) M*K);
    CK(cudaMemcpy(q.data(), dq, q.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(s.data(), ds, s.size()*4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(Wr.data(), dWr, Wr.size()*4, cudaMemcpyDeviceToHost));
    std::vector<int8_t> xn((size_t) N*K);
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < K/128; ++c)
            for (int t = 0; t < 4; ++t) for (int j = 0; j < 4; ++j) for (int h = 0; h < 2; ++h) for (int b = 0; b < 4; ++b)
                xn[(size_t) n*K + 128*c + 32*j + 16*h + 4*t + b] = q[(size_t) n*K + 128*c + t*32 + j*8 + h*4 + b];
    std::vector<int16_t> L((size_t) M*K); std::vector<float> D((size_t) M*(K/128));
    for (int sl = 0; sl < M/16; ++sl)
        for (int u = 0; u < K/128; ++u) {
            const uint8_t * blk = SPAN ? Wb.data() + ((size_t) sl*(K/1024) + (u >> 3))*sb
                                       : Wb.data() + ((size_t) sl*(K/128) + u)*ub;
            const uint16_t * dh = (const uint16_t *) blk;
            const uint32_t * wd = (const uint32_t *) (blk + 32) + (SPAN ? (size_t) (u & 7)*16*4*KB : 0);
            for (int r = 0; r < 16; ++r) {
                D[(size_t) (sl*16 + r)*(K/128) + u] = ai4_fp16_to_fp32(dh[r]);
                for (int k = 0; k < 128; ++k)
                    L[(size_t) (sl*16 + r)*K + u*128 + k] = (int16_t) ai4_level(aik_get_state(wd + (k/8)*4*KB, KB, r*8 + k%8));
            }
        }
    std::vector<double> Yi((size_t) M*N), Yf((size_t) M*N);
    for (int n = 0; n < N; ++n)
        for (int m = 0; m < M; m += rstep) {
            double a = 0, af = 0;
            for (int c = 0; c < K/128; ++c) {
                int64_t p = 0;
                for (int k = 128*c; k < 128*c + 128; ++k) p += (int64_t) L[(size_t) m*K + k]*xn[(size_t) n*K + k];
                a += (double) p*D[(size_t) m*(K/128) + c]*s[(size_t) n*ns + (c >> xsh)];
            }
            for (int k = 0; k < K; ++k) af += (double) Wr[(size_t) m*K + k]*X[(size_t) n*K + k];
            Yi[(size_t) n*M + m] = a; Yf[(size_t) n*M + m] = af;
        }
    double rms = 0, rmsf = 0; size_t nchk = 0;
    for (int n = 0; n < N; ++n)
        for (int m = 0; m < M; m += rstep) { const size_t i = (size_t) n*M + m; rms += Yi[i]*Yi[i]; rmsf += Yf[i]*Yf[i]; nchk++; }
    rms = sqrt(rms/nchk);
    bool ok = true;
    std::vector<float> Y((size_t) M*N);
    for (int v = 0; v < 4; ++v) {
        if ((v < 2) != (N <= 8)) continue;
        CK(cudaMemset(dY, 0xff, (size_t) M*N*4));
        if (xsh) run_u<LAY, 1>(v, dW, dq, ds, dY, M, N, K); else run_u<LAY, 0>(v, dW, dq, ds, dY, M, N, K);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(Y.data(), dY, Y.size()*4, cudaMemcpyDeviceToHost));
        double emax = 0, ef = 0; bool nan = false;
        for (size_t i = 0; i < Y.size(); ++i) {
            if (!std::isfinite(Y[i])) nan = true;
            if (((int) (i % M)) % rstep != 0) continue;
            emax = std::max(emax, fabs(Y[i] - Yi[i]));
            ef += (Y[i] - Yf[i])*(Y[i] - Yf[i]);
        }
        const double rel = emax/rms, nmse = ef/rmsf;
        const bool pass = !nan && rel < 1e-5 && nmse < 5e-4;
        ok &= pass;
        printf("k%d%s %-23s M %5d K %5d N %4d %-10s max|err|/rms(int ref) %.2e  NMSE vs f32 reconstruct %.2e  %s\n",
               KB, SPAN ? "s" : "u", label, M, K, N, run_u_name[v], rel, nmse, pass ? "ok" : "FAIL");
    }
    cudaFree(dW); cudaFree(dX); cudaFree(dq); cudaFree(ds); cudaFree(dY); cudaFree(dWr);
    return ok;
}

static bool check_u(void) {
    std::mt19937 rng(4321);
    bool ok = true;
    for (int N : {1, 3, 8}) { ok &= verify_u<5>(random_unit(5, 256, 2048, rng), 256, 2048, N, 1, "random", rng);
                              ok &= verify_u<4>(random_unit(4, 256, 2048, rng), 256, 2048, N, 1, "random", rng); }
    ok &= verify_u<5>(random_unit(5, 144, 5120, rng), 144, 5120, 1, 1, "random", rng);
    ok &= verify_u<5>(random_unit(5, 48, 640, rng), 48, 640, 7, 1, "random H128", rng);
    ok &= verify_u<5>(random_unit(5, 64, 1152, rng), 64, 1152, 5, 1, "random H128", rng);
    for (int N : {9, 64, 65, 200}) ok &= verify_u<5>(random_unit(5, 256, 2048, rng), 256, 2048, N, 1, "random", rng);
    ok &= verify_u<5>(random_unit(5, 144, 3072, rng), 144, 3072, 130, 1, "random", rng);
    ok &= verify_u<5>(random_unit(5, 48, 640, rng), 48, 640, 70, 1, "random H128", rng);
    ok &= verify_u<5>(random_unit(5, 400, 1024, rng), 400, 1024, 600, 1, "random", rng);
    // the span layouts (K % 1024 == 0 only)
    for (int N : {1, 3, 8}) { ok &= verify_u<5, true>(random_span(5, 256, 2048, rng), 256, 2048, N, 1, "random span", rng);
                              ok &= verify_u<4, true>(random_span(4, 256, 2048, rng), 256, 2048, N, 1, "random span", rng); }
    ok &= verify_u<4, true>(random_span(4, 144, 5120, rng), 144, 5120, 1, 1, "random span", rng);
    ok &= verify_u<5, true>(random_span(5, 16, 1024, rng), 16, 1024, 1, 1, "random span", rng);
    ok &= verify_u<5, true>(random_span(5, 48, 2048, rng), 48, 2048, 7, 1, "random span", rng);
    for (int N : {9, 64, 65, 200}) ok &= verify_u<5, true>(random_span(5, 256, 2048, rng), 256, 2048, N, 1, "random span", rng);
    for (int N : {9, 65}) ok &= verify_u<4, true>(random_span(4, 256, 2048, rng), 256, 2048, N, 1, "random span", rng);
    ok &= verify_u<5, true>(random_span(5, 144, 3072, rng), 144, 3072, 130, 1, "random span", rng);
    ok &= verify_u<5, true>(random_span(5, 400, 1024, rng), 400, 1024, 600, 1, "random span", rng);
    printf("ucheck: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// U (k=4) vs U5 (k=5) timing on the attention shapes, N = 1..8 and 512
static void bench_u(int reps) {
    struct shape { int M, K; const char * name; };
    const shape shapes[] = { {10240, 5120, "attn_qkv"}, {6144, 5120, "attn_gate"}, {5120, 6144, "ssm_out"},
                             {12288, 5120, "attn_q"}, {5120, 6144, "attn_output"}, {17408, 5120, "ffn_up"} };
    const int widths[] = { 1, 2, 3, 4, 5, 6, 7, 8, 512 };
    std::mt19937 rng(7);
    int l2 = 0; CK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
    printf("\nbench_u: median of %d reps (us); weights rotated over >= 4x L2\n", reps);
    printf("%-11s %6s %6s %5s %-4s %-10s %9s %9s\n", "shape", "M", "K", "N", "type", "kernel", "kern_us", "GB/s");
    for (const shape & sh : shapes) {
        for (int arm = 0; arm < 4; ++arm) {
            const int  kb   = (arm & 1) ? 5 : 4;
            const bool span = (arm & 2) != 0;
            if (span && sh.K % 1024 != 0) continue;
            const size_t wb = span ? (size_t) (sh.M/16)*(sh.K/1024)*(32 + 2048*kb)
                                   : (size_t) (sh.M/16)*(sh.K/128)*(32 + 256*kb);
            const int ncopy = (int) std::max<size_t>(1, (4*(size_t) l2 + wb - 1)/wb);
            std::vector<uint8_t> Wb = span ? random_span(kb, sh.M, sh.K, rng) : random_unit(kb, sh.M, sh.K, rng);
            uint8_t * dW; CK(cudaMalloc(&dW, wb*ncopy));
            for (int c = 0; c < ncopy; ++c) CK(cudaMemcpy(dW + c*wb, Wb.data(), wb, cudaMemcpyHostToDevice));
            const int NMAX = 512;
            float * dX; int8_t * dq; float * ds; float * dY;
            CK(cudaMalloc(&dX, (size_t) NMAX*sh.K*4)); CK(cudaMalloc(&dq, (size_t) NMAX*sh.K));
            CK(cudaMalloc(&ds, (size_t) NMAX*(sh.K/256)*4)); CK(cudaMalloc(&dY, (size_t) NMAX*sh.M*4));
            std::vector<float> X((size_t) NMAX*sh.K); std::normal_distribution<float> nd(0.0f, 1.0f);
            for (auto & v : X) v = nd(rng);
            CK(cudaMemcpy(dX, X.data(), X.size()*4, cudaMemcpyHostToDevice));
            rq4::launch_quantize_act(dX, sh.K, dq, ds, sh.K, NMAX, 0);
            cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
            for (int N : widths) {
                const int r = N <= 8 ? reps : std::max(5, reps/10);
                for (int v = 0; v < 4; ++v) {
                    if ((v < 2) != (N <= 8)) continue;
                    std::vector<float> tk;
                    for (int i = 0; i < r + 2; ++i) {
                        const uint8_t * W = dW + (size_t) (i % ncopy)*wb;
                        CK(cudaEventRecord(e0));
                        if (span) { if (kb == 5) run_u<ai4k::LAY_S5, 1>(v, W, dq, ds, dY, sh.M, N, sh.K);
                                    else         run_u<ai4k::LAY_S,  1>(v, W, dq, ds, dY, sh.M, N, sh.K); }
                        else      { if (kb == 5) run_u<ai4k::LAY_U5, 1>(v, W, dq, ds, dY, sh.M, N, sh.K);
                                    else         run_u<ai4k::LAY_U,  1>(v, W, dq, ds, dY, sh.M, N, sh.K); }
                        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
                        float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); if (i >= 2) tk.push_back(ms*1e3f);
                    }
                    CK(cudaGetLastError());
                    const float us = median(tk);
                    char tn[8]; snprintf(tn, sizeof(tn), "ai%d%c", kb, span ? 's' : 'u');
                    printf("%-11s %6d %6d %5d %-4s %-10s %9.2f %9.1f\n", sh.name, sh.M, sh.K, N, tn, run_u_name[v], us, wb/(us*1e3));
                }
            }
            cudaFree(dW); cudaFree(dX); cudaFree(dq); cudaFree(ds); cudaFree(dY);
            cudaEventDestroy(e0); cudaEventDestroy(e1);
        }
    }
}

int main(int argc, char ** argv) {
    const std::string mode = argc > 1 ? argv[1] : "all";
    const int reps = (argc > 2 && mode != "gguf") ? atoi(argv[2]) : 200;
    bool ok = true;
    if (mode == "check" || mode == "all") {
        std::mt19937 rng(1234);
        for (int N : {1, 3, 8}) ok &= check(256, 2048, N, rng);
        ok &= check(144, 5120, 1, rng);
        ok &= check(48, 1024, 7, rng);
        for (int N : {9, 64, 65, 200}) ok &= check(256, 2048, N, rng);
        ok &= check(144, 3072, 130, rng);
        ok &= check(400, 1024, 600, rng);
        printf("check: %s\n", ok ? "PASS" : "FAIL");
    }
    if (mode == "ucheck") {
        return check_u() ? 0 : 1;
    }
    if (mode == "ubench") {
        return check_u() ? (bench_u(reps), 0) : 1;
    }
    if (mode == "gguf") {
        if (argc < 3) { fprintf(stderr, "usage: %s gguf <file.gguf>\n", argv[0]); return 2; }
        return check_gguf(argv[2]) ? 0 : 1;
    }
    if (ok && (mode == "bench" || mode == "all")) {
        bench(reps);
    }
    return ok ? 0 : 1;
}
