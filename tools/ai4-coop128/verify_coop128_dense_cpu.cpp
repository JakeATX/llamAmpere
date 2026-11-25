// CPU-only check of the dense (ID = false) gemm_s8_coop128 index math (ai4-coop128.cuh, GGML_CUDA_AI4_COOP128_DENSE=1)
// against the dense gemm<128> reference sum order, on the PLAIN activation prep (row-major [N][K] int8 and
// [N][K >> (7 + XSH)] float scales, as quantize_act writes it and a PREFILL-GLUE producer caches it).
// Modelled per CTA / warp / lane: grid ((M/16 + 3)/4, (N + 127)/128), n0 = blockIdx.y*128, zero-filled loads past N and
// past the last 16-row slab, the 128 B/column XOR-swizzled activation tile, the 2-warp j split, the Xs index
// (kc >> XSH), the masked stores Y[(n0 + c)*ldy + r] and the 65,535-token pieces of ggml_cuda_mul_mat_ai4
// (X + t0*K, Y + t0*M). Checks: every output written exactly once, no read outside X / Xs, bitwise equal to the
// reference. Host model of the index math, not the CUDA kernel (that is test-backend-ops MUL_MAT ait4_u).
// Build: g++ -O2 -std=c++17 verify_coop128_dense_cpu.cpp -o verify_coop128_dense_cpu
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

struct shape { int M, N, K; };

int fails = 0;

// One dense launch of the modelled kernel over an N-token piece: Xq / Xs / Y already offset to the piece.
void coop_dense(const std::vector<int8_t> & W, const std::vector<float> & Wd, const int8_t * Xq, const float * Xs,
                size_t xq_avail, size_t xs_avail, float * Y, std::vector<int> & wr, size_t y_base,
                int M, int N, int K, int xsh) {
    const int nslab = M / 16;
    const int nk    = K / 128;
    const int xsn   = K >> (7 + xsh);
    const int gx    = (M / 16 + 3) / 4;
    const int gy    = (N + 127) / 128;
    for (int by = 0; by < gy; ++by) {
        const int n0 = by * 128;
        for (int bx = 0; bx < gx; ++bx) {
            const int slab0 = bx * 4;
            // per-thread accumulators [mw][nw][lane][nt][4]
            std::vector<float> acc((size_t) 4 * 2 * 32 * 8 * 4, 0.0f);
            for (int kc = 0; kc < nk; ++kc) {
                // stage: activation tile (swizzled) + scales, zero fill where n0 + c >= N
                std::vector<uint8_t> sx(128 * 128, 0);
                std::vector<float>   sxd(128, 0.0f);
                for (int i = 0; i < 128 * 8; ++i) {
                    const int c = i >> 3, p = i & 7;
                    const bool ok = (n0 + c) < N;
                    const int xp = p ^ (c & 7);
                    if (ok) {
                        const size_t off = (size_t) (n0 + c) * K + kc * 128 + p * 16;
                        if (off + 16 > xq_avail) { printf("FAIL Xq OOB\n"); ++fails; return; }
                        std::memcpy(&sx[c * 128 + xp * 16], Xq + off, 16);
                    }
                }
                for (int c = 0; c < 128; ++c) {
                    if ((n0 + c) < N) {
                        const size_t off = (size_t) (n0 + c) * xsn + (kc >> xsh);
                        if (off >= xs_avail) { printf("FAIL Xs OOB\n"); ++fails; return; }
                        sxd[c] = Xs[off];
                    }
                }
                for (int mw = 0; mw < 4; ++mw) {
                    const int slab = slab0 + mw;
                    for (int nw = 0; nw < 2; ++nw) {
                        for (int lane = 0; lane < 32; ++lane) {
                            const int g = lane >> 2, t = lane & 3;
                            for (int nt = 0; nt < 8; ++nt) {
                                // m16n8k32 output fragment: rows g / g + 8, columns 2t / 2t + 1 of this n8 tile
                                int p[4] = {0, 0, 0, 0};
                                for (int jp = 0; jp < 2; ++jp) {
                                    for (int jj = 0; jj < 2; ++jj) {
                                        const int j = jp * 2 + jj;   // j0 = 2jp then j1 = 2jp + 1, as the kernel
                                        for (int oi = 0; oi < 4; ++oi) {
                                            const int rr = (oi >> 1) ? g + 8 : g;
                                            const int cc = nw * 64 + nt * 8 + 2 * (lane & 3) + (oi & 1);
                                            const int r  = slab * 16 + rr;
                                            // B column cc is held by the lanes with g' = cc & 7; sum over its 4 t' quads
                                            for (int tq = 0; tq < 4; ++tq) {
                                                const int ob = tq * 32 + j * 8;
                                                const int xo = (((ob >> 4) ^ (cc & 7)) << 4) + (ob & 15);
                                                for (int b = 0; b < 8; ++b) {
                                                    const int w = slab < nslab ? W[(size_t) r * K + kc * 128 + ob + b] : 0;
                                                    p[oi] += w * (int) (int8_t) sx[cc * 128 + xo + b];
                                                }
                                            }
                                        }
                                    }
                                }
                                (void) t;
                                const float d0 = slab < nslab ? Wd[(size_t) (slab * 16 + g) * nk + kc] : 0.0f;
                                const float d1 = slab < nslab ? Wd[(size_t) (slab * 16 + g + 8) * nk + kc] : 0.0f;
                                const float dx0 = sxd[nw * 64 + nt * 8 + 2 * (lane & 3)];
                                const float dx1 = sxd[nw * 64 + nt * 8 + 2 * (lane & 3) + 1];
                                float * a = &acc[((((size_t) mw * 2 + nw) * 32 + lane) * 8 + nt) * 4];
                                a[0] += (float) p[0] * (d0 * dx0);
                                a[1] += (float) p[1] * (d0 * dx1);
                                a[2] += (float) p[2] * (d1 * dx0);
                                a[3] += (float) p[3] * (d1 * dx1);
                            }
                        }
                    }
                }
            }
            // masked stores
            for (int mw = 0; mw < 4; ++mw) {
                if (slab0 + mw >= nslab) continue;
                for (int nw = 0; nw < 2; ++nw) {
                    for (int lane = 0; lane < 32; ++lane) {
                        const int g = lane >> 2, t = lane & 3;
                        const int r = (slab0 + mw) * 16 + g;
                        for (int nt = 0; nt < 8; ++nt) {
                            const float * a = &acc[((((size_t) mw * 2 + nw) * 32 + lane) * 8 + nt) * 4];
                            const int c = nw * 64 + nt * 8 + 2 * t;
                            for (int h = 0; h < 2; ++h) {
                                if (n0 + c + h < N) {
                                    const size_t y0 = (size_t) (n0 + c + h) * M + r;
                                    Y[y0]     = a[h];
                                    Y[y0 + 8] = a[2 + h];
                                    wr[y_base + y0]++;
                                    wr[y_base + y0 + 8]++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void run(const shape & s, std::mt19937 & rng) {
    const int M = s.M, T = s.N, K = s.K;
    const int xsh = K % 256 == 0 ? 1 : 0;
    const int nk = K / 128, xsn = K >> (7 + xsh);
    std::vector<int8_t> W((size_t) M * K), Xq((size_t) T * K);
    std::vector<float>  Wd((size_t) M * nk), Xs((size_t) T * xsn);
    for (auto & v : W)  v = (int8_t) ((int) (rng() % 15) - 7);
    for (auto & v : Xq) v = (int8_t) ((int) (rng() % 255) - 127);
    for (auto & v : Wd) v = 0.001f * (1.0f + (float) (rng() % 97));
    for (auto & v : Xs) v = 0.003f * (1.0f + (float) (rng() % 89));

    // reference: dense gemm<128> order (int32 chunk sum -> * (d * dx) -> float acc over kc)
    std::vector<float> Yr((size_t) T * M);
    for (int c = 0; c < T; ++c) {
        for (int r = 0; r < M; ++r) {
            float acc = 0.0f;
            for (int kc = 0; kc < nk; ++kc) {
                int p = 0;
                for (int k = 0; k < 128; ++k) p += (int) W[(size_t) r * K + kc * 128 + k] * (int) Xq[(size_t) c * K + kc * 128 + k];
                acc += (float) p * (Wd[(size_t) r * nk + kc] * Xs[(size_t) c * xsn + (kc >> xsh)]);
            }
            Yr[(size_t) c * M + r] = acc;
        }
    }

    std::vector<float> Y((size_t) T * M, -12345.0f);
    std::vector<int>   wr((size_t) T * M, 0);
    // the 65,535-token pieces of ggml_cuda_mul_mat_ai4
    for (int t0 = 0; t0 < T; t0 += 65535) {
        const int nt = T - t0 < 65535 ? T - t0 : 65535;
        coop_dense(W, Wd, Xq.data() + (size_t) t0 * K, Xs.data() + (size_t) t0 * xsn,
                   Xq.size() - (size_t) t0 * K, Xs.size() - (size_t) t0 * xsn,
                   Y.data() + (size_t) t0 * M, wr, (size_t) t0 * M, M, nt, K, xsh);
    }
    int bad = 0, multi = 0;
    for (size_t i = 0; i < Y.size(); ++i) {
        if (wr[i] != 1) { ++multi; continue; }
        uint32_t a, b;
        std::memcpy(&a, &Y[i], 4);
        std::memcpy(&b, &Yr[i], 4);
        if (a != b) ++bad;
    }
    const bool ok = bad == 0 && multi == 0;
    printf("[%s] dense M=%d N=%d K=%d XSH=%d: %d bit mismatches, %d outputs not written exactly once\n",
           ok ? "PASS" : "FAIL", M, T, K, xsh, bad, multi);
    if (!ok) ++fails;
}

} // namespace

int main() {
    std::mt19937 rng(321);
    const shape shapes[] = {
        {48, 512, 1024},    // existing tbo case: 3 slabs (partial 64-row CTA), exact token tiles
        {80, 513, 384},     // H128, 1-token last tile (second warp idle)
        {2304, 600, 2304},  // Inkling width, partial last tile (88 tokens: warp 1 has 24)
        {1040, 1000, 1152}, // H128 K, 65 slabs (one-slab last CTA), 104-token last tile
        {16, 640, 256},     // one slab, five full tiles
        {16, 65535 + 700, 128}, // two 65,535-token pieces
    };
    for (const shape & s : shapes) run(s, rng);
    if (fails) { printf("FAIL: %d shapes\n", fails); return 1; }
    printf("[PASS] coop128 dense: %zu shapes bitwise equal to the gemm<128> order, every output written once\n",
           sizeof(shapes) / sizeof(shapes[0]));
    return 0;
}
