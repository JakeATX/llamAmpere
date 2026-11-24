// CPU-only check of the gemm_s8_coop128 (ai4-coop128.cuh) work split against the gemm<128> reference sum order:
// the 2-warp j-split (each warp of a slab decodes 2 of 4 j-steps), the 128 B/column XOR-swizzled activation tile and
// the per-warp column skip, over skewed expert route counts. Host model of the index math, not the CUDA kernel;
// the kernel itself is covered by test-backend-ops MUL_MAT_ID ait4_u with GGML_CUDA_AI4_COOP128=1.
// Build: g++ -O2 -std=c++17 verify_coop128_cpu.cpp -o verify_coop128_cpu
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main() {
    std::mt19937 rng(123);
    {
        // Test skewed expert token counts including N=1, 7, 32, 63, 64, 65, 95, 128 (exercises the nw=1 column skip)
        const int test_Ns[] = {1, 7, 32, 63, 64, 65, 95, 128};
        const int K_t = 256, M_t = 64;
        std::vector<int8_t> W_s8(M_t * K_t);
        std::vector<float> W_d(M_t * (K_t / 128));
        for (auto & v : W_s8) v = (int8_t)((int)(rng() % 255) - 127);
        for (auto & v : W_d) v = 0.01f * (1.0f + float(rng() % 10));

        for (int N_exp : test_Ns) {
            std::vector<int8_t> Xq_t(N_exp * K_t);
            std::vector<float> Xs_t(N_exp * (K_t / 256));
            for (auto & v : Xq_t) v = (int8_t)((int)(rng() % 255) - 127);
            for (auto & v : Xs_t) v = 0.02f * (1.0f + float(rng() % 10));

            // Reference: gemm<128> (int32 sum over K=128 -> magic_i2f -> * (d * dx) -> float acc)
            std::vector<float> Y_rc(N_exp * M_t, 0.0f);
            for (int c = 0; c < N_exp; ++c) {
                for (int r = 0; r < M_t; ++r) {
                    float acc = 0.0f;
                    for (int kc = 0; kc < K_t / 128; ++kc) {
                        int p = 0;
                        for (int k = 0; k < 128; ++k) p += int(W_s8[r * K_t + kc * 128 + k]) * int(Xq_t[c * K_t + kc * 128 + k]);
                        acc += float(p) * (W_d[r * (K_t / 128) + kc] * Xs_t[c * (K_t / 256) + (kc >> 1)]);
                    }
                    Y_rc[c * M_t + r] = acc;
                }
            }

            // Cooperative 2-warp simulation (nw=0 handles cols 0..63, nw=1 handles cols 64..127 with tail_skip)
            std::vector<float> Y_coop(N_exp * M_t, 0.0f);
            for (int mw = 0; mw < 4; ++mw) {
                for (int nw = 0; nw < 2; ++nw) {
                    int n_warp_cols = N_exp - nw * 64;
                    if (n_warp_cols <= 0) continue; // mirror of (!ID || n_warp_cols > 0)
                    for (int nt = 0; nt < 8; ++nt) {
                        if (nt * 8 >= n_warp_cols) continue; // mirror of the per-warp column skip
                        for (int c_in = 0; c_in < 8; ++c_in) {
                            int c = nw * 64 + nt * 8 + c_in;
                            if (c >= N_exp) continue;
                            for (int r_in = 0; r_in < 16; ++r_in) {
                                int r = mw * 16 + r_in;
                                float acc = 0.0f;
                                for (int kc = 0; kc < K_t / 128; ++kc) {
                                    // Verify XOR-swizzle round-trip for every 8B piece across jp=0..1, j=0..3
                                    uint8_t sx_row[128];
                                    for (int p16 = 0; p16 < 8; ++p16) {
                                        int xp = p16 ^ (c & 7);
                                        std::memcpy(&sx_row[xp * 16], &Xq_t[c * K_t + kc * 128 + p16 * 16], 16);
                                    }
                                    int p = 0;
                                    for (int jp = 0; jp < 2; ++jp) {
                                        for (int jj = 0; jj < 2; ++jj) {
                                            int j = jp * 2 + jj;
                                            for (int t = 0; t < 4; ++t) {
                                                int ob = t * 32 + j * 8;
                                                int xo = (((ob >> 4) ^ (c & 7)) << 4) + (ob & 15);
                                                for (int b = 0; b < 8; ++b) {
                                                    int8_t xv = (int8_t)sx_row[xo + b];
                                                    p += int(W_s8[r * K_t + kc * 128 + ob + b]) * int(xv);
                                                }
                                            }
                                        }
                                    }
                                    acc += float(p) * (W_d[r * (K_t / 128) + kc] * Xs_t[c * (K_t / 256) + (kc >> 1)]);
                                }
                                Y_coop[c * M_t + r] = acc;
                            }
                        }
                    }
                }
            }

            // must be bit-exact
            for (size_t i = 0; i < Y_rc.size(); ++i) {
                uint32_t u0, u1;
                std::memcpy(&u0, &Y_rc[i], 4);
                std::memcpy(&u1, &Y_coop[i], 4);
                if (u0 != u1) {
                    printf("FAIL coop128 bit mismatch at N=%d idx=%zu: rc=%g coop=%g\n", N_exp, i, Y_rc[i], Y_coop[i]);
                    return 1;
                }
            }
        }
        printf("[PASS] coop128: gemm_s8_coop128 2-warp j-split + XOR swizzle + column skip:\n"
               "       8/8 skewed token counts (N = 1, 7, 32, 63, 64, 65, 95, 128) match the gemm<128> integer sums (index model only)\n");
    }
    return 0;
}
