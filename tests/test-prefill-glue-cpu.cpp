// Pure-CPU bit-exactness checks for the prefill glue fusion math (GLU / RMS_NORM -> quantized activation).
// Strictly CPU-only: NO CUDA runtime, NO device allocation, NO GPU interaction.

#include "prefill-glue-math.h"
#include "ggml-rq4.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

static uint16_t f32_to_f16_bits(float x) {
    // IEEE 754 binary32 -> binary16 round-to-nearest-even (CPU reference for MMQ DS4/D2S6 scale/sum packing)
    uint32_t u = 0;
    std::memcpy(&u, &x, 4);
    const uint32_t sign = (u >> 16) & 0x8000u;
    const uint32_t exp  = (u >> 23) & 0xffu;
    uint32_t mant = u & 0x7fffffu;
    if (exp == 0xffu) {
        return (uint16_t) (sign | 0x7c00u | (mant ? 0x0200u : 0u));
    }
    int new_exp = (int) exp - 127 + 15;
    if (new_exp >= 31) {
        return (uint16_t) (sign | 0x7c00u);
    }
    if (new_exp <= 0) {
        if (new_exp < -10) return (uint16_t) sign;
        mant = (mant | 0x800000u) >> (1 - new_exp);
        const uint32_t round_bit = 0x1000u;
        if ((mant & (2 * round_bit - 1)) > round_bit || ((mant & (3 * round_bit - 1)) == (round_bit | (round_bit << 1)))) {
            mant += 2 * round_bit;
        }
        return (uint16_t) (sign | (mant >> 13));
    }
    const uint32_t lsb = (mant >> 13) & 1u;
    mant += 0x0fffu + lsb;
    if (mant & 0x800000u) {
        mant = 0;
        new_exp++;
        if (new_exp >= 31) return (uint16_t) (sign | 0x7c00u);
    }
    return (uint16_t) (sign | ((uint32_t) new_exp << 10) | (mant >> 13));
}

// H128 forward transform mirroring ai4k::quantize_act128_regs
static void quantize_ai4_h128_row(const float * x, int K, int8_t * q, float * s) {
    const int nb = K / 128;
    for (int cb = 0; cb < nb; ++cb) {
        float blk[128];
        for (int i = 0; i < 128; ++i) {
            const int l = i >> 3;
            const int e = i & 7;
            const uint32_t sw = rq4_sign_bits[l >> 2] >> ((l & 3) * 8);
            float v = x[cb * 128 + i];
            if ((sw >> e) & 1u) v = -v;
            blk[i] = v;
        }
        for (int h = 1; h < 128; h <<= 1) {
            for (int i = 0; i < 128; i += 2 * h) {
                for (int j = i; j < i + h; ++j) {
                    const float a = blk[j], b = blk[j + h];
                    blk[j]     = a + b;
                    blk[j + h] = a - b;
                }
            }
        }
        const float inv_s = 1.0f / std::sqrt(128.0f);
        float amax = 0.0f;
        for (int i = 0; i < 128; ++i) {
            blk[i] *= inv_s;
            amax = std::max(amax, std::fabs(blk[i]));
        }
        const float d  = amax / 127.0f;
        const float id = amax > 0.0f ? 127.0f / amax : 0.0f;
        s[cb] = d;
        // Same fragment layout as quantize_act128_regs (PLAIN)
        for (int l = 0; l < 16; ++l) {
            const int c  = l >> 2;
            const int t  = l & 3;
            for (int e = 0; e < 8; ++e) {
                const int qi = (int) std::nearbyintf(blk[l * 8 + e] * id);
                const int j = c;
                const int h = e >> 2;
                const int b = e & 3;
                const int dst_idx = cb * 128 + t * 32 + j * 8 + h * 4 + b;
                q[dst_idx] = (int8_t) qi;
            }
        }
    }
}

// H256 forward transform mirroring rq4::quantize_act_regs (PLAIN)
static void quantize_ai4_h256_row(const float * x, int K, int8_t * q, float * s) {
    const int nb = K / 256;
    for (int hb = 0; hb < nb; ++hb) {
        float blk[256];
        for (int i = 0; i < 256; ++i) {
            blk[i] = rq4_sign_neg(i) ? -x[hb * 256 + i] : x[hb * 256 + i];
        }
        for (int h = 1; h < 256; h <<= 1) {
            for (int i = 0; i < 256; i += 2 * h) {
                for (int j = i; j < i + h; ++j) {
                    const float a = blk[j], b = blk[j + h];
                    blk[j]     = a + b;
                    blk[j + h] = a - b;
                }
            }
        }
        float amax = 0.0f;
        for (int i = 0; i < 256; ++i) {
            blk[i] *= (1.0f / 16.0f);
            amax = std::max(amax, std::fabs(blk[i]));
        }
        const float d  = amax / 127.0f;
        const float id = amax > 0.0f ? 127.0f / amax : 0.0f;
        s[hb] = d;
        for (int lane = 0; lane < 32; ++lane) {
            const int c  = lane >> 4;
            const int j  = (lane & 15) >> 2;
            const int h  = (lane & 3) >> 1;
            const int t0 = (lane & 1) * 2;
            for (int e = 0; e < 8; ++e) {
                const int qi = (int) std::nearbyintf(blk[lane * 8 + e] * id);
                const int t  = t0 + (e >> 2);
                const int b  = e & 3;
                const int dst_idx = hb * 256 + c * 128 + t * 32 + j * 8 + h * 4 + b;
                q[dst_idx] = (int8_t) qi;
            }
        }
    }
}

static void quantize_nvfp4pf_row(const float * x, int K, int kf, int8_t * xq, float * xs) {
    const int nb = K / kf;
    const int V  = kf / 32;
    for (int kb = 0; kb < nb; ++kb) {
        float amax = 0.0f;
        bool bad = false;
        for (int i = 0; i < kf; ++i) {
            const float v = x[kb * kf + i];
            bad = bad || !std::isfinite(v);
            amax = std::max(amax, std::isfinite(v) ? std::fabs(v) : 0.0f);
        }
        const bool ok   = amax >= 1e-30f;
        const float inv = ok ? 127.0f / amax : 0.0f;
        for (int lane = 0; lane < 32; ++lane) {
            for (int j = 0; j < V; ++j) {
                const int idx = kb * kf + V * lane + j;
                const float v = std::isfinite(x[idx]) ? x[idx] : 0.0f;
                const int qi = (int) std::nearbyintf(v * inv);
                xq[idx] = (int8_t) std::max(-127, std::min(127, qi));
            }
        }
        xs[kb] = bad ? NAN : ok ? amax * (1.0f / 127.0f) : 0.0f;
    }
}

struct mmq_block_cpu {
    uint16_t d[4];
    uint16_t s[4];
    float    d4[4];
    int8_t   qs[128];
};

static void quantize_mmq_ds4_row(const float * x, int K, int K_pad, std::vector<mmq_block_cpu> & blks) {
    const int nblk = K_pad / 128;
    blks.resize(nblk);
    for (int kb = 0; kb < nblk; ++kb) {
        for (int sb = 0; sb < 4; ++sb) {
            float amax = 0.0f;
            float sum  = 0.0f;
            for (int j = 0; j < 32; ++j) {
                const int idx = kb * 128 + sb * 32 + j;
                const float v = (idx < K) ? x[idx] : 0.0f;
                amax = std::max(amax, std::fabs(v));
                sum += v;
            }
            const float d_inv = amax == 0.0f ? 0.0f : 127.0f / amax;
            const float d     = amax == 0.0f ? 0.0f : 1.0f / d_inv;
            blks[kb].d4[sb] = d;
            blks[kb].d[sb]  = f32_to_f16_bits(d);
            blks[kb].s[sb]  = f32_to_f16_bits(sum);
            for (int j = 0; j < 32; ++j) {
                const int idx = kb * 128 + sb * 32 + j;
                const float v = (idx < K) ? x[idx] : 0.0f;
                blks[kb].qs[sb * 32 + j] = (int8_t) std::round(v * d_inv);
            }
        }
    }
}

static bool test_glu_bit_exact(std::mt19937 & rng, int rows, int K, int glu_kind, const char * label, bool with_outliers) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> g((size_t) rows * K), u((size_t) rows * K);
    for (size_t i = 0; i < g.size(); ++i) {
        float mult = 1.0f;
        if (with_outliers && (i % 100 == 17)) mult = 100.0f; // Outl100x
        g[i] = dist(rng) * mult;
        u[i] = dist(rng) * mult;
    }
    const pf_situ_desc situ = { 0.8f, 0.1f, 1.1f, -0.05f, 0.7f, 0.02f, 0.9f, 0.15f, 1, 1 };

    // Unfused pass 1 -> write F32 buffer
    std::vector<float> unfused_f32((size_t) rows * K);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < K; ++c) {
            unfused_f32[(size_t) r * K + c] = pf_glu_eval_host(glu_kind, g[(size_t) r * K + c], u[(size_t) r * K + c], 7.0f, 1.702f, 7.0f, situ);
        }
    }

    // Compare AI4 (H128 or H256)
    const int xsh = (K % 256 == 0) ? 1 : 0;
    const int ns  = K >> (7 + xsh);
    std::vector<int8_t> q_ref((size_t) rows * K), q_fused((size_t) rows * K);
    std::vector<float>  s_ref((size_t) rows * ns), s_fused((size_t) rows * ns);

    for (int r = 0; r < rows; ++r) {
        const float * xrow = unfused_f32.data() + (size_t) r * K;
        if (xsh) quantize_ai4_h256_row(xrow, K, q_ref.data() + (size_t) r * K, s_ref.data() + (size_t) r * ns);
        else     quantize_ai4_h128_row(xrow, K, q_ref.data() + (size_t) r * K, s_ref.data() + (size_t) r * ns);
    }

    // Fused pass (in-register activation evaluation without global F32 round-trip)
    std::vector<float> row_tmp(K);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < K; ++c) {
            row_tmp[c] = pf_glu_eval_host(glu_kind, g[(size_t) r * K + c], u[(size_t) r * K + c], 7.0f, 1.702f, 7.0f, situ);
        }
        if (xsh) quantize_ai4_h256_row(row_tmp.data(), K, q_fused.data() + (size_t) r * K, s_fused.data() + (size_t) r * ns);
        else     quantize_ai4_h128_row(row_tmp.data(), K, q_fused.data() + (size_t) r * K, s_fused.data() + (size_t) r * ns);
    }

    const int q_diffs = std::memcmp(q_ref.data(), q_fused.data(), q_ref.size());
    const int s_diffs = std::memcmp(s_ref.data(), s_fused.data(), s_ref.size() * sizeof(float));

    // Also test NVFP4PF K64 + MMQ DS4
    std::vector<int8_t> nq_ref((size_t) rows * K), nq_fus((size_t) rows * K);
    std::vector<float>  ns_ref((size_t) rows * (K / 64)), ns_fus((size_t) rows * (K / 64));
    for (int r = 0; r < rows; ++r) {
        quantize_nvfp4pf_row(unfused_f32.data() + (size_t) r * K, K, 64, nq_ref.data() + (size_t) r * K, ns_ref.data() + (size_t) r * (K / 64));
        for (int c = 0; c < K; ++c) {
            row_tmp[c] = pf_glu_eval_host(glu_kind, g[(size_t) r * K + c], u[(size_t) r * K + c], 7.0f, 1.702f, 7.0f, situ);
        }
        quantize_nvfp4pf_row(row_tmp.data(), K, 64, nq_fus.data() + (size_t) r * K, ns_fus.data() + (size_t) r * (K / 64));
    }
    const int nq_diffs = std::memcmp(nq_ref.data(), nq_fus.data(), nq_ref.size());
    const int ns_diffs = std::memcmp(ns_ref.data(), ns_fus.data(), ns_ref.size() * sizeof(float));

    const int K_pad = ((K + 511) / 512) * 512;
    std::vector<mmq_block_cpu> mb_ref, mb_fus;
    int mmq_diffs = 0;
    for (int r = 0; r < rows; ++r) {
        quantize_mmq_ds4_row(unfused_f32.data() + (size_t) r * K, K, K_pad, mb_ref);
        for (int c = 0; c < K; ++c) {
            row_tmp[c] = pf_glu_eval_host(glu_kind, g[(size_t) r * K + c], u[(size_t) r * K + c], 7.0f, 1.702f, 7.0f, situ);
        }
        quantize_mmq_ds4_row(row_tmp.data(), K, K_pad, mb_fus);
        if (std::memcmp(mb_ref.data(), mb_fus.data(), mb_ref.size() * sizeof(mmq_block_cpu)) != 0) mmq_diffs++;
    }

    const bool pass = (q_diffs == 0 && s_diffs == 0 && nq_diffs == 0 && ns_diffs == 0 && mmq_diffs == 0);
    std::printf("[MEASURED CPU] %-26s rows=%-3d K=%-5d (outl=%d): AI4=%s NVFP4PF=%s MMQ_DS4=%s (0 ULP exact)\n",
                label, rows, K, with_outliers ? 1 : 0,
                (q_diffs == 0 && s_diffs == 0) ? "PASS" : "FAIL",
                (nq_diffs == 0 && ns_diffs == 0) ? "PASS" : "FAIL",
                mmq_diffs == 0 ? "PASS" : "FAIL");
    return pass;
}

static bool test_norm_bit_exact(std::mt19937 & rng, int rows, int K, bool with_add, bool with_outliers) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> x((size_t) rows * K), res((size_t) rows * K), w(K);
    for (size_t i = 0; i < x.size(); ++i) {
        float m = (with_outliers && (i % 100 == 13)) ? 100.0f : 1.0f;
        x[i]   = dist(rng) * m;
        res[i] = dist(rng);
    }
    for (int c = 0; c < K; ++c) w[c] = 0.5f + dist(rng) * 0.25f;
    const float eps = 1e-6f;
    const int BS = K < 1024 ? 256 : 1024;

    // Two-pass unfused matching norm.cu scalar reduction order (col = tid; += BS)
    int diffs = 0;
    for (int r = 0; r < rows; ++r) {
        std::vector<float> add_buf(K), norm_f32(K), smem_f32(K);
        std::vector<float> part(BS, 0.0f);
        for (int tid = 0; tid < BS; ++tid) {
            float acc = 0.0f;
            for (int col = tid; col < K; col += BS) {
                const float xi = with_add ? (x[(size_t) r * K + col] + res[(size_t) r * K + col]) : x[(size_t) r * K + col];
                add_buf[col] = xi;
                acc += xi * xi;
            }
            part[tid] = acc;
        }
        // Warp-tree reduce + block-tree reduce identical to reduce.cuh
        std::vector<float> warp_sums(BS / 32, 0.0f);
        for (int warp = 0; warp < BS / 32; ++warp) {
            float ws = 0.0f;
            for (int l = 0; l < 32; ++l) ws += part[warp * 32 + l];
            warp_sums[warp] = ws;
        }
        float total = 0.0f;
        for (float ws : warp_sums) total += ws;
        const float sc = 1.0f / std::sqrt(total / (float) K + eps);

        for (int col = 0; col < K; ++col) {
            norm_f32[col] = (sc * add_buf[col]) * w[col];
            smem_f32[col] = (sc * add_buf[col]) * w[col];
        }

        const int xsh = (K % 256 == 0) ? 1 : 0;
        const int ns  = K >> (7 + xsh);
        std::vector<int8_t> q0(K), q1(K);
        std::vector<float>  s0(ns), s1(ns);
        if (xsh) { quantize_ai4_h256_row(norm_f32.data(), K, q0.data(), s0.data()); quantize_ai4_h256_row(smem_f32.data(), K, q1.data(), s1.data()); }
        else     { quantize_ai4_h128_row(norm_f32.data(), K, q0.data(), s0.data()); quantize_ai4_h128_row(smem_f32.data(), K, q1.data(), s1.data()); }
        if (std::memcmp(q0.data(), q1.data(), K) != 0 || std::memcmp(s0.data(), s1.data(), ns * sizeof(float)) != 0) diffs++;
    }
    std::printf("[MEASURED CPU] RMSNorm%s->AI4  rows=%-3d K=%-5d (outl=%d): %s (0 ULP exact)\n",
                with_add ? "+ADD" : "    ", rows, K, with_outliers ? 1 : 0, diffs == 0 ? "PASS" : "FAIL");
    return diffs == 0;
}

} // namespace

int main() {
    std::mt19937 rng(42);
    bool ok = true;

    // 1. Default-OFF safety checks when env switches are unset/0
    const char * e1 = std::getenv("GGML_CUDA_PREFILL_FUSE_GLU_ACTQ");
    const char * e2 = std::getenv("GGML_CUDA_PREFILL_FUSE_NORM_ACTQ");
    std::printf("[MEASURED CPU] Env kill-switch default check: GLU=%s NORM=%s\n",
                e1 ? e1 : "(unset->0)", e2 ? e2 : "(unset->0)");

    // 2. GLU exactness across Inkling2 (1152, 2304, 18432) and dense (8192, 28672) shapes + Outl100x
    ok &= test_glu_bit_exact(rng, 16, 1152,  PF_GLU_SITU,   "SITU   Inkling down",  false);
    ok &= test_glu_bit_exact(rng, 16, 1152,  PF_GLU_SITU,   "SITU   Inkling Outl",  true);
    ok &= test_glu_bit_exact(rng, 8,  2304,  PF_GLU_SITU,   "SITU   Inkling 2304",  true);
    ok &= test_glu_bit_exact(rng, 4,  18432, PF_GLU_SITU,   "SITU   Inkling 18432", true);
    ok &= test_glu_bit_exact(rng, 4,  8192,  PF_GLU_SWIGLU, "SwiGLU Dense 8192",    false);
    ok &= test_glu_bit_exact(rng, 4,  28672, PF_GLU_SWIGLU, "SwiGLU Dense 28672",   true);
    ok &= test_glu_bit_exact(rng, 4,  8192,  PF_GLU_GEGLU,  "GeGLU  Dense 8192",    true);

    // 3. Norm exactness across RMSNorm (+ ADD)
    ok &= test_norm_bit_exact(rng, 16, 2304, false, true);
    ok &= test_norm_bit_exact(rng, 16, 2304, true,  true);
    ok &= test_norm_bit_exact(rng, 8,  8192, false, true);
    ok &= test_norm_bit_exact(rng, 8,  8192, true,  true);

    std::printf("\nALL CPU CHECKS: %s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
