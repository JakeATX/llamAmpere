// CPU-only mathematical & reference verification for inkattn-fast and fattn-sjkvarn-fast (Inkling-2-Mini 100K SpecDecode).
// Runs strictly on CPU (no CUDA runtime or GPU device access).
// Build & run: g++ -O3 -std=c++17 tests/test-ink200-attn-cpu.cpp -o /tmp/test-ink200-attn-cpu && /tmp/test-ink200-attn-cpu

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

static uint16_t f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFF) == 0xFF) {
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x0200u : 0u));
    }
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        int shift = 14 - exp;
        uint32_t rb = 1u << (shift - 1);
        uint32_t rem = mant & ((1u << shift) - 1u);
        uint32_t res = mant >> shift;
        if (rem > rb || (rem == rb && (res & 1u))) res++;
        return (uint16_t)(sign | res);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    uint32_t rem = mant & 0x1FFFu;
    uint32_t res = (exp << 10) | (mant >> 13);
    if (rem > 0x1000u || (rem == 0x1000u && (res & 1u))) res++;
    return (uint16_t)(sign | res);
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)(h & 0x8000u)) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t x;
    if (exp == 0) {
        if (mant == 0) {
            x = sign;
        } else {
            exp = 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FFu;
            x = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        x = sign | 0x7F800000u | (mant << 13);
    } else {
        x = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &x, 4);
    return f;
}

static float h_round(float f) { return f16_to_f32(f32_to_f16(f)); }

// Load real blk.0.attn_rel_bias.weight [16, 1024] from GGUF if available (read-only).
static bool load_gguf_rel_bias(const char * path, std::vector<float> & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[4];
    f.read(magic, 4);
    if (memcmp(magic, "GGUF", 4) != 0) return false;
    uint32_t ver = 0; uint64_t nt = 0, nkv = 0;
    f.read((char *)&ver, 4); f.read((char *)&nt, 8); f.read((char *)&nkv, 8);
    auto rd_str = [&]() -> std::string {
        uint64_t n = 0; f.read((char *)&n, 8);
        std::string s(n, '\0'); f.read(&s[0], n);
        return s;
    };
    auto skip_val = [&](uint32_t t, auto & self) -> void {
        if (t == 0 || t == 1 || t == 7) f.seekg(1, std::ios::cur);
        else if (t == 2 || t == 3) f.seekg(2, std::ios::cur);
        else if (t == 4 || t == 5 || t == 6) f.seekg(4, std::ios::cur);
        else if (t == 10 || t == 11 || t == 12) f.seekg(8, std::ios::cur);
        else if (t == 8) (void)rd_str();
        else if (t == 9) {
            uint32_t et = 0; uint64_t cnt = 0;
            f.read((char *)&et, 4); f.read((char *)&cnt, 8);
            for (uint64_t i = 0; i < cnt; ++i) self(et, self);
        }
    };
    for (uint64_t i = 0; i < nkv; ++i) {
        (void)rd_str();
        uint32_t vt = 0; f.read((char *)&vt, 4);
        skip_val(vt, skip_val);
    }
    uint64_t found_offs = 0, found_n = 0;
    bool found = false;
    for (uint64_t i = 0; i < nt; ++i) {
        std::string name = rd_str();
        uint32_t nd = 0; f.read((char *)&nd, 4);
        uint64_t n = 1;
        for (uint32_t d = 0; d < nd; ++d) { uint64_t v = 0; f.read((char *)&v, 8); n *= v; }
        uint32_t ttype = 0; uint64_t offs = 0;
        f.read((char *)&ttype, 4); f.read((char *)&offs, 8);
        if (name == "blk.0.attn_rel_bias.weight" && ttype == 0) {
            found = true; found_offs = offs; found_n = n;
        }
    }
    if (!found) return false;
    uint64_t pos = (uint64_t)f.tellg();
    uint64_t data_start = (pos + 31u) & ~31ull;
    f.seekg(data_start + found_offs, std::ios::beg);
    out.resize(found_n);
    f.read((char *)out.data(), found_n * sizeof(float));
    return (bool)f;
}

// Fast 256-point orthogonal Walsh-Hadamard Transform (exact match to SJ-KVaRN WHT-256)
static void wht256(float * x) {
    for (int h = 1; h < 256; h <<= 1) {
        for (int i = 0; i < 256; i += (h << 1)) {
            for (int j = i; j < i + h; ++j) {
                float u = x[j], v = x[j + h];
                x[j]     = u + v;
                x[j + h] = u - v;
            }
        }
    }
    const float inv = 1.0f / 16.0f;
    for (int i = 0; i < 256; ++i) x[i] *= inv;
}

static bool test_q8_magic_exhaustive() {
    int mismatches = 0;
    for (uint32_t dh = 0; dh < 65536; ++dh) {
        if (((dh >> 10) & 0x1Fu) == 31u) continue;
        float df = f16_to_f32((uint16_t)dh);
        for (int q = -128; q <= 127; ++q) {
            uint16_t ref = f32_to_f16(df * (float)q);
            uint8_t u = (uint8_t)((uint8_t)q ^ 0x80u);
            uint16_t magic_h = (uint16_t)(0x6400u | u);
            uint16_t q_h = f32_to_f16(f16_to_f32(magic_h) - 1152.0f);
            uint16_t fast = f32_to_f16(f16_to_f32((uint16_t)dh) * f16_to_f32(q_h));
            if (ref != fast) mismatches++;
        }
    }
    printf("[PASS=%d] Test 1: Exhaustive Q8_0 FP16-magic dequant (16,252,928 pairs): mismatches=%d\n",
           mismatches == 0, mismatches);
    return mismatches == 0;
}

static bool test_inkattn_swa_skip_and_cyclic(const std::vector<float> & real_rel_bias) {
    // Inkling-2-Mini SWA geometry: D=128, H=24, Hkv=8 (g=3), M=5, n_kv=3072 ring buffer with 1024 live cells wrapped!
    const int D = 128, g = 3, T = 5, R = g * T, n_kv = 3072, NT = n_kv / 16, W = 48;
    const int E = 1024, n_rel = 16;
    const float scale = 1.0f / sqrtf((float)D);
    std::mt19937 rng(2026);
    std::normal_distribution<float> nd(0.0f, 0.7f);

    // Build rel bias table [g, E+1, T] from real GGUF blk.0.attn_rel_bias if loaded
    std::vector<float> tab(g * (E + 1) * T, 0.0f);
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < g; ++h) {
            float rvec[16];
            for (int c = 0; c < n_rel; ++c) rvec[c] = nd(rng) * 0.5f;
            float sink_logit = nd(rng);
            for (int e = 0; e <= E; ++e) {
                float acc = 0.0f;
                if (e < E) {
                    for (int c = 0; c < n_rel; ++c) {
                        float bv = !real_rel_bias.empty() ? real_rel_bias[e * n_rel + c] : nd(rng) * 0.3f;
                        acc += bv * rvec[c];
                    }
                }
                tab[(t * (E + 1) + e) * g + h] = acc - sink_logit;
            }
        }
    }

    // Q [R, D], K/V [n_kv, D] (quantized to Q8_0!), mask [T, n_kv] (wrapped ring window at offset 2500 -> wraps past 3072!)
    std::vector<float> Q(R * D), K(n_kv * D), V(n_kv * D), gate(R * D), sinks(g, 0.0f);
    for (float & x : Q) x = h_round(nd(rng) * scale);
    for (float & x : gate) x = nd(rng);
    // Quantize K/V to Q8_0 blocks of 32 and dequant back with exact Q8_0 f16
    for (int k = 0; k < n_kv; ++k) {
        for (int b = 0; b < D / 32; ++b) {
            float kv[32], vv[32], amaxk = 0, amaxv = 0;
            for (int i = 0; i < 32; ++i) {
                kv[i] = nd(rng); vv[i] = nd(rng);
                amaxk = std::max(amaxk, fabsf(kv[i])); amaxv = std::max(amaxv, fabsf(vv[i]));
            }
            float dk = h_round(amaxk / 127.0f), dv = h_round(amaxv / 127.0f);
            for (int i = 0; i < 32; ++i) {
                int qk = std::clamp((int)lrintf(dk > 0 ? kv[i] / dk : 0.0f), -127, 127);
                int qv = std::clamp((int)lrintf(dv > 0 ? vv[i] / dv : 0.0f), -127, 127);
                K[k * D + b * 32 + i] = h_round(dk * (float)qk);
                V[k * D + b * 32 + i] = h_round(dv * (float)qv);
            }
        }
    }

    std::vector<float> mask(T * n_kv, -INFINITY);
    std::vector<int32_t> rp(T * n_kv, E);
    const int ring_start = 2500; // wraps around 3072 -> tests split contiguous window across ring wrap!
    for (int t = 0; t < T; ++t) {
        for (int dist = 0; dist < 1024; ++dist) {
            int cell = (ring_start + t + (1023 - dist)) % n_kv;
            mask[t * n_kv + cell] = 0.0f;
            rp[t * n_kv + cell] = dist;
        }
    }

    // Reference: full dense scan (all 192 tiles, no skip)
    std::vector<float> O_ref(R * D, 0.0f), lse_ref(R, 0.0f);
    for (int r = 0; r < R; ++r) {
        int tq = r / g, h = r % g;
        float mx = -INFINITY;
        std::vector<float> sc(n_kv, -INFINITY);
        for (int k = 0; k < n_kv; ++k) {
            float m = mask[tq * n_kv + k];
            if (m == -INFINITY) continue;
            float bias = tab[(tq * (E + 1) + rp[tq * n_kv + k]) * g + h];
            float mb = h_round(m + bias);
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) dot += Q[r * D + d] * K[k * D + d];
            sc[k] = dot + mb;
            mx = std::max(mx, sc[k]);
        }
        float sm = 0.0f;
        for (int k = 0; k < n_kv; ++k) {
            if (sc[k] == -INFINITY) continue;
            float p = expf(sc[k] - mx);
            sm += p;
            for (int d = 0; d < D; ++d) O_ref[r * D + d] += p * V[k * D + d];
        }
        // Zero sink merge
        float m2 = std::max(mx, sinks[h]);
        float f = expf(mx - m2);
        sm = sm * f + expf(sinks[h] - m2);
        for (int d = 0; d < D; ++d) {
            float gval = 1.0f / (1.0f + expf(-gate[r * D + d]));
            O_ref[r * D + d] = (O_ref[r * D + d] * f / sm) * gval;
        }
        lse_ref[r] = m2 + logf(sm);
    }

    // Fast path simulation: cyclic striding across W=48 warps, dead-tile skip, guarded O *= alpha, skip empty Op write!
    std::vector<float> Op(W * R * D, -99999.0f); // poison sentinel to verify dead warps are never read!
    std::vector<float> Mp_x(W * R, -INFINITY), Mp_y(W * R, 0.0f);
    int tiles_visited = 0, tiles_skipped = 0, max_live_per_warp = 0, min_live_per_warp = 999999;

    for (int wi = 0; wi < W; ++wi) {
        std::vector<float> O(R * D, 0.0f);
        std::vector<float> mx(R, -INFINITY), sm(R, 0.0f);
        int live_w = 0;
        for (int tile = wi; tile < NT; tile += W) {
            tiles_visited++;
            int k0 = 16 * tile;
            bool any_live = false;
            for (int r = 0; r < R && !any_live; ++r) {
                int tq = r / g;
                for (int kk = 0; kk < 16; ++kk) {
                    if (mask[tq * n_kv + k0 + kk] != -INFINITY) { any_live = true; break; }
                }
            }
            if (!any_live) { tiles_skipped++; continue; }
            live_w++;
            for (int r = 0; r < R; ++r) {
                int tq = r / g, h = r % g;
                float s16[16], tm = -INFINITY;
                for (int kk = 0; kk < 16; ++kk) {
                    int key = k0 + kk;
                    float m = mask[tq * n_kv + key];
                    if (m == -INFINITY) { s16[kk] = -INFINITY; continue; }
                    float mb = h_round(m + tab[(tq * (E + 1) + rp[tq * n_kv + key]) * g + h]);
                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d) dot += Q[r * D + d] * K[key * D + d];
                    s16[kk] = dot + mb;
                    tm = std::max(tm, s16[kk]);
                }
                float mn = std::max(mx[r], tm);
                float mu = (mn == -INFINITY) ? 0.0f : mn;
                if (mn != mx[r]) {
                    float a = expf(mx[r] - mu);
                    mx[r] = mn;
                    sm[r] *= a;
                    for (int d = 0; d < D; ++d) O[r * D + d] *= a;
                }
                for (int kk = 0; kk < 16; ++kk) {
                    float p = expf(s16[kk] - mu);
                    sm[r] += p;
                    float ph = h_round(p);
                    for (int d = 0; d < D; ++d) O[r * D + d] += ph * V[(k0 + kk) * D + d];
                }
            }
        }
        max_live_per_warp = std::max(max_live_per_warp, live_w);
        min_live_per_warp = std::min(min_live_per_warp, live_w);
        for (int r = 0; r < R; ++r) {
            Mp_x[wi * R + r] = mx[r];
            Mp_y[wi * R + r] = sm[r];
            if (mx[r] != -INFINITY) {
                for (int d = 0; d < D; ++d) Op[(wi * R + r) * D + d] = O[r * D + d];
            }
        }
    }

    // Combine
    std::vector<float> O_fast(R * D, 0.0f), lse_fast(R, 0.0f);
    for (int r = 0; r < R; ++r) {
        int h = r % g;
        float M = -INFINITY;
        for (int w = 0; w < W; ++w) M = std::max(M, Mp_x[w * R + r]);
        float L = 0.0f;
        for (int w = 0; w < W; ++w) {
            if (Mp_x[w * R + r] == -INFINITY) continue;
            float f = expf(Mp_x[w * R + r] - M);
            L += f * Mp_y[w * R + r];
            for (int d = 0; d < D; ++d) O_fast[r * D + d] += f * Op[(w * R + r) * D + d];
        }
        float M2 = std::max(M, sinks[h]);
        float f = M == -INFINITY ? 0.0f : expf(M - M2);
        L = L * f + expf(sinks[h] - M2);
        for (int d = 0; d < D; ++d) {
            float gval = 1.0f / (1.0f + expf(-gate[r * D + d]));
            O_fast[r * D + d] = (O_fast[r * D + d] * f / L) * gval;
        }
        lse_fast[r] = M2 + logf(L);
    }

    double num = 0, den = 0, max_lse = 0;
    for (size_t i = 0; i < O_ref.size(); ++i) {
        double df = O_ref[i] - O_fast[i];
        num += df * df; den += (double)O_ref[i] * O_ref[i];
    }
    for (int r = 0; r < R; ++r) max_lse = std::max(max_lse, fabs((double)lse_ref[r] - lse_fast[r]));
    double nmse = num / std::max(den, 1e-30);
    bool ok = (nmse < 1e-6) && (max_lse < 1e-5) && (tiles_skipped > 120) && (max_live_per_warp <= 2);
    printf("[PASS=%d] Test 2: inkattn-fast SWA (n_kv=3072, swa=1024, M=5, Q8_0, real_rel_bias=%d):\n",
           ok, !real_rel_bias.empty());
    printf("         tiles_skipped=%d/%d (%.1f%%), live_tiles/warp=[%d..%d], NMSE=%.3e, max_LSE_err=%.3e\n",
           tiles_skipped, tiles_visited, 100.0 * tiles_skipped / tiles_visited,
           min_live_per_warp, max_live_per_warp, nmse, max_lse);
    return ok;
}

static bool test_sjkvarn_fast_modes() {
    // Realistic WHT-256 rotated 4/4 SJ-KVaRN across 64 records (8,192 tokens) with Inkling-2-Mini tau=1.25
    const int D = 256, NREC = 64, G = 128;
    const float scale = 1.25f / sqrtf(128.0f);
    std::mt19937 rng(777);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);

    double nmse1 = 0, nmse2 = 0, nmse3 = 0, lse1 = 0, lse2 = 0, lse3 = 0;
    for (int trial = 0; trial < 4; ++trial) {
        std::vector<float> Q(D, 0.0f);
        // Pair-packed [q, 0] rotated by WHT-256 just like Inkling-2-Mini!
        for (int d = 0; d < 128; ++d) Q[d] = nd(rng) * scale;
        wht256(Q.data());

        auto run = [&](int mode, std::vector<float> & out, float & lse_out) {
            const bool use_raw_k = (mode == 1 || mode == 2);
            const bool use_f16_k = (mode == 3);
            const bool use_f16_v = (mode == 2 || mode == 3);
            std::vector<float> acc(D, 0.0f), prev_vch(D, 1.0f);
            float kq_max = -1e30f, kq_sum = 0.0f, ps = 0.0f, pz = 0.0f;
            bool have_prev = false;
            std::mt19937 lrng(5000 + trial);

            for (int r = 0; r < NREC; ++r) {
                std::vector<float> kscale(D), kzero(D), vch(D);
                for (int d = 0; d < D; ++d) {
                    kscale[d] = h_round(0.06f + 0.14f * ud(lrng));
                    kzero[d]  = h_round(-0.7f + 0.15f * nd(lrng));
                    vch[d]    = h_round(0.5f + 0.9f * ud(lrng));
                }
                if (have_prev) {
                    for (int d = 0; d < D; ++d) {
                        if (!use_f16_v) {
                            bool hi = ((d % 16) >= 8);
                            acc[d] += (hi ? 16.0f * pz : pz) - 1024.0f * ps;
                        } else {
                            acc[d] += pz;
                        }
                        acc[d] *= (prev_vch[d] / vch[d]);
                    }
                    ps = 0.0f; pz = 0.0f;
                }
                have_prev = true;
                prev_vch = vch;

                std::vector<float> Qp(D);
                float c_r = 0.0f, q_sum = 0.0f;
                for (int d = 0; d < D; ++d) {
                    Qp[d] = h_round(h_round(Q[d]) * kscale[d]);
                    c_r += Q[d] * kzero[d];
                    q_sum += Qp[d];
                }
                float c_r_lo = c_r - 1024.0f * q_sum;
                float c_r_hi = 16.0f * c_r - 1024.0f * q_sum;

                for (int sl = 0; sl < G / 16; ++sl) {
                    float ktok[16], vsc[16], vze[16], kq[16];
                    int kcode[16][D], vcode[16][D];
                    for (int t = 0; t < 16; ++t) {
                        ktok[t] = h_round(0.6f + 0.8f * ud(lrng));
                        vsc[t]  = h_round(0.08f + 0.12f * ud(lrng));
                        vze[t]  = h_round(-0.6f + 0.15f * nd(lrng));
                        bool needle = (r == 12 && sl == 2 && t == 7) || (r == 47 && sl == 5 && t == 13);
                        for (int d = 0; d < D; ++d) {
                            kcode[t][d] = needle ? (Q[d] > 0 ? 15 : 0) : (lrng() & 15);
                            vcode[t][d] = lrng() & 15;
                        }
                    }
                    for (int t = 0; t < 16; ++t) {
                        if (use_f16_k) {
                            float dot_h = 0.0f;
                            for (int c = 0; c < 16; ++c) {
                                float ts = 0.0f;
                                for (int k = 0; k < 16; ++k) ts += (float)kcode[t][c*16+k] * Qp[c*16+k];
                                dot_h = h_round(dot_h + ts);
                            }
                            kq[t] = ktok[t] * (dot_h + c_r);
                        } else if (use_raw_k) {
                            float dot = 0.0f;
                            for (int d = 0; d < D; ++d) {
                                float rv = (t < 8) ? (1024.0f + kcode[t][d]) : (1024.0f + 16.0f * kcode[t][d]);
                                dot += rv * Qp[d];
                            }
                            kq[t] = (t < 8) ? ktok[t] * (dot + c_r_lo) : (0.0625f * ktok[t]) * (dot + c_r_hi);
                        } else {
                            float dot = 0.0f;
                            for (int d = 0; d < D; ++d) dot += (float)kcode[t][d] * Qp[d];
                            kq[t] = ktok[t] * (dot + c_r);
                        }
                    }
                    float mx = kq[0];
                    for (int t = 1; t < 16; ++t) mx = std::max(mx, kq[t]);
                    float nm = std::max(kq_max, mx);
                    if (nm != kq_max) {
                        float sc = expf(kq_max - nm);
                        kq_max = nm; kq_sum *= sc; ps *= sc; pz *= sc;
                        for (int d = 0; d < D; ++d) acc[d] *= sc;
                    }
                    float P[16], Pp[16];
                    for (int t = 0; t < 16; ++t) {
                        P[t] = expf(kq[t] - nm);
                        kq_sum += P[t];
                        Pp[t] = h_round(h_round(P[t]) * vsc[t]);
                        if (!use_f16_v) ps += Pp[t];
                        pz += P[t] * vze[t];
                    }
                    for (int d = 0; d < D; ++d) {
                        if (!use_f16_v) {
                            bool hi = ((d % 16) >= 8);
                            float s = 0.0f;
                            for (int t = 0; t < 16; ++t) s += (1024.0f + (hi ? 16.0f : 1.0f) * vcode[t][d]) * Pp[t];
                            acc[d] += s;
                        } else {
                            float s = 0.0f;
                            for (int t = 0; t < 16; ++t) s += (float)vcode[t][d] * Pp[t];
                            acc[d] += h_round(s);
                        }
                    }
                }
            }
            for (int d = 0; d < D; ++d) {
                if (!use_f16_v) {
                    bool hi = ((d % 16) >= 8);
                    acc[d] += (hi ? 16.0f * pz : pz) - 1024.0f * ps;
                    acc[d] *= (hi ? 0.0625f * prev_vch[d] : prev_vch[d]);
                } else {
                    acc[d] += pz;
                    acc[d] *= prev_vch[d];
                }
            }
            out.resize(D);
            for (int d = 0; d < D; ++d) out[d] = acc[d] / kq_sum;
            wht256(out.data());
            lse_out = kq_max + logf(kq_sum);
        };

        auto nmse = [&](const std::vector<float> & a, const std::vector<float> & b) {
            double num = 0, den = 0;
            for (int d = 0; d < 128; ++d) { double df = a[d] - b[d]; num += df * df; den += a[d] * a[d]; }
            return num / std::max(den, 1e-30);
        };

        std::vector<float> o0, o1, o2, o3;
        float l0, l1, l2, l3;
        run(0, o0, l0); run(1, o1, l1); run(2, o2, l2); run(3, o3, l3);
        nmse1 = std::max(nmse1, nmse(o0, o1)); lse1 = std::max(lse1, fabs((double)l0 - l1));
        nmse2 = std::max(nmse2, nmse(o0, o2)); lse2 = std::max(lse2, fabs((double)l0 - l2));
        nmse3 = std::max(nmse3, nmse(o0, o3)); lse3 = std::max(lse3, fabs((double)l0 - l3));
    }
    bool ok = (nmse1 < 1e-7) && (nmse2 < 1e-5) && (nmse3 < 5e-4);
    printf("[PASS=%d] Test 3: fattn-sjkvarn-fast WHT-256 8K-tok 4/4 (modes 1/2/3 vs baseline):\n", ok);
    printf("         FAST=1 (FP32 HMMA + raw_pairs K):          NMSE=%.3e, LSE_err=%.3e\n", nmse1, lse1);
    printf("         FAST=2 (FP32 raw K + per-strip FP16 V):    NMSE=%.3e, LSE_err=%.3e\n", nmse2, lse2);
    printf("         FAST=3 (FP16 KQ + per-strip FP16 V):       NMSE=%.3e, LSE_err=%.3e\n", nmse3, lse3);
    return ok;
}

int main() {
    std::vector<float> real_rel;
    bool loaded = load_gguf_rel_bias("/mnt/data/inkling2/Inkling-2-Mini-NVFP4-exact.gguf", real_rel);
    printf("Loaded real GGUF blk.0.attn_rel_bias.weight: %s (%zu floats)\n", loaded ? "YES" : "NO", real_rel.size());
    bool ok = true;
    ok &= test_q8_magic_exhaustive();
    ok &= test_inkattn_swa_skip_and_cyclic(real_rel);
    ok &= test_sjkvarn_fast_modes();
    printf("OVERALL: %s\n", ok ? "ALL PASS" : "FAILED");
    return ok ? 0 : 1;
}
