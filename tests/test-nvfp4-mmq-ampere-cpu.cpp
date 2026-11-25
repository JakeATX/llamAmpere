// Standalone CPU verification harness for Ampere NVFP4 MMQ (BP-1B block-128 INT8 IMMA & BP-1A bit-exact BF16 HMMA).
// Strictly CPU-only (zero CUDA runtime calls) — safe to build & run while GPUs are busy.
//
// Build & run:
//   g++ -O3 -std=c++17 tests/test-nvfp4-mmq-ampere-cpu.cpp -o /tmp/test-nvfp4-mmq-ampere-cpu
//   /tmp/test-nvfp4-mmq-ampere-cpu [/path/to/Inkling-2-Mini-NVFP4-exact.gguf]

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

static const int8_t kvalues_fp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

struct block_nvfp4 {
    uint8_t d[4];
    uint8_t qs[32];
};

static inline uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static inline float    u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

// Bit-exact PTX prmt.b32 (default mode) emulator
static inline uint32_t cpu_byte_perm(uint32_t a, uint32_t b, uint32_t sel) {
    uint8_t src[8];
    for (int i = 0; i < 4; ++i) {
        src[i]     = (uint8_t)((a >> (8 * i)) & 0xFFu);
        src[4 + i] = (uint8_t)((b >> (8 * i)) & 0xFFu);
    }
    uint32_t out = 0;
    for (int i = 0; i < 4; ++i) {
        uint32_t s = (sel >> (4 * i)) & 0xFu;
        uint8_t byte = (s < 8) ? src[s] : ((src[s & 7] & 0x80u) ? 0xFFu : 0x00u);
        out |= ((uint32_t)byte) << (8 * i);
    }
    return out;
}

// Reference ggml_ue4m3_to_fp32 (ggml-impl.h)
static inline float ue4m3_to_fp32_ref(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    int exp = (x >> 3) & 0xF;
    int man = x & 0x7;
    float raw = (exp == 0) ? ldexpf((float)man, -9) : ldexpf(1.0f + (float)man / 8.0f, exp - 7);
    return raw * 0.5f;
}

// Exact mirror of mmq-nvfp4-ampere.cuh primitives:
static inline float nvfp4_ue4m3_to_fp32_fast(uint8_t u) {
    uint32_t x     = (uint32_t)u & 0x7Fu;
    uint32_t is_sn = x < 8u;
    uint32_t base  = is_sn ? 0x3C000000u : 0x3B800000u;
    float    sub   = is_sn ? 0.0078125f  : 0.0f;
    float    val   = u2f(base + (x << 20)) - sub;
    return (x == 0x7Fu) ? 0.0f : val;
}

static inline float nvfp4_fast_rcp_no_mufu(float a) {
    float y = u2f(0x7EF311C2u - f2u(a));
    y = y * fmaf(-a, y, 2.0f);
    y = y * fmaf(-a, y, 2.0f);
    return (a > 0.0f) ? y : 0.0f;
}

static inline float nvfp4_magic_i2f(int x) {
    return u2f((uint32_t)(x + 0x4B400000)) - 12582912.0f;
}

static inline int nvfp4_magic_f2i(float x) {
    return (int)f2u(x + 12582912.0f) - 0x4B400000;
}

static inline void nvfp4_lut_i8_regs(uint32_t q4, uint32_t t0, uint32_t t1, uint32_t t2, uint32_t t3, int8_t out_lo[4], int8_t out_hi[4]) {
    uint32_t idx = q4 & 0x77777777u;
    uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
    uint32_t lo0 = cpu_byte_perm(t0, t1, idx);
    uint32_t hi0 = cpu_byte_perm(t2, t3, idx);
    uint32_t lo1 = cpu_byte_perm(t0, t1, idx >> 16);
    uint32_t hi1 = cpu_byte_perm(t2, t3, idx >> 16);
    uint32_t m0  = cpu_byte_perm(lo0, hi0, sel);
    uint32_t m1  = cpu_byte_perm(lo1, hi1, sel >> 16);
    uint32_t x   = cpu_byte_perm(m0, m1, 0x6420u);
    uint32_t y   = cpu_byte_perm(m0, m1, 0x7531u);
    memcpy(out_lo, &x, 4);
    memcpy(out_hi, &y, 4);
}

static inline void nvfp4_build_bf16_lut(float d, uint32_t & lo0, uint32_t & lo1, uint32_t & hi0, uint32_t & hi1) {
    uint32_t f1  = f2u(d);
    uint32_t f2  = f2u(d * 2.0f);
    uint32_t f3  = f2u(d * 3.0f);
    uint32_t f4  = f2u(d * 4.0f);
    uint32_t f6  = f2u(d * 6.0f);
    uint32_t f8  = f2u(d * 8.0f);
    uint32_t f12 = f2u(d * 12.0f);

    uint32_t p01  = cpu_byte_perm(0u, f1, 0x7610u);
    uint32_t p23  = cpu_byte_perm(f2, f3, 0x7632u);
    uint32_t p46  = cpu_byte_perm(f4, f6, 0x7632u);
    uint32_t p812 = cpu_byte_perm(f8, f12, 0x7632u);

    lo0 = cpu_byte_perm(p01,  p23,  0x6420u);
    hi0 = cpu_byte_perm(p01,  p23,  0x7531u);
    lo1 = cpu_byte_perm(p46,  p812, 0x6420u);
    hi1 = cpu_byte_perm(p46,  p812, 0x7531u);
}

static inline void nvfp4_decode_q4_bf16x8(uint32_t q4, uint32_t lo0, uint32_t lo1, uint32_t hi0, uint32_t hi1, uint32_t out[4]) {
    uint32_t idx = q4 & 0x77777777u;
    uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);

    uint32_t l0  = cpu_byte_perm(lo0, lo1, idx);
    uint32_t hp0 = cpu_byte_perm(hi0, hi1, idx);
    uint32_t h0  = cpu_byte_perm(hp0, hp0 ^ 0x80808080u, sel);

    uint32_t l1  = cpu_byte_perm(lo0, lo1, idx >> 16);
    uint32_t hp1 = cpu_byte_perm(hi0, hi1, idx >> 16);
    uint32_t h1  = cpu_byte_perm(hp1, hp1 ^ 0x80808080u, sel >> 16);

    out[0] = cpu_byte_perm(l0, h0, 0x6240u);
    out[1] = cpu_byte_perm(l1, h1, 0x6240u);
    out[2] = cpu_byte_perm(l0, h0, 0x7351u);
    out[3] = cpu_byte_perm(l1, h1, 0x7351u);
}

static inline uint16_t fp32_to_bf16_rn(float x) {
    uint32_t u = f2u(x);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (uint16_t)(u >> 16) | 0x0040u;
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

static int run_unit_tests() {
    std::mt19937 rng(12345);
    int fails = 0;

    for (int u = 0; u < 128; ++u) {
        if (f2u(ue4m3_to_fp32_ref((uint8_t)u)) != f2u(nvfp4_ue4m3_to_fp32_fast((uint8_t)u))) {
            printf("FAIL: UE4M3 bit mismatch at u=%d\n", u);
            fails++;
        }
    }

    double max_rcp_rel = 0.0;
    for (int u = 1; u < 127; ++u) {
        float s = nvfp4_ue4m3_to_fp32_fast((uint8_t)u);
        float r = nvfp4_fast_rcp_no_mufu(s);
        double rel = fabs((double)r - 1.0 / (double)s) * (double)s;
        max_rcp_rel = std::max(max_rcp_rel, rel);
    }
    if (max_rcp_rel > 1e-5) {
        printf("FAIL: no-MUFU rcp error %.3e > 1e-5\n", max_rcp_rel);
        fails++;
    }

    for (int x = -2097152; x <= 2097152; x += 997) {
        if (nvfp4_magic_i2f(x) != (float)x) {
            printf("FAIL: magic_i2f mismatch at x=%d\n", x);
            fails++;
        }
    }

    for (int u = 0; u < 128; ++u) {
        float d = nvfp4_ue4m3_to_fp32_fast((uint8_t)u);
        uint32_t lo0, lo1, hi0, hi1;
        nvfp4_build_bf16_lut(d, lo0, lo1, hi0, hi1);
        for (int t = 0; t < 128; ++t) {
            uint32_t q4 = (t == 0) ? 0xF7E6D5C4u : (t == 1) ? 0xB3A29180u : rng();
            uint32_t out[4];
            nvfp4_decode_q4_bf16x8(q4, lo0, lo1, hi0, hi1, out);
            uint16_t lo[4] = {(uint16_t)(out[0]&0xFFFFu), (uint16_t)(out[0]>>16), (uint16_t)(out[1]&0xFFFFu), (uint16_t)(out[1]>>16)};
            uint16_t hi[4] = {(uint16_t)(out[2]&0xFFFFu), (uint16_t)(out[2]>>16), (uint16_t)(out[3]&0xFFFFu), (uint16_t)(out[3]>>16)};
            for (int b = 0; b < 4; ++b) {
                uint8_t byte = (uint8_t)((q4 >> (8*b)) & 0xFFu);
                if (u2f((uint32_t)lo[b] << 16) != (float)kvalues_fp4[byte & 0xF] * d ||
                    u2f((uint32_t)hi[b] << 16) != (float)kvalues_fp4[byte >> 4]  * d) {
                    fails++;
                }
            }
        }
    }

    for (int R = 0; R <= 43349; R += 313) {
        uint32_t v1=(R+2048u)>>12, v2=(2*R+2048u)>>12, v3=(3*R+2048u)>>12;
        uint32_t v4=(4*R+2048u)>>12, v6=(6*R+2048u)>>12, v8=(8*R+2048u)>>12, v12=(12*R+2048u)>>12;
        int8_t ref[16] = {0,(int8_t)v1,(int8_t)v2,(int8_t)v3,(int8_t)v4,(int8_t)v6,(int8_t)v8,(int8_t)v12,
                          0,(int8_t)-v1,(int8_t)-v2,(int8_t)-v3,(int8_t)-v4,(int8_t)-v6,(int8_t)-v8,(int8_t)-v12};
        uint32_t t0 = (v1<<8)|(v2<<16)|(v3<<24);
        uint32_t t1 = v4|(v6<<8)|(v8<<16)|(v12<<24);
        uint32_t t2 = (0x80808080u - t0) ^ 0x80808080u;
        uint32_t t3 = (0x80808080u - t1) ^ 0x80808080u;
        for (int t = 0; t < 32; ++t) {
            uint32_t q4 = (t == 0) ? 0xF7E6D5C4u : rng();
            int8_t lo[4], hi[4];
            nvfp4_lut_i8_regs(q4, t0, t1, t2, t3, lo, hi);
            for (int b = 0; b < 4; ++b) {
                uint8_t byte = (uint8_t)((q4 >> (8*b)) & 0xFFu);
                if (lo[b] != ref[byte & 0xF] || hi[b] != ref[byte >> 4]) fails++;
            }
        }
    }

    printf("[UNIT] UE4M3 bit-exact, no-MUFU rcp (max_rel=%.3e), magic_i2f, PRMT BP-1A/BP-1B: %s (%d failures)\n",
           max_rcp_rel, fails == 0 ? "PASS" : "FAIL", fails);
    return fails;
}

struct TensorInfo { std::string name; std::vector<uint64_t> dims; uint32_t type; uint64_t offset; };

static std::string read_gguf_str(std::ifstream & f) {
    uint64_t len = 0; f.read((char*)&len, 8);
    std::string s(len, '\0'); f.read(s.data(), len);
    return s;
}

static void skip_gguf_val(std::ifstream & f, uint32_t t) {
    static const int sz[13] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
    if (t < 13 && sz[t] > 0) f.seekg(sz[t], std::ios::cur);
    else if (t == 8) read_gguf_str(f);
    else if (t == 9) {
        uint32_t et = 0; uint64_t cnt = 0;
        f.read((char*)&et, 4); f.read((char*)&cnt, 8);
        if (et < 13 && sz[et] > 0) f.seekg(sz[et] * cnt, std::ios::cur);
        else for (uint64_t i = 0; i < cnt; ++i) skip_gguf_val(f, et);
    }
}

struct Metrics { double rel_rms; double cos_sim; };
static Metrics compute_metrics(const std::vector<float> & ref, const std::vector<float> & got) {
    double dot = 0, nr = 0, ng = 0, e2 = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double r = ref[i], g = got[i], e = r - g;
        dot += r * g; nr += r * r; ng += g * g; e2 += e * e;
    }
    return { sqrt(e2 / std::max(nr, 1e-30)), dot / std::max(sqrt(nr * ng), 1e-30) };
}

static int run_gguf_eval(const char * path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { printf("ERROR: cannot open %s\n", path); return 1; }

    char magic[4]; f.read(magic, 4);
    uint32_t ver; uint64_t n_tensors, n_kv;
    f.read((char*)&ver, 4); f.read((char*)&n_tensors, 8); f.read((char*)&n_kv, 8);
    for (uint64_t i = 0; i < n_kv; ++i) {
        read_gguf_str(f); uint32_t t; f.read((char*)&t, 4); skip_gguf_val(f, t);
    }
    std::vector<TensorInfo> tensors;
    for (uint64_t i = 0; i < n_tensors; ++i) {
        TensorInfo ti;
        ti.name = read_gguf_str(f);
        uint32_t ndims; f.read((char*)&ndims, 4);
        ti.dims.resize(ndims);
        f.read((char*)ti.dims.data(), 8 * ndims);
        f.read((char*)&ti.type, 4); f.read((char*)&ti.offset, 8);
        if (ti.type == 40) tensors.push_back(ti);
    }
    uint64_t data_off = ((uint64_t)f.tellg() + 31ull) & ~31ull;

    std::vector<std::string> targets = {
        "blk.0.ffn_gate_up_exps.weight", "blk.0.ffn_down_exps.weight",
        "blk.10.ffn_gate_up_exps.weight", "blk.10.ffn_down_exps.weight",
        "blk.15.ffn_gate_up_exps.weight", "blk.15.ffn_down_exps.weight"
    };

    std::mt19937 rng(42);
    std::normal_distribution<float> norm(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    const int M_rows = 256, N_toks = 16;

    for (const auto & tname : targets) {
        const TensorInfo * ti = nullptr;
        for (const auto & t : tensors) if (t.name == tname) { ti = &t; break; }
        if (!ti) continue;

        int K = (int)ti->dims[0], nb = K / 64;
        std::vector<block_nvfp4> raw(M_rows * nb);
        f.seekg(data_off + ti->offset, std::ios::beg);
        f.read((char*)raw.data(), raw.size() * sizeof(block_nvfp4));

        std::vector<float> W_ref(M_rows * K), W_1b_rec(M_rows * K), W_1a_rec(M_rows * K);
        std::vector<int8_t> W_i8(M_rows * K);
        std::vector<float> W_s128(M_rows * (K / 128));
        std::vector<uint16_t> W_bf16(M_rows * K);

        for (int r = 0; r < M_rows; ++r) {
            const block_nvfp4 * row = &raw[r * nb];
            // Exact ref + BP-1A PRMT decode
            for (int b = 0; b < nb; ++b) {
                const uint32_t * qs32 = reinterpret_cast<const uint32_t *>(row[b].qs);
                for (int s = 0; s < 4; ++s) {
                    float d = nvfp4_ue4m3_to_fp32_fast(row[b].d[s]);
                    uint32_t lo0, lo1, hi0, hi1;
                    nvfp4_build_bf16_lut(d, lo0, lo1, hi0, hi1);
                    uint32_t q0[4], q1[4];
                    nvfp4_decode_q4_bf16x8(qs32[2*s + 0], lo0, lo1, hi0, hi1, q0);
                    nvfp4_decode_q4_bf16x8(qs32[2*s + 1], lo0, lo1, hi0, hi1, q1);
                    uint32_t dst_w[8] = {q0[0], q0[1], q1[0], q1[1], q0[2], q0[3], q1[2], q1[3]};
                    memcpy(&W_bf16[r*K + b*64 + s*16], dst_w, 32);
                    for (int j = 0; j < 8; ++j) {
                        uint8_t byte = row[b].qs[s*8 + j];
                        W_ref[r*K + b*64 + s*16 + j]     = kvalues_fp4[byte & 0xF] * d;
                        W_ref[r*K + b*64 + s*16 + j + 8] = kvalues_fp4[byte >> 4]  * d;
                    }
                    for (int k = 0; k < 16; ++k) {
                        W_1a_rec[r*K + b*64 + s*16 + k] = u2f((uint32_t)W_bf16[r*K + b*64 + s*16 + k] << 16);
                    }
                }
            }
            // Exact BP-1B PRMT block-128 scale-folded encode
            for (int c = 0; c < K / 128; ++c) {
                float sc[8], s_max = 0.0f;
                for (int b = 0; b < 2; ++b)
                    for (int s = 0; s < 4; ++s) {
                        sc[4*b + s] = nvfp4_ue4m3_to_fp32_fast(row[2*c + b].d[s]);
                        s_max = std::max(s_max, sc[4*b + s]);
                    }
                W_s128[r*(K/128) + c] = s_max * (12.0f / 127.0f);
                float inv_q12 = (4096.0f * (127.0f / 12.0f)) * nvfp4_fast_rcp_no_mufu(s_max);
                for (int b = 0; b < 2; ++b) {
                    const uint32_t * qs32 = reinterpret_cast<const uint32_t *>(row[2*c + b].qs);
                    for (int s = 0; s < 4; ++s) {
                        uint32_t R = (uint32_t)std::min(std::max(nvfp4_magic_f2i(sc[4*b + s] * inv_q12), 0), 43349);
                        uint32_t v1=(R+2048u)>>12, v2=(2*R+2048u)>>12, v3=(3*R+2048u)>>12;
                        uint32_t v4=(4*R+2048u)>>12, v6=(6*R+2048u)>>12, v8=(8*R+2048u)>>12, v12=(12*R+2048u)>>12;
                        uint32_t t0 = (v1<<8)|(v2<<16)|(v3<<24);
                        uint32_t t1 = v4|(v6<<8)|(v8<<16)|(v12<<24);
                        uint32_t t2 = (0x80808080u - t0) ^ 0x80808080u;
                        uint32_t t3 = (0x80808080u - t1) ^ 0x80808080u;
                        int8_t * dst = &W_i8[r*K + c*128 + b*64 + s*16];
                        nvfp4_lut_i8_regs(qs32[2*s + 0], t0, t1, t2, t3, dst + 0, dst + 8);
                        nvfp4_lut_i8_regs(qs32[2*s + 1], t0, t1, t2, t3, dst + 4, dst + 12);
                    }
                }
                for (int i = 0; i < 128; ++i)
                    W_1b_rec[r*K + c*128 + i] = (float)W_i8[r*K + c*128 + i] * W_s128[r*(K/128) + c];
            }
        }
        auto wm1a = compute_metrics(W_ref, W_1a_rec);
        auto wm1b = compute_metrics(W_ref, W_1b_rec);
        printf("[GGUF] %s (K=%d): W_relRMS BP-1A=%.6f%% | BP-1B=%.4f%% (cos=%.6f)\n",
               tname.c_str(), K, wm1a.rel_rms*100, wm1b.rel_rms*100, wm1b.cos_sim);

        for (int act_mode = 0; act_mode < 2; ++act_mode) {
            std::vector<float> X(N_toks * K), ch_scale(K, 1.0f);
            if (act_mode == 1) {
                for (int k = 0; k < K; ++k) {
                    float u = uni(rng);
                    if (u < 0.02f) ch_scale[k] = 15.0f;
                    else if (u < 0.07f) ch_scale[k] = 4.0f;
                }
            }
            for (int t = 0; t < N_toks; ++t)
                for (int k = 0; k < K; ++k) {
                    float v = norm(rng) * ch_scale[k];
                    if (act_mode == 1 && uni(rng) < 0.005f) v *= 8.0f;
                    X[t*K + k] = v;
                }

            std::vector<int8_t> X_q128(N_toks * K);
            std::vector<float>  X_d128(N_toks * (K/128));
            std::vector<uint16_t> X_bf16(N_toks * K);
            for (int t = 0; t < N_toks; ++t) {
                for (int c = 0; c < K / 128; ++c) {
                    float amax = 0;
                    for (int i = 0; i < 128; ++i) amax = std::max(amax, fabsf(X[t*K + c*128 + i]));
                    float id = amax > 0 ? 127.0f / amax : 0.0f;
                    X_d128[t*(K/128) + c] = amax / 127.0f;
                    for (int i = 0; i < 128; ++i) X_q128[t*K + c*128 + i] = (int8_t)lrintf(X[t*K + c*128 + i] * id);
                }
                for (int k = 0; k < K; ++k) X_bf16[t*K + k] = fp32_to_bf16_rn(X[t*K + k]);
            }

            std::vector<float> Y_ref(N_toks * M_rows), Y_1b(N_toks * M_rows), Y_1a(N_toks * M_rows);
            for (int t = 0; t < N_toks; ++t) {
                for (int r = 0; r < M_rows; ++r) {
                    double s_ref = 0.0; float s_1b = 0.0f, s_1a = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        s_ref += (double)W_ref[r*K + k] * (double)X[t*K + k];
                        s_1a  += u2f((uint32_t)W_bf16[r*K + k] << 16) * u2f((uint32_t)X_bf16[t*K + k] << 16);
                    }
                    for (int c = 0; c < K / 128; ++c) {
                        int acc = 0;
                        for (int i = 0; i < 128; ++i) acc += (int)W_i8[r*K + c*128 + i] * (int)X_q128[t*K + c*128 + i];
                        s_1b = fmaf(nvfp4_magic_i2f(acc), W_s128[r*(K/128) + c] * X_d128[t*(K/128) + c], s_1b);
                    }
                    Y_ref[t*M_rows + r] = (float)s_ref;
                    Y_1b[t*M_rows + r]  = s_1b;
                    Y_1a[t*M_rows + r]  = s_1a;
                }
            }
            auto m1b = compute_metrics(Y_ref, Y_1b);
            auto m1a = compute_metrics(Y_ref, Y_1a);
            printf("       %s -> BP-1B relRMS=%.4f%% cos=%.6f | BP-1A relRMS=%.4f%% cos=%.6f\n",
                   act_mode == 0 ? "Gaussian  " : "Heavy-Tail",
                   m1b.rel_rms*100, m1b.cos_sim, m1a.rel_rms*100, m1a.cos_sim);
        }
    }
    return 0;
}

int main(int argc, char ** argv) {
    if (run_unit_tests() != 0) return 1;
    if (argc > 1) return run_gguf_eval(argv[1]);
    return 0;
}
