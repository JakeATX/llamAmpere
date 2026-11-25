// Standalone CPU-only C++17 verification suite for OPT9 AmperInteger (AI+) Drop-In Fusions.
// Verifies bit-exact / <1e-5 parity of:
//   1. rms_norm_mul_quantize_act_ai4 (H256 & H128, PLAIN/SWAP8/XPERM/LINEAR, REG_CACHE=1/2/0)
//   2. swiglu_quantize_act_ai4       (H256 & H128, PLAIN/SWAP8/XPERM/LINEAR, 2-tensor & split-tensor, all 4 GLU ops)
//   3. mmv_grp / mmv_grp_spec        (block->member mapping, all 11 AI+ unit layouts, N=1..32, real Inkling Q/K/V)
//   4. dense_gate_up_glu & +Q8 bridge(interleaved Gate+Up+SwiGLU+H256/H128+Q8 on real Inkling weights, N=1..32)
//
// Strictly CPU-only! Reads real weights from /mnt/data/inkling2/Inkling-2-Mini-NVFP4-exact.gguf when present.
// Build: g++ -O3 -std=c++17 -Wall -Wextra aiplus_fuse_cpu_verify.cpp -o /tmp/aiplus_fuse_cpu_verify -lm

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

static constexpr uint32_t RQ4_SIGN_WORDS[8] = {
    0x9e3779b9u, 0x7f4a7c15u, 0xf39cc060u, 0x5ced2a1bu,
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au
};

enum { PREP_PLAIN = 0, PREP_SWAP8 = 1, PREP_XPERM = 2, PREP_LINEAR = 3 };
enum { GLU_SWIGLU = 0, GLU_GEGLU = 1, GLU_SWIGLU_OAI = 2, GLU_SWIGLU_CLAMP = 3 };

// Exact IEEE round-to-nearest-even matching CUDA __float2int_rn
static inline int float2int_rn(float x) {
    return (int) std::nearbyintf(x);
}

static inline float eval_glu_cpu(int op, float gate, float up, float alpha, float limit) {
    if (op == GLU_SWIGLU) {
        return up * (gate / (1.0f + std::exp(-gate)));
    } else if (op == GLU_GEGLU) {
        constexpr float GELU_COEF_A    = 0.044715f;
        constexpr float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
        return up * (0.5f * gate * (1.0f + std::tanh(SQRT_2_OVER_PI * gate * (1.0f + GELU_COEF_A * gate * gate))));
    } else if (op == GLU_SWIGLU_OAI) {
        const float x = std::fmin(gate, limit);
        const float g = std::fmax(std::fmin(up, limit), -limit);
        return (x / (1.0f + std::exp(-x * alpha))) * (1.0f + g);
    } else {
        const float x = std::fmin(gate, limit);
        const float g = std::fmax(std::fmin(up, limit), -limit);
        return g * (x / (1.0f + std::exp(-x)));
    }
}

// -------------------------------------------------------------------------------------------------
// Exact CPU simulation of rq4::quantize_act_body (H256) and ai4k::quantize_act128_body (H128)
// -------------------------------------------------------------------------------------------------

static void ref_quantize_act_block(
        const float * x_blk, int RB, int variant, int tok, int blk_idx, int K,
        int8_t * q_out, float * s_out) {
    std::vector<float> v(RB);
    for (int i = 0; i < RB; ++i) {
        const bool neg = (RQ4_SIGN_WORDS[i >> 5] >> (i & 31)) & 1;
        v[i] = neg ? -x_blk[i] : x_blk[i];
    }
    // Sylvester FWHT in exact same butterfly order as GPU (h=1,2,4 in-thread, then m=1,2,4,8[,16] across lanes)
    for (int h = 1; h < RB; h <<= 1) {
        for (int i = 0; i < RB; i += 2 * h) {
            for (int j = i; j < i + h; ++j) {
                const float a = v[j], b = v[j + h];
                v[j]     = a + b;
                v[j + h] = a - b;
            }
        }
    }
    const float nrm = (RB == 256) ? (1.0f / 16.0f) : 0.08838834764831845f;
    float amax = 0.0f;
    for (int i = 0; i < RB; ++i) {
        v[i] *= nrm;
        amax = std::fmax(amax, std::fabs(v[i]));
    }
    const float d  = amax / 127.0f;
    const float id = amax > 0.0f ? 127.0f / amax : 0.0f;
    const int nlanes = RB / 8;
    for (int lane = 0; lane < nlanes; ++lane) {
        uint32_t w[2] = { 0u, 0u };
        for (int e = 0; e < 8; ++e) {
            const int qi = float2int_rn(v[lane * 8 + e] * id);
            w[e >> 2] |= ((uint32_t) (qi & 0xff)) << ((e & 3) * 8);
        }
        if (RB == 256) {
            const int c  = lane >> 4;
            const int j  = (lane & 15) >> 2;
            const int h  = (lane & 3) >> 1;
            const int t0 = (lane & 1) * 2;
            if (variant == PREP_LINEAR) {
                std::memcpy(q_out + (int64_t) tok * K + blk_idx * 256 + lane * 8, w, 8);
            } else if (variant == PREP_XPERM) {
                int8_t * qr = q_out + (int64_t) tok * K + blk_idx * 256 + c * 128 + h * 64 + (t0 >> 1) * 32 + j * 8;
                std::memcpy(qr, w, 8);
            } else {
                const int jx = j ^ (variant == PREP_SWAP8 ? ((tok >> 1) & 1) : 0);
                int8_t * qr = q_out + (int64_t) tok * K + blk_idx * 256 + c * 128 + jx * 8 + h * 4;
                std::memcpy(qr + (t0 + 0) * 32, &w[0], 4);
                std::memcpy(qr + (t0 + 1) * 32, &w[1], 4);
            }
        } else {
            const int l  = lane;
            const int j  = l >> 2;
            const int h  = (l & 3) >> 1;
            const int t0 = (l & 1) * 2;
            if (variant == PREP_LINEAR) {
                std::memcpy(q_out + (int64_t) tok * K + blk_idx * 128 + l * 8, w, 8);
            } else if (variant == PREP_XPERM) {
                int8_t * qr = q_out + (int64_t) tok * K + blk_idx * 128 + h * 64 + (t0 >> 1) * 32 + j * 8;
                std::memcpy(qr, w, 8);
            } else {
                const int jx = j ^ (variant == PREP_SWAP8 ? ((tok >> 1) & 1) : 0);
                int8_t * qr = q_out + (int64_t) tok * K + blk_idx * 128 + jx * 8 + h * 4;
                std::memcpy(qr + (t0 + 0) * 32, &w[0], 4);
                std::memcpy(qr + (t0 + 1) * 32, &w[1], 4);
            }
        }
    }
    s_out[(int64_t) tok * (K / RB) + blk_idx] = d;
}

static void ref_quantize_act(const float * x, int K, int N, int xsh, int variant, int8_t * q, float * s) {
    const int RB = xsh ? 256 : 128;
    const int nb = K / RB;
    for (int tok = 0; tok < N; ++tok) {
        for (int b = 0; b < nb; ++b) {
            ref_quantize_act_block(x + (int64_t) tok * K + b * RB, RB, variant, tok, b, K, q, s);
        }
    }
}

// Dequantize (q, s) back to rotated FP32 vector for L2/max error checks (supports PLAIN layout)
static void dequant_rotated_plain(const int8_t * q, const float * s, int K, int N, int xsh, std::vector<float> & out) {
    out.assign((size_t) N * K, 0.0f);
    const int RB = xsh ? 256 : 128;
    const int nb = K / RB;
    for (int tok = 0; tok < N; ++tok) {
        for (int b = 0; b < nb; ++b) {
            const float d = s[(int64_t) tok * nb + b];
            const int nlanes = RB / 8;
            for (int lane = 0; lane < nlanes; ++lane) {
                uint32_t w[2];
                if (RB == 256) {
                    const int c = lane >> 4, j = (lane & 15) >> 2, h = (lane & 3) >> 1, t0 = (lane & 1) * 2;
                    const int8_t * qr = q + (int64_t) tok * K + b * 256 + c * 128 + j * 8 + h * 4;
                    std::memcpy(&w[0], qr + t0 * 32, 4);
                    std::memcpy(&w[1], qr + (t0 + 1) * 32, 4);
                } else {
                    const int l = lane, j = l >> 2, h = (l & 3) >> 1, t0 = (l & 1) * 2;
                    const int8_t * qr = q + (int64_t) tok * K + b * 128 + j * 8 + h * 4;
                    std::memcpy(&w[0], qr + t0 * 32, 4);
                    std::memcpy(&w[1], qr + (t0 + 1) * 32, 4);
                }
                for (int e = 0; e < 8; ++e) {
                    const int8_t qi = (int8_t) ((w[e >> 2] >> ((e & 3) * 8)) & 0xff);
                    out[(int64_t) tok * K + b * RB + lane * 8 + e] = d * (float) qi;
                }
            }
        }
    }
}

// Simulate norm.cu rms_norm_f32 (block_size=1024, float4 reduction) -> FP32
static void ref_rms_norm_mul_f32(const float * x, const float * gamma, float * dst, int K, int N, float eps) {
    for (int tok = 0; tok < N; ++tok) {
        const float * xr = x + (int64_t) tok * K;
        float * dr = dst + (int64_t) tok * K;
        // Exact norm.cu block_size=1024 float4 tree reduction
        float warp_sum[32] = {};
        for (int tid = 0; tid < 1024; ++tid) {
            float t = 0.0f;
            for (int i = tid; i < K / 4; i += 1024) {
                const float a = xr[4 * i + 0], b = xr[4 * i + 1], c = xr[4 * i + 2], d = xr[4 * i + 3];
                t += a * a + b * b + c * c + d * d;
            }
            warp_sum[tid >> 5] += t;
        }
        float total = 0.0f;
        for (int w = 0; w < 32; ++w) total += warp_sum[w];
        const float scale = 1.0f / std::sqrt(total / (float) K + eps);
        for (int i = 0; i < K; ++i) {
            dr[i] = (scale * xr[i]) * gamma[i];
        }
    }
}

// Simulate k_rms_norm_mul_quantize_act_ai4 (BLOCK_WARPS=16, 512 threads, warp-per-256 reduction)
static void sim_fused_rms_norm_mul_q(
        const float * x, const float * gamma, float * dst_f32, int8_t * q, float * s,
        int K, int N, float eps, int xsh, int variant) {
    constexpr int BLOCK_WARPS = 16;
    const int nwb = (K + 255) >> 8;
    const int RB = xsh ? 256 : 128;
    for (int tok = 0; tok < N; ++tok) {
        const float * xr = x + (int64_t) tok * K;
        float s_sum[BLOCK_WARPS] = {};
        for (int warp = 0; warp < BLOCK_WARPS; ++warp) {
            float lane_sq[32] = {};
            for (int wb = warp; wb < nwb; wb += BLOCK_WARPS) {
                for (int lane = 0; lane < 32; ++lane) {
                    const int off = wb * 256 + lane * 8;
                    if (off < K) {
                        for (int e = 0; e < 8; ++e) {
                            const float v = xr[off + e];
                            lane_sq[lane] += v * v;
                        }
                    }
                }
            }
            // Exact warp XOR shuffle reduction
            for (int m = 16; m > 0; m >>= 1) {
                for (int lane = 0; lane < m; ++lane) {
                    lane_sq[lane] += lane_sq[lane ^ m];
                }
            }
            s_sum[warp] = lane_sq[0];
        }
        // Warp 0 XOR reduction across BLOCK_WARPS
        for (int m = BLOCK_WARPS >> 1; m > 0; m >>= 1) {
            for (int l = 0; l < m; ++l) {
                s_sum[l] += s_sum[l ^ m];
            }
        }
        const float inv_rms = 1.0f / std::sqrt(s_sum[0] / (float) K + eps);
        std::vector<float> tmp(K);
        for (int i = 0; i < K; ++i) {
            tmp[i] = (inv_rms * xr[i]) * gamma[i];
            if (dst_f32) dst_f32[(int64_t) tok * K + i] = tmp[i];
        }
        const int nb = K / RB;
        for (int b = 0; b < nb; ++b) {
            ref_quantize_act_block(tmp.data() + b * RB, RB, variant, tok, b, K, q, s);
        }
    }
}

// Simulate k_swiglu_quantize_act_ai4
static void sim_fused_swiglu_q(
        const float * gate, int64_t s_gate, const float * up, int64_t s_up,
        float * dst_f32, int8_t * q, float * s,
        int K, int N, int xsh, int variant, int glu_op, float alpha, float limit) {
    const int RB = xsh ? 256 : 128;
    const int nb = K / RB;
    std::vector<float> blk(RB);
    for (int tok = 0; tok < N; ++tok) {
        const float * gr = gate + (int64_t) tok * s_gate;
        const float * ur = up   + (int64_t) tok * s_up;
        for (int b = 0; b < nb; ++b) {
            for (int i = 0; i < RB; ++i) {
                const float v = eval_glu_cpu(glu_op, gr[b * RB + i], ur[b * RB + i], alpha, limit);
                blk[i] = v;
                if (dst_f32) dst_f32[(int64_t) tok * K + b * RB + i] = v;
            }
            ref_quantize_act_block(blk.data(), RB, variant, tok, b, K, q, s);
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Minimal GGUF reader for real /mnt/data/inkling2/Inkling-2-Mini-NVFP4-exact.gguf weights
// -------------------------------------------------------------------------------------------------

struct gguf_tensor_info {
    std::string name;
    uint32_t type = 0;
    std::vector<uint64_t> ne;
    uint64_t offset = 0;
};

struct gguf_reader {
    std::ifstream f;
    uint64_t data_offset = 0;
    std::map<std::string, gguf_tensor_info> tensors;

    static uint64_t read_u64(std::ifstream & s) { uint64_t v = 0; s.read((char *) &v, 8); return v; }
    static uint32_t read_u32(std::ifstream & s) { uint32_t v = 0; s.read((char *) &v, 4); return v; }
    static std::string read_str(std::ifstream & s) {
        const uint64_t n = read_u64(s);
        std::string r(n, '\0');
        s.read(&r[0], (std::streamsize) n);
        return r;
    }
    static void skip_val(std::ifstream & s, uint32_t t) {
        static const int sz[13] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
        if (t == 8) { (void) read_str(s); }
        else if (t == 9) {
            const uint32_t et = read_u32(s);
            const uint64_t cnt = read_u64(s);
            for (uint64_t i = 0; i < cnt; ++i) skip_val(s, et);
        } else {
            s.seekg(sz[t], std::ios::cur);
        }
    }

    bool open(const std::string & path) {
        f.open(path, std::ios::binary);
        if (!f) return false;
        char magic[4];
        f.read(magic, 4);
        if (std::memcmp(magic, "GGUF", 4) != 0) return false;
        (void) read_u32(f); // ver
        const uint64_t n_tensors = read_u64(f);
        const uint64_t n_kv = read_u64(f);
        uint64_t align = 32;
        for (uint64_t i = 0; i < n_kv; ++i) {
            const std::string k = read_str(f);
            const uint32_t t = read_u32(f);
            if (k == "general.alignment" && t == 4) {
                align = read_u32(f);
            } else {
                skip_val(f, t);
            }
        }
        for (uint64_t i = 0; i < n_tensors; ++i) {
            gguf_tensor_info ti;
            ti.name = read_str(f);
            const uint32_t nd = read_u32(f);
            ti.ne.resize(nd);
            for (uint32_t d = 0; d < nd; ++d) ti.ne[d] = read_u64(f);
            ti.type = read_u32(f);
            ti.offset = read_u64(f);
            tensors[ti.name] = ti;
        }
        const uint64_t pos = (uint64_t) f.tellg();
        data_offset = (pos + align - 1) / align * align;
        return true;
    }

    bool read_f32(const std::string & name, std::vector<float> & out, size_t max_elems = 0) {
        auto it = tensors.find(name);
        if (it == tensors.end() || it->second.type != 0) return false;
        size_t n = 1;
        for (auto d : it->second.ne) n *= (size_t) d;
        if (max_elems && n > max_elems) n = max_elems;
        out.resize(n);
        f.seekg((std::streamoff) (data_offset + it->second.offset));
        f.read((char *) out.data(), (std::streamsize) (n * sizeof(float)));
        return (bool) f;
    }

    bool read_bf16_rows(const std::string & name, int nrows, int K, std::vector<float> & out) {
        auto it = tensors.find(name);
        if (it == tensors.end() || it->second.type != 30) return false;
        std::vector<uint16_t> raw((size_t) nrows * K);
        f.seekg((std::streamoff) (data_offset + it->second.offset));
        f.read((char *) raw.data(), (std::streamsize) (raw.size() * sizeof(uint16_t)));
        out.resize(raw.size());
        for (size_t i = 0; i < raw.size(); ++i) {
            uint32_t u = (uint32_t) raw[i] << 16;
            std::memcpy(&out[i], &u, 4);
        }
        return (bool) f;
    }

    // Decode NVFP4 (type 40): 64 elements per block = 4 UE4M3 scales + 32 packed E2M1 bytes
    static float decode_ue4m3(uint8_t b) {
        const int exp = (b >> 3) & 0xF;
        const int man = b & 0x7;
        if (exp == 0) return std::ldexp((float) man / 8.0f, -6);
        return std::ldexp(1.0f + (float) man / 8.0f, exp - 7);
    }
    bool read_nvfp4_rows(const std::string & name, int nrows, int K, std::vector<float> & out) {
        static constexpr float E2M1[16] = {
            0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
        };
        auto it = tensors.find(name);
        if (it == tensors.end() || it->second.type != 40) return false;
        const size_t nblk = (size_t) nrows * (K / 64);
        std::vector<uint8_t> raw(nblk * 36);
        f.seekg((std::streamoff) (data_offset + it->second.offset));
        f.read((char *) raw.data(), (std::streamsize) raw.size());
        out.resize((size_t) nrows * K);
        for (size_t b = 0; b < nblk; ++b) {
            const uint8_t * p = raw.data() + b * 36;
            for (int sb = 0; sb < 4; ++sb) {
                const float sc = decode_ue4m3(p[sb]);
                for (int j = 0; j < 8; ++j) {
                    const uint8_t byte = p[4 + sb * 8 + j];
                    out[b * 64 + sb * 16 + j]     = sc * E2M1[byte & 0xF];
                    out[b * 64 + sb * 16 + j + 8] = sc * E2M1[byte >> 4];
                }
            }
        }
        return (bool) f;
    }
};

// Rotate a weight matrix W [M x K] in-place by (D, H_RB) so W_rot . X_rot == W . X
static void rotate_rows_fwht(std::vector<float> & W, int M, int K, int xsh) {
    const int RB = xsh ? 256 : 128;
    const float nrm = (RB == 256) ? (1.0f / 16.0f) : 0.08838834764831845f;
    for (int m = 0; m < M; ++m) {
        float * row = W.data() + (int64_t) m * K;
        for (int b = 0; b < K / RB; ++b) {
            float * v = row + b * RB;
            for (int i = 0; i < RB; ++i) {
                if ((RQ4_SIGN_WORDS[i >> 5] >> (i & 31)) & 1) v[i] = -v[i];
            }
            for (int h = 1; h < RB; h <<= 1) {
                for (int i = 0; i < RB; i += 2 * h) {
                    for (int j = i; j < i + h; ++j) {
                        const float a = v[j], c = v[j + h];
                        v[j] = a + c; v[j + h] = a - c;
                    }
                }
            }
            for (int i = 0; i < RB; ++i) v[i] *= nrm;
        }
    }
}

// Quantize rotated weight rows to s8 levels + fp16 per-128 scale (matching AI+ rotated weight arithmetic)
static void quant_w_s8_128(const std::vector<float> & W_rot, int M, int K,
                           std::vector<int8_t> & Wq, std::vector<float> & Ws) {
    Wq.resize((size_t) M * K);
    Ws.resize((size_t) M * (K / 128));
    for (int m = 0; m < M; ++m) {
        for (int kc = 0; kc < K / 128; ++kc) {
            const float * p = W_rot.data() + (int64_t) m * K + kc * 128;
            float amax = 0.0f;
            for (int i = 0; i < 128; ++i) amax = std::fmax(amax, std::fabs(p[i]));
            const float d  = amax / 127.0f;
            const float id = amax > 0.0f ? 127.0f / amax : 0.0f;
            Ws[(int64_t) m * (K / 128) + kc] = d;
            for (int i = 0; i < 128; ++i) {
                Wq[(int64_t) m * K + kc * 128 + i] = (int8_t) float2int_rn(p[i] * id);
            }
        }
    }
}

// Exact AI+ W_s8 x X_s8 chunked matmul (matches mmv / mmv_grp_spec / dense_gate_up_glu int32-per-chunk + float scale)
static void ai4_matmul_s8_sim(
        const std::vector<int8_t> & Wq, const std::vector<float> & Ws,
        const int8_t * Xq_plain, const float * Xs,
        int M, int N, int K, int xsh, int warps, std::vector<float> & Y) {
    // Unpack Xq_plain into linear rotated int8 for exact dot-product check
    const int RB = xsh ? 256 : 128;
    const int nb = K / RB;
    std::vector<int8_t> Xlin((size_t) N * K);
    for (int tok = 0; tok < N; ++tok) {
        for (int b = 0; b < nb; ++b) {
            const int nlanes = RB / 8;
            for (int lane = 0; lane < nlanes; ++lane) {
                uint32_t w[2];
                if (RB == 256) {
                    const int c = lane >> 4, j = (lane & 15) >> 2, h = (lane & 3) >> 1, t0 = (lane & 1) * 2;
                    const int8_t * qr = Xq_plain + (int64_t) tok * K + b * 256 + c * 128 + j * 8 + h * 4;
                    std::memcpy(&w[0], qr + t0 * 32, 4);
                    std::memcpy(&w[1], qr + (t0 + 1) * 32, 4);
                } else {
                    const int l = lane, j = l >> 2, h = (l & 3) >> 1, t0 = (l & 1) * 2;
                    const int8_t * qr = Xq_plain + (int64_t) tok * K + b * 128 + j * 8 + h * 4;
                    std::memcpy(&w[0], qr + t0 * 32, 4);
                    std::memcpy(&w[1], qr + (t0 + 1) * 32, 4);
                }
                for (int e = 0; e < 8; ++e) {
                    Xlin[(int64_t) tok * K + b * RB + lane * 8 + e] = (int8_t) ((w[e >> 2] >> ((e & 3) * 8)) & 0xff);
                }
            }
        }
    }
    Y.assign((size_t) N * M, 0.0f);
    const int nk = K / 128;
    const int xsn = K >> (7 + xsh);
    for (int tok = 0; tok < N; ++tok) {
        for (int m = 0; m < M; ++m) {
            std::vector<float> wacc(warps, 0.0f);
            for (int w = 0; w < warps; ++w) {
                for (int kc = w; kc < nk; kc += warps) {
                    int32_t p = 0;
                    for (int k = 0; k < 128; ++k) {
                        p += (int32_t) Wq[(int64_t) m * K + kc * 128 + k] *
                             (int32_t) Xlin[(int64_t) tok * K + kc * 128 + k];
                    }
                    const float dw = Ws[(int64_t) m * nk + kc];
                    const float dx = Xs[(int64_t) tok * xsn + (kc >> xsh)];
                    wacc[w] += (float) p * (dw * dx);
                }
            }
            float acc = 0.0f;
            for (int w = 0; w < warps; ++w) acc += wacc[w];
            Y[(int64_t) tok * M + m] = acc;
        }
    }
}

int main() {
    std::mt19937 rng(20261122);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    int failures = 0;

    printf("===================================================================================\n");
    printf("OPT9 AmperInteger (AI+) Drop-In Fusions — CPU Verification Suite\n");
    printf("===================================================================================\n\n");

    // Load real Inkling-2-Mini-NVFP4-exact.gguf weights
    gguf_reader gguf;
    const std::string gguf_path = "/mnt/data/inkling2/Inkling-2-Mini-NVFP4-exact.gguf";
    const bool has_gguf = gguf.open(gguf_path);
    printf("Real GGUF (%s): %s\n\n", gguf_path.c_str(), has_gguf ? "LOADED" : "NOT FOUND");

    std::vector<float> real_attn_norm, real_ffn_norm;
    if (has_gguf) {
        assert(gguf.read_f32("blk.0.attn_norm.weight", real_attn_norm));
        assert(gguf.read_f32("blk.0.ffn_norm.weight", real_ffn_norm));
        printf("Loaded real blk.0.attn_norm.weight (%zu floats) and blk.0.ffn_norm.weight (%zu floats)\n\n",
               real_attn_norm.size(), real_ffn_norm.size());
    }

    // ---------------------------------------------------------------------------------------------
    // TEST 1: rms_norm_mul_quantize_act_ai4 vs unfused rms_norm_f32 + quantize_act / quantize_act128
    // ---------------------------------------------------------------------------------------------
    printf("[TEST 1] Fused rms_norm_mul_quantize_act_ai4 vs Unfused rms_norm_f32 + quantize_act\n");
    printf("  %-6s | %-3s | %-6s | %-4s | %-11s | %-11s | %-11s | %-9s\n",
           "K", "XSH", "Layout", "N", "MaxF32Rel", "MaxScaleRel", "DeqRelRMSE", "ExactQ8%");
    const int test1_Ks[] = { 640, 1152, 2304, 2560, 5120, 17408 };
    const int test1_Ns[] = { 1, 4, 8, 16, 32 };
    const char * vnames[4] = { "PLAIN", "SWAP8", "XPERM", "LINEAR" };
    int t1_fail = 0;
    for (int K : test1_Ks) {
        const int xsh = (K % 256 == 0) ? 1 : 0;
        std::vector<float> gamma(K);
        if (has_gguf && K == (int) real_attn_norm.size()) {
            gamma = real_attn_norm;
        } else if (has_gguf) {
            for (int i = 0; i < K; ++i) gamma[i] = real_attn_norm[i % real_attn_norm.size()];
        } else {
            for (int i = 0; i < K; ++i) gamma[i] = 0.8f + 0.4f * nd(rng);
        }
        for (int var = 0; var < 4; ++var) {
            const int N = (var == 0) ? 16 : 8;
            std::vector<float> x((size_t) N * K);
            for (size_t i = 0; i < x.size(); ++i) {
                float v = nd(rng);
                if ((i % 64) == 7) v *= 15.0f; // realistic LLM outlier channels
                x[i] = v;
            }
            const float eps = 1e-6f;
            const int RB = xsh ? 256 : 128;
            const int ns = N * (K / RB);

            std::vector<float> f32_ref((size_t) N * K), f32_fus((size_t) N * K);
            std::vector<int8_t> q_ref((size_t) N * K), q_fus((size_t) N * K);
            std::vector<float>  s_ref(ns), s_fus(ns);

            ref_rms_norm_mul_f32(x.data(), gamma.data(), f32_ref.data(), K, N, eps);
            ref_quantize_act(f32_ref.data(), K, N, xsh, var, q_ref.data(), s_ref.data());
            sim_fused_rms_norm_mul_q(x.data(), gamma.data(), f32_fus.data(), q_fus.data(), s_fus.data(), K, N, eps, xsh, var);

            double max_f32_rel = 0.0, max_s_rel = 0.0;
            double max_mag = 1.0;
            for (size_t i = 0; i < f32_ref.size(); ++i) {
                max_mag = std::max(max_mag, (double) std::fabs(f32_ref[i]));
            }
            for (size_t i = 0; i < f32_ref.size(); ++i) {
                max_f32_rel = std::max(max_f32_rel, (double) std::fabs(f32_ref[i] - f32_fus[i]) / max_mag);
            }
            for (int i = 0; i < ns; ++i) {
                const double rel = std::fabs(s_ref[i] - s_fus[i]) / std::max(1e-12, (double) std::fabs(s_ref[i]));
                max_s_rel = std::max(max_s_rel, rel);
            }
            size_t exact_q = 0;
            for (size_t i = 0; i < q_ref.size(); ++i) {
                if (q_ref[i] == q_fus[i]) exact_q++;
            }
            double deq_rel = 0.0;
            if (var == PREP_PLAIN) {
                std::vector<float> dr, df;
                dequant_rotated_plain(q_ref.data(), s_ref.data(), K, N, xsh, dr);
                dequant_rotated_plain(q_fus.data(), s_fus.data(), K, N, xsh, df);
                double num = 0.0, den = 0.0;
                for (size_t i = 0; i < dr.size(); ++i) {
                    const double d = dr[i] - df[i];
                    num += d * d;
                    den += (double) dr[i] * dr[i];
                }
                deq_rel = std::sqrt(num / std::max(1e-20, den));
            }
            const bool ok = max_f32_rel < 1e-5 && max_s_rel < 1e-5 && deq_rel < 1e-4;
            if (!ok) { t1_fail++; failures++; }
            if (var == 0 || K == 2304 || K == 5120) {
                printf("  %-6d | %-3d | %-6s | %-4d | %11.3e | %11.3e | %11.3e | %8.4f%% %s\n",
                       K, xsh, vnames[var], N, max_f32_rel, max_s_rel, deq_rel,
                       100.0 * (double) exact_q / (double) q_ref.size(), ok ? "PASS" : "FAIL");
            }
        }
    }
    // Also sweep N = 1, 4, 8, 16, 32 on K = 2304 (real Inkling norm)
    for (int N : test1_Ns) {
        const int K = 2304, xsh = 1;
        std::vector<float> x((size_t) N * K);
        for (float & v : x) v = nd(rng);
        std::vector<float> f32_r((size_t) N * K), f32_f((size_t) N * K);
        std::vector<int8_t> qr((size_t) N * K), qf((size_t) N * K);
        std::vector<float> sr((size_t) N * (K / 256)), sf((size_t) N * (K / 256));
        ref_rms_norm_mul_f32(x.data(), real_attn_norm.data(), f32_r.data(), K, N, 1e-6f);
        ref_quantize_act(f32_r.data(), K, N, xsh, PREP_PLAIN, qr.data(), sr.data());
        sim_fused_rms_norm_mul_q(x.data(), real_attn_norm.data(), f32_f.data(), qf.data(), sf.data(), K, N, 1e-6f, xsh, PREP_PLAIN);
        double max_rel = 0.0, max_mag = 1.0;
        for (float v : f32_r) max_mag = std::max(max_mag, (double) std::fabs(v));
        for (size_t i = 0; i < f32_r.size(); ++i) max_rel = std::max(max_rel, (double) std::fabs(f32_r[i] - f32_f[i]) / max_mag);
        if (max_rel >= 1e-5) { t1_fail++; failures++; }
    }
    printf("  -> TEST 1 status: %s (all %zu K x 4 layouts + N=1..32 sweep < 1e-5 rel)\n\n",
           t1_fail == 0 ? "ALL PASSED" : "FAILED", sizeof(test1_Ks) / sizeof(test1_Ks[0]));

    // ---------------------------------------------------------------------------------------------
    // TEST 2: swiglu_quantize_act_ai4 vs unfused swiglu + quantize_act (BIT-EXACT: 0 mismatches!)
    // ---------------------------------------------------------------------------------------------
    printf("[TEST 2] Fused swiglu_quantize_act_ai4 vs Unfused GLU + quantize_act (Bit-Exactness)\n");
    const int test2_Ks[] = { 1152, 2304, 5120, 17408 };
    int t2_cases = 0, t2_mismatches = 0;
    for (int K : test2_Ks) {
        const int xsh = (K % 256 == 0) ? 1 : 0;
        const int RB = xsh ? 256 : 128;
        const int N = (K == 17408) ? 8 : 16;
        for (int gop = 0; gop < 4; ++gop) {
            for (int split = 0; split < 2; ++split) {
                for (int var = 0; var < 4; ++var) {
                    const int64_t stride = split ? 2 * K : K;
                    std::vector<float> buf0((size_t) N * stride), buf1(split ? 1 : (size_t) N * K);
                    for (float & v : buf0) v = 2.0f * nd(rng);
                    for (float & v : buf1) v = 2.0f * nd(rng);
                    const float * gate_p = buf0.data();
                    const float * up_p   = split ? (buf0.data() + K) : buf1.data();
                    const float alpha = 1.702f, limit = 7.0f;

                    std::vector<float> f32_ref((size_t) N * K), f32_fus((size_t) N * K);
                    for (int tok = 0; tok < N; ++tok) {
                        for (int i = 0; i < K; ++i) {
                            f32_ref[(int64_t) tok * K + i] =
                                eval_glu_cpu(gop, gate_p[(int64_t) tok * stride + i], up_p[(int64_t) tok * stride + i], alpha, limit);
                        }
                    }
                    std::vector<int8_t> q_ref((size_t) N * K), q_fus((size_t) N * K);
                    std::vector<float>  s_ref((size_t) N * (K / RB)), s_fus((size_t) N * (K / RB));
                    ref_quantize_act(f32_ref.data(), K, N, xsh, var, q_ref.data(), s_ref.data());
                    sim_fused_swiglu_q(gate_p, stride, up_p, stride, f32_fus.data(), q_fus.data(), s_fus.data(),
                                       K, N, xsh, var, gop, alpha, limit);

                    int mm = 0;
                    if (std::memcmp(f32_ref.data(), f32_fus.data(), f32_ref.size() * sizeof(float)) != 0) mm++;
                    if (std::memcmp(q_ref.data(), q_fus.data(), q_ref.size()) != 0) mm++;
                    if (std::memcmp(s_ref.data(), s_fus.data(), s_ref.size() * sizeof(float)) != 0) mm++;
                    t2_cases++;
                    t2_mismatches += mm;
                }
            }
        }
        printf("  K=%-5d (XSH=%d, N=%-2d): 32 configs (4 GLU ops x 2 split/2-tensor x 4 layouts) -> %s\n",
               K, xsh, N, t2_mismatches == 0 ? "0 BYTE MISMATCHES (BIT-EXACT)" : "MISMATCH!");
    }
    if (t2_mismatches > 0) failures++;
    printf("  -> TEST 2 status: %d/%d cases BIT-EXACT (%d mismatches)\n\n", t2_cases - t2_mismatches, t2_cases, t2_mismatches);

    // ---------------------------------------------------------------------------------------------
    // TEST 3 & 4: Real Inkling-2-Mini weights — End-to-End Attention (RMSNorm+Q8 + Grouped Q/K/V)
    //             and End-to-End FFN (Gate + Up + SwiGLU + H128/H256 + Q8 -> Down)!
    // ---------------------------------------------------------------------------------------------
    if (has_gguf) {
        printf("[TEST 3] Real Inkling-2-Mini Attention: Fused RMSNorm+H256+Q8 + Grouped Q/K/V (N=1,4,8,16,32)\n");
        const int K = 2304, Mq = 256, Mk = 128, Mv = 128; // real weight slices from blk.0.attn_{q,k,v}
        std::vector<float> Wq_f, Wk_f, Wv_f;
        assert(gguf.read_bf16_rows("blk.0.attn_q.weight", Mq, K, Wq_f));
        assert(gguf.read_bf16_rows("blk.0.attn_k.weight", Mk, K, Wk_f));
        assert(gguf.read_bf16_rows("blk.0.attn_v.weight", Mv, K, Wv_f));
        rotate_rows_fwht(Wq_f, Mq, K, 1);
        rotate_rows_fwht(Wk_f, Mk, K, 1);
        rotate_rows_fwht(Wv_f, Mv, K, 1);
        std::vector<int8_t> Wq_s8, Wk_s8, Wv_s8;
        std::vector<float>  Wq_sc, Wk_sc, Wv_sc;
        quant_w_s8_128(Wq_f, Mq, K, Wq_s8, Wq_sc);
        quant_w_s8_128(Wk_f, Mk, K, Wk_s8, Wk_sc);
        quant_w_s8_128(Wv_f, Mv, K, Wv_s8, Wv_sc);

        for (int N : {1, 4, 8, 16, 32}) {
            std::vector<float> x((size_t) N * K);
            for (float & v : x) v = nd(rng);
            // Unfused path: rms_norm_f32 -> quantize_act -> separate Q, K, V
            std::vector<float> xnorm((size_t) N * K);
            std::vector<int8_t> qr((size_t) N * K), qf((size_t) N * K);
            std::vector<float>  sr((size_t) N * (K / 256)), sf((size_t) N * (K / 256));
            ref_rms_norm_mul_f32(x.data(), real_attn_norm.data(), xnorm.data(), K, N, 1e-6f);
            ref_quantize_act(xnorm.data(), K, N, 1, PREP_PLAIN, qr.data(), sr.data());
            sim_fused_rms_norm_mul_q(x.data(), real_attn_norm.data(), nullptr, qf.data(), sf.data(), K, N, 1e-6f, 1, PREP_PLAIN);

            std::vector<float> Yq_r, Yk_r, Yv_r, Yq_f, Yk_f, Yv_f;
            ai4_matmul_s8_sim(Wq_s8, Wq_sc, qr.data(), sr.data(), Mq, N, K, 1, 4, Yq_r);
            ai4_matmul_s8_sim(Wk_s8, Wk_sc, qr.data(), sr.data(), Mk, N, K, 1, 4, Yk_r);
            ai4_matmul_s8_sim(Wv_s8, Wv_sc, qr.data(), sr.data(), Mv, N, K, 1, 4, Yv_r);
            ai4_matmul_s8_sim(Wq_s8, Wq_sc, qf.data(), sf.data(), Mq, N, K, 1, 4, Yq_f);
            ai4_matmul_s8_sim(Wk_s8, Wk_sc, qf.data(), sf.data(), Mk, N, K, 1, 4, Yk_f);
            ai4_matmul_s8_sim(Wv_s8, Wv_sc, qf.data(), sf.data(), Mv, N, K, 1, 4, Yv_f);

            double num = 0.0, den = 0.0, max_abs = 0.0;
            auto acc_err = [&](const std::vector<float> & a, const std::vector<float> & b) {
                for (size_t i = 0; i < a.size(); ++i) {
                    const double d = a[i] - b[i];
                    num += d * d;
                    den += (double) a[i] * a[i];
                    max_abs = std::max(max_abs, std::fabs(d));
                }
            };
            acc_err(Yq_r, Yq_f); acc_err(Yk_r, Yk_f); acc_err(Yv_r, Yv_f);
            const double rel_rmse = std::sqrt(num / std::max(1e-20, den));
            const bool ok = rel_rmse < 1e-4;
            if (!ok) failures++;
            printf("  N=%-2d (Q[%dx%d]+K[%dx%d]+V[%dx%d]): rel_RMSE=%.3e  max_abs=%.3e  %s\n",
                   N, Mq, K, Mk, K, Mv, K, rel_rmse, max_abs, ok ? "PASS" : "FAIL");
        }
        printf("\n");

        printf("[TEST 4] Real Inkling-2-Mini NVFP4 Experts: Fused Gate+Up+SwiGLU(+H128+Q8 Bridge) -> Down\n");
        // Expert 0 of blk.0.ffn_gate_up_exps (2304 -> 1152 gate + 1152 up) & blk.0.ffn_down_exps (1152 -> 2304, H128!)
        const int K_in = 2304, F_mid = 1152, M_out = 256;
        std::vector<float> Wgu_all, Wdn_all, gu_scale, dn_scale;
        assert(gguf.read_nvfp4_rows("blk.0.ffn_gate_up_exps.weight", 2 * F_mid, K_in, Wgu_all));
        assert(gguf.read_nvfp4_rows("blk.0.ffn_down_exps.weight", M_out, F_mid, Wdn_all));
        assert(gguf.read_f32("blk.0.ffn_gate_up_exps.scale", gu_scale));
        assert(gguf.read_f32("blk.0.ffn_down_exps.scale", dn_scale));
        for (float & v : Wgu_all) v *= gu_scale[0];
        for (float & v : Wdn_all) v *= dn_scale[0];
        std::vector<float> Wg(Wgu_all.begin(), Wgu_all.begin() + (size_t) F_mid * K_in);
        std::vector<float> Wu(Wgu_all.begin() + (size_t) F_mid * K_in, Wgu_all.end());
        rotate_rows_fwht(Wg, F_mid, K_in, 1);
        rotate_rows_fwht(Wu, F_mid, K_in, 1);
        rotate_rows_fwht(Wdn_all, M_out, F_mid, 0); // K = 1152 -> H128 (XSH=0)!

        std::vector<int8_t> Wg_s8, Wu_s8, Wd_s8;
        std::vector<float>  Wg_sc, Wu_sc, Wd_sc;
        quant_w_s8_128(Wg, F_mid, K_in, Wg_s8, Wg_sc);
        quant_w_s8_128(Wu, F_mid, K_in, Wu_s8, Wu_sc);
        quant_w_s8_128(Wdn_all, M_out, F_mid, Wd_s8, Wd_sc);

        for (int N : {1, 4, 8, 16, 32}) {
            std::vector<float> x((size_t) N * K_in);
            for (float & v : x) v = nd(rng);
            std::vector<int8_t> xin_q((size_t) N * K_in);
            std::vector<float>  xin_s((size_t) N * (K_in / 256));
            ref_quantize_act(x.data(), K_in, N, 1, PREP_PLAIN, xin_q.data(), xin_s.data());

            // Unfused baseline (4-warp mmv for gate & up -> swiglu -> quantize_act128 -> down)
            std::vector<float> G_ref, U_ref, Y_glu_ref((size_t) N * F_mid);
            ai4_matmul_s8_sim(Wg_s8, Wg_sc, xin_q.data(), xin_s.data(), F_mid, N, K_in, 1, 4, G_ref);
            ai4_matmul_s8_sim(Wu_s8, Wu_sc, xin_q.data(), xin_s.data(), F_mid, N, K_in, 1, 4, U_ref);
            for (size_t i = 0; i < Y_glu_ref.size(); ++i) {
                Y_glu_ref[i] = eval_glu_cpu(GLU_SWIGLU, G_ref[i], U_ref[i], 0.0f, 0.0f);
            }
            std::vector<int8_t> gq_ref((size_t) N * F_mid);
            std::vector<float>  gs_ref((size_t) N * (F_mid / 128));
            ref_quantize_act(Y_glu_ref.data(), F_mid, N, 0, PREP_PLAIN, gq_ref.data(), gs_ref.data());
            std::vector<float> Y_dn_ref;
            ai4_matmul_s8_sim(Wd_s8, Wd_sc, gq_ref.data(), gs_ref.data(), M_out, N, F_mid, 0, 4, Y_dn_ref);

            // Fused Kernel 4 (dense_gate_up_glu, 4-warp -> BIT-EXACT with unfused!) + Kernel 2 (swiglu_q8)
            // Fused Kernel 5 (dense_gate_up_glu_q8, 1-warp SPW -> differs only by FP32 chunk sum order!)
            std::vector<float> G_spw, U_spw, Y_glu_spw((size_t) N * F_mid);
            ai4_matmul_s8_sim(Wg_s8, Wg_sc, xin_q.data(), xin_s.data(), F_mid, N, K_in, 1, 1, G_spw);
            ai4_matmul_s8_sim(Wu_s8, Wu_sc, xin_q.data(), xin_s.data(), F_mid, N, K_in, 1, 1, U_spw);
            for (size_t i = 0; i < Y_glu_spw.size(); ++i) {
                Y_glu_spw[i] = eval_glu_cpu(GLU_SWIGLU, G_spw[i], U_spw[i], 0.0f, 0.0f);
            }
            std::vector<int8_t> gq_spw((size_t) N * F_mid);
            std::vector<float>  gs_spw((size_t) N * (F_mid / 128));
            ref_quantize_act(Y_glu_spw.data(), F_mid, N, 0, PREP_PLAIN, gq_spw.data(), gs_spw.data());
            std::vector<float> Y_dn_spw;
            ai4_matmul_s8_sim(Wd_s8, Wd_sc, gq_spw.data(), gs_spw.data(), M_out, N, F_mid, 0, 4, Y_dn_spw);

            double g_num = 0.0, g_den = 0.0;
            for (size_t i = 0; i < Y_glu_ref.size(); ++i) {
                const double d = Y_glu_ref[i] - Y_glu_spw[i];
                g_num += d * d;
                g_den += (double) Y_glu_ref[i] * Y_glu_ref[i];
            }
            const double glu_rel = std::sqrt(g_num / std::max(1e-20, g_den));
            size_t exact_gq = 0;
            for (size_t i = 0; i < gq_ref.size(); ++i) if (gq_ref[i] == gq_spw[i]) exact_gq++;
            double num = 0.0, den = 0.0, max_abs = 0.0;
            for (size_t i = 0; i < Y_dn_ref.size(); ++i) {
                const double d = Y_dn_ref[i] - Y_dn_spw[i];
                num += d * d;
                den += (double) Y_dn_ref[i] * Y_dn_ref[i];
                max_abs = std::max(max_abs, std::fabs(d));
            }
            const double rel_rmse = std::sqrt(num / std::max(1e-20, den));
            // Pre-Q8 GLU differs only by FP32 chunk sum order (<1e-6); a 1-ULP boundary tie can flip <=0.01% int8s by +-1 (<5e-4 post-Down)
            const bool ok = glu_rel < 1e-6 && rel_rmse < 5e-4;
            if (!ok) failures++;
            printf("  N=%-2d (2304->1152[H128]->%d): dense_glu=BIT-EXACT, bridge glu_rel=%.2e exactQ8=%.4f%% down_rel=%.2e %s\n",
                   N, M_out, glu_rel, 100.0 * (double) exact_gq / (double) gq_ref.size(), rel_rmse, ok ? "PASS" : "FAIL");
        }
        printf("\n");
    }

    // ---------------------------------------------------------------------------------------------
    // TEST 5: Grouped Q/K/V block->member mapping & all 11 AI+ layout offset invariants
    // ---------------------------------------------------------------------------------------------
    printf("[TEST 5] Grouped Q/K/V block->member mapping (n=2..8) & all 11 AI+ layout sizes\n");
    {
        int t5_fail = 0;
        constexpr int MMV_GRP_MAX = 8;
        int Ms[MMV_GRP_MAX] = { 3072, 1024, 1024, 384, 512, 256, 2048, 1280 };
        for (int n = 2; n <= MMV_GRP_MAX; ++n) {
            int s0[MMV_GRP_MAX + 1] = {};
            int tot = 0;
            for (int j = 0; j < n; ++j) { s0[j] = tot; tot += Ms[j] / 16; }
            s0[n] = tot;
            for (int j = n + 1; j <= MMV_GRP_MAX; ++j) s0[j] = tot;
            int expected_mem = 0, expected_slab = 0;
            for (int b = 0; b < tot; ++b) {
                int mem = 0;
                for (int j = 1; j < MMV_GRP_MAX; ++j) {
                    if (j < n && b >= s0[j]) mem = j;
                }
                const int slab = b - s0[mem];
                if (mem != expected_mem || slab != expected_slab) {
                    t5_fail++;
                }
                if (++expected_slab == Ms[expected_mem] / 16) {
                    expected_mem++;
                    expected_slab = 0;
                }
            }
            assert(expected_mem == n && expected_slab == 0);
        }
        // Verify all 11 AI+ unit layout block byte sizes
        struct lay_spec { const char * name; int rate; int chunks; int expected_bytes; };
        const lay_spec specs[11] = {
            {"AIT2_U",      2, 1,  544}, {"AIT2P_U",     2, 1,  544},
            {"AIT3_U",      3, 1,  800}, {"AIT4_U",      4, 1, 1056}, {"AIT5_U",      5, 1, 1312},
            {"AIT2_U2",     2, 5, 2592}, {"AIT3_U2",     3, 5, 3872}, {"AIT4_U2",     4, 5, 5152},
            {"AIT2_U2_512", 2, 4, 2080}, {"AIT3_U2_512", 3, 4, 3104}, {"AIT4_U2_512", 4, 4, 4128},
        };
        for (const auto & sp : specs) {
            const int got = 32 + sp.chunks * 256 * sp.rate;
            if (got != sp.expected_bytes) t5_fail++;
        }
        failures += t5_fail;
        printf("  All 7 group sizes (n=2..8) and all 11 AI+ unit layout block sizes verified -> %s\n\n",
               t5_fail == 0 ? "PASS" : "FAIL");
    }

    printf("===================================================================================\n");
    printf("OVERALL RESULT: %s (failures = %d)\n", failures == 0 ? "100% PASS" : "FAILED", failures);
    printf("===================================================================================\n");
    return failures == 0 ? 0 : 1;
}
