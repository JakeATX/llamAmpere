#pragma once

// KVarN-style sealed KV records: scalar reference codec shared by the CPU backend, the tests and
// the CUDA kernels' host side. Math follows github.com/huawei-csl/KVarN (variance balancing in the
// log domain, asymmetric per-row RTN, scale absorption) with one deliberate divergence: the balancing
// runs a fixed number of iterations and keeps the FINAL scales instead of the reference's best-so-far
// selection, which flips on float32 last bits once the imbalance reaches its fixed point and would make
// CPU and CUDA sealers disagree. The record layout is this project's own (token-major payloads for both
// K and V), see ggml_kvarn_seal in ggml.h.

#include "ggml.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace ggml_kvarn {

struct layout {
    int    D, G, bits_k, bits_v;
    size_t k_row;      // bytes per token row of the K payload
    size_t v_row;      // bytes per token row of the V payload
    size_t k_payload;  // offsets
    size_t v_payload;
    size_t k_scale;    // fp16[D]
    size_t k_zero;     // fp16[D]
    size_t k_tok;      // fp16[G]
    size_t v_ch;       // fp16[D]
    size_t v_scale;    // fp16[G]
    size_t v_zero;     // fp16[G]
    size_t bytes;
};

inline layout make_layout(int D, int G, int bits_k, int bits_v) {
    layout l;
    l.D = D; l.G = G; l.bits_k = bits_k; l.bits_v = bits_v;
    l.k_row     = (size_t) D * bits_k / 8;
    l.v_row     = (size_t) D * bits_v / 8;
    l.k_payload = 0;
    l.v_payload = l.k_payload + l.k_row * G;
    l.k_scale   = l.v_payload + l.v_row * G;
    l.k_zero    = l.k_scale + 2 * (size_t) D;
    l.k_tok     = l.k_zero  + 2 * (size_t) D;
    l.v_ch      = l.k_tok   + 2 * (size_t) G;
    l.v_scale   = l.v_ch    + 2 * (size_t) D;
    l.v_zero    = l.v_scale + 2 * (size_t) G;
    l.bytes     = l.v_zero  + 2 * (size_t) G;
    return l;
}

// orthonormal Sylvester Hadamard, in place, n a power of two
inline void hadamard(float * x, int n) {
    for (int h = 1; h < n; h <<= 1) {
        for (int i = 0; i < n; i += 2*h) {
            for (int j = i; j < i + h; ++j) {
                const float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    const float s = 1.0f / sqrtf((float) n);
    for (int i = 0; i < n; ++i) {
        x[i] *= s;
    }
}

// Reduction contract shared with the CUDA sealer (kvarn-seal.cu) so both produce identical records:
// a vector of n values (n % KVARN_RED_LANES == 0) is split into KVARN_RED_LANES contiguous chunks, each
// chunk is summed sequentially in double, and the chunk partials are combined as a balanced xor
// butterfly ((p0+p1)+(p2+p3))+((p4+p5)+(p6+p7)). The second moment uses an explicit double fma so
// neither compiler can change the contraction. Everything else is IEEE float/double arithmetic.
#define KVARN_RED_LANES 8

inline double reduce_lanes(const double * p) {
    double q[KVARN_RED_LANES];
    for (int i = 0; i < KVARN_RED_LANES; ++i) q[i] = p[i];
    for (int w = 1; w < KVARN_RED_LANES; w <<= 1) {
        for (int i = 0; i < KVARN_RED_LANES; i += 2*w) {
            q[i] = q[i] + q[i + w];
        }
    }
    return q[0];
}

// sample standard deviation (N-1) of a strided float sequence, see the reduction contract above
inline float sample_std(const float * x, int n, size_t stride) {
    const int chunk = n / KVARN_RED_LANES;
    double p[KVARN_RED_LANES];
    for (int l = 0; l < KVARN_RED_LANES; ++l) {
        double acc = 0.0;
        for (int i = 0; i < chunk; ++i) {
            acc += (double) x[(size_t) (l*chunk + i)*stride];
        }
        p[l] = acc;
    }
    const double mean = reduce_lanes(p) / n;
    for (int l = 0; l < KVARN_RED_LANES; ++l) {
        double acc = 0.0;
        for (int i = 0; i < chunk; ++i) {
            const double d = (double) x[(size_t) (l*chunk + i)*stride] - mean;
            acc = fma(d, d, acc);
        }
        p[l] = acc;
    }
    const double m2 = reduce_lanes(p);
    return (float) sqrt(m2 / (n > 1 ? (n - 1) : 1));
}

// log-domain variance balancing of a row-major [R][C] tile: exactly `iters` alternating column/row
// passes, final scales s_row[R], s_col[C] such that cur = M / s_col / s_row has balanced row and
// column spreads (no best-so-far selection, see the header comment). R and C must be multiples of
// KVARN_RED_LANES.
inline void balance(const float * M, int R, int C, int iters, float * s_row, float * s_col, float * cur /* [R*C] scratch */) {
    std::vector<double> log_sc(C, 0.0), log_sr(R, 0.0); // log domain in double (CUDA fast-math expf/logf differ, double exp/log do not)
    for (int c = 0; c < C; ++c) s_col[c] = 1.0f;
    for (int r = 0; r < R; ++r) s_row[r] = 1.0f;

    auto recompute = [&]() {
        for (int r = 0; r < R; ++r) {
            for (int c = 0; c < C; ++c) {
                cur[(size_t) r*C + c] = M[(size_t) r*C + c] / s_col[c] / s_row[r];
            }
        }
    };

    recompute();
    for (int it = 0; it < iters; ++it) {
        for (int c = 0; c < C; ++c) {
            const float sd = std::min(std::max(sample_std(cur + c, R, C), 1e-3f), 1e3f);
            log_sc[c] = std::min(std::max(log_sc[c] + log((double) sd), -0.3), 10.0);
            s_col[c] = (float) exp(log_sc[c]);
        }
        recompute();
        for (int r = 0; r < R; ++r) {
            const float sd = std::min(std::max(sample_std(cur + (size_t) r*C, C, 1), 1e-3f), 1e3f);
            log_sr[r] = std::min(std::max(log_sr[r] + log((double) sd), -0.3), 10.0);
            s_row[r] = (float) exp(log_sr[r]);
        }
        recompute();
    }
}

inline uint32_t rtn_code(float x, float lo, float step, uint32_t qmax) {
    const float y = (x - lo) / step;
    float r = nearbyintf(y); // ties to even under the default rounding mode
    if (r < 0.0f) r = 0.0f;
    if (r > (float) qmax) r = (float) qmax;
    return (uint32_t) r;
}

// Physical slot of value idx within a payload row. 4-bit rows interleave each 8-value word so that the CUDA
// decoder pulls two adjacent values with one mask: nibble k of a word holds value 2k (k < 4) or 2(k-4)+1
// (k >= 4), i.e. values (0,1) sit in nibbles (0,4), (2,3) in (1,5), (4,5) in (2,6), (6,7) in (3,7).
// Other bit widths are a plain little-endian bit stream (value idx at bit idx*bits).
inline int phys_slot(int idx, int bits) {
    if (bits != 4) {
        return idx;
    }
    const int word = idx >> 3, k = idx & 7;
    return word*8 + ((k & 1) ? 4 + (k >> 1) : (k >> 1));
}

// value idx of a row at bit phys_slot(idx)*bits, little-endian across bytes
inline void pack_value(uint8_t * row, int idx, int bits, uint32_t v) {
    const uint32_t bit = (uint32_t) phys_slot(idx, bits) * bits;
    uint32_t byte = bit >> 3;
    const uint32_t shift = bit & 7;
    uint32_t w = (v & ((1u << bits) - 1)) << shift;
    row[byte] |= (uint8_t) w;
    if (shift + bits > 8) {
        row[byte + 1] |= (uint8_t) (w >> 8);
    }
}

inline uint32_t unpack_value(const uint8_t * row, int idx, int bits) {
    const uint32_t bit = (uint32_t) phys_slot(idx, bits) * bits;
    const uint32_t byte = bit >> 3;
    const uint32_t shift = bit & 7;
    uint32_t w = row[byte];
    if (shift + bits > 8) {
        w |= (uint32_t) row[byte + 1] << 8;
    }
    return (w >> shift) & ((1u << bits) - 1);
}

inline void put_half(uint8_t * rec, size_t off, float v) {
    const ggml_fp16_t h = GGML_FP32_TO_FP16(v);
    memcpy(rec + off, &h, 2);
}
inline float get_half(const uint8_t * rec, size_t off) {
    ggml_fp16_t h;
    memcpy(&h, rec + off, 2);
    return GGML_FP16_TO_FP32(h);
}

// Seal one (head, group): K and V are the rotated fp16 rows, row t at K + t*row_stride (elements),
// D contiguous values per row. Writes l.bytes bytes to rec.
inline void seal_group(const ggml_fp16_t * K, const ggml_fp16_t * V, size_t row_stride, const layout & l, int iters, uint8_t * rec) {
    const int D = l.D, G = l.G;
    std::vector<float> tile((size_t) D*G), bal((size_t) D*G), s_row(std::max(D, G)), s_col(std::max(D, G));
    memset(rec, 0, l.bytes);

    // K: tile [D][G] (rows = channels, cols = tokens)
    for (int t = 0; t < G; ++t) {
        for (int d = 0; d < D; ++d) {
            tile[(size_t) d*G + t] = GGML_FP16_TO_FP32(K[t*row_stride + d]);
        }
    }
    balance(tile.data(), D, G, iters, s_row.data(), s_col.data(), bal.data());
    {
        const uint32_t qmax = (1u << l.bits_k) - 1;
        for (int d = 0; d < D; ++d) {
            const float * row = bal.data() + (size_t) d*G;
            float lo = row[0], hi = row[0];
            for (int t = 1; t < G; ++t) { lo = std::min(lo, row[t]); hi = std::max(hi, row[t]); }
            const float step = std::max((hi - lo) / (float) qmax, 1e-10f);
            put_half(rec, l.k_scale + 2*d, s_row[d] * step);
            put_half(rec, l.k_zero  + 2*d, s_row[d] * lo);
            for (int t = 0; t < G; ++t) {
                pack_value(rec + l.k_payload + (size_t) t*l.k_row, d, l.bits_k, rtn_code(row[t], lo, step, qmax));
            }
        }
        for (int t = 0; t < G; ++t) {
            put_half(rec, l.k_tok + 2*t, s_col[t]);
        }
    }

    // V: tile [G][D] (rows = tokens, cols = channels)
    for (int t = 0; t < G; ++t) {
        for (int d = 0; d < D; ++d) {
            tile[(size_t) t*D + d] = GGML_FP16_TO_FP32(V[t*row_stride + d]);
        }
    }
    balance(tile.data(), G, D, iters, s_row.data(), s_col.data(), bal.data());
    {
        const uint32_t qmax = (1u << l.bits_v) - 1;
        for (int t = 0; t < G; ++t) {
            const float * row = bal.data() + (size_t) t*D;
            float lo = row[0], hi = row[0];
            for (int d = 1; d < D; ++d) { lo = std::min(lo, row[d]); hi = std::max(hi, row[d]); }
            const float step = std::max((hi - lo) / (float) qmax, 1e-10f);
            put_half(rec, l.v_scale + 2*t, s_row[t] * step);
            put_half(rec, l.v_zero  + 2*t, s_row[t] * lo);
            for (int d = 0; d < D; ++d) {
                pack_value(rec + l.v_payload + (size_t) t*l.v_row, d, l.bits_v, rtn_code(row[d], lo, step, qmax));
            }
        }
        for (int d = 0; d < D; ++d) {
            put_half(rec, l.v_ch + 2*d, s_col[d]);
        }
    }
}

// rotated-domain reconstruction of token t of a record
inline void decode_k_row(const uint8_t * rec, const layout & l, int t, float * out) {
    const float tok = get_half(rec, l.k_tok + 2*t);
    const uint8_t * row = rec + l.k_payload + (size_t) t*l.k_row;
    for (int d = 0; d < l.D; ++d) {
        const float q = (float) unpack_value(row, d, l.bits_k);
        out[d] = (q * get_half(rec, l.k_scale + 2*d) + get_half(rec, l.k_zero + 2*d)) * tok;
    }
}
inline void decode_v_row(const uint8_t * rec, const layout & l, int t, float * out) {
    const float sc = get_half(rec, l.v_scale + 2*t);
    const float zp = get_half(rec, l.v_zero  + 2*t);
    const uint8_t * row = rec + l.v_payload + (size_t) t*l.v_row;
    for (int d = 0; d < l.D; ++d) {
        const float q = (float) unpack_value(row, d, l.bits_v);
        out[d] = (q * sc + zp) * get_half(rec, l.v_ch + 2*d);
    }
}

} // namespace ggml_kvarn
