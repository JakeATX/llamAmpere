#include "kvarn-seal.cuh"
#include "../ggml-kvarn.h"

// One block per (record, tile): blockIdx.x = 2*record + is_V. The fp16 tile lives in shared memory
// token-major ([G][D], exactly the source rows); the balanced value cur(r, c) = M(r, c) / s_col[c] / s_row[r]
// is recomputed on the fly with IEEE division. K tiles use rows = channels (r = d, c = t), V tiles rows =
// tokens (r = t, c = d), so for both the per-row quantities (lo, step -> scale/zero) index the row and the
// column scale is the remaining metadata vector (Ktok / Vch). Reductions follow the KVARN_RED_LANES contract
// of ggml-kvarn.h: 8 consecutive lanes own contiguous chunks, sequential double sums, xor butterfly.

#define KVARN_SEAL_THREADS 256
#define KVARN_SEAL_MAX_DIM 256

struct kvarn_seal_params {
    int D, G, iters, hkv;
    int bits_k, bits_v;
    size_t rec_bytes;
    size_t k_payload, v_payload, k_scale, k_zero, k_tok, v_ch, v_scale, v_zero;
    size_t k_nb1, v_nb1; // bytes per token row of the sources
    int n_groups_max;    // > 0: descriptor-driven (dynamic) seal
};

static __device__ __forceinline__ float kvarn_cur(const half * tile, const float * s_row, const float * s_col,
                                                  const int r, const int c, const int sr, const int sc) {
    const float m = __half2float(tile[r*sr + c*sc]);
    return __fdiv_rn(__fdiv_rn(m, s_col[c]), s_row[r]);
}

// one balancing pass over n_vec vectors of n = CHUNK*KVARN_RED_LANES elements; vector index is the column
// (col_pass) or the row of the tile; updates log_s[vec] and s_out[vec] (the vector's own scale only)
template <int CHUNK, bool col_pass>
static __device__ __forceinline__ void kvarn_std_pass(const half * tile, float * s_row, float * s_col,
                                                      double * log_s, float * s_out,
                                                      const int n_vec, const int sr, const int sc) {
    constexpr int n = CHUNK*KVARN_RED_LANES;
    for (int task = threadIdx.x; task < n_vec*KVARN_RED_LANES; task += KVARN_SEAL_THREADS) {
        const int vec  = task / KVARN_RED_LANES;
        const int lane = task % KVARN_RED_LANES;
        float vals[CHUNK];
#pragma unroll
        for (int i = 0; i < CHUNK; ++i) {
            const int idx = lane*CHUNK + i;
            vals[i] = col_pass ? kvarn_cur(tile, s_row, s_col, idx, vec, sr, sc)
                               : kvarn_cur(tile, s_row, s_col, vec, idx, sr, sc);
        }
        double acc = 0.0;
#pragma unroll
        for (int i = 0; i < CHUNK; ++i) {
            acc += (double) vals[i];
        }
#pragma unroll
        for (int w = 1; w < KVARN_RED_LANES; w <<= 1) {
            acc += __shfl_xor_sync(0xFFFFFFFF, acc, w, 32);
        }
        const double mean = acc / (double) n;
        double m2 = 0.0;
#pragma unroll
        for (int i = 0; i < CHUNK; ++i) {
            const double d = (double) vals[i] - mean;
            m2 = fma(d, d, m2);
        }
#pragma unroll
        for (int w = 1; w < KVARN_RED_LANES; w <<= 1) {
            m2 += __shfl_xor_sync(0xFFFFFFFF, m2, w, 32);
        }
        if (lane == 0) {
            float sd = (float) sqrt(m2 / (double) (n - 1));
            sd = fminf(fmaxf(sd, 1e-3f), 1e3f);
            double ls = log_s[vec] + log((double) sd);
            ls = fmin(fmax(ls, -0.3), 10.0);
            log_s[vec] = ls;
            s_out[vec] = (float) exp(ls);
        }
    }
}

template <bool col_pass>
static __device__ __forceinline__ void kvarn_std_pass_n(const half * tile, float * s_row, float * s_col,
                                                        double * log_s, float * s_out,
                                                        const int n_vec, const int n, const int sr, const int sc) {
    switch (n) {
        case  32: kvarn_std_pass< 4, col_pass>(tile, s_row, s_col, log_s, s_out, n_vec, sr, sc); break;
        case  64: kvarn_std_pass< 8, col_pass>(tile, s_row, s_col, log_s, s_out, n_vec, sr, sc); break;
        case 128: kvarn_std_pass<16, col_pass>(tile, s_row, s_col, log_s, s_out, n_vec, sr, sc); break;
        case 256: kvarn_std_pass<32, col_pass>(tile, s_row, s_col, log_s, s_out, n_vec, sr, sc); break;
        default: break; // rejected by supports_op
    }
}

static __device__ __forceinline__ int kvarn_phys_slot(const int idx, const int bits) {
    if (bits != 4) {
        return idx;
    }
    const int word = idx >> 3, k = idx & 7;
    return word*8 + ((k & 1) ? 4 + (k >> 1) : (k >> 1));
}

static __device__ __forceinline__ uint32_t kvarn_rtn(const float x, const float lo, const float step, const float qmaxf) {
    float r = rintf(__fdiv_rn(__fsub_rn(x, lo), step));
    r = fmaxf(r, 0.0f);
    r = fminf(r, qmaxf);
    return (uint32_t) r;
}

static __global__ void k_kvarn_seal(const char * __restrict__ ksrc, const char * __restrict__ vsrc,
                                    uint8_t * __restrict__ body, const int32_t * __restrict__ desc,
                                    const kvarn_seal_params p) {
    extern __shared__ half tile[]; // [G][D] token-major
    __shared__ float  s_row[KVARN_SEAL_MAX_DIM];
    __shared__ float  s_col[KVARN_SEAL_MAX_DIM];
    __shared__ double log_sr[KVARN_SEAL_MAX_DIM];
    __shared__ double log_sc[KVARN_SEAL_MAX_DIM];
    __shared__ float  q_lo[KVARN_SEAL_MAX_DIM];
    __shared__ float  q_step[KVARN_SEAL_MAX_DIM];

    const bool is_V = blockIdx.x & 1;
    const int  D = p.D, G = p.G;
    const int  tid = threadIdx.x;
    int rec  = blockIdx.x >> 1; // record index into body
    int row0;                   // first source token row of the group
    if (desc == nullptr) {
        row0 = (rec / p.hkv)*G;
    } else {
        const int S     = desc[GGML_KVARN_DESC_S];
        const int cap   = desc[GGML_KVARN_DESC_CAP];
        const int B     = desc[GGML_KVARN_DESC_B];
        const int B_old = desc[GGML_KVARN_DESC_B_OLD];
        const int n_new = B > B_old ? (B - B_old) / G : 0;
        const int g     = rec / p.hkv;
        if (g >= n_new) {
            return;
        }
        row0 = S + (B_old + g*G - S) % cap; // cap % G == 0 -> the group never wraps
        rec  = ((B_old - S)/G + g)*p.hkv + rec % p.hkv;
    }
    const int h = rec % p.hkv;

    // tile load: token rows row0 + t, channels h*D .. h*D + D
    {
        const char * src = (is_V ? vsrc : ksrc) + (size_t) h*D*sizeof(half);
        const size_t nb1 = is_V ? p.v_nb1 : p.k_nb1;
        const int chunks_per_row = D/8; // uint4 = 8 halves
        for (int i = tid; i < G*chunks_per_row; i += KVARN_SEAL_THREADS) {
            const int t = i / chunks_per_row, c = i % chunks_per_row;
            const uint4 x = *(const uint4 *) (src + (size_t) (row0 + t)*nb1 + (size_t) c*16);
            *(uint4 *) (tile + (size_t) t*D + c*8) = x;
        }
    }
    // (R, C, sr, sc): K rows = channels (stride 1), cols = tokens (stride D); V rows = tokens, cols = channels
    const int R  = is_V ? G : D;
    const int C  = is_V ? D : G;
    const int sr = is_V ? D : 1;
    const int sc = is_V ? 1 : D;
    for (int i = tid; i < KVARN_SEAL_MAX_DIM; i += KVARN_SEAL_THREADS) {
        s_row[i] = 1.0f; s_col[i] = 1.0f; log_sr[i] = 0.0; log_sc[i] = 0.0;
    }
    __syncthreads();

    for (int it = 0; it < p.iters; ++it) {
        kvarn_std_pass_n<true >(tile, s_row, s_col, log_sc, s_col, C, R, sr, sc); // columns: vectors over rows
        __syncthreads();
        kvarn_std_pass_n<false>(tile, s_row, s_col, log_sr, s_row, R, C, sr, sc); // rows: vectors over columns
        __syncthreads();
    }

    uint8_t * out = body + (size_t) rec*p.rec_bytes;
    const int      bits    = is_V ? p.bits_v : p.bits_k;
    const uint32_t qmax    = (1u << bits) - 1;
    const float    qmaxf   = (float) qmax;
    const size_t   payload = is_V ? p.v_payload : p.k_payload;
    const size_t   o_scale = is_V ? p.v_scale : p.k_scale;
    const size_t   o_zero  = is_V ? p.v_zero  : p.k_zero;
    const size_t   o_col   = is_V ? p.v_ch    : p.k_tok;
    const size_t   row_bytes = (size_t) D*bits/8;

    // phase A: per row lo/hi over the columns, 2 threads per row
    for (int task = tid; task < 2*R; task += KVARN_SEAL_THREADS) {
        const int r = task >> 1, half_ = task & 1;
        const int c0 = half_*(C/2), c1 = c0 + C/2;
        float lo = kvarn_cur(tile, s_row, s_col, r, c0, sr, sc), hi = lo;
        for (int c = c0 + 1; c < c1; ++c) {
            const float x = kvarn_cur(tile, s_row, s_col, r, c, sr, sc);
            lo = fminf(lo, x);
            hi = fmaxf(hi, x);
        }
        lo = fminf(lo, __shfl_xor_sync(0xFFFFFFFF, lo, 1, 32));
        hi = fmaxf(hi, __shfl_xor_sync(0xFFFFFFFF, hi, 1, 32));
        const float step = fmaxf(__fdiv_rn(__fsub_rn(hi, lo), qmaxf), 1e-10f);
        if (half_ == 0) {
            q_lo[r]   = lo;
            q_step[r] = step;
            ((half *) (out + o_scale))[r] = __float2half_rn(__fmul_rn(s_row[r], step));
            ((half *) (out + o_zero))[r]  = __float2half_rn(__fmul_rn(s_row[r], lo));
        }
    }
    for (int c = tid; c < C; c += KVARN_SEAL_THREADS) {
        ((half *) (out + o_col))[c] = __float2half_rn(s_col[c]);
    }
    __syncthreads();

    // phase B: payload rows are tokens; value i of token t is (r, c) = (i, t) for K and (t, i) for V.
    // 2 threads per token row, each packs D/2 values = D*bits/16 bytes (byte aligned for D >= 32).
    for (int task = tid; task < 2*G; task += KVARN_SEAL_THREADS) {
        const int t = task >> 1, half_ = task & 1;
        const int i0 = half_*(D/2);
        uint8_t * dst = out + payload + (size_t) t*row_bytes + (size_t) half_*(row_bytes/2);
        if (bits == 4) {
            // 8 values per 32-bit word, interleaved slots (values 2k -> nibble k, 2k+1 -> nibble k+4)
            uint32_t * dst32 = (uint32_t *) dst;
            for (int w = 0; w < D/16; ++w) {
                uint32_t word = 0;
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const int i = i0 + w*8 + k;
                    const int r = is_V ? t : i, c = is_V ? i : t;
                    const uint32_t q = kvarn_rtn(kvarn_cur(tile, s_row, s_col, r, c, sr, sc), q_lo[r], q_step[r], qmaxf);
                    word |= q << (4*kvarn_phys_slot(k, 4));
                }
                dst32[w] = word;
            }
        } else {
            uint8_t buf[KVARN_SEAL_MAX_DIM/2]; // <= D/2 bytes at 8 bits
            const int nbytes = (D/2)*bits/8;
            for (int b = 0; b < nbytes; ++b) buf[b] = 0;
            for (int j = 0; j < D/2; ++j) {
                const int i = i0 + j;
                const int r = is_V ? t : i, c = is_V ? i : t;
                const uint32_t q = kvarn_rtn(kvarn_cur(tile, s_row, s_col, r, c, sr, sc), q_lo[r], q_step[r], qmaxf);
                const uint32_t bit = (uint32_t) kvarn_phys_slot(j, bits)*bits;
                const uint32_t byte = bit >> 3, shift = bit & 7;
                const uint32_t wv = (q & qmax) << shift;
                buf[byte] |= (uint8_t) wv;
                if (shift + bits > 8) {
                    buf[byte + 1] |= (uint8_t) (wv >> 8);
                }
            }
            for (int b = 0; b < nbytes; ++b) dst[b] = buf[b];
        }
    }
}

bool ggml_cuda_kvarn_seal_supported(const ggml_tensor * op) {
    const ggml_tensor * k = op->src[0];
    const ggml_tensor * v = op->src[1];
    if (k == nullptr || v == nullptr || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 || op->type != GGML_TYPE_I8) {
        return false;
    }
    const int32_t D = ggml_get_op_params_i32(op, 0);
    const int32_t G = ggml_get_op_params_i32(op, 1);
    const int32_t bits_k = ggml_get_op_params_i32(op, 2);
    const int32_t bits_v = ggml_get_op_params_i32(op, 3);
    if (D < 32 || D > KVARN_SEAL_MAX_DIM || (D & (D - 1)) != 0) return false;
    if (G < 32 || G > KVARN_SEAL_MAX_DIM || (G & (G - 1)) != 0) return false;
    if (bits_k < 2 || bits_k > 8 || bits_v < 2 || bits_v > 8) return false;
    if (k->nb[0] != sizeof(ggml_fp16_t) || v->nb[0] != sizeof(ggml_fp16_t)) return false;
    if (k->nb[1] % 16 != 0 || v->nb[1] % 16 != 0) return false; // uint4 row loads
    return true;
}

void ggml_cuda_kvarn_seal(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * k = dst->src[0];
    const ggml_tensor * v = dst->src[1];
    GGML_ASSERT(ggml_cuda_kvarn_seal_supported(dst));

    kvarn_seal_params p;
    p.D      = ggml_get_op_params_i32(dst, 0);
    p.G      = ggml_get_op_params_i32(dst, 1);
    p.bits_k = ggml_get_op_params_i32(dst, 2);
    p.bits_v = ggml_get_op_params_i32(dst, 3);
    p.iters  = ggml_get_op_params_i32(dst, 4);
    p.hkv    = (int) (k->ne[0] / p.D);
    p.n_groups_max = ggml_get_op_params_i32(dst, 5);
    const ggml_tensor * desc = dst->src[2];
    const int n_groups = p.n_groups_max > 0 ? p.n_groups_max : (int) (k->ne[1] / p.G);
    const ggml_kvarn::layout l = ggml_kvarn::make_layout(p.D, p.G, p.bits_k, p.bits_v);
    p.rec_bytes = l.bytes;
    p.k_payload = l.k_payload; p.v_payload = l.v_payload;
    p.k_scale = l.k_scale; p.k_zero = l.k_zero; p.k_tok = l.k_tok;
    p.v_ch = l.v_ch; p.v_scale = l.v_scale; p.v_zero = l.v_zero;
    p.k_nb1 = k->nb[1]; p.v_nb1 = v->nb[1];
    if (p.n_groups_max > 0) {
        GGML_ASSERT(desc != nullptr && desc->type == GGML_TYPE_I32);
        GGML_ASSERT((int64_t) ggml_nelements(dst) % ((int64_t) l.bytes * p.hkv) == 0);
    } else {
        GGML_ASSERT((int64_t) ggml_nelements(dst) == (int64_t) l.bytes * p.hkv * n_groups);
    }

    const size_t smem = (size_t) p.G * p.D * sizeof(half);
    static bool smem_set[GGML_CUDA_MAX_DEVICES] = { false };
    const int id = ggml_cuda_get_device();
    if (!smem_set[id]) {
        CUDA_CHECK(cudaFuncSetAttribute(k_kvarn_seal, cudaFuncAttributeMaxDynamicSharedMemorySize, KVARN_SEAL_MAX_DIM*KVARN_SEAL_MAX_DIM/2*sizeof(half)));
        smem_set[id] = true;
    }
    GGML_ASSERT(smem <= (size_t) KVARN_SEAL_MAX_DIM*KVARN_SEAL_MAX_DIM/2*sizeof(half));

    const int n_rec = p.hkv * n_groups;
    if (n_rec == 0) {
        return;
    }
    k_kvarn_seal<<<2*n_rec, KVARN_SEAL_THREADS, smem, ctx.stream()>>>(
        (const char *) k->data, (const char *) v->data, (uint8_t *) dst->data,
        p.n_groups_max > 0 ? (const int32_t *) desc->data : nullptr, p);
    CUDA_CHECK(cudaGetLastError());
}
