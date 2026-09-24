#include "kvarn-seal.cuh"
#include "turbo-quant.cuh"
#include "../ggml-kvarn.h"

// One block per (record, tile): blockIdx.x = 2*record + is_V. The fp16 tile lives in shared memory
// token-major ([G][D], exactly the source rows); the balanced value cur(r, c) = M(r, c) / s_col[c] / s_row[r]
// is recomputed on the fly with IEEE division. K tiles use rows = channels (r = d, c = t), V tiles rows =
// tokens (r = t, c = d), so for both the per-row quantities (lo, step -> scale/zero) index the row and the
// column scale is the remaining metadata vector (Ktok / Vch). Reductions follow the KVARN_RED_LANES contract
// of ggml-kvarn.h: 8 consecutive lanes own contiguous chunks, sequential double sums, xor butterfly.

#define KVARN_SEAL_THREADS 256
#define KVARN_SEAL_MAX_DIM 256

// trellis-coded 4-bit body (body type GGML_TYPE_I16, ggml-kvarn.h "kvarn4t"): warp-parallel Viterbi, one warp per
// (channel tile, fragment lane) sequence of G/2 values. Lane owns windows w = lane + 32*i (i < NWIN/32), i.e. states
// lane + 32*(i % SPL) and new nibble j = i / SPL; the next state of window w is w >> 4 = (lane >> 4) + 2*i.
namespace kvarn_tr = ggml_kvarn::trellis;
static_assert(kvarn_tr::L >= 9 && kvarn_tr::L <= 10, "CUDA trellis sealer supports window lengths 9..10");
#define KVARN_TR_WPL  (kvarn_tr::NWIN/32)   // windows per lane
#define KVARN_TR_SPL  (kvarn_tr::NSTATE/32) // states per lane
#define KVARN_TR_NVAL 64                     // values per sequence = G/2, G <= 128
#define KVARN_TR_ARGW (KVARN_TR_NVAL*kvarn_tr::NSTATE/8)
static __device__ uint16_t kvarn_seal_cb_k[kvarn_tr::NWIN] = KVARN_CB_K_INIT;
static __device__ uint16_t kvarn_seal_cb_v[kvarn_tr::NWIN] = KVARN_CB_V_INIT;

// 3-bit trellis body (ggml-kvarn.h "kvarn3t"): warp-parallel Viterbi over one SEQ-channel sequence of a token row.
// Lane owns windows w = lane + 32*i (i < NWIN/32 = 16): state lane + 32*(i % 2), new code j = i / 2, next state
// w >> 3 = (lane >> 3) + 4*i. The traceback packs the 3-bit codes into the SEQ*3/32 words of the sequence.
namespace kvarn_tr3 = ggml_kvarn::trellis3;
#define KVARN_TR3_WPL   (kvarn_tr3::NWIN/32)              // 16 windows per lane
#define KVARN_TR3_SPL   (kvarn_tr3::NSTATE/32)            // 2 states per lane
#define KVARN_TR3_NVAL  (kvarn_tr3::SEQ)                  // 128 values per sequence
#define KVARN_TR3_ARGW  (KVARN_TR3_NVAL*kvarn_tr3::NSTATE/8) // 1024 words per warp (global scratch, p.tr3_arg)
#define KVARN_TR3_WORDS (kvarn_tr3::SEQ*kvarn_tr3::BITS/32)  // 12 payload words per sequence
static __device__ uint16_t kvarn_seal_cb3_k[kvarn_tr3::NWIN] = KVARN_CB3_TRAINED_K_INIT; // built-in trained (ggml-kvarn-cb-lowbits.h), overridden by the env copy below
static __device__ uint16_t kvarn_seal_cb3_v[kvarn_tr3::NWIN] = KVARN_CB3_TRAINED_V_INIT;
// 2-bit trellis body (ggml-kvarn.h "kvarn2t"): the same warp layout at 256 windows (8 per lane), 4 branches, L = 8
namespace kvarn_tr2 = ggml_kvarn::trellis2;
static __device__ uint16_t kvarn_seal_cb2_k[kvarn_tr2::NWIN] = KVARN_CB2_TRAINED_K_INIT;
static __device__ uint16_t kvarn_seal_cb2_v[kvarn_tr2::NWIN] = KVARN_CB2_TRAINED_V_INIT;

struct kvarn_trellis_cb_registry {
    std::vector<std::pair<const void *, const void *>> syms;
    std::vector<std::pair<const void *, const void *>> syms3;
    std::vector<std::pair<const void *, const void *>> syms2;
    bool done[GGML_CUDA_MAX_DEVICES] = { false };
};
static kvarn_trellis_cb_registry & kvarn_trellis_cb_reg() {
    static kvarn_trellis_cb_registry r;
    return r;
}
void ggml_cuda_kvarn_trellis_cb_register(const void * sym_k, const void * sym_v) {
    kvarn_trellis_cb_reg().syms.push_back({sym_k, sym_v});
}
void ggml_cuda_kvarn_trellis_cb3_register(const void * sym_k, const void * sym_v) {
    kvarn_trellis_cb_reg().syms3.push_back({sym_k, sym_v});
}
void ggml_cuda_kvarn_trellis_cb2_register(const void * sym_k, const void * sym_v) {
    kvarn_trellis_cb_reg().syms2.push_back({sym_k, sym_v});
}
static int kvarn_trellis_cb_copy(const std::vector<std::pair<const void *, const void *>> & syms, const uint16_t * hk, const uint16_t * hv, size_t nwin) {
    int n_ok = 0;
    for (const auto & s : syms) {
        // a translation unit whose device code dropped its unreferenced copy has no symbol: skip it
        const cudaError_t ek = cudaMemcpyToSymbol(s.first,  hk, nwin*sizeof(uint16_t));
        const cudaError_t ev = cudaMemcpyToSymbol(s.second, hv, nwin*sizeof(uint16_t));
        if (ek == cudaSuccess && ev == cudaSuccess) {
            ++n_ok;
        } else {
            (void) cudaGetLastError();
        }
    }
    return n_ok;
}
void ggml_cuda_kvarn_trellis_cb_init() {
    kvarn_trellis_cb_registry & r = kvarn_trellis_cb_reg();
    const int id = ggml_cuda_get_device();
    if (r.done[id]) {
        return;
    }
    r.done[id] = true;
    if (getenv("GGML_KVARN_TRELLIS_CB") != nullptr) {
        const int n_ok = kvarn_trellis_cb_copy(r.syms, ggml_kvarn::trellis::cb(false), ggml_kvarn::trellis::cb(true), kvarn_tr::NWIN);
        GGML_LOG_INFO("%s: trellis codebook override copied to %d of %zu device symbol pairs (device %d)\n", __func__, n_ok, r.syms.size(), id);
    }
    if (getenv("GGML_KVARN_TRELLIS_CB3") != nullptr) {
        const int n_ok = kvarn_trellis_cb_copy(r.syms3, ggml_kvarn::trellis3::cb(false), ggml_kvarn::trellis3::cb(true), kvarn_tr3::NWIN);
        GGML_LOG_INFO("%s: trellis3 codebook override copied to %d of %zu device symbol pairs (device %d)\n", __func__, n_ok, r.syms3.size(), id);
    }
    if (getenv("GGML_KVARN_TRELLIS_CB2") != nullptr) {
        const int n_ok = kvarn_trellis_cb_copy(r.syms2, ggml_kvarn::trellis2::cb(false), ggml_kvarn::trellis2::cb(true), kvarn_tr2::NWIN);
        GGML_LOG_INFO("%s: trellis2 codebook override copied to %d of %zu device symbol pairs (device %d)\n", __func__, n_ok, r.syms2.size(), id);
    }
}
struct kvarn_seal_cb_registrar {
    kvarn_seal_cb_registrar() {
        ggml_cuda_kvarn_trellis_cb_register((const void *) kvarn_seal_cb_k, (const void *) kvarn_seal_cb_v);
        ggml_cuda_kvarn_trellis_cb3_register((const void *) kvarn_seal_cb3_k, (const void *) kvarn_seal_cb3_v);
        ggml_cuda_kvarn_trellis_cb2_register((const void *) kvarn_seal_cb2_k, (const void *) kvarn_seal_cb2_v);
    }
};
static kvarn_seal_cb_registrar kvarn_seal_cb_registrar_instance;

struct kvarn_seal_params {
    int D, G, iters, hkv;
    int trellis;
    int refit;           // trellis3 per-row affine refit mode (ggml_kvarn::trellis3::refit_mode)
    uint32_t * tr3_arg;  // trellis3 Viterbi back-pointers, global scratch: [block][warp][KVARN_TR3_ARGW] (static + tile smem leave no room)
    int bits_k, bits_v;
    int type_k, type_v;
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

// inverse of ggml_kvarn::code_bit for 4-bit fragment order: (word, nibble) -> (token, channel) of a G x D tile
static __device__ __forceinline__ void kvarn_frag_pos(const int word, const int nib, const bool is_V, const int D,
                                                      int & t, int & d) {
    const int lane = word & 31;
    const int tile = (word >> 5) % (D/16);
    const int s    = (word >> 5) / (D/16);
    const int l = nib & 3, e = nib >> 2;
    // A-fragment (tile<16,8,half2>) register l of a lane: row (l%2)*8 + lane/4, half2 column (l/2)*4 + lane%4
    const int row = (l & 1)*8 + (lane >> 2);
    const int col = 2*((l >> 1)*4 + (lane & 3)) + e;
    if (!is_V) {
        t = s*16 + row; d = tile*16 + col;
    } else {
        d = tile*16 + row; t = s*16 + col;
    }
}

static __device__ __forceinline__ uint32_t kvarn_rtn(const float x, const float lo, const float step, const float qmaxf) {
    float r = rintf(__fdiv_rn(__fsub_rn(x, lo), step));
    r = fmaxf(r, 0.0f);
    r = fminf(r, qmaxf);
    return (uint32_t) r;
}

// Arithmetic contract shared with ggml_kvarn::trellis_encode (records bit-identical to the CPU reference):
// y = (cur - lo)/step, d = y - cb[w], e = d*d, cost = e + M[w >> 4], all IEEE without contraction; strict < over
// ascending j per state; start state 0, no termination.
static __device__ void kvarn_trellis_phase_b(const half * tile, const float * s_row, const float * s_col,
                                             const float * q_lo, const float * q_step, const int sr, const int sc,
                                             const bool is_V, const int D, const int G, uint32_t * dst32,
                                             float * ys, float * M, uint32_t * arg) {
    using namespace kvarn_tr;
    constexpr int WPL = KVARN_TR_WPL, SPL = KVARN_TR_SPL;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    constexpr int nwarps = KVARN_SEAL_THREADS/32;
    const int ntiles_c = D/16, nstrips = G/16, n = G/2;
    const uint16_t * cb = is_V ? kvarn_seal_cb_v : kvarn_seal_cb_k;
    float cbf[WPL];
#pragma unroll
    for (int i = 0; i < WPL; ++i) {
        cbf[i] = __half2float(__ushort_as_half(cb[lane + 32*i]));
    }
    for (int seq = warp; seq < ntiles_c*32; seq += nwarps) {
        const int c = seq >> 5, fl = seq & 31;
        for (int i = lane; i < n; i += 32) {
            const int word = ((i >> 3)*ntiles_c + c)*32 + fl;
            int t, d;
            kvarn_frag_pos(word, i & 7, is_V, D, t, d);
            const int r = is_V ? t : d, col = is_V ? d : t;
            ys[i] = __fdiv_rn(__fsub_rn(kvarn_cur(tile, s_row, s_col, r, col, sr, sc), q_lo[r]), q_step[r]);
        }
        for (int s = lane; s < NSTATE; s += 32) M[s] = 0.0f;
        __syncwarp();
        for (int i = n - 1; i >= 0; --i) {
            const float y = ys[i];
            float best[SPL]; int bj[SPL];
#pragma unroll
            for (int k = 0; k < SPL; ++k) { best[k] = INFINITY; bj[k] = 0; }
#pragma unroll
            for (int w = 0; w < WPL; ++w) {
                const float d   = __fsub_rn(y, cbf[w]);
                const float e   = __fmul_rn(d, d);
                const float cst = __fadd_rn(e, M[(lane >> 4) + 2*w]);
                const int k = w % SPL;
                if (cst < best[k]) { best[k] = cst; bj[k] = w / SPL; }
            }
            __syncwarp();
#pragma unroll
            for (int k = 0; k < SPL; ++k) {
                M[lane + 32*k] = best[k];
                uint32_t v = (uint32_t) bj[k] << (4*(lane & 7));
                v |= __shfl_xor_sync(0xFFFFFFFF, v, 1, 32);
                v |= __shfl_xor_sync(0xFFFFFFFF, v, 2, 32);
                v |= __shfl_xor_sync(0xFFFFFFFF, v, 4, 32);
                if ((lane & 7) == 0) {
                    arg[(i*NSTATE + lane + 32*k) >> 3] = v;
                }
            }
            __syncwarp();
        }
        uint32_t word = 0;
        int s = 0;
        for (int i = 0; i < n; ++i) {
            const int j = (arg[(i*NSTATE + s) >> 3] >> (4*(s & 7))) & 15;
            if ((i >> 3) == lane) {
                word |= (uint32_t) j << (4*(i & 7));
            }
            s = (s + NSTATE*j) >> 4;
        }
        if (lane < nstrips) {
            dst32[(lane*ntiles_c + c)*32 + fl] = word;
        }
        __syncwarp();
    }
}

// Arithmetic contract shared with ggml_kvarn::trellis_lb_encode<BITS> (records bit-identical to the CPU reference).
// BITS = 3: 16 windows per lane, next state (lane >> 3) + 4*w; BITS = 2: 8 windows per lane, next state (lane >> 2) + 8*w.
template<int BITS>
static __device__ void kvarn_trellis_lb_phase_b(const half * tile, const float * s_row, const float * s_col,
                                                const float * q_lo, const float * q_step, const int sr, const int sc,
                                                const bool is_V, const int D, const int G, uint32_t * dst32,
                                                float * ys, float * M, uint32_t * arg) {
    static_assert(BITS == 2 || BITS == 3, "low-bit trellis sealer");
    constexpr int NSTATE = kvarn_tr3::NSTATE, SEQ = kvarn_tr3::SEQ;
    constexpr int WPL = (1 << (6 + BITS))/32, SPL = KVARN_TR3_SPL, n = KVARN_TR3_NVAL, WORDS = SEQ*BITS/32;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    constexpr int nwarps = KVARN_SEAL_THREADS/32;
    const int nseq_row = D/SEQ;
    const uint16_t * cb = BITS == 3 ? (is_V ? kvarn_seal_cb3_v : kvarn_seal_cb3_k) : (is_V ? kvarn_seal_cb2_v : kvarn_seal_cb2_k);
    float cbf[WPL];
#pragma unroll
    for (int i = 0; i < WPL; ++i) {
        cbf[i] = __half2float(__ushort_as_half(cb[lane + 32*i]));
    }
    for (int seq = warp; seq < G*nseq_row; seq += nwarps) {
        const int t = seq / nseq_row, h = seq % nseq_row;
        for (int i = lane; i < n; i += 32) {
            const int d = h*SEQ + i;
            const int r = is_V ? t : d, c = is_V ? d : t;
            ys[i] = __fdiv_rn(__fsub_rn(kvarn_cur(tile, s_row, s_col, r, c, sr, sc), q_lo[r]), q_step[r]);
        }
        for (int s = lane; s < NSTATE; s += 32) M[s] = 0.0f;
        __syncwarp();
        for (int i = n - 1; i >= 0; --i) {
            const float y = ys[i];
            float best[SPL]; int bj[SPL];
#pragma unroll
            for (int k = 0; k < SPL; ++k) { best[k] = INFINITY; bj[k] = 0; }
#pragma unroll
            for (int w = 0; w < WPL; ++w) {
                const float d   = __fsub_rn(y, cbf[w]);
                const float e   = __fmul_rn(d, d);
                const float cst = __fadd_rn(e, M[(lane >> BITS) + (32 >> BITS)*w]);
                const int k = w % SPL;
                if (cst < best[k]) { best[k] = cst; bj[k] = w / SPL; }
            }
            __syncwarp();
#pragma unroll
            for (int k = 0; k < SPL; ++k) {
                M[lane + 32*k] = best[k];
                uint32_t v = (uint32_t) bj[k] << (4*(lane & 7));
                v |= __shfl_xor_sync(0xFFFFFFFF, v, 1, 32);
                v |= __shfl_xor_sync(0xFFFFFFFF, v, 2, 32);
                v |= __shfl_xor_sync(0xFFFFFFFF, v, 4, 32);
                if ((lane & 7) == 0) {
                    arg[(i*NSTATE + lane + 32*k) >> 3] = v;
                }
            }
            __syncwarp();
        }
        uint32_t word = 0;
        int s = 0;
        for (int i = 0; i < n; ++i) {
            const int j = (arg[(i*NSTATE + s) >> 3] >> (4*(s & 7))) & 15;
            const int b = BITS*i - 32*lane;
            if (b > -BITS && b < 32) {
                word |= b >= 0 ? ((uint32_t) j << b) : ((uint32_t) j >> -b);
            }
            s = (s + NSTATE*j) >> BITS;
        }
        if (lane < WORDS) {
            dst32[((size_t) t*D + (size_t) h*SEQ)*BITS/32 + lane] = word;
        }
        __syncwarp();
    }
}

// per-row affine refit of a coded low-bit trellis payload (ggml_kvarn::trellis_lb_refit_rows<BITS>): one thread per row,
// sequential double sums in column order
template<int BITS>
static __device__ void kvarn_trellis_lb_refit(const half * tile, const float * s_row, const float * s_col, const int sr, const int sc,
                                              const bool is_V, const int D, const int G, const int R, const int C,
                                              const uint8_t * payload, uint8_t * out, const size_t o_scale, const size_t o_zero,
                                              const int bits, const int mode) {
    const uint16_t * cb = BITS == 3 ? (is_V ? kvarn_seal_cb3_v : kvarn_seal_cb3_k) : (is_V ? kvarn_seal_cb2_v : kvarn_seal_cb2_k);
    for (int r = threadIdx.x; r < R; r += KVARN_SEAL_THREADS) {
        double Sx = 0.0, Sy = 0.0, Sxx = 0.0, Syy = 0.0, Sxy = 0.0;
        for (int c = 0; c < C; ++c) {
            const int t = is_V ? r : c, d = is_V ? c : r;
            const double x  = (double) kvarn_cur(tile, s_row, s_col, r, c, sr, sc);
            const double yh = (double) __half2float(__ushort_as_half(cb[ggml_kvarn::trellis_lb_window<BITS>(payload, t, d, D)]));
            Sx = __dadd_rn(Sx, x); Sy = __dadd_rn(Sy, yh);
            Sxx = __dadd_rn(Sxx, __dmul_rn(x, x)); Syy = __dadd_rn(Syy, __dmul_rn(yh, yh)); Sxy = __dadd_rn(Sxy, __dmul_rn(x, yh));
        }
        float a, b;
        if (!ggml_kvarn::trellis3_refit_solve(mode, (double) C, Sx, Sy, Sxx, Syy, Sxy, a, b)) { continue; }
        const int ri = is_V ? r : ggml_kvarn::k_ch_idx(r, D, G, bits);
        ((half *) (out + o_scale))[ri] = __float2half_rn(__fmul_rn(s_row[r], a));
        ((half *) (out + o_zero))[ri]  = __float2half_rn(__fmul_rn(s_row[r], b));
    }
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
    __shared__ float    tr_y[KVARN_SEAL_THREADS/32][KVARN_TR3_NVAL > KVARN_TR_NVAL ? KVARN_TR3_NVAL : KVARN_TR_NVAL];
    __shared__ float    tr_M[KVARN_SEAL_THREADS/32][kvarn_tr::NSTATE];
    __shared__ uint32_t tr_arg[KVARN_SEAL_THREADS/32][KVARN_TR_ARGW];

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
        const int type = is_V ? p.type_v : p.type_k;
        const size_t head_bytes = type == GGML_TYPE_TQ6_0 ? (D/QK_TQ6)*sizeof(block_tq6_0) : type == GGML_TYPE_Q8_0 ? (D/QK8_0)*sizeof(block_q8_0) : D*sizeof(half);
        const char * src = (is_V ? vsrc : ksrc) + (size_t) h*head_bytes;
        const size_t nb1 = is_V ? p.v_nb1 : p.k_nb1;
        const int chunks_per_row = D/8; // uint4 = 8 halves
        for (int i = tid; i < G*chunks_per_row; i += KVARN_SEAL_THREADS) {
            const int t = i / chunks_per_row, c = i % chunks_per_row;
            if (type == GGML_TYPE_TQ6_0) {
                const block_tq6_0 * b = ((const block_tq6_0 *) (src + (size_t) (row0 + t)*nb1)) + c/(QK_TQ6/8);
                const float norm = __half2float(b->norm);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    tile[(size_t) t*D + c*8 + j] = __float2half_rn(tq6_dequant_element(b, (c*8+j)%QK_TQ6, norm));
                }
            } else if (type == GGML_TYPE_Q8_0) {
                const block_q8_0 & b = ((const block_q8_0 *) (src + (size_t) (row0 + t)*nb1))[c/4];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    tile[(size_t) t*D + c*8 + j] = __float2half_rn(__half2float(b.d)*b.qs[(c%4)*8 + j]);
                }
            } else {
                const uint4 x = *(const uint4 *) (src + (size_t) (row0 + t)*nb1 + (size_t) c*16);
                *(uint4 *) (tile + (size_t) t*D + c*8) = x;
            }
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
            // K rows are channels: Kscale/Kzero are stored in fragment order; V rows are tokens (natural order)
            const int ri = is_V ? r : ggml_kvarn::k_ch_idx(r, D, G, bits);
            ((half *) (out + o_scale))[ri] = __float2half_rn(__fmul_rn(s_row[r], step));
            ((half *) (out + o_zero))[ri]  = __float2half_rn(__fmul_rn(s_row[r], lo));
        }
    }
    for (int c = tid; c < C; c += KVARN_SEAL_THREADS) {
        // K columns are tokens (Ktok, natural order); V columns are channels (Vch, fragment order)
        const int ci = is_V ? ggml_kvarn::v_ch_idx(c, D, G, bits) : c;
        ((half *) (out + o_col))[ci] = __float2half_rn(s_col[c]);
    }
    __syncthreads();

    // phase B: 4-bit payloads in fragment order (ggml_kvarn::code_bit), one 32-bit word = 8 codes per task;
    // other widths as a token-major bit stream, 2 threads per token row.
    if (ggml_kvarn::frag_order(bits, D, G)) {
        uint32_t * dst32 = (uint32_t *) (out + payload);
        if (p.trellis) {
            const int warp = tid >> 5;
            kvarn_trellis_phase_b(tile, s_row, s_col, q_lo, q_step, sr, sc, is_V, D, G, dst32,
                                  tr_y[warp], tr_M[warp], tr_arg[warp]);
            return;
        }
        const int n_words = G*D/8;
        for (int w = tid; w < n_words; w += KVARN_SEAL_THREADS) {
            uint32_t word = 0;
#pragma unroll
            for (int nib = 0; nib < 8; ++nib) {
                int t, d;
                kvarn_frag_pos(w, nib, is_V, D, t, d);
                const int r = is_V ? t : d, c = is_V ? d : t;
                const uint32_t q = kvarn_rtn(kvarn_cur(tile, s_row, s_col, r, c, sr, sc), q_lo[r], q_step[r], qmaxf);
                word |= q << (4*nib);
            }
            dst32[w] = word;
        }
    } else if (p.trellis && (bits == 3 || bits == 2)) {
        uint32_t * dst32 = (uint32_t *) (out + payload);
        uint32_t * tr3_arg = p.tr3_arg + ((size_t) blockIdx.x*(KVARN_SEAL_THREADS/32) + (tid >> 5))*KVARN_TR3_ARGW;
        if (bits == 3) {
            kvarn_trellis_lb_phase_b<3>(tile, s_row, s_col, q_lo, q_step, sr, sc, is_V, D, G, dst32, tr_y[tid >> 5], tr_M[tid >> 5], tr3_arg);
        } else {
            kvarn_trellis_lb_phase_b<2>(tile, s_row, s_col, q_lo, q_step, sr, sc, is_V, D, G, dst32, tr_y[tid >> 5], tr_M[tid >> 5], tr3_arg);
        }
        if (p.refit) {
            __syncthreads();
            if (bits == 3) {
                kvarn_trellis_lb_refit<3>(tile, s_row, s_col, sr, sc, is_V, D, G, R, C, out + payload, out, o_scale, o_zero, bits, p.refit);
            } else {
                kvarn_trellis_lb_refit<2>(tile, s_row, s_col, sr, sc, is_V, D, G, R, C, out + payload, out, o_scale, o_zero, bits, p.refit);
            }
        }
    } else {
        for (int task = tid; task < 2*G; task += KVARN_SEAL_THREADS) {
            const int t = task >> 1, half_ = task & 1;
            const int i0 = half_*(D/2);
            uint8_t * dst = out + payload + (size_t) t*row_bytes + (size_t) half_*(row_bytes/2);
            uint8_t buf[KVARN_SEAL_MAX_DIM]; // <= D/2 bytes at 8 bits
            const int nbytes = (D/2)*bits/8;
            for (int b = 0; b < nbytes; ++b) buf[b] = 0;
            for (int j = 0; j < D/2; ++j) {
                const int i = i0 + j;
                const int r = is_V ? t : i, c = is_V ? i : t;
                const uint32_t q = kvarn_rtn(kvarn_cur(tile, s_row, s_col, r, c, sr, sc), q_lo[r], q_step[r], qmaxf);
                const uint32_t bit = (uint32_t) j*bits;
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

// Each thread converts one 128-value block in the existing rotation basis.
static __global__ void k_tiered_tq_seal(const char * K, const char * V, char * body,
        const int32_t * desc, kvarn_seal_params p) {
    int record = blockIdx.x;
    const int head = record%p.hkv;
    const int group = record/p.hkv;
    int row0 = group*p.G;
    if (desc) {
        const int S = desc[GGML_KVARN_DESC_S], old = desc[GGML_KVARN_DESC_B_OLD];
        const int count = (desc[GGML_KVARN_DESC_B] - old)/p.G;
        if (group >= count) return;
        row0 = S + (old + group*p.G - S)%desc[GGML_KVARN_DESC_CAP];
        record += ((old-S)/p.G)*p.hkv;
    }
    const int is_v = blockIdx.y;
    const int type = is_v ? p.type_v : p.type_k;
    const size_t stride = is_v ? p.v_nb1 : p.k_nb1;
    const int blocks_per_row = p.D/QK_TURBO4;
    const size_t head_bytes = type == GGML_TYPE_TQ6_0 ? blocks_per_row*sizeof(block_tq6_0) :
        type == GGML_TYPE_Q8_0 ? (p.D/QK8_0)*sizeof(block_q8_0) : p.D*sizeof(half);
    for (int task = threadIdx.x; task < p.G*blocks_per_row; task += blockDim.x) {
        const int token = task/blocks_per_row, block = task%blocks_per_row;
        const char * row = (is_v ? V : K) + (row0+token)*stride + head*head_bytes;
        block_turbo4_0 * out = (block_turbo4_0 *) (body + record*p.rec_bytes) + is_v*p.G*blocks_per_row + task;
        float norm2 = 0.0f;
        for (int j = 0; j < QK_TURBO4; ++j) {
            const int col = block*QK_TURBO4+j;
            float value;
            if (type == GGML_TYPE_TQ6_0) {
                const block_tq6_0 * b = (const block_tq6_0 *) row + block;
                value = tq6_dequant_element(b, j, __half2float(b->norm));
            } else if (type == GGML_TYPE_Q8_0) {
                const block_q8_0 & b = ((const block_q8_0 *) row)[col/QK8_0];
                value = __fmul_rn(__half2float(b.d), (float) b.qs[col%QK8_0]);
            } else value = __half2float(((const half *) row)[col]);
            value = __half2float(__float2half_rn(value));
            norm2 = __fadd_rn(norm2, __fmul_rn(value, value));
        }
        const float norm = __fsqrt_rn(norm2);
        const float inv = norm > 1e-10f ? __fdiv_rn(1.0f, norm) : 0.0f;
        float recon2 = 0.0f;
        for (int j = 0; j < QK_TURBO4; j += 2) {
            uint8_t packed = 0;
            for (int e = 0; e < 2; ++e) {
                const int col = block*QK_TURBO4+j+e;
                float value;
                if (type == GGML_TYPE_TQ6_0) {
                    const block_tq6_0 * b = (const block_tq6_0 *) row + block;
                    value = tq6_dequant_element(b, j+e, __half2float(b->norm));
                } else if (type == GGML_TYPE_Q8_0) {
                    const block_q8_0 & b = ((const block_q8_0 *) row)[col/QK8_0];
                    value = __fmul_rn(__half2float(b.d), (float) b.qs[col%QK8_0]);
                } else value = __half2float(((const half *) row)[col]);
                value = __half2float(__float2half_rn(value));
                const uint8_t code = turbo_nearest_centroid_4bit(__fmul_rn(value, inv));
                packed |= code << (4*e);
                const float centroid = TURBO_CENTROIDS_4BIT[code];
                recon2 = __fadd_rn(recon2, __fmul_rn(centroid, centroid));
            }
            out->qs[j/2] = packed;
        }
        const float recon = __fsqrt_rn(recon2);
        out->norm = __float2half_rn(recon > 1e-10f ? __fdiv_rn(norm, recon) : norm);
    }
}

bool ggml_cuda_kvarn_seal_supported(const ggml_tensor * op) {
    const ggml_tensor * k = op->src[0];
    const ggml_tensor * v = op->src[1];
    if (k == nullptr || v == nullptr || (k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_Q8_0 && k->type != GGML_TYPE_TQ6_0) || (v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_Q8_0 && v->type != GGML_TYPE_TQ6_0) || op->type != GGML_TYPE_I8) {
        return false;
    }
    const int32_t D = ggml_get_op_params_i32(op, 0);
    const int32_t G = ggml_get_op_params_i32(op, 1);
    const int32_t bits_k = ggml_get_op_params_i32(op, 2);
    const int32_t bits_v = ggml_get_op_params_i32(op, 3);
    if (D < 32 || D > KVARN_SEAL_MAX_DIM || (D & (D - 1)) != 0) return false;
    if (ggml_get_op_params_i32(op, 7) == GGML_TYPE_TURBO4_0 && D % QK_TURBO4 != 0) return false;
    if (ggml_get_op_params_i32(op, 7) == GGML_TYPE_I16) {
        for (const int32_t bits : {bits_k, bits_v}) {
            if (bits == 4) { if (G > 2*KVARN_TR_NVAL || G < 16) return false; }
            else if (bits == 3 || bits == 2) { if (D % kvarn_tr3::SEQ != 0) return false; }
            else return false;
        }
    }
    if ((k->type == GGML_TYPE_TQ6_0 || v->type == GGML_TYPE_TQ6_0) && D % QK_TQ6 != 0) return false;
    if (G < 32 || G > KVARN_SEAL_MAX_DIM || (G & (G - 1)) != 0) return false;
    if (bits_k < 2 || bits_k > 8 || bits_v < 2 || bits_v > 8) return false;
    if (k->nb[0] != ggml_type_size(k->type) || v->nb[0] != ggml_type_size(v->type)) return false;
    if ((k->type == GGML_TYPE_F16 && k->nb[1] % 16 != 0) ||
        (v->type == GGML_TYPE_F16 && v->nb[1] % 16 != 0)) return false; // uint4 row loads
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
    const bool turbo4_body = ggml_get_op_params_i32(dst, 7) == GGML_TYPE_TURBO4_0;
    p.trellis = ggml_get_op_params_i32(dst, 7) == GGML_TYPE_I16;
    p.refit   = p.trellis && (p.bits_k == 3 || p.bits_v == 3 || p.bits_k == 2 || p.bits_v == 2) ? ggml_kvarn::trellis3::refit_mode() : 0;
    p.rec_bytes = turbo4_body ? 2*p.G*ggml_row_size(GGML_TYPE_TURBO4_0, p.D) : l.bytes;
    p.k_payload = l.k_payload; p.v_payload = l.v_payload;
    p.k_scale = l.k_scale; p.k_zero = l.k_zero; p.k_tok = l.k_tok;
    p.v_ch = l.v_ch; p.v_scale = l.v_scale; p.v_zero = l.v_zero;
    p.k_nb1 = k->nb[1]; p.v_nb1 = v->nb[1];
    p.type_k = k->type; p.type_v = v->type;
    if (p.n_groups_max > 0) {
        GGML_ASSERT(desc != nullptr && desc->type == GGML_TYPE_I32);
        GGML_ASSERT((int64_t) ggml_nelements(dst) % ((int64_t) p.rec_bytes * p.hkv) == 0);
    } else {
        GGML_ASSERT((int64_t) ggml_nelements(dst) == (int64_t) p.rec_bytes * p.hkv * n_groups);
    }

    if (p.trellis) {
        ggml_cuda_kvarn_trellis_cb_init();
    }
    if (turbo4_body) {
        GGML_ASSERT(p.D % QK_TURBO4 == 0);
        k_tiered_tq_seal<<<dim3(p.hkv*n_groups, 2), 128, 0, ctx.stream()>>>(
            (const char *) k->data, (const char *) v->data, (char *) dst->data,
            desc ? (const int32_t *) desc->data : nullptr, p);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const bool tr3 = p.trellis && (p.bits_k == 3 || p.bits_v == 3 || p.bits_k == 2 || p.bits_v == 2);
    const size_t smem = (size_t) p.G * p.D * sizeof(half);
    const size_t smem_max = (size_t) KVARN_SEAL_MAX_DIM*KVARN_SEAL_MAX_DIM/2*sizeof(half);
    static bool smem_set[GGML_CUDA_MAX_DEVICES] = { false };
    const int id = ggml_cuda_get_device();
    if (!smem_set[id]) {
        CUDA_CHECK(cudaFuncSetAttribute(k_kvarn_seal, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max));
        smem_set[id] = true;
    }
    GGML_ASSERT(smem <= smem_max);

    const int n_rec = p.hkv * n_groups;
    if (n_rec == 0) {
        return;
    }
    ggml_cuda_pool_alloc<uint32_t> tr3_arg_alloc(ctx.pool());
    p.tr3_arg = nullptr;
    if (tr3) {
        p.tr3_arg = tr3_arg_alloc.alloc((size_t) 2*n_rec*(KVARN_SEAL_THREADS/32)*KVARN_TR3_ARGW);
    }
    k_kvarn_seal<<<2*n_rec, KVARN_SEAL_THREADS, smem, ctx.stream()>>>(
        (const char *) k->data, (const char *) v->data, (uint8_t *) dst->data,
        p.n_groups_max > 0 ? (const int32_t *) desc->data : nullptr, p);
    CUDA_CHECK(cudaGetLastError());
}
