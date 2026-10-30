// [HCFUSED] Flash-Next hyper-connection mix for small batches (verify/decode), opt-in through LLAMA_HC_FUSED=1 in
// the model graph. The unfused graph runs rms_norm, mul, the W_down / W_inject matmuls, scale, silu, the W_up matmul
// and the gated mean as separate kernels; at 1..8 tokens every one of them is launch- or occupancy-bound.
//
// Here it is two kernels (v2; three with GGML_CUDA_HC_FUSED_FINISH=0, which runs down_finish as its own launch):
//   down_partial: split-K over 512-column chunks of the W_down and W_inject rows. Each CTA stages x*w_norm for its
//                 chunk in shared memory (the per-stream rms scale r is a per-chunk constant, so it is applied after
//                 the dot product), and each row block writes the chunk's sum of squares.
//   down_finish:  r from the chunk sums of squares, the chunk partials combined with r, then silu(lo_scale*v) for the
//                 R low-rank rows and 2*sigmoid(inj_scale*v) for the injection rows. In v2 the last CTA of each row
//                 block to arrive does this for its own rows, with the same sums in the same order.
//   up_gate:      one warp per embedding column i, eight lanes per stream; the W_up row of (stream, i) is dotted with
//                 the activations, and the sigmoid gate, the xn product and the mean over the streams are applied in
//                 registers.
// Activations stay f32 throughout (the unfused mul_mat_f rounds them to BF16 first), so results differ from the
// unfused graph by rounding only.
// [HCPD] GGML_CUDA_HC_POST_DOWN=1 (opt-in; graph match ggml_cuda_try_fuse_hc_post_down in ggml-cuda.cu): the identity-
// comb hc_post that produces the mix input (h = block_out*post + residual per stream) runs inside down_partial. Every
// CTA computes the h values it stages from the hc_post inputs, row block 0 writes them to the hc_post output for
// up_gate and the later readers, and the expression is dsv4_hc_post_f32<false>'s, so h is the same float bit for bit.

#include "hc-fused.cuh"

#include <cstdlib>
#include <mutex>
#include <type_traits>
#include <unordered_map>

static constexpr int HCF_CHUNK      = 512;  // columns per split-K chunk; n_embd must be a multiple
static constexpr int HCF_DOWN_WARPS_DEF = 8;
static constexpr int HCF_DOWN_ROWS_DEF = 4;  // rows per warp (default; LLAMA_HC_DOWN_ROWS=1|2 gives more, smaller CTAs)
static constexpr int HCF_UP_WARPS   = 16;   // embedding columns per CTA
static constexpr int HCF_MAX_T      = 8;

static __device__ __forceinline__ float hcf_bf16_lo(uint32_t v) { return __uint_as_float(v << 16); }
static __device__ __forceinline__ float hcf_bf16_hi(uint32_t v) { return __uint_as_float(v & 0xFFFF0000u); }

// [HCFUSED v2] Per-launch arrival counters for the fused finish. Each backend context gets its own slot of
// HCF_CNT_RB counters (one per row block); the last CTA of a row block resets its counter, so they are zero
// between launches and a captured graph can replay them. The mixes of one context run in stream order (each needs
// the previous one's output), so one slot per context is enough.
static constexpr int HCF_CNT_SLOTS = 16;
static constexpr int HCF_CNT_RB    = 64;   // max row blocks: n_rows <= 64*32
static __device__ unsigned int hcf_arrive[HCF_CNT_SLOTS*HCF_CNT_RB];

// [HCPD] hc_post inputs folded into down_partial: blk [n_embd, T] (row stride sb1), post [hc, T] (strides sp0, sp1),
// residual and h contiguous [n_embd, hc, T] like x
struct hcf_post_fold {
    const float * blk;
    const float * post;
    const float * res;
    float *       h;
    int64_t       sb1;
    int64_t       sp0;
    int64_t       sp1;
};

// dsv4_hc_post_f32<false>'s value: its b*p + r compiles to one fused multiply-add (FFMA.FTZ in its SASS), so this
// spells the same single rounding out instead of leaving it to contraction
static __device__ __forceinline__ float hcf_post_val(const float b, const float p, const float r) {
    return __fmaf_rn(b, p, r);
}

// four h values of token t at columns j..j+3 of stream s (i0 = j - s*n_embd), from the hc_post inputs
static __device__ __forceinline__ float4 hcf_post_val4(const hcf_post_fold & pf, const int t, const int s, const int i0,
        const int j, const int hc_dim) {
    const float4 bv = *(const float4 *) (pf.blk + (int64_t) t*pf.sb1 + i0);
    const float  pv = pf.post[s*pf.sp0 + t*pf.sp1];
    const float4 rv = *(const float4 *) (pf.res + (int64_t) t*hc_dim + j);
    return make_float4(hcf_post_val(bv.x, pv, rv.x), hcf_post_val(bv.y, pv, rv.y),
                       hcf_post_val(bv.z, pv, rv.z), hcf_post_val(bv.w, pv, rv.w));
}

static constexpr int HCF_OWNER_ROWS = 16;
static constexpr int HCF_OWNER_MAX_ROWS = 336;
static constexpr int HCF_OWNER_RB = HCF_OWNER_MAX_ROWS/HCF_OWNER_ROWS;
static __device__ unsigned int hcf_owner_arrive[HCF_CNT_SLOTS*HCF_OWNER_RB];

template <int WIDTH = 32>
static __device__ __forceinline__ float hcf_owner_sum(float v, const unsigned mask = 0xffffffffu) {
#pragma unroll
    for (int d = WIDTH/2; d; d >>= 1) {
        v = __fadd_rn(v, __shfl_xor_sync(mask, v, d, WIDTH));
    }
    return v;
}

// Stage each weighted stream once. The row kernel applies its RMS scale after the dot product.
template <bool FOLD>
static __global__ void __launch_bounds__(256) hcf_row_owner_stage(
        const float * __restrict__ x, const float * __restrict__ wn, const int n_embd, const int n_rows,
        const int n_tokens, const float eps, float * __restrict__ xn, float * __restrict__ dst, const hcf_post_fold pf) {
    __shared__ float sums[8];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int s = blockIdx.x, t = blockIdx.y, hc_dim = 4*n_embd;
    float sq = 0.0f;
    for (int j = tid*4; j < n_embd; j += 256*4) {
        const int src = s*n_embd + j;
        float4 xv;
        if constexpr (FOLD) {
            xv = hcf_post_val4(pf, t, s, j, src, hc_dim);
            *(float4 *) (pf.h + (int64_t) t*hc_dim + src) = xv;
        } else {
            xv = *(const float4 *) (x + (int64_t) t*hc_dim + src);
        }
        const float4 gv = *(const float4 *) (wn + src);
        sq += xv.x*xv.x + xv.y*xv.y + xv.z*xv.z + xv.w*xv.w;
        *(float4 *) (xn + (int64_t) t*hc_dim + src) = make_float4(xv.x*gv.x, xv.y*gv.y, xv.z*gv.z, xv.w*gv.w);
    }
    const float v = hcf_owner_sum(sq);
    if (lane == 0) {
        sums[warp] = v;
    }
    __syncthreads();
    if (warp == 0) {
        const float total = hcf_owner_sum(lane < 8 ? sums[lane] : 0.0f);
        if (lane == 0) {
            dst[n_rows*n_tokens + 4*t + s] = rsqrtf(total/n_embd + eps);
        }
    }
}

template <int T>
static __global__ void __launch_bounds__(256) hcf_down_row_owner(
        const float * __restrict__ xn, const char * __restrict__ wd, const int64_t sd,
        const char * __restrict__ wi, const int64_t si, const int R, const int n_rows, const int n_embd,
        const float lo_scale, const float inj_scale, float * __restrict__ dst) {
    const int lane = threadIdx.x & 31, row = blockIdx.x*8 + (threadIdx.x >> 5);
    if (row >= n_rows) {
        return;
    }
    const char * wr = row < R ? wd + (int64_t) row*sd : wi + (int64_t) (row - R)*si;
    const int hc_dim = 4*n_embd;
    float out[T] = {};
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        float acc[T] = {};
        for (int j = lane*8; j < n_embd; j += 256) {
            const uint4 pk = __ldg((const uint4 *) wr + (s*n_embd + j)/8);
            const uint32_t p[4] = { pk.x, pk.y, pk.z, pk.w };
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const float * src = xn + (int64_t) t*hc_dim + s*n_embd + j;
                const float4 a = __ldg((const float4 *) src);
                const float4 b = __ldg((const float4 *) (src + 4));
                const float v[8] = { a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w };
                float dot = 0.0f;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    dot = __fmaf_rn(hcf_bf16_lo(p[k]), v[2*k], dot);
                    dot = __fmaf_rn(hcf_bf16_hi(p[k]), v[2*k + 1], dot);
                }
                acc[t] = __fadd_rn(acc[t], dot);
            }
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
            out[t] += __ldg(dst + n_rows*T + 4*t + s)*hcf_owner_sum(acc[t]);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < T; ++t) {
            if (row < R) {
                const float a = lo_scale*out[t];
                dst[t*R + row] = a/(1.0f + expf(-a));
            } else {
                dst[R*T + t*(n_rows - R) + row - R] = 2.0f/(1.0f + expf(-inj_scale*out[t]));
            }
        }
    }
}

// Stripes split each 256-column tile. Keep the four stream sums separate
// until their RMS scales are known. The last stripe finishes its row block with coalesced partial reads.
template <int T, bool FOLD, bool LOOKAHEAD = false, bool PACK = false, int STRIPES = 4, bool STREAM_STAGE = false>
static __global__ void __launch_bounds__(128) hcf_down_owner(
        const float * __restrict__ x, const float * __restrict__ wn,
        const char * __restrict__ wd, const int64_t sd, const char * __restrict__ wi, const int64_t si,
        const int R, const int n_rows, const int n_embd, const float eps, const float lo_scale, const float inj_scale,
        float * __restrict__ partial, float * __restrict__ ssq, const int cnt_off,
        float * __restrict__ dst, const hcf_post_fold pf, const bool finish8) {
    extern __shared__ __align__(16) float sh[];
    __shared__ float sq_w[4][T][4];
    __shared__ float rr[T][4];
    __shared__ int last;
    static_assert(STRIPES == 4 || STRIPES == 8, "unsupported stripe count");
    constexpr int VALUES = 32/STRIPES, TILE = 256/STRIPES, OWNERS = 8*STRIPES;
    using packed_t = std::conditional_t<STRIPES == 4, uint4, uint2>;
    const int hc_dim = 4*n_embd, slice = hc_dim/STRIPES;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int local = tid & 7, stripe = blockIdx.y, owner = stripe*8 + local;
    const int row = blockIdx.x*HCF_OWNER_ROWS + tid/8;
    // Keep the indexed legacy scratch array out of the streamed kernel's frame.
    if constexpr (STREAM_STAGE) {
        const int stream_slice = n_embd/STRIPES;
#pragma unroll
        for (int t = 0; t < T; ++t) {
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                float sq = 0.0f;
                for (int j = tid*4; j < stream_slice; j += 128*4) {
                    const int src = s*n_embd + (j/TILE)*256 + stripe*TILE + j%TILE;
                    float4 xv;
                    if constexpr (FOLD) {
                        xv = hcf_post_val4(pf, t, s, src - s*n_embd, src, hc_dim);
                        if (blockIdx.x == 0) {
                            *(float4 *) (pf.h + (int64_t) t*hc_dim + src) = xv;
                        }
                    } else {
                        xv = *(const float4 *) (x + (int64_t) t*hc_dim + src);
                    }
                    const float4 gv = *(const float4 *) (wn + src);
                    sq += xv.x*xv.x + xv.y*xv.y + xv.z*xv.z + xv.w*xv.w;
                    *(float4 *) (sh + t*slice + s*stream_slice + j) = make_float4(xv.x*gv.x, xv.y*gv.y, xv.z*gv.z, xv.w*gv.w);
                }
                const float v = hcf_owner_sum(sq);
                if (lane == 0) {
                    sq_w[warp][t][s] = v;
                }
            }
        }
    } else {
        float sq[T][4] = {};
        for (int i = tid; i < T*(slice/4); i += 128) {
            const int t = i/(slice/4), j = (i % (slice/4))*4;
            const int src = (j/TILE)*256 + stripe*TILE + j%TILE;
            const int s = src/n_embd;
            float4 xv;
            if constexpr (FOLD) {
                xv = hcf_post_val4(pf, t, s, src - s*n_embd, src, hc_dim);
                if (blockIdx.x == 0) {
                    *(float4 *) (pf.h + (int64_t) t*hc_dim + src) = xv;
                }
            } else {
                xv = *(const float4 *) (x + (int64_t) t*hc_dim + src);
            }
            const float4 gv = *(const float4 *) (wn + src);
            sq[t][s] += xv.x*xv.x + xv.y*xv.y + xv.z*xv.z + xv.w*xv.w;
            *(float4 *) (sh + t*slice + j) = make_float4(xv.x*gv.x, xv.y*gv.y, xv.z*gv.z, xv.w*gv.w);
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const float v = hcf_owner_sum(sq[t][s]);
                if (lane == 0) {
                    sq_w[warp][t][s] = v;
                }
            }
        }
    }
    __syncthreads();
    if (tid < T*4) {
        float v = 0.0f;
        for (int w = 0; w < 4; ++w) {
            v += sq_w[w][tid/4][tid%4];
        }
        ssq[(((int64_t) blockIdx.x*STRIPES + stripe)*T + tid/4)*4 + tid%4] = v;
    }
    const unsigned mask = PACK ? __ballot_sync(0xffffffffu, row < n_rows) : 0;
    if (row < n_rows) {
        const char * wr = row < R ? wd + (int64_t) row*sd : wi + (int64_t) (row - R)*si;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            float acc[T] = {};
            const int first = s*(n_embd/256), end = (s + 1)*(n_embd/256);
            packed_t next;
            if constexpr (LOOKAHEAD) {
                next = __ldg((const packed_t *) wr + first*OWNERS + owner);
            }
            for (int q = first; q < end; ++q) {
                packed_t pk;
                if constexpr (LOOKAHEAD) {
                    pk = next;
                    if (q + 1 < end) {
                        next = __ldg((const packed_t *) wr + (q + 1)*OWNERS + owner);
                    }
                } else {
                    pk = __ldg((const packed_t *) wr + q*OWNERS + owner);
                }
                uint32_t p[4] = { pk.x, pk.y, 0, 0 };
                if constexpr (STRIPES == 4) {
                    p[2] = pk.z;
                    p[3] = pk.w;
                }
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    const float * xv = sh + t*slice + q*TILE + local*VALUES;
                    float v = 0.0f;
#pragma unroll
                    for (int k = 0; k < VALUES/2; ++k) {
                        v = __fmaf_rn(hcf_bf16_lo(p[k]), xv[2*k], v);
                        v = __fmaf_rn(hcf_bf16_hi(p[k]), xv[2*k + 1], v);
                    }
                    acc[t] = __fadd_rn(acc[t], v);
                }
            }
#pragma unroll
            for (int t = 0; t < T; ++t) {
                if constexpr (PACK) {
                    float v = acc[t];
#pragma unroll
                    for (int d = 4; d; d >>= 1) {
                        v = __fadd_rn(v, __shfl_xor_sync(mask, v, d, 8));
                    }
                    if (local == 0) {
                        partial[(((int64_t) s*T + t)*n_rows + row)*STRIPES + stripe] = v;
                    }
                } else {
                    partial[(((int64_t) s*T + t)*n_rows + row)*OWNERS + owner] = acc[t];
                }
            }
        }
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        last = atomicAdd(hcf_owner_arrive + cnt_off + blockIdx.x, 1u) == STRIPES - 1;
    }
    __syncthreads();
    if (!last) {
        return;
    }
    __threadfence();
    if (tid < T*4) {
        float v = 0.0f;
        for (int st = 0; st < STRIPES; ++st) {
            v += __ldcg(ssq + (((int64_t) blockIdx.x*STRIPES + st)*T + tid/4)*4 + tid%4);
        }
        const float r = rsqrtf(v/n_embd + eps);
        rr[tid/4][tid%4] = r;
        if (blockIdx.x == 0) {
            dst[n_rows*T + tid] = r;
        }
    }
    __syncthreads();
    const bool finish_pack = PACK && finish8;
    const int rows_pw = finish_pack ? 4 : 1;
    const int out_lane = finish_pack ? lane % 8 : lane;
    // Packed rows use at most eight values; all 16 rows can finish in one pass.
    for (int base = warp*rows_pw; base < HCF_OWNER_ROWS; base += 4*rows_pw) {
        const int r = blockIdx.x*HCF_OWNER_ROWS + base + (finish_pack ? lane/8 : 0);
        const unsigned out_mask = finish_pack ? __ballot_sync(0xffffffffu, r < n_rows) : 0xffffffffu;
        if (r >= n_rows) {
            continue;
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
            float v = 0.0f;
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                float p;
                if constexpr (PACK) {
                    p = out_lane < STRIPES ? __ldcg(partial + (((int64_t) s*T + t)*n_rows + r)*STRIPES + out_lane) : 0.0f;
                } else {
                    const float * src = partial + (((int64_t) s*T + t)*n_rows + r)*OWNERS;
                    p = __ldcg(src + lane);
                    if constexpr (STRIPES == 8) {
                        p = __fadd_rn(p, __ldcg(src + lane + 32));
                    }
                }
                v += rr[t][s]*(finish_pack ? hcf_owner_sum<8>(p, out_mask) : hcf_owner_sum(p));
            }
            if (out_lane == 0) {
                if (r < R) {
                    const float a = lo_scale*v;
                    dst[t*R + r] = a/(1.0f + expf(-a));
                } else {
                    dst[R*T + t*(n_rows - R) + r - R] = 2.0f/(1.0f + expf(-inj_scale*v));
                }
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        atomicExch(hcf_owner_arrive + cnt_off + blockIdx.x, 0u);
    }
}

// grid (ceil(n_rows/32), n_chunks), block 256
// P   [n_chunks][n_rows][T]       partial dot products (without r)
// ssq [row block][n_chunks][T]    partial sums of x^2 (only row block 0 when cnt == nullptr)
// cnt != nullptr: the last CTA to finish a row block (over all chunks) also runs the finish for its 32 rows
// (same sums in the same order as hcf_down_finish, so the result is bit-identical to the two-kernel path).
template <int T, bool FOLD, int HCF_DOWN_ROWS, bool WARP_FINISH = false, bool LATE_WEIGHTS = false, int HCF_DOWN_WARPS = HCF_DOWN_WARPS_DEF>
static __global__ void __launch_bounds__(HCF_DOWN_WARPS*WARP_SIZE)
hcf_down_partial(
        const float * __restrict__ x, const float * __restrict__ wn,
        const char * __restrict__ wd, const int64_t sd,
        const char * __restrict__ wi, const int64_t si,
        const int R, const int n_rows, const int hc_dim,
        float * __restrict__ P, float * __restrict__ ssq,
        const int cnt_off, const int n_embd, const float eps, const float lo_scale, const float inj_scale,
        float * __restrict__ dst, const hcf_post_fold pf) {
    __shared__ float4 xs[T][HCF_CHUNK/4];
    __shared__ float  rr[4][T];
    __shared__ int    last_s;

    const int c    = blockIdx.y;
    const int j0   = c*HCF_CHUNK;
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int n_chunks = gridDim.y;
    const bool fuse_finish = cnt_off >= 0;

    // each lane covers 4 columns in each of 4 groups of 128 (8-byte loads, a 256-byte span per warp per group);
    // the weight loads go out first so their DRAM latency overlaps the activation staging below
    constexpr int NG = HCF_CHUNK/128;
    const int k_base = blockIdx.x*(HCF_DOWN_WARPS*HCF_DOWN_ROWS) + warp*HCF_DOWN_ROWS;

    constexpr int NP = LATE_WEIGHTS ? 1 : NG;
    uint2 w[HCF_DOWN_ROWS][NP];
#pragma unroll
    for (int q = 0; q < HCF_DOWN_ROWS; ++q) {
        const int k = k_base + q;
        if (k < n_rows) {
            const char * row = k < R ? wd + (int64_t) k*sd : wi + (int64_t) (k - R)*si;
            const uint2 * rp = (const uint2 *) (row + (int64_t) j0*2);
#pragma unroll
            for (int g = 0; g < NP; ++g) {
                w[q][g] = rp[g*WARP_SIZE + lane];
            }
        } else {
#pragma unroll
            for (int g = 0; g < NP; ++g) {
                w[q][g] = make_uint2(0, 0);
            }
        }
    }

    // a chunk never straddles two streams (n_embd is a multiple of HCF_CHUNK)
    const int s_c  = j0 / n_embd;
    const int i0_c = j0 - s_c*n_embd;

    for (int idx = threadIdx.x; idx < T*(HCF_CHUNK/4); idx += blockDim.x) {
        const int t = idx / (HCF_CHUNK/4);
        const int e = idx % (HCF_CHUNK/4);
        float4 xv;
        if constexpr (FOLD) {
            xv = hcf_post_val4(pf, t, s_c, i0_c + 4*e, j0 + 4*e, hc_dim);
            if (blockIdx.x == 0) {
                *((float4 *) (pf.h + (int64_t) t*hc_dim + j0) + e) = xv;
            }
        } else {
            xv = *((const float4 *) (x + (int64_t) t*hc_dim + j0) + e);
        }
        const float4 gv = *((const float4 *) (wn + j0) + e);
        xs[t][e] = make_float4(xv.x*gv.x, xv.y*gv.y, xv.z*gv.z, xv.w*gv.w);
    }

    if (fuse_finish || blockIdx.x == 0) {
        for (int t = warp; t < T; t += HCF_DOWN_WARPS) {
            const float4 * xr = (const float4 *) (x + (int64_t) t*hc_dim + j0);
            float s = 0.0f;
#pragma unroll
            for (int it = 0; it < HCF_CHUNK/4/WARP_SIZE; ++it) {
                float4 v;
                if constexpr (FOLD) {
                    const int e = it*WARP_SIZE + lane;
                    v = hcf_post_val4(pf, t, s_c, i0_c + 4*e, j0 + 4*e, hc_dim);
                } else {
                    v = xr[it*WARP_SIZE + lane];
                }
                s += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
            }
            s = warp_reduce_sum(s);
            if (lane == 0) {
                ssq[((int64_t) blockIdx.x*n_chunks + c)*T + t] = s;
            }
        }
    }

    __syncthreads();

    float acc[HCF_DOWN_ROWS][T];
#pragma unroll
    for (int q = 0; q < HCF_DOWN_ROWS; ++q) {
#pragma unroll
        for (int t = 0; t < T; ++t) {
            acc[q][t] = 0.0f;
        }
    }

#pragma unroll
    for (int g = 0; g < NG; ++g) {
        float wf[HCF_DOWN_ROWS][4];
#pragma unroll
        for (int q = 0; q < HCF_DOWN_ROWS; ++q) {
            uint2 v;
            if constexpr (LATE_WEIGHTS) {
                v = w[q][0];
                if (g > 0 && k_base + q < n_rows) {
                    const int k = k_base + q;
                    const char * row = k < R ? wd + (int64_t) k*sd : wi + (int64_t) (k - R)*si;
                    const uint2 * rp = (const uint2 *) (row + (int64_t) j0*2);
                    v = rp[g*WARP_SIZE + lane];
                }
            } else {
                v = w[q][g];
            }
            wf[q][0] = hcf_bf16_lo(v.x);
            wf[q][1] = hcf_bf16_hi(v.x);
            wf[q][2] = hcf_bf16_lo(v.y);
            wf[q][3] = hcf_bf16_hi(v.y);
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float4 xv = xs[t][g*WARP_SIZE + lane];
#pragma unroll
            for (int q = 0; q < HCF_DOWN_ROWS; ++q) {
                acc[q][t] += wf[q][0]*xv.x + wf[q][1]*xv.y + wf[q][2]*xv.z + wf[q][3]*xv.w;
            }
        }
    }

#pragma unroll
    for (int q = 0; q < HCF_DOWN_ROWS; ++q) {
#pragma unroll
        for (int t = 0; t < T; ++t) {
            acc[q][t] = warp_reduce_sum(acc[q][t]);
        }
    }

    if (lane == 0) {
#pragma unroll
        for (int q = 0; q < HCF_DOWN_ROWS; ++q) {
            const int k = k_base + q;
            if (k < n_rows) {
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    P[((int64_t) c*n_rows + k)*T + t] = acc[q][t];
                }
            }
        }
    }

    if (!fuse_finish) {
        return;
    }

    // publish P / ssq, then count this CTA in; the last of the n_chunks CTAs of this row block does the finish
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned int prev = atomicAdd(&hcf_arrive[cnt_off + blockIdx.x], 1u);
        last_s = prev == (unsigned int) (n_chunks - 1);
    }
    __syncthreads();
    if (!last_s) {
        return;
    }
    __threadfence();

    constexpr int hc = 4;
    const int cps = n_embd/HCF_CHUNK;  // chunks per stream
    const float * ssq_rb = ssq + (int64_t) blockIdx.x*n_chunks*T;
    float r_warp[hc] = {};
    if constexpr (WARP_FINISH) {
#pragma unroll
        for (int s = 0; s < hc; ++s) {
            if (lane < T) {
                float sq = 0.0f;
                for (int cc = 0; cc < cps; ++cc) {
                    sq += __ldcg(ssq_rb + (s*cps + cc)*T + lane);
                }
                r_warp[s] = rsqrtf(sq/n_embd + eps);
                if (blockIdx.x == 0 && warp == 0) {
                    dst[n_rows*T + lane*hc + s] = r_warp[s];
                }
            }
        }
    } else {
        if (threadIdx.x < hc*T) {
            const int s = threadIdx.x / T;
            const int t = threadIdx.x % T;
            float sq = 0.0f;
            for (int cc = 0; cc < cps; ++cc) {
                sq += __ldcg(ssq_rb + (s*cps + cc)*T + t);
            }
            const float rv = rsqrtf(sq/n_embd + eps);
            rr[s][t] = rv;
            if (blockIdx.x == 0) {
                dst[n_rows*T + t*hc + s] = rv;
            }
        }
        __syncthreads();
    }

    for (int base = 0; base < (HCF_DOWN_WARPS*HCF_DOWN_ROWS)*T; base += blockDim.x) {
        const int o = base + threadIdx.x;
        const int k = blockIdx.x*(HCF_DOWN_WARPS*HCF_DOWN_ROWS) + o / T;
        const int t = o % T;
        float r_token[hc];
        if constexpr (WARP_FINISH) {
#pragma unroll
            for (int s = 0; s < hc; ++s) {
                r_token[s] = __shfl_sync(0xffffffff, r_warp[s], t, WARP_SIZE);
            }
        }
        if (o >= (HCF_DOWN_WARPS*HCF_DOWN_ROWS)*T || k >= n_rows) {
            continue;
        }
        float v = 0.0f;
#pragma unroll
        for (int s = 0; s < hc; ++s) {
            float part = 0.0f;
#pragma unroll 8
            for (int cc = 0; cc < cps; ++cc) {
                part += __ldcg(P + ((int64_t) (s*cps + cc)*n_rows + k)*T + t);
            }
            if constexpr (WARP_FINISH) {
                v += r_token[s] * part;
            } else {
                v += rr[s][t] * part;
            }
        }
        if (k < R) {
            const float a = lo_scale*v;
            dst[t*R + k] = a/(1.0f + expf(-a));
        } else {
            dst[R*T + t*(n_rows - R) + (k - R)] = 2.0f/(1.0f + expf(-inj_scale*v));
        }
    }

    if (threadIdx.x == 0) {
        hcf_arrive[cnt_off + blockIdx.x] = 0u;
    }
}

// one thread per (row, token); dst = [act R*T | inj hc_inj*T | r hc*T]
static __global__ void hcf_down_finish(
        const float * __restrict__ P, const float * __restrict__ ssq,
        const int T, const int R, const int hc_inj, const int hc, const int n_embd,
        const float eps, const float lo_scale, const float inj_scale,
        float * __restrict__ dst) {
    const int n_rows   = R + hc_inj;
    const int n_chunks = hc*n_embd/HCF_CHUNK;
    const int cps      = n_embd/HCF_CHUNK;  // chunks per stream
    const int idx      = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= (n_rows + hc)*T) {
        return;
    }

    if (idx >= n_rows*T) {
        // r segment
        const int u = idx - n_rows*T;
        const int t = u / hc;
        const int s = u % hc;
        float sq = 0.0f;
        for (int cc = 0; cc < cps; ++cc) {
            sq += ssq[(s*cps + cc)*T + t];
        }
        dst[n_rows*T + t*hc + s] = rsqrtf(sq/n_embd + eps);
        return;
    }

    const int k = idx / T;
    const int t = idx % T;

    float v = 0.0f;
    for (int s = 0; s < hc; ++s) {
        float sq   = 0.0f;
        float part = 0.0f;
        for (int cc = 0; cc < cps; ++cc) {
            const int c = s*cps + cc;
            sq   += ssq[c*T + t];
            part += P[((int64_t) c*n_rows + k)*T + t];
        }
        v += rsqrtf(sq/n_embd + eps) * part;
    }
    GGML_UNUSED(n_chunks);

    if (k < R) {
        const float a = lo_scale*v;
        dst[t*R + k] = a/(1.0f + expf(-a));
    } else {
        dst[R*T + t*hc_inj + (k - R)] = 2.0f/(1.0f + expf(-inj_scale*v));
    }
}

// grid ceil(n_embd/16), block 512, dynamic shared R*T floats; hc == 4 (one 8-lane group per stream), R <= 64*HCF_UP_MAX_IT
// The first HCF_UP_PRE passes of the W_up row (all of it for R <= 320) and the x / r / w_norm values are loaded before
// the activations are staged, so a warp's global loads are in flight together (v1 issued one 16-byte weight load per
// lane per loop pass); the sums are unchanged. Two CTAs per SM keep all 160 CTAs of n_embd = 2560 in one wave.
static constexpr int HCF_UP_MAX_IT = 8;
static constexpr int HCF_UP_PRE    = 5;
// [MOEGAP HCUPQ8] Q = 1 (plain) or 2 (IQ4_XS swizzle): the mix output is also written as q8_1 blocks, the bytes
// of quantize_row_q8_1_cuda on dst (see hcf_up_q8_block). The two CTAs of a 32-column block count in on cnt; the
// second one quantizes the block from dst and resets the counter, so the counters are zero between launches.
static constexpr int HCF_UP_CNT = 256;  // q8 blocks per counter slot: n_embd <= 8192
static __device__ unsigned int hcf_up_arrive[HCF_CNT_SLOTS*HCF_UP_CNT];

// quantize_q8_1<swizzle> for one 32-value block held one value per lane (same expressions, so the same bytes)
template <bool swizzle_iq4>
static __device__ __forceinline__ void hcf_up_q8_block(const float xi, block_q8_1 * y, const int iqs) {
    float amax = fabsf(xi);
    float sum = 0.0f;

    amax = warp_reduce_max<QK8_1>(amax);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    if constexpr (swizzle_iq4) {
        y->qs[2*(iqs % 16) + iqs/16] = q;
    } else {
        y->qs[iqs] = q;
    }
    sum = warp_reduce_sum<QK8_1>((float) q);

    if (iqs > 0) {
        return;
    }

    y->ds = make_half2(d, d * sum);
}

template <int T, int Q = 0, int W = HCF_UP_WARPS, int PRE = HCF_UP_PRE>
static __global__ void __launch_bounds__(W*WARP_SIZE, 32/W)
hcf_up_gate(
        const float * __restrict__ x, const float * __restrict__ wn,
        const float * __restrict__ act, const float * __restrict__ r,
        const char * __restrict__ wu, const int64_t su,
        const int n_embd, const int R, const float out_scale,
        float * __restrict__ dst, block_q8_1 * __restrict__ q8 = nullptr, const int cnt_off = 0) {
    extern __shared__ float act_s[];  // [T][R]

    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int i    = blockIdx.x*W + warp;
    const bool live = i < n_embd;

    constexpr int hc = 4;
    const int s  = lane / 8;
    const int l8 = lane % 8;
    const int j  = s*n_embd + (live ? i : 0);

    const uint4 * rp = (const uint4 * ) (wu + (int64_t) j*su);
    const int n_it = R/64;  // uint4 (8 values) per lane per pass, 8 lanes per row

    uint4 wv[PRE];
#pragma unroll
    for (int it = 0; it < PRE; ++it) {
        wv[it] = (live && it < n_it) ? rp[it*8 + l8] : make_uint4(0, 0, 0, 0);
    }
    float xn[T];
    const float wnj = live ? wn[j] : 0.0f;
#pragma unroll
    for (int t = 0; t < T; ++t) {
        xn[t] = live ? x[(int64_t) t*hc*n_embd + j] : 0.0f;
    }
    __shared__ float r_s[HCF_MAX_T*hc];
    for (int idx = threadIdx.x; idx < T*R/4; idx += blockDim.x) {
        ((float4 *) act_s)[idx] = ((const float4 *) act)[idx];
    }
    if (threadIdx.x < T*hc) {
        r_s[threadIdx.x] = r[threadIdx.x];
    }
    __syncthreads();

    if (!live) {
        return;
    }

    float g[T];
#pragma unroll
    for (int t = 0; t < T; ++t) {
        g[t] = 0.0f;
    }

#pragma unroll
    for (int it = 0; it < HCF_UP_MAX_IT; ++it) {
        if (it >= n_it) {
            break;
        }
        const int    u  = it*8 + l8;
        const uint4  q  = it < PRE ? wv[it < PRE ? it : 0] : rp[u];
        const float  w0 = hcf_bf16_lo(q.x), w1 = hcf_bf16_hi(q.x), w2 = hcf_bf16_lo(q.y), w3 = hcf_bf16_hi(q.y);
        const float  w4 = hcf_bf16_lo(q.z), w5 = hcf_bf16_hi(q.z), w6 = hcf_bf16_lo(q.w), w7 = hcf_bf16_hi(q.w);
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float4 a0 = ((const float4 *) (act_s + t*R))[2*u + 0];
            const float4 a1 = ((const float4 *) (act_s + t*R))[2*u + 1];
            g[t] += w0*a0.x + w1*a0.y + w2*a0.z + w3*a0.w + w4*a1.x + w5*a1.y + w6*a1.z + w7*a1.w;
        }
    }

    float out[T];
#pragma unroll
    for (int t = 0; t < T; ++t) {
        float gt = g[t];
        gt += __shfl_xor_sync(0xFFFFFFFF, gt, 4);
        gt += __shfl_xor_sync(0xFFFFFFFF, gt, 2);
        gt += __shfl_xor_sync(0xFFFFFFFF, gt, 1);
        const float xnv = (xn[t]*r_s[t*hc + s])*wnj;
        float v = xnv*(1.0f/(1.0f + expf(-gt)));
        v += __shfl_xor_sync(0xFFFFFFFF, v, 8);
        v += __shfl_xor_sync(0xFFFFFFFF, v, 16);
        out[t] = v;
    }

#pragma unroll
    for (int t = 0; t < T; ++t) {
        if (lane == t) {
            dst[(int64_t) t*n_embd + i] = out_scale*out[t];
        }
    }

    if constexpr (Q != 0) {
        // every warp is live here (the host requires n_embd % 32 == 0)
        static_assert(32 % W == 0, "two or more CTAs per q8 block");
        constexpr unsigned int n_cta = 32 / W;
        __shared__ unsigned int last_s;
        const int blk = blockIdx.x*W / 32;
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned int * cnt = hcf_up_arrive + cnt_off;
            const unsigned int old = atomicAdd(&cnt[blk], 1u);
            last_s = old == n_cta - 1;
            if (old == n_cta - 1) {
                cnt[blk] = 0;
            }
        }
        __syncthreads();
        if (last_s) {
            __threadfence();
            for (int t = warp; t < T; t += W) {
                const float xi = __ldcg(dst + (int64_t) t*n_embd + blk*32 + lane);
                hcf_up_q8_block<Q == 2>(xi, q8 + (int64_t) t*(n_embd/QK8_1) + blk, lane);
            }
        }
    }
}

static bool hcf_common_supported(const ggml_tensor * x, const ggml_tensor * wn) {
    return x->type == GGML_TYPE_F32 && wn->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(x) && ggml_is_contiguous(wn) &&
        x->ne[1] == 4 && x->ne[2] >= 1 && x->ne[2] <= HCF_MAX_T && x->ne[3] == 1 &&
        x->ne[0] % HCF_CHUNK == 0;
}

static bool hcf_weight_ok(const ggml_tensor * w) {
    return w->type == GGML_TYPE_BF16 && w->nb[0] == 2 && w->nb[1] % 16 == 0 &&
        ((uintptr_t) w->data) % 16 == 0;
}

bool ggml_cuda_dsv4_hc_down_supported(const ggml_tensor * op) {
    const ggml_tensor * x  = op->src[0];
    const ggml_tensor * wd = op->src[2];
    const ggml_tensor * wi = op->src[3];
    // data pointers may still be null at the supports check; the alignment test then passes trivially
    return hcf_common_supported(x, op->src[1]) && hcf_weight_ok(wd) && (wi == nullptr || hcf_weight_ok(wi));
}

bool ggml_cuda_dsv4_hc_up_supported(const ggml_tensor * op) {
    const ggml_tensor * x  = op->src[0];
    const ggml_tensor * wu = op->src[3];
    return hcf_common_supported(x, op->src[1]) && hcf_weight_ok(wu) && wu->ne[0] % 64 == 0 &&
        wu->ne[0] <= 64*HCF_UP_MAX_IT;
}

// [HCFUSED v2] arrival-counter slot for this backend context (-1: none left, or GGML_CUDA_HC_FUSED_FINISH=0,
// which keeps the separate finish kernel)
static int hcf_cnt_slot(const ggml_backend_cuda_context & ctx) {
    static const bool fuse = [] { const char * e = getenv("GGML_CUDA_HC_FUSED_FINISH"); return !e || atoi(e) != 0; }();
    if (!fuse) {
        return -1;
    }
    static std::mutex mtx;
    static std::unordered_map<const void *, int> slots;
    static int next = 0;  // one numbering for all devices: virtual devices may share a GPU (and its counters)
    std::lock_guard<std::mutex> lock(mtx);
    const auto it = slots.find(&ctx);
    if (it != slots.end()) {
        return it->second;
    }
    int slot = -1;
    if (next < HCF_CNT_SLOTS) {
        slot = next++;
    }
    slots[&ctx] = slot;
    return slot;
}

static int hcf_owner_cnt_slot(const ggml_backend_cuda_context & ctx) {
    static std::mutex mtx;
    static std::unordered_map<const void *, int> slots;
    static int next = 0;
    std::lock_guard<std::mutex> lock(mtx);
    const auto it = slots.find(&ctx);
    if (it != slots.end()) {
        return it->second;
    }
    const int slot = next < HCF_CNT_SLOTS ? next++ : -1;
    slots[&ctx] = slot;
    return slot;
}

template <int T>
static void hcf_down_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * hpost) {
    const ggml_tensor * x  = dst->src[0];
    const ggml_tensor * wn = dst->src[1];
    const ggml_tensor * wd = dst->src[2];
    const ggml_tensor * wi = dst->src[3];

    const float   eps       = ggml_get_op_params_f32(dst, 0);
    const int32_t hc_inj    = ggml_get_op_params_i32(dst, 1);
    const float   lo_scale  = ggml_get_op_params_f32(dst, 2);
    const float   inj_scale = ggml_get_op_params_f32(dst, 3);

    const int n_embd   = (int) x->ne[0];
    const int hc       = (int) x->ne[1];
    const int hc_dim   = hc*n_embd;
    const int R        = (int) wd->ne[1];
    const int n_rows   = R + hc_inj;
    const int n_chunks = hc_dim/HCF_CHUNK;
    // rows per warp: 4 by default; LLAMA_HC_DOWN_ROWS=2|1 launches 2x|4x as many CTAs (the same per-row sums in the
    // same order, so the result is bit-identical). The counter slots hold 64 row blocks, so a larger count falls back.
    static const int rows_env = [] {
        const char * e = getenv("LLAMA_HC_DOWN_ROWS");
        const int v = e ? atoi(e) : HCF_DOWN_ROWS_DEF;
        return v == 1 || v == 2 ? v : HCF_DOWN_ROWS_DEF;
    }();
    static const bool warp_finish = [] {
        const char * e = getenv("LLAMA_HC_DOWN_WARP_FINISH");
        return e && atoi(e) == 1;
    }();
    // Keep one weight group live across activation staging; fetch the rest at use.
    static const bool late_weights = [] {
        const char * e = getenv("LLAMA_HC_DOWN_LATE_WEIGHTS");
        return e && atoi(e) == 1;
    }();
    static const int warps_env = [] {
        const char * e = getenv("LLAMA_HC_DOWN_WARPS");
        return e && atoi(e) == 4 ? 4 : HCF_DOWN_WARPS_DEF;
    }();
    const int down_w = warps_env;
    const int rows_pw = (n_rows + down_w*rows_env - 1)/(down_w*rows_env) <= HCF_CNT_RB ? rows_env : HCF_DOWN_ROWS_DEF;
    const int n_rb     = (n_rows + down_w*rows_pw - 1)/(down_w*rows_pw);

    GGML_ASSERT(((uintptr_t) x->data) % 16 == 0 && ((uintptr_t) wn->data) % 16 == 0);
    GGML_ASSERT(hc == 4);

    cudaStream_t stream = ctx.stream();

    const int slot    = n_rb <= HCF_CNT_RB ? hcf_cnt_slot(ctx) : -1;
    const int cnt_off = slot >= 0 ? slot*HCF_CNT_RB : -1;
    hcf_post_fold pf = {};
    if (hpost != nullptr) {
        pf.blk  = (const float *) hpost->src[0]->data;
        pf.res  = (const float *) hpost->src[1]->data;
        pf.post = (const float *) hpost->src[2]->data;
        pf.h    = (float *) hpost->data;
        pf.sb1  = hpost->src[0]->nb[1]/sizeof(float);
        pf.sp0  = hpost->src[2]->nb[0]/sizeof(float);
        pf.sp1  = hpost->src[2]->nb[1]/sizeof(float);
        GGML_ASSERT(pf.h == x->data);
    }
    static const bool row_owner = [] {
        const char * e = getenv("LLAMA_HC_DOWN_ROW_OWNER");
        return e && atoi(e) == 1;
    }();
    if (row_owner && slot >= 0 && n_embd % 256 == 0) {
        ggml_cuda_pool_alloc<float> xn(ctx.pool(), (size_t) T*hc_dim);
        auto stage = [&](auto fold) {
            constexpr bool F = decltype(fold)::value;
            static const bool reported = [=] {
                fprintf(stderr, "[hc-down-row-owner] T=%d rows=%d embd=%d folded=%d\n", T, n_rows, n_embd, F);
                return true;
            }();
            GGML_UNUSED(reported);
            hcf_row_owner_stage<F><<<dim3(4, T), 256, 0, stream>>>((const float *) x->data, (const float *) wn->data,
                    n_embd, n_rows, T, eps, xn.get(), (float *) dst->data, pf);
        };
        if (hpost != nullptr) {
            stage(std::true_type{});
        } else {
            stage(std::false_type{});
        }
        hcf_down_row_owner<T><<<(n_rows + 7)/8, 256, 0, stream>>>(xn.get(),
                (const char *) wd->data, wd->nb[1], wi ? (const char *) wi->data : nullptr, wi ? wi->nb[1] : 0,
                R, n_rows, n_embd, lo_scale, inj_scale, (float *) dst->data);
        return;
    }
    static const bool owner = [] {
        const char * e = getenv("LLAMA_HC_DOWN_OWNER");
        return e && atoi(e) == 1;
    }();
    static const bool owner_lookahead = [] {
        const char * e = getenv("LLAMA_HC_DOWN_OWNER_LOOKAHEAD");
        return e && atoi(e) == 1;
    }();
    static const bool owner_pack = [] {
        const char * e = getenv("LLAMA_HC_DOWN_OWNER_PACK");
        return e && atoi(e) == 1;
    }();
    static const bool owner_finish8 = [] {
        const char * e = getenv("LLAMA_HC_DOWN_OWNER_FINISH8");
        return e && atoi(e) == 1;
    }();
    static const int owner_stripes = [] {
        const char * e = getenv("LLAMA_HC_DOWN_OWNER_STRIPES");
        return e && atoi(e) == 8 ? 8 : 4;
    }();
    static const bool owner_stream_stage = [] {
        const char * e = getenv("LLAMA_HC_DOWN_OWNER_STREAM_STAGE");
        return e && atoi(e) == 1;
    }();
    const int owner_rb = (n_rows + HCF_OWNER_ROWS - 1)/HCF_OWNER_ROWS;
    const size_t owner_smem = (size_t) T*n_embd*sizeof(float)*4/owner_stripes;
    const size_t owner_static = (5*T*4 + 1)*sizeof(float);
    const size_t owner_limit = std::min<size_t>(81920, ggml_cuda_info().devices[ctx.device].smpbo > owner_static ?
            ggml_cuda_info().devices[ctx.device].smpbo - owner_static : 0);
    // A disabled fused finish or exhausted slot table keeps the original two-kernel fallback.
    const int owner_slot = owner && slot >= 0 && n_rows <= HCF_OWNER_MAX_ROWS && n_embd % 256 == 0 &&
            owner_smem <= owner_limit ? hcf_owner_cnt_slot(ctx) : -1;
    if (owner_slot >= 0) {
        auto select = [&](auto fold, auto lookahead, auto pack, auto stripes, auto stage) {
            constexpr bool F = decltype(fold)::value;
            constexpr bool L = decltype(lookahead)::value;
            constexpr bool P = decltype(pack)::value;
            constexpr int S = decltype(stripes)::value;
            constexpr bool SS = decltype(stage)::value;
            auto k = hcf_down_owner<T, F, L, P, S, SS>;
            CUDA_SET_SHARED_MEMORY_LIMIT(k, owner_limit);
            static const bool reported = [=] {
                fprintf(stderr, "[hc-down-owner] T=%d rows=%d embd=%d smem=%zu folded=%d lookahead=%d pack=%d finish8=%d stripes=%d stream_stage=%d\n",
                        T, n_rows, n_embd, owner_smem, F, L, P, P && owner_finish8, S, SS);
                return true;
            }();
            GGML_UNUSED(reported);
            return k;
        };
        auto select_stage = [&](auto fold, auto lookahead, auto pack, auto stripes) {
            return owner_stream_stage ? select(fold, lookahead, pack, stripes, std::true_type{}) :
                    select(fold, lookahead, pack, stripes, std::false_type{});
        };
        auto select_stripes = [&](auto fold, auto lookahead, auto pack) {
            return owner_stripes == 8 ? select_stage(fold, lookahead, pack, std::integral_constant<int, 8>{}) :
                    select_stage(fold, lookahead, pack, std::integral_constant<int, 4>{});
        };
        auto select_pack = [&](auto fold, auto lookahead) {
            return owner_pack ? select_stripes(fold, lookahead, std::true_type{}) : select_stripes(fold, lookahead, std::false_type{});
        };
        auto kernel = hpost != nullptr ?
                (owner_lookahead ? select_pack(std::true_type{}, std::true_type{}) : select_pack(std::true_type{}, std::false_type{})) :
                (owner_lookahead ? select_pack(std::false_type{}, std::true_type{}) : select_pack(std::false_type{}, std::false_type{}));
        // Sum each eight-lane row group before the global store to cut the partial count by eight.
        ggml_cuda_pool_alloc<float> partial(ctx.pool(), (size_t) 4*T*n_rows*owner_stripes*(owner_pack ? 1 : 8));
        ggml_cuda_pool_alloc<float> sq(ctx.pool(), (size_t) owner_rb*owner_stripes*T*4);
        kernel<<<dim3(owner_rb, owner_stripes), 128, owner_smem, stream>>>(
            (const float *) x->data, (const float *) wn->data,
            (const char *) wd->data, wd->nb[1], wi ? (const char *) wi->data : nullptr, wi ? wi->nb[1] : 0,
            R, n_rows, n_embd, eps, lo_scale, inj_scale, partial.get(), sq.get(), owner_slot*HCF_OWNER_RB,
            (float *) dst->data, pf, owner_finish8);
        return;
    }
    ggml_cuda_pool_alloc<float> P(ctx.pool(), (size_t) n_chunks*n_rows*T);
    ggml_cuda_pool_alloc<float> ssq(ctx.pool(), (size_t) (cnt_off >= 0 ? n_rb : 1)*n_chunks*T);
    const dim3 grid(n_rb, n_chunks, 1);
    auto launch = [&](auto finish, auto late, auto warps) {
        constexpr bool WF = decltype(finish)::value;
        constexpr bool LW = decltype(late)::value;
        constexpr int NW = decltype(warps)::value;
        auto kernel = hpost != nullptr ?
            (rows_pw == 1 ? hcf_down_partial<T, true, 1, WF, LW, NW> : rows_pw == 2 ? hcf_down_partial<T, true, 2, WF, LW, NW> : hcf_down_partial<T, true, HCF_DOWN_ROWS_DEF, WF, LW, NW>) :
            (rows_pw == 1 ? hcf_down_partial<T, false, 1, WF, LW, NW> : rows_pw == 2 ? hcf_down_partial<T, false, 2, WF, LW, NW> : hcf_down_partial<T, false, HCF_DOWN_ROWS_DEF, WF, LW, NW>);
        kernel<<<grid, NW*WARP_SIZE, 0, stream>>>(
            (const float *) x->data, (const float *) wn->data,
            (const char *) wd->data, wd->nb[1],
            wi ? (const char *) wi->data : nullptr, wi ? wi->nb[1] : 0,
            R, n_rows, hc_dim, P.get(), ssq.get(),
            cnt_off, n_embd, eps, lo_scale, inj_scale, (float *) dst->data, pf);
    };
    auto launch_warps = [&](auto finish, auto late) {
        if (down_w == 4) {
            launch(finish, late, std::integral_constant<int, 4>{});
        } else {
            launch(finish, late, std::integral_constant<int, HCF_DOWN_WARPS_DEF>{});
        }
    };
    if (late_weights) {
        if (warp_finish) {
            launch_warps(std::true_type{}, std::true_type{});
        } else {
            launch_warps(std::false_type{}, std::true_type{});
        }
    } else {
        if (warp_finish) {
            launch_warps(std::true_type{}, std::false_type{});
        } else {
            launch_warps(std::false_type{}, std::false_type{});
        }
    }

    if (cnt_off < 0) {
        const int n_out = (n_rows + hc)*T;
        hcf_down_finish<<<(n_out + 255)/256, 256, 0, stream>>>(
            P.get(), ssq.get(), T, R, hc_inj, hc, n_embd, eps, lo_scale, inj_scale, (float *) dst->data);
    }
}

// [MOEGAP HCUPQ8] counter slot of a backend context for the q8 epilogue (-1: none left)
static int hcf_up_cnt_slot(const ggml_backend_cuda_context & ctx) {
    static std::mutex mtx;
    static std::unordered_map<const void *, int> slots;
    static int next = 0;
    std::lock_guard<std::mutex> lock(mtx);
    const auto it = slots.find(&ctx);
    if (it != slots.end()) {
        return it->second;
    }
    const int slot = next < HCF_CNT_SLOTS ? next++ : -1;
    slots[&ctx] = slot;
    return slot;
}

template <int T, int PRE>
static void hcf_up_launch_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst, void * q8, bool swizzle) {
    const ggml_tensor * x    = dst->src[0];
    const ggml_tensor * wn   = dst->src[1];
    const ggml_tensor * down = dst->src[2];
    const ggml_tensor * wu   = dst->src[3];

    const float   out_scale = ggml_get_op_params_f32(dst, 0);
    const int32_t hc_inj    = ggml_get_op_params_i32(dst, 1);

    const int n_embd = (int) x->ne[0];
    const int hc     = (int) x->ne[1];
    const int R      = (int) wu->ne[0];

    const float * act = (const float *) down->data;
    const float * r   = act + (int64_t) (R + hc_inj)*T;

    GGML_ASSERT(((uintptr_t) act) % 16 == 0);
    const size_t smem = (size_t) R*T*sizeof(float);
    GGML_ASSERT(smem <= 48*1024);
    GGML_UNUSED(hc);

    // embedding columns per CTA: 16 by default; LLAMA_HC_UP_WARPS=8|4 launches 2x|4x as many smaller CTAs (every row
    // is still one warp with the same sums, so the result is bit-identical)
    static const int up_w = [] {
        const char * e = getenv("LLAMA_HC_UP_WARPS");
        const int v = e ? atoi(e) : HCF_UP_WARPS;
        return v == 4 || v == 8 ? v : HCF_UP_WARPS;
    }();
    const dim3 grid((n_embd + up_w - 1)/up_w);
    const dim3 block(up_w*WARP_SIZE);
    if (q8 == nullptr) {
        auto kernel = up_w == 4 ? hcf_up_gate<T, 0, 4, PRE> : up_w == 8 ? hcf_up_gate<T, 0, 8, PRE> : hcf_up_gate<T, 0, HCF_UP_WARPS, PRE>;
        kernel<<<grid, block, smem, ctx.stream()>>>(
            (const float *) x->data, (const float *) wn->data, act, r,
            (const char *) wu->data, wu->nb[1], n_embd, R, out_scale, (float *) dst->data, nullptr, 0);
    } else {
        const int slot = hcf_up_cnt_slot(ctx);
        GGML_ASSERT(slot >= 0 && n_embd % QK8_1 == 0 && n_embd/QK8_1 <= HCF_UP_CNT);
        auto kernel = swizzle ?
            (up_w == 4 ? hcf_up_gate<T, 2, 4, PRE> : up_w == 8 ? hcf_up_gate<T, 2, 8, PRE> : hcf_up_gate<T, 2, HCF_UP_WARPS, PRE>) :
            (up_w == 4 ? hcf_up_gate<T, 1, 4, PRE> : up_w == 8 ? hcf_up_gate<T, 1, 8, PRE> : hcf_up_gate<T, 1, HCF_UP_WARPS, PRE>);
        kernel<<<grid, block, smem, ctx.stream()>>>(
            (const float *) x->data, (const float *) wn->data, act, r,
            (const char *) wu->data, wu->nb[1], n_embd, R, out_scale, (float *) dst->data, (block_q8_1 *) q8, slot*HCF_UP_CNT);
    }
}

template <int T>
static void hcf_up_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst, void * q8, bool swizzle) {
    // Fewer early weight loads use fewer registers; the dot products keep their order.
    static const bool preload_two = [] {
        const char * e = getenv("LLAMA_HC_UP_PRELOAD");
        return e != nullptr && atoi(e) == 2;
    }();
    if (preload_two) {
        hcf_up_launch_impl<T, 2>(ctx, dst, q8, swizzle);
    } else {
        hcf_up_launch_impl<T, HCF_UP_PRE>(ctx, dst, q8, swizzle);
    }
}

#define HCF_DISPATCH(fn, T_, ...)                                   \
    switch (T_) {                                                   \
        case 1: fn<1>(__VA_ARGS__); break;                          \
        case 2: fn<2>(__VA_ARGS__); break;                          \
        case 3: fn<3>(__VA_ARGS__); break;                          \
        case 4: fn<4>(__VA_ARGS__); break;                          \
        case 5: fn<5>(__VA_ARGS__); break;                          \
        case 6: fn<6>(__VA_ARGS__); break;                          \
        case 7: fn<7>(__VA_ARGS__); break;                          \
        case 8: fn<8>(__VA_ARGS__); break;                          \
        default: GGML_ABORT("hc fused: unsupported token count");   \
    }

void ggml_cuda_op_dsv4_hc_down(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    HCF_DISPATCH(hcf_down_launch, (int) dst->src[0]->ne[2], ctx, dst, nullptr);
}

bool ggml_cuda_dsv4_hc_post_down_supported(const ggml_tensor * hpost, const ggml_tensor * down) {
    const ggml_tensor * x   = down->src[0];
    const ggml_tensor * blk = hpost->src[0];
    const ggml_tensor * res = hpost->src[1];
    const ggml_tensor * pw  = hpost->src[2];
    const int64_t E = x->ne[0], hc = x->ne[1], T = x->ne[2];
    const auto f32 = [](const ggml_tensor * t) { return t != nullptr && t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float); };
    return ggml_cuda_dsv4_hc_down_supported(down) && hpost->src[3] == nullptr &&
        f32(hpost) && f32(blk) && f32(res) && f32(pw) && x->data == hpost->data &&
        ggml_is_contiguous(hpost) && ggml_nelements(hpost) == ggml_nelements(x) &&
        hpost->ne[0] == E && hpost->ne[1] == hc && hpost->ne[2] == T && hpost->ne[3] == 1 &&
        blk->ne[0] == E && blk->ne[1] == T && blk->ne[2] == 1 && blk->ne[3] == 1 && blk->nb[1] % 16 == 0 &&
        ((uintptr_t) blk->data) % 16 == 0 &&
        ggml_is_contiguous(res) && res->ne[0] == E && res->ne[1] == hc && res->ne[2] == T && res->ne[3] == 1 &&
        ((uintptr_t) res->data) % 16 == 0 &&
        pw->ne[0] == hc && pw->ne[1] == T && pw->ne[2] == 1 && pw->ne[3] == 1;
}

void ggml_cuda_op_dsv4_hc_post_down(ggml_backend_cuda_context & ctx, const ggml_tensor * hpost, ggml_tensor * down) {
    HCF_DISPATCH(hcf_down_launch, (int) down->src[0]->ne[2], ctx, down, hpost);
}

void ggml_cuda_op_dsv4_hc_up(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    HCF_DISPATCH(hcf_up_launch, (int) dst->src[0]->ne[2], ctx, dst, nullptr, false);
}

bool ggml_cuda_dsv4_hc_up_q8_supported(const ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    const int64_t n_embd = dst->src[0]->ne[0];
    return n_embd % QK8_1 == 0 && n_embd/QK8_1 <= HCF_UP_CNT && hcf_up_cnt_slot(ctx) >= 0 &&
        dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst) && ggml_nelements(dst) == n_embd*dst->src[0]->ne[2];
}

void ggml_cuda_op_dsv4_hc_up_q8(ggml_backend_cuda_context & ctx, ggml_tensor * dst, void * q8, bool swizzle) {
    HCF_DISPATCH(hcf_up_launch, (int) dst->src[0]->ne[2], ctx, dst, q8, swizzle);
}

// [GDNGATES] Flash-Next / Qwen3-Next gated-delta-net gates in one kernel (opt-in, GGML_CUDA_GDN_GATES=1; the graph
// match is ggml_cuda_try_fuse_gdn_gates in ggml-cuda.cu). The unfused graph is six kernels per linear-attention layer:
//   alpha = W_alpha x; gate = softplus(alpha + dt) * a;   beta = sigmoid(W_beta x).
// One CTA per output row (alpha rows first, then beta rows). The dot product follows mul_mat_vec_f for BF16 with a
// 256-thread block exactly (same per-thread column stride, same fused multiply-adds, same warp and block reduction),
// and the activations are the same expressions as the unary kernels, so the outputs are bit-identical to the
// unfused graph whenever that graph runs mul_mat_vec_f (K a multiple of 512, at most 8 tokens).
static constexpr int GDNG_BLOCK = 256;

template <int ncols_dst>
static __global__ void __launch_bounds__(GDNG_BLOCK) gdn_gates_kernel(
        const nv_bfloat16 * __restrict__ w_alpha, const nv_bfloat16 * __restrict__ w_beta, const float * __restrict__ y,
        const float * __restrict__ dt, const float * __restrict__ a, float * __restrict__ gate, float * __restrict__ beta,
        const int ncols2, const int64_t stride_row_alpha, const int64_t stride_row_beta, const int stride_col_y2,
        const int nrows) {
    const bool is_beta = (int) blockIdx.x >= nrows;
    const int  row     = is_beta ? blockIdx.x - nrows : blockIdx.x;
    const int  tid     = threadIdx.x;

    const nv_bfloat162 * x2 = (const nv_bfloat162 *) (is_beta ? w_beta + row*stride_row_beta : w_alpha + row*stride_row_alpha);
    const float2       * y2 = (const float2 *) y;

    __shared__ float buf_iw[WARP_SIZE];
    if (tid < WARP_SIZE) {
        buf_iw[tid] = 0.0f;
    }
    __syncthreads();

    float sumf[ncols_dst] = {0.0f};
    for (int col2 = tid; col2 < ncols2; col2 += GDNG_BLOCK) {
        const nv_bfloat162 tmpx = x2[col2];
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            const float2 tmpy = y2[j*stride_col_y2 + col2];
            ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
            ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);
        }
    }

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        sumf[j] = warp_reduce_sum<WARP_SIZE>(sumf[j]);
        buf_iw[tid/WARP_SIZE] = sumf[j];
        __syncthreads();
        if (tid < WARP_SIZE) {
            sumf[j] = buf_iw[tid];
            sumf[j] = warp_reduce_sum<WARP_SIZE>(sumf[j]);
        }
        __syncthreads();
    }

    if (tid >= ncols_dst) {
        return;
    }

    const float v = sumf[tid];
    if (is_beta) {
        beta[tid*nrows + row] = 1.0f / (1.0f + expf(-v));
    } else {
        const float x  = v + dt[row];
        const float sp = (x > 20.0f) ? x : logf(1.0f + expf(x));
        gate[tid*nrows + row] = sp * a[row];
    }
}

template <int T>
static void gdn_gates_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * w_alpha, const ggml_tensor * w_beta,
        const ggml_tensor * y, const ggml_tensor * dt, const ggml_tensor * a, ggml_tensor * gate, ggml_tensor * beta) {
    const int nrows = (int) w_alpha->ne[1];
    gdn_gates_kernel<T><<<2*nrows, GDNG_BLOCK, 0, ctx.stream()>>>(
        (const nv_bfloat16 *) w_alpha->data, (const nv_bfloat16 *) w_beta->data, (const float *) y->data,
        (const float *) dt->data, (const float *) a->data, (float *) gate->data, (float *) beta->data,
        (int) (w_alpha->ne[0]/2), (int64_t) (w_alpha->nb[1]/sizeof(nv_bfloat16)), (int64_t) (w_beta->nb[1]/sizeof(nv_bfloat16)),
        (int) (y->nb[1]/sizeof(float)/2), nrows);
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_gdn_gates_supported(const ggml_tensor * w_alpha, const ggml_tensor * w_beta, const ggml_tensor * y,
        const ggml_tensor * dt, const ggml_tensor * a, const ggml_tensor * gate, const ggml_tensor * beta) {
    const int64_t K = w_alpha->ne[0], rows = w_alpha->ne[1], T = y->ne[1];
    return w_alpha->type == GGML_TYPE_BF16 && w_beta->type == GGML_TYPE_BF16 &&
        ggml_are_same_shape(w_alpha, w_beta) && w_alpha->ne[2] == 1 && w_alpha->ne[3] == 1 &&
        w_alpha->nb[0] == sizeof(nv_bfloat16) && w_beta->nb[0] == sizeof(nv_bfloat16) &&
        K % 512 == 0 && rows >= 1 && rows <= 1024 &&
        y->type == GGML_TYPE_F32 && y->ne[0] == K && T >= 1 && T <= HCF_MAX_T && y->ne[2] == 1 && y->ne[3] == 1 &&
        y->nb[0] == sizeof(float) && y->nb[1] % (2*sizeof(float)) == 0 &&
        dt->type == GGML_TYPE_F32 && ggml_is_contiguous(dt) && ggml_nelements(dt) == rows && dt->ne[0] == rows &&
        a->type  == GGML_TYPE_F32 && ggml_is_contiguous(a)  && ggml_nelements(a)  == rows && a->ne[0]  == rows &&
        gate->type == GGML_TYPE_F32 && ggml_is_contiguous(gate) && ggml_nelements(gate) == rows*T &&
        beta->type == GGML_TYPE_F32 && ggml_is_contiguous(beta) && ggml_nelements(beta) == rows*T;
}

void ggml_cuda_gdn_gates(ggml_backend_cuda_context & ctx, const ggml_tensor * w_alpha, const ggml_tensor * w_beta,
        const ggml_tensor * y, const ggml_tensor * dt, const ggml_tensor * a, ggml_tensor * gate, ggml_tensor * beta) {
    HCF_DISPATCH(gdn_gates_launch, (int) y->ne[1], ctx, w_alpha, w_beta, y, dt, a, gate, beta);
}

// [SHEXPGATE] Shared-expert gate, residual-side: dst = other + x * sigmoid(dot(w, y_t)) per token t, replacing the
// vector matmul (cuBLAS dot + reduce at 1..8 tokens), the sigmoid, the mul and the add (opt-in, GGML_CUDA_SHEXP_GATE=1;
// graph match ggml_cuda_try_fuse_shexp_gate in ggml-cuda.cu). The mul and the add are rounded separately, as in the
// unfused graph; only the dot product's summation order differs (valid-math reordering).
static constexpr int SHG_BLOCK = 256;
static constexpr int SHG_PER_THREAD = 4;

static __global__ void __launch_bounds__(SHG_BLOCK) shexp_gate_kernel(
        const float * __restrict__ w, const float * __restrict__ y, const int64_t s_y,
        const float * __restrict__ x, const int64_t s_x, const float * __restrict__ other, const int64_t s_o,
        float * __restrict__ dst, const int64_t s_d, const int K, const int E) {
    const int t   = blockIdx.y;
    const int tid = threadIdx.x;
    const float * yt = y + t*s_y;

    float s = 0.0f;
    for (int k = tid; k < K; k += SHG_BLOCK) {
        s += w[k]*yt[k];
    }
    s = warp_reduce_sum<WARP_SIZE>(s);
    __shared__ float buf[SHG_BLOCK/WARP_SIZE];
    if (tid % WARP_SIZE == 0) {
        buf[tid/WARP_SIZE] = s;
    }
    __syncthreads();
    s = 0.0f;
#pragma unroll
    for (int i = 0; i < SHG_BLOCK/WARP_SIZE; ++i) {
        s += buf[i];
    }
    const float sg = 1.0f / (1.0f + expf(-s));

    const int i0 = blockIdx.x*SHG_BLOCK*SHG_PER_THREAD + tid;
#pragma unroll
    for (int r = 0; r < SHG_PER_THREAD; ++r) {
        const int i = i0 + r*SHG_BLOCK;
        if (i < E) {
            dst[t*s_d + i] = __fadd_rn(other[t*s_o + i], __fmul_rn(x[t*s_x + i], sg));
        }
    }
}

bool ggml_cuda_shexp_gate_supported(const ggml_tensor * w, const ggml_tensor * y, const ggml_tensor * x,
        const ggml_tensor * other, const ggml_tensor * dst) {
    const int64_t K = w->ne[0], T = y->ne[1], E = x->ne[0];
    const auto rows_ok = [&](const ggml_tensor * t) {
        return t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float) && t->ne[0] == E && t->ne[1] == T &&
            t->ne[2] == 1 && t->ne[3] == 1;
    };
    return w->type == GGML_TYPE_F32 && ggml_is_contiguous(w) && ggml_nelements(w) == K && K <= (1 << 20) &&
        y->type == GGML_TYPE_F32 && y->nb[0] == sizeof(float) && y->ne[0] == K && T >= 1 && T <= HCF_MAX_T &&
        y->ne[2] == 1 && y->ne[3] == 1 && E >= 1 && E <= (1 << 20) &&
        rows_ok(x) && rows_ok(other) && rows_ok(dst);
}

void ggml_cuda_shexp_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * y,
        const ggml_tensor * x, const ggml_tensor * other, ggml_tensor * dst) {
    const int K = (int) w->ne[0], E = (int) x->ne[0], T = (int) y->ne[1];
    const dim3 grid((E + SHG_BLOCK*SHG_PER_THREAD - 1)/(SHG_BLOCK*SHG_PER_THREAD), T);
    shexp_gate_kernel<<<grid, SHG_BLOCK, 0, ctx.stream()>>>(
        (const float *) w->data, (const float *) y->data, (int64_t) (y->nb[1]/sizeof(float)),
        (const float *) x->data, (int64_t) (x->nb[1]/sizeof(float)), (const float *) other->data, (int64_t) (other->nb[1]/sizeof(float)),
        (float *) dst->data, (int64_t) (dst->nb[1]/sizeof(float)), K, E);
    CUDA_CHECK(cudaGetLastError());
}
