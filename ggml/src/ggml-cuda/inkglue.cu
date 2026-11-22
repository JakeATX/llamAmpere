// INKGLUE: Inkling 2 decode glue fusions (one kernel for a run of small graph nodes). Every fused kernel repeats the
// unfused kernels' arithmetic in the same order, so the outputs match them bit for bit (see each kernel). The graph
// matchers live in ggml-cuda.cu (ggml_cuda_try_fuse_inkglue); GGML_CUDA_INKGLUE=0 turns all of them off.

#include "inkglue.cuh"

// ---------------------------------------------------------------------------------------------------------------------
// SITU: out = (b*tanh(g/b) * sigmoid(g)) * (lb*tanh(u/lb)), 9 nodes (SCALE TANH SCALE SIGMOID MUL SCALE TANH SCALE MUL).
// Same per-element expressions as scale_f32 (scale*x + bias), op_tanh (tanhf), op_sigmoid (1/(1 + expf(-x))) and the
// MULs (a*b), with every intermediate rounded to f32 as when it was stored.
static __global__ void k_ink_situ(const char * __restrict__ g, const char * __restrict__ u, float * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const int64_t g_nb1, const int64_t g_nb2, const int64_t u_nb1, const int64_t u_nb2,
        const ink_situ_params p) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t n = ne0*ne1*ne2;
    if (i >= n) {
        return;
    }
    const int64_t i0 = i % ne0;
    const int64_t i1 = (i / ne0) % ne1;
    const int64_t i2 = i / (ne0*ne1);

    const float gv = *(const float *) (g + i2*g_nb2 + i1*g_nb1 + i0*sizeof(float));
    const float uv = *(const float *) (u + i2*u_nb2 + i1*u_nb1 + i0*sizeof(float));

    const float a1 = p.ga_s*gv + p.ga_b;
    const float a2 = tanhf(a1);
    const float a3 = p.gb_s*a2 + p.gb_b;
    const float sg = 1.0f / (1.0f + expf(-gv));
    const float a4 = p.sig_first ? sg*a3 : a3*sg;
    const float u1 = p.ua_s*uv + p.ua_b;
    const float u2 = tanhf(u1);
    const float u3 = p.ub_s*u2 + p.ub_b;
    dst[i] = p.act_first ? a4*u3 : u3*a4;
}

void ggml_cuda_ink_situ(ggml_backend_cuda_context & ctx, const ggml_tensor * g, const ggml_tensor * u, ggml_tensor * dst,
        const ink_situ_params & p) {
    GGML_ASSERT(ggml_is_contiguous(dst) && dst->type == GGML_TYPE_F32);
    const int64_t n = ggml_nelements(dst);
    const int bs = 256;
    k_ink_situ<<<(unsigned) ((n + bs - 1)/bs), bs, 0, ctx.stream()>>>((const char *) g->data, (const char *) u->data,
        (float *) dst->data, dst->ne[0], dst->ne[1], dst->ne[2], g->nb[1], g->nb[2], u->nb[1], u->nb[2], p);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------------------------------
// MoE weighted expert sum: out[i, t] = sum_k e[i, k, t]*w[k, t] (MUL -> CONT(PERMUTE) -> SUM_ROWS). One warp per output:
// lane k holds the rounded product (the MUL), and warp_reduce_sum adds them in the order reduce_rows_f32 does with one
// column per thread (k < 32 columns, every other lane 0).
static __global__ void k_ink_moe_wsum(const char * __restrict__ e, const char * __restrict__ w, float * __restrict__ dst,
        const int n_embd, const int n_used, const int n_tok,
        const int64_t e_nb1, const int64_t e_nb2, const int64_t w_nb1, const int64_t w_nb2) {
    const int warp = (blockIdx.x*blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    if (warp >= n_embd*n_tok) {
        return;
    }
    const int i = warp % n_embd;
    const int t = warp / n_embd;
    float v = 0.0f;
    if (lane < n_used) {
        const float ev = *(const float *) (e + t*e_nb2 + lane*e_nb1 + i*sizeof(float));
        const float wv = *(const float *) (w + t*w_nb2 + lane*w_nb1);
        v = 0.0f + __fmul_rn(ev, wv); // the MUL rounds to memory, reduce_rows starts from 0
    }
    v = warp_reduce_sum(v);
    if (lane == 0) {
        dst[(int64_t) t*n_embd + i] = v;
    }
}

void ggml_cuda_ink_moe_wsum(ggml_backend_cuda_context & ctx, const ggml_tensor * e, const ggml_tensor * w, ggml_tensor * dst) {
    const int n_embd = (int) e->ne[0];
    const int n_used = (int) e->ne[1];
    const int n_tok  = (int) e->ne[2];
    GGML_ASSERT(n_used <= WARP_SIZE && ggml_nelements(dst) == (int64_t) n_embd*n_tok && ggml_is_contiguous(dst));
    const int bs = 256;
    const int64_t threads = (int64_t) n_embd*n_tok*WARP_SIZE;
    k_ink_moe_wsum<<<(unsigned) ((threads + bs - 1)/bs), bs, 0, ctx.stream()>>>((const char *) e->data, (const char *) w->data,
        (float *) dst->data, n_embd, n_used, n_tok, e->nb[1], e->nb[2], w->nb[1], w->nb[2]);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------------------------------
// clip: [MUL(x, m) ->] L2_NORM(eps) -> SCALE(s, b). One warp per row as l2_norm_f32<WARP_SIZE> (rows of < 1024): the same
// per-lane partial sums and warp_reduce_sum, the same rsqrtf(fmaxf(sum, eps*eps)), then scale_f32's s*v + b.
// x rows: [ncols, nrows, nch] with strides; m (optional) [ncols, nrows] contiguous, broadcast over the channels.
static __global__ void k_ink_clip(const float * x, const float * __restrict__ m, float * dst,
        const int ncols, const int nrows, const int64_t sx1, const int64_t sx2, const float eps, const float s, const float b) {
    const int row = blockIdx.x;
    const int ch  = blockIdx.y;
    const int tid = threadIdx.x;
    const float * xr = x + ch*sx2 + row*sx1;
    const float * mr = m ? m + (int64_t) row*ncols : nullptr;
    float * dr = dst + ((int64_t) ch*nrows + row)*ncols;

    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += WARP_SIZE) {
        const float xi = mr ? xr[col]*mr[col] : xr[col];
        tmp += xi * xi;
    }
    tmp = warp_reduce_sum(tmp);
    const float scale = rsqrtf(fmaxf(tmp, eps * eps));
    for (int col = tid; col < ncols; col += WARP_SIZE) {
        const float xi = mr ? xr[col]*mr[col] : xr[col];
        const float v  = scale * xi;
        dr[col] = s*v + b;
    }
}

void ggml_cuda_ink_clip(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * m, ggml_tensor * dst,
        float eps, float s, float b) {
    GGML_ASSERT(x->type == GGML_TYPE_F32 && x->nb[0] == sizeof(float) && x->ne[3] == 1 && ggml_is_contiguous(dst));
    GGML_ASSERT(x->ne[0] < 1024 && x->ne[2] <= 65535);
    const int64_t sx1 = x->nb[1]/sizeof(float);
    const int64_t sx2 = x->nb[2]/sizeof(float);
    const dim3 grid((unsigned) x->ne[1], (unsigned) x->ne[2], 1);
    k_ink_clip<<<grid, WARP_SIZE, 0, ctx.stream()>>>((const float *) x->data, m ? (const float *) m->data : nullptr,
        (float *) dst->data, (int) x->ne[0], (int) x->ne[1], sx1, sx2, eps, s, b);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------------------------------
// relative-position attention mask, the dense Inkling path (llm_graph_context::build_attn_rel + build_attn):
//   table = MUL_MAT(PAD(bias, one zero row), r) - sink    [E + 1, H, T]
//   out[j, t, h] = cast(table[rel_pos[j, t], h, t] + mask[j, t])    [n_kv, T, H], F16 (FA) or F32
// mode tab (table given, the MUL_MAT ran): replaces SUB, CONT(PERMUTE), GET_ROWS, CONT(PERMUTE), the mask cast, ADD and
// the output cast with the same arithmetic (exact). mode full (bias + r given): also PAD and MUL_MAT; the dot over the
// n_rel bias columns is a k-ascending fmaf chain from 0 (a rounding-level match of the sgemm). rel_pos == E is the pad row.
template <typename T_out, typename T_mask, bool full>
static __global__ void k_ink_rel_mask(const float * __restrict__ tab, const float * __restrict__ bias, const float * __restrict__ r,
        const char * __restrict__ sinks, const int32_t * __restrict__ rel_pos, const T_mask * __restrict__ mask, T_out * __restrict__ dst,
        const int n_rel, const int E, const int H, const int n_kv, const int T,
        const int64_t a_s1, const int64_t a_s2, const int64_t sk_nb1, const int64_t sk_nb2, const int64_t rp_s1, const int64_t mask_s1) {
    extern __shared__ float sm[];
    float * rs = sm;                    // [H][n_rel] (full)
    float * ss = sm + (full ? H*n_rel : 0); // [H]
    const int t = blockIdx.y;
    if (full) {
        for (int k = threadIdx.x; k < H*n_rel; k += blockDim.x) {
            rs[k] = r[t*a_s2 + (k / n_rel)*a_s1 + k % n_rel];
        }
    }
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        ss[h] = *(const float *) (sinks + t*sk_nb2 + h*sk_nb1);
    }
    __syncthreads();
    const int j = blockIdx.x*blockDim.x + threadIdx.x;
    if (j >= n_kv) {
        return;
    }
    const int   e = rel_pos[t*rp_s1 + j];
    const float m = ggml_cuda_cast<float>(mask[t*mask_s1 + j]);
    if (full) {
        float b[32];
#pragma unroll
        for (int c = 0; c < 32; ++c) {
            b[c] = c < n_rel && e < E ? bias[(int64_t) e*n_rel + c] : 0.0f;
        }
        for (int h = 0; h < H; ++h) {
            float acc = 0.0f;
#pragma unroll
            for (int c = 0; c < 32; ++c) {
                if (c < n_rel) {
                    acc = fmaf(b[c], rs[h*n_rel + c], acc);
                }
            }
            const float tv = acc - ss[h];
            dst[((int64_t) h*T + t)*n_kv + j] = ggml_cuda_cast<T_out>(tv + m);
        }
    } else {
        for (int h = 0; h < H; ++h) {
            const float tv = tab[t*a_s2 + h*a_s1 + e] - ss[h];
            dst[((int64_t) h*T + t)*n_kv + j] = ggml_cuda_cast<T_out>(tv + m);
        }
    }
}

void ggml_cuda_ink_rel_mask(ggml_backend_cuda_context & ctx, const ggml_tensor * tab, const ggml_tensor * bias, const ggml_tensor * r,
        const ggml_tensor * sinks, const ggml_tensor * rel_pos, const ggml_tensor * mask, ggml_tensor * dst) {
    const bool full = tab == nullptr;
    const ggml_tensor * a = full ? r : tab;
    const int n_rel = full ? (int) bias->ne[0] : 0;
    const int E     = full ? (int) bias->ne[1] : 0;
    const int H     = (int) a->ne[1];
    const int T     = (int) a->ne[2];
    const int n_kv  = (int) dst->ne[0];
    GGML_ASSERT(n_rel <= 32 && dst->ne[1] == T && dst->ne[2] == H && dst->ne[3] == 1 && ggml_is_contiguous(dst));
    GGML_ASSERT(a->type == GGML_TYPE_F32 && a->nb[0] == sizeof(float) && sinks->type == GGML_TYPE_F32 && rel_pos->type == GGML_TYPE_I32);
    const dim3 grid((unsigned) ((n_kv + 127)/128), (unsigned) T, 1);
    const size_t smem = (size_t) ((full ? H*n_rel : 0) + H)*sizeof(float);
    const int64_t a_s1 = a->nb[1]/sizeof(float), a_s2 = a->nb[2]/sizeof(float);
    const int64_t rp_s1 = rel_pos->nb[1]/sizeof(int32_t);
    const int64_t m_s1 = mask->nb[1]/ggml_element_size(mask);
    const float * tp = full ? nullptr : (const float *) tab->data;
    const float * bp = full ? (const float *) bias->data : nullptr;
    const float * rp = full ? (const float *) r->data : nullptr;
#define INK_REL_LAUNCH(TO, TM, FULL) k_ink_rel_mask<TO, TM, FULL><<<grid, 128, smem, ctx.stream()>>>(tp, bp, rp, \
        (const char *) sinks->data, (const int32_t *) rel_pos->data, (const TM *) mask->data, (TO *) dst->data, \
        n_rel, E, H, n_kv, T, a_s1, a_s2, sinks->nb[1], sinks->nb[2], rp_s1, m_s1)
#define INK_REL_LAUNCH2(TO, TM) if (full) { INK_REL_LAUNCH(TO, TM, true); } else { INK_REL_LAUNCH(TO, TM, false); }
    if (dst->type == GGML_TYPE_F16) {
        if (mask->type == GGML_TYPE_F16) { INK_REL_LAUNCH2(half, half); } else { INK_REL_LAUNCH2(half, float); }
    } else {
        if (mask->type == GGML_TYPE_F16) { INK_REL_LAUNCH2(float, half); } else { INK_REL_LAUNCH2(float, float); }
    }
#undef INK_REL_LAUNCH2
#undef INK_REL_LAUNCH
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------------------------------
// LOOP_GATE x G: loop-gate.cu's k_loop_gate_down / k_loop_gate_apply with the gate picked by blockIdx (z for down, y for
// apply). Every statement of the per-gate body is the original's, so each gate's output is bit-identical.
#define INK_LG_CHUNKS 8
#define INK_LG_WARPS  4
#define INK_LG_MAX_T  8
#define INK_LG_MAX_R  128
#define INK_LG_APPLY_WARPS 8
#define INK_LG_APPLY_ROWS  2

struct ink_lg_args {
    const char * x[INK_MULTI_MAX];
    const char * m[INK_MULTI_MAX];
    int64_t      x_nb1[INK_MULTI_MAX];
    int64_t      m_nb1[INK_MULTI_MAX];
    const void * down[INK_MULTI_MAX];
    const void * gain[INK_MULTI_MAX];
    const void * forget[INK_MULTI_MAX];
    float      * dst[INK_MULTI_MAX];
    float        eps[INK_MULTI_MAX];
};

template <typename TW, int RB>
static __global__ void k_ink_lg_down(const ink_lg_args a, float * __restrict__ part_all, const int n, const int R, const int T) {
    const int g = blockIdx.z;
    const char * __restrict__ x = a.x[g];
    const char * __restrict__ m = a.m[g];
    const int64_t x_nb1 = a.x_nb1[g];
    const int64_t m_nb1 = a.m_nb1[g];
    const TW * __restrict__ down = (const TW *) a.down[g];
    float * __restrict__ part = part_all + (int64_t) g*T*2*INK_LG_CHUNKS*(R + 1);

    // RB consecutive rows r per block (one x / m load feeds all of them); each (r, chunk) partial keeps the original
    // lane order and warp reduction, so the partials are bit-identical for any RB
    const int r0   = blockIdx.x*RB;
    const int c    = blockIdx.y*INK_LG_WARPS + threadIdx.x/WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int half = c / INK_LG_CHUNKS;
    const int cl   = (n + INK_LG_CHUNKS - 1)/INK_LG_CHUNKS;
    const int i0   = (c % INK_LG_CHUNKS)*cl;
    const int i1   = min(n, i0 + cl);

    const char   * src = half ? m : x;
    const int64_t  nb1 = half ? m_nb1 : x_nb1;
    const TW     * w   = down + (int64_t) r0*2*n + half*n;

    float d[RB][INK_LG_MAX_T] = {{0.0f}};
    float ss[INK_LG_MAX_T]    = {0.0f};
#pragma unroll 4
    for (int i = i0 + lane; i < i1; i += WARP_SIZE) {
        float wi[RB];
#pragma unroll
        for (int b = 0; b < RB; ++b) {
            wi[b] = r0 + b < R ? ggml_cuda_cast<float>(w[(int64_t) b*2*n + i]) : 0.0f;
        }
#pragma unroll
        for (int t = 0; t < INK_LG_MAX_T; ++t) {
            if (t < T) {
                const float v = ((const float *) (src + t*nb1))[i];
#pragma unroll
                for (int b = 0; b < RB; ++b) {
                    d[b][t] += wi[b]*v;
                }
                ss[t] += v*v;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < INK_LG_MAX_T; ++t) {
        if (t < T) {
            if (r0 == 0) {
                ss[t] = warp_reduce_sum(ss[t]);
            }
            float * p = part + ((int64_t) t*2*INK_LG_CHUNKS + c)*(R + 1);
#pragma unroll
            for (int b = 0; b < RB; ++b) {
                d[b][t] = warp_reduce_sum(d[b][t]);
                if (lane == 0 && r0 + b < R) {
                    p[r0 + b] = d[b][t];
                }
            }
            if (lane == 0 && r0 == 0) {
                p[R] = ss[t];
            }
        }
    }
}

template <typename TW, int ROWS>
static __global__ void k_ink_lg_apply(const ink_lg_args a, const float * __restrict__ part_all, const int n, const int R, const int T) {
    const int g = blockIdx.y;
    const char * __restrict__ x = a.x[g];
    const char * __restrict__ m = a.m[g];
    const int64_t x_nb1 = a.x_nb1[g];
    const int64_t m_nb1 = a.m_nb1[g];
    const TW * __restrict__ gain   = (const TW *) a.gain[g];
    const TW * __restrict__ forget = (const TW *) a.forget[g];
    const float * __restrict__ part = part_all + (int64_t) g*T*2*INK_LG_CHUNKS*(R + 1);
    float * __restrict__ dst = a.dst[g];
    const float eps = a.eps[g];

    __shared__ float gs[INK_LG_MAX_T*INK_LG_MAX_R];
    __shared__ float rs[2*INK_LG_MAX_T];

    for (int j = threadIdx.x; j < 2*T; j += blockDim.x) {
        const float * p = part + ((int64_t) (j/2)*2*INK_LG_CHUNKS + (j % 2)*INK_LG_CHUNKS)*(R + 1);
        float s = 0.0f;
        for (int c = 0; c < INK_LG_CHUNKS; ++c) {
            s += p[c*(R + 1) + R];
        }
        rs[j] = rsqrtf(s/n + eps);
    }
    __syncthreads();
    for (int j = threadIdx.x; j < T*R; j += blockDim.x) {
        const int t = j / R;
        const int r = j % R;
        const float * p = part + (int64_t) t*2*INK_LG_CHUNKS*(R + 1) + r;
        float gx = 0.0f;
        float gm = 0.0f;
        for (int c = 0; c < INK_LG_CHUNKS; ++c) {
            gx += p[c*(R + 1)];
            gm += p[(INK_LG_CHUNKS + c)*(R + 1)];
        }
        gs[t*R + r] = rs[2*t]*gx + rs[2*t + 1]*gm;
    }
    __syncthreads();

    const int lane = threadIdx.x % WARP_SIZE;
    const int row0 = (blockIdx.x*INK_LG_APPLY_WARPS + threadIdx.x/WARP_SIZE)*ROWS;
#pragma unroll
    for (int k = 0; k < ROWS; ++k) {
        const int i = row0 + k;
        if (i >= n) {
            return;
        }
        float ag[INK_LG_MAX_T] = {0.0f};
        float af[INK_LG_MAX_T] = {0.0f};
        const TW * wg = gain   + (int64_t) i*R;
        const TW * wf = forget + (int64_t) i*R;
        for (int r = lane; r < R; r += WARP_SIZE) {
            const float av = ggml_cuda_cast<float>(wg[r]);
            const float bv = ggml_cuda_cast<float>(wf[r]);
#pragma unroll
            for (int t = 0; t < INK_LG_MAX_T; ++t) {
                if (t < T) {
                    ag[t] += av*gs[t*R + r];
                    af[t] += bv*gs[t*R + r];
                }
            }
        }
#pragma unroll
        for (int t = 0; t < INK_LG_MAX_T; ++t) {
            if (t < T) {
                ag[t] = warp_reduce_sum(ag[t]);
                af[t] = warp_reduce_sum(af[t]);
                if (lane == t) {
                    const float xi = ((const float *) (x + t*x_nb1))[i];
                    const float mi = ((const float *) (m + t*m_nb1))[i];
                    dst[(int64_t) t*n + i] = (xi - tanhf(af[t])*xi) + tanhf(ag[t])*mi;
                }
            }
        }
    }
}

// LGTUNE: rows r per down block (GGML_CUDA_LG_RB 1/2/4) and output rows per apply warp (GGML_CUDA_LG_ROWS 2/4/8);
// launch shape only, every output is bit-identical for any setting. Default RB=4 ROWS=4 (HOME 3090 Ti screen: fastest of 8
// shapes); GGML_CUDA_LG_RB=1 GGML_CUDA_LG_ROWS=2 restores the old shape.
int ggml_cuda_lg_tune_rb() {
    static const int v = [] { const char * e = getenv("GGML_CUDA_LG_RB"); const int k = e && *e ? atoi(e) : 4; return k == 1 || k == 2 ? k : 4; }();
    return v;
}
int ggml_cuda_lg_tune_rows() {
    static const int v = [] { const char * e = getenv("GGML_CUDA_LG_ROWS"); const int k = e && *e ? atoi(e) : 4; return k == 2 || k == 8 ? k : 4; }();
    return v;
}

template <typename TW, int RB, int ROWS>
static void ink_lg_launch_t(ggml_backend_cuda_context & ctx, const ink_lg_args & a, int G, int n, int R, int T) {
    ggml_cuda_pool_alloc<float> part(ctx.pool(), (size_t) G*T*2*INK_LG_CHUNKS*(R + 1));
    cudaStream_t stream = ctx.stream();
    k_ink_lg_down<TW, RB><<<dim3((R + RB - 1)/RB, 2*INK_LG_CHUNKS/INK_LG_WARPS, G), INK_LG_WARPS*WARP_SIZE, 0, stream>>>(a, part.get(), n, R, T);
    const int rows_per_block = INK_LG_APPLY_WARPS*ROWS;
    k_ink_lg_apply<TW, ROWS><<<dim3((n + rows_per_block - 1)/rows_per_block, G, 1), INK_LG_APPLY_WARPS*WARP_SIZE, 0, stream>>>(
        a, part.get(), n, R, T);
    CUDA_CHECK(cudaGetLastError());
}

template <typename TW, int RB>
static void ink_lg_launch_r(ggml_backend_cuda_context & ctx, const ink_lg_args & a, int G, int n, int R, int T) {
    switch (ggml_cuda_lg_tune_rows()) {
        case 4:  ink_lg_launch_t<TW, RB, 4>(ctx, a, G, n, R, T); break;
        case 8:  ink_lg_launch_t<TW, RB, 8>(ctx, a, G, n, R, T); break;
        default: ink_lg_launch_t<TW, RB, 2>(ctx, a, G, n, R, T); break;
    }
}

template <typename TW>
static void ink_lg_launch(ggml_backend_cuda_context & ctx, const ink_lg_args & a, int G, int n, int R, int T) {
    switch (ggml_cuda_lg_tune_rb()) {
        case 2:  ink_lg_launch_r<TW, 2>(ctx, a, G, n, R, T); break;
        case 4:  ink_lg_launch_r<TW, 4>(ctx, a, G, n, R, T); break;
        default: ink_lg_launch_r<TW, 1>(ctx, a, G, n, R, T); break;
    }
}

bool ggml_cuda_ink_loop_gate_multi(ggml_backend_cuda_context & ctx, ggml_tensor * const * gates, int G) {
    if (G < 2 || G > INK_MULTI_MAX) {
        return false;
    }
    const ggml_tensor * d0 = gates[0];
    const int n = (int) d0->src[0]->ne[0];
    const int T = (int) d0->src[0]->ne[1];
    const int R = (int) d0->src[2]->ne[1];
    const ggml_type wt = d0->src[2]->type;
    if (T < 1 || T > INK_LG_MAX_T || R > INK_LG_MAX_R) {
        return false;
    }
    ink_lg_args a{};
    for (int k = 0; k < G; ++k) {
        const ggml_tensor * d = gates[k];
        if (d->op != GGML_OP_LOOP_GATE || d->src[0]->ne[0] != n || d->src[0]->ne[1] != T || d->src[2]->ne[1] != R ||
                d->src[2]->type != wt || d->src[3]->type != wt || d->src[4]->type != wt) {
            return false;
        }
        a.x[k] = (const char *) d->src[0]->data;  a.x_nb1[k] = d->src[0]->nb[1];
        a.m[k] = (const char *) d->src[1]->data;  a.m_nb1[k] = d->src[1]->nb[1];
        a.down[k] = d->src[2]->data; a.gain[k] = d->src[3]->data; a.forget[k] = d->src[4]->data;
        a.dst[k] = (float *) d->data;
        a.eps[k] = ggml_get_op_params_f32(d, 0);
    }
    switch (wt) {
        case GGML_TYPE_F16:  ink_lg_launch<half>(ctx, a, G, n, R, T);        break;
        case GGML_TYPE_BF16: ink_lg_launch<nv_bfloat16>(ctx, a, G, n, R, T); break;
        case GGML_TYPE_F32:  ink_lg_launch<float>(ctx, a, G, n, R, T);       break;
        default: return false;
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------
// RMS_NORM -> MUL(gamma) x G rows (one block per row): rms_norm_f32<block_size, 1, false, true>'s scalar path term for
// term (tmp += xi*xi over col = tid + k*block_size, block_reduce SUM, rsqrtf(tmp/ncols + eps), scale*x*gamma). With q8 the
// loop runs to ncols_padded and also writes q8_1 like quantize_q8_1 (the add_rms_norm_mul_f32 q8_mode 1 code, norm.cu).
struct ink_rn_args {
    const float * x[INK_MULTI_MAX];
    const float * g[INK_MULTI_MAX];
    float       * dst[INK_MULTI_MAX];
    block_q8_1  * q8[INK_MULTI_MAX];
    float         eps[INK_MULTI_MAX];
};

template <int block_size>
static __global__ void k_ink_rms_norm_mul_multi(const ink_rn_args a, const int ncols, const int ncols_padded) {
    static_assert(block_size % QK8_1 == 0, "q8_1 blocks must not straddle warps");
    const int k   = blockIdx.x;
    const int tid = threadIdx.x;
    const float * x     = a.x[k];
    const float * gamma = a.g[k];
    float       * dst   = a.dst[k];
    block_q8_1  * y     = a.q8[k];

    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + a.eps[k]);

    if (y == nullptr) {
        for (int col = tid; col < ncols; col += block_size) {
            dst[col] = scale * x[col] * gamma[col];
        }
        return;
    }
    const int lane = tid % QK8_1;
    for (int col = tid; col - lane < ncols_padded; col += block_size) {
        float xi = 0.0f;
        if (col < ncols) {
            xi = scale * x[col] * gamma[col];
            dst[col] = xi;
        }
        float amax = fabsf(xi);
        amax = warp_reduce_max<QK8_1>(amax);

        const float  d = amax / 127.0f;
        const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

        const int ib = col / QK8_1;
        y[ib].qs[lane] = q;
        const float qsum = warp_reduce_sum<QK8_1>((float) q);
        if (lane == 0) {
            y[ib].ds = make_half2(d, d * qsum);
        }
    }
}

bool ggml_cuda_ink_rms_norm_mul_multi(ggml_backend_cuda_context & ctx, const ggml_tensor * const * x,
        const ggml_tensor * const * gamma, ggml_tensor * const * dst, void * const * q8, const float * eps, int G) {
    if (G < 1 || G > INK_MULTI_MAX) {
        return false;
    }
    const int64_t ncols = x[0]->ne[0];
    if (ncols > INT_MAX) {
        return false;
    }
    ink_rn_args a{};
    for (int k = 0; k < G; ++k) {
        a.x[k] = (const float *) x[k]->data;
        a.g[k] = (const float *) gamma[k]->data;
        a.dst[k] = (float *) dst[k]->data;
        a.q8[k] = (block_q8_1 *) q8[k];
        a.eps[k] = eps[k];
    }
    const int ncols_padded = (int) GGML_PAD(ncols, MATRIX_ROW_PADDING);
    cudaStream_t stream = ctx.stream();
    // rms_norm_f32_cuda's block size choice (the reduction order depends on it)
    if (ncols < 1024) {
        k_ink_rms_norm_mul_multi<256><<<G, 256, 32*sizeof(float), stream>>>(a, (int) ncols, ncols_padded);
    } else {
        k_ink_rms_norm_mul_multi<1024><<<G, 1024, 32*sizeof(float), stream>>>(a, (int) ncols, ncols_padded);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
