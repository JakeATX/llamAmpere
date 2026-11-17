#include "lsa-comp.cuh"

#define LSA_COMP_THREADS   256
#define LSA_COMP_GMAX      8    // q heads per kv head (= warps per CTA: one warp per head in the softmax pass)
#define LSA_COMP_DMAX      512  // K head size
#define LSA_COMP_CMAX      256  // mean blocks per CTA
#define LSA_COMP_NSPLITMAX 256  // chunks per (row, kv head)

// partial p = (t*H + h)*nsplit + s: [Dv] weighted V sum, then the chunk max and the sum of exp(s_b - max)
static __global__ void __launch_bounds__(LSA_COMP_THREADS) k_lsa_comp_part(
        const char * __restrict__ q, const size_t q_nb1, const size_t q_nb2,
        const char * __restrict__ kmv, const size_t k_nb1, const char * __restrict__ vmv, const size_t v_nb1,
        const char * __restrict__ nd, const size_t nd_nb1, float * __restrict__ part,
        const int D, const int Dv, const int H, const int G, const int nbs, const int C, const int nsplit,
        const float scale, const float bias) {
    const int s    = blockIdx.x;
    const int t    = blockIdx.y;
    const int g    = blockIdx.z;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nw   = blockDim.x >> 5;

    __shared__ float2 qs[LSA_COMP_GMAX*LSA_COMP_DMAX/2];
    __shared__ float  sc[LSA_COMP_GMAX*LSA_COMP_CMAX];
    __shared__ int    keep[LSA_COMP_CMAX];
    __shared__ int    kidx[LSA_COMP_CMAX]; // kept block offsets in order, then nkept
    __shared__ int    nkept;
    __shared__ float  ms[LSA_COMP_GMAX], ls[LSA_COMP_GMAX];

    const int b0 = s*C;
    const int nb = min(C, nbs - b0);
    const int D2 = D/2;

    for (int i = tid; i < G*D2; i += blockDim.x) {
        const int j = i / D2, d2 = i % D2;
        const float2 v = ((const float2 *) (q + (size_t) (g*G + j)*q_nb1 + (size_t) t*q_nb2))[d2];
        qs[j*(LSA_COMP_DMAX/2) + d2] = make_float2(scale*v.x, scale*v.y);
    }
    __syncthreads();

    const float * ndr = (const float *) (nd + (size_t) t*nd_nb1);
    for (int bl = warp; bl < nb; bl += nw) {
        const float n = ndr[b0 + bl];
        if (!(n >= 0.5f)) {
            if (lane < G) {
                sc[lane*LSA_COMP_CMAX + bl] = -INFINITY;
            }
            if (lane == 0) {
                keep[bl] = 0;
            }
            continue;
        }
        const half2 * kr = (const half2 *) (kmv + (size_t) (b0 + bl)*k_nb1) + g*D2;
        float a[LSA_COMP_GMAX];
#pragma unroll
        for (int j = 0; j < LSA_COMP_GMAX; ++j) {
            a[j] = 0.0f;
        }
        for (int d2 = lane; d2 < D2; d2 += 32) {
            const float2 k = __half22float2(kr[d2]);
#pragma unroll
            for (int j = 0; j < LSA_COMP_GMAX; ++j) {
                if (j < G) {
                    const float2 qq = qs[j*(LSA_COMP_DMAX/2) + d2];
                    a[j] += k.x*qq.x + k.y*qq.y;
                }
            }
        }
        const float lb = logf(n) + bias;
#pragma unroll
        for (int j = 0; j < LSA_COMP_GMAX; ++j) {
            if (j < G) {
                const float r = warp_reduce_sum(a[j]);
                if (lane == 0) {
                    sc[j*LSA_COMP_CMAX + bl] = r + lb;
                }
            }
        }
        if (lane == 0) {
            keep[bl] = 1;
        }
    }
    __syncthreads();

    // compact the kept blocks (ballot over the chunk, order preserved) so the V loop has no branch and can
    // keep several independent mean-row loads in flight
    if (warp == LSA_COMP_THREADS/32 - 1) {
        int base = 0;
        for (int b = 0; b < nb; b += 32) {
            const int  bl = b + lane;
            const bool k  = bl < nb && keep[bl];
            const unsigned m = __ballot_sync(0xffffffffu, k);
            if (k) {
                kidx[base + __popc(m & ((1u << lane) - 1u))] = bl;
            }
            base += __popc(m);
        }
        if (lane == 0) {
            nkept = base;
        }
    }
    if (warp < G) {
        float * sj = sc + warp*LSA_COMP_CMAX;
        float mx = -INFINITY;
        for (int bl = lane; bl < nb; bl += 32) {
            mx = fmaxf(mx, sj[bl]);
        }
        mx = warp_reduce_max(mx);
        float l = 0.0f;
        for (int bl = lane; bl < nb; bl += 32) {
            const float w = mx == -INFINITY ? 0.0f : expf(sj[bl] - mx);
            sj[bl] = w;
            l += w;
        }
        l = warp_reduce_sum(l);
        if (lane == 0) {
            ms[warp] = mx;
            ls[warp] = l;
        }
    }
    __syncthreads();

    for (int d = tid; d < Dv; d += blockDim.x) {
        float a[LSA_COMP_GMAX];
#pragma unroll
        for (int j = 0; j < LSA_COMP_GMAX; ++j) {
            a[j] = 0.0f;
        }
        const half * vcol = (const half *) (vmv + (size_t) b0*v_nb1) + g*Dv + d;
        const size_t vstr = v_nb1/sizeof(half);
        const int    nk   = nkept;
        int i = 0;
        // same summation order as one block at a time; only the loads are issued 4 ahead
        for (; i + 4 <= nk; i += 4) {
            int   bi[4];
            float v[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                bi[u] = kidx[i + u];
                v[u]  = __half2float(vcol[(size_t) bi[u]*vstr]);
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
#pragma unroll
                for (int j = 0; j < LSA_COMP_GMAX; ++j) {
                    if (j < G) {
                        a[j] += sc[j*LSA_COMP_CMAX + bi[u]]*v[u];
                    }
                }
            }
        }
        for (; i < nk; ++i) {
            const int   bl = kidx[i];
            const float v  = __half2float(vcol[(size_t) bl*vstr]);
#pragma unroll
            for (int j = 0; j < LSA_COMP_GMAX; ++j) {
                if (j < G) {
                    a[j] += sc[j*LSA_COMP_CMAX + bl]*v;
                }
            }
        }
        for (int j = 0; j < G; ++j) {
            part[((size_t) (t*H + g*G + j)*nsplit + s)*(Dv + 2) + d] = a[j];
        }
    }
    if (tid < G) {
        float * p = part + ((size_t) (t*H + g*G + tid)*nsplit + s)*(Dv + 2) + Dv;
        p[0] = ms[tid];
        p[1] = ls[tid];
    }
}

// one CTA per (head, row): softmax over [chunk partials | lse] and the merged output
static __global__ void k_lsa_comp_combine(
        const float * __restrict__ part, const char * __restrict__ o_kept, const size_t o_nb1,
        const float * __restrict__ lse, char * __restrict__ dst, const size_t dst_nb1,
        const int H, const int Dv, const int nsplit) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    __shared__ float fs[LSA_COMP_NSPLITMAX];
    __shared__ float hdr[2];

    const float * pp = part + (size_t) (t*H + h)*nsplit*(Dv + 2);
    const float   L  = lse[h + t*H];
    if (threadIdx.x == 0) {
        float M = L;
        for (int s = 0; s < nsplit; ++s) {
            M = fmaxf(M, pp[(size_t) s*(Dv + 2) + Dv]);
        }
        float wk = 0.0f, z = 0.0f;
        if (M != -INFINITY) {
            wk = L == -INFINITY ? 0.0f : expf(L - M);
            z  = wk;
            for (int s = 0; s < nsplit; ++s) {
                const float m = pp[(size_t) s*(Dv + 2) + Dv];
                const float f = m == -INFINITY ? 0.0f : expf(m - M);
                fs[s] = f;
                z += f*pp[(size_t) s*(Dv + 2) + Dv + 1];
            }
        }
        hdr[0] = M == -INFINITY ? 1.0f : wk; // nothing to merge: out = o_kept
        hdr[1] = M == -INFINITY ? 1.0f : z;
        if (M == -INFINITY) {
            for (int s = 0; s < nsplit; ++s) {
                fs[s] = 0.0f;
            }
        }
    }
    __syncthreads();
    const float wk = hdr[0];
    const float iz = 1.0f/hdr[1];
    const float * orow = (const float *) (o_kept + (size_t) t*o_nb1) + (size_t) h*Dv;
    float       * out  = (float *) (dst + (size_t) t*dst_nb1) + (size_t) h*Dv;
    for (int d = threadIdx.x; d < Dv; d += blockDim.x) {
        float a = wk*orow[d];
        for (int s = 0; s < nsplit; ++s) {
            a += fs[s]*pp[(size_t) s*(Dv + 2) + d];
        }
        out[d] = a*iz;
    }
}

bool ggml_cuda_lsa_comp_merge_supported(const ggml_tensor * op) {
    const ggml_tensor * o_kept = op->src[0];
    const ggml_tensor * lse    = op->src[1];
    const ggml_tensor * q      = op->src[2];
    const ggml_tensor * kmv    = op->src[3];
    const ggml_tensor * vmv    = op->src[4];
    const ggml_tensor * nd     = op->src[5];
    if (!o_kept || !lse || !q || !kmv || !vmv || !nd || op->type != GGML_TYPE_F32) {
        return false;
    }
    const int32_t H_kv = ggml_get_op_params_i32(op, 0);
    const int64_t D = q->ne[0], H = q->ne[1], T = q->ne[2], nbs = nd->ne[0];
    if (H_kv < 1 || H % H_kv != 0) {
        return false;
    }
    return D % 2 == 0 && D <= LSA_COMP_DMAX && H/H_kv <= LSA_COMP_GMAX && T <= 65535 && H <= 65535 &&
           nbs <= (int64_t) LSA_COMP_CMAX*LSA_COMP_NSPLITMAX && q->nb[1] % 8 == 0 && q->nb[2] % 8 == 0 &&
           q->view_offs % 8 == 0 && kmv->nb[1] % 4 == 0 && kmv->view_offs % 4 == 0 && vmv->type == GGML_TYPE_F16 &&
           vmv->nb[1] % 2 == 0;
}

void ggml_cuda_lsa_comp_merge(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * o_kept = dst->src[0];
    const ggml_tensor * lse    = dst->src[1];
    const ggml_tensor * q      = dst->src[2];
    const ggml_tensor * kmv    = dst->src[3];
    const ggml_tensor * vmv    = dst->src[4];
    const ggml_tensor * nd     = dst->src[5];

    const int32_t H_kv  = ggml_get_op_params_i32(dst, 0);
    const float   scale = ggml_get_op_params_f32(dst, 1);
    const float   bias  = ggml_get_op_params_f32(dst, 2);

    const int D = (int) q->ne[0], H = (int) q->ne[1], T = (int) q->ne[2], nbs = (int) nd->ne[0];
    const int Dv = (int) (vmv->ne[0]/H_kv);
    const int G  = H/H_kv;
    GGML_ASSERT(ggml_cuda_lsa_comp_merge_supported(dst));

    cudaStream_t stream = ctx.stream();
    const int nsm = ggml_cuda_info().devices[ctx.device].nsm;

    int nsplit = 0, C = 1;
    if (nbs > 0) {
        const int lo   = (nbs + LSA_COMP_CMAX - 1)/LSA_COMP_CMAX;
        const int hi   = std::max(lo, std::min((nbs + 31)/32, LSA_COMP_NSPLITMAX));
        // chunk count from the kv heads only (as at T = 1): extra verify rows add CTAs instead of making each
        // CTA's serial V loop longer. GGML_LSA_COMP_TSPLIT=1 restores the old split (divide by T) for A/B.
        static const bool tsplit = [] {
            const char * e = getenv("GGML_LSA_COMP_TSPLIT");
            return e && atoi(e) != 0;
        }();
        const int want = tsplit ? (2*nsm + H_kv*T - 1)/(H_kv*T) : (2*nsm + H_kv - 1)/H_kv;
        nsplit = std::min(std::max(want, lo), hi);
        C      = (nbs + nsplit - 1)/nsplit;
        nsplit = (nbs + C - 1)/C;
    }
    ggml_cuda_pool_alloc<float> part(ctx.pool(), std::max<size_t>(1, (size_t) H*T*nsplit*(Dv + 2)));
    if (nsplit > 0) {
        const dim3 grid(nsplit, T, H_kv);
        k_lsa_comp_part<<<grid, LSA_COMP_THREADS, 0, stream>>>(
            (const char *) q->data, q->nb[1], q->nb[2], (const char *) kmv->data, kmv->nb[1],
            (const char *) vmv->data, vmv->nb[1], (const char *) nd->data, nd->nb[1], part.get(),
            D, Dv, H, G, nbs, C, nsplit, scale, bias);
        CUDA_CHECK(cudaGetLastError());
    }
    const int nt = std::min(256, (Dv + 31)/32*32);
    k_lsa_comp_combine<<<dim3(H, T), nt, 0, stream>>>(
        part.get(), (const char *) o_kept->data, o_kept->nb[1], (const float *) lse->data,
        (char *) dst->data, dst->nb[1], H, Dv, nsplit);
    CUDA_CHECK(cudaGetLastError());
}
