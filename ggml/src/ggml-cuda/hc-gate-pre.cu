// [HCGP] GGML_CUDA_HC_GATE_PRE=1 (opt-in): Flash-Next hyper-connection mixer, gate GEMM + gated HC_PRE in one kernel.
//
// Unfused, the mixer runs gate = mul_mat(w_up, lo) (BF16 weight [K, hc*n_embd], lo f32 [K, T], cuBLAS with lo
// rounded to BF16 and f32 accumulation) and writes gate in f32 ([hc*n_embd, T], 503 MB at a 12,288-token ubatch); HC_PRE
// then reads gate and xn and writes mixed[e, t] = scale * sum_c xn[e, c, t] * sigmoid(gate[c*n_embd + e, t]).
// Here one block computes the gate rows of all four streams for 32 embedding positions and 128 tokens with BF16 tensor
// core MMA (m16n8k16, f32 accumulate), so each thread holds the four gate values of the same (e, t) in registers and
// writes mixed directly. lo is rounded to BF16 with the same round-to-nearest-even conversion cuBLAS gets, and the
// epilogue evaluates the HC_PRE expression in the same order; only the order of the f32 accumulation over K differs
// from cuBLAS (valid-math rounding). With GGML_CUDA_HC_PRE_XN the norm did not write xn (it left one scale per row in
// the first float of that row of xn's buffer), and xn is rebuilt from the norm input exactly as dsv4_hc_pre_xn_f32 does:
// (row_scale * x) * gamma, the norm's two products in its order.

#include "common.cuh"
#include "convert.cuh"
#include "hc-gate-pre.cuh"

static constexpr int HGP_HC      = 4;
static constexpr int HGP_E       = 32;              // embedding positions per block
static constexpr int HGP_BM      = HGP_HC * HGP_E;  // 128 gate rows per block (4 streams x 32)
static constexpr int HGP_BN      = 128;             // tokens per block
static constexpr int HGP_BK      = 32;              // K per stage (64 B rows)
static constexpr int HGP_STAGES  = 3;
static constexpr int HGP_THREADS = 256;
static constexpr int HGP_GROUP_E = 8;               // e-blocks per raster group (keeps both operands' working set in L2)

static __device__ __forceinline__ uint32_t hgp_smem_u32(const void * p) {
    return (uint32_t) __cvta_generic_to_shared(p);
}

static __device__ __forceinline__ void hgp_cp_async16(uint32_t dst, const void * src, int src_bytes) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(src_bytes));
#else
    GGML_UNUSED(dst); GGML_UNUSED(src); GGML_UNUSED(src_bytes);
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ void hgp_cp_commit() {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int N>
static __device__ __forceinline__ void hgp_cp_wait() {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

static __device__ __forceinline__ void hgp_ldmatrix_x4(uint32_t & r0, uint32_t & r1, uint32_t & r2, uint32_t & r3, uint32_t addr) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
#else
    GGML_UNUSED(r0); GGML_UNUSED(r1); GGML_UNUSED(r2); GGML_UNUSED(r3); GGML_UNUSED(addr);
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ void hgp_mma_bf16(float * d, const uint32_t * a, const uint32_t * b) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    GGML_UNUSED(d); GGML_UNUSED(a); GGML_UNUSED(b);
    NO_DEVICE_CODE;
#endif
}

// smem rows are 64 B (4 chunks of 16 B); chunk c of row r is stored at chunk c ^ ((r >> 1) & 3) so that the 8 rows an
// ldmatrix phase reads land in 8 different 16 B bank groups
static __device__ __forceinline__ int hgp_swz(int row, int chunk) {
    return row * HGP_BK + ((chunk ^ ((row >> 1) & 3)) << 3);
}

template <bool raw>
static __global__ void __launch_bounds__(HGP_THREADS, 2) hc_gate_pre_kernel(
        const nv_bfloat16 * __restrict__ w,      // [K, hc*n_embd], row r = c*n_embd + e at r*lda
        const nv_bfloat16 * __restrict__ lo,     // [K, T] BF16, token t at t*K
        const float       * __restrict__ xn,     // !raw: xn[e + c*sx1 + t*sx2]; raw: the norm input xr, same strides
        const float       * __restrict__ rs,     // raw: the scale of row t*hc + c at rs[(t*hc + c)*n_embd] (HC_PRE_XN)
        const float       * __restrict__ gamma,  // raw: gamma[e + c*sg1] (sg1 == 0: one row broadcast over the streams)
        float             * __restrict__ dst,    // dst[e + t*sd1]
        const int K, const int n_embd, const int T, const int64_t lda,
        const int64_t sx1, const int64_t sx2, const int64_t sg1, const int64_t sd1, const float scale) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    __shared__ __align__(128) nv_bfloat16 sA[HGP_STAGES][HGP_BM * HGP_BK];
    __shared__ __align__(128) nv_bfloat16 sB[HGP_STAGES][HGP_BN * HGP_BK];

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm   = warp & 1;   // 16 e each
    const int wn   = warp >> 1;  // 32 tokens each

    // grouped raster: HGP_GROUP_E consecutive blocks share a token tile, a wave covers a compact e x token region
    const int n_eb = n_embd / HGP_E;
    const int n_tb = (T + HGP_BN - 1) / HGP_BN;
    const int bid  = blockIdx.x;
    const int per_group = HGP_GROUP_E * n_tb;
    const int group   = bid / per_group;
    const int first_e = group * HGP_GROUP_E;
    const int ge      = min(n_eb - first_e, HGP_GROUP_E);
    const int in_grp  = bid - group * per_group;
    const int eb      = first_e + in_grp % ge;
    const int tb      = in_grp / ge;

    const int e0 = eb * HGP_E;
    const int t0 = tb * HGP_BN;
    const int KT = K / HGP_BK;

    const auto load_stage = [&](const int s, const int kt) {
        const int k0 = kt * HGP_BK;
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int id  = tid + p * HGP_THREADS;
            const int row = id >> 2;
            const int ch  = id & 3;
            const int c   = row >> 5;
            const int e   = e0 + (row & 31);
            const nv_bfloat16 * src = w + (int64_t) (c * n_embd + e) * lda + k0 + ch * 8;
            hgp_cp_async16(hgp_smem_u32(&sA[s][hgp_swz(row, ch)]), src, 16);
        }
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int id  = tid + p * HGP_THREADS;
            const int row = id >> 2;
            const int ch  = id & 3;
            const int t   = t0 + row;
            const bool ok = t < T;
            const nv_bfloat16 * src = lo + (int64_t) (ok ? t : 0) * K + k0 + ch * 8;
            hgp_cp_async16(hgp_smem_u32(&sB[s][hgp_swz(row, ch)]), src, ok ? 16 : 0);
        }
    };

    float acc[HGP_HC][4][4];
#pragma unroll
    for (int c = 0; c < HGP_HC; ++c) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                acc[c][j][r] = 0.0f;
            }
        }
    }

#pragma unroll
    for (int s = 0; s < HGP_STAGES - 1; ++s) {
        if (s < KT) {
            load_stage(s, s);
        }
        hgp_cp_commit();
    }

    for (int kt = 0; kt < KT; ++kt) {
        hgp_cp_wait<HGP_STAGES - 2>();
        __syncthreads();
        {
            const int nk = kt + HGP_STAGES - 1;
            if (nk < KT) {
                load_stage(nk % HGP_STAGES, nk);
            }
            hgp_cp_commit();
        }
        const int s = kt % HGP_STAGES;
#pragma unroll
        for (int kk = 0; kk < HGP_BK / 16; ++kk) {
            uint32_t a[HGP_HC][4];
            uint32_t b[4][2];
#pragma unroll
            for (int c = 0; c < HGP_HC; ++c) {
                const int row = c * HGP_E + wm * 16 + (lane & 15);
                const int ch  = kk * 2 + (lane >> 4);
                hgp_ldmatrix_x4(a[c][0], a[c][1], a[c][2], a[c][3], hgp_smem_u32(&sA[s][hgp_swz(row, ch)]));
            }
#pragma unroll
            for (int jp = 0; jp < 2; ++jp) {
                const int row = wn * 32 + jp * 16 + (lane & 7) + ((lane >> 4) << 3);
                const int ch  = kk * 2 + ((lane >> 3) & 1);
                hgp_ldmatrix_x4(b[2*jp][0], b[2*jp][1], b[2*jp + 1][0], b[2*jp + 1][1], hgp_smem_u32(&sB[s][hgp_swz(row, ch)]));
            }
#pragma unroll
            for (int c = 0; c < HGP_HC; ++c) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    hgp_mma_bf16(acc[c][j], a[c], b[j]);
                }
            }
        }
    }
    hgp_cp_wait<0>();

    // epilogue: acc[c][j][h*2 + cc] is gate row c*n_embd + e, token t with
    // e = e0 + wm*16 + (lane >> 2) + 8*h, t = t0 + wn*32 + 8*j + 2*(lane & 3) + cc
    const int g = lane >> 2;
    const int q = lane & 3;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int cc = 0; cc < 2; ++cc) {
            const int t = t0 + wn * 32 + 8 * j + 2 * q + cc;
            if (t >= T) {
                continue;
            }
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int e = e0 + wm * 16 + g + 8 * h;
                float sum = 0.0f;
#pragma unroll
                for (int c = 0; c < HGP_HC; ++c) {
                    float xv;
                    if constexpr (raw) {
                        xv = rs[((int64_t) t * HGP_HC + c) * n_embd] * xn[e + c * sx1 + (int64_t) t * sx2] * gamma[e + c * sg1];
                    } else {
                        xv = xn[e + c * sx1 + (int64_t) t * sx2];
                    }
                    const float wv = 1.0f / (1.0f + expf(-acc[c][j][h * 2 + cc]));
                    sum += xv * wv;
                }
                dst[e + (int64_t) t * sd1] = scale * sum;
            }
        }
    }
#else
    GGML_UNUSED_VARS(w, lo, xn, rs, gamma, dst, K, n_embd, T, lda, sx1, sx2, sg1, sd1, scale);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
}

bool ggml_cuda_hc_gate_pre_supported(const ggml_tensor * mm, const ggml_tensor * hc_pre, int cc) {
    const ggml_tensor * w  = mm->src[0];
    const ggml_tensor * lo = mm->src[1];
    const ggml_tensor * x  = hc_pre->src[0];
    if (!(GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE)) {
        return false;
    }
    if (w->type != GGML_TYPE_BF16 || lo->type != GGML_TYPE_F32 || mm->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 ||
        hc_pre->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t K = w->ne[0];
    if (K % HGP_BK != 0 || w->nb[0] != sizeof(nv_bfloat16) || (w->nb[1] % 16) != 0 || ((uintptr_t) w->data % 16) != 0 ||
        w->ne[2] != 1 || w->ne[3] != 1) {
        return false;
    }
    if (!ggml_is_contiguous(lo) || lo->ne[0] != K || lo->ne[2] != 1 || lo->ne[3] != 1 || !ggml_is_contiguous(mm)) {
        return false;
    }
    if (x->ne[1] != HGP_HC || x->ne[3] != 1 || x->nb[0] != sizeof(float) || x->ne[0] % HGP_E != 0 ||
        w->ne[1] != x->ne[0] * HGP_HC || lo->ne[1] != x->ne[2] || mm->ne[0] != w->ne[1] || mm->ne[1] != lo->ne[1]) {
        return false;
    }
    if (hc_pre->nb[0] != sizeof(float) || hc_pre->ne[0] != x->ne[0] || hc_pre->ne[1] != x->ne[2]) {
        return false;
    }
    // the kernel takes K, n_embd and T as int; offsets are formed in 64-bit
    if ((int64_t) x->ne[2] > INT32_MAX / 2 || K > 65536) {
        return false;
    }
    return true;
}

// the HC_PRE_XN rebuild inputs this kernel can index: the norm input with xn's shape (f32, unit element stride) and a
// gamma of [n_embd, 1 or hc] (the MUL's broadcast over the streams; one gamma for every token)
bool ggml_cuda_hc_gate_pre_xn_supported(const ggml_tensor * xr, const ggml_tensor * gamma, const ggml_tensor * hc_pre) {
    const ggml_tensor * x = hc_pre->src[0];
    return xr->type == GGML_TYPE_F32 && xr->nb[0] == sizeof(float) && xr->ne[0] == x->ne[0] && xr->ne[1] == x->ne[1] &&
           xr->ne[2] == x->ne[2] && xr->ne[3] == 1 &&
           gamma->type == GGML_TYPE_F32 && gamma->nb[0] == sizeof(float) && gamma->ne[0] == x->ne[0] &&
           (gamma->ne[1] == 1 || gamma->ne[1] == x->ne[1]) && gamma->ne[2] == 1 && gamma->ne[3] == 1 &&
           gamma->nb[1] % sizeof(float) == 0;
}

void ggml_cuda_op_hc_gate_pre(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, ggml_tensor * hc_pre,
                              const ggml_tensor * xr, const float * rs, const ggml_tensor * gamma) {
    const ggml_tensor * w  = mm->src[0];
    const ggml_tensor * lo = mm->src[1];
    const ggml_tensor * x  = hc_pre->src[0];

    const int K      = (int) w->ne[0];
    const int n_embd = (int) x->ne[0];
    const int T      = (int) x->ne[2];
    const float scale = ggml_get_op_params_f32(hc_pre, 0);
    GGML_ASSERT(ggml_get_op_params_i32(hc_pre, 1) != 0); // gated only

    cudaStream_t stream = ctx.stream();

    // lo -> BF16 with the conversion the cuBLAS BF16 path applies to src1
    ggml_cuda_pool_alloc<nv_bfloat16> lo_bf16(ctx.pool(), (size_t) K * T);
    const to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
    GGML_ASSERT(to_bf16 != nullptr);
    to_bf16(lo->data, lo_bf16.get(), (int64_t) K * T, stream);

    const int n_eb = n_embd / HGP_E;
    const int n_tb = (T + HGP_BN - 1) / HGP_BN;
    const dim3 grid(n_eb * n_tb, 1, 1);
    const dim3 block(HGP_THREADS, 1, 1);

    const int64_t lda = (int64_t) (w->nb[1] / sizeof(nv_bfloat16));
    const int64_t sd1 = (int64_t) (hc_pre->nb[1] / sizeof(float));

    if (xr != nullptr) {
        GGML_ASSERT(rs != nullptr && gamma != nullptr);
        GGML_ASSERT(ggml_cuda_hc_gate_pre_xn_supported(xr, gamma, hc_pre));
        const int64_t sg1 = gamma->ne[1] == 1 ? 0 : (int64_t) (gamma->nb[1] / sizeof(float));
        hc_gate_pre_kernel<true><<<grid, block, 0, stream>>>(
                (const nv_bfloat16 *) w->data, lo_bf16.get(), (const float *) xr->data, rs, (const float *) gamma->data,
                (float *) hc_pre->data, K, n_embd, T, lda,
                (int64_t) (xr->nb[1] / sizeof(float)), (int64_t) (xr->nb[2] / sizeof(float)), sg1, sd1, scale);
    } else {
        hc_gate_pre_kernel<false><<<grid, block, 0, stream>>>(
                (const nv_bfloat16 *) w->data, lo_bf16.get(), (const float *) x->data, nullptr, nullptr,
                (float *) hc_pre->data, K, n_embd, T, lda,
                (int64_t) (x->nb[1] / sizeof(float)), (int64_t) (x->nb[2] / sizeof(float)), 0, sd1, scale);
    }
    CUDA_CHECK(cudaGetLastError());
}
