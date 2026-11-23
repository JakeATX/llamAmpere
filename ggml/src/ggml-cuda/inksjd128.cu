// INKPREFILL: SJ-KVaRN 4/4 prefill attention for packed D128 heads (Inkling 2 full-attention visits).
//
// build_attn_rel_sj (llama-graph.cpp) packs two 128-wide KV heads into one 256-wide SJ-KVaRN cache head and pads each Q
// head with zeros on the other KV head's half ([q, 0] or [0, q]); the node carries GGML_INK_SJ_D128_PARAM. The generic
// expand-once prefill route (fattn-sjkvarn-prefill.cu) then runs D = 256 attention: half of every Q.K and P.V product
// is spent on the zero half / on the other head's V lanes, which the graph drops at the unpack.
//
// Here: each packed head's K/V rows are expanded to f16 as in fattn_sj_kvarn_expand_prefill (same loaders), rotated back
// to the head basis in shared memory (the cache stores the plain 256-point Hadamard H/16 of each row; H/16 is its own
// inverse, applied with the same butterfly sequence as sj_kvarn_rot256_row8), and the two real heads run as D = 128
// f16 MMA attention (GQA g) on the unrotated Q halves. The output is written in the unrotated basis, lanes of the other
// half zero, which is what the fused-rotation contract (GGML_SJKVARN_ROT_PARAM) asks of the node.
// Rounding differs from the D = 256 route (f16 K/V in the head basis instead of the rotated one); math is the same.
// GGML_CUDA_INK_SJ_D128=0 = off (the generic route).

#include "inksjd128.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "qsa-resident.cuh"

#include <cstdlib>

static bool ink_sj_d128_enabled() {
    static const bool on = [] {
        const char * v = getenv("GGML_CUDA_INK_SJ_D128");
        return !(v && *v && atoi(v) == 0);
    }();
    return on;
}

// One packed KV head, positions [0, n_kv): 16 rows per block, 256 threads. Loads exactly as fattn_sj_kvarn_expand_prefill
// (4/4 body records, sink / ring rows), then each row is rotated back in shared memory and written as f16.
static __global__ void k_ink_sj_d128_expand(
        const char * K, const char * V, const char * body, const int32_t * desc,
        const size_t k_row_stride, const size_t k_head_stride,
        const size_t v_row_stride, const size_t v_head_stride,
        half2 * K_out, half2 * V_out, const int head,
        const char * mask, const size_t mask_pitch, const int mask_queries) {
    constexpr int D2 = 128;
    constexpr int rows = 16;
    extern __shared__ half2 tile[];
    const int tid = threadIdx.y*32 + threadIdx.x;
    const int lane = threadIdx.x;
    const int p0 = blockIdx.x*rows;
    const fattn_sj_kvarn_ctx kv = fattn_sj_kvarn_make_ctx(body, desc, head, 2*D2, 4, 4, k_head_stride, v_head_stride);
    const size_t out_offset = (size_t) p0*D2;

    // rotate rows 2w, 2w + 1 of the tile back (warp w) and store them to out
    auto unrotate_store = [&](half2 * out) {
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
            const int r = 2*threadIdx.y + rr;
            float x[8];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = __half22float2(tile[r*D2 + 4*lane + j]);
                x[2*j] = f.x; x[2*j + 1] = f.y;
            }
            sj_kvarn_rot256_row8(x, lane);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                out[out_offset + r*D2 + 4*lane + j] = __floats2half2_rn(x[2*j], x[2*j + 1]);
            }
        }
    };

    if (p0 >= kv.S && p0 < kv.B) {
        if ((reinterpret_cast<uintptr_t>(body) & 1) && mask) {
            bool visible = false;
            for (int q0 = 0; q0 < mask_queries; q0 += 16) {
                const int q = q0 + tid/rows;
                const int p = p0 + tid%rows;
                const bool live = q < mask_queries && __half2float(reinterpret_cast<const half *>(mask + q*mask_pitch)[p]) > -INFINITY;
                if (__syncthreads_or(live)) {
                    visible = true;
                    break;
                }
            }
            if (!visible) {
                for (int i = tid; i < rows*D2; i += 256) {
                    K_out[out_offset + i] = make_half2(0.0f, 0.0f);
                    V_out[out_offset + i] = make_half2(0.0f, 0.0f);
                }
                return;
            }
        }
        const int group = (p0 - kv.S)/kv.G;
        const int token = (p0 - kv.S)%kv.G;
        const char * rec = fattn_sj_kvarn_rec(kv, group);
        flash_attn_ext_sj_kvarn_load_tile<D2, false, rows, 256, D2, false, false>(rec, token, kv, tile, rows);
        __syncthreads();
        unrotate_store(K_out);
        __syncthreads();
        flash_attn_ext_sj_kvarn_load_tile<D2, false, rows, 256, D2, false, true>(rec, token, kv, tile, rows);
        __syncthreads();
        unrotate_store(V_out);
    } else {
        const int visible_end = desc[GGML_SJKVARN_DESC_N];
        static_assert(D2 % 32 == 0, "fattn_sj_kvarn_ring_pair_warp: warp-uniform rows");
        const float tq6_mag = TQ6_CENTROIDS[32 + threadIdx.x];
        half2 * tile_v = tile + rows*D2;
        for (int i = tid; i < rows*D2; i += 256) {
            const int pos = p0 + i/D2;
            const int col = i%D2;
            half2 k = make_half2(0.0f, 0.0f);
            half2 v = make_half2(0.0f, 0.0f);
            if (pos < visible_end) {
                const int row = pos < kv.S ? pos : kv.S + (pos - kv.S)%kv.cap;
                const bool f16_sink = pos < kv.S && kv.sink_type == GGML_TYPE_F16;
                const char * kh = K + (size_t) head*k_head_stride;
                const char * vh = V + (size_t) head*v_head_stride;
                k = fattn_sj_kvarn_ring_pair_warp(f16_sink ? fattn_sj_kvarn_sink_row(kh, kv, k_row_stride, row, false) : kh + (size_t) row*k_row_stride, col, f16_sink ? GGML_TYPE_F16 : kv.type_k, tq6_mag);
                v = fattn_sj_kvarn_ring_pair_warp(f16_sink ? fattn_sj_kvarn_sink_row(vh, kv, v_row_stride, row, true) : vh + (size_t) row*v_row_stride, col, f16_sink ? GGML_TYPE_F16 : kv.type_v, tq6_mag);
            }
            tile[i] = k;
            tile_v[i] = v;
        }
        __syncthreads();
        unrotate_store(K_out);
        __syncthreads();
        // (V rows sit behind the K rows: shift the tile view for the second store)
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
            const int r = 2*threadIdx.y + rr;
            float x[8];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = __half22float2(tile_v[r*D2 + 4*lane + j]);
                x[2*j] = f.x; x[2*j + 1] = f.y;
            }
            sj_kvarn_rot256_row8(x, lane);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                V_out[out_offset + r*D2 + 4*lane + j] = __floats2half2_rn(x[2*j], x[2*j + 1]);
            }
        }
    }
}

// FA output [128, g, T] (and LSE [g, T]) of the q heads [head_first, head_first + g) into dst [256, H, T] at lanes
// [half*128, half*128 + 128), the other half zero
static __global__ void k_ink_sj_d128_scatter(const float * __restrict__ src, char * __restrict__ dst, const int g, const int T,
        const int head_first, const int half, const size_t nb1, const size_t nb2,
        const float * __restrict__ lse_src, float * __restrict__ lse_dst, const int n_head) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; // over T*g*256
    if (i >= (int64_t) T*g*256) {
        return;
    }
    const int lane = i % 256;
    const int64_t row = i / 256; // t*g + c
    const int c = row % g;
    const int t = row / g;
    const int d = lane - half*128;
    const float v = d >= 0 && d < 128 ? src[row*128 + d] : 0.0f;
    *(float *) (dst + (size_t) t*nb2 + (size_t) (head_first + c)*nb1 + (size_t) lane*sizeof(float)) = v;
    if (lse_src && lane == 0) {
        lse_dst[(size_t) t*n_head + head_first + c] = lse_src[row];
    }
}

bool ggml_cuda_ink_sj_d128_try(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!ink_sj_d128_enabled()) {
        return false;
    }
    if (ggml_get_op_params_i32(dst, GGML_INK_SJ_D128_PARAM) != GGML_INK_SJ_D128_MAGIC) {
        if (dst->src[0] && dst->src[0]->ne[0] == 256 && dst->src[0]->ne[1] >= 128) {
            static bool told_flag = false;
            if (!told_flag) {
                told_flag = true;
                GGML_LOG_WARN("[INKSJD128] SJ node without the packed flag: %s, T %d\n", dst->name, (int) dst->src[0]->ne[1]);
            }
        }
        return false;
    }
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * snk  = dst->src[4];
    ggml_tensor * lse = ggml_flash_attn_ext_get_lse(dst);
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const int n_kv = ggml_get_op_params_i32(dst, 6);
    if (!ggml_cuda_fattn_sj_kvarn_rot(dst) || !ampere_mma_available(cc) || dst->src[5] == nullptr || dst->src[6] == nullptr ||
            ggml_get_op_params_i32(dst, 5) != ((4 << 8) | 4) || ggml_get_op_params_i32(dst, 7) == GGML_TYPE_I16 ||
            ggml_get_op_params_i32(dst, 7) == GGML_TYPE_TURBO4_0 || ggml_get_op_params_i32(dst, 4) != 0 ||
            Q->type != GGML_TYPE_F32 || Q->ne[0] != 256 || Q->nb[0] != sizeof(float) || Q->ne[1] < 128 || Q->ne[3] != 1 ||
            K->ne[0] != 256 || V->ne[0] != 256 || K->ne[3] != 1 || V->ne[3] != 1 || K->ne[2] != V->ne[2] || K->ne[2] < 1 ||
            (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_Q8_0 && K->type != GGML_TYPE_TQ6_0) ||
            (V->type != GGML_TYPE_F16 && V->type != GGML_TYPE_Q8_0 && V->type != GGML_TYPE_TQ6_0) ||
            Q->ne[2] % (2*K->ne[2]) != 0 || n_kv <= 0 || n_kv % 256 != 0 ||
            !mask || mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[3] != 1 || mask->ne[0] < n_kv || mask->ne[1] < Q->ne[1] ||
            (snk && (snk->type != GGML_TYPE_F32 || snk->ne[0] != Q->ne[2])) ||
            dst->type != GGML_TYPE_F32 || dst->ne[0] != 256 || dst->ne[1] != Q->ne[2] || dst->ne[2] != Q->ne[1] || dst->nb[0] != sizeof(float) ||
            (lse && (lse->type != GGML_TYPE_F32 || !ggml_is_contiguous(lse) || lse->ne[0] != Q->ne[2] || lse->ne[1] != Q->ne[1])) ||
            (dst->src[9] && !lse) || dst->src[7] || dst->src[8] || ggml_get_op_params_f32(dst, 1) != 0.0f || ggml_get_op_params_f32(dst, 2) != 0.0f) {
        if (Q->ne[1] >= 128) {
            static bool told = false;
            if (!told) {
                told = true;
                GGML_LOG_WARN("[INKSJD128] packed node not taken (generic route): T %d, mask %s ne2 %d, op5 %x op7 %d op4 %d, rot %d, lse %d, src7 %d src8 %d src9 %d\n",
                    (int) Q->ne[1], mask ? ggml_type_name(mask->type) : "none", mask ? (int) mask->ne[2] : -1,
                    ggml_get_op_params_i32(dst, 5), ggml_get_op_params_i32(dst, 7), ggml_get_op_params_i32(dst, 4),
                    (int) ggml_cuda_fattn_sj_kvarn_rot(dst), lse != nullptr, dst->src[7] != nullptr, dst->src[8] != nullptr, dst->src[9] != nullptr);
            }
        }
        return false;
    }
    const char * body = ggml_cuda_qsa_resident_body(dst->src[5]);
    if (reinterpret_cast<uintptr_t>(body) & 3) {
        return false; // QSA / LSA resident maps: the generic route
    }
    const int T      = (int) Q->ne[1];
    const int n_head = (int) Q->ne[2];
    const int P      = (int) K->ne[2];
    const int g      = n_head/(2*P); // q heads per 128-wide KV head
    cudaStream_t stream = ctx.stream();

    const size_t kv_elems = (size_t) n_kv*256;                   // one packed head, halves
    const size_t kv_bytes = GGML_PAD(2*kv_elems*sizeof(half), 128);
    const size_t o_bytes  = GGML_PAD((size_t) g*T*128*sizeof(float), 128);
    const size_t l_bytes  = lse ? GGML_PAD((size_t) g*T*sizeof(float), 128) : 0;
    const size_t need     = kv_bytes + o_bytes + l_bytes;
    const size_t offset   = GGML_PAD(ggml_nbytes(dst), 128);
    const bool reserved = dst->buffer != nullptr && dst->view_src == nullptr && (uintptr_t) dst->data % 128 == 0 &&
        offset + need <= ggml_backend_buffer_get_alloc_size(dst->buffer, dst);
    if (!reserved && need > (size_t(256) << 20)) {
        return false; // no reservation behind dst (the generic plan did not fit): keep the VMM pool bounded
    }
    ggml_cuda_pool_alloc<char> pool(ctx.pool());
    char * scratch = reserved ? (char *) dst->data + offset : pool.alloc(need);
    half  * Ks   = (half *) scratch;
    half  * Vs   = Ks + kv_elems;
    float * outs = (float *) (scratch + kv_bytes);
    float * lses = lse ? (float *) (scratch + kv_bytes + o_bytes) : nullptr;

    for (int hp = 0; hp < P; ++hp) {
        const ggml_cuda_kernel_launch_params lp(dim3(n_kv/16, 1, 1), dim3(32, 8, 1), 2*16*128*sizeof(half2), stream);
        ggml_cuda_kernel_launch(k_ink_sj_d128_expand, lp,
            (const char *) K->data, (const char *) V->data, body, (const int32_t *) dst->src[6]->data,
            K->nb[1], K->nb[2], V->nb[1], V->nb[2], (half2 *) Ks, (half2 *) Vs, hp,
            (const char *) mask->data, mask->nb[1], T);
        CUDA_CHECK(cudaGetLastError());

        for (int half_i = 0; half_i < 2; ++half_i) {
            const int head_first = 2*g*hp + half_i*g;
            ggml_tensor q = *Q;
            q.ne[0] = 128; q.ne[2] = g;
            q.data = (char *) Q->data + (size_t) head_first*Q->nb[2] + (size_t) half_i*128*sizeof(float);
            q.view_src = nullptr; q.view_offs = 0; q.buffer = nullptr;
            ggml_tensor k = *K;
            ggml_tensor v = *V;
            for (ggml_tensor * t : {&k, &v}) {
                t->type = GGML_TYPE_F16;
                t->ne[0] = 128; t->ne[1] = n_kv; t->ne[2] = 1; t->ne[3] = 1;
                t->nb[0] = sizeof(half);
                t->nb[1] = 256*sizeof(half);
                t->nb[2] = kv_elems*sizeof(half);
                t->nb[3] = t->nb[2];
                t->view_src = nullptr; t->view_offs = 0; t->buffer = nullptr;
            }
            k.data = Ks + half_i*128;
            v.data = Vs + half_i*128;
            ggml_tensor s;
            if (snk) {
                s = *snk;
                s.ne[0] = g;
                s.data = (char *) snk->data + (size_t) head_first*snk->nb[0];
                s.view_src = nullptr; s.view_offs = 0;
            }
            ggml_tensor l;
            if (lse) {
                l = *lse;
                l.ne[0] = g; l.ne[1] = T;
                l.nb[1] = g*sizeof(float); l.nb[2] = l.nb[1]*T; l.nb[3] = l.nb[2];
                l.data = lses; l.buffer = nullptr; l.view_src = nullptr; l.view_offs = 0;
            }
            ggml_tensor out = *dst;
            for (int i = 4; i < GGML_MAX_OP_PARAMS/(int) sizeof(int32_t); ++i) {
                ggml_set_op_params_i32(&out, i, 0);
            }
            for (int i = 0; i < GGML_MAX_SRC; ++i) {
                out.src[i] = nullptr;
            }
            out.src[0] = &q;
            out.src[1] = &k;
            out.src[2] = &v;
            out.src[3] = const_cast<ggml_tensor *>(mask);
            out.src[4] = snk ? &s : nullptr;
            out.src[9] = lse ? &l : nullptr;
            out.ne[0] = 128; out.ne[1] = g; out.ne[2] = T; out.ne[3] = 1;
            out.nb[0] = sizeof(float);
            for (int d = 1; d < GGML_MAX_DIMS; ++d) {
                out.nb[d] = out.nb[d - 1]*out.ne[d - 1];
            }
            out.data = outs; out.buffer = nullptr; out.view_src = nullptr; out.view_offs = 0;
            ggml_cuda_flash_attn_ext_mma_f16_case<128, 128, 64, 1>(ctx, &out);

            const int64_t n = (int64_t) T*g*256;
            k_ink_sj_d128_scatter<<<(unsigned) ((n + 255)/256), 256, 0, stream>>>(outs, (char *) dst->data, g, T, head_first, half_i,
                dst->nb[1], dst->nb[2], lses, lse ? (float *) lse->data : nullptr, n_head);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    static bool noted = false;
    if (!noted) {
        noted = true;
        GGML_LOG_WARN("[INKSJD128] packed D128 SJ-KVaRN prefill: %d packed heads x 2 at D 128 (GQA %d), T %d, n_kv %d\n", P, g, T, n_kv);
    }
    return true;
}
