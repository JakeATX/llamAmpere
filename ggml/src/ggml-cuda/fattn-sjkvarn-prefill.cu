#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"

#include <cstdlib>

#ifndef GGML_SJKVARN_TRELLIS_PREFILL_MIB_DEFAULT
#define GGML_SJKVARN_TRELLIS_PREFILL_MIB_DEFAULT 64
#endif

// Expand each packed tile once, then share the result across all query tiles in ordinary f16 attention.
static __global__ void fattn_sj_kvarn_expand_prefill(
        const char * K, const char * V, const char * body, const int32_t * desc,
        const size_t k_row_stride, const size_t k_head_stride,
        const size_t v_row_stride, const size_t v_head_stride,
        half2 * K_out, half2 * V_out, const int n_out, const int p_begin, const int head_first, const bool skip_body,
        const char * mask, const size_t mask_pitch, const int mask_queries) {
    constexpr int D2 = 128;
    constexpr int rows = 16;
    extern __shared__ half2 tile[];
    const int tid = threadIdx.y*32 + threadIdx.x;
    const int p0 = p_begin + blockIdx.x*rows; // [TRPF] positions [p_begin, p_begin + n_out) land at rows [0, n_out)
    const int head = head_first + blockIdx.y;
    const fattn_sj_kvarn_ctx kv = fattn_sj_kvarn_make_ctx(body, desc, head, 2*D2, 4, 4, k_head_stride, v_head_stride);
    const size_t out_offset = ((size_t) blockIdx.y*n_out + (p0 - p_begin))*D2;

    if (p0 >= kv.S && p0 < kv.B) {
        if (skip_body) {
            return; // trellis body: ggml_cuda_sj_kvarn_trellis_expand_body (fattn-sjkvarn-lowbits.cu) writes these rows
        }
        if ((reinterpret_cast<uintptr_t>(body) & 1) && mask) {
            bool visible = false;
            for (int q0 = 0; q0 < mask_queries; q0 += 16) {
                const int q = q0 + tid/rows;
                const bool live = q < mask_queries && __half2float(reinterpret_cast<const half *>(mask + q*mask_pitch)[p0 + tid%rows]) > -CUDART_INF_F;
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
        for (int i = tid; i < rows*D2; i += 256) {
            K_out[out_offset + i] = tile[i];
        }
        __syncthreads();
        flash_attn_ext_sj_kvarn_load_tile<D2, false, rows, 256, D2, false, true>(rec, token, kv, tile, rows);
        __syncthreads();
        for (int i = tid; i < rows*D2; i += 256) {
            V_out[out_offset + i] = tile[i];
        }
    } else {
        const int visible_end = desc[GGML_SJKVARN_DESC_N];
        // fattn_sj_kvarn_ring_pair_warp (#127): a warp's 32 consecutive i share one row (D2 % 32 == 0), so pos, f16_sink
        // and the row type are warp-uniform and every lane reaches the shuffle
        static_assert(D2 % 32 == 0, "fattn_sj_kvarn_ring_pair_warp: warp-uniform rows");
        const float tq6_mag = TQ6_CENTROIDS[32 + threadIdx.x];
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
            K_out[out_offset + i] = k;
            V_out[out_offset + i] = v;
        }
    }
}

static __global__ void fattn_sj_kvarn_scatter_prefill(
        const float * src, char * dst, const size_t elements, const int query_heads,
        const int head_first, const size_t nb0, const size_t nb1, const size_t nb2) {
    const size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= elements) {
        return;
    }
    const int channel = i%256;
    const size_t head_row = i/256;
    const int head = head_first + head_row%query_heads;
    const size_t row = head_row/query_heads;
    *(float *) (dst + row*nb2 + (size_t) head*nb1 + (size_t) channel*nb0) = src[i];
}

static size_t fattn_sj_kvarn_prefill_budget() {
    static const size_t budget = [] {
        const char * value = getenv("GGML_SJKVARN_PREFILL_MIB");
        if (value == nullptr) {
            return size_t(256) << 20;
        }
        char * end = nullptr;
        const long mib = strtol(value, &end, 10);
        if (end == value || *end != '\0' || mib < 0 || mib > 16384) {
            return size_t(256) << 20;
        }
        return size_t(mib) << 20;
    }();
    return budget;
}

// [TRPF] 2/3-bit trellis bodies (3/3, 3/2, 2/2 with body type I16) take the same expand-once path: the body records are
// decoded by ggml_cuda_sj_kvarn_trellis_expand_body with the fused kernel's own tile loader (identical f16 values), the
// sink/ring rows by fattn_sj_kvarn_expand_prefill. Before this, n_q >= 128 trellis attention re-decoded every KV tile once
// per query tile in the one-stage low-bit tile kernel. GGML_SJKVARN_TRELLIS_EXPAND=0 restores that path.
// [TRPF] Scratch cap for trellis bodies (GGML_SJKVARN_TRELLIS_PREFILL_MIB, MiB). The trellis pairs are the low-VRAM
// configurations (the 12 GB default runs 3/3t at 204,800), so they get their own, smaller cap: when all KV heads' f16
// K/V do not fit, the KV axis is cut into chunks that do (fattn_sj_kvarn_prefill_plan::chunked) instead of head groups.
static size_t fattn_sj_kvarn_prefill_trellis_budget() {
    static const size_t budget = [] {
        constexpr size_t def = size_t(GGML_SJKVARN_TRELLIS_PREFILL_MIB_DEFAULT) << 20;
        const char * value = getenv("GGML_SJKVARN_TRELLIS_PREFILL_MIB");
        if (value == nullptr) {
            return def;
        }
        char * end = nullptr;
        const long mib = strtol(value, &end, 10);
        if (end == value || *end != '\0' || mib < 0 || mib > 16384) {
            return def;
        }
        return size_t(mib) << 20;
    }();
    return budget;
}

static bool fattn_sj_kvarn_prefill_trellis(const ggml_tensor * dst) {
    static const bool enabled = [] {
        const char * value = getenv("GGML_SJKVARN_TRELLIS_EXPAND");
        return value == nullptr || atoi(value) != 0;
    }();
    const int bits_pair = ggml_get_op_params_i32(dst, 5);
    return enabled && ggml_get_op_params_i32(dst, 7) == GGML_TYPE_I16 &&
        (bits_pair == ((3 << 8) | 3) || bits_pair == ((3 << 8) | 2) || bits_pair == ((2 << 8) | 2));
}

void ggml_cuda_sj_kvarn_trellis_expand_body(int bits_pair, const char * body, const int32_t * desc,
        size_t k_head_stride, size_t v_head_stride, half2 * K_out, half2 * V_out, int n_out, int p_begin, int heads,
        int head_first, cudaStream_t stream); // fattn-sjkvarn-lowbits.cu
void ggml_cuda_sj_kvarn_prefill_partial(ggml_backend_cuda_context & ctx, ggml_tensor * dst, float * parts, float2 * meta); // fattn.cu
void ggml_cuda_sj_kvarn_prefill_accumulate(const float * parts, const float2 * meta, float * acc, float2 * acc_meta,
        int nrows, bool initialize, cudaStream_t stream); // fattn.cu
void ggml_cuda_sj_kvarn_prefill_normalize(const float * acc, const float2 * acc_meta, float * dst, int nrows,
        cudaStream_t stream); // fattn.cu

// One plan decides eligibility and sizes for the executor below and for the allocator
// (ggml_cuda_flash_attn_ext_sj_kvarn_prefill_alloc_size, called from ggml_cuda_flash_attn_ext_get_alloc_size), so they
// cannot disagree. n_kv_max >= n_kv (0 = n_kv) only widens the reservation bound; the executor's grouping depends on
// n_kv alone, exactly as before.
struct fattn_sj_kvarn_prefill_plan {
    bool   ok               = false;
    bool   grouped          = false;
    int    heads_per_group  = 0;
    size_t kv_elements_per_head    = 0;
    size_t output_elements_per_head = 0;
    size_t kv_elements      = 0;     // one group's f16 K elements (V has as many)
    size_t output_bytes     = 0;     // one group's f32 output temporary (grouped only)
    size_t scratch_bytes    = 0;     // 2*kv_elements*2 + output_bytes (chunked: K/V chunk + parts + metas)
    // [TRPF] chunked (trellis only): all KV heads, KV positions in chunks of chunk_kv, queries in launches of work_q;
    // each launch writes unnormalized partials that are folded into dst (numerator) + acc_meta (max, sum).
    bool   chunked          = false;
    int    chunk_kv         = 0;
    int    work_q           = 0;
    size_t parts_bytes      = 0;     // work_q*n_head*256 floats
    size_t meta_bytes       = 0;     // work_q*n_head float2
};

static fattn_sj_kvarn_prefill_plan fattn_sj_kvarn_prefill_make_plan(const int cc, const ggml_tensor * dst) {
    fattn_sj_kvarn_prefill_plan p;
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const int n_kv = ggml_get_op_params_i32(dst, 6);
    const int bits_pair = ggml_get_op_params_i32(dst, 5);
    const bool trellis = fattn_sj_kvarn_prefill_trellis(dst);
    const size_t budget = trellis ? fattn_sj_kvarn_prefill_trellis_budget() : fattn_sj_kvarn_prefill_budget();
    if (budget == 0 || Q->ne[1] < 128 || !ampere_mma_available(cc) || (ggml_get_op_params_i32(dst, 7) == GGML_TYPE_I16 && !trellis) ||
            Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1 ||
            Q->ne[0] != 256 || K->ne[0] != 256 || V->ne[0] != 256 ||
            (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_Q8_0 && K->type != GGML_TYPE_TQ6_0) ||
            (V->type != GGML_TYPE_F16 && V->type != GGML_TYPE_Q8_0 && V->type != GGML_TYPE_TQ6_0) ||
            K->ne[2] != V->ne[2] || K->ne[2] <= 0 || K->ne[2] > 65535 ||
            dst->src[5] == nullptr || dst->src[6] == nullptr ||
            (bits_pair != ((4 << 8) | 4) && !trellis) || n_kv <= 0 || n_kv % 256 != 0) {
        return p;
    }
    // [#139] a fused-rotation node is planned like the unflagged copy that ggml_cuda_flash_attn_ext_sj_kvarn_rot_unfused
    // dispatches, so the allocator reserves the same scratch; the executor below asserts the flag is gone.

    const int gqa = Q->ne[2]/K->ne[2];
    const size_t kv_elements_per_head = size_t(n_kv)*256;
    const size_t kv_bytes_per_head = 2*kv_elements_per_head*sizeof(half);
    const size_t output_elements_per_head = size_t(Q->ne[1])*gqa*256;
    const size_t output_bytes_per_head = output_elements_per_head*sizeof(float);
    const bool grouped = size_t(K->ne[2]) > budget/kv_bytes_per_head;
    int heads_per_group = K->ne[2];
    static const bool kv_chunks = [] {
        const char * value = getenv("GGML_SJKVARN_PREFILL_KV_CHUNKS");
        return value == nullptr || atoi(value) != 0;
    }();
    // [TRPF] test hook: GGML_SJKVARN_PREFILL_CHUNK_KV=<multiple of 256> forces the chunked plan with at most that many
    // KV positions per chunk, so the op tests (small KV) exercise the partial + fold path.
    static const int force_chunk_kv = [] {
        const char * value = getenv("GGML_SJKVARN_PREFILL_CHUNK_KV");
        const int v = value ? atoi(value) : 0;
        return v >= 256 && v % 256 == 0 ? v : 0;
    }();
    if ((grouped || force_chunk_kv > 0) && trellis && kv_chunks && dst->src[3] != nullptr && dst->src[3]->ne[2] == 1 && dst->src[3]->ne[3] == 1 &&
            Q->ne[2] % K->ne[2] == 0 && gqa <= 8 && ggml_is_contiguous(dst)) {
        // [TRPF] KV chunks: K/V chunk (all heads) + one launch's partials + metas + the running (max, sum) per row
        const size_t n_head = Q->ne[2];
        const size_t acc_meta_bytes = size_t(Q->ne[1])*n_head*sizeof(float2);
        const size_t per_query = n_head*(256*sizeof(float) + sizeof(float2));
        int64_t work_q = Q->ne[1];
        while (work_q > 64 && acc_meta_bytes + work_q*per_query > budget/2) {
            work_q = GGML_PAD(work_q/2, 64);
        }
        const size_t fixed = acc_meta_bytes + size_t(work_q)*per_query;
        const size_t kv_bytes_per_token = size_t(K->ne[2])*2*256*sizeof(half);
        int64_t chunk_kv = fixed < budget ? int64_t((budget - fixed)/kv_bytes_per_token/256*256) : 0;
        if (force_chunk_kv > 0) {
            chunk_kv = std::min<int64_t>(std::max<int64_t>(chunk_kv, 256), force_chunk_kv);
        }
        if (chunk_kv >= 256) {
            p.ok = true;
            p.chunked = true;
            p.chunk_kv = int(std::min<int64_t>(chunk_kv, n_kv));
            p.work_q = int(work_q);
            p.parts_bytes = size_t(work_q)*n_head*256*sizeof(float);
            p.meta_bytes = size_t(work_q)*n_head*sizeof(float2);
            p.heads_per_group = K->ne[2];
            p.kv_elements_per_head = size_t(p.chunk_kv)*256;
            p.kv_elements = p.kv_elements_per_head*K->ne[2];
            p.output_elements_per_head = output_elements_per_head;
            p.scratch_bytes = GGML_PAD(2*p.kv_elements*sizeof(half), 128) + GGML_PAD(p.parts_bytes, 128) +
                GGML_PAD(p.meta_bytes, 128) + acc_meta_bytes;
            return p;
        }
    }
    if (grouped) {
        static const bool head_groups = [] {
            const char * value = getenv("GGML_SJKVARN_PREFILL_HEAD_GROUPS");
            return value == nullptr || atoi(value) != 0;
        }();
        // Each group includes complete GQA groups. Head-specific masks need a separate slicing policy.
        if (!head_groups || dst->src[3] == nullptr || dst->src[3]->ne[2] != 1) {
            return p;
        }
        heads_per_group = std::min(size_t(K->ne[2]), budget/(kv_bytes_per_head + output_bytes_per_head));
        if (heads_per_group == 0) {
            return p;
        }
    }

    // Include the output temporary in the byte cap. Other FA workspace remains owned by its launcher.
    p.ok = true;
    p.grouped = grouped;
    p.heads_per_group = heads_per_group;
    p.kv_elements_per_head = kv_elements_per_head;
    p.output_elements_per_head = output_elements_per_head;
    p.kv_elements = kv_elements_per_head*heads_per_group;
    p.output_bytes = grouped ? output_bytes_per_head*heads_per_group : 0;
    p.scratch_bytes = 2*p.kv_elements*sizeof(half) + p.output_bytes;
    return p;
}

// Bytes to reserve behind dst in the compute buffer for the expand scratch: dst padded to 128, then an upper bound on
// the scratch of every n_kv' <= this graph's n_kv (the allocator sizes each graph with its own dst). Ungrouped
// scratch is n_head_kv * f16 K+V bytes <= budget; grouped scratch is heads_per_group * (f16 K+V + output) <= budget.
// Returns 0 when the prefill expand does not apply. Before 2026-09-27 this scratch came from the CUDA VMM pool,
// which never shrinks, so it stayed resident beside the compute buffer.
size_t ggml_cuda_flash_attn_ext_sj_kvarn_prefill_alloc_size(const int device, const ggml_tensor * dst) {
    const fattn_sj_kvarn_prefill_plan p = fattn_sj_kvarn_prefill_make_plan(ggml_cuda_info().devices[device].cc, dst);
    if (!p.ok) {
        return 0;
    }
    const size_t n_head_kv = dst->src[1]->ne[2];
    const size_t budget = fattn_sj_kvarn_prefill_trellis(dst) ? fattn_sj_kvarn_prefill_trellis_budget() : fattn_sj_kvarn_prefill_budget();
    const size_t bound = std::max(p.scratch_bytes,
        std::min(budget, n_head_kv*2*size_t(ggml_get_op_params_i32(dst, 6))*256*sizeof(half)));
    return GGML_PAD(ggml_nbytes(dst), 128) + bound;
}

// [TRPF] host predicate for the low-bit dispatch: true when the executor below will take this node (a fused-rotation
// node is planned like its unflagged copy, so the caller can route it through the separate rotation passes first).
bool ggml_cuda_flash_attn_ext_sj_kvarn_prefill_applies(const int device, const ggml_tensor * dst) {
    return fattn_sj_kvarn_prefill_make_plan(ggml_cuda_info().devices[device].cc, dst).ok;
}

bool ggml_cuda_flash_attn_ext_sj_kvarn_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const int n_kv = ggml_get_op_params_i32(dst, 6);
    const fattn_sj_kvarn_prefill_plan plan = fattn_sj_kvarn_prefill_make_plan(ggml_cuda_info().devices[ctx.device].cc, dst);
    if (!plan.ok) {
        return false;
    }
    const bool trellis = fattn_sj_kvarn_prefill_trellis(dst);
    GGML_ASSERT(!ggml_cuda_fattn_sj_kvarn_rot(dst)); // [#139] no in-kernel rotation: fattn.cu runs it in separate passes
    const int gqa = Q->ne[2]/K->ne[2];
    const bool grouped = plan.grouped;
    const int heads_per_group = plan.heads_per_group;
    const size_t kv_elements_per_head = plan.kv_elements_per_head;
    const size_t output_elements_per_head = plan.output_elements_per_head;
    const size_t kv_elements = plan.kv_elements;
    const size_t output_bytes = plan.output_bytes;

    // Scratch: the region ggml_cuda_flash_attn_ext_sj_kvarn_prefill_alloc_size reserved behind dst in the compute buffer,
    // or the pool when dst was not allocated through the buffer type (a view, no buffer) or the region is short.
    const size_t offset = GGML_PAD(ggml_nbytes(dst), 128);
    const bool reserved = dst->buffer != nullptr && dst->view_src == nullptr && (uintptr_t) dst->data % 128 == 0 &&
        offset + plan.scratch_bytes <= ggml_backend_buffer_get_alloc_size(dst->buffer, dst);
    ggml_cuda_pool_alloc<half> scratch_pool(ctx.pool());
    half * scratch_ptr = reserved ? (half *) ((char *) dst->data + offset) :
        scratch_pool.alloc(2*kv_elements + output_bytes/sizeof(half));

    float * output = grouped ? (float *) (scratch_ptr + 2*kv_elements) : (float *) dst->data;

    if (plan.chunked) {
        // [TRPF] KV chunks: expand positions [c0, c0 + cn) of every KV head, run the f16 MMA kernel in partial mode on
        // them (one launch per work_q queries), fold each launch into dst (unnormalized numerator) and acc_meta
        // (running max, sum), and normalize once at the end. Attention sinks count once: only the first chunk sees them.
        const int heads = K->ne[2];
        const int n_head = Q->ne[2];
        const int chunk_kv = plan.chunk_kv;
        half * K_chunk = scratch_ptr;
        char * tail = (char *) scratch_ptr + GGML_PAD(2*kv_elements*sizeof(half), 128);
        float  * parts    = (float *) tail;
        float2 * meta     = (float2 *) (tail + GGML_PAD(plan.parts_bytes, 128));
        float2 * acc_meta = (float2 *) (tail + GGML_PAD(plan.parts_bytes, 128) + GGML_PAD(plan.meta_bytes, 128));
        float  * acc      = (float *) dst->data;
        const ggml_tensor * mask = dst->src[3];
        for (int c0 = 0; c0 < n_kv; c0 += chunk_kv) {
            const int cn = std::min(chunk_kv, n_kv - c0);
            const size_t elements = size_t(cn)*256*heads;
            const ggml_cuda_kernel_launch_params launch_params(dim3(cn/16, heads, 1), dim3(32, 8, 1), 16*128*sizeof(half2), ctx.stream());
            ggml_cuda_kernel_launch(fattn_sj_kvarn_expand_prefill, launch_params,
                (const char *) K->data, (const char *) V->data,
                ggml_cuda_qsa_resident_body(dst->src[5]), (const int32_t *) dst->src[6]->data,
                K->nb[1], K->nb[2], V->nb[1], V->nb[2],
                (half2 *) K_chunk, (half2 *) (K_chunk + elements), cn, c0, 0, true,
                dst->src[3] && dst->src[3]->ne[2] == 1 && dst->src[3]->ne[3] == 1 ? (const char *) dst->src[3]->data : nullptr,
                dst->src[3] ? dst->src[3]->nb[1] : 0, Q->ne[1]);
            CUDA_CHECK(cudaGetLastError());
            ggml_cuda_sj_kvarn_trellis_expand_body(ggml_get_op_params_i32(dst, 5), ggml_cuda_qsa_resident_body(dst->src[5]),
                (const int32_t *) dst->src[6]->data, K->nb[2], V->nb[2],
                (half2 *) K_chunk, (half2 *) (K_chunk + elements), cn, c0, heads, 0, ctx.stream());

            ggml_tensor k = *K;
            ggml_tensor v = *V;
            for (ggml_tensor * t : {&k, &v}) {
                t->type = GGML_TYPE_F16;
                t->ne[1] = cn;
                t->ne[2] = heads;
                t->nb[0] = sizeof(half);
                for (int d = 1; d < GGML_MAX_DIMS; ++d) {
                    t->nb[d] = t->nb[d - 1]*t->ne[d - 1];
                }
                t->view_src = nullptr;
                t->view_offs = 0;
            }
            k.data = K_chunk;
            v.data = K_chunk + elements;

            for (int64_t q0 = 0; q0 < Q->ne[1]; q0 += plan.work_q) {
                const int64_t qn = std::min<int64_t>(plan.work_q, Q->ne[1] - q0);
                ggml_tensor q = *Q;
                q.data = (char *) Q->data + q0*Q->nb[1];
                q.ne[1] = qn;
                ggml_tensor m = *mask;
                m.data = (char *) mask->data + q0*mask->nb[1] + size_t(c0)*mask->nb[0];
                m.ne[0] = cn;
                m.ne[1] = qn;
                m.view_src = nullptr;
                m.view_offs = 0;
                ggml_tensor out = *dst;
                out.src[0] = &q;
                out.src[1] = &k;
                out.src[2] = &v;
                out.src[3] = &m;
                out.src[4] = c0 == 0 ? dst->src[4] : nullptr;
                out.src[5] = nullptr;
                out.src[6] = nullptr;
                ggml_set_op_params_i32(&out, 5, 0);
                ggml_set_op_params_i32(&out, 6, 0);
                out.ne[2] = qn;
                for (int d = 2; d < GGML_MAX_DIMS; ++d) {
                    out.nb[d] = out.nb[d - 1]*out.ne[d - 1];
                }
                out.data = parts;
                out.buffer = nullptr;
                out.view_src = nullptr;
                out.view_offs = 0;
                ggml_cuda_sj_kvarn_prefill_partial(ctx, &out, parts, meta);
                const size_t row0 = size_t(q0)*n_head;
                ggml_cuda_sj_kvarn_prefill_accumulate(parts, meta, acc + row0*256, acc_meta + row0,
                    int(qn*n_head), c0 == 0, ctx.stream());
            }
        }
        ggml_cuda_sj_kvarn_prefill_normalize(acc, acc_meta, acc, int(Q->ne[1]*n_head), ctx.stream());
        return true;
    }

    for (int head_first = 0; head_first < K->ne[2]; head_first += heads_per_group) {
        const int heads = std::min(heads_per_group, int(K->ne[2]) - head_first);
        const size_t elements = kv_elements_per_head*heads;
        const dim3 blocks(n_kv/16, heads, 1);
        const dim3 threads(32, 8, 1);
        const ggml_cuda_kernel_launch_params launch_params(blocks, threads, 16*128*sizeof(half2), ctx.stream());
        ggml_cuda_kernel_launch(fattn_sj_kvarn_expand_prefill, launch_params,
            (const char *) K->data, (const char *) V->data,
            ggml_cuda_qsa_resident_body(dst->src[5]), (const int32_t *) dst->src[6]->data,
            K->nb[1], K->nb[2], V->nb[1], V->nb[2],
            (half2 *) scratch_ptr, (half2 *) (scratch_ptr + elements), n_kv, 0, head_first, trellis,
            dst->src[3] && dst->src[3]->ne[2] == 1 && dst->src[3]->ne[3] == 1 ? (const char *) dst->src[3]->data : nullptr,
            dst->src[3] ? dst->src[3]->nb[1] : 0, Q->ne[1]);
        CUDA_CHECK(cudaGetLastError());
        if (trellis) {
            ggml_cuda_sj_kvarn_trellis_expand_body(ggml_get_op_params_i32(dst, 5), ggml_cuda_qsa_resident_body(dst->src[5]),
                (const int32_t *) dst->src[6]->data, K->nb[2], V->nb[2],
                (half2 *) scratch_ptr, (half2 *) (scratch_ptr + elements), n_kv, 0, heads, head_first, ctx.stream());
        }

        ggml_tensor q = *Q;
        q.ne[2] = heads*gqa;
        q.data = (char *) Q->data + (size_t) head_first*gqa*Q->nb[2];
        ggml_tensor k = *K;
        ggml_tensor v = *V;
        for (ggml_tensor * t : {&k, &v}) {
            t->type = GGML_TYPE_F16;
            t->ne[1] = n_kv;
            t->ne[2] = heads;
            t->nb[0] = sizeof(half);
            for (int d = 1; d < GGML_MAX_DIMS; ++d) {
                t->nb[d] = t->nb[d - 1]*t->ne[d - 1];
            }
            t->view_src = nullptr;
            t->view_offs = 0;
        }
        k.data = scratch_ptr;
        v.data = scratch_ptr + elements;
        ggml_tensor out = *dst;
        out.src[0] = &q;
        out.src[1] = &k;
        out.src[2] = &v;
        out.src[5] = nullptr;
        out.src[6] = nullptr;
        ggml_set_op_params_i32(&out, 5, 0);
        ggml_set_op_params_i32(&out, 6, 0);
        ggml_tensor sinks;
        if (grouped) {
            out.ne[1] = q.ne[2];
            out.nb[0] = sizeof(float);
            for (int d = 1; d < GGML_MAX_DIMS; ++d) {
                out.nb[d] = out.nb[d - 1]*out.ne[d - 1];
            }
            out.data = output;
            out.buffer = nullptr;
            out.view_src = nullptr;
            out.view_offs = 0;
            if (dst->src[4] != nullptr) {
                sinks = *dst->src[4];
                sinks.ne[0] = q.ne[2];
                sinks.data = (char *) sinks.data + (size_t) head_first*gqa*sinks.nb[0];
                out.src[4] = &sinks;
            }
        }

        if (gqa > 4) {
            ggml_cuda_flash_attn_ext_mma_f16_case<256, 256, 8, 8>(ctx, &out);
        } else if (gqa > 2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<256, 256, 16, 4>(ctx, &out);
        } else if (gqa > 1) {
            ggml_cuda_flash_attn_ext_mma_f16_case<256, 256, 32, 2>(ctx, &out);
        } else {
            ggml_cuda_flash_attn_ext_mma_f16_case<256, 256, 64, 1>(ctx, &out);
        }
        if (grouped) {
            const size_t output_elements = output_elements_per_head*heads;
            const ggml_cuda_kernel_launch_params copy_params(dim3((output_elements + 255)/256), dim3(256), 0, ctx.stream());
            ggml_cuda_kernel_launch(fattn_sj_kvarn_scatter_prefill, copy_params,
                output, (char *) dst->data, output_elements, heads*gqa, head_first*gqa,
                dst->nb[0], dst->nb[1], dst->nb[2]);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    return true;
}
