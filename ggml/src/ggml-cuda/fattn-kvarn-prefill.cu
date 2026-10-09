#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"

#include <cstdlib>

// Expand each packed tile once, then share the result across all query tiles in ordinary f16 attention.
static __global__ void fattn_kvarn_expand_prefill(
        const char * K, const char * V, const char * body, const int32_t * desc,
        const size_t k_row_stride, const size_t k_head_stride,
        const size_t v_row_stride, const size_t v_head_stride,
        half2 * K_out, half2 * V_out, const int n_kv, const int head_first) {
    constexpr int D2 = 128;
    constexpr int rows = 16;
    extern __shared__ half2 tile[];
    const int tid = threadIdx.y*32 + threadIdx.x;
    const int p0 = blockIdx.x*rows;
    const int head = head_first + blockIdx.y;
    const fattn_kvarn_ctx kv = fattn_kvarn_make_ctx(body, desc, head, 2*D2, 4, 4, k_head_stride, v_head_stride);
    const size_t out_offset = ((size_t) blockIdx.y*n_kv + p0)*D2;

    if (p0 >= kv.S && p0 < kv.B) {
        const int group = (p0 - kv.S)/kv.G;
        const int token = (p0 - kv.S)%kv.G;
        const char * rec = kv.body + (size_t) group*kv.rec_stride;
        flash_attn_ext_kvarn_load_tile<D2, false, rows, 256, D2, false, false>(rec, token, kv, tile, rows);
        __syncthreads();
        for (int i = tid; i < rows*D2; i += 256) {
            K_out[out_offset + i] = tile[i];
        }
        __syncthreads();
        flash_attn_ext_kvarn_load_tile<D2, false, rows, 256, D2, false, true>(rec, token, kv, tile, rows);
        __syncthreads();
        for (int i = tid; i < rows*D2; i += 256) {
            V_out[out_offset + i] = tile[i];
        }
    } else {
        const int visible_end = desc[GGML_KVARN_DESC_N];
        // fattn_kvarn_ring_pair_warp (#127): a warp's 32 consecutive i share one row (D2 % 32 == 0), so pos, f16_sink
        // and the row type are warp-uniform and every lane reaches the shuffle
        static_assert(D2 % 32 == 0, "fattn_kvarn_ring_pair_warp: warp-uniform rows");
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
                k = fattn_kvarn_ring_pair_warp(f16_sink ? fattn_kvarn_sink_row(kh, kv, k_row_stride, row, false) : kh + (size_t) row*k_row_stride, col, f16_sink ? GGML_TYPE_F16 : kv.type_k, tq6_mag);
                v = fattn_kvarn_ring_pair_warp(f16_sink ? fattn_kvarn_sink_row(vh, kv, v_row_stride, row, true) : vh + (size_t) row*v_row_stride, col, f16_sink ? GGML_TYPE_F16 : kv.type_v, tq6_mag);
            }
            K_out[out_offset + i] = k;
            V_out[out_offset + i] = v;
        }
    }
}

static __global__ void fattn_kvarn_scatter_prefill(
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

static size_t fattn_kvarn_prefill_budget() {
    static const size_t budget = [] {
        const char * value = getenv("GGML_KVARN_PREFILL_MIB");
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

// One plan decides eligibility and sizes for the executor below and for the allocator
// (ggml_cuda_flash_attn_ext_kvarn_prefill_alloc_size, called from ggml_cuda_flash_attn_ext_get_alloc_size), so they
// cannot disagree. n_kv_max >= n_kv (0 = n_kv) only widens the reservation bound; the executor's grouping depends on
// n_kv alone, exactly as before.
struct fattn_kvarn_prefill_plan {
    bool   ok               = false;
    bool   grouped          = false;
    int    heads_per_group  = 0;
    size_t kv_elements_per_head    = 0;
    size_t output_elements_per_head = 0;
    size_t kv_elements      = 0;     // one group's f16 K elements (V has as many)
    size_t output_bytes     = 0;     // one group's f32 output temporary (grouped only)
    size_t scratch_bytes    = 0;     // 2*kv_elements*2 + output_bytes
};

static fattn_kvarn_prefill_plan fattn_kvarn_prefill_make_plan(const int cc, const ggml_tensor * dst) {
    fattn_kvarn_prefill_plan p;
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const int n_kv = ggml_get_op_params_i32(dst, 6);
    const size_t budget = fattn_kvarn_prefill_budget();
    if (budget == 0 || Q->ne[1] < 128 || !ampere_mma_available(cc) || ggml_get_op_params_i32(dst, 7) == GGML_TYPE_I16 ||
            Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1 ||
            Q->ne[0] != 256 || K->ne[0] != 256 || V->ne[0] != 256 ||
            (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_Q8_0 && K->type != GGML_TYPE_TQ6_0) ||
            (V->type != GGML_TYPE_F16 && V->type != GGML_TYPE_Q8_0 && V->type != GGML_TYPE_TQ6_0) ||
            K->ne[2] != V->ne[2] || K->ne[2] <= 0 || K->ne[2] > 65535 ||
            dst->src[5] == nullptr || dst->src[6] == nullptr ||
            ggml_get_op_params_i32(dst, 5) != ((4 << 8) | 4) || n_kv <= 0 || n_kv % 256 != 0) {
        return p;
    }
    // [#139] a fused-rotation node is planned like the unflagged copy that ggml_cuda_flash_attn_ext_kvarn_rot_unfused
    // dispatches, so the allocator reserves the same scratch; the executor below asserts the flag is gone.

    const int gqa = Q->ne[2]/K->ne[2];
    const size_t kv_elements_per_head = size_t(n_kv)*256;
    const size_t kv_bytes_per_head = 2*kv_elements_per_head*sizeof(half);
    const size_t output_elements_per_head = size_t(Q->ne[1])*gqa*256;
    const size_t output_bytes_per_head = output_elements_per_head*sizeof(float);
    const bool grouped = size_t(K->ne[2]) > budget/kv_bytes_per_head;
    int heads_per_group = K->ne[2];
    if (grouped) {
        static const bool head_groups = [] {
            const char * value = getenv("GGML_KVARN_PREFILL_HEAD_GROUPS");
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
size_t ggml_cuda_flash_attn_ext_kvarn_prefill_alloc_size(const int device, const ggml_tensor * dst) {
    const fattn_kvarn_prefill_plan p = fattn_kvarn_prefill_make_plan(ggml_cuda_info().devices[device].cc, dst);
    if (!p.ok) {
        return 0;
    }
    const size_t n_head_kv = dst->src[1]->ne[2];
    const size_t bound = std::max(p.scratch_bytes,
        std::min(fattn_kvarn_prefill_budget(), n_head_kv*2*p.kv_elements_per_head*sizeof(half)));
    return GGML_PAD(ggml_nbytes(dst), 128) + bound;
}

bool ggml_cuda_flash_attn_ext_kvarn_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const int n_kv = ggml_get_op_params_i32(dst, 6);
    const fattn_kvarn_prefill_plan plan = fattn_kvarn_prefill_make_plan(ggml_cuda_info().devices[ctx.device].cc, dst);
    if (!plan.ok) {
        return false;
    }
    GGML_ASSERT(!ggml_cuda_fattn_kvarn_rot(dst)); // [#139] no in-kernel rotation: fattn.cu runs it in separate passes
    const int gqa = Q->ne[2]/K->ne[2];
    const bool grouped = plan.grouped;
    const int heads_per_group = plan.heads_per_group;
    const size_t kv_elements_per_head = plan.kv_elements_per_head;
    const size_t output_elements_per_head = plan.output_elements_per_head;
    const size_t kv_elements = plan.kv_elements;
    const size_t output_bytes = plan.output_bytes;

    // Scratch: the region ggml_cuda_flash_attn_ext_kvarn_prefill_alloc_size reserved behind dst in the compute buffer,
    // or the pool when dst was not allocated through the buffer type (a view, no buffer) or the region is short.
    const size_t offset = GGML_PAD(ggml_nbytes(dst), 128);
    const bool reserved = dst->buffer != nullptr && dst->view_src == nullptr && (uintptr_t) dst->data % 128 == 0 &&
        offset + plan.scratch_bytes <= ggml_backend_buffer_get_alloc_size(dst->buffer, dst);
    ggml_cuda_pool_alloc<half> scratch_pool(ctx.pool());
    half * scratch_ptr = reserved ? (half *) ((char *) dst->data + offset) :
        scratch_pool.alloc(2*kv_elements + output_bytes/sizeof(half));

    float * output = grouped ? (float *) (scratch_ptr + 2*kv_elements) : (float *) dst->data;
    static const bool trace = [] {
        const char * value = getenv("GGML_KVARN_PREFILL_TRACE");
        return value != nullptr && value[0] == '1';
    }();

    for (int head_first = 0; head_first < K->ne[2]; head_first += heads_per_group) {
        const int heads = std::min(heads_per_group, int(K->ne[2]) - head_first);
        const size_t elements = kv_elements_per_head*heads;
        const dim3 blocks(n_kv/16, heads, 1);
        const dim3 threads(32, 8, 1);
        const ggml_cuda_kernel_launch_params launch_params(blocks, threads, 16*128*sizeof(half2), ctx.stream());
        ggml_cuda_kernel_launch(fattn_kvarn_expand_prefill, launch_params,
            (const char *) K->data, (const char *) V->data,
            (const char *) dst->src[5]->data, (const int32_t *) dst->src[6]->data,
            K->nb[1], K->nb[2], V->nb[1], V->nb[2],
            (half2 *) scratch_ptr, (half2 *) (scratch_ptr + elements), n_kv, head_first);
        CUDA_CHECK(cudaGetLastError());

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

        if (trace) {
            GGML_LOG_INFO("KVarN prefill: queries=%lld positions=%d kv_heads=%d first_head=%d scratch=%zu KiB%s%s\n",
                (long long) Q->ne[1], n_kv, heads, head_first,
                (2*kv_elements*sizeof(half) + output_bytes)/1024, grouped ? " grouped" : "", reserved ? " reserved" : " pool");
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
            ggml_cuda_kernel_launch(fattn_kvarn_scatter_prefill, copy_params,
                output, (char *) dst->data, output_elements, heads*gqa, head_first*gqa,
                dst->nb[0], dst->nb[1], dst->nb[2]);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    return true;
}
