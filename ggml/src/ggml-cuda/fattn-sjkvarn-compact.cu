#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "../ggml-sjkvarn.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
namespace {

// fork: keep the union in original position order; the n_kv_max contract bounds its fixed graph-safe storage.
// Two phases over 2048-position chunks so the whole card scans the mask (one block per chunk) instead of one block
// walking all positions: phase 1 counts the live positions of each chunk, phase 2 turns the chunk counts into offsets
// and writes the ordered position list plus the -1 padding. Both phases use the same liveness test, so the list is the
// exact union of the finite mask entries of every query row.
constexpr int FATTN_SJ_COMPACT_CHUNK = 256*8;

static __device__ __forceinline__ uint32_t fattn_sj_compact_live_ballot(
        const half * mask, const int pitch, const int queries, const int positions, const int p) {
    bool live = false;
    if (p < positions) {
        for (int q = 0; q < queries; ++q) {
            live |= __half2float(mask[size_t(q)*pitch + p]) != -INFINITY;
        }
    }
    return __ballot_sync(0xFFFFFFFF, live);
}

static __global__ void fattn_sj_compact_count(
        const half * mask, const int pitch, const int queries, const int positions, int32_t * chunk_counts) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps = 256/warp_size;
    __shared__ int warp_counts[nwarps];
    const int tid = threadIdx.x;
    const int warp = tid/warp_size;
    const int lane = tid%warp_size;
    const int p0 = blockIdx.x*FATTN_SJ_COMPACT_CHUNK;
    int count = 0;
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        const int p = p0 + (warp*8 + item)*warp_size + lane;
        count += __popc(fattn_sj_compact_live_ballot(mask, pitch, queries, positions, p));
    }
    if (lane == 0) {
        warp_counts[warp] = count;
    }
    __syncthreads();
    if (tid == 0) {
        int total = 0;
        for (int w = 0; w < nwarps; ++w) {
            total += warp_counts[w];
        }
        chunk_counts[blockIdx.x] = total;
    }
}

static __global__ void fattn_sj_compact_indices(
        const half * mask, const int pitch, const int queries, const int positions,
        const int32_t * chunk_counts, const int n_chunks, int32_t * indices, int32_t * total_out, const int capacity) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps = 256/warp_size;
    __shared__ int warp_offsets[nwarps];
    __shared__ int before, total;
    const int tid = threadIdx.x;
    const int warp = tid/warp_size;
    const int lane = tid%warp_size;
    if (tid == 0) {
        int sum_before = 0, sum_all = 0;
        for (int c = 0; c < n_chunks; ++c) {
            sum_all += chunk_counts[c];
            sum_before += c < blockIdx.x ? chunk_counts[c] : 0;
        }
        before = sum_before;
        total = sum_all;
    }
    __syncthreads();
    if (total > capacity) {
        if (tid == 0 && blockIdx.x == 0) {
            printf("SJ compact: mask union %d exceeds n_kv_max bound %d\n", total, capacity);
            __trap();
        }
        return;
    }
    const int p0 = blockIdx.x*FATTN_SJ_COMPACT_CHUNK;
    uint32_t selected[8];
    int count = 0;
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        const int p = p0 + (warp*8 + item)*warp_size + lane;
        selected[item] = fattn_sj_compact_live_ballot(mask, pitch, queries, positions, p);
        count += __popc(selected[item]);
    }
    if (lane == 0) {
        warp_offsets[warp] = count;
    }
    __syncthreads();
    if (tid == 0) {
        int prefix = 0;
        for (int w = 0; w < nwarps; ++w) {
            const int count_w = warp_offsets[w];
            warp_offsets[w] = prefix;
            prefix += count_w;
        }
    }
    __syncthreads();
    const uint32_t earlier = lane == 0 ? 0 : (uint32_t(1) << lane) - 1;
    int item_prefix = 0;
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        const int dst = before + warp_offsets[warp] + item_prefix + __popc(selected[item] & earlier);
        if (selected[item] & (uint32_t(1) << lane)) {
            indices[dst] = p0 + (warp*8 + item)*warp_size + lane;
        }
        item_prefix += __popc(selected[item]);
    }
    // -1 padding after the last live position, spread over all blocks
    for (int i = total + blockIdx.x*256 + tid; i < capacity; i += gridDim.x*256) {
        indices[i] = -1;
    }
    if (blockIdx.x == 0 && tid == 0) {
        // [0] live count, [1] the attended row bound: live count padded to whole 256-row strides, at least one stride
        total_out[0] = total;
        total_out[1] = max(256, int(GGML_PAD(total, 256)));
    }
}

static __global__ void fattn_sj_compact_mask(
        const half * src, const int pitch, const int32_t * indices, half * dst, const int capacity, const int queries) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < capacity*queries) {
        const int q = i/capacity;
        const int p = indices[i%capacity];
        dst[i] = p < 0 ? __float2half(-INFINITY) : src[size_t(q)*pitch + p];
    }
}

static __device__ __forceinline__ half fattn_sj_compact_code(
        const char * payload, const int token, const int d, const bool is_v) {
    const uint32_t bit = ggml_sj_kvarn::code_bit(token, d, is_v, 256, 128, 4);
    const uint32_t word = __ldg((const uint32_t *) payload + bit/32);
    return __float2half_rn(float((word >> (bit%32)) & 15));
}

static __global__ void fattn_sj_compact_gather(
        const char * K, const char * V, const char * body, const int32_t * desc,
        const size_t k_row_stride, const size_t k_head_stride, const size_t v_row_stride, const size_t v_head_stride,
        const int32_t * indices, const int32_t * live_count, half2 * k_out, half2 * v_out, const int capacity) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int row_out = blockIdx.x*blockDim.y + threadIdx.y;
    // rows at and past live_count[1] lie beyond the device KV bound handed to the attention kernel, which never loads them
    if (row_out >= capacity || row_out >= live_count[1]) {
        return;
    }
    const int p = indices[row_out];
    const int head = blockIdx.y;
    const fattn_sj_kvarn_ctx kv = fattn_sj_kvarn_make_ctx(body, desc, head, 256, 4, 4, k_head_stride, v_head_stride);
    const float tq6_mag = TQ6_CENTROIDS[32 + threadIdx.x];
    const size_t out = (size_t(head)*capacity + row_out)*128;
    if (kv.G != 128 || kv.body_type != GGML_TYPE_F32) {
        if (threadIdx.x == 0) {
            printf("SJ compact: expected scalar4/4 D256/G128 descriptor\n");
            __trap();
        }
        return;
    }
    if (p >= kv.S && p < kv.B) {
        const int group = (p - kv.S)/kv.G;
        const int token = (p - kv.S)%kv.G;
        const char * rec = fattn_sj_kvarn_rec(kv, group);
        const half2 ktok = __half2half2(__ldg((const half *) (rec + kv.k_tok) + token));
        const half2 vs = __half2half2(__ldg((const half *) (rec + kv.v_scale) + token));
        const half2 vz = __half2half2(__ldg((const half *) (rec + kv.v_zero) + token));
        for (int pair = threadIdx.x; pair < 128; pair += warp_size) {
            const int d = 2*pair;
            const int kd = ggml_sj_kvarn::k_ch_idx(d, 256, 128, 4);
            const half2 ks = __ldg((const half2 *) (rec + kv.k_scale) + kd/2);
            const half2 kz = __ldg((const half2 *) (rec + kv.k_zero) + kd/2);
            const half2 kcode = __halves2half2(fattn_sj_compact_code(rec, token, d, false), fattn_sj_compact_code(rec, token, d + 1, false));
            k_out[out + pair] = __hmul2(__hfma2(kcode, ks, kz), ktok);
            const half2 vcode = __halves2half2(fattn_sj_compact_code(rec + kv.v_payload, token, d, true), fattn_sj_compact_code(rec + kv.v_payload, token, d + 1, true));
            const half2 vch = __halves2half2(
                __ldg((const half *) (rec + kv.v_ch) + ggml_sj_kvarn::v_ch_idx(d, 256, 128, 4)),
                __ldg((const half *) (rec + kv.v_ch) + ggml_sj_kvarn::v_ch_idx(d + 1, 256, 128, 4)));
            v_out[out + pair] = __hmul2(__hfma2(vcode, vs, vz), vch);
        }
    } else {
        const int row = p < kv.S ? p : kv.S + (p - kv.S)%kv.cap;
        const bool sink = p >= 0 && p < kv.S && kv.sink_type == GGML_TYPE_F16;
        const char * kh = K + size_t(head)*k_head_stride;
        const char * vh = V + size_t(head)*v_head_stride;
        const char * kr = p < 0 ? nullptr : sink ? fattn_sj_kvarn_sink_row(kh, kv, k_row_stride, row, false) : kh + size_t(row)*k_row_stride;
        const char * vr = p < 0 ? nullptr : sink ? fattn_sj_kvarn_sink_row(vh, kv, v_row_stride, row, true) : vh + size_t(row)*v_row_stride;
        for (int pair = threadIdx.x; pair < 128; pair += warp_size) {
            k_out[out + pair] = p < 0 ? make_half2(0.0f, 0.0f) : fattn_sj_kvarn_ring_pair_warp(kr, pair, sink ? GGML_TYPE_F16 : kv.type_k, tq6_mag);
            v_out[out + pair] = p < 0 ? make_half2(0.0f, 0.0f) : fattn_sj_kvarn_ring_pair_warp(vr, pair, sink ? GGML_TYPE_F16 : kv.type_v, tq6_mag);
        }
    }
}

static __global__ void fattn_sj_compact_compare(const float * ref, const float * actual, const int n) {
    __shared__ float difference[256], reference[256], maximum[256];
    __shared__ int words[256], invalid[256];
    const int tid = threadIdx.x;
    float diff = 0.0f, norm = 0.0f, error = 0.0f;
    int mismatches = 0, nonfinite = 0;
    for (int i = tid; i < n; i += 256) {
        const float a = ref[i], b = actual[i];
        const float delta = a - b;
        diff += delta*delta;
        norm += a*a;
        error = fmaxf(error, fabsf(delta));
        mismatches += __float_as_uint(a) != __float_as_uint(b);
        nonfinite |= !isfinite(a) || !isfinite(b);
    }
    difference[tid] = diff;
    reference[tid] = norm;
    maximum[tid] = error;
    words[tid] = mismatches;
    invalid[tid] = nonfinite;
    __syncthreads();
    for (int step = 128; step > 0; step /= 2) {
        if (tid < step) {
            difference[tid] += difference[tid + step];
            reference[tid] += reference[tid + step];
            maximum[tid] = fmaxf(maximum[tid], maximum[tid + step]);
            words[tid] += words[tid + step];
            invalid[tid] |= invalid[tid + step];
        }
        __syncthreads();
    }
    if (tid == 0) {
        const float relative_l2_squared = difference[0]/fmaxf(reference[0], 1e-30f);
        printf("SJ compact check: relative_l2_squared=%g maxabs=%g changed_words=%d/%d nonfinite=%d\n", relative_l2_squared, maximum[0], words[0], n, invalid[0]);
        if (invalid[0] || !isfinite(relative_l2_squared) || relative_l2_squared > 1e-4f) {
            __trap();
        }
    }
}

static __global__ void fattn_sj_compact_gather_check(
        const char * body, const int32_t * desc, const size_t k_head_stride, const size_t v_head_stride,
        const int32_t * indices, const half2 * k, const half2 * v, const int capacity) {
    const int row = blockIdx.x;
    const int p = indices[row];
    const int head = blockIdx.y;
    const fattn_sj_kvarn_ctx kv = fattn_sj_kvarn_make_ctx(body, desc, head, 256, 4, 4, k_head_stride, v_head_stride);
    if (p < kv.S || p >= kv.B) {
        return;
    }
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int tid = threadIdx.y*warp_size + threadIdx.x;
    const int token = (p - kv.S)%128;
    const int first = token/16*16;
    const char * rec = fattn_sj_kvarn_rec(kv, (p - kv.S)/128);
    extern __shared__ half2 tile[];
    for (int is_v = 0; is_v < 2; ++is_v) {
        if (is_v) {
            flash_attn_ext_sj_kvarn_load_tile<128, false, 16, 128, 128, false, true>(rec, first, kv, tile, 16);
        } else {
            flash_attn_ext_sj_kvarn_load_tile<128, false, 16, 128, 128, false, false>(rec, first, kv, tile, 16);
        }
        __syncthreads();
        const uint32_t expected = ((const uint32_t *) tile)[(token - first)*128 + tid];
        const uint32_t actual = ((const uint32_t *) (is_v ? v : k))[(size_t(head)*capacity + row)*128 + tid];
        if (expected != actual) {
            printf("SJ compact gather: row=%d head=%d V=%d pair=%d differs %08x vs %08x\n", row, head, is_v, tid, expected, actual);
            __trap();
        }
        __syncthreads();
    }
}

} // namespace
#endif

bool ggml_cuda_flash_attn_ext_sj_kvarn_compact(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        void (*run_reference)(ggml_backend_cuda_context &, ggml_tensor *)) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(ctx, dst, run_reference);
    return false;
#else
    static const int mode = [] {
        const char * value = getenv("GGML_SJKVARN_COMPACT");
        return value && strcmp(value, "1") == 0 ? 1 : value && strcmp(value, "2") == 0 ? 2 : 0;
    }();
    if (mode == 0) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const int64_t bound = ggml_get_op_params_i32(dst, 4);
    const int64_t positions = ggml_get_op_params_i32(dst, 6);
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    float max_bias;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(max_bias));
    if (cc != 860 || !ampere_mma_available(cc) || max_bias != 0.0f || dst->src[5] == nullptr || dst->src[6] == nullptr ||
            ggml_get_op_params_i32(dst, 5) != ((4 << 8) | 4) || ggml_get_op_params_i32(dst, 7) != GGML_TYPE_F32 ||
            ggml_cuda_fattn_sj_kvarn_rot(dst) || Q->type != GGML_TYPE_F32 || Q->ne[0] != 256 || Q->nb[0] != sizeof(float) ||
            (Q->ne[1] != 1 && Q->ne[1] != 4 && Q->ne[1] != 5) || Q->ne[2] != 24 || Q->ne[3] != 1 ||
            K->ne[0] != 256 || V->ne[0] != 256 || K->ne[2] != 2 || V->ne[2] != 2 || K->ne[3] != 1 || V->ne[3] != 1 ||
            (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_Q8_0 && K->type != GGML_TYPE_TQ6_0) || V->type != K->type ||
            dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) || mask == nullptr || mask->type != GGML_TYPE_F16 ||
            mask->nb[0] != sizeof(half) || mask->ne[0] < positions || mask->ne[1] < Q->ne[1] || mask->ne[2] != 1 || mask->ne[3] != 1 ||
            Q->nb[1]%16 != 0 || mask->nb[1]%16 != 0 || mask->nb[1]/sizeof(half) > INT32_MAX ||
            bound <= 0 || bound > 4096 || positions <= 0 || positions > (1 << 20) || positions%256 != 0) {
        return false;
    }
    const int capacity = int(GGML_PAD(std::min(positions, bound*Q->ne[1]), 256));
    if (capacity >= positions/2) {
        return false;
    }
    const int queries = int(Q->ne[1]);
    const size_t elements = size_t(capacity)*256*2;
    ggml_cuda_pool_alloc<int32_t> indices(ctx.pool(), capacity);
    ggml_cuda_pool_alloc<half> compact_mask(ctx.pool(), size_t(capacity)*queries);
    ggml_cuda_pool_alloc<half> kv(ctx.pool(), 2*elements);
    ggml_cuda_pool_alloc<float> reference(ctx.pool());
    if (mode == 2) {
        reference.alloc(ggml_nelements(dst));
        run_reference(ctx, dst);
        CUDA_CHECK(cudaMemcpyAsync(reference.ptr, dst->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, ctx.stream()));
    }
    const int n_chunks = int((positions + FATTN_SJ_COMPACT_CHUNK - 1)/FATTN_SJ_COMPACT_CHUNK);
    ggml_cuda_pool_alloc<int32_t> chunk_counts(ctx.pool(), n_chunks);
    ggml_cuda_pool_alloc<int32_t> live_count(ctx.pool(), 2);
    fattn_sj_compact_count<<<n_chunks, 256, 0, ctx.stream()>>>(
        (const half *) mask->data, int(mask->nb[1]/sizeof(half)), queries, int(positions), chunk_counts.ptr);
    CUDA_CHECK(cudaGetLastError());
    fattn_sj_compact_indices<<<n_chunks, 256, 0, ctx.stream()>>>(
        (const half *) mask->data, int(mask->nb[1]/sizeof(half)), queries, int(positions),
        chunk_counts.ptr, n_chunks, indices.ptr, live_count.ptr, capacity);
    CUDA_CHECK(cudaGetLastError());
    fattn_sj_compact_mask<<<(capacity*queries + 255)/256, 256, 0, ctx.stream()>>>(
        (const half *) mask->data, int(mask->nb[1]/sizeof(half)), indices.ptr, compact_mask.ptr, capacity, queries);
    CUDA_CHECK(cudaGetLastError());
    fattn_sj_compact_gather<<<dim3((capacity + 3)/4, 2), dim3(32, 4), 0, ctx.stream()>>>(
        (const char *) K->data, (const char *) V->data, ggml_cuda_qsa_resident_body(dst->src[5]), (const int32_t *) dst->src[6]->data,
        K->nb[1], K->nb[2], V->nb[1], V->nb[2], indices.ptr, live_count.ptr, (half2 *) kv.ptr, (half2 *) (kv.ptr + elements), capacity);
    CUDA_CHECK(cudaGetLastError());
    if (mode == 2) {
        fattn_sj_compact_gather_check<<<dim3(capacity, 2), dim3(32, 4), 16*128*sizeof(half2), ctx.stream()>>>(
            ggml_cuda_qsa_resident_body(dst->src[5]), (const int32_t *) dst->src[6]->data, K->nb[2], V->nb[2],
            indices.ptr, (const half2 *) kv.ptr, (const half2 *) (kv.ptr + elements), capacity);
        CUDA_CHECK(cudaGetLastError());
    }

    ggml_tensor k = *K, v = *V, m = *mask, out = *dst;
    for (ggml_tensor * t : {&k, &v}) {
        t->type = GGML_TYPE_F16;
        t->ne[1] = capacity;
        t->nb[0] = sizeof(half);
        for (int d = 1; d < GGML_MAX_DIMS; ++d) {
            t->nb[d] = t->nb[d - 1]*t->ne[d - 1];
        }
        t->view_src = nullptr;
        t->view_offs = 0;
        t->buffer = nullptr;
    }
    k.data = kv.ptr;
    v.data = kv.ptr + elements;
    m.data = compact_mask.ptr;
    m.ne[0] = capacity;
    m.ne[1] = queries;
    m.nb[0] = sizeof(half);
    for (int d = 1; d < GGML_MAX_DIMS; ++d) {
        m.nb[d] = m.nb[d - 1]*m.ne[d - 1];
    }
    m.view_src = nullptr;
    m.view_offs = 0;
    out.src[1] = &k;
    out.src[2] = &v;
    out.src[3] = &m;
    out.src[5] = nullptr;
    out.src[6] = nullptr;
    for (int i : {4, 5, 6, 7}) {
        ggml_set_op_params_i32(&out, i, 0);
    }
    // one query tile and one sequence: the kernel stops at live_count[1] instead of walking the whole capacity
    ggml_cuda_fattn_kv_max_override = live_count.ptr + 1;
    if (queries == 1) {
        ggml_cuda_flash_attn_ext_mma_f16_case<256, 256, 1, 8>(ctx, &out, false);
    } else if (queries == 4) {
        ggml_cuda_flash_attn_ext_mma_f16_case<256, 256, 4, 8>(ctx, &out, false);
    } else {
        ggml_cuda_flash_attn_ext_mma_f16_case<256, 256, 8, 8>(ctx, &out, false);
    }
    ggml_cuda_fattn_kv_max_override = nullptr;
    if (mode == 2) {
        fattn_sj_compact_compare<<<1, 256, 0, ctx.stream()>>>(reference.ptr, (const float *) dst->data, int(ggml_nelements(dst)));
        CUDA_CHECK(cudaGetLastError());
    }
    ++ctx.fusion_stats.sj_compact;
    return true;
#endif
}
