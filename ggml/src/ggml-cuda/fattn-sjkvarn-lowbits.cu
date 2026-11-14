#include "fattn-sjkvarn-lowbits-stream.cuh"
#include "sjkvarn-seal.cuh"
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <string>

// GGML_SJKVARN_LIVE_TILES=1 (opt-in): the trellis tile kernel walks only the KV tiles (nbatch_fa positions) that some query
// row can see, split evenly over its stream-k blocks. Without it every block gets an even share of ALL tiles and the
// dead-tile check runs after the split, so with an LSA mask (sink + KEEP picks + tail) a few blocks carry the live tiles.
// A dead tile adds exp(-inf) = 0 and leaves the running max alone, so the sum is the same; the block split moves, so the
// float grouping (and possibly the last bits) can differ from the dense walk.
static bool sj_kvarn_live_tiles() {
    static const bool v = [] { const char * e = getenv("GGML_SJKVARN_LIVE_TILES"); return e != nullptr && e[0] == '1'; }();
    return v;
}

// flags[t] = 1 iff some row q < n_q has a visible entry in columns [t*nbatch_fa, (t+1)*nbatch_fa): one warp per tile.
// F16 mask: entry != -inf (the tile kernel's TILE_SKIP test). Packed I32 mask [#175]: one bit per column, set = visible,
// tiles start on a word (nbatch_fa % 32 == 0), so a nonzero word is a visible column.
static __global__ void sj_kvarn_live_tile_flags(const char * __restrict__ mask, const int64_t row_bytes, const int n_q,
        const int packed, const int nbatch_fa, const int n_tiles, uint8_t * __restrict__ flags) {
    const int t    = blockIdx.x*(blockDim.x/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    if (t >= n_tiles) {
        return;
    }
    bool live = false;
    for (int q = 0; q < n_q && !live; ++q) {
        const char * row = mask + q*row_bytes;
        bool f = false;
        if (packed) {
            for (int w = lane; w < nbatch_fa/32; w += WARP_SIZE) {
                f |= ((const uint32_t *) row)[t*(nbatch_fa/32) + w] != 0u;
            }
        } else {
            for (int c = lane; c < nbatch_fa; c += WARP_SIZE) {
                f |= __half2float(((const half *) row)[t*nbatch_fa + c]) != -INFINITY;
            }
        }
        live = __any_sync(0xFFFFFFFF, f);
    }
    if (lane == 0) {
        flags[t] = live ? 1 : 0;
    }
}

// list[0] = live count, list[1 ..] = live tile ids ascending (one block of 1024: ballot scan per 1024 tiles, the
// sj_live_record_list pattern of the 4/4 stream kernel)
static __global__ void __launch_bounds__(1024) sj_kvarn_live_tile_list(const uint8_t * __restrict__ flags, const int n_tiles,
        int32_t * __restrict__ list) {
    __shared__ int warp_counts[32];
    __shared__ int base;
    const int tid = threadIdx.x, lane = tid % WARP_SIZE, warp = tid / WARP_SIZE, nwarps = blockDim.x / WARP_SIZE;
    if (tid == 0) {
        base = 0;
    }
    __syncthreads();
    for (int t0 = 0; t0 < n_tiles; t0 += blockDim.x) {
        const int t = t0 + tid;
        const bool live = t < n_tiles && flags[t] != 0;
        const uint32_t bal = __ballot_sync(0xFFFFFFFF, live);
        if (lane == 0) {
            warp_counts[warp] = __popc(bal);
        }
        __syncthreads();
        int before = base;
        for (int w = 0; w < warp; ++w) {
            before += warp_counts[w];
        }
        if (live) {
            list[1 + before + __popc(bal & ((1u << lane) - 1u))] = t;
        }
        __syncthreads();
        if (tid == 0) {
            for (int w = 0; w < nwarps; ++w) {
                base += warp_counts[w];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        list[0] = base;
    }
}

template <int DKQ, int DV, int ncols1, int ncols2, int bits_k, int bits_v, bool rot = false>
void ggml_cuda_lowbits_prefill_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    constexpr ggml_type type_K = GGML_TYPE_I8;
    constexpr ggml_type type_V = GGML_TYPE_I8;
    constexpr int ncols = ncols1 * ncols2;

    GGML_ASSERT(dst->src[5] != nullptr && dst->src[6] != nullptr);
    GGML_ASSERT(ggml_get_op_params_i32(KQV, 5) == (((bits_k & 7) << 8) | bits_v)); // bits_k | 8 / | 16: word-load / byte-load trellis build
    GGML_ASSERT(DKQ == 256 && DV == 256);
    GGML_ASSERT(rot == ggml_cuda_fattn_sj_kvarn_rot(dst)); // [#139] the rot build rotates Q and the output in-kernel
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_F16 || dst->src[1]->type == GGML_TYPE_Q8_0 || dst->src[1]->type == GGML_TYPE_TQ6_0);
    GGML_ASSERT(dst->src[2]->type == GGML_TYPE_F16 || dst->src[2]->type == GGML_TYPE_Q8_0 || dst->src[2]->type == GGML_TYPE_TQ6_0);

    const int  nthreads       = ggml_cuda_fattn_mma_get_nthreads      (DKQ, DV, ncols, cc);
    const int  nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols, cc);
    const int  nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols, cc);
    const int  nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols, cc);
    const int  nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols, cc);
    if (dst->src[0]->ne[1] <= 8 && ggml_cuda_lsa_page_full_stage(dst->src[5])) {
        GGML_ASSERT(nbatch_fa <= GGML_LSA_PAGE_CELLS && GGML_LSA_PAGE_CELLS % nbatch_fa == 0);
    }
    const bool Q_in_reg       = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols, cc);

    const int cols_per_warp  = std::min(ncols, get_cols_per_warp(cc));
    const int warp_size_host = ggml_cuda_info().devices[ctx.device].warp_size;
    const int nwarps         = nthreads / warp_size_host;

    constexpr bool V_is_K_view = false;

    const int stride_tile_K = ggml_cuda_fattn_smem_swizzle::tile_stride(nbatch_K2, cc);
    const int stride_tile_V = ggml_cuda_fattn_smem_swizzle::tile_stride(nbatch_V2, cc);
    const size_t nbytes_shared_KV_1stage = nbatch_fa            * std::max(stride_tile_K, stride_tile_V) * sizeof(half2);
    const size_t nbytes_shared_Q         = ncols                * (DKQ/2 + 4)                             * sizeof(half2);
    const size_t nbytes_shared_mask      = ncols1               * (nbatch_fa/2 + 4)                       * sizeof(half2);
    const size_t nbytes_shared_combine   = nwarps*cols_per_warp * (nbatch_combine + 4)                    * sizeof(half2);

    const size_t nbytes_shared_KV = nbytes_shared_KV_1stage;

    const size_t nbytes_shared_total = std::max(nbytes_shared_combine, Q_in_reg ?
        std::max(nbytes_shared_Q,  nbytes_shared_KV + nbytes_shared_mask) :
                 nbytes_shared_Q + nbytes_shared_KV + nbytes_shared_mask);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

#if defined(GGML_USE_HIP)
    using fattn_kernel_ptr_t = const void*;
#else
    using fattn_kernel_ptr_t = fattn_kernel_t;
#endif // defined(GGML_USE_HIP)
    fattn_kernel_t fattn_kernel;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, /* use_sparse */ false, type_K, type_V, bits_k, bits_v, rot>;

#if !defined(GGML_USE_MUSA)
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
            shared_memory_limit_raised[id] = true;
        }
#endif // !defined(GGML_USE_MUSA)
    } else if constexpr (rot) {
        GGML_ABORT("SJ-KVaRN fused rotation with logit softcap (the caller takes the separate passes)");
    } else {
        constexpr bool use_logit_softcap = true;
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, /* use_sparse */ false, type_K, type_V, bits_k, bits_v>;

#if !defined(GGML_USE_MUSA)
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
            shared_memory_limit_raised[id] = true;
        }
#endif // !defined(GGML_USE_MUSA)
    }

    // GGML_SJKVARN_LIVE_TILES: trellis build, one shared mask (no per-head or per-sequence mask), no LSA block list
    ggml_cuda_pool_alloc<uint8_t> live_flags(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> live_list(ctx.pool());
    if constexpr ((bits_k & 24) != 0 && ncols2 > 1) {
        const ggml_tensor * mask = dst->src[3];
        const int64_t positions = ggml_get_op_params_i32(KQV, 6);
        const int64_t n_tiles   = (positions + nbatch_fa - 1) / nbatch_fa;
        const bool    packed    = mask && mask->type == GGML_TYPE_I32;
        // columns a row holds: f16 entries, or 32 per I32 word
        const int64_t mask_cols = !mask ? 0 : packed ? (mask->nb[1]/4)*32 : mask->ne[0];
        // the kernel reads mask plane 0 for every head (mask + nb33*sequence, no nb32 term), so the flags read plane 0 too
        const bool use = mask && (mask->type == GGML_TYPE_F16 || packed) && mask->ne[3] == 1 &&
                dst->src[0]->ne[3] == 1 && mask->ne[1] >= dst->src[0]->ne[1] && positions > 0 && n_tiles*nbatch_fa <= mask_cols &&
                nbatch_fa % 32 == 0 && ggml_flash_attn_ext_get_lsa_block(KQV) == 0 && !ggml_flash_attn_ext_get_lsa_vk(KQV);
        if (sj_kvarn_live_tiles()) {
            // route proof for end-to-end gates: one line for the first launch that takes the walk, one for the first that cannot
            static std::atomic_flag noted[2] = {ATOMIC_FLAG_INIT, ATOMIC_FLAG_INIT};
            if (!noted[use ? 1 : 0].test_and_set()) {
                GGML_LOG_INFO("%s: SJ-KVaRN trellis live-tile walk %s (mask %s, n_q %d, tiles %d)\n", __func__,
                    use ? "ON" : "OFF: unsupported mask/shape", mask ? ggml_type_name(mask->type) : "none",
                    (int) dst->src[0]->ne[1], (int) n_tiles);
            }
        }
        if (sj_kvarn_live_tiles() && use) {
            live_flags.alloc(n_tiles);
            live_list.alloc(n_tiles + 1);
            constexpr int warps_per_block = 8;
            sj_kvarn_live_tile_flags<<<(n_tiles + warps_per_block - 1)/warps_per_block, warps_per_block*WARP_SIZE, 0, ctx.stream()>>>(
                (const char *) mask->data, (int64_t) mask->nb[1], (int) dst->src[0]->ne[1], packed ? 1 : 0, nbatch_fa, (int) n_tiles,
                live_flags.ptr);
            CUDA_CHECK(cudaGetLastError());
            sj_kvarn_live_tile_list<<<1, 1024, 0, ctx.stream()>>>(live_flags.ptr, (int) n_tiles, live_list.ptr);
            CUDA_CHECK(cudaGetLastError());
            ggml_cuda_fattn_sj_live_tiles = live_list.ptr;
        }
    }

    // K/V stay in their ring storage format; only shared tiles are decoded. stream_k = true.
    launch_fattn<DV, ncols1, ncols2>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, nbatch_fa,
         /*need_f16_K=*/false, /*need_f16_V=*/false, /*stream_k=*/true, /*use_sparse=*/false, warp_size_host);
    ggml_cuda_fattn_sj_live_tiles = nullptr;
}


// [TRPF] Trellis body expand for the prefill path (fattn-sjkvarn-prefill.cu): every sealed record of the visible body is
// decoded ONCE per attention call into the f16 K/V scratch ([head][n_kv][128 half2], row = position), which the plain f16
// MMA kernel then shares across all query tiles. The values come from flash_attn_ext_sj_kvarn_load_tile, the same loader
// (word-load decode, staged codebooks, same half2 affine) the fused low-bit kernel runs per tile, so they are bit-identical
// to what that kernel feeds its MMAs. Rows outside [S, B) are written by fattn_sj_kvarn_expand_prefill (skip_body).
// One CTA = 64 positions (4 strips of 16) of one KV head; strips outside the body are skipped individually.
template<int bits_k, int bits_v>
static __global__ void __launch_bounds__(256) fattn_sj_kvarn_trellis_expand_body_kernel(
        const char * __restrict__ body, const int32_t * __restrict__ desc,
        const size_t k_head_stride, const size_t v_head_stride,
        half2 * __restrict__ K_out, half2 * __restrict__ V_out, const int n_out, const int p_begin, const int head_first) {
    constexpr int D2   = 128;
    constexpr int rows = 16;
    constexpr int per_block = 64;
    __shared__ __align__(16) half2 tile_K[rows*D2];
    __shared__ __align__(16) half2 tile_V[rows*D2];
    const int head = head_first + blockIdx.y;
    const fattn_sj_kvarn_lowbits_ctx kv = fattn_sj_kvarn_lowbits_make_ctx(body, desc, head, 2*D2, bits_k | 8, bits_v, k_head_stride, v_head_stride);
    const int pb = p_begin + blockIdx.x*per_block; // positions [p_begin, p_begin + n_out) -> rows [0, n_out)
    if (pb + per_block <= kv.S || pb >= kv.B) {
        return; // block-uniform: before the staging barrier
    }
    fattn_sj_kvarn_trellis_cb_stage<bits_k, bits_v>();
    const int tid = threadIdx.y*32 + threadIdx.x;
#pragma unroll 1
    for (int p0 = pb; p0 < pb + per_block; p0 += rows) {
        if (p0 < kv.S || p0 >= kv.B) {
            continue; // block-uniform
        }
        const int group = (p0 - kv.S)/kv.G;
        const int token = (p0 - kv.S)%kv.G;
        const char * rec = fattn_sj_kvarn_rec(kv, group);
        flash_attn_ext_sj_kvarn_load_tile<D2, false, rows, 256, D2, false, false>(rec, token, kv, tile_K, rows);
        flash_attn_ext_sj_kvarn_load_tile<D2, false, rows, 256, D2, false, true >(rec, token, kv, tile_V, rows);
        __syncthreads();
        const size_t out_offset = ((size_t) blockIdx.y*n_out + (p0 - p_begin))*D2;
        const uint4 * sk = (const uint4 *) tile_K;
        const uint4 * sv = (const uint4 *) tile_V;
        uint4 * gk = (uint4 *) (K_out + out_offset);
        uint4 * gv = (uint4 *) (V_out + out_offset);
        for (int i = tid; i < rows*D2/4; i += 256) {
            gk[i] = sk[i];
            gv[i] = sv[i];
        }
        __syncthreads();
    }
}

void ggml_cuda_sj_kvarn_trellis_expand_body(const int bits_pair, const char * body, const int32_t * desc,
        const size_t k_head_stride, const size_t v_head_stride, half2 * K_out, half2 * V_out, const int n_out, const int p_begin,
        const int heads, const int head_first, cudaStream_t stream) {
    ggml_cuda_sj_kvarn_trellis_cb_init();
    GGML_ASSERT(n_out % 64 == 0 && p_begin % 64 == 0);
    const dim3 blocks(n_out/64, heads, 1);
    const dim3 threads(32, 8, 1);
    switch (bits_pair) {
        case (3 << 8) | 3: fattn_sj_kvarn_trellis_expand_body_kernel<3,3><<<blocks, threads, 0, stream>>>(body, desc, k_head_stride, v_head_stride, K_out, V_out, n_out, p_begin, head_first); break;
        case (3 << 8) | 2: fattn_sj_kvarn_trellis_expand_body_kernel<3,2><<<blocks, threads, 0, stream>>>(body, desc, k_head_stride, v_head_stride, K_out, V_out, n_out, p_begin, head_first); break;
        case (2 << 8) | 2: fattn_sj_kvarn_trellis_expand_body_kernel<2,2><<<blocks, threads, 0, stream>>>(body, desc, k_head_stride, v_head_stride, K_out, V_out, n_out, p_begin, head_first); break;
        default: GGML_ABORT("SJ-KVaRN trellis expand: unsupported pair %d/%d", bits_pair >> 8, bits_pair & 0xFF);
    }
    CUDA_CHECK(cudaGetLastError());
}

template<int bits_k, int bits_v, bool expand>
static __global__ void __launch_bounds__(256) fattn_sj_kvarn_strip_master_expand_kernel(
        ggml_cuda_qsa_clock_map map, const int32_t * desc, size_t table_records, size_t k_head_stride, size_t v_head_stride,
        uint8_t * scratch, half2 * K_out, half2 * V_out, int n_out, int p_begin, int head_first) {
    constexpr int record_bytes = 4096*(bits_k + bits_v) + 2304;
    const int head = head_first + int(blockIdx.y), pb = p_begin + int(blockIdx.x)*64;
    const int sink = desc[GGML_SJKVARN_DESC_S], end = desc[GGML_SJKVARN_DESC_B];
    if (sink < 0 || sink%128 || end < sink || (end - sink)%128 ||
        desc[GGML_SJKVARN_DESC_D] != 256 || desc[GGML_SJKVARN_DESC_G] != 128 ||
        desc[GGML_SJKVARN_DESC_HKV] != int(map.heads) || desc[GGML_SJKVARN_DESC_RECBYTES] != record_bytes ||
        desc[GGML_SJKVARN_DESC_BODY_TYPE] != GGML_TYPE_I16 ||
        (desc[GGML_SJKVARN_DESC_TABLE] != 0 && desc[GGML_SJKVARN_DESC_TABLE] != 1) || !(fattn_sj_kvarn_trtok & 1)) { __trap(); return; }
    if (pb + 64 <= sink || pb >= end) return;
    const int32_t * table = desc[GGML_SJKVARN_DESC_TABLE] ? desc + GGML_SJKVARN_DESC_N_ENTRIES : nullptr;
    int record;
    if (!sj_kvarn_lowbits::strip_master_expand_location(sink, end, pb, table, table_records, map.records, record)) { __trap(); return; }
    uint8_t * rec = scratch + (size_t(blockIdx.y)*gridDim.x + blockIdx.x)*record_bytes;
    const int tid = threadIdx.y*ggml_cuda_get_physical_warp_size() + threadIdx.x;
    if constexpr (!expand) {
        const int slot = map.table[record];
        if (slot < -1 || (slot >= 0 && uint32_t(slot) >= map.slots)) { __trap(); return; }
        if (slot >= 0) {
            const auto * src = reinterpret_cast<const uint4 *>(map.card + (size_t(slot)*map.heads + head)*record_bytes);
            for (int q = tid; q < record_bytes/16; q += 256) reinterpret_cast<uint4 *>(rec)[q] = src[q];
        } else {
            sj_kvarn_lowbits::strip_master_layout l;
            sj_kvarn_lowbits::make_strip_master_layout(bits_k, bits_v, l);
            const auto * src = reinterpret_cast<const uint4 *>(map.strip_master + (size_t(record)*map.heads + head)*l.bytes);
            for (int q = tid; q < l.bytes/16; q += 256) {
                const uint4 value = src[q];
                const uint8_t * bytes = reinterpret_cast<const uint8_t *>(&value);
                for (int b = 0; b < 16; ++b) {
                    const int from = sj_kvarn_lowbits::strip_master_source_byte(l, q*16 + b);
                    if (from >= 0 && sj_kvarn_lowbits::strip_master_byte(l, from) == q*16 + b) rec[from] = bytes[b];
                }
            }
        }
    } else {
        __shared__ __align__(16) half2 tile_K[16*128], tile_V[16*128];
        // A prior launch wrote the canonical scratch; the old decoder's read-only loads are safe here.
        const auto kv = fattn_sj_kvarn_lowbits_make_ctx(map.card, desc, head, 256, bits_k | 8, bits_v, k_head_stride, v_head_stride);
        fattn_sj_kvarn_trellis_cb_stage<bits_k, bits_v>();
        for (int p0 = pb; p0 < pb + 64; p0 += 16) {
            const int token = (p0 - sink)%128;
            flash_attn_ext_sj_kvarn_load_tile_raw<128, false, 16, 256, 128, false, false>(reinterpret_cast<const char *>(rec), token, kv, tile_K, 16);
            flash_attn_ext_sj_kvarn_load_tile_raw<128, false, 16, 256, 128, false, true >(reinterpret_cast<const char *>(rec), token, kv, tile_V, 16);
            __syncthreads();
            const size_t out = (size_t(blockIdx.y)*n_out + p0 - p_begin)*128;
            for (int i = tid; i < 16*128/4; i += 256) {
                reinterpret_cast<uint4 *>(K_out + out)[i] = reinterpret_cast<const uint4 *>(tile_K)[i];
                reinterpret_cast<uint4 *>(V_out + out)[i] = reinterpret_cast<const uint4 *>(tile_V)[i];
            }
            __syncthreads();
        }
    }
}

bool ggml_cuda_sj_kvarn_strip_master_expand(ggml_backend_cuda_context & ctx, const ggml_cuda_qsa_clock_map & map,
        const ggml_tensor * desc, void * scratch, size_t scratch_capacity, half2 * K_out, half2 * V_out, size_t output_capacity,
        int n_out, int p_begin, int heads, int head_first, size_t k_head_stride, size_t v_head_stride) {
    static const bool enabled = [] { const char * e = getenv("GGML_CUDA_LSA_RESIDENT_STRIP_MASTER"); return e && e[0] == '1' && !e[1]; }();
    sj_kvarn_lowbits::strip_master_layout l;
    sj_kvarn_lowbits::strip_master_expand_plan plan;
    if (!enabled || ggml_cuda_info().devices[ctx.device].warp_size != 32 || !ampere_mma_available(ggml_cuda_info().devices[ctx.device].cc) ||
        !map.strip_master || !map.card || !map.table || map.cells != 128 || map.group != 128 ||
        !map.stage_all || map.stage_n < map.records || !map.slots || map.slots > map.records || map.stride > size_t(INT32_MAX) ||
        !sj_kvarn_lowbits::strip_master_clock_layout(map.strip_master_bits >> 8, map.strip_master_bits & 255,
            map.records, map.heads, map.stride, map.stage_stride, map.strip_master_capacity, l) ||
        map.heads > uint32_t(INT32_MAX/512) || !desc || desc->type != GGML_TYPE_I32 || !ggml_is_contiguous(desc) ||
        desc->ne[0] < GGML_SJKVARN_DESC_N_ENTRIES || desc->ne[0] > INT32_MAX || desc->ne[1] != 1 || desc->ne[2] != 1 || desc->ne[3] != 1 ||
        !desc->data || !scratch || !K_out || !V_out || n_out <= 0 || n_out%64 || p_begin < 0 || p_begin%64 || n_out > INT32_MAX - p_begin ||
        heads <= 0 || heads > 65535 || head_first < 0 || uint32_t(head_first) >= map.heads || uint32_t(heads) > map.heads - uint32_t(head_first) ||
        k_head_stride > size_t(INT32_MAX) || v_head_stride > size_t(INT32_MAX) ||
        !sj_kvarn_lowbits::make_strip_master_expand_plan(map.strip_master_bits >> 8, map.strip_master_bits & 255,
            n_out, heads, SIZE_MAX, plan) || plan.rows != n_out ||
        output_capacity < plan.output_bytes || scratch_capacity < plan.scratch_bytes ||
        reinterpret_cast<uintptr_t>(map.strip_master)%16 || reinterpret_cast<uintptr_t>(map.card)%16 ||
        reinterpret_cast<uintptr_t>(map.table)%alignof(int32_t) || reinterpret_cast<uintptr_t>(desc->data)%alignof(int32_t) ||
        reinterpret_cast<uintptr_t>(scratch)%16 || reinterpret_cast<uintptr_t>(K_out)%16 || reinterpret_cast<uintptr_t>(V_out)%16) return false;
    ggml_cuda_sj_kvarn_trellis_cb_init();
    const dim3 blocks(n_out/64, heads, 1), threads(32, 8, 1);
    const size_t table_records = size_t(desc->ne[0]) - GGML_SJKVARN_DESC_N_ENTRIES;
#define LSA_MASTER_EXPAND(K, V, EXPAND) do { fattn_sj_kvarn_strip_master_expand_kernel<K,V,EXPAND><<<blocks, threads, 0, ctx.stream()>>>(map, static_cast<const int32_t *>(desc->data), table_records, k_head_stride, v_head_stride, static_cast<uint8_t *>(scratch), K_out, V_out, n_out, p_begin, head_first); CUDA_CHECK(cudaGetLastError()); } while (0)
    switch (map.strip_master_bits) {
        case (3 << 8) | 3: LSA_MASTER_EXPAND(3, 3, false); LSA_MASTER_EXPAND(3, 3, true); break;
        case (3 << 8) | 2: LSA_MASTER_EXPAND(3, 2, false); LSA_MASTER_EXPAND(3, 2, true); break;
        case (2 << 8) | 3: LSA_MASTER_EXPAND(2, 3, false); LSA_MASTER_EXPAND(2, 3, true); break;
        case (2 << 8) | 2: LSA_MASTER_EXPAND(2, 2, false); LSA_MASTER_EXPAND(2, 2, true); break;
        default: return false;
    }
#undef LSA_MASTER_EXPAND
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_flash_attn_ext_sj_kvarn_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst);         // fattn-sjkvarn-prefill.cu
bool ggml_cuda_flash_attn_ext_sj_kvarn_prefill_applies(int device, const ggml_tensor * dst);              // fattn-sjkvarn-prefill.cu

// GGML_SJKVARN_NO_DIRECT=1 (same switch as the 4/4 path): every width through the tile loader, so a scalar arm can be
// compared with a trellis arm on one decoder
static bool sj_kvarn_lowbits_no_direct() {
    static const bool v = [] { const char * e = getenv("GGML_SJKVARN_NO_DIRECT"); return e != nullptr && e[0] == '1'; }();
    return v;
}

// GGML_SJKVARN_TRELLIS_WORDS (default on; =0 selects the byte-load kernel): trellis tile loads read the payload as aligned
// 32-bit words instead of two byte loads per code (fattn_sj_kvarn_trellis_lb_word_w, bit-identical values). Kernel build
// bits_k | 8 (byte-load: bits_k | 16); both stage the codebooks in shared memory (#126).
// Measured 2026-10-01: +19.3% (3/3) / +17.8% (3/2) decode at 100K, identical text.
static bool sj_kvarn_trellis_words() {
    static const bool v = [] { const char * e = getenv("GGML_SJKVARN_TRELLIS_WORDS"); return e == nullptr || e[0] != '0'; }();
    return v;
}
// [TRW] GGML_SJKVARN_TRELLIS_WIDTH (default on, Jake 2026-11-14; =0/off/false/no turns it off): 3/3 trellis decode/verify (n_q <= 8) takes the width kernel
// (fattn-sjkvarn-lowbits-stream.cuh, template bits_k | 8) instead of the tile build. Same exclusions as the scalar width
// route (GGML_SJKVARN_NO_DIRECT, LSA full stage incl. the compact strip stage). LSE nodes (LLAMA_LSA_COMP=2 + COMP_LB=1)
// stay on the tile build unless GGML_SJKVARN_TRELLIS_WIDTH_LSE=1 (below): the width launch is the low-bit launch_fattn
// (fattn-sjkvarn-lowbits-common.cuh), which then keeps >= 2 parallel blocks and its combine writes lse = max + log(sum).
static bool sj_kvarn_trellis_width() {
    static const bool v = [] {
        const char * e = getenv("GGML_SJKVARN_TRELLIS_WIDTH");
        if (e == nullptr || *e == 0) { return true; }
        std::string w(e);
        for (char & c : w) { c = (char) tolower((unsigned char) c); }
        return !(w == "0" || w == "off" || w == "false" || w == "no");
    }();
    return v;
}
// opt-in on top of the width route: FA nodes with an LSE output also take it (LSE from the parallel-block combine)
static bool sj_kvarn_trellis_width_lse() {
    static const bool v = [] {
        const char * e = getenv("GGML_SJKVARN_TRELLIS_WIDTH_LSE");
        const bool on = e && e[0] == '1';
        if (on) { GGML_LOG_WARN("%s: SJ-KVaRN trellis width route also takes LSE nodes (combine LSE)\n", __func__); }
        return on;
    }();
    return v;
}
static bool sj_kvarn_trellis_width_applies(const ggml_tensor * dst) {
    const int bits = ggml_get_op_params_i32(dst,5);
    return sj_kvarn_trellis_width() && bits == ((3 << 8) | 3) && ggml_get_op_params_i32(dst,7) == GGML_TYPE_I16 &&
        dst->src[0]->ne[1] <= 8 && !sj_kvarn_lowbits_no_direct() && !ggml_cuda_lsa_page_full_stage(dst->src[5]) &&
        (ggml_flash_attn_ext_get_lse(dst) == nullptr || sj_kvarn_trellis_width_lse());
    // LSE nodes (LLAMA_LSA_COMP=2 + COMP_LB=1) stay on the tile build (its stream-k fixup writes the LSE) unless
    // GGML_SJKVARN_TRELLIS_WIDTH_LSE=1: then the width launch writes it through the parallel-block combine
}
// route proof for the gate: one line per process on the first n_q <= 8 call that takes it, one on the first that does not
static void sj_kvarn_trellis_width_note(const bool taken, const int n_q) {
    static std::atomic_flag done_on = ATOMIC_FLAG_INIT, done_off = ATOMIC_FLAG_INIT;
    if (taken ? !done_on.test_and_set() : !done_off.test_and_set()) {
        GGML_LOG_WARN("%s: SJ-KVaRN trellis width route %s (n_q=%d)\n", __func__, taken ? "ON" : "OFF", n_q);
    }
}

// route proof for end-to-end gates: one line on the first trellis attention call that takes the word-load build
static void sj_kvarn_trellis_words_note() {
    static std::atomic_flag done = ATOMIC_FLAG_INIT;
    if (!done.test_and_set()) {
        GGML_LOG_INFO("%s: SJ-KVaRN trellis attention uses the word-load decode (GGML_SJKVARN_TRELLIS_WORDS=0 disables)\n", __func__);
    }
}

template<int bits_k, int bits_v, bool rot = false>
static void ggml_cuda_lowbits_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if constexpr (bits_k == 4 && bits_v == 4) {
        static_assert(!rot, "the 4/4 trellis4 tile build takes the separate rotation passes");
        // 4/4 reaches this path only as the token-axis trellis4 body (explicit nonzero GGML_SJKVARN_TRELLIS_TOKENS, fattn.cu)
        GGML_ASSERT(ggml_get_op_params_i32(dst,7) == GGML_TYPE_I16 && ggml_sj_kvarn::trellis3::tokens4());
        ggml_cuda_sj_kvarn_trellis_cb_init();
        ggml_cuda_lowbits_prefill_case<256,256,8,8,4,4>(ctx,dst);
    } else if (ggml_get_op_params_i32(dst,7) == GGML_TYPE_I16) {
        // trellis-coded body: the stream kernel has no codebook path, the tile loader decodes every width from the
        // codebooks staged in shared memory (#126). Only 3/3, 3/2 and 2/2 seal a 2/3-bit trellis body (llama-context.cpp).
        if constexpr (bits_k <= 3 && bits_v <= 3) {
            // A5: an LSE node never has >= 128 rows (graph guard); with a strip master the tile kernel would reject it
            GGML_ASSERT(ggml_flash_attn_ext_get_lse(dst) == nullptr || dst->src[0]->ne[1] < 128);
            if constexpr (!rot) {
                // [TRPF] n_q >= 128: expand each record once into f16, then the plain f16 MMA kernel (fattn-sjkvarn-prefill.cu)
                // fork (A5): the expand path writes no LSE output; the graph never asks for one at n_q >= 128
                if (ggml_flash_attn_ext_get_lse(dst) == nullptr && ggml_cuda_flash_attn_ext_sj_kvarn_prefill(ctx, dst)) {
                    return;
                }
            }
            ggml_cuda_sj_kvarn_trellis_cb_init();
            if constexpr (bits_k == 3 && bits_v == 3) {
                if (sj_kvarn_trellis_width()) {
                    const bool taken = sj_kvarn_trellis_width_applies(dst);
                    if (dst->src[0]->ne[1] <= 8) { // prompt batches never take it; an OFF line means an n_q <= 8 call missed it
                        sj_kvarn_trellis_width_note(taken, (int) dst->src[0]->ne[1]);
                    }
                    if (taken) {
                        ggml_cuda_sj_kvarn_lowbits_width<bits_k | 8,bits_v,rot>(ctx,dst);
                        return;
                    }
                }
            }
            if (sj_kvarn_trellis_words()) {
                sj_kvarn_trellis_words_note();
                ggml_cuda_lowbits_prefill_case<256,256,8,8,bits_k | 8,bits_v,rot>(ctx,dst);
            } else if constexpr (!rot) {
                ggml_cuda_lowbits_prefill_case<256,256,8,8,bits_k | 16,bits_v>(ctx,dst);
            } else {
                GGML_ABORT("SJ-KVaRN fused rotation on the byte-load trellis build (the caller takes the separate passes)");
            }
        } else {
            GGML_ABORT("SJ-KVaRN trellis body on a %d/%d pair (trellis needs 3/3, 3/2, 2/2 or the 4/4 trellis4 body)", bits_k, bits_v);
        }
    } else if (dst->src[0]->ne[1] <= 8 && !sj_kvarn_lowbits_no_direct() && !ggml_cuda_lsa_page_full_stage(dst->src[5])) {
        GGML_ASSERT(ggml_flash_attn_ext_get_lse(dst) == nullptr && "SJ-KVaRN scalar low-bit width kernel writes no LSE output");
        ggml_cuda_sj_kvarn_lowbits_width<bits_k,bits_v,rot>(ctx,dst);
    } else {
        ggml_cuda_lowbits_prefill_case<256,256,8,8,bits_k,bits_v,rot>(ctx,dst);
    }
}

void ggml_cuda_flash_attn_sj_kvarn_lowbits(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// [#139] fused Q / output rotation: the tile build (Ampere table: one 128-half2 combine batch at 8x8 columns) and the
// word-load trellis build rotate in-kernel, the width kernel takes a Q pre-pass and rotates its output in-kernel; the
// 4/4 trellis4 body, the byte-load trellis build, logit softcap and other MMA tables take the separate passes around
// this same dispatch.
template<int bits_k, int bits_v>
static void ggml_cuda_lowbits_case_rot(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!ggml_cuda_fattn_sj_kvarn_rot(dst)) {
        ggml_cuda_lowbits_case<bits_k,bits_v>(ctx,dst);
        return;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    float logit_softcap;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    const bool trellis = ggml_get_op_params_i32(dst,7) == GGML_TYPE_I16;
    bool in_kernel = !(bits_k == 4 && bits_v == 4) && logit_softcap == 0.0f &&
        ggml_cuda_fattn_mma_get_nbatch_combine(256, 256, 64, cc) == 128 && ggml_cuda_fattn_sj_kvarn_rot_q_aligned(dst->src[0]) &&
        (!trellis || sj_kvarn_trellis_words());
    if constexpr (bits_k == 4 && bits_v == 4) {
        in_kernel = false;
    }
    if (trellis && ggml_cuda_flash_attn_ext_sj_kvarn_prefill_applies(ggml_cuda_get_device(), dst)) {
        in_kernel = false; // [TRPF] the expand path runs plain f16 attention: separate rotation passes around it
    }
    if (!in_kernel) {
        ggml_cuda_flash_attn_ext_sj_kvarn_rot_unfused(ctx, dst, ggml_cuda_flash_attn_sj_kvarn_lowbits);
        return;
    }
    if constexpr (!(bits_k == 4 && bits_v == 4)) {
        ggml_cuda_lowbits_case<bits_k,bits_v,true>(ctx,dst);
    }
}

// [#175] true when ggml_cuda_lowbits_case sends this node to the tile build (ggml_cuda_lowbits_prefill_case), whose mask
// loader reads the packed I32 mask natively; the width kernel takes the f16 expansion in fattn.cu. Mirrors the branches above.
bool ggml_cuda_flash_attn_sj_kvarn_lowbits_tile_route(const ggml_tensor * dst) {
    const int bits = ggml_get_op_params_i32(dst,5);
    if (bits == ((4 << 8) | 4) || ggml_get_op_params_i32(dst,7) == GGML_TYPE_I16) {
        return !sj_kvarn_trellis_width_applies(dst); // [TRW] the width kernel reads the f16 mask expansion
    }
    return !(dst->src[0]->ne[1] <= 8 && !sj_kvarn_lowbits_no_direct());
}

void ggml_cuda_flash_attn_sj_kvarn_lowbits(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const bool master = ggml_cuda_qsa_clock_strip_master_map(ctx, dst->src[5]) != nullptr;
    if (!sj_kvarn_lowbits::strip_master_prefill_fallback_safe(master, dst->src[0]->ne[1])) {
        // Reject before fused rotation can bypass the expand executor.
        GGML_ASSERT(ggml_cuda_flash_attn_ext_sj_kvarn_prefill_applies(ctx.device, dst));
    }
    switch (ggml_get_op_params_i32(dst,5)) {
        case (4 << 8) | 3: ggml_cuda_lowbits_case_rot<4,3>(ctx,dst); break;
        case (3 << 8) | 3: ggml_cuda_lowbits_case_rot<3,3>(ctx,dst); break;
        case (4 << 8) | 2: ggml_cuda_lowbits_case_rot<4,2>(ctx,dst); break;
        case (2 << 8) | 4: ggml_cuda_lowbits_case_rot<2,4>(ctx,dst); break;
        case (3 << 8) | 2: ggml_cuda_lowbits_case_rot<3,2>(ctx,dst); break;
        case (2 << 8) | 2: ggml_cuda_lowbits_case_rot<2,2>(ctx,dst); break;
        case (4 << 8) | 4: ggml_cuda_lowbits_case_rot<4,4>(ctx,dst); break; // token-axis trellis4 only (explicit nonzero GGML_SJKVARN_TRELLIS_TOKENS)
        default: GGML_ABORT("Unsupported SJ-KVaRN lower-bit pair");
    }
}
