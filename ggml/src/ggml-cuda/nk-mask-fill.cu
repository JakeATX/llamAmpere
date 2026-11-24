// NEONKERN: vectorized device-side F16 prefix-mask fill (LLAMA_NK_DEV_MASK=1).
//
// Fills contiguous F16 [n_kv, n_rows] masks directly in device memory:
//   dst[row, j] = (j < clamp(n_visible[row], 0, n_kv)) ? 0.0h (0x0000) : -INFh (0xFC00)
//
// - Vectorized 16-byte (uint4 = 8 halfs) multi-CTA stores when the row pointer is 16-byte aligned,
//   saturating device memory bandwidth (~2 us for 1.6 MB on RTX 3090 vs ~500-750 us host build + PCIe).
// - Inline kernel-arg counts for n_rows <= 32 (zero mapped host ring, zero event sync, zero mutex).
// - Mapped pinned ring fallback for n_rows > 32 (up to 65,535 rows).

#include "nk-mask-fill.cuh"
#include "common.cuh"
#include "ggml-backend-impl.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>

static constexpr int NK_MASK_SMALL_MAX_ROWS = 32;
static constexpr int NK_MASK_MAX_ROWS       = 65535; // CUDA gridDim.y limit

struct nk_prefix_mask_small_args {
    int32_t nv[NK_MASK_SMALL_MAX_ROWS];
};

static std::atomic<int64_t> g_nk_dev_mask_launches{0};

int64_t ggml_cuda_nk_dev_mask_count() {
    return g_nk_dev_mask_launches.load(std::memory_order_relaxed);
}

static bool nk_dev_mask_trace() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_NK_TRACE");
        return e != nullptr && atoi(e) != 0;
    }();
    return on;
}

static __device__ __forceinline__ uint32_t nk_pack_half2_prefix(int64_t j, int64_t nv) {
    const uint32_t lo = (j     < nv) ? 0x0000u : 0xFC00u;
    const uint32_t hi = (j + 1 < nv) ? 0x0000u : 0xFC00u;
    return lo | (hi << 16);
}

static __device__ __forceinline__ void nk_fill_prefix_row_f16(half * __restrict__ r, int64_t n_kv, int64_t c) {
    const int64_t nv = c < 0 ? 0 : (c > n_kv ? n_kv : c);
    const __half_raw keep_raw = { 0x0000 };
    const __half_raw drop_raw = { 0xFC00 };
    const half keep = half(keep_raw);
    const half drop = half(drop_raw);

    const int64_t tid    = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t) gridDim.x  * blockDim.x;

    if ((((uintptr_t) r) & 15u) == 0u) {
        const int64_t n_vec = n_kv >> 3;
        const uint4 v_keep = make_uint4(0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u);
        const uint4 v_drop = make_uint4(0xFC00FC00u, 0xFC00FC00u, 0xFC00FC00u, 0xFC00FC00u);
        uint4 * __restrict__ rv = (uint4 *) r;

        for (int64_t idx = tid; idx < n_vec; idx += stride) {
            const int64_t j0 = idx << 3;
            if (j0 + 8 <= nv) {
                rv[idx] = v_keep;
            } else if (j0 >= nv) {
                rv[idx] = v_drop;
            } else {
                rv[idx] = make_uint4(
                    nk_pack_half2_prefix(j0 + 0, nv),
                    nk_pack_half2_prefix(j0 + 2, nv),
                    nk_pack_half2_prefix(j0 + 4, nv),
                    nk_pack_half2_prefix(j0 + 6, nv));
            }
        }

        if (blockIdx.x == 0) {
            const int64_t tail_j = (n_vec << 3) + threadIdx.x;
            if (tail_j < n_kv) {
                r[tail_j] = tail_j < nv ? keep : drop;
            }
        }
    } else {
        for (int64_t j = tid; j < n_kv; j += stride) {
            r[j] = j < nv ? keep : drop;
        }
    }
}

static __global__ void k_nk_fill_prefix_mask_small(
        half * __restrict__ dst, const nk_prefix_mask_small_args args, int64_t n_kv, int64_t n_rows) {
    const int64_t row = blockIdx.y;
    if (row >= n_rows) {
        return;
    }
    nk_fill_prefix_row_f16(dst + row * n_kv, n_kv, args.nv[row]);
}

static __global__ void k_nk_fill_prefix_mask_ring(
        half * __restrict__ dst, const int32_t * __restrict__ n_visible, int64_t n_kv, int64_t n_rows) {
    const int64_t row = blockIdx.y;
    if (row >= n_rows) {
        return;
    }
    nk_fill_prefix_row_f16(dst + row * n_kv, n_kv, n_visible[row]);
}

bool ggml_backend_cuda_nk_fill_prefix_mask(
        ggml_backend_t backend, ggml_tensor * dst, const int32_t * n_visible, int64_t n_rows, int64_t n_kv) {
    constexpr int    N_SLOTS    = 4;
    constexpr size_t SLOT_BYTES = 256u << 10;

    if (!backend || !ggml_backend_is_cuda(backend) || !dst || !dst->data || !n_visible ||
            dst->type != GGML_TYPE_F16 || !ggml_is_contiguous(dst) || dst->ne[0] != n_kv ||
            n_kv <= 0 || n_rows <= 0 || n_rows > NK_MASK_MAX_ROWS ||
            n_rows != dst->ne[1] * dst->ne[2] * dst->ne[3] ||
            (size_t) n_rows * sizeof(int32_t) > SLOT_BYTES) {
        return false;
    }

    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    if (ctx->scratch_borrowed) {
        return false;
    }

    constexpr int threads = 256;
    const int64_t n_work = n_kv >= 8 ? (n_kv + 7) / 8 : n_kv;
    const unsigned blocks_x = (unsigned) std::min<int64_t>(64, std::max<int64_t>(1, (n_work + threads - 1) / threads));
    const dim3 grid(blocks_x, (unsigned) n_rows);

    static std::once_flag once_log;
    std::call_once(once_log, [] {
        GGML_LOG_INFO("[NK_DEV_MASK] active: vectorized device prefix-mask fill (LLAMA_NK_DEV_MASK=1)\n");
    });

    ggml_cuda_set_device(ctx->device);
    cudaStream_t stream = ctx->stream();

    const bool use_small = n_rows <= NK_MASK_SMALL_MAX_ROWS;
    if (use_small) {
        nk_prefix_mask_small_args args = {};
        for (int64_t i = 0; i < n_rows; ++i) {
            args.nv[i] = n_visible[i];
        }
        k_nk_fill_prefix_mask_small<<<grid, threads, 0, stream>>>((half *) dst->data, args, n_kv, n_rows);
        CUDA_CHECK(cudaGetLastError());
    } else {
        struct ring {
            std::mutex  mu;
            bool        failed = false;
            char      * host   = nullptr;
            char      * dev    = nullptr;
            cudaEvent_t ev[N_SLOTS] = {};
            int         next = 0;
        };
        static ring rings[GGML_CUDA_MAX_DEVICES];
        auto & r = rings[ctx->device];
        std::lock_guard<std::mutex> lock(r.mu);
        if (r.failed) {
            return false;
        }
        if (!r.host) {
            if (cudaHostAlloc((void **) &r.host, N_SLOTS * SLOT_BYTES, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
                    cudaHostGetDevicePointer((void **) &r.dev, r.host, 0) != cudaSuccess) {
                (void) cudaGetLastError();
                r.failed = true;
                GGML_LOG_WARN("%s: no mapped count ring; falling back to host mask fill\n", __func__);
                return false;
            }
            for (auto & e : r.ev) {
                CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
            }
        }
        const int slot = r.next;
        r.next = (r.next + 1) % N_SLOTS;
        CUDA_CHECK(cudaEventSynchronize(r.ev[slot]));
        std::memcpy(r.host + slot * SLOT_BYTES, n_visible, (size_t) n_rows * sizeof(int32_t));
        k_nk_fill_prefix_mask_ring<<<grid, threads, 0, stream>>>((half *) dst->data,
                (const int32_t *) (r.dev + slot * SLOT_BYTES), n_kv, n_rows);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(r.ev[slot], stream));
    }

    const int64_t launch_idx = g_nk_dev_mask_launches.fetch_add(1, std::memory_order_relaxed) + 1;
    if (nk_dev_mask_trace() && launch_idx <= 64) {
        GGML_LOG_INFO("[NK_DEV_MASK] launch#%lld path=%s n_rows=%lld n_kv=%lld grid=(%u,%u) nv0=%d nv_last=%d\n",
                (long long) launch_idx, use_small ? "inline" : "ring",
                (long long) n_rows, (long long) n_kv, blocks_x, (unsigned) n_rows,
                n_visible[0], n_visible[n_rows - 1]);
    }
    return true;
}
