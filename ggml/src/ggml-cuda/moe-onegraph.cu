// fork (ONEGRAPH, backlog #101): GPU-anchored MoE expert-cache lookup. See ggml/include/ggml-moe-onegraph.h.
#include "moe-onegraph.cuh"
#include "ggml-moe-onegraph.h"

#include <cstring>
#include <mutex>
#include <set>

static __device__ __forceinline__ unsigned long long og_ld_volatile_u64(const unsigned long long * p) {
    unsigned long long v;
    asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

static __device__ __forceinline__ int og_ld_volatile_i32(const int * p) {
    int v;
    asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

static __device__ __forceinline__ unsigned long long og_now_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// kernel 1: one block of GGML_MOE_OG_MAX_IDS threads
static __global__ void k_moe_og_publish(const int32_t * __restrict__ ids, int n_ids, int n_used, const int32_t * map_row,
        int n_expert, int n_slots, unsigned long long * ctrl, ggml_moe_og_ring * ring, int layer,
        int32_t * __restrict__ out, unsigned long long timeout_ns) {
    __shared__ unsigned long long s_seq;
    const int t = threadIdx.x;
    const int e = t < n_ids ? ids[t] : -1;
    const bool valid = e >= 0 && e < n_expert;
    const int slot0 = valid ? og_ld_volatile_i32(map_row + e) : 0;
    const int miss  = __syncthreads_or(valid && slot0 >= n_slots);

    if (t == 0) {
        const unsigned long long seq = ctrl[0] + 1;   // one block: no other writer of the counter
        ctrl[0] = seq;
        ctrl[1] = seq;
        s_seq = seq;
        ggml_moe_og_rec * rec = &ring->rec[seq % GGML_MOE_OG_RING];
        *(volatile unsigned long long *) &rec->seq = 0;   // seqlock: a reader that sees 0 or a new value retries
        __threadfence_system();
        rec->layer  = layer;
        rec->n_ids  = n_ids;
        rec->n_used = n_used;
        rec->need   = miss ? 1 : 0;
    }
    __syncthreads();
    const unsigned long long seq = s_seq;
    if (t < n_ids) {
        ring->rec[seq % GGML_MOE_OG_RING].ids[t] = e;
    }
    __threadfence_system();
    __syncthreads();
    if (t == 0) {
        ggml_moe_og_rec * rec = &ring->rec[seq % GGML_MOE_OG_RING];
        *(volatile unsigned long long *) &rec->seq = seq;
        *(volatile unsigned long long *) &ring->last = seq;
        __threadfence_system();
        unsigned long long state = 0;
        if (miss) {
            state = 1;
            const unsigned long long t0 = og_now_ns();
            while (og_ld_volatile_u64((const unsigned long long *) &rec->done) != seq) {
#if __CUDA_ARCH__ >= 700
                __nanosleep(256);
#endif
                if (og_now_ns() - t0 > timeout_ns) {
                    if (*(volatile unsigned long long *) &ring->error == 0) {
                        *(volatile unsigned long long *) &ring->error = seq;
                    }
                    __threadfence_system();
                    state = 2;
                    break;
                }
            }
        }
        ctrl[2] = state;
    }
    if (!miss && t < n_ids) {
        out[t] = slot0;
    }
}

// kernel 2: the answer's copies (all blocks), then the map changes and the slots (block 0). The answer is read once
// per block into shared memory (each read of host memory is a PCIe round trip).
static __global__ void k_moe_og_apply(const int32_t * __restrict__ ids, int n_ids, int32_t * map, int row, int n_expert,
        int n_slots, const unsigned long long * ctrl, const ggml_moe_og_ring * ring, int32_t * __restrict__ out) {
    __shared__ unsigned long long s_copy[GGML_MOE_OG_MAX_COPY*3];
    __shared__ int s_upd[GGML_MOE_OG_MAX_UPD*2];
    __shared__ int s_n[2];
    const unsigned long long state = ctrl[2];
    if (state == 0) {
        return;   // all hits: kernel 1 wrote the slots
    }
    const unsigned long long seq = ctrl[1];
    const ggml_moe_og_rec * rec = &ring->rec[seq % GGML_MOE_OG_RING]; // answered before kernel 1 ended: plain loads
    const int t = threadIdx.x;
    if (t == 0) {
        s_n[0] = state == 1 ? max(0, min(rec->n_copy, GGML_MOE_OG_MAX_COPY)) : 0;
        s_n[1] = state == 1 ? max(0, min(rec->n_upd,  GGML_MOE_OG_MAX_UPD))  : 0;
    }
    __syncthreads();
    const int n_copy = s_n[0], n_upd = s_n[1];
    const unsigned long long * c64 = (const unsigned long long *) rec->copy;
    for (int i = t; i < n_copy*3; i += blockDim.x) {
        s_copy[i] = c64[i];
    }
    if (blockIdx.x == 0) {
        const int * u32 = (const int *) rec->upd;
        for (int i = t; i < n_upd*2; i += blockDim.x) {
            s_upd[i] = u32[i];
        }
    }
    __syncthreads();

    // each copy is split over the blocks, 16 bytes per thread and step when aligned
    const size_t step = (size_t) gridDim.x*blockDim.x;
    const size_t i0   = (size_t) blockIdx.x*blockDim.x + t;
    for (int c = 0; c < n_copy; ++c) {
        const char * src = (const char *) (uintptr_t) s_copy[3*c + 0];
        char       * dst = (char *) (uintptr_t) s_copy[3*c + 1];
        const size_t bytes = (size_t) s_copy[3*c + 2];
        if ((((uintptr_t) src | (uintptr_t) dst | bytes) & 15) == 0) {
            const int4 * s4 = (const int4 *) src;
            int4       * d4 = (int4 *) dst;
            for (size_t i = i0; i < bytes/16; i += step) {
                d4[i] = s4[i];
            }
        } else {
            for (size_t i = i0; i < bytes; i += step) {
                dst[i] = src[i];
            }
        }
    }
    if (blockIdx.x != 0) {
        return;
    }
    int slot = 0;
    if (t < n_ids) {
        const int e = ids[t];
        if (e >= 0 && e < n_expert) {
            const int key = row*n_expert + e;
            slot = map[key];
            for (int u = 0; u < n_upd; ++u) {
                if (s_upd[2*u] == key) {
                    slot = s_upd[2*u + 1];
                }
            }
        }
        if (slot < 0 || slot >= n_slots) {
            slot = 0;   // only after a timeout (the host aborts on ring->error); never an out-of-range slot
        }
    }
    __syncthreads();
    for (int u = t; u < n_upd; u += blockDim.x) {
        map[s_upd[2*u]] = s_upd[2*u + 1];   // distinct keys: the worker writes one change per entry
    }
    if (t < n_ids) {
        out[t] = slot;
    }
}

bool ggml_cuda_moe_og_supported(const ggml_tensor * op) {
    const ggml_tensor * ids = op->src[0];
    const ggml_tensor * map = op->src[1];
    return ids && map && op->src[2] && ids->type == GGML_TYPE_I32 && map->type == GGML_TYPE_I32 &&
        ggml_is_contiguous(ids) && ggml_nelements(ids) <= GGML_MOE_OG_MAX_IDS && op->type == GGML_TYPE_I32;
}

void ggml_cuda_moe_og_slots(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * ids  = dst->src[0];
    const ggml_tensor * map  = dst->src[1];
    ggml_tensor       * ctrl = dst->src[2];
    GGML_ASSERT(ggml_cuda_moe_og_supported(dst));

    const int32_t * p = (const int32_t *) dst->op_params;
    const int row = p[0], layer = p[1], n_slots = p[2], n_used = p[3], timeout_ms = p[4];
    int64_t ring_i;
    memcpy(&ring_i, p + 6, sizeof(ring_i));
    auto * ring = (ggml_moe_og_ring *) (intptr_t) ring_i;

    const int n_ids    = (int) ggml_nelements(ids);
    const int n_expert = (int) map->ne[0];
    const int32_t * map_row = (const int32_t *) ((const char *) map->data + (size_t) row * map->nb[1]);
    const unsigned long long timeout_ns = (unsigned long long) (timeout_ms > 0 ? timeout_ms : 10000) * 1000000ULL;

    int32_t * map_base = (int32_t *) map->data;
    k_moe_og_publish<<<1, GGML_MOE_OG_MAX_IDS, 0, ctx.stream()>>>((const int32_t *) ids->data, n_ids, n_used, map_row, n_expert,
        n_slots, (unsigned long long *) ctrl->data, ring, layer, (int32_t *) dst->data, timeout_ns);
    CUDA_CHECK(cudaGetLastError());
    k_moe_og_apply<<<64, 256, 0, ctx.stream()>>>((const int32_t *) ids->data, n_ids, map_base, row, n_expert, n_slots,
        (const unsigned long long *) ctrl->data, ring, (int32_t *) dst->data);
    CUDA_CHECK(cudaGetLastError());
}

static std::mutex og_pin_mutex;
static std::set<void *> og_pinned; // ranges registered here (others were device-accessible already)

bool ggml_backend_cuda_moe_og_pin(void * ptr, size_t size) {
    int dev = 0, ro = 0, same = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    // already device-accessible at the same address (pinned host buffers)
    cudaPointerAttributes attr = {};
    if (cudaPointerGetAttributes(&attr, ptr) == cudaSuccess && attr.type == cudaMemoryTypeHost && attr.devicePointer == ptr) {
        return true;
    }
    (void) cudaGetLastError();
    if (cudaDeviceGetAttribute(&same, cudaDevAttrCanUseHostPointerForRegisteredMem, dev) != cudaSuccess || !same) {
        (void) cudaGetLastError();
        GGML_LOG_WARN("%s: the device cannot use host pointers of registered memory\n", __func__);
        return false;
    }
    if (cudaDeviceGetAttribute(&ro, cudaDevAttrHostRegisterReadOnlySupported, dev) != cudaSuccess) {
        (void) cudaGetLastError();
        ro = 0;
    }
    // read-only registration covers read-only file mappings; without it only writable memory registers
    const unsigned int flags = cudaHostRegisterPortable | cudaHostRegisterMapped | (ro ? cudaHostRegisterReadOnly : 0);
    const cudaError_t err = ggml_cuda_host_register(ptr, size, flags);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_WARN("%s: cannot register %.1f MiB (read-only registration %s): %s\n", __func__, size/1048576.0,
                ro ? "supported" : "not supported", cudaGetErrorString(err));
        return false;
    }
    void * dptr = nullptr;
    if (cudaHostGetDevicePointer(&dptr, ptr, 0) != cudaSuccess || dptr != ptr) {
        (void) cudaGetLastError();
        ggml_cuda_host_unregister(ptr);
        GGML_LOG_WARN("%s: registered memory is not at the same device address\n", __func__);
        return false;
    }
    std::lock_guard<std::mutex> lock(og_pin_mutex);
    og_pinned.insert(ptr);
    return true;
}

void ggml_backend_cuda_moe_og_unpin(void * ptr) {
    {
        std::lock_guard<std::mutex> lock(og_pin_mutex);
        if (og_pinned.erase(ptr) == 0) {
            return;
        }
    }
    ggml_cuda_host_unregister(ptr);
}
