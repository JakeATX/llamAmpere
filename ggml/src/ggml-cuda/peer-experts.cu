// Peer expert tier transport, see peer-experts.cuh (GGML_PEER_EXPERTS, default OFF).

#include "common.cuh"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "peer-experts.cuh"

#include <atomic>
#include <cstring>

// host layout: [abort line][per slot, per dir: doorbell line, status line][data areas, 256-byte aligned]
static constexpr size_t PEER_LINE = 128;

struct ggml_cuda_peer_mbox {
    int      n_slots   = 0;
    size_t   bytes[2]  = { 0, 0 };
    size_t   data_off  = 0;     // offset of the first data area
    size_t   slot_size = 0;     // data bytes per slot (both dirs, aligned)
    size_t   total     = 0;
    uint8_t * host     = nullptr;   // cudaHostAllocPortable | cudaHostAllocMapped
};

static size_t peer_align(size_t n, size_t a) { return (n + a - 1) / a * a; }

static uint32_t * peer_abort_word(const ggml_cuda_peer_mbox * mb) { return (uint32_t *) mb->host; }
static uint32_t * peer_bell(const ggml_cuda_peer_mbox * mb, int slot, int dir) {
    return (uint32_t *) (mb->host + PEER_LINE * (1 + 4 * (size_t) slot + 2 * dir));
}
static uint32_t * peer_status_word(const ggml_cuda_peer_mbox * mb, int slot, int dir) {
    return (uint32_t *) (mb->host + PEER_LINE * (1 + 4 * (size_t) slot + 2 * dir + 1));
}
static float * peer_data(const ggml_cuda_peer_mbox * mb, int slot, int dir) {
    return (float *) (mb->host + mb->data_off + mb->slot_size * slot + (dir ? peer_align(mb->bytes[0], 256) : 0));
}

ggml_cuda_peer_mbox * ggml_backend_cuda_peer_mbox_new(int n_slots, size_t bytes_in, size_t bytes_out) {
    if (n_slots <= 0 || n_slots > 1024) {
        return nullptr;
    }
    auto * mb = new ggml_cuda_peer_mbox;
    mb->n_slots   = n_slots;
    mb->bytes[0]  = bytes_in;
    mb->bytes[1]  = bytes_out;
    mb->data_off  = peer_align(PEER_LINE * (1 + 4 * (size_t) n_slots), 256);
    mb->slot_size = peer_align(bytes_in, 256) + peer_align(bytes_out, 256);
    mb->total     = mb->data_off + mb->slot_size * n_slots;
    void * p = nullptr;
    // portable: every device context may map it; mapped: kernels read and write it over PCIe (no P2P needed)
    if (cudaHostAlloc(&p, mb->total, cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess) {
        (void) cudaGetLastError();
        delete mb;
        return nullptr;
    }
    memset(p, 0, mb->total);
    mb->host = (uint8_t *) p;
    return mb;
}

void ggml_backend_cuda_peer_mbox_free(ggml_cuda_peer_mbox * mb) {
    if (!mb) {
        return;
    }
    if (mb->host) {
        (void) cudaFreeHost(mb->host);
    }
    delete mb;
}

uint32_t ggml_backend_cuda_peer_status(const ggml_cuda_peer_mbox * mb, int slot, int dir) {
    if (!mb || slot < 0 || slot >= mb->n_slots || dir < 0 || dir > 1) {
        return 0;
    }
    return reinterpret_cast<std::atomic<uint32_t> *>(peer_status_word(mb, slot, dir))->load(std::memory_order_acquire);
}

void ggml_backend_cuda_peer_abort(ggml_cuda_peer_mbox * mb, uint32_t value) {
    if (!mb) {
        return;
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    reinterpret_cast<std::atomic<uint32_t> *>(peer_abort_word(mb))->store(value, std::memory_order_release);
}

static __device__ __forceinline__ uint64_t peer_globaltimer_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// one block: every data store must be visible before the doorbell, and one block makes that a __syncthreads
static __global__ void k_peer_post(const float * __restrict__ src, volatile float * dst, int n,
        volatile uint32_t * bell, uint32_t seq) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        dst[i] = src[i];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        *bell = seq;
    }
}

static __global__ void k_peer_wait(const volatile uint32_t * bell, const volatile uint32_t * abort_word, uint32_t seq,
        const volatile float * src, float * __restrict__ dst, int n, volatile uint32_t * status, uint64_t timeout_ns) {
    __shared__ int ok;
    if (threadIdx.x == 0) {
        const uint64_t t0 = peer_globaltimer_ns();
        int good = 1;
        while ((int32_t) (*bell - seq) < 0) {
            if (*abort_word != 0 || peer_globaltimer_ns() - t0 > timeout_ns) {
                good = 0;
                break;
            }
            __nanosleep(128);
        }
        __threadfence_system();
        if (!good) {
            *status = seq;
        }
        ok = good;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        dst[i] = ok ? src[i] : 0.0f;
    }
}

static bool peer_args_ok(ggml_backend_t backend, const ggml_cuda_peer_mbox * mb, int slot, int dir, const ggml_tensor * t) {
    return backend && ggml_backend_is_cuda(backend) && mb && mb->host && slot >= 0 && slot < mb->n_slots &&
        dir >= 0 && dir <= 1 && t && t->data && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_I32) && ggml_is_contiguous(t) &&
        ggml_nbytes(t) <= mb->bytes[dir];
}

template <typename T>
static bool peer_dev_ptr(T * host, T ** dev) {
    void * d = nullptr;
    if (cudaHostGetDevicePointer(&d, (void *) host, 0) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    *dev = (T *) d;
    return true;
}

bool ggml_backend_cuda_peer_post(ggml_backend_t backend, ggml_cuda_peer_mbox * mb, int slot, int dir,
        const ggml_tensor * src, uint32_t seq) {
    if (!peer_args_ok(backend, mb, slot, dir, src)) {
        return false;
    }
    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_set_device(ctx->device);
    float * d_data = nullptr;
    uint32_t * d_bell = nullptr;
    if (!peer_dev_ptr(peer_data(mb, slot, dir), &d_data) || !peer_dev_ptr(peer_bell(mb, slot, dir), &d_bell)) {
        return false;
    }
    k_peer_post<<<1, 1024, 0, ctx->stream()>>>((const float *) src->data, d_data, (int) ggml_nelements(src), d_bell, seq);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_backend_cuda_peer_wait(ggml_backend_t backend, ggml_cuda_peer_mbox * mb, int slot, int dir,
        ggml_tensor * dst, uint32_t seq, uint64_t timeout_ns) {
    if (!peer_args_ok(backend, mb, slot, dir, dst)) {
        return false;
    }
    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_set_device(ctx->device);
    float * d_data = nullptr;
    uint32_t * d_bell = nullptr, * d_abort = nullptr, * d_status = nullptr;
    if (!peer_dev_ptr(peer_data(mb, slot, dir), &d_data) || !peer_dev_ptr(peer_bell(mb, slot, dir), &d_bell) ||
            !peer_dev_ptr(peer_abort_word(mb), &d_abort) || !peer_dev_ptr(peer_status_word(mb, slot, dir), &d_status)) {
        return false;
    }
    k_peer_wait<<<1, 1024, 0, ctx->stream()>>>(d_bell, d_abort, seq, d_data, (float *) dst->data,
            (int) ggml_nelements(dst), d_status, timeout_ns);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
