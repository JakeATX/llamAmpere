// Peer expert tier transport, see peer-experts.cuh (GGML_PEER_EXPERTS, default OFF).

#include "common.cuh"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "peer-experts.cuh"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

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
// in_status (dir 1 posts only): the status word of the slot's dir-0 wait. If that wait failed for this seq, the
// rows are not valid: the failure goes into the dir-1 status word, so the main card falls back instead of
// accumulating rows computed from zeros.
static __global__ void k_peer_post(const float * __restrict__ src, volatile float * dst, int n,
        volatile uint32_t * bell, uint32_t seq, const volatile uint32_t * in_status, volatile uint32_t * out_status) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        dst[i] = src[i];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        if (in_status && *in_status == seq) {
            *out_status = seq;
        }
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
    uint32_t * d_bell = nullptr, * d_in_status = nullptr, * d_out_status = nullptr;
    if (!peer_dev_ptr(peer_data(mb, slot, dir), &d_data) || !peer_dev_ptr(peer_bell(mb, slot, dir), &d_bell)) {
        return false;
    }
    if (dir == 1 && (!peer_dev_ptr(peer_status_word(mb, slot, 0), &d_in_status) ||
            !peer_dev_ptr(peer_status_word(mb, slot, 1), &d_out_status))) {
        return false;
    }
    k_peer_post<<<1, 1024, 0, ctx->stream()>>>((const float *) src->data, d_data, (int) ggml_nelements(src), d_bell, seq,
            d_in_status, d_out_status);
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

// ---- copy hook (see peer-experts.cuh) ----

static constexpr int PEER_LINK_SLOTS = 32;

struct ggml_cuda_peer_link {
    ggml_cuda_peer_mbox *    mb  = nullptr;
    uint32_t                 seq = 0;
    std::vector<cudaEvent_t> done;      // per slot, on the dst device: recorded after the dst stream's wait
    std::vector<uint32_t>    posted;    // per slot: last seq posted (0 = never)
    std::vector<uint32_t>    seen_fail; // per slot: last failed seq already counted
};

static std::atomic<bool>     g_peer_copy_on{false};
static std::atomic<uint64_t> g_peer_copy_fail{0};
static std::mutex            g_peer_copy_mtx;
static size_t                g_peer_copy_max = 1u << 20; // GGML_PEER_EXPERTS_COPY_MAX_KIB (default 1024)
static uint64_t              g_peer_copy_wait_ns = 10ull * 1000 * 1000 * 1000; // GGML_PEER_EXPERTS_WAIT_MS (default 10000)
static ggml_cuda_peer_link   g_peer_links[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_DEVICES];

void ggml_backend_cuda_peer_copy_enable(bool on) {
    std::lock_guard<std::mutex> lock(g_peer_copy_mtx);
    if (on) {
        if (const char * e = getenv("GGML_PEER_EXPERTS_COPY_MAX_KIB"); e && atoi(e) > 0) {
            g_peer_copy_max = (size_t) atoi(e) * 1024;
        }
        if (const char * e = getenv("GGML_PEER_EXPERTS_WAIT_MS"); e && atoi(e) > 0) {
            g_peer_copy_wait_ns = (uint64_t) atoi(e) * 1000 * 1000;
        }
    }
    g_peer_copy_on.store(on, std::memory_order_release);
}

bool ggml_cuda_peer_copy_on(void) {
    return g_peer_copy_on.load(std::memory_order_relaxed);
}

uint64_t ggml_backend_cuda_peer_copy_failures(void) {
    std::lock_guard<std::mutex> lock(g_peer_copy_mtx);
    for (auto & row : g_peer_links) {
        for (auto & l : row) {
            if (!l.mb) {
                continue;
            }
            for (int s = 0; s < PEER_LINK_SLOTS; ++s) {
                const uint32_t st = ggml_backend_cuda_peer_status(l.mb, s, 0);
                if (st != 0 && st != l.seen_fail[s]) {
                    l.seen_fail[s] = st;
                    g_peer_copy_fail++;
                }
            }
        }
    }
    return g_peer_copy_fail.load();
}

bool ggml_cuda_peer_copy(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    if (src->type != dst->type || (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_I32) ||
            !ggml_is_contiguous(src) || !ggml_is_contiguous(dst) || ggml_nbytes(src) != ggml_nbytes(dst) ||
            ggml_nbytes(src) == 0 || src->data == nullptr || dst->data == nullptr) {
        return false;
    }
    auto * ctx_src = static_cast<ggml_backend_cuda_context *>(backend_src->context);
    auto * ctx_dst = static_cast<ggml_backend_cuda_context *>(backend_dst->context);
    const int ds = ctx_src->device, dd = ctx_dst->device;
    if (ds < 0 || dd < 0 || ds >= GGML_CUDA_MAX_DEVICES || dd >= GGML_CUDA_MAX_DEVICES || ds == dd) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_peer_copy_mtx);
    if (!g_peer_copy_on.load(std::memory_order_relaxed) || ggml_nbytes(src) > g_peer_copy_max) {
        return false;
    }
    ggml_cuda_peer_link & l = g_peer_links[ds][dd];
    if (!l.mb) {
        l.mb = ggml_backend_cuda_peer_mbox_new(PEER_LINK_SLOTS, g_peer_copy_max, 0);
        if (!l.mb) {
            return false;
        }
        ggml_cuda_set_device(dd);
        l.done.assign(PEER_LINK_SLOTS, nullptr);
        for (auto & ev : l.done) {
            CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        }
        l.posted.assign(PEER_LINK_SLOTS, 0);
        l.seen_fail.assign(PEER_LINK_SLOTS, 0);
    }
    if (ggml_nbytes(src) > l.mb->bytes[0]) {
        return false; // the limit grew after this ring was made
    }
    // doorbells start at 0 and are compared as (bell - seq) >= 0: seq 0 is never used
    if (++l.seq == 0) {
        ++l.seq;
    }
    const uint32_t seq  = l.seq;
    const int      slot = (int) (seq % PEER_LINK_SLOTS);
    if (l.posted[slot] != 0) {
        // the dst stream must have read the slot's previous rows before the src stream overwrites them
        ggml_cuda_set_device(ds);
        CUDA_CHECK(cudaStreamWaitEvent(ctx_src->stream(), l.done[slot], 0));
    }
    if (!ggml_backend_cuda_peer_post(backend_src, l.mb, slot, 0, src, seq) ||
            !ggml_backend_cuda_peer_wait(backend_dst, l.mb, slot, 0, dst, seq, g_peer_copy_wait_ns)) {
        // cannot happen after the checks above; the copy state is then unknown, so stop using the hook
        g_peer_copy_on.store(false);
        GGML_ABORT("peer copy: mailbox post/wait refused after the argument checks");
    }
    ggml_cuda_set_device(dd);
    CUDA_CHECK(cudaEventRecord(l.done[slot], ctx_dst->stream()));
    l.posted[slot] = seq;
    return true;
}
