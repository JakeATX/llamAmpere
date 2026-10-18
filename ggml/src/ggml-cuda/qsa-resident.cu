#include "qsa-resident.cuh"
#include "ggml-backend-impl.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct qsa_resident_type {
    ggml_backend_buffer_type buft;
    size_t stride;
    uint32_t slots;
    int device;
    std::string name;
};

struct qsa_resident_tensor {
    ggml_cuda_qsa_resident_map map{};
    ggml_cuda_qsa_resident_map * device_map = nullptr;
};

struct qsa_resident_buffer {
    void * host = nullptr;
    int device;
    std::vector<std::unique_ptr<qsa_resident_tensor>> tensors;
};

static const char * qsa_resident_name(ggml_backend_buffer_type_t buft) {
    return static_cast<qsa_resident_type *>(buft->context)->name.c_str();
}

bool ggml_cuda_qsa_resident_is_buft(ggml_backend_buffer_type_t buft) {
    return buft && buft->iface.get_name == qsa_resident_name;
}

static qsa_resident_tensor * qsa_resident_entry(const ggml_tensor * tensor) {
    while (tensor->view_src) {
        tensor = tensor->view_src;
    }
    return tensor->buffer && ggml_cuda_qsa_resident_is_buft(tensor->buffer->buft) ?
        static_cast<qsa_resident_tensor *>(tensor->extra) : nullptr;
}

static void qsa_resident_sync(ggml_backend_buffer_t buffer) {
    ggml_cuda_set_device(static_cast<qsa_resident_buffer *>(buffer->context)->device);
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void qsa_resident_reset(ggml_backend_buffer_t buffer) {
    qsa_resident_sync(buffer);
    for (const auto & entry : static_cast<qsa_resident_buffer *>(buffer->context)->tensors) {
        CUDA_CHECK(cudaMemset(entry->map.tags, 0xff, entry->map.slots*sizeof(int32_t)));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void qsa_resident_free(ggml_backend_buffer_t buffer) {
    auto * b = static_cast<qsa_resident_buffer *>(buffer->context);
    qsa_resident_sync(buffer);
    for (const auto & e : b->tensors) {
        CUDA_CHECK(cudaFree(e->map.card));
        CUDA_CHECK(cudaFree(e->map.tags));
        CUDA_CHECK(cudaFree(e->device_map));
    }
    if (b->host) CUDA_CHECK(cudaFreeHost(b->host));
    delete b;
}

static void * qsa_resident_base(ggml_backend_buffer_t buffer) {
    return static_cast<qsa_resident_buffer *>(buffer->context)->host;
}

static ggml_status qsa_resident_init(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    if (tensor->view_src) {
        return GGML_STATUS_SUCCESS;
    }
    auto * t = static_cast<qsa_resident_type *>(buffer->buft->context);
    if (tensor->type != GGML_TYPE_I8 || ggml_nbytes(tensor) % t->stride != 0) {
        return GGML_STATUS_FAILED;
    }
    auto entry = std::make_unique<qsa_resident_tensor>();
    auto & m = entry->map;
    void * mapped = nullptr;
    const auto * b = static_cast<qsa_resident_buffer *>(buffer->context);
    CUDA_CHECK(cudaHostGetDevicePointer(&mapped, b->host, 0));
    m.host = static_cast<const char *>(mapped) + (static_cast<const char *>(tensor->data) - static_cast<const char *>(b->host));
    m.stride = t->stride;
    m.slots = std::min<size_t>(t->slots, ggml_nbytes(tensor)/t->stride);
    if (!m.slots) {
        return GGML_STATUS_FAILED;
    }
    ggml_cuda_set_device(t->device);
    cudaError_t error = cudaMalloc(&m.card, m.stride*m.slots);
    if (error == cudaSuccess) error = cudaMalloc(&m.tags, m.slots*sizeof(int32_t));
    if (error == cudaSuccess) error = cudaMalloc(&entry->device_map, sizeof(m));
    if (error != cudaSuccess) {
        (void) cudaGetLastError();
        if (m.card) cudaFree(m.card);
        if (m.tags) cudaFree(m.tags);
        if (entry->device_map) cudaFree(entry->device_map);
        return GGML_STATUS_ALLOC_FAILED;
    }
    CUDA_CHECK(cudaMemset(m.tags, 0xff, m.slots*sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(entry->device_map, &m, sizeof(m), cudaMemcpyHostToDevice));
    tensor->extra = entry.get();
    static_cast<qsa_resident_buffer *>(buffer->context)->tensors.push_back(std::move(entry));
    return GGML_STATUS_SUCCESS;
}

static void qsa_resident_set(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    qsa_resident_reset(buffer);
    memcpy(static_cast<char *>(tensor->data) + offset, data, size);
}

static void qsa_resident_get(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    qsa_resident_sync(buffer);
    memcpy(data, static_cast<const char *>(tensor->data) + offset, size);
}

static void qsa_resident_memset(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    qsa_resident_reset(buffer);
    memset(static_cast<char *>(tensor->data) + offset, value, size);
}

static void qsa_resident_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    qsa_resident_reset(buffer);
    if (buffer->size) memset(qsa_resident_base(buffer), value, buffer->size);
}

static const ggml_backend_buffer_i qsa_resident_buffer_i = {
    qsa_resident_free, qsa_resident_base, qsa_resident_init,
    qsa_resident_memset, qsa_resident_set, qsa_resident_get,
    nullptr, nullptr, nullptr, qsa_resident_clear, nullptr,
};

static ggml_backend_buffer_t qsa_resident_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    auto * t = static_cast<qsa_resident_type *>(buft->context);
    auto b = std::make_unique<qsa_resident_buffer>();
    b->device = t->device;
    ggml_cuda_set_device(t->device);
    if (size == 0) return ggml_backend_buffer_init(buft, qsa_resident_buffer_i, b.release(), size);
    const cudaError_t error = cudaHostAlloc(&b->host, std::max<size_t>(size, 1), cudaHostAllocMapped);
    if (error != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_ERROR("%s: pinned allocation failed: %s\n", __func__, cudaGetErrorString(error));
        return nullptr;
    }
    return ggml_backend_buffer_init(buft, qsa_resident_buffer_i, b.release(), size);
}

static size_t qsa_resident_alignment(ggml_backend_buffer_type_t) { return 256; }

// Report a device buffer so all host access uses the synchronized get/set callbacks.
static bool qsa_resident_host(ggml_backend_buffer_type_t) { return false; }

ggml_backend_buffer_type_t ggml_backend_cuda_qsa_resident_buffer_type(ggml_backend_dev_t dev, size_t stride, uint32_t slots) {
    if (!dev || !stride || stride % 4 || !slots || stride > std::numeric_limits<size_t>::max()/slots) {
        return nullptr;
    }
    int device = -1;
    for (int i = 0; i < ggml_backend_cuda_get_device_count(); ++i) {
        if (ggml_backend_buft_get_device(ggml_backend_cuda_buffer_type(i)) == dev) device = i;
    }
    if (device < 0) return nullptr;
    static std::mutex mutex;
    static std::vector<std::unique_ptr<qsa_resident_type>> types;
    std::lock_guard<std::mutex> lock(mutex);
    for (const auto & t : types) {
        if (t->device == device && t->stride == stride && t->slots == slots) return &t->buft;
    }
    auto t = std::make_unique<qsa_resident_type>();
    t->stride = stride;
    t->slots = slots;
    t->device = device;
    t->name = "CUDA_QSA_Resident_" + std::to_string(device) + "_" + std::to_string(stride) + "_" + std::to_string(slots);
    t->buft = {{qsa_resident_name, qsa_resident_alloc, nullptr, qsa_resident_alignment, nullptr, nullptr, nullptr, qsa_resident_host}, dev, t.get()};
    auto * result = &t->buft;
    types.push_back(std::move(t));
    return result;
}

size_t ggml_backend_cuda_qsa_resident_device_bytes(ggml_backend_buffer_type_t buft, const ggml_tensor * body) {
    const auto * t = static_cast<const qsa_resident_type *>(buft->context);
    const size_t slots = std::min<size_t>(t->slots, ggml_nbytes(body)/t->stride);
    return slots*(t->stride + sizeof(int32_t)) + sizeof(ggml_cuda_qsa_resident_map);
}

const char * ggml_cuda_qsa_resident_body(const ggml_tensor * body) {
    auto * entry = qsa_resident_entry(body);
    // Body records are aligned; the low bit identifies a device-side split map.
    return entry ? reinterpret_cast<const char *>(reinterpret_cast<uintptr_t>(entry->device_map) | 1) :
        static_cast<const char *>(body->data);
}

static __global__ void qsa_resident_invalidate(ggml_cuda_qsa_resident_map m, const int32_t * desc) {
    const int slot = blockIdx.x*blockDim.x + threadIdx.x;
    if (slot >= m.slots) return;
    if (!desc) {
        m.tags[slot] = -1;
        return;
    }
    const int S = desc[GGML_SJKVARN_DESC_S], G = desc[GGML_SJKVARN_DESC_G];
    const int first = (desc[GGML_SJKVARN_DESC_B_OLD] - S)/G;
    const int end = (desc[GGML_SJKVARN_DESC_B] - S)/G;
    const int tag = m.tags[slot];
    for (int g = first; g < end; ++g) {
        const int record = desc[GGML_SJKVARN_DESC_TABLE] ? desc[GGML_SJKVARN_DESC_N_ENTRIES + g] : g;
        if (tag == record) {
            m.tags[slot] = -1;
            return;
        }
    }
}

void ggml_cuda_qsa_resident_invalidate(ggml_backend_cuda_context & ctx, const ggml_tensor * body) {
    auto * e = qsa_resident_entry(body);
    if (!e) return;
    const auto * desc = body->src[2];
    qsa_resident_invalidate<<<(e->map.slots + 127)/128, 128, 0, ctx.stream()>>>(e->map,
        desc ? static_cast<const int32_t *>(desc->data) : nullptr);
    CUDA_CHECK(cudaGetLastError());
}

static __global__ void qsa_resident_fetch(ggml_cuda_qsa_resident_map m, const int32_t * desc,
        const char * mask, size_t mask_pitch, int nqueries, int mask_n) {
    const int slot = blockIdx.x;
    const int slots = m.slots;
    const int S = desc[GGML_SJKVARN_DESC_S];
    const int G = desc[GGML_SJKVARN_DESC_G];
    const int ng = (desc[GGML_SJKVARN_DESC_B] - S)/G;
    int selected = -1;
    // One writer per slot; colliding groups keep using their authoritative RAM records.
    const bool paged = desc[GGML_SJKVARN_DESC_TABLE] != 0;
    const int start = paged ? ng - 1 : slot + (ng > slot ? (ng - 1 - slot)/slots : 0)*slots;
    for (int g = start; g >= 0 && g < ng; g -= paged ? 1 : slots) {
        const int record = paged ? desc[GGML_SJKVARN_DESC_N_ENTRIES + g] : g;
        if (record % slots != slot) continue;
        bool live = false;
        for (int j = threadIdx.x; j < G*nqueries; j += blockDim.x) {
            const int pos = S + g*G + j%G;
            if (pos < mask_n) live |= __half2float(reinterpret_cast<const half *>(mask + (j/G)*mask_pitch)[pos]) > -CUDART_INF_F;
        }
        if (__syncthreads_or(live)) {
            selected = record;
            break;
        }
    }
    if (selected < 0 || m.tags[slot] == selected) return;
    for (size_t i = threadIdx.x*sizeof(uint32_t); i < m.stride; i += blockDim.x*sizeof(uint32_t)) {
        *reinterpret_cast<uint32_t *>(m.card + slot*m.stride + i) =
            *reinterpret_cast<const uint32_t *>(m.host + selected*m.stride + i);
    }
    __syncthreads();
    if (threadIdx.x == 0) m.tags[slot] = selected;
}

void ggml_cuda_qsa_resident_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    if (!dst->src[5] || !dst->src[6]) return;
    auto * e = qsa_resident_entry(dst->src[5]);
    if (!e) return;
    const auto * mask = dst->src[3];
    if (!mask || mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[3] != 1 || dst->src[0]->ne[1] > 8) return;
    qsa_resident_fetch<<<e->map.slots, 128, 0, ctx.stream()>>>(e->map,
        static_cast<const int32_t *>(dst->src[6]->data), static_cast<const char *>(mask->data), mask->nb[1],
        std::min(mask->ne[1], dst->src[0]->ne[1]), mask->ne[0]);
    CUDA_CHECK(cudaGetLastError());
}
