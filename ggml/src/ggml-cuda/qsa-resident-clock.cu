#include "qsa-resident-clock.cuh"
#include "ggml-backend-impl.h"
#include "../ggml-sjkvarn.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct qsa_clock_type {
    ggml_backend_buffer_type buft;
    size_t stride;
    uint32_t slots;
    uint32_t group;
    uint32_t heads;
    int device;
    std::string name;
};

struct qsa_clock_tensor {
    ggml_cuda_qsa_clock_map map{};
    ggml_cuda_qsa_clock_map * device_map = nullptr;
};

struct qsa_clock_buffer {
    void * host = nullptr;
    int device;
    std::vector<std::unique_ptr<qsa_clock_tensor>> tensors;
};

static const char * qsa_clock_name(ggml_backend_buffer_type_t buft) {
    return static_cast<qsa_clock_type *>(buft->context)->name.c_str();
}

bool ggml_cuda_qsa_clock_is_buft(ggml_backend_buffer_type_t buft) {
    return buft && buft->iface.get_name == qsa_clock_name;
}

static qsa_clock_tensor * qsa_clock_entry(const ggml_tensor * tensor) {
    while (tensor->view_src) {
        tensor = tensor->view_src;
    }
    return tensor->buffer && ggml_cuda_qsa_clock_is_buft(tensor->buffer->buft) ?
        static_cast<qsa_clock_tensor *>(tensor->extra) : nullptr;
}

static void qsa_clock_sync(ggml_backend_buffer_t buffer) {
    ggml_cuda_set_device(static_cast<qsa_clock_buffer *>(buffer->context)->device);
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void qsa_clock_reset(ggml_backend_buffer_t buffer) {
    qsa_clock_sync(buffer);
    for (const auto & entry : static_cast<qsa_clock_buffer *>(buffer->context)->tensors) {
        auto & m = entry->map;
        CUDA_CHECK(cudaMemset(m.tags, 0xff, m.slots*sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(m.table, 0xff, size_t(m.records)*(m.group/4)*sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(m.stamp, 0, m.slots*sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(m.ref, 0, m.slots*sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(m.ctl, 0, 16*sizeof(uint32_t)));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void qsa_clock_free(ggml_backend_buffer_t buffer) {
    auto * b = static_cast<qsa_clock_buffer *>(buffer->context);
    qsa_clock_sync(buffer);
    for (const auto & e : b->tensors) {
        uint64_t counters[8]{};
        CUDA_CHECK(cudaMemcpy(counters, e->map.ctl, sizeof(counters), cudaMemcpyDeviceToHost));
        GGML_LOG_INFO("QSA CLOCK: slots=%u calls=%llu page_misses=%llu selected_cells=%llu last_misses=%u overflow=%u\n",
            e->map.slots, (unsigned long long) counters[6], (unsigned long long) counters[4],
            (unsigned long long) counters[5], uint32_t(counters[1]), uint32_t(counters[2]));
        CUDA_CHECK(cudaFree(e->map.card));
        CUDA_CHECK(cudaFree(e->map.tags));
        CUDA_CHECK(cudaFree(e->map.meta));
        CUDA_CHECK(cudaFree(e->map.table));
        CUDA_CHECK(cudaFree(e->map.stamp));
        CUDA_CHECK(cudaFree(e->map.ref));
        CUDA_CHECK(cudaFree(e->map.miss_page));
        CUDA_CHECK(cudaFree(e->map.miss_slot));
        CUDA_CHECK(cudaFree(e->map.ctl));
        CUDA_CHECK(cudaFree(e->device_map));
    }
    if (b->host) CUDA_CHECK(cudaFreeHost(b->host));
    delete b;
}

static void * qsa_clock_base(ggml_backend_buffer_t buffer) {
    return static_cast<qsa_clock_buffer *>(buffer->context)->host;
}

static ggml_status qsa_clock_init(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    if (tensor->view_src) {
        return GGML_STATUS_SUCCESS;
    }
    auto * t = static_cast<qsa_clock_type *>(buffer->buft->context);
    if (tensor->type != GGML_TYPE_I8 || ggml_nbytes(tensor) % t->stride != 0) {
        return GGML_STATUS_FAILED;
    }
    auto entry = std::make_unique<qsa_clock_tensor>();
    auto & m = entry->map;
    void * mapped = nullptr;
    const auto * b = static_cast<qsa_clock_buffer *>(buffer->context);
    CUDA_CHECK(cudaHostGetDevicePointer(&mapped, b->host, 0));
    m.host = static_cast<const char *>(mapped) + (static_cast<const char *>(tensor->data) - static_cast<const char *>(b->host));
    m.stride = t->stride;
    GGML_ASSERT(ggml_nbytes(tensor)/t->stride <= INT32_MAX/(t->group/4));
    m.records = ggml_nbytes(tensor)/t->stride;
    m.group = t->group;
    m.heads = t->heads;
    m.slots = std::min<size_t>(t->slots, m.records)*(m.group/4);
    if (!m.slots) {
        return GGML_STATUS_FAILED;
    }
    ggml_cuda_set_device(t->device);
    cudaError_t error = cudaMalloc(&m.card, size_t(m.slots)*m.heads*GGML_QSA_PAGE_HEAD_BYTES);
    if (error == cudaSuccess) error = cudaMalloc(&m.meta, size_t(m.records)*m.heads*GGML_QSA_META_HEAD_BYTES);
    if (error == cudaSuccess) error = cudaMalloc(&m.table, size_t(m.records)*(m.group/4)*sizeof(int32_t));
    if (error == cudaSuccess) error = cudaMalloc(&m.tags, m.slots*sizeof(int32_t));
    if (error == cudaSuccess) error = cudaMalloc(&m.stamp, m.slots*sizeof(uint32_t));
    if (error == cudaSuccess) error = cudaMalloc(&m.ref, m.slots*sizeof(int32_t));
    if (error == cudaSuccess) error = cudaMalloc(&m.miss_page, m.slots*sizeof(int32_t));
    if (error == cudaSuccess) error = cudaMalloc(&m.miss_slot, m.slots*sizeof(int32_t));
    if (error == cudaSuccess) error = cudaMalloc(&m.ctl, 16*sizeof(uint32_t));
    if (error == cudaSuccess) error = cudaMalloc(&entry->device_map, sizeof(m));
    if (error != cudaSuccess) {
        (void) cudaGetLastError();
        for (void * ptr : {static_cast<void *>(m.card), static_cast<void *>(m.meta), static_cast<void *>(m.table),
                static_cast<void *>(m.tags), static_cast<void *>(m.stamp), static_cast<void *>(m.ref),
                static_cast<void *>(m.miss_page), static_cast<void *>(m.miss_slot), static_cast<void *>(m.ctl),
                static_cast<void *>(entry->device_map)}) {
            if (ptr) cudaFree(ptr);
        }
        return GGML_STATUS_ALLOC_FAILED;
    }
    CUDA_CHECK(cudaMemset(m.tags, 0xff, m.slots*sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(m.table, 0xff, size_t(m.records)*(m.group/4)*sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(m.stamp, 0, m.slots*sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(m.ref, 0, m.slots*sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(m.ctl, 0, 16*sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(m.meta, 0, size_t(m.records)*m.heads*GGML_QSA_META_HEAD_BYTES));
    CUDA_CHECK(cudaMemcpy(entry->device_map, &m, sizeof(m), cudaMemcpyHostToDevice));
    tensor->extra = entry.get();
    static_cast<qsa_clock_buffer *>(buffer->context)->tensors.push_back(std::move(entry));
    return GGML_STATUS_SUCCESS;
}

static void qsa_clock_written(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, size_t offset, size_t size);

static void qsa_clock_set(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    qsa_clock_sync(buffer);
    memcpy(static_cast<char *>(tensor->data) + offset, data, size);
    qsa_clock_written(buffer, tensor, offset, size);
}

static void qsa_clock_get(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    qsa_clock_sync(buffer);
    memcpy(data, static_cast<const char *>(tensor->data) + offset, size);
}

static void qsa_clock_memset(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    qsa_clock_sync(buffer);
    memset(static_cast<char *>(tensor->data) + offset, value, size);
    qsa_clock_written(buffer, tensor, offset, size);
}

static void qsa_clock_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    qsa_clock_reset(buffer);
    if (buffer->size) memset(qsa_clock_base(buffer), value, buffer->size);
    for (const auto & e : static_cast<qsa_clock_buffer *>(buffer->context)->tensors) {
        CUDA_CHECK(cudaMemset(e->map.meta, value, size_t(e->map.records)*e->map.heads*GGML_QSA_META_HEAD_BYTES));
    }
}

static const ggml_backend_buffer_i qsa_clock_buffer_i = {
    qsa_clock_free, qsa_clock_base, qsa_clock_init,
    qsa_clock_memset, qsa_clock_set, qsa_clock_get,
    nullptr, nullptr, nullptr, qsa_clock_clear, nullptr,
};

static ggml_backend_buffer_t qsa_clock_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    auto * t = static_cast<qsa_clock_type *>(buft->context);
    auto b = std::make_unique<qsa_clock_buffer>();
    b->device = t->device;
    ggml_cuda_set_device(t->device);
    if (size == 0) return ggml_backend_buffer_init(buft, qsa_clock_buffer_i, b.release(), size);
    const cudaError_t error = cudaHostAlloc(&b->host, std::max<size_t>(size, 1), cudaHostAllocMapped);
    if (error != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_ERROR("%s: pinned allocation failed: %s\n", __func__, cudaGetErrorString(error));
        return nullptr;
    }
    return ggml_backend_buffer_init(buft, qsa_clock_buffer_i, b.release(), size);
}

static size_t qsa_clock_alignment(ggml_backend_buffer_type_t) { return 256; }

// Report a device buffer so all host access uses the synchronized get/set callbacks.
static bool qsa_clock_host(ggml_backend_buffer_type_t) { return false; }

ggml_backend_buffer_type_t ggml_backend_cuda_qsa_clock_buffer_type(ggml_backend_dev_t dev, size_t stride, uint32_t slots) {
    const uint32_t group = 128;
    const uint32_t heads = stride/(128*256 + 2*(3*256 + 3*128));
    if (!dev || !stride || stride % 4 || !slots || group != 128 || !heads || stride != size_t(heads)*(128*256 + 2*(3*256 + 3*128)) || stride > std::numeric_limits<size_t>::max()/slots) {
        return nullptr;
    }
    int device = -1;
    for (int i = 0; i < ggml_backend_cuda_get_device_count(); ++i) {
        if (ggml_backend_buft_get_device(ggml_backend_cuda_buffer_type(i)) == dev) device = i;
    }
    if (device < 0) return nullptr;
    static std::mutex mutex;
    static std::vector<std::unique_ptr<qsa_clock_type>> types;
    std::lock_guard<std::mutex> lock(mutex);
    for (const auto & t : types) {
        if (t->device == device && t->stride == stride && t->slots == slots) return &t->buft;
    }
    auto t = std::make_unique<qsa_clock_type>();
    t->stride = stride;
    t->slots = slots;
    t->group = group;
    t->heads = heads;
    t->device = device;
    t->name = "CUDA_QSA_Clock_" + std::to_string(device) + "_" + std::to_string(stride) + "_" + std::to_string(slots);
    GGML_LOG_INFO("QSA CLOCK: selected-page policy enabled, record_groups=%u pages=%llu heads=%u\n", slots,
        (unsigned long long) slots*(group/4), heads);
    t->buft = {{qsa_clock_name, qsa_clock_alloc, nullptr, qsa_clock_alignment, nullptr, nullptr, nullptr, qsa_clock_host}, dev, t.get()};
    auto * result = &t->buft;
    types.push_back(std::move(t));
    return result;
}

size_t ggml_backend_cuda_qsa_clock_device_bytes(ggml_backend_buffer_type_t buft, const ggml_tensor * body) {
    const auto * t = static_cast<const qsa_clock_type *>(buft->context);
    const size_t slots = std::min<size_t>(t->slots, ggml_nbytes(body)/t->stride);
    const size_t pages = slots*(t->group/4);
    const size_t records = ggml_nbytes(body)/t->stride;
    return pages*(t->heads*GGML_QSA_PAGE_HEAD_BYTES + 5*sizeof(int32_t)) +
        records*(t->heads*GGML_QSA_META_HEAD_BYTES + (t->group/4)*sizeof(int32_t)) +
        16*sizeof(uint32_t) + sizeof(ggml_cuda_qsa_clock_map);
}

const char * ggml_cuda_qsa_clock_body(const ggml_tensor * body) {
    auto * entry = qsa_clock_entry(body);
    // Body records are aligned; the low bit identifies a device-side split map.
    return entry ? reinterpret_cast<const char *>(reinterpret_cast<uintptr_t>(entry->device_map) | 3) :
        static_cast<const char *>(body->data);
}

// Repack nibbles, not reconstructed values. Channel scales retain their original fragment order.
static __device__ void qsa_clock_copy_page(ggml_cuda_qsa_clock_map m, int page, int slot, int head) {
    const int record = page/(m.group/4);
    const int first = (page%(m.group/4))*4;
    const char * src = m.host + size_t(record)*m.stride + size_t(head)*(m.stride/m.heads);
    char * dst = m.card + (size_t(slot)*m.heads + head)*GGML_QSA_PAGE_HEAD_BYTES;
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
        const bool is_v = i >= 512;
        const int j = i%512, token = first + j/128, d = 2*(j%128);
        const uint32_t b0 = ggml_sj_kvarn::code_bit(token, d, is_v, 256, 128, 4);
        const uint32_t b1 = ggml_sj_kvarn::code_bit(token, d + 1, is_v, 256, 128, 4);
        const auto * payload = reinterpret_cast<const uint32_t *>(src + (is_v ? 16384 : 0));
        dst[i] = ((payload[b0/32] >> (b0%32)) & 15) | (((payload[b1/32] >> (b1%32)) & 15) << 4);
    }
    for (int i = threadIdx.x; i < 12; i += blockDim.x) {
        const int offset = i < 4 ? 33792 : i < 8 ? 34560 : 34816;
        reinterpret_cast<uint16_t *>(dst + 1024)[i] =
            reinterpret_cast<const uint16_t *>(src + offset)[first + i%4];
    }
}

static __global__ void qsa_clock_refresh_records(ggml_cuda_qsa_clock_map m, const int32_t * desc,
        int first_record, int end_record) {
    int record = first_record + blockIdx.x;
    if (desc) {
        const int S = desc[GGML_SJKVARN_DESC_S], G = desc[GGML_SJKVARN_DESC_G];
        const int g = (desc[GGML_SJKVARN_DESC_B_OLD] - S)/G + blockIdx.x;
        if (g >= (desc[GGML_SJKVARN_DESC_B] - S)/G) return;
        record = desc[GGML_SJKVARN_DESC_TABLE] ? desc[GGML_SJKVARN_DESC_N_ENTRIES + g] : g;
    } else if (record >= end_record) {
        return;
    }
    if (record < 0 || record >= m.records) return;
    const int head = blockIdx.y;
    const char * src = m.host + size_t(record)*m.stride + size_t(head)*(m.stride/m.heads);
    char * meta = m.meta + (size_t(record)*m.heads + head)*GGML_QSA_META_HEAD_BYTES;
    for (int i = threadIdx.x; i < 384; i += blockDim.x) {
        reinterpret_cast<uint32_t *>(meta)[i] =
            reinterpret_cast<const uint32_t *>(src + (i < 256 ? 32768 : 33024))[i];
    }
    for (int p = 0; p < m.group/4; ++p) {
        const int page = record*(m.group/4) + p;
        const int slot = m.table[page];
        if (slot >= 0) qsa_clock_copy_page(m, page, slot, head);
    }
}

static void qsa_clock_written(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, size_t offset, size_t size) {
    auto * e = qsa_clock_entry(tensor);
    if (!e || !size) return;
    const auto * root = tensor;
    while (root->view_src) root = root->view_src;
    const size_t base = static_cast<const char *>(tensor->data) - static_cast<const char *>(root->data);
    const size_t first = (base + offset)/e->map.stride;
    const size_t end = (base + offset + size - 1)/e->map.stride + 1;
    GGML_ASSERT(end <= e->map.records);
    qsa_clock_refresh_records<<<dim3(end - first, e->map.heads), 128>>>(e->map, nullptr, first, end);
    CUDA_CHECK(cudaGetLastError());
    qsa_clock_sync(buffer);
}

void ggml_cuda_qsa_clock_refresh(ggml_backend_cuda_context & ctx, const ggml_tensor * body) {
    auto * e = qsa_clock_entry(body);
    if (!e) return;
    const auto * desc = body->src[2];
    const int max_records = ggml_get_op_params_i32(body, 5);
    GGML_ASSERT(desc && max_records > 0);
    qsa_clock_refresh_records<<<dim3(max_records, e->map.heads), 128, 0, ctx.stream()>>>(
        e->map, static_cast<const int32_t *>(desc->data), 0, e->map.records);
    CUDA_CHECK(cudaGetLastError());
}

static __global__ void qsa_clock_resolve(ggml_cuda_qsa_clock_map m, const int32_t * desc,
        const int32_t * ids, int width, int queries, const char * mask, size_t mask_pitch, int mask_n) {
    __shared__ uint32_t epoch, assigned, hand;
    if (threadIdx.x == 0) {
        epoch = m.ctl[0] + 1;
        m.ctl[0] = epoch;
        m.ctl[2] = 0;
        m.ctl[3] = 1;
        assigned = 0;
        hand = m.ctl[1];
    }
    __syncthreads();
    if (epoch == 0) {
        for (int i = threadIdx.x; i < m.slots; i += blockDim.x) m.stamp[i] = ~0u;
        __syncthreads();
    }
    const int S = desc[GGML_SJKVARN_DESC_S], B = desc[GGML_SJKVARN_DESC_B];
    for (int i = threadIdx.x; i < width*queries; i += blockDim.x) {
        const int p = ids[i];
        if (p < S || p >= B || p >= mask_n ||
                __half2float(reinterpret_cast<const half *>(mask + (i/width)*mask_pitch)[p]) == -INFINITY) continue;
        const int g = (p - S)/m.group;
        const int record = desc[GGML_SJKVARN_DESC_TABLE] ? desc[GGML_SJKVARN_DESC_N_ENTRIES + g] : g;
        const int page = record*(m.group/4) + ((p - S)%m.group)/4;
        if (record < 0 || record >= m.records) { atomicExch(m.ctl + 4, 1u); continue; }
        const int slot = atomicCAS(m.table + page, -1, -2);
        if (slot >= 0) {
            atomicExch(m.stamp + slot, epoch);
            atomicExch(m.ref + slot, 1);
        } else if (slot == -1) {
            const uint32_t miss = atomicAdd(m.ctl + 2, 1u);
            if (miss < m.slots) m.miss_page[miss] = page;
        }
    }
    __syncthreads();
    if (m.ctl[2] > m.slots || m.ctl[4]) {
        if (threadIdx.x == 0) { printf("QSA CLOCK: selection exceeds resident capacity or record bounds\n"); __trap(); }
        return;
    }
    const uint32_t lanes = min(uint32_t(blockDim.x), m.slots);
    uint32_t scanned = 0;
    // Each sweep visits disjoint slots. A second visit consumes the CLOCK reference bit.
    while (assigned < m.ctl[2] && scanned < 3*m.slots + lanes) {
        const uint32_t slot = (hand + scanned + threadIdx.x)%m.slots;
        if (threadIdx.x < lanes && m.stamp[slot] != epoch && atomicExch(m.ref + slot, 0) == 0) {
            const uint32_t item = atomicAdd(&assigned, 1u);
            if (item < m.ctl[2]) {
                const int old = m.tags[slot];
                if (old >= 0) m.table[old] = -1;
                const int page = m.miss_page[item];
                m.tags[slot] = page;
                m.stamp[slot] = epoch;
                m.ref[slot] = 1;
                m.miss_slot[item] = slot;
                m.table[page] = slot;
            }
        }
        scanned += lanes;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (assigned < m.ctl[2]) {
            m.ctl[4] = 1;
            printf("QSA CLOCK: all slots are protected by this selection\n");
            __trap();
        }
        m.ctl[1] = (hand + scanned)%m.slots;
        reinterpret_cast<unsigned long long *>(m.ctl + 8)[0] += m.ctl[2];
        reinterpret_cast<unsigned long long *>(m.ctl + 10)[0] += uint64_t(width)*queries;
        reinterpret_cast<unsigned long long *>(m.ctl + 12)[0] += 1;
    }
}

static __global__ void qsa_clock_copy_misses(ggml_cuda_qsa_clock_map m) {
    for (uint32_t i = blockIdx.x; i < m.ctl[2]; i += gridDim.x) {
        qsa_clock_copy_page(m, m.miss_page[i], m.miss_slot[i], blockIdx.y);
    }
}

void ggml_cuda_qsa_clock_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    if (!dst->src[5] || !dst->src[6]) return;
    auto * e = qsa_clock_entry(dst->src[5]);
    if (!e) return;
    const auto * mask = dst->src[3];
    const auto * ids = dst->src[7];
    if (dst->src[0]->ne[1] > 8 || !ids) {
        // Prefill and non-indexed attention use the existing host-backed expansion path.
        CUDA_CHECK(cudaMemsetAsync(e->map.ctl + 3, 0, sizeof(uint32_t), ctx.stream()));
        return;
    }
    GGML_ASSERT(mask && mask->type == GGML_TYPE_F16 && mask->ne[2] == 1 && mask->ne[3] == 1);
    GGML_ASSERT(ids && ids->type == GGML_TYPE_I32 && ggml_is_contiguous(ids));
    GGML_ASSERT(ids->ne[1] == dst->src[0]->ne[1] && ids->ne[2] == 1 && ids->ne[3] == 1);
    qsa_clock_resolve<<<1, 1024, 0, ctx.stream()>>>(e->map,
        static_cast<const int32_t *>(dst->src[6]->data), static_cast<const int32_t *>(ids->data),
        ids->ne[0], ids->ne[1], static_cast<const char *>(mask->data), mask->nb[1], mask->ne[0]);
    CUDA_CHECK(cudaGetLastError());
    qsa_clock_copy_misses<<<dim3(128, e->map.heads), 128, 0, ctx.stream()>>>(e->map);
    CUDA_CHECK(cudaGetLastError());
}
