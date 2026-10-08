#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-mma-turbo.cuh"
#include "fattn-mma-kvarn.cuh"
#include "fattn-kvarn-rot.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn.cuh"
#include "fattn-prefill-policy.h"
#include "fattn-i8qk.cuh"
#include "convert.cuh"
#include "kv-stream-span-tuner.h"
#include "ledger.cuh"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <unordered_map>
#include <vector>

struct ggml_cuda_kv_stream_resident_cache;

namespace {

constexpr size_t KV_STREAM_NO_REQUEST = std::numeric_limits<size_t>::max();
constexpr uint32_t KV_STREAM_NO_LAYER = std::numeric_limits<uint32_t>::max();
constexpr uint32_t KV_STREAM_COPY_BATCH_PAGES = 32;
constexpr uint32_t KV_STREAM_DECODE_SPAN_PAGES = KV_STREAM_COPY_BATCH_PAGES;

struct kv_stream_graph_request {
    const char * k_data = nullptr;
    const char * v_data = nullptr;
    size_t k_nb1 = 0;
    size_t k_nb2 = 0;
    size_t v_nb1 = 0;
    size_t v_nb2 = 0;
    int64_t n_head_kv = 0;
    int64_t token_begin = 0;
    int64_t token_count = 0;
    size_t k_row_bytes = 0;
    size_t v_row_bytes = 0;
    size_t k_head_bytes = 0;
    size_t k_bytes = 0;
    size_t v_offset = 0;
    size_t v_head_bytes = 0;
    size_t v_bytes = 0;
    uint32_t layer = 0;
    uint32_t slot = 0;
    uint32_t ready_slot = 0;
    bool mutable_tail = false;
    bool eligible = false;
    bool scheduled = false;
    bool consumed = false;
    bool deadline_sample = false;
};

} // namespace

struct ggml_cuda_kv_stream_transfer_ring {
    char * pool_data = nullptr;
    size_t page_bytes = 0;
    char * conversion_data = nullptr;
    size_t conversion_bytes = 0;
    uint32_t capacity_slots = 0;
    uint32_t active_slots = 0;
    uint32_t forced_decode_span_pages = 0;
    uint32_t graph_decode_span_pages = UINT32_MAX;
    cudaStream_t copy_stream = nullptr;
    cudaEvent_t producer_ready = nullptr;
    cudaEvent_t eval_start = nullptr;
    cudaEvent_t eval_end = nullptr;
    cudaEvent_t copy_sample_start = nullptr;
    cudaEvent_t copy_sample_end = nullptr;
    std::vector<cudaEvent_t> ready;
    std::vector<cudaEvent_t> consumed;
    std::vector<uint8_t> slot_used;
    std::vector<size_t> slot_request;
    uint32_t * ready_flags_host = nullptr;
    uint32_t * ready_flags_device = nullptr;
    uint64_t * deadline_counters_host = nullptr;
    uint64_t * deadline_counters_device = nullptr;

    bool graph_active = false;
    bool graph_decode = true;
    ggml_cuda_kv_stream_span_tuner span_tuner;
    uint32_t graph_layer_count = 0;
    uint32_t current_layer = KV_STREAM_NO_LAYER;
    size_t next_request = 0;
    ggml_cuda_kv_stream_resident_cache * graph_resident_cache = nullptr;
    std::vector<kv_stream_graph_request> graph_requests;
    std::unordered_map<const void *, uint32_t> graph_layer_by_k;
    std::unordered_map<const void *, std::vector<size_t>> graph_request_by_k_page;

    uint64_t asynchronous_page_uploads = 0;
    uint64_t host_to_device_copy_commands = 0;
    uint64_t compute_stream_waits = 0;
    uint64_t stage_slot_reuses = 0;
    uint64_t cross_layer_prefetches = 0;
    uint32_t current_occupancy = 0;
    uint32_t ring_peak_occupancy = 0;
    uint32_t current_ring_peak_occupancy = 0;
    uint32_t last_ring_peak_occupancy = 0;
    uint64_t current_epoch_uploads = 0;
    double last_copy_engine_busy_ratio = 0.0;
    uint32_t copy_sample_uploads = 0;
    bool timing_pending = false;
    bool timing_current = false;
    bool last_graph_decode = false;
    bool last_graph_bounded = false;
    bool last_graph_streamed = false;
    bool copy_sample_recorded = false;
};

ggml_cuda_kv_stream_transfer_ring * ggml_cuda_kv_stream_transfer_ring_new(
        void * pool_data, size_t page_bytes, uint32_t stage_slots,
        void * conversion_data, size_t conversion_bytes,
        uint32_t forced_decode_span_pages) {
    if (pool_data == nullptr || page_bytes == 0 || stage_slots == 0 ||
            (conversion_bytes != 0 && conversion_data == nullptr)) {
        return nullptr;
    }

    auto * ring = new ggml_cuda_kv_stream_transfer_ring;
    ring->pool_data = static_cast<char *>(pool_data);
    ring->page_bytes = page_bytes;
    ring->conversion_data = static_cast<char *>(conversion_data);
    ring->conversion_bytes = conversion_bytes;
    ring->capacity_slots = stage_slots;
    ring->active_slots = stage_slots;
    ring->ready.resize(stage_slots, nullptr);
    ring->forced_decode_span_pages = forced_decode_span_pages;
    ring->consumed.resize(stage_slots, nullptr);
    ring->slot_used.resize(stage_slots, 0);
    ring->slot_request.resize(stage_slots, KV_STREAM_NO_REQUEST);

    auto cleanup = [&]() {
        for (cudaEvent_t event : ring->ready) {
            if (event != nullptr) {
                (void) cudaEventDestroy(event);
            }
        }
        for (cudaEvent_t event : ring->consumed) {
            if (event != nullptr) {
                (void) cudaEventDestroy(event);
            }
        }
        if (ring->producer_ready != nullptr) {
            (void) cudaEventDestroy(ring->producer_ready);
        }
        if (ring->eval_start != nullptr) {
            (void) cudaEventDestroy(ring->eval_start);
        }
        if (ring->eval_end != nullptr) {
            (void) cudaEventDestroy(ring->eval_end);
        }
        if (ring->copy_sample_start != nullptr) {
            (void) cudaEventDestroy(ring->copy_sample_start);
        }
        if (ring->copy_sample_end != nullptr) {
            (void) cudaEventDestroy(ring->copy_sample_end);
        }
        if (ring->copy_stream != nullptr) {
            (void) cudaStreamDestroy(ring->copy_stream);
        }
        if (ring->ready_flags_host != nullptr) {
            (void) cudaFreeHost(ring->ready_flags_host);
        }
        if (ring->deadline_counters_host != nullptr) {
            (void) cudaFreeHost(ring->deadline_counters_host);
        }
        delete ring;
    };

    if (cudaHostAlloc(reinterpret_cast<void **>(&ring->ready_flags_host),
            stage_slots*sizeof(uint32_t), cudaHostAllocMapped) != cudaSuccess ||
        cudaHostGetDevicePointer(reinterpret_cast<void **>(&ring->ready_flags_device),
            ring->ready_flags_host, 0) != cudaSuccess ||
        cudaHostAlloc(reinterpret_cast<void **>(&ring->deadline_counters_host),
            2*sizeof(uint64_t), cudaHostAllocMapped) != cudaSuccess ||
        cudaHostGetDevicePointer(reinterpret_cast<void **>(&ring->deadline_counters_device),
            ring->deadline_counters_host, 0) != cudaSuccess) {
        (void) cudaGetLastError();
        cleanup();
        return nullptr;
    }
    std::fill_n(ring->ready_flags_host, stage_slots, 0u);
    std::fill_n(ring->deadline_counters_host, 2, uint64_t(0));

    if (cudaStreamCreateWithFlags(&ring->copy_stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&ring->producer_ready, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreate(&ring->eval_start) != cudaSuccess ||
        cudaEventCreate(&ring->eval_end) != cudaSuccess ||
        cudaEventCreate(&ring->copy_sample_start) != cudaSuccess ||
        cudaEventCreate(&ring->copy_sample_end) != cudaSuccess) {
        (void) cudaGetLastError();
        cleanup();
        return nullptr;
    }
    for (uint32_t slot = 0; slot < stage_slots; ++slot) {
        if (cudaEventCreateWithFlags(&ring->ready[slot], cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&ring->consumed[slot], cudaEventDisableTiming) != cudaSuccess) {
            (void) cudaGetLastError();
            cleanup();
            return nullptr;
        }
    }
    return ring;
}

void ggml_cuda_kv_stream_transfer_ring_free(ggml_cuda_kv_stream_transfer_ring * ring) {
    if (ring == nullptr) {
        return;
    }
    CUDA_CHECK(cudaStreamSynchronize(ring->copy_stream));
    for (cudaEvent_t event : ring->ready) {
        CUDA_CHECK(cudaEventDestroy(event));
    }
    for (cudaEvent_t event : ring->consumed) {
        CUDA_CHECK(cudaEventDestroy(event));
    }
    CUDA_CHECK(cudaEventDestroy(ring->producer_ready));
    CUDA_CHECK(cudaEventDestroy(ring->eval_start));
    CUDA_CHECK(cudaEventDestroy(ring->eval_end));
    CUDA_CHECK(cudaEventDestroy(ring->copy_sample_start));
    CUDA_CHECK(cudaEventDestroy(ring->copy_sample_end));
    CUDA_CHECK(cudaStreamDestroy(ring->copy_stream));
    CUDA_CHECK(cudaFreeHost(ring->ready_flags_host));
    CUDA_CHECK(cudaFreeHost(ring->deadline_counters_host));
    delete ring;
}

bool ggml_cuda_kv_stream_transfer_ring_set_active_slots(
        ggml_cuda_kv_stream_transfer_ring * ring, uint32_t stage_slots) {
    if (ring == nullptr || stage_slots == 0 || stage_slots > ring->capacity_slots) {
        return false;
    }
    if (ring->active_slots != stage_slots) {
        ring->span_tuner.reset();
        ring->timing_pending = false;
        ring->timing_current = false;
    }
    ring->active_slots = stage_slots;
    return true;
}

void ggml_cuda_kv_stream_transfer_ring_set_conversion_data(
        ggml_cuda_kv_stream_transfer_ring * ring, void * conversion_data) {
    if (ring != nullptr) {
        ring->conversion_data = static_cast<char *>(conversion_data);
    }
}

void ggml_cuda_kv_stream_transfer_ring_reset_span_tuner(
        ggml_cuda_kv_stream_transfer_ring * ring) {
    if (ring != nullptr) {
        ring->span_tuner.reset();
        ring->timing_pending = false;
        ring->timing_current = false;
    }
}

bool ggml_cuda_kv_stream_transfer_ring_observe_decode_latency(
        ggml_cuda_kv_stream_transfer_ring * ring, double elapsed_ms) {
    if (ring == nullptr) {
        return false;
    }
    if (ring->forced_decode_span_pages != 0) {
        return true;
    }
    const bool was_selected = ring->span_tuner.selected();
    ring->span_tuner.observe(
        elapsed_ms, ring->last_graph_decode && ring->last_graph_streamed,
        ring->last_graph_bounded);
    if (!was_selected && ring->span_tuner.selected()) {
        GGML_LOG_WARN(
            "%s: selected %s decode spans from end-to-end latency "
            "(unbounded %.3f ms, %u-page %.3f ms)\n",
            __func__, ring->span_tuner.use_bounded() ? "bounded" : "unbounded",
            ring->span_tuner.unbounded_average_ms(), KV_STREAM_DECODE_SPAN_PAGES,
            ring->span_tuner.bounded_average_ms());
    }
    return true;
}

ggml_cuda_kv_stream_transfer_stats ggml_cuda_kv_stream_transfer_ring_get_stats(
        const ggml_cuda_kv_stream_transfer_ring * ring) {
    if (ring == nullptr) {
        return {};
    }
    return {
        ring->asynchronous_page_uploads,
        ring->host_to_device_copy_commands,
        ring->compute_stream_waits,
        ring->stage_slot_reuses,
        ring->cross_layer_prefetches,
        ring->deadline_counters_host[0],
        ring->deadline_counters_host[1],
        ring->ring_peak_occupancy,
    };
}

struct ggml_cuda_kv_stream_resident_cache {
    char * pool_data = nullptr;
    size_t pool_bytes = 0;
    size_t scratch_bytes = 0;
    size_t page_bytes = 0;
    uint32_t layer_count = 0;
    uint32_t page_tokens = 0;
    uint32_t resident_pages_per_layer = 0;
    uint32_t decode_active_pages = 0;
    uint32_t next_layer = 0;
    std::vector<uint32_t> layer_pages;
    std::vector<size_t> layer_offsets;
    std::unordered_map<const void *, uint32_t> layer_by_k;
    std::unordered_map<const void *, uint32_t> layer_by_data;
    std::unordered_map<const void *, void *> mirror_by_data;
    std::vector<uint8_t> loaded;
    std::vector<uint8_t> dirty;
    std::vector<uint8_t> precise_dirty_tracking;
    std::vector<int64_t> dirty_rows;
    std::vector<uint32_t> mutable_pages;
    bool all_pages_mutable = false;
    ggml_cuda_kv_stream_resident_stats stats;
};

static bool kv_stream_resident_cache_layout(
        const ggml_cuda_kv_stream_resident_cache * cache,
        size_t pool_bytes,
        size_t scratch_bytes,
        uint32_t decode_active_pages,
        uint32_t & resident_pages_per_layer,
        std::vector<uint32_t> & layer_pages,
        std::vector<size_t> & layer_offsets) {
    if (cache == nullptr || scratch_bytes > pool_bytes ||
            scratch_bytes%cache->page_bytes != 0) {
        return false;
    }
    const size_t resident_pages_total =
        (pool_bytes - scratch_bytes)/cache->page_bytes;
    const size_t pages_per_layer = resident_pages_total/cache->layer_count;
    if (pages_per_layer > UINT32_MAX ||
            pages_per_layer > std::numeric_limits<size_t>::max()/cache->layer_count) {
        return false;
    }

    resident_pages_per_layer = uint32_t(pages_per_layer);
    const size_t controlled_resident_pages = pages_per_layer*cache->layer_count;
    layer_pages.assign(cache->layer_count, resident_pages_per_layer);
    if (decode_active_pages > resident_pages_per_layer) {
        const uint64_t total_active_pages =
            uint64_t(decode_active_pages)*cache->layer_count;
        if (total_active_pages < controlled_resident_pages) {
            return false;
        }
        const uint64_t streamed_pages =
            total_active_pages - controlled_resident_pages;
        const uint64_t ring_pages = scratch_bytes/cache->page_bytes;
        if (ring_pages == 0 || streamed_pages == 0) {
            return false;
        }

        // Reduce the number of split-attention layers while each streamed
        // working set fits in both the shared ring and the active pages owned
        // by one layer. When even one split per model layer exceeds the ring,
        // that layer streams in multiple waves. Spread split layers across
        // model order so resident layers provide prefetch windows.
        const auto ceil_div = [](uint64_t numerator, uint64_t denominator) {
            return numerator/denominator + (numerator%denominator != 0);
        };
        const uint64_t splits_for_ring = ceil_div(streamed_pages, ring_pages);
        const uint64_t splits_for_layer_capacity =
            ceil_div(streamed_pages, decode_active_pages);
        const uint64_t ring_bounded_splits =
            std::min<uint64_t>(cache->layer_count, splits_for_ring);
        const uint64_t split_layers_wide =
            std::max(ring_bounded_splits, splits_for_layer_capacity);
        if (split_layers_wide == 0 || split_layers_wide > cache->layer_count) {
            return false;
        }
        const uint32_t split_layers = uint32_t(split_layers_wide);
        const uint64_t base_streamed = streamed_pages/split_layers;
        const uint64_t remainder = streamed_pages%split_layers;
        layer_pages.assign(cache->layer_count, decode_active_pages);
        for (uint32_t split = 0; split < split_layers; ++split) {
            const uint32_t layer = uint32_t(
                uint64_t(split)*cache->layer_count/split_layers);
            const uint64_t layer_streamed = base_streamed + (split < remainder ? 1 : 0);
            if (layer_streamed > decode_active_pages) {
                return false;
            }
            layer_pages[layer] = decode_active_pages - uint32_t(layer_streamed);
        }
    }

    layer_offsets.assign(size_t(cache->layer_count) + 1, 0);
    for (uint32_t layer = 0; layer < cache->layer_count; ++layer) {
        layer_offsets[layer + 1] = layer_offsets[layer] + layer_pages[layer];
    }
    return layer_offsets.back() == controlled_resident_pages;
}

static size_t kv_stream_resident_index(
        const ggml_cuda_kv_stream_resident_cache * cache,
        uint32_t layer,
        uint32_t page) {
    GGML_ASSERT(layer < cache->layer_count && page < cache->layer_pages[layer]);
    return cache->layer_offsets[layer] + page;
}

ggml_cuda_kv_stream_resident_cache * ggml_cuda_kv_stream_resident_cache_new(
        void * pool_data, size_t pool_bytes, size_t scratch_bytes, size_t page_bytes,
        uint32_t layer_count, uint32_t page_tokens) {
    if (pool_data == nullptr || scratch_bytes == 0 || scratch_bytes >= pool_bytes ||
            page_bytes == 0 || scratch_bytes%page_bytes != 0 ||
            layer_count == 0 || page_tokens != 256) {
        return nullptr;
    }

    const size_t resident_pages = (pool_bytes - scratch_bytes)/(page_bytes*layer_count);
    if (resident_pages == 0 || resident_pages > UINT32_MAX) {
        return nullptr;
    }

    auto * cache = new ggml_cuda_kv_stream_resident_cache;
    cache->pool_data = static_cast<char *>(pool_data);
    cache->pool_bytes = pool_bytes;
    cache->scratch_bytes = scratch_bytes;
    cache->page_bytes = page_bytes;
    cache->layer_count = layer_count;
    cache->page_tokens = page_tokens;
    cache->resident_pages_per_layer = resident_pages;
    cache->layer_pages.assign(layer_count, uint32_t(resident_pages));
    cache->layer_offsets.resize(size_t(layer_count) + 1);
    for (uint32_t layer = 0; layer <= layer_count; ++layer) {
        cache->layer_offsets[layer] = size_t(layer)*resident_pages;
    }
    cache->loaded.resize(cache->layer_offsets.back(), 0);
    cache->dirty.resize(cache->layer_offsets.back(), 0);
    cache->precise_dirty_tracking.resize(layer_count, 0);
    return cache;
}

void ggml_cuda_kv_stream_resident_cache_free(ggml_cuda_kv_stream_resident_cache * cache) {
    delete cache;
}

void ggml_cuda_kv_stream_resident_cache_reset(ggml_cuda_kv_stream_resident_cache * cache) {
    if (cache == nullptr) {
        return;
    }
    std::fill(cache->loaded.begin(), cache->loaded.end(), 0);
    std::fill(cache->dirty.begin(), cache->dirty.end(), 0);
    std::fill(cache->precise_dirty_tracking.begin(), cache->precise_dirty_tracking.end(), 0);
    cache->dirty_rows.clear();
    cache->mutable_pages.clear();
    cache->all_pages_mutable = false;
    cache->layer_by_k.clear();
    cache->layer_by_data.clear();
    cache->mirror_by_data.clear();
    cache->next_layer = 0;
    cache->stats = {};
}

bool ggml_cuda_kv_stream_resident_cache_reshape(
        ggml_cuda_kv_stream_resident_cache * cache,
        size_t pool_bytes,
        size_t scratch_bytes,
        uint32_t active_pages_per_layer) {
    if (cache == nullptr) {
        return false;
    }

    // KV_STREAM_KEEP in any position leaves that dimension untouched, so one
    // entry point covers a pool resize, a scratch repartition, a decode layout
    // switch, and any combination of them.
    if (pool_bytes == KV_STREAM_KEEP_BYTES) {
        pool_bytes = cache->pool_bytes;
    }
    if (scratch_bytes == KV_STREAM_KEEP_BYTES) {
        scratch_bytes = cache->scratch_bytes;
    }
    if (active_pages_per_layer == KV_STREAM_KEEP_PAGES) {
        active_pages_per_layer = cache->decode_active_pages;
    }

    uint32_t pages_per_layer = 0;
    std::vector<uint32_t> layer_pages;
    std::vector<size_t> layer_offsets;
    if (!kv_stream_resident_cache_layout(
            cache, pool_bytes, scratch_bytes, active_pages_per_layer,
            pages_per_layer, layer_pages, layer_offsets)) {
        return false;
    }

    const bool pool_changed = cache->pool_bytes != pool_bytes;
    const bool scratch_changed = cache->scratch_bytes != scratch_bytes;
    const bool layout_changed = cache->layer_pages != layer_pages;
    cache->pool_bytes = pool_bytes;
    cache->decode_active_pages = active_pages_per_layer;
    cache->resident_pages_per_layer = pages_per_layer;
    if (!scratch_changed && !layout_changed) {
        return true;
    }

    // Publish the scratch boundary and the layout together. Both change
    // physical K/V addresses, so one invalidation is required; applying them
    // separately would discard and reload the same resident working set twice.
    // Resident storage uses separate contiguous K and V planes per layer, so a
    // layout change moves both the layer base and the V-plane boundary and
    // page-sized byte slots cannot be migrated. Reload lazily from the
    // authoritative host cache instead.
    cache->scratch_bytes = scratch_bytes;
    cache->layer_pages = std::move(layer_pages);
    cache->layer_offsets = std::move(layer_offsets);
    cache->loaded.assign(cache->layer_offsets.back(), 0);
    cache->dirty.assign(cache->layer_offsets.back(), 0);
    cache->precise_dirty_tracking.assign(cache->layer_count, 0);
    cache->dirty_rows.clear();
    cache->mutable_pages.clear();
    cache->all_pages_mutable = false;
    cache->layer_by_k.clear();
    cache->layer_by_data.clear();
    cache->mirror_by_data.clear();
    cache->next_layer = 0;
    if (pool_changed || scratch_changed) {
        cache->stats = {};
    }
    return true;
}

uint32_t ggml_cuda_kv_stream_resident_cache_pages_per_layer(
        const ggml_cuda_kv_stream_resident_cache * cache) {
    return cache == nullptr ? 0 : cache->resident_pages_per_layer;
}

uint32_t ggml_cuda_kv_stream_resident_cache_decode_active_pages(
        const ggml_cuda_kv_stream_resident_cache * cache) {
    return cache == nullptr ? 0 : cache->decode_active_pages;
}

ggml_cuda_kv_stream_resident_stats ggml_cuda_kv_stream_resident_cache_get_stats(
        const ggml_cuda_kv_stream_resident_cache * cache) {
    return cache == nullptr ? ggml_cuda_kv_stream_resident_stats{} : cache->stats;
}

bool ggml_cuda_kv_stream_resident_cache_mark_dirty_rows(
        ggml_cuda_kv_stream_resident_cache * cache,
        const int64_t * rows, size_t count) {
    if (cache == nullptr || (rows == nullptr && count != 0)) {
        return false;
    }

    cache->mutable_pages.clear();
    cache->all_pages_mutable = false;
    if (count == 0) {
        cache->dirty_rows.clear();
    } else {
        cache->dirty_rows.assign(rows, rows + count);
    }
    std::fill(cache->dirty.begin(), cache->dirty.end(), 0);
    for (uint32_t layer = 0; layer < cache->layer_count; ++layer) {
        cache->precise_dirty_tracking[layer] = 1;
    }
    for (size_t i = 0; i < count; ++i) {
        if (rows[i] < 0) {
            cache->all_pages_mutable = true;
            std::fill(cache->dirty.begin(), cache->dirty.end(), 1);
            return true;
        }
        const uint32_t page = uint32_t(uint64_t(rows[i])/cache->page_tokens);
        if (std::find(cache->mutable_pages.begin(), cache->mutable_pages.end(), page) ==
                cache->mutable_pages.end()) {
            cache->mutable_pages.push_back(page);
        }
        for (uint32_t layer = 0; layer < cache->layer_count; ++layer) {
            if (page < cache->layer_pages[layer]) {
                cache->dirty[kv_stream_resident_index(cache, layer, page)] = 1;
            }
        }
    }
    return true;
}

bool ggml_cuda_kv_stream_resident_cache_all_layers_fit(
        const ggml_cuda_kv_stream_resident_cache * cache,
        uint32_t active_pages) {
    if (cache == nullptr || active_pages == 0) {
        return false;
    }
    return std::all_of(cache->layer_pages.begin(), cache->layer_pages.end(),
        [active_pages](uint32_t pages) { return pages >= active_pages; });
}

bool ggml_cuda_kv_stream_resident_cache_get_mirror(
        ggml_cuda_kv_stream_resident_cache * cache,
        const ggml_tensor * target,
        void ** data) {
    if (cache == nullptr || target == nullptr || data == nullptr ||
            cache->dirty_rows.empty()) {
        return false;
    }
    const auto layer_it = cache->layer_by_data.find(target->data);
    const auto mirror_it = cache->mirror_by_data.find(target->data);
    if (layer_it == cache->layer_by_data.end() || mirror_it == cache->mirror_by_data.end()) {
        return false;
    }
    const uint64_t capacity = uint64_t(cache->layer_pages[layer_it->second])*cache->page_tokens;
    for (const int64_t row : cache->dirty_rows) {
        if (row < 0 || uint64_t(row) >= capacity) {
            return false;
        }
    }
    *data = mirror_it->second;
    return true;
}

void ggml_cuda_kv_stream_resident_cache_mark_mirrored(
        ggml_cuda_kv_stream_resident_cache * cache,
        const ggml_tensor * target) {
    if (cache == nullptr || target == nullptr) {
        return;
    }
    const auto layer_it = cache->layer_by_data.find(target->data);
    if (layer_it == cache->layer_by_data.end()) {
        return;
    }
    const uint32_t layer = layer_it->second;
    for (const int64_t row : cache->dirty_rows) {
        if (row < 0) {
            return;
        }
        const uint32_t page = uint32_t(uint64_t(row)/cache->page_tokens);
        if (page < cache->layer_pages[layer]) {
            cache->dirty[kv_stream_resident_index(cache, layer, page)] = 0;
        }
    }
}

void ggml_cuda_kv_stream_resident_cache_mark_dirty(
        ggml_cuda_kv_stream_resident_cache * cache,
        const ggml_tensor * target, const ggml_tensor * indices) {
    if (cache == nullptr || target == nullptr || indices == nullptr) {
        return;
    }
    const auto layer_it = cache->layer_by_data.find(target->data);
    if (layer_it == cache->layer_by_data.end()) {
        return;
    }
    const uint32_t layer = layer_it->second;
    GGML_ASSERT(layer < cache->layer_count);
    if (indices->buffer == nullptr || !ggml_backend_buffer_is_host(indices->buffer) ||
            !ggml_is_contiguous(indices) ||
            (indices->type != GGML_TYPE_I32 && indices->type != GGML_TYPE_I64)) {
        return;
    }

    cache->precise_dirty_tracking[layer] = 1;
    const size_t begin = cache->layer_offsets[layer];
    const size_t end = cache->layer_offsets[layer + 1];
    const size_t count = ggml_nelements(indices);
    for (size_t i = 0; i < count; ++i) {
        const int64_t row = indices->type == GGML_TYPE_I32 ?
            static_cast<const int32_t *>(indices->data)[i] :
            static_cast<const int64_t *>(indices->data)[i];
        if (row < 0) {
            std::fill(cache->dirty.begin() + begin, cache->dirty.begin() + end, 1);
            return;
        }
        const uint64_t page = uint64_t(row)/cache->page_tokens;
        if (page < cache->layer_pages[layer]) {
            cache->dirty[begin + page] = 1;
        }
    }
}

static uint32_t kv_stream_resident_layer(
        ggml_cuda_kv_stream_resident_cache * cache, const void * k_key) {
    GGML_ASSERT(cache != nullptr);
    auto [it, inserted] = cache->layer_by_k.emplace(k_key, cache->next_layer);
    if (inserted) {
        GGML_ASSERT(cache->next_layer < cache->layer_count);
        ++cache->next_layer;
    }
    return it->second;
}

static bool kv_stream_page_mutable(
        const ggml_cuda_kv_stream_resident_cache * cache,
        uint32_t page) {
    return cache->all_pages_mutable ||
        std::find(cache->mutable_pages.begin(), cache->mutable_pages.end(), page) !=
            cache->mutable_pages.end();
}

namespace {

constexpr int KV_STREAM_HEAD_DIM = 256;
constexpr int KV_STREAM_MAX_PARTS_PER_CHUNK = 16;
constexpr int KV_STREAM_QUERY_WORKSPACE_TOKENS = 256;

static int kv_stream_parts_per_chunk() {
    static const int parts = []() {
        const char * value = getenv("GGML_CUDA_KV_STREAM_PARTS");
        const int parsed = value == nullptr ? 16 : atoi(value);
        return parsed == 2 || parsed == 4 || parsed == 8 || parsed == 16 ? parsed : 16;
    }();
    return parts;
}

static int64_t kv_stream_block_tokens(const ggml_tensor * dst, size_t stage_bytes) {
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const size_t k_row_bytes = ggml_row_size(K->type, K->ne[0]);
    const size_t v_row_bytes = ggml_row_size(V->type, V->ne[0]);
    const size_t bytes_per_token = k_row_bytes*K->ne[2] + v_row_bytes*V->ne[2];
    if (bytes_per_token == 0) {
        return 0;
    }

    int64_t tokens = std::min<int64_t>(K->ne[1], stage_bytes/bytes_per_token);
    tokens = tokens/FATTN_KQ_STRIDE*FATTN_KQ_STRIDE;
    while (tokens > 0) {
        const size_t k_bytes = k_row_bytes*tokens*K->ne[2];
        const size_t v_offset = GGML_PAD(k_bytes, 128);
        const size_t v_bytes = v_row_bytes*tokens*V->ne[2];
        if (v_offset <= stage_bytes && v_bytes <= stage_bytes - v_offset) {
            return tokens;
        }
        tokens -= FATTN_KQ_STRIDE;
    }
    return 0;
}

template<int D>
static __global__ void kv_stream_accumulate_chunk_results(
        const float * parts,
        const float2 * meta,
        float * accumulator,
        float2 * accumulator_meta,
        int nrows,
        bool initialize,
        int nparts) {
    ggml_cuda_pdl_lc();
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= nrows || tid >= D) {
        return;
    }
    ggml_cuda_pdl_sync();

    __shared__ float old_maximum;
    __shared__ float maximum;
    __shared__ float old_scale;
    __shared__ float denominator;

    const int base = row*nparts;
    if (tid == 0) {
        old_maximum = initialize ? -FLT_MAX : accumulator_meta[row].x;
        maximum = old_maximum;
        for (int part = 0; part < nparts; ++part) {
            maximum = fmaxf(maximum, meta[base + part].x);
        }

        old_scale = initialize ? 0.0f : expf(old_maximum - maximum);
        denominator = initialize ? 0.0f : old_scale*accumulator_meta[row].y;
        for (int part = 0; part < nparts; ++part) {
            const float weight = expf(meta[base + part].x - maximum);
            denominator += weight*meta[base + part].y;
        }
        accumulator_meta[row] = make_float2(maximum, denominator);
    }
    __syncthreads();

    float numerator = initialize ? 0.0f : old_scale*accumulator[row*D + tid];
    for (int part = 0; part < nparts; ++part) {
        const float weight = expf(meta[base + part].x - maximum);
        numerator += weight*parts[(base + part)*D + tid];
    }
    accumulator[row*D + tid] = numerator;
}

template<int D>
static __global__ void kv_stream_normalize_chunk_results(
        const float * accumulator,
        const float2 * accumulator_meta,
        float * dst,
        int nrows) {
    ggml_cuda_pdl_lc();
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= nrows || tid >= D) {
        return;
    }
    ggml_cuda_pdl_sync();

    // A row whose mask is entirely -inf (e.g. a padding row filled in just to
    // round the grid out to a uniform block/row count, never written back to
    // the real output) has a zero softmax normalizer here. Guard the divide
    // so that stays a harmless zero rather than a silent NaN/Inf that could
    // propagate if such a row's output were ever read by mistake.
    const float denom = accumulator_meta[row].y;
    dst[row*D + tid] = denom > 0.0f ? accumulator[row*D + tid]/denom : 0.0f;
}

#ifdef GGML_CUDA_FA_ALL_QUANTS
using kv_stream_native_partial_fn = void (*)(
    ggml_backend_cuda_context &, ggml_tensor *, float *, float2 *, int);

template<ggml_type type_K>
static kv_stream_native_partial_fn kv_stream_resolve_native_partial_for_v(ggml_type type_v) {
    // TurboQuant cross-type native kernels are only instantiated for a
    // deliberately narrow set of pairings (turbo K/V combined with {f16,
    // q8_0} and with each other) - see fattn-vec.cuh's EXTERN_DECL_FATTN_VEC_CASE
    // blocks for turbo2/3/4. Every other case below must be compile-time
    // excluded (if constexpr, not a runtime check) whenever it isn't one of
    // those instantiated pairs, or taking the function's address would be an
    // undefined reference at link time for a pairing nobody built.
    constexpr bool k_is_turbo =
        type_K == GGML_TYPE_TURBO2_0 || type_K == GGML_TYPE_TURBO3_0 || type_K == GGML_TYPE_TURBO4_0;
    switch (type_v) {
        case GGML_TYPE_F16:
            return &ggml_cuda_flash_attn_ext_vec_partial_case<
                KV_STREAM_HEAD_DIM, type_K, GGML_TYPE_F16>;
        case GGML_TYPE_Q4_0:
            if constexpr (!k_is_turbo) {
                return &ggml_cuda_flash_attn_ext_vec_partial_case<
                    KV_STREAM_HEAD_DIM, type_K, GGML_TYPE_Q4_0>;
            }
            return nullptr;
        case GGML_TYPE_Q4_1:
            if constexpr (!k_is_turbo) {
                return &ggml_cuda_flash_attn_ext_vec_partial_case<
                    KV_STREAM_HEAD_DIM, type_K, GGML_TYPE_Q4_1>;
            }
            return nullptr;
        case GGML_TYPE_Q5_0:
            if constexpr (!k_is_turbo) {
                return &ggml_cuda_flash_attn_ext_vec_partial_case<
                    KV_STREAM_HEAD_DIM, type_K, GGML_TYPE_Q5_0>;
            }
            return nullptr;
        case GGML_TYPE_Q5_1:
            if constexpr (!k_is_turbo) {
                return &ggml_cuda_flash_attn_ext_vec_partial_case<
                    KV_STREAM_HEAD_DIM, type_K, GGML_TYPE_Q5_1>;
            }
            return nullptr;
        case GGML_TYPE_Q8_0:
            return &ggml_cuda_flash_attn_ext_vec_partial_case<
                KV_STREAM_HEAD_DIM, type_K, GGML_TYPE_Q8_0>;
        case GGML_TYPE_BF16:
            if constexpr (!k_is_turbo) {
                return &ggml_cuda_flash_attn_ext_vec_partial_case<
                    KV_STREAM_HEAD_DIM, type_K, GGML_TYPE_BF16>;
            }
            return nullptr;
        case GGML_TYPE_TURBO2_0:
            if constexpr (type_K == GGML_TYPE_F16 || type_K == GGML_TYPE_Q8_0 || k_is_turbo) {
                return &ggml_cuda_flash_attn_ext_vec_partial_case<
                    KV_STREAM_HEAD_DIM, type_K, GGML_TYPE_TURBO2_0>;
            }
            return nullptr;
        case GGML_TYPE_TURBO3_0:
            if constexpr (type_K == GGML_TYPE_F16 || type_K == GGML_TYPE_Q8_0 || k_is_turbo) {
                return &ggml_cuda_flash_attn_ext_vec_partial_case<
                    KV_STREAM_HEAD_DIM, type_K, GGML_TYPE_TURBO3_0>;
            }
            return nullptr;
        case GGML_TYPE_TURBO4_0:
            if constexpr (type_K == GGML_TYPE_F16 || type_K == GGML_TYPE_Q8_0 || k_is_turbo) {
                return &ggml_cuda_flash_attn_ext_vec_partial_case<
                    KV_STREAM_HEAD_DIM, type_K, GGML_TYPE_TURBO4_0>;
            }
            return nullptr;
        default:
            return nullptr;
    }
}

static kv_stream_native_partial_fn kv_stream_resolve_native_partial(
        ggml_type type_k, ggml_type type_v) {
#define KV_STREAM_NATIVE_K_CASE(type_K) \
        case type_K: return kv_stream_resolve_native_partial_for_v<type_K>(type_v)
    switch (type_k) {
        KV_STREAM_NATIVE_K_CASE(GGML_TYPE_F16);
        KV_STREAM_NATIVE_K_CASE(GGML_TYPE_Q4_0);
        KV_STREAM_NATIVE_K_CASE(GGML_TYPE_Q4_1);
        KV_STREAM_NATIVE_K_CASE(GGML_TYPE_Q5_0);
        KV_STREAM_NATIVE_K_CASE(GGML_TYPE_Q5_1);
        KV_STREAM_NATIVE_K_CASE(GGML_TYPE_Q8_0);
        KV_STREAM_NATIVE_K_CASE(GGML_TYPE_BF16);
        KV_STREAM_NATIVE_K_CASE(GGML_TYPE_TURBO2_0);
        KV_STREAM_NATIVE_K_CASE(GGML_TYPE_TURBO3_0);
        KV_STREAM_NATIVE_K_CASE(GGML_TYPE_TURBO4_0);
        default: return nullptr;
    }
#undef KV_STREAM_NATIVE_K_CASE
}
#endif // GGML_CUDA_FA_ALL_QUANTS

} // namespace

struct ggml_backend_cuda_kv_stream_type_capabilities
ggml_backend_cuda_kv_stream_get_type_capabilities(ggml_type type) {
    ggml_backend_cuda_kv_stream_type_capabilities result{};

    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        // TurboQuant: WHT-rotated, per-group nearest-centroid quantized K/V
        // types. Real backing storage (storage=true falls out below, since
        // they aren't the Q8_1/Q8_K auxiliary types), writable directly from
        // F32 via set_rows_cuda_turbo{2,3,4} (set-rows.cu), and dequantizable
        // to F16 via ggml_get_to_fp16_cuda (convert.cu) - so they qualify for
        // ATTENTION_F16 mode below whenever direct_attention isn't available
        // for a given pair (see the direct_attention switch below - a native
        // kernel exists but is only instantiated for a subset of type pairs).
        // The F16 round-trip is correct as-is regardless: Q is unconditionally
        // pre-rotated at the graph level whenever K/V is a turbo type
        // (llama-graph.cpp), and turbo's dequantize functions deliberately
        // do not invert that rotation, so <dequantized-to-F16 K, pre-rotated
        // Q> is exactly right with no extra work.
        case GGML_TYPE_TURBO2_0:
        case GGML_TYPE_TURBO3_0:
        case GGML_TYPE_TURBO4_0:
            result.classified = true;
            break;
        default:
            return result;
    }

    result.storage = type != GGML_TYPE_Q8_1 && type != GGML_TYPE_Q8_K;

    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_TURBO2_0:
        case GGML_TYPE_TURBO3_0:
        case GGML_TYPE_TURBO4_0:
            result.online_write = true;
            break;
        default:
            break;
    }

    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_TURBO2_0:
        case GGML_TYPE_TURBO3_0:
        case GGML_TYPE_TURBO4_0:
            result.decode_f16 = true;
            break;
        default:
            break;
    }

    switch (type) {
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        // TurboQuant: a native decode-inline attention kernel already exists
        // for turbo K/V (fattn-vec.cuh, plus a fused MMA path in
        // fattn-mma-f16.cuh) and is used unconditionally by the ordinary
        // non-streaming FA dispatch - the comment above the `classified`
        // switch previously claiming no such kernel exists was stale. Only
        // instantiated for turbo paired with {f16, q8_0, turbo2, turbo3,
        // turbo4} (see kv_stream_resolve_native_partial_for_v below, which
        // gates on exactly this set); other pairings fall back to
        // ATTENTION_F16 via the storage/online_write/decode_f16 capabilities
        // set above.
        case GGML_TYPE_TURBO2_0:
        case GGML_TYPE_TURBO3_0:
        case GGML_TYPE_TURBO4_0:
            result.direct_attention = true;
            break;
        default:
            break;
    }

    result.requires_initialization =
        type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ2_S;
    result.requires_importance_matrix =
        type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS ||
        type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M;
    result.auxiliary = type == GGML_TYPE_Q8_1 || type == GGML_TYPE_Q8_K;
    return result;
}

#ifdef GGML_CUDA_FA_ALL_QUANTS
// The original direct_attention type set (F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0)
// is a fully-connected cross product - every pair among them has a native
// kernel instantiated. TurboQuant breaks that assumption: turbo only pairs
// natively with {f16, q8_0, turbo2, turbo3, turbo4} (see
// kv_stream_resolve_native_partial_for_v above), not with Q4_0/Q4_1/Q5_0/
// Q5_1/BF16. direct_attention is a per-type capability flag, so without this
// explicit pairwise check, get_attention_mode would wrongly select DIRECT
// for e.g. (Q4_0, TURBO2_0) - a pair with no instantiated kernel - and hit
// the GGML_ASSERT in ggml_cuda_flash_attn_ext_streamed that guards against
// exactly that.
static bool kv_stream_direct_attention_pair_supported(ggml_type type_k, ggml_type type_v) {
    const auto is_turbo = [](ggml_type t) {
        return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0;
    };
    if (!is_turbo(type_k) && !is_turbo(type_v)) {
        return true;
    }
    const auto turbo_compatible = [&](ggml_type t) {
        return t == GGML_TYPE_F16 || t == GGML_TYPE_Q8_0 || is_turbo(t);
    };
    return turbo_compatible(type_k) && turbo_compatible(type_v);
}
#endif // GGML_CUDA_FA_ALL_QUANTS

ggml_backend_cuda_kv_stream_attention_mode
ggml_backend_cuda_kv_stream_get_attention_mode(ggml_type type_k, ggml_type type_v) {
    const auto capabilities_k = ggml_backend_cuda_kv_stream_get_type_capabilities(type_k);
    const auto capabilities_v = ggml_backend_cuda_kv_stream_get_type_capabilities(type_v);
#ifdef GGML_CUDA_FA_ALL_QUANTS
    if (capabilities_k.direct_attention && capabilities_v.direct_attention &&
            kv_stream_direct_attention_pair_supported(type_k, type_v)) {
        return GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_DIRECT;
    }
#endif // GGML_CUDA_FA_ALL_QUANTS
    if (capabilities_k.storage && capabilities_v.storage &&
            capabilities_k.online_write && capabilities_v.online_write &&
            capabilities_k.decode_f16 && capabilities_v.decode_f16 &&
            !capabilities_k.requires_importance_matrix && !capabilities_v.requires_importance_matrix) {
        return GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_F16;
    }
    return GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_UNSUPPORTED;
}

// TurboQuant K/V storage zero-pads non-128-aligned head dims to the next
// multiple of 128 (see llama-kv-cache.cpp) so the WHT rotation always sees
// full 128-wide groups. Streaming page/workspace geometry must size rows
// off that padded width, not the raw model head_dim, or pages allocated
// here would be too small for what llama-kv-cache.cpp actually writes.
static inline uint32_t ggml_cuda_kv_stream_padded_head_dim(ggml_type type, uint32_t head_dim) {
    const bool is_turbo =
        type == GGML_TYPE_TURBO2_0 || type == GGML_TYPE_TURBO3_0 || type == GGML_TYPE_TURBO4_0;
    if (!is_turbo) {
        return head_dim;
    }
    return ((head_dim + 127) / 128) * 128;
}

// NOTE: an earlier version of this function special-cased head_dim_v == 0 to
// mean "no separate V storage" for MLA-shaped (has_v == false) layers, on
// the assumption that such layers' attention op never needs a real V
// region. That assumption is wrong: DSV4's compressed-cache attention (see
// models/deepseek4.cpp, build_attn_mha called with k_all passed as BOTH K
// and V) and plain MLA's graph-reconstructed V (llama-graph.cpp) both feed
// a real, non-null V tensor into the streamed FLASH_ATTN_EXT op regardless
// of whether the persistent cache stores V separately - reserving zero V
// bytes in the page corrupted those reads. Confirmed empirically: streaming
// vs non-streaming KL-divergence on a real DeepSeek-V4 model showed a real,
// arena-size-independent mismatch (mean KLD 0.07, max 4.3) that vanished
// once this was reverted. Determining the *correct* V-region size for each
// K-only architecture (it's not simply head_dim_k, and not simply the raw
// model head_dim_v either - see llama_hparams::n_embd_head_v_mla() vs plain
// n_embd_head_v()) needs dedicated per-architecture investigation before
// any such special-casing is reintroduced.
bool ggml_cuda_kv_stream_page_bytes(
        ggml_type type_k, ggml_type type_v,
        uint32_t head_dim_k, uint32_t head_dim_v, uint32_t head_count,
        uint32_t page_tokens, size_t * page_bytes) {
    if (head_dim_k == 0 || head_dim_v == 0 || head_count == 0 || page_tokens == 0 || page_bytes == nullptr) {
        return false;
    }

    const auto capabilities_k = ggml_backend_cuda_kv_stream_get_type_capabilities(type_k);
    const auto capabilities_v = ggml_backend_cuda_kv_stream_get_type_capabilities(type_v);
    if (!capabilities_k.storage || !capabilities_v.storage) {
        return false;
    }

    const uint32_t stored_head_dim_k = ggml_cuda_kv_stream_padded_head_dim(type_k, head_dim_k);
    const uint32_t stored_head_dim_v = ggml_cuda_kv_stream_padded_head_dim(type_v, head_dim_v);

    const uint32_t block_size_k = ggml_blck_size(type_k);
    const uint32_t block_size_v = ggml_blck_size(type_v);
    if (block_size_k == 0 || block_size_v == 0 ||
            stored_head_dim_k % block_size_k != 0 || stored_head_dim_v % block_size_v != 0) {
        return false;
    }

    const size_t maximum = std::numeric_limits<size_t>::max();
    const size_t row_bytes_k = ggml_row_size(type_k, stored_head_dim_k);
    const size_t row_bytes_v = ggml_row_size(type_v, stored_head_dim_v);
    if (row_bytes_k > maximum/head_count || row_bytes_v > maximum/head_count) {
        return false;
    }
    const size_t token_bytes_k = row_bytes_k*head_count;
    const size_t token_bytes_v = row_bytes_v*head_count;
    if (token_bytes_k > maximum/page_tokens || token_bytes_v > maximum/page_tokens) {
        return false;
    }
    const size_t page_bytes_k = token_bytes_k*page_tokens;
    const size_t page_bytes_v = token_bytes_v*page_tokens;
    if (page_bytes_k > maximum - 127) {
        return false;
    }
    const size_t page_offset_v = (page_bytes_k + 127) & ~size_t(127);
    if (page_bytes_v > maximum - page_offset_v) {
        return false;
    }

    *page_bytes = page_offset_v + page_bytes_v;
    return true;
}

bool ggml_cuda_kv_stream_workspace_bytes(
        ggml_type type_k, ggml_type type_v,
        uint32_t head_dim_k, uint32_t head_dim_v, uint32_t head_count,
        uint32_t page_tokens, size_t * workspace_bytes) {
    if (workspace_bytes == nullptr) {
        return false;
    }
    const auto mode =
        ggml_backend_cuda_kv_stream_get_attention_mode(type_k, type_v);
    if (mode == GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_DIRECT) {
        *workspace_bytes = 0;
        return true;
    }
    if (mode != GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_F16) {
        return false;
    }
    // The F16 conversion workspace must match the row width Q is padded/rotated
    // to at the graph level, which for turbo K/V is the padded (not raw) head
    // dim - see ggml_cuda_kv_stream_padded_head_dim().
    const uint32_t stored_head_dim_k = ggml_cuda_kv_stream_padded_head_dim(type_k, head_dim_k);
    const uint32_t stored_head_dim_v = ggml_cuda_kv_stream_padded_head_dim(type_v, head_dim_v);
    return ggml_cuda_kv_stream_page_bytes(
        GGML_TYPE_F16, GGML_TYPE_F16,
        stored_head_dim_k, stored_head_dim_v, head_count, page_tokens, workspace_bytes);
}

// ggml_cuda_flash_attn_ext_streamed's chunked partial-attention reduction
// (parts/meta/accumulator/accumulator_meta below) allocates its scratch
// buffers from the device's general ctx.pool() - a separate CUDA memory
// pool from the phase arena, not carved out of it. A caller that sizes the
// arena to exactly fill remaining VRAM can construct successfully and still
// hit a hard, unrecoverable abort the first time this pool needs to grow
// mid-request (see docs/kv-stream.md's "Sizing the arena"). This returns
// the worst case those four allocations can reach, so a caller can confirm
// that much *additional* free VRAM survives reserving the arena, and refuse
// construction up front instead.
bool ggml_cuda_kv_stream_transient_workspace_bytes(
        uint32_t n_head_q, uint32_t head_dim_v, uint32_t n_ubatch, size_t * transient_bytes) {
    if (n_head_q == 0 || head_dim_v == 0 || n_ubatch == 0 || transient_bytes == nullptr) {
        return false;
    }

    // parts/meta size off max(KV_STREAM_MAX_PARTS_PER_CHUNK*KV_STREAM_QUERY_WORKSPACE_TOKENS, n_ubatch)
    // query-rows worth of per-head partial state - the first term bounds the
    // non-MMA chunked-reduction path (partial_count <= 16, workspace_queries
    // <= 256 regardless of n_ubatch), the second bounds the MMA-prefill path
    // (partial_count == 1, workspace_queries == the real ubatch token count,
    // unclamped). accumulator/accumulator_meta are always exactly one
    // n_ubatch-worth of the real destination tensor.
    const uint64_t max_parts_workspace_rows =
        uint64_t(KV_STREAM_MAX_PARTS_PER_CHUNK)*uint64_t(KV_STREAM_QUERY_WORKSPACE_TOKENS);
    const uint64_t worst_case_rows = std::max<uint64_t>(max_parts_workspace_rows, n_ubatch);

    const size_t maximum = std::numeric_limits<size_t>::max();
    if (worst_case_rows > maximum/n_head_q) {
        return false;
    }
    const uint64_t worst_case_row_heads = worst_case_rows*n_head_q;
    if (worst_case_row_heads > maximum/head_dim_v || worst_case_row_heads > maximum/sizeof(float2)) {
        return false;
    }
    const size_t parts_bytes = size_t(worst_case_row_heads)*head_dim_v*sizeof(float);
    const size_t meta_bytes  = size_t(worst_case_row_heads)*sizeof(float2);

    if (uint64_t(n_ubatch) > maximum/n_head_q) {
        return false;
    }
    const uint64_t ubatch_row_heads = uint64_t(n_ubatch)*n_head_q;
    if (ubatch_row_heads > maximum/head_dim_v || ubatch_row_heads > maximum/sizeof(float2)) {
        return false;
    }
    const size_t accumulator_bytes      = size_t(ubatch_row_heads)*head_dim_v*sizeof(float);
    const size_t accumulator_meta_bytes = size_t(ubatch_row_heads)*sizeof(float2);

    if (parts_bytes > maximum - meta_bytes ||
            parts_bytes + meta_bytes > maximum - accumulator_bytes ||
            parts_bytes + meta_bytes + accumulator_bytes > maximum - accumulator_meta_bytes) {
        return false;
    }

    *transient_bytes = parts_bytes + meta_bytes + accumulator_bytes + accumulator_meta_bytes;
    return true;
}

bool ggml_cuda_flash_attn_ext_streamed_supported(const ggml_tensor * dst, size_t stage_bytes) {
    if (dst == nullptr || dst->op != GGML_OP_FLASH_ATTN_EXT) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    return Q != nullptr && K != nullptr && V != nullptr &&
        Q->type == GGML_TYPE_F32 &&
        ggml_backend_cuda_kv_stream_get_attention_mode(K->type, V->type) != GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_UNSUPPORTED &&
        Q->ne[0] == KV_STREAM_HEAD_DIM && V->ne[0] == KV_STREAM_HEAD_DIM &&
        Q->ne[1] >= 1 && Q->ne[3] == 1 && K->ne[3] == 1 && V->ne[3] == 1 &&
        K->ne[1] == V->ne[1] && K->ne[2] == V->ne[2] &&
        K->ne[1] % FATTN_KQ_STRIDE == 0 &&
        K->nb[0] == ggml_element_size(K) && V->nb[0] == ggml_element_size(V) &&
        K->nb[1] >= ggml_row_size(K->type, K->ne[0]) &&
        V->nb[1] >= ggml_row_size(V->type, V->ne[0]) &&
        K->nb[1] == ggml_row_size(K->type, K->ne[0])*K->ne[2] &&
        K->nb[2] == ggml_row_size(K->type, K->ne[0]) &&
        V->nb[1] == ggml_row_size(V->type, V->ne[0])*V->ne[2] &&
        V->nb[2] == ggml_row_size(V->type, V->ne[0]) &&
        (mask == nullptr || (mask->type == GGML_TYPE_F16 && ggml_is_contiguous(mask))) &&
        sinks == nullptr && kv_stream_block_tokens(dst, stage_bytes) > 0;
}

namespace {

static __global__ void kv_stream_record_deadline(
        const uint32_t * ready_flag,
        uint64_t * samples,
        uint64_t * misses) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        atomicAdd(reinterpret_cast<unsigned long long *>(samples), 1ULL);
        if (*ready_flag == 0) {
            atomicAdd(reinterpret_cast<unsigned long long *>(misses), 1ULL);
        }
    }
}

static bool kv_stream_collect_timing(ggml_cuda_kv_stream_transfer_ring * ring) {
    if (!ring->timing_pending) {
        return true;
    }
    const cudaError_t status = cudaEventQuery(ring->eval_end);
    if (status == cudaErrorNotReady) {
        return false;
    }
    CUDA_CHECK(status);

    float eval_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&eval_ms, ring->eval_start, ring->eval_end));
    double busy_ratio = 0.0;
    if (ring->copy_sample_recorded && eval_ms > 0.0f && ring->current_epoch_uploads > 0) {
        float copy_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(
            &copy_ms, ring->copy_sample_start, ring->copy_sample_end));
        GGML_ASSERT(ring->copy_sample_uploads > 0);
        busy_ratio = std::min(1.0, double(copy_ms)*double(ring->current_epoch_uploads)/
            (double(ring->copy_sample_uploads)*double(eval_ms)));
    }
    ring->last_copy_engine_busy_ratio = busy_ratio;
    ring->last_ring_peak_occupancy = ring->current_ring_peak_occupancy;
    ring->timing_pending = false;
    return true;
}

static void kv_stream_graph_upload(
        ggml_cuda_kv_stream_transfer_ring * ring,
        kv_stream_graph_request & request,
        uint32_t slot) {
    // Preserve the host cache's compact token-major layout. This makes each
    // page one contiguous K transfer plus one contiguous V transfer.
    char * k_stage = ring->pool_data + size_t(slot)*request.k_bytes;
    char * v_stage = ring->pool_data +
        size_t(ring->active_slots)*request.k_bytes + size_t(slot)*request.v_bytes;
    if (ring->slot_used[slot]) {
        CUDA_CHECK(cudaStreamWaitEvent(ring->copy_stream, ring->consumed[slot], 0));
        ++ring->stage_slot_reuses;
    }
    if (request.deadline_sample) {
        CUDA_CHECK(cudaMemsetAsync(
            ring->ready_flags_device + slot, 0, sizeof(uint32_t), ring->copy_stream));
    }
    if (ring->timing_current && !ring->copy_sample_recorded) {
        CUDA_CHECK(cudaEventRecord(ring->copy_sample_start, ring->copy_stream));
    }
    GGML_ASSERT(request.k_nb1 == request.k_row_bytes*request.n_head_kv);
    GGML_ASSERT(request.k_nb2 == request.k_row_bytes);
    GGML_ASSERT(request.v_nb1 == request.v_row_bytes*request.n_head_kv);
    GGML_ASSERT(request.v_nb2 == request.v_row_bytes);
    CUDA_CHECK(cudaMemcpyAsync(
        k_stage, request.k_data + request.token_begin*request.k_nb1,
        request.k_bytes, cudaMemcpyHostToDevice, ring->copy_stream));
    CUDA_CHECK(cudaMemcpyAsync(
        v_stage, request.v_data + request.token_begin*request.v_nb1,
        request.v_bytes, cudaMemcpyHostToDevice, ring->copy_stream));
    ring->host_to_device_copy_commands += 2;
    if (request.deadline_sample) {
        CUDA_CHECK(cudaMemsetAsync(
            ring->ready_flags_device + slot, 1, sizeof(uint32_t), ring->copy_stream));
    }
    if (ring->timing_current && !ring->copy_sample_recorded) {
        CUDA_CHECK(cudaEventRecord(ring->copy_sample_end, ring->copy_stream));
        ring->copy_sample_recorded = true;
        ring->copy_sample_uploads = 1;
    }
    CUDA_CHECK(cudaEventRecord(ring->ready[slot], ring->copy_stream));
    request.slot = slot;
    request.scheduled = true;
    ring->slot_used[slot] = 1;
    request.ready_slot = slot;
    ring->slot_request[slot] = size_t(&request - ring->graph_requests.data());
    ++ring->current_occupancy;
    ring->ring_peak_occupancy = std::max(
        ring->ring_peak_occupancy, ring->current_occupancy);
    ring->current_ring_peak_occupancy = std::max(
        ring->current_ring_peak_occupancy, ring->current_occupancy);
    ++ring->asynchronous_page_uploads;
    ++ring->current_epoch_uploads;
    if (ring->graph_resident_cache != nullptr) {
        ring->graph_resident_cache->stats.host_to_device_bytes +=
            request.k_bytes + request.v_bytes;
    }
    if (ring->current_layer != KV_STREAM_NO_LAYER && request.layer > ring->current_layer) {
        ++ring->cross_layer_prefetches;
    }
}

static bool kv_stream_graph_request_follows(
        const kv_stream_graph_request & previous,
        const kv_stream_graph_request & request) {
    return request.eligible && !request.scheduled && !request.consumed &&
        request.layer == previous.layer &&
        request.k_data == previous.k_data && request.v_data == previous.v_data &&
        request.k_nb1 == previous.k_nb1 && request.v_nb1 == previous.v_nb1 &&
        request.k_bytes == previous.k_bytes && request.v_bytes == previous.v_bytes &&
        request.token_count == previous.token_count &&
        request.token_begin == previous.token_begin + previous.token_count;
}

static void kv_stream_graph_upload_batch(
        ggml_cuda_kv_stream_transfer_ring * ring,
        size_t first_request,
        uint32_t first_slot,
        uint32_t batch_pages) {
    GGML_ASSERT(batch_pages > 1);
    GGML_ASSERT(first_request + batch_pages <= ring->graph_requests.size());
    GGML_ASSERT(first_slot + batch_pages <= ring->active_slots);
    auto & first = ring->graph_requests[first_request];

    for (uint32_t page = 0; page < batch_pages; ++page) {
        auto & request = ring->graph_requests[first_request + page];
        const uint32_t slot = first_slot + page;
        GGML_ASSERT(request.eligible && !request.scheduled && !request.consumed);
        GGML_ASSERT(ring->slot_request[slot] == KV_STREAM_NO_REQUEST);
        if (ring->slot_used[slot]) {
            CUDA_CHECK(cudaStreamWaitEvent(ring->copy_stream, ring->consumed[slot], 0));
            ++ring->stage_slot_reuses;
        }
        if (request.deadline_sample) {
            CUDA_CHECK(cudaMemsetAsync(
                ring->ready_flags_device + slot, 0, sizeof(uint32_t), ring->copy_stream));
        }
    }

    if (ring->timing_current && !ring->copy_sample_recorded) {
        CUDA_CHECK(cudaEventRecord(ring->copy_sample_start, ring->copy_stream));
    }
    GGML_ASSERT(first.k_nb1 == first.k_row_bytes*first.n_head_kv);
    GGML_ASSERT(first.k_nb2 == first.k_row_bytes);
    GGML_ASSERT(first.v_nb1 == first.v_row_bytes*first.n_head_kv);
    GGML_ASSERT(first.v_nb2 == first.v_row_bytes);
    char * k_stage = ring->pool_data + size_t(first_slot)*first.k_bytes;
    char * v_stage = ring->pool_data +
        size_t(ring->active_slots)*first.k_bytes + size_t(first_slot)*first.v_bytes;
    CUDA_CHECK(cudaMemcpyAsync(
        k_stage, first.k_data + first.token_begin*first.k_nb1,
        size_t(batch_pages)*first.k_bytes, cudaMemcpyHostToDevice, ring->copy_stream));
    CUDA_CHECK(cudaMemcpyAsync(
        v_stage, first.v_data + first.token_begin*first.v_nb1,
        size_t(batch_pages)*first.v_bytes, cudaMemcpyHostToDevice, ring->copy_stream));
    ring->host_to_device_copy_commands += 2;

    for (uint32_t page = 0; page < batch_pages; ++page) {
        auto & request = ring->graph_requests[first_request + page];
        if (request.deadline_sample) {
            CUDA_CHECK(cudaMemsetAsync(
                ring->ready_flags_device + first_slot + page, 1,
                sizeof(uint32_t), ring->copy_stream));
        }
    }
    if (ring->timing_current && !ring->copy_sample_recorded) {
        CUDA_CHECK(cudaEventRecord(ring->copy_sample_end, ring->copy_stream));
        ring->copy_sample_recorded = true;
        ring->copy_sample_uploads = batch_pages;
    }
    CUDA_CHECK(cudaEventRecord(ring->ready[first_slot], ring->copy_stream));

    for (uint32_t page = 0; page < batch_pages; ++page) {
        auto & request = ring->graph_requests[first_request + page];
        const uint32_t slot = first_slot + page;
        request.slot = slot;
        request.ready_slot = first_slot;
        request.scheduled = true;
        ring->slot_used[slot] = 1;
        ring->slot_request[slot] = first_request + page;
        ++ring->current_occupancy;
        ++ring->asynchronous_page_uploads;
        ++ring->current_epoch_uploads;
        if (ring->graph_resident_cache != nullptr) {
            ring->graph_resident_cache->stats.host_to_device_bytes +=
                request.k_bytes + request.v_bytes;
        }
        if (ring->current_layer != KV_STREAM_NO_LAYER && request.layer > ring->current_layer) {
            ++ring->cross_layer_prefetches;
        }
    }
    ring->ring_peak_occupancy = std::max(
        ring->ring_peak_occupancy, ring->current_occupancy);
    ring->current_ring_peak_occupancy = std::max(
        ring->current_ring_peak_occupancy, ring->current_occupancy);
}

static uint32_t kv_stream_graph_batch_pages(
        const ggml_cuda_kv_stream_transfer_ring * ring,
        uint32_t first_slot) {
    if (!ring->graph_active || first_slot >= ring->active_slots ||
            ring->next_request >= ring->graph_requests.size() ||
            ring->slot_request[first_slot] != KV_STREAM_NO_REQUEST) {
        return 0;
    }
    const auto & first = ring->graph_requests[ring->next_request];
    if (!first.eligible) {
        return 0;
    }
    GGML_ASSERT(!first.scheduled && !first.consumed);

    const uint32_t maximum = std::min<uint32_t>({
        KV_STREAM_COPY_BATCH_PAGES,
        ring->active_slots - first_slot,
        uint32_t(ring->graph_requests.size() - ring->next_request),
    });
    uint32_t pages = 1;
    while (pages < maximum) {
        if (ring->slot_request[first_slot + pages] != KV_STREAM_NO_REQUEST ||
                !kv_stream_graph_request_follows(
                    ring->graph_requests[ring->next_request + pages - 1],
                    ring->graph_requests[ring->next_request + pages])) {
            break;
        }
        ++pages;
    }
    return pages;
}

static uint32_t kv_stream_graph_schedule_batch(
        ggml_cuda_kv_stream_transfer_ring * ring,
        uint32_t first_slot) {
    const uint32_t batch_pages = kv_stream_graph_batch_pages(ring, first_slot);
    if (batch_pages == 0) {
        return 0;
    }
    const size_t first_request = ring->next_request;
    // Probe every immutable copy batch at its actual compute deadline. Mutable
    // tails are produced by this graph and are excluded from prefetch quality
    // feedback so they do not force unnecessary resident-page demotions.
    ring->graph_requests[first_request].deadline_sample =
        !ring->graph_requests[first_request].mutable_tail;
    ring->next_request += batch_pages;
    if (batch_pages == 1) {
        kv_stream_graph_upload(
            ring, ring->graph_requests[first_request], first_slot);
    } else {
        kv_stream_graph_upload_batch(
            ring, first_request, first_slot, batch_pages);
    }
    return batch_pages;
}

static void kv_stream_graph_fill_free_slots(ggml_cuda_kv_stream_transfer_ring * ring) {
    for (uint32_t slot = 0; slot < ring->active_slots; ++slot) {
        if (ring->slot_request[slot] != KV_STREAM_NO_REQUEST) {
            continue;
        }
        const uint32_t scheduled = kv_stream_graph_schedule_batch(ring, slot);
        if (scheduled == 0) {
            break;
        }
        slot += scheduled - 1;
    }
}

static bool kv_stream_graph_layer_begin(
        ggml_cuda_kv_stream_transfer_ring * ring,
        const void * k_key,
        cudaStream_t compute_stream) {
    if (!ring->graph_active) {
        return false;
    }
    const auto layer_it = ring->graph_layer_by_k.find(k_key);
    if (layer_it == ring->graph_layer_by_k.end()) {
        return false;
    }

    ring->current_layer = layer_it->second;
    CUDA_CHECK(cudaEventRecord(ring->producer_ready, compute_stream));
    CUDA_CHECK(cudaStreamWaitEvent(ring->copy_stream, ring->producer_ready, 0));
    for (auto & request : ring->graph_requests) {
        if (request.layer == ring->current_layer && request.mutable_tail) {
            request.eligible = true;
        }
    }
    kv_stream_graph_fill_free_slots(ring);
    return true;
}

static size_t kv_stream_graph_request_index(
        const ggml_cuda_kv_stream_transfer_ring * ring,
        const void * k_key,
        uint32_t page) {
    const auto it = ring->graph_request_by_k_page.find(k_key);
    if (it == ring->graph_request_by_k_page.end() || page >= it->second.size()) {
        return KV_STREAM_NO_REQUEST;
    }
    return it->second[page];
}

static void kv_stream_graph_release(
        ggml_cuda_kv_stream_transfer_ring * ring,
        size_t request_index,
        cudaStream_t compute_stream) {
    GGML_ASSERT(request_index < ring->graph_requests.size());
    auto & request = ring->graph_requests[request_index];
    GGML_ASSERT(request.scheduled && !request.consumed);
    const uint32_t slot = request.slot;
    GGML_ASSERT(ring->slot_request[slot] == request_index);
    CUDA_CHECK(cudaEventRecord(ring->consumed[slot], compute_stream));
    request.consumed = true;
    ring->slot_request[slot] = KV_STREAM_NO_REQUEST;
    GGML_ASSERT(ring->current_occupancy > 0);
    --ring->current_occupancy;
}

} // namespace

void ggml_cuda_kv_stream_graph_begin(ggml_cuda_kv_stream_transfer_ring * ring) {
    GGML_ASSERT(ring != nullptr);
    const bool timing_available = kv_stream_collect_timing(ring);
    ring->graph_active = true;
    ring->graph_decode = true;
    ring->graph_decode_span_pages = ring->forced_decode_span_pages != 0 ?
        ring->forced_decode_span_pages :
        (ring->span_tuner.use_bounded() ? KV_STREAM_DECODE_SPAN_PAGES : UINT32_MAX);
    ring->graph_layer_count = 0;
    ring->current_layer = KV_STREAM_NO_LAYER;
    ring->next_request = 0;
    ring->graph_resident_cache = nullptr;
    ring->graph_requests.clear();
    ring->graph_layer_by_k.clear();
    ring->graph_request_by_k_page.clear();
    ring->current_occupancy = 0;
    if (timing_available) {
        ring->current_ring_peak_occupancy = 0;
        ring->current_epoch_uploads = 0;
        ring->copy_sample_recorded = false;
        ring->copy_sample_uploads = 0;
    }
    ring->timing_current = timing_available;
    std::fill(ring->slot_request.begin(), ring->slot_request.end(), KV_STREAM_NO_REQUEST);
}

bool ggml_cuda_kv_stream_graph_add_attention(
        ggml_cuda_kv_stream_transfer_ring * ring,
        ggml_cuda_kv_stream_resident_cache * resident_cache,
        const ggml_tensor * dst) {
    GGML_ASSERT(ring != nullptr);
    const ggml_tensor * K = dst == nullptr ? nullptr : dst->src[1];
    const ggml_tensor * V = dst == nullptr ? nullptr : dst->src[2];
    if (resident_cache == nullptr || dst == nullptr ||
            !ggml_cuda_flash_attn_ext_streamed_supported(dst, ring->page_bytes)) {
        return false;
    }
    if (ring->graph_resident_cache != nullptr && ring->graph_resident_cache != resident_cache) {
        return false;
    }
    ring->graph_decode = ring->graph_decode && dst->src[0]->ne[1] == 1;
    if (dst->src[0]->ne[1] != 1) {
        // Graphs are rebuilt across warmup, prompt chunks, and slot reuse.
        // Relearn pointer-to-layer identity once per prefill graph while the
        // resident page contents are refreshed by the local multi-token path.
        if (ring->graph_resident_cache == nullptr) {
            resident_cache->layer_by_k.clear();
            resident_cache->layer_by_data.clear();
            resident_cache->next_layer = 0;
            ring->graph_resident_cache = resident_cache;
        }
        const uint32_t resident_layer = kv_stream_resident_layer(resident_cache, K->data);
        resident_cache->layer_by_data[K->data] = resident_layer;
        resident_cache->layer_by_data[V->data] = resident_layer;
        char * layer_base = resident_cache->pool_data + resident_cache->scratch_bytes +
            resident_cache->layer_offsets[resident_layer]*resident_cache->page_bytes;
        resident_cache->mirror_by_data[K->data] = layer_base;
        resident_cache->mirror_by_data[V->data] = layer_base +
            size_t(resident_cache->layer_pages[resident_layer])*K->nb[1]*resident_cache->page_tokens;
        return false;
    }
    ring->graph_resident_cache = resident_cache;
    // Resident placement is stable across evaluations, while deadlines must
    // follow this graph's finalized execution order.
    const uint32_t resident_layer = kv_stream_resident_layer(resident_cache, K->data);
    resident_cache->layer_by_data[K->data] = resident_layer;
    resident_cache->layer_by_data[V->data] = resident_layer;
    char * layer_base = resident_cache->pool_data + resident_cache->scratch_bytes +
        resident_cache->layer_offsets[resident_layer]*resident_cache->page_bytes;
    resident_cache->mirror_by_data[K->data] = layer_base;
    resident_cache->mirror_by_data[V->data] = layer_base +
        size_t(resident_cache->layer_pages[resident_layer])*K->nb[1]*resident_cache->page_tokens;

    const int64_t block_tokens = resident_cache->page_tokens;
    const int nchunks = int((K->ne[1] + block_tokens - 1)/block_tokens);
    if (uint32_t(nchunks) <= resident_cache->layer_pages[resident_layer]) {
        return true;
    }
    const uint32_t layer = ring->graph_layer_count++;
    ring->graph_layer_by_k[K->data] = layer;
    auto & page_requests = ring->graph_request_by_k_page[K->data];
    page_requests.assign(nchunks, KV_STREAM_NO_REQUEST);

    for (int chunk = 0; chunk < nchunks; ++chunk) {
        const uint32_t page = uint32_t(chunk);
        if (page < resident_cache->layer_pages[resident_layer]) {
            continue;
        }
        const int64_t token_begin = chunk*block_tokens;
        const int64_t token_count = std::min<int64_t>(block_tokens, K->ne[1] - token_begin);
        const size_t k_row_bytes = ggml_row_size(K->type, K->ne[0]);
        const size_t v_row_bytes = ggml_row_size(V->type, V->ne[0]);

        kv_stream_graph_request request;
        request.k_data = static_cast<const char *>(K->data);
        request.v_data = static_cast<const char *>(V->data);
        request.k_nb1 = K->nb[1];
        request.k_nb2 = K->nb[2];
        request.v_nb1 = V->nb[1];
        request.v_nb2 = V->nb[2];
        request.n_head_kv = K->ne[2];
        request.token_begin = token_begin;
        request.token_count = token_count;
        request.k_row_bytes = k_row_bytes;
        request.v_row_bytes = v_row_bytes;
        request.k_head_bytes = k_row_bytes*token_count;
        request.k_bytes = request.k_head_bytes*K->ne[2];
        request.v_offset = GGML_PAD(request.k_bytes, 128);
        request.v_head_bytes = v_row_bytes*token_count;
        request.v_bytes = request.v_head_bytes*V->ne[2];
        request.layer = layer;
        request.mutable_tail = chunk == nchunks - 1 ||
            kv_stream_page_mutable(resident_cache, page);
        request.eligible = !request.mutable_tail;
        GGML_ASSERT(request.v_offset + request.v_bytes == ring->page_bytes);

        page_requests[page] = ring->graph_requests.size();
        ring->graph_requests.push_back(request);
    }
    return true;
}

void ggml_cuda_kv_stream_graph_finalize(
        ggml_cuda_kv_stream_transfer_ring * ring, cudaStream_t compute_stream) {
    GGML_ASSERT(ring != nullptr);
    ring->last_graph_decode = ring->graph_decode;
    ring->last_graph_bounded = ring->graph_decode_span_pages != UINT32_MAX;
    ring->last_graph_streamed = !ring->graph_requests.empty();
    if (ring->graph_requests.empty()) {
        ring->timing_current = false;
        return;
    }
    if (ring->timing_current) {
        CUDA_CHECK(cudaEventRecord(ring->eval_start, compute_stream));
    }
    kv_stream_graph_fill_free_slots(ring);
}

void ggml_cuda_kv_stream_graph_end(
        ggml_cuda_kv_stream_transfer_ring * ring, cudaStream_t compute_stream) {
    GGML_ASSERT(ring != nullptr);
    if (ring->timing_current) {
        CUDA_CHECK(cudaEventRecord(ring->eval_end, compute_stream));
        ring->timing_pending = true;
        ring->timing_current = false;
    }
}

double ggml_cuda_kv_stream_copy_engine_busy_ratio(
        ggml_cuda_kv_stream_transfer_ring * ring) {
    if (ring == nullptr) {
        return 0.0;
    }
    (void) kv_stream_collect_timing(ring);
    return ring->last_copy_engine_busy_ratio;
}

uint32_t ggml_cuda_kv_stream_last_ring_peak_occupancy(
        const ggml_cuda_kv_stream_transfer_ring * ring) {
    return ring == nullptr ? 0 : ring->last_ring_peak_occupancy;
}

void ggml_cuda_flash_attn_ext_streamed(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        ggml_cuda_kv_stream_transfer_ring * transfer_ring,
        ggml_cuda_kv_stream_resident_cache * resident_cache) {
    GGML_ASSERT(transfer_ring != nullptr);
    void * stage_data = transfer_ring->pool_data;
    const size_t stage_bytes = transfer_ring->page_bytes;
    GGML_ASSERT(ggml_cuda_flash_attn_ext_streamed_supported(dst, stage_bytes));

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const auto attention_mode =
        ggml_backend_cuda_kv_stream_get_attention_mode(K->type, V->type);
    const bool convert_to_f16 =
        attention_mode == GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_F16;
#ifdef GGML_CUDA_FA_ALL_QUANTS
    const kv_stream_native_partial_fn native_partial =
        attention_mode == GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_DIRECT ?
            kv_stream_resolve_native_partial(K->type, V->type) : nullptr;
    GGML_ASSERT(attention_mode != GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_DIRECT ||
        native_partial != nullptr);
#endif // GGML_CUDA_FA_ALL_QUANTS
    const to_fp16_cuda_t converter_k = convert_to_f16 && K->type != GGML_TYPE_F16 ?
        ggml_get_to_fp16_cuda(K->type) : nullptr;
    const to_fp16_cuda_t converter_v = convert_to_f16 && V->type != GGML_TYPE_F16 ?
        ggml_get_to_fp16_cuda(V->type) : nullptr;
    GGML_ASSERT(!convert_to_f16 || K->type == GGML_TYPE_F16 || converter_k != nullptr);
    GGML_ASSERT(!convert_to_f16 || V->type == GGML_TYPE_F16 || converter_v != nullptr);
    const int64_t block_tokens = resident_cache != nullptr ?
        resident_cache->page_tokens : kv_stream_block_tokens(dst, stage_bytes);
    const int nchunks = (K->ne[1] + block_tokens - 1)/block_tokens;
    const int nrows = ggml_nrows(dst);
    const uint32_t maximum_streamed_span_pages = Q->ne[1] == 1 ?
        transfer_ring->graph_decode_span_pages : UINT32_MAX;

    struct chunk_descriptor {
        int64_t token_begin = 0;
        int64_t token_count = 0;
        size_t k_row_bytes = 0;
        size_t v_row_bytes = 0;
        size_t k_head_bytes = 0;
        size_t k_bytes = 0;
        size_t v_offset = 0;
        size_t v_head_bytes = 0;
        size_t v_bytes = 0;
        char * k_stage = nullptr;
        char * v_stage = nullptr;
        size_t k_stage_token_stride = 0;
        size_t k_stage_head_stride = 0;
        size_t v_stage_token_stride = 0;
        size_t v_stage_head_stride = 0;
        bool upload = true;
        bool resident_refresh = false;
        bool streamed = false;
        uint32_t slot = 0;
        size_t request_index = KV_STREAM_NO_REQUEST;
    };

    uint32_t resident_layer = 0;
    if (resident_cache != nullptr) {
        resident_layer = kv_stream_resident_layer(resident_cache, K->data);
    }
    const uint32_t resident_layer_pages = resident_cache == nullptr ?
        0 : resident_cache->layer_pages[resident_layer];
    const bool graph_planned = kv_stream_graph_layer_begin(
        transfer_ring, K->data, ctx.stream());

    std::vector<chunk_descriptor> chunks(nchunks);
    std::vector<size_t> streamed_chunks;
    streamed_chunks.reserve(nchunks);

    for (int chunk = 0; chunk < nchunks; ++chunk) {
        auto & desc = chunks[chunk];
        const int64_t token_begin = chunk*block_tokens;
        const int64_t token_count = std::min<int64_t>(block_tokens, K->ne[1] - token_begin);
        const size_t k_row_bytes = ggml_row_size(K->type, K->ne[0]);
        const size_t v_row_bytes = ggml_row_size(V->type, V->ne[0]);
        const size_t k_head_bytes = k_row_bytes*token_count;
        const size_t k_bytes = k_head_bytes*K->ne[2];
        const size_t v_offset = GGML_PAD(k_bytes, 128);
        const size_t v_head_bytes = v_row_bytes*token_count;
        const size_t v_bytes = v_head_bytes*V->ne[2];
        GGML_ASSERT(v_offset <= stage_bytes && v_bytes <= stage_bytes - v_offset);

        desc.token_begin = token_begin;
        desc.token_count = token_count;
        desc.k_row_bytes = k_row_bytes;
        desc.v_row_bytes = v_row_bytes;
        desc.k_head_bytes = k_head_bytes;
        desc.k_bytes = k_bytes;
        desc.v_offset = v_offset;
        desc.v_head_bytes = v_head_bytes;
        desc.v_bytes = v_bytes;
        desc.k_stage = static_cast<char *>(stage_data);
        desc.v_stage = desc.k_stage + v_offset;
        desc.k_stage_token_stride = k_row_bytes*K->ne[2];
        desc.k_stage_head_stride = k_row_bytes;
        desc.v_stage_token_stride = v_row_bytes*V->ne[2];
        desc.v_stage_head_stride = v_row_bytes;

        if (resident_cache != nullptr) {
            GGML_ASSERT(token_count == resident_cache->page_tokens);
            GGML_ASSERT(v_offset + v_bytes == resident_cache->page_bytes);

            const uint32_t page = token_begin/resident_cache->page_tokens;
            if (page < resident_layer_pages) {
                const size_t resident_index =
                    kv_stream_resident_index(resident_cache, resident_layer, page);
                // Keep each layer's resident K and V in separate token-major
                // planes so resident pages form one directly consumable span.
                char * layer_base = resident_cache->pool_data + resident_cache->scratch_bytes +
                    resident_cache->layer_offsets[resident_layer]*resident_cache->page_bytes;
                const size_t resident_k_plane_bytes =
                    size_t(resident_layer_pages)*k_bytes;
                desc.k_stage = layer_base + size_t(page)*k_bytes;
                desc.v_stage = layer_base + resident_k_plane_bytes + size_t(page)*v_bytes;
                if (resident_cache->loaded[resident_index]) {
                    ++resident_cache->stats.resident_hits;
                    desc.upload = resident_cache->precise_dirty_tracking[resident_layer] ?
                        resident_cache->dirty[resident_index] :
                        (dst->src[0]->ne[1] > 1 || chunk == nchunks - 1);
                    desc.resident_refresh = desc.upload;
                } else {
                    ++resident_cache->stats.resident_misses;
                    resident_cache->loaded[resident_index] = 1;
                }
            } else {
                ++resident_cache->stats.streamed_pages;
                desc.streamed = true;
            }
        } else {
            desc.streamed = true;
        }

        if (desc.streamed) {
            const size_t stream_index = streamed_chunks.size();
            if (graph_planned) {
                desc.request_index = kv_stream_graph_request_index(
                    transfer_ring, K->data, uint32_t(chunk));
                GGML_ASSERT(desc.request_index != KV_STREAM_NO_REQUEST);
            } else {
                desc.slot = uint32_t(stream_index%transfer_ring->active_slots);
                desc.k_stage = transfer_ring->pool_data + size_t(desc.slot)*desc.k_bytes;
                desc.v_stage = transfer_ring->pool_data +
                    size_t(transfer_ring->active_slots)*desc.k_bytes +
                    size_t(desc.slot)*desc.v_bytes;
            }
            streamed_chunks.push_back(chunk);
        }
    }

    const bool use_mma_prefill = !convert_to_f16 &&
        Q->ne[1] > 1 && Q->ne[0] == 256 && V->ne[0] == 256 &&
        mask != nullptr && Q->ne[2] % K->ne[2] == 0 && Q->ne[2]/K->ne[2] <= 8;
    const int partial_count = use_mma_prefill ? 1 : kv_stream_parts_per_chunk();
    GGML_ASSERT(partial_count > 0 && partial_count <= KV_STREAM_MAX_PARTS_PER_CHUNK);
    GGML_ASSERT(Q->ne[1] > 0 && nrows % Q->ne[1] == 0);
    const int64_t rows_per_query = nrows/Q->ne[1];
    const int64_t workspace_queries = use_mma_prefill ? Q->ne[1] :
        std::min<int64_t>(Q->ne[1], KV_STREAM_QUERY_WORKSPACE_TOKENS);
    const size_t workspace_rows = size_t(workspace_queries*rows_per_query);
    const size_t workspace_elements = workspace_rows*dst->ne[0];

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<float> parts(pool);
    ggml_cuda_pool_alloc<float2> meta(pool);
    ggml_cuda_pool_alloc<float> accumulator(pool);
    ggml_cuda_pool_alloc<float2> accumulator_meta(pool);
    const bool needs_partial_reduction = convert_to_f16 || (!streamed_chunks.empty() && nchunks > 1);
    if (needs_partial_reduction) {
        parts.alloc(size_t(partial_count)*workspace_elements);
        meta.alloc(size_t(partial_count)*workspace_rows);
        accumulator.alloc(ggml_nelements(dst));
        accumulator_meta.alloc(nrows);
    }

    auto upload = [&](const chunk_descriptor & desc, cudaStream_t stream) {
        if (!desc.upload) {
            return;
        }
        GGML_ASSERT(K->nb[1] == desc.k_stage_token_stride && K->nb[2] == desc.k_stage_head_stride);
        GGML_ASSERT(V->nb[1] == desc.v_stage_token_stride && V->nb[2] == desc.v_stage_head_stride);
        size_t dirty_row_count = 0;
        if (resident_cache != nullptr && desc.resident_refresh &&
                !resident_cache->all_pages_mutable) {
            for (const int64_t row : resident_cache->dirty_rows) {
                if (row >= desc.token_begin && row < desc.token_begin + desc.token_count) {
                    ++dirty_row_count;
                }
            }
        }
        const size_t dirty_bytes = dirty_row_count*(desc.k_stage_token_stride + desc.v_stage_token_stride);
        if (dirty_row_count > 0 && dirty_bytes < desc.k_bytes + desc.v_bytes) {
            for (const int64_t row : resident_cache->dirty_rows) {
                if (row < desc.token_begin || row >= desc.token_begin + desc.token_count) {
                    continue;
                }
                const size_t page_row = size_t(row - desc.token_begin);
                CUDA_CHECK(cudaMemcpyAsync(
                    desc.k_stage + page_row*desc.k_stage_token_stride,
                    static_cast<const char *>(K->data) + row*K->nb[1],
                    desc.k_stage_token_stride, cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(
                    desc.v_stage + page_row*desc.v_stage_token_stride,
                    static_cast<const char *>(V->data) + row*V->nb[1],
                    desc.v_stage_token_stride, cudaMemcpyHostToDevice, stream));
            }
            transfer_ring->host_to_device_copy_commands += 2*dirty_row_count;
            resident_cache->stats.host_to_device_bytes += dirty_bytes;
            return;
        }
        CUDA_CHECK(cudaMemcpyAsync(
            desc.k_stage, static_cast<const char *>(K->data) + desc.token_begin*K->nb[1],
            desc.k_bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(
            desc.v_stage, static_cast<const char *>(V->data) + desc.token_begin*V->nb[1],
            desc.v_bytes, cudaMemcpyHostToDevice, stream));
        transfer_ring->host_to_device_copy_commands += 2;
        if (resident_cache != nullptr) {
            resident_cache->stats.host_to_device_bytes += desc.k_bytes + desc.v_bytes;
        }
    };

    auto schedule_streamed = [&](size_t stream_index) {
        auto & desc = chunks[streamed_chunks[stream_index]];
        const uint32_t slot = desc.slot;
        if (transfer_ring->slot_used[slot]) {
            CUDA_CHECK(cudaStreamWaitEvent(
                transfer_ring->copy_stream, transfer_ring->consumed[slot], 0));
            ++transfer_ring->stage_slot_reuses;
        }
        upload(desc, transfer_ring->copy_stream);
        CUDA_CHECK(cudaEventRecord(transfer_ring->ready[slot], transfer_ring->copy_stream));
        transfer_ring->slot_used[slot] = 1;
        ++transfer_ring->asynchronous_page_uploads;
    };

    if (!graph_planned && !streamed_chunks.empty()) {
        // SET_ROWS and all other producers for this layer are ordered before
        // this marker on the compute stream. The copy stream may then run
        // independently while attention consumes previously prepared pages.
        CUDA_CHECK(cudaEventRecord(transfer_ring->producer_ready, ctx.stream()));
        CUDA_CHECK(cudaStreamWaitEvent(
            transfer_ring->copy_stream, transfer_ring->producer_ready, 0));
        const size_t initial = std::min<size_t>(
            transfer_ring->active_slots, streamed_chunks.size());
        for (size_t i = 0; i < initial; ++i) {
            schedule_streamed(i);
        }
    }

    size_t stream_index = 0;
    for (int chunk = 0; chunk < nchunks; ++chunk) {
        auto & desc = chunks[chunk];
        uint32_t streamed_span_pages = 0;
        if (desc.streamed) {
            streamed_span_pages = 1;
            uint32_t ready_slot = desc.slot;
            if (graph_planned) {
                auto & request = transfer_ring->graph_requests[desc.request_index];
                GGML_ASSERT(request.scheduled && !request.consumed);
                desc.slot = request.slot;
                ready_slot = request.ready_slot;
                desc.k_stage = transfer_ring->pool_data + size_t(desc.slot)*desc.k_bytes;
                desc.v_stage = transfer_ring->pool_data +
                    size_t(transfer_ring->active_slots)*desc.k_bytes +
                    size_t(desc.slot)*desc.v_bytes;
                if (request.deadline_sample) {
                    kv_stream_record_deadline<<<1, 1, 0, ctx.stream()>>>(
                        transfer_ring->ready_flags_device + desc.slot,
                        transfer_ring->deadline_counters_device + 0,
                        transfer_ring->deadline_counters_device + 1);
                    CUDA_CHECK(cudaGetLastError());
                }
            }
            CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), transfer_ring->ready[ready_slot], 0));
            ++transfer_ring->compute_stream_waits;

            // Coalesce ready pages that occupy consecutive plane slots. The
            // head stride remains the full active-ring plane width, while the
            // tensor's token extent grows across adjacent slots.
            while (!convert_to_f16 &&
                    streamed_span_pages < maximum_streamed_span_pages &&
                    chunk + int(streamed_span_pages) < nchunks) {
                auto & candidate = chunks[chunk + streamed_span_pages];
                uint32_t candidate_ready_slot = candidate.slot;
                if (!candidate.streamed) {
                    break;
                }
                if (graph_planned) {
                    auto & request = transfer_ring->graph_requests[candidate.request_index];
                    if (!request.scheduled || request.consumed) {
                        break;
                    }
                    candidate.slot = request.slot;
                    candidate_ready_slot = request.ready_slot;
                    candidate.k_stage = transfer_ring->pool_data +
                        size_t(candidate.slot)*candidate.k_bytes;
                    candidate.v_stage = transfer_ring->pool_data +
                        size_t(transfer_ring->active_slots)*candidate.k_bytes +
                        size_t(candidate.slot)*candidate.v_bytes;
                }
                if (candidate.slot != desc.slot + streamed_span_pages ||
                        candidate.token_begin != desc.token_begin +
                            int64_t(streamed_span_pages)*block_tokens) {
                    break;
                }
                if (graph_planned) {
                    auto & request = transfer_ring->graph_requests[candidate.request_index];
                    if (request.deadline_sample) {
                        kv_stream_record_deadline<<<1, 1, 0, ctx.stream()>>>(
                            transfer_ring->ready_flags_device + candidate.slot,
                            transfer_ring->deadline_counters_device + 0,
                            transfer_ring->deadline_counters_device + 1);
                        CUDA_CHECK(cudaGetLastError());
                    }
                }
                if (candidate_ready_slot != ready_slot) {
                    CUDA_CHECK(cudaStreamWaitEvent(
                        ctx.stream(), transfer_ring->ready[candidate_ready_slot], 0));
                    ++transfer_ring->compute_stream_waits;
                    ready_slot = candidate_ready_slot;
                }
                ++streamed_span_pages;
            }
            desc.token_count = int64_t(streamed_span_pages)*block_tokens;
            if (resident_cache != nullptr) {
                ++resident_cache->stats.streamed_attention_spans;
                resident_cache->stats.streamed_pages_attended += streamed_span_pages;
            }
        } else {
            if (convert_to_f16) {
                // Generic quantized K/V is converted one page at a time into
                // the bounded workspace. Keep resident pages separate so the
                // fallback never creates a context-sized F16 allocation.
                upload(desc, ctx.stream());
                if (desc.upload && resident_cache->precise_dirty_tracking[resident_layer]) {
                    resident_cache->dirty[kv_stream_resident_index(
                        resident_cache, resident_layer, uint32_t(chunk))] = 0;
                }
                ++resident_cache->stats.resident_attention_spans;
                ++resident_cache->stats.resident_pages_attended;
            } else {
                // Native kernels consume the resident K/V prefix in one span.
                if (chunk > 0) {
                    continue;
                }
                const uint32_t resident_span_pages =
                    std::min<uint32_t>(uint32_t(nchunks), resident_layer_pages);
                GGML_ASSERT(resident_span_pages > 0);
                for (uint32_t page = 0; page < resident_span_pages; ++page) {
                    upload(chunks[page], ctx.stream());
                    if (chunks[page].upload &&
                            resident_cache->precise_dirty_tracking[resident_layer]) {
                        resident_cache->dirty[
                            kv_stream_resident_index(resident_cache, resident_layer, page)] = 0;
                    }
                }
                desc.token_count = int64_t(resident_span_pages)*block_tokens;
                desc.k_head_bytes = desc.k_row_bytes*desc.token_count;
                desc.k_bytes = desc.k_head_bytes*K->ne[2];
                desc.v_head_bytes = desc.v_row_bytes*desc.token_count;
                desc.v_bytes = desc.v_head_bytes*V->ne[2];
                ++resident_cache->stats.resident_attention_spans;
                resident_cache->stats.resident_pages_attended += resident_span_pages;
            }
        }

        ggml_tensor staged_k = *K;
        ggml_tensor staged_v = *V;
        staged_k.data = desc.k_stage;
        staged_k.ne[1] = desc.token_count;
        staged_k.nb[1] = desc.k_stage_token_stride;
        staged_k.nb[2] = desc.k_stage_head_stride;
        staged_k.nb[3] = desc.k_stage_token_stride*desc.token_count;
        staged_v.data = desc.v_stage;
        staged_v.ne[1] = desc.token_count;
        staged_v.nb[1] = desc.v_stage_token_stride;
        staged_v.nb[2] = desc.v_stage_head_stride;
        staged_v.nb[3] = desc.v_stage_token_stride*desc.token_count;

        ggml_tensor converted_k{};
        ggml_tensor converted_v{};
        if (convert_to_f16) {
            GGML_ASSERT(transfer_ring->conversion_data != nullptr);
            const size_t k_elements = size_t(K->ne[0])*desc.token_count*K->ne[2];
            const size_t v_elements = size_t(V->ne[0])*desc.token_count*V->ne[2];
            const size_t k_f16_bytes = k_elements*sizeof(half);
            const size_t v_f16_offset = GGML_PAD(k_f16_bytes, 128);
            const size_t v_f16_bytes = v_elements*sizeof(half);
            GGML_ASSERT(v_f16_offset <= transfer_ring->conversion_bytes &&
                v_f16_bytes <= transfer_ring->conversion_bytes - v_f16_offset);

            auto convert_page = [&](const ggml_tensor & src, half * converted,
                    size_t elements, to_fp16_cuda_t converter) {
                if (converter == nullptr) {
                    CUDA_CHECK(cudaMemcpyAsync(
                        converted, src.data, elements*sizeof(half),
                        cudaMemcpyDeviceToDevice, ctx.stream()));
                    return;
                }
                converter(src.data, converted, elements, ctx.stream());
            };

            auto * converted_k_data = reinterpret_cast<half *>(transfer_ring->conversion_data);
            auto * converted_v_data = reinterpret_cast<half *>(
                transfer_ring->conversion_data + v_f16_offset);
            convert_page(staged_k, converted_k_data, k_elements, converter_k);
            convert_page(staged_v, converted_v_data, v_elements, converter_v);

            converted_k = staged_k;
            converted_k.type = GGML_TYPE_F16;
            converted_k.data = converted_k_data;
            converted_k.nb[0] = sizeof(half);
            converted_k.nb[1] = size_t(K->ne[0])*K->ne[2]*sizeof(half);
            converted_k.nb[2] = size_t(K->ne[0])*sizeof(half);
            converted_k.nb[3] = converted_k.nb[1]*converted_k.ne[1];
            converted_v = staged_v;
            converted_v.type = GGML_TYPE_F16;
            converted_v.data = converted_v_data;
            converted_v.nb[0] = sizeof(half);
            converted_v.nb[1] = size_t(V->ne[0])*V->ne[2]*sizeof(half);
            converted_v.nb[2] = size_t(V->ne[0])*sizeof(half);
            converted_v.nb[3] = converted_v.nb[1]*converted_v.ne[1];
        }
        ggml_tensor staged_mask{};
        ggml_tensor * staged_mask_ptr = nullptr;
        if (mask != nullptr) {
            staged_mask = *mask;
            staged_mask.data = static_cast<char *>(mask->data) + desc.token_begin*mask->nb[0];
            staged_mask.ne[0] = desc.token_count;
            staged_mask_ptr = &staged_mask;
        }

        ggml_tensor staged_dst = *dst;
        staged_dst.src[1] = convert_to_f16 ? &converted_k : &staged_k;
        staged_dst.src[2] = convert_to_f16 ? &converted_v : &staged_v;
        staged_dst.src[3] = staged_mask_ptr;

        // Preserve normal CUDA flash attention when the active cache is fully resident or fits in one streamed page.
        // This avoids a partial reduction and keeps logits identical to a non-streamed cache.
        if (!convert_to_f16 && (streamed_chunks.empty() || nchunks == 1)) {
            ggml_cuda_flash_attn_ext(ctx, &staged_dst);
            if (desc.streamed) {
                if (graph_planned) {
                    kv_stream_graph_release(transfer_ring, desc.request_index, ctx.stream());
                    kv_stream_graph_fill_free_slots(transfer_ring);
                } else {
                    CUDA_CHECK(cudaEventRecord(transfer_ring->consumed[desc.slot], ctx.stream()));
                }
            }
            return;
        }

        GGML_ASSERT(needs_partial_reduction);

        // Keep each staged KV span alive while all bounded query tiles consume it.
        // This bounds vector scratch without issuing another H2D transfer for the span.
        for (int64_t query_begin = 0; query_begin < Q->ne[1]; query_begin += workspace_queries) {
            const int64_t query_count = std::min<int64_t>(
                workspace_queries, Q->ne[1] - query_begin);
            ggml_tensor query_q = *Q;
            query_q.data = static_cast<char *>(Q->data) + query_begin*Q->nb[1];
            query_q.ne[1] = query_count;

            ggml_tensor query_mask{};
            ggml_tensor * query_mask_ptr = nullptr;
            if (staged_mask_ptr != nullptr) {
                query_mask = *staged_mask_ptr;
                query_mask.data = static_cast<char *>(staged_mask_ptr->data) +
                    query_begin*mask->nb[1];
                query_mask.ne[1] = query_count;
                query_mask_ptr = &query_mask;
            }

            ggml_tensor query_dst = staged_dst;
            query_dst.ne[1] = query_count;
            query_dst.src[0] = &query_q;
            query_dst.src[3] = query_mask_ptr;

            if (use_mma_prefill) {
                ggml_cuda_flash_attn_ext_mma_f16_partial_case<256, 256, 8, 8>(
                    ctx, &query_dst, parts.ptr, meta.ptr);
                if (resident_cache != nullptr) {
                    ++resident_cache->stats.mma_prefill_attention_spans;
                }
            } else {
                if (convert_to_f16) {
                    ggml_cuda_flash_attn_ext_vec_partial_case<
                        KV_STREAM_HEAD_DIM, GGML_TYPE_F16, GGML_TYPE_F16>(
                            ctx, &query_dst, parts.ptr, meta.ptr, partial_count);
                } else {
#ifdef GGML_CUDA_FA_ALL_QUANTS
                    GGML_ASSERT(native_partial != nullptr);
                    native_partial(ctx, &query_dst, parts.ptr, meta.ptr, partial_count);
#else
                    GGML_ABORT("native quantized KV streaming requires GGML_CUDA_FA_ALL_QUANTS");
#endif // GGML_CUDA_FA_ALL_QUANTS
                }
            }

            const int tile_nrows = int(query_count*rows_per_query);
            const size_t row_offset = size_t(query_begin*rows_per_query);
            const dim3 blocks(tile_nrows, 1, 1);
            const dim3 threads(KV_STREAM_HEAD_DIM, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params(blocks, threads, 0, ctx.stream());
            ggml_cuda_kernel_launch(
                kv_stream_accumulate_chunk_results<KV_STREAM_HEAD_DIM>, launch_params,
                parts.ptr, meta.ptr, accumulator.ptr + row_offset*dst->ne[0],
                accumulator_meta.ptr + row_offset, tile_nrows, chunk == 0, partial_count);
            CUDA_CHECK(cudaGetLastError());
        }

        if (desc.streamed) {
            for (uint32_t page = 0; page < streamed_span_pages; ++page) {
                auto & member = chunks[chunk + page];
                if (graph_planned) {
                    kv_stream_graph_release(
                        transfer_ring, member.request_index, ctx.stream());
                } else {
                    CUDA_CHECK(cudaEventRecord(
                        transfer_ring->consumed[member.slot], ctx.stream()));
                    const size_t next = stream_index + page + transfer_ring->active_slots;
                    if (next < streamed_chunks.size()) {
                        schedule_streamed(next);
                    }
                }
            }
            stream_index += streamed_span_pages;
            if (graph_planned) {
                kv_stream_graph_fill_free_slots(transfer_ring);
            }
            chunk += int(streamed_span_pages) - 1;
        }
    }

    const dim3 blocks(nrows, 1, 1);
    const dim3 threads(KV_STREAM_HEAD_DIM, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks, threads, 0, ctx.stream());
    ggml_cuda_kernel_launch(kv_stream_normalize_chunk_results<KV_STREAM_HEAD_DIM>, launch_params,
        accumulator.ptr, accumulator_meta.ptr, static_cast<float *>(dst->data), nrows);
    CUDA_CHECK(cudaGetLastError());
}

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include "../ggml-kvarn.h"

// Flash-attention path census, opt-in via GGML_FATTN_PATH_STATS=1.
// Counts every ggml_cuda_flash_attn_ext dispatch keyed by (path, K type, V type,
// n_q = Q->ne[1], gqa_ratio, ncols2 when the MMA path packs GQA). Dumped to stderr at
// exit. Purpose (W5): prove or kill the premise that an f16 draft K/V cache routes the
// drafter's width-1 attention onto the GQA-packed MMA path instead of VEC.
static bool ggml_cuda_fattn_path_stats_enabled() {
    static const bool enabled = [] { const char * e = getenv("GGML_FATTN_PATH_STATS"); return e && e[0] == '1'; }();
    return enabled;
}

static void ggml_cuda_fattn_path_note(const char * path, const ggml_tensor * dst, int ncols2) {
    {
        // fallback ledger (GGML_LEDGER=1): kernel family per K/V pair and query width
        const ggml_tensor * Q = dst->src[0];
        const ggml_tensor * K = dst->src[1];
        const ggml_tensor * V = dst->src[2];
        const uint64_t id = ggml_cuda_ledger_mix(ggml_cuda_ledger_mix(ggml_cuda_ledger_mix(ggml_cuda_ledger_mix(
            (uint64_t) (uintptr_t) path, (uint64_t) K->type), (uint64_t) V->type), (uint64_t) Q->ne[0]),
            (uint64_t) ggml_cuda_ledger_width_bucket(Q->ne[1]) * 64 + (uint64_t) (ncols2 + 1));
        ggml_cuda_ledger_count("cuda.fattn", id, [&](char * buf, size_t size) {
            char w[16];
            ggml_cuda_ledger_width_str(Q->ne[1], w, sizeof(w));
            snprintf(buf, size, "path=%s K=%s V=%s D=%d n_q=%s ncols2=%d", path, ggml_type_name(K->type), ggml_type_name(V->type),
                (int) Q->ne[0], w, ncols2);
        });
    }
    if (!ggml_cuda_fattn_path_stats_enabled()) {
        return;
    }
    static std::mutex mtx;
    static std::map<std::string, uint64_t> counts;
    static const bool registered = [] {
        atexit([] {
            std::lock_guard<std::mutex> lock(mtx);
            fprintf(stderr, "fattn_path_stats: begin (%zu keys)\n", counts.size());
            for (const auto & kv : counts) {
                fprintf(stderr, "fattn_path_stats: %s count=%llu\n", kv.first.c_str(), (unsigned long long) kv.second);
            }
            fprintf(stderr, "fattn_path_stats: end\n");
        });
        return true;
    }();
    GGML_UNUSED(registered);
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    char key[256];
    snprintf(key, sizeof(key), "path=%s K=%s V=%s D=%d n_q=%d gqa=%d ncols2=%d kv_len_bucket=%dk",
        path, ggml_type_name(K->type), ggml_type_name(V->type), (int) Q->ne[0], (int) Q->ne[1],
        (int) (Q->ne[2] / K->ne[2]), ncols2, (int) (K->ne[1] / 1024));
    std::lock_guard<std::mutex> lock(mtx);
    counts[key]++;
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// one list per group of ncols1 queries: a column is selected if any query of the group can see it
template <int ncols1, bool oob>
__launch_bounds__(256, 1)
static __global__ void flash_attn_mask_to_sparse_indices(
        const half * mask_ptr, int32_t * indices_ptr, int32_t * counts_ptr, const int ne30, const int n_queries,
        const int n_kv_max, const int64_t s31, const int64_t s33) {
    ggml_cuda_pdl_sync();

    constexpr int values_per_lane = 8;
    const int tid      = threadIdx.x;
    const int warp     = tid / WARP_SIZE;
    const int lane     = tid % WARP_SIZE;
    const int sequence = blockIdx.y;
    const int group    = blockIdx.x;

    const int q0 = group*ncols1;
    const int q1 = min(q0 + ncols1, n_queries);

    const half * mask = mask_ptr + sequence*s33 + q0*s31;
    int32_t * indices = indices_ptr + (int64_t(sequence)*gridDim.x + group)*n_kv_max;

    __shared__ int warp_offsets[256/WARP_SIZE];
    __shared__ int row_count;
    __shared__ int chunk_count;

    if (tid == 0) {
        row_count = 0;
    }
    __syncthreads();

    for (int i0 = 0; i0 < ne30; i0 += blockDim.x*values_per_lane) {
        uint32_t selected_warp[values_per_lane];
        int warp_count = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*WARP_SIZE + lane;
            bool selected = false;
            if (i < ne30) {
#pragma unroll
                for (int q = 0; q < ncols1; ++q) {
                    selected |= (!oob || q < q1 - q0) && isfinite(__half2float(mask[q*s31 + i]));
                }
            }
            selected_warp[item] = __ballot_sync(0xFFFFFFFF, selected);
            warp_count += __popc(selected_warp[item]);
        }

        if (lane == 0) {
            warp_offsets[warp] = warp_count;
        }
        __syncthreads();

        if (tid == 0) {
            int offset = 0;
#pragma unroll
            for (int iw = 0; iw < 256/WARP_SIZE; ++iw) {
                const int count = warp_offsets[iw];
                warp_offsets[iw] = offset;
                offset += count;
            }
            chunk_count = offset;
        }
        __syncthreads();

        const uint32_t lane_mask = lane == 0 ? 0 : (1u << lane) - 1;
        int warp_item_offset = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*WARP_SIZE + lane;
            const int dst = row_count + warp_offsets[warp] + warp_item_offset + __popc(selected_warp[item] & lane_mask);
            if ((selected_warp[item] & (uint32_t(1) << lane)) && dst < n_kv_max) {
                indices[dst] = i;
            }
            warp_item_offset += __popc(selected_warp[item]);
        }
        __syncthreads();

        if (tid == 0) {
            row_count += chunk_count;
        }
        __syncthreads();
    }

    const int count = min(row_count, n_kv_max);
    for (int i = count + tid; i < n_kv_max; i += blockDim.x) {
        indices[i] = -1;
    }
    if (tid == 0) {
        counts_ptr[int64_t(sequence)*gridDim.x + group] = count;
    }
    __syncthreads();

    // the dependent grid reads indices, signal once the row is complete
    ggml_cuda_pdl_lc();
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

void ggml_cuda_flash_attn_ext_compact_mask(
        const ggml_tensor * mask, int32_t * indices, int32_t * counts, int32_t n_queries, int32_t ncols1, int32_t n_kv_max, cudaStream_t stream) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(mask, indices, counts, n_queries, ncols1, n_kv_max, stream);
    GGML_ABORT("sparse flash attention is only supported on NVIDIA CUDA");
#else
    const int64_t s31 = mask->nb[1] / sizeof(half);
    const int64_t s33 = mask->nb[3] / sizeof(half);
    const dim3 blocks_num((n_queries + ncols1 - 1)/ncols1, mask->ne[3], 1);
    const dim3 block_dim(256, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks_num, block_dim, 0, stream);
    // the last group of queries is partial only if ncols1 does not divide n_queries
    GGML_ASSERT(ncols1 == 1 || ncols1 == 8);
    const auto kernel = ncols1 == 1       ? flash_attn_mask_to_sparse_indices<1, false> :
                        n_queries % 8 != 0 ? flash_attn_mask_to_sparse_indices<8, true>  :
                                             flash_attn_mask_to_sparse_indices<8, false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
        (const half *) mask->data, indices, counts, int(mask->ne[0]), n_queries, n_kv_max, s31, s33);
    CUDA_CHECK(cudaGetLastError());
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

bool ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(const int cc, const ggml_tensor * dst, const int ncols1, const int ncols2) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(cc, dst, ncols1, ncols2);
    return false;
#else
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);

    // the dense kernel handles up to 64/ncols2 queries per K/V pass, the single-query gather has to beat that
    const int64_t n_gather = (ncols1 == 1 ? std::min<int64_t>(Q->ne[1], 64/ncols2) : ncols1) * (int64_t) n_kv_max;

    return GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) &&
        mask != nullptr && n_kv_max > 0 && max_bias == 0.0f && logit_softcap == 0.0f &&
        mask->ne[0] == K->ne[1] && mask->ne[1] >= Q->ne[1] && mask->ne[2] == 1 &&
        K->ne[1] >= std::max<int64_t>(4096, 2*n_gather);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst, bool preserve_cand = false) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if constexpr (ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, 1, ncols2)) {
        // a sparse variant at the full tile width gathers the union of its queries once, prefer it for large batches
        constexpr bool has_wide_sparse = ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, 64/ncols2, ncols2);
        if (!(has_wide_sparse && Q->ne[1] > 32/ncols2) && ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1, ncols2)) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 1, ncols2>(ctx, dst, preserve_cand);
            return;
        }
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst, preserve_cand);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if (Q->ne[1] <= 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst, preserve_cand);
            return;
        }
    }

    if (Q->ne[1] <= 32/ncols2 || (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) ||
            (GGML_CUDA_CC_IS_AMD(cc) && DKQ > 256)) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst, preserve_cand);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst, preserve_cand);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst, bool preserve_cand = false) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations. Quantized tensors
    //     (incl. turbo2/3/4) are skipped here: their loaders dequantize into SMEM via the
    //     swizzled/padded tile helpers rather than reading nb[] directly, so the 16-byte-stride
    //     alignment this loop checks for doesn't apply to them.
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // [HQ4] opt-in prefill tile probe (fattn-prefill-policy.h); unset = the rule below
    if constexpr (DKQ == 256 && DV == 256) {
        const int arm = ggml_cuda_fa_prefill_ncols2_env();
        if (arm != 0) {
            switch (ggml_cuda_fa_prefill_ncols2_pick(arm, cc, DKQ, DV, Q->ne[1], gqa_ratio, Q->ne[3] == 1, use_gqa_opt)) {
                case 1:
                    ggml_cuda_fattn_path_note("mma_f16_prefill_probe", dst, 1);
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst, preserve_cand);
                    return;
                case 2:
                    ggml_cuda_fattn_path_note("mma_f16_prefill_probe", dst, 2);
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst, preserve_cand);
                    return;
                case 8:
                    ggml_cuda_fattn_path_note("mma_f16_prefill_probe", dst, 8);
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst, preserve_cand);
                    return;
                default:
                    break;
            }
        }
    }

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_fattn_path_note("mma_f16", dst, 8);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst, preserve_cand);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_fattn_path_note("mma_f16", dst, 4);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst, preserve_cand);
            return;
        }

        if constexpr (DKQ <= 256) {
            if (use_gqa_opt && gqa_ratio % 2 == 0) {
                ggml_cuda_fattn_path_note("mma_f16", dst, 2);
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst, preserve_cand);
                return;
            }

            ggml_cuda_fattn_path_note("mma_f16", dst, 1);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst, preserve_cand);
            return;
        } else {
            GGML_ABORT("fatal error");
        }
    }

    // On RDNA it is preferable to minimize wasted compute vs. duplicate I/O for the mask.
    if (amd_wmma_available(cc)) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst, preserve_cand);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst, preserve_cand);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 2 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst, preserve_cand);
            return;
        }
    }

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_fattn_path_note("mma_f16", dst, 8);
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst, preserve_cand);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_fattn_path_note("mma_f16", dst, 4);
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst, preserve_cand);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 1) {
        ggml_cuda_fattn_path_note("mma_f16", dst, 2);
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst, preserve_cand);
        return;
    }

    if constexpr (DKQ <= 256) {
        ggml_cuda_fattn_path_note("mma_f16", dst, 1);
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst, preserve_cand);
    } else {
        GGML_ABORT("fatal error");
    }
}

// ---------------------------------------------------------------------------
// turbo4 fused MMA decode dispatch (mirrors the f16 switch helpers, type-parametric).
// Only reached from the gate for turbo4 K==V, D in {128,256}, Q->ne[1] <= 4, turing MMA.
//
// The reachable (ncols1, ncols2) set for Q->ne[1] in {1..4} with GQA-packing is exactly
// {(1,8),(2,8),(4,8),(2,4),(4,4),(4,2),(8,1)} — the 7 compiled instances per D. Each ncols2
// has an explicit dispatcher so ONLY those pairs are instantiated (an unguarded ncols1=8/ncols2
// fallthrough would also instantiate uncompiled cases like (8,4) -> link error).

// SLOWKV: fused routes for the non-default D256 KV pairs on sm86 (GGML_VL_AT_Q5G6 keeps tq5_0/turbo4 alone).
// GGML_SLOWKV picks the pairs:
//   unset, empty or "0" = off (routing and kernels exactly as before); "1" or "all" = all four;
//   tq6_0/tq6_0 is always on since FA66 (see ggml_cuda_fattn_slowkv_pair);
//   otherwise a comma list of q8 (q8_0/q8_0), turbo4 (turbo4_0/turbo4_0), tq6 (tq6_0/tq6_0), q5_1 (q8_0 K / q5_1 V).
// Routes per pair:
//   - width 5, GQA 6 (compact_applies): the compact tile, 5 queries x 6 heads in the 32-column (4,8) tile (see
//     fattn-query-layout.cuh). q8_0/q8_0 swaps its (8,8) width-5 tile for it.
//   - turbo4/turbo4 and tq6_0/tq6_0 widths 5-8 that do not take the compact tile (wide_applies): the fused (8,8) tile.
//     Without the switch these pairs are fused only up to width 4 and fall to the generic f16 MMA route above it,
//     which converts the whole K and V cache to f16 per call.
//   - q8_0/q5_1 widths 1-8 (q5_1_applies): the fused tiles with the q5_1 V loader (flash_attn_ext_q5_1_load_tile)
//     instead of the generic route (full-cache f16 conversion at every width).
// The switch is read once per process.
enum : int {
    GGML_CUDA_SLOWKV_Q8_Q8   = 1,
    GGML_CUDA_SLOWKV_T4_T4   = 2,
    GGML_CUDA_SLOWKV_TQ6_TQ6 = 4,
    GGML_CUDA_SLOWKV_Q8_Q51  = 8,
};

static int ggml_cuda_fattn_slowkv_pairs() {
    static const int pairs = [] {
        const char * e = getenv("GGML_SLOWKV");
        if (e == nullptr || e[0] == '\0' || strcmp(e, "0") == 0) {
            return 0;
        }
        if (strcmp(e, "1") == 0 || strcmp(e, "all") == 0) {
            return GGML_CUDA_SLOWKV_Q8_Q8 | GGML_CUDA_SLOWKV_T4_T4 | GGML_CUDA_SLOWKV_TQ6_TQ6 | GGML_CUDA_SLOWKV_Q8_Q51;
        }
        int m = 0;
        const char * p = e;
        while (*p) {
            const char * end = strchr(p, ',');
            const size_t n = end ? size_t(end - p) : strlen(p);
            if (n == 2 && strncmp(p, "q8", 2) == 0) {
                m |= GGML_CUDA_SLOWKV_Q8_Q8;
            } else if (n == 6 && strncmp(p, "turbo4", 6) == 0) {
                m |= GGML_CUDA_SLOWKV_T4_T4;
            } else if (n == 3 && strncmp(p, "tq6", 3) == 0) {
                m |= GGML_CUDA_SLOWKV_TQ6_TQ6;
            } else if (n == 4 && strncmp(p, "q5_1", 4) == 0) {
                m |= GGML_CUDA_SLOWKV_Q8_Q51;
            } else if (n > 0) {
                GGML_LOG_WARN("GGML_SLOWKV: ignoring unknown pair \"%.*s\" (known: q8, turbo4, tq6, q5_1, 1, all)\n", (int) n, p);
            }
            p += n;
            if (*p == ',') {
                ++p;
            }
        }
        return m;
    }();
    return pairs;
}

static bool ggml_cuda_fattn_slowkv_pair(const ggml_type type_K, const ggml_type type_V) {
    // FA66: matched tq6_0/tq6_0 (the MTP drafter cache under a KVarN 4/4 trunk) takes the fused SLOWKV tiles at widths
    // 5-8 by default, independent of GGML_SLOWKV (the "tq6" entry is kept and is now a no-op).
    if (type_K == GGML_TYPE_TQ6_0 && type_V == GGML_TYPE_TQ6_0) {
        return true;
    }
    const int pairs = ggml_cuda_fattn_slowkv_pairs();
    if (pairs == 0) {
        return false;
    }
    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q5_1) {
        return (pairs & GGML_CUDA_SLOWKV_Q8_Q51) != 0;
    }
    if (type_K != type_V) {
        return false;
    }
    switch (type_K) {
        case GGML_TYPE_Q8_0:     return (pairs & GGML_CUDA_SLOWKV_Q8_Q8)   != 0;
        case GGML_TYPE_TURBO4_0: return (pairs & GGML_CUDA_SLOWKV_T4_T4)   != 0;
        case GGML_TYPE_TQ6_0:    return (pairs & GGML_CUDA_SLOWKV_TQ6_TQ6) != 0;
        default:                 return false;
    }
}

// Common SLOWKV preconditions: an enabled pair on a cc 8.6 CUDA device with D256 for Q/K/V.
static bool ggml_cuda_fattn_slowkv_base(const ggml_tensor * dst, const int device) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    if (!ggml_cuda_fattn_slowkv_pair(K->type, V->type)) {
        return false;
    }
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(device);
    return false;
#else
    return ggml_cuda_info().devices[device].cc == 860 && Q->ne[0] == 256 && K->ne[0] == 256 && V->ne[0] == 256;
#endif
}

// The ncols2 switch's use_gqa_opt test (mask present, no ALiBi, KV a multiple of FATTN_KQ_STRIDE, 16-byte strides on
// the non-quantized tensors). With GQA > 4 it sends a call to the ncols1 == 8 dispatch.
static bool ggml_cuda_fattn_slowkv_gqa_opt(const ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    if (mask == nullptr) {
        return false;
    }
    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    if (max_bias != 0.0f || K->ne[1] % FATTN_KQ_STRIDE != 0) {
        return false;
    }
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                return false;
            }
        }
    }
    return true;
}

// True when a SLOWKV pair's call takes the compact tile. The fused gate uses it to admit width 5 for turbo4/turbo4 and
// tq6_0/tq6_0, and the ncols1 == 8 dispatch below uses it to pick the tile, so the two cannot disagree. It repeats the
// ncols2 switch's use_gqa_opt test so an admitted call always reaches the ncols1 == 8 dispatch (GQA 6 > 4).
static bool ggml_cuda_fattn_slowkv_compact_applies(const ggml_tensor * dst, const int device) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    if (!ggml_cuda_fattn_slowkv_base(dst, device)) {
        return false;
    }
    if (Q->ne[1] != 5 || Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1 || Q->ne[2] != 6*K->ne[2]) {
        return false;
    }
    if (mask == nullptr || mask->ne[1] < 5 || mask->ne[2] != 1 || mask->ne[3] != 1) {
        return false;
    }
    return ggml_cuda_fattn_slowkv_gqa_opt(dst);
}

// True when a turbo4/turbo4 or tq6_0/tq6_0 call of width 5-8 takes the fused (8,8) tile: the switch is on for the pair,
// the call reaches the ncols1 == 8 dispatch (use_gqa_opt, GQA > 4) and the compact tile does not apply. The fused gate
// admits widths 5-8 for these pairs on this predicate only, and the dispatch picks (8,8) for every admitted width > 4
// that is not compact, so gate and tile cannot disagree.
static bool ggml_cuda_fattn_slowkv_wide_applies(const ggml_tensor * dst, const int device) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    if (K->type != V->type || (K->type != GGML_TYPE_TURBO4_0 && K->type != GGML_TYPE_TQ6_0)) {
        return false;
    }
    if (Q->ne[1] < 5 || Q->ne[1] > 8 || !ggml_cuda_fattn_slowkv_base(dst, device)) {
        return false;
    }
    if (Q->ne[2] % K->ne[2] != 0 || Q->ne[2] / K->ne[2] <= 4 || !ggml_cuda_fattn_slowkv_gqa_opt(dst)) {
        return false;
    }
    return !ggml_cuda_fattn_slowkv_compact_applies(dst, device);
}

// True when a q8_0 K / q5_1 V call of width 1-8 takes the fused tiles (q5_1 V loader). The tile follows the usual
// GQA-packed dispatch: (2,8)/(4,8) up to width 4, compact at width 5 when compact_applies, (8,8) above.
static bool ggml_cuda_fattn_slowkv_q5_1_applies(const ggml_tensor * dst, const int device) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    return K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q5_1 && Q->ne[1] >= 1 && Q->ne[1] <= 8 &&
        ggml_cuda_fattn_slowkv_base(dst, device);
}

template <int DKQ, int DV, ggml_type type_K, ggml_type type_V>
static void ggml_cuda_flash_attn_ext_mma_turbo_dispatch_ncols1_8(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    // Pack five queries and six heads into the 32-column tile.
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if constexpr (DKQ == 256 && DV == 256 && ((type_K == type_V &&
                  (type_K == GGML_TYPE_Q8_0 || type_K == GGML_TYPE_TURBO4_0 || type_K == GGML_TYPE_TQ6_0)) ||
                  (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q5_1))) {
        if (ggml_cuda_fattn_slowkv_compact_applies(dst, ctx.device)) {
            ggml_cuda_fattn_path_note("q5g6_compact32", dst, 8);
            ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 4, 8, type_K, type_V, true>(ctx, dst);
            return;
        }
    }
    // SLOWKV widths 5-8 of turbo4/turbo4 and tq6_0/tq6_0: the fused gate admits them only when wide_applies holds
    if constexpr (DKQ == 256 && DV == 256 && type_K == type_V && (type_K == GGML_TYPE_TURBO4_0 || type_K == GGML_TYPE_TQ6_0)) {
        if (Q->ne[1] > 4) {
            ggml_cuda_fattn_path_note("slowkv_wide88", dst, 8);
            ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 8, 8, type_K, type_V>(ctx, dst);
            return;
        }
    }
    if constexpr (DKQ == 256 && DV == 256 && type_K == GGML_TYPE_TQ5_0 && type_V == GGML_TYPE_TURBO4_0) {
        static const bool compact = [] {
            const char * e = getenv("GGML_VL_AT_Q5G6");
            if (e == nullptr) {
                e = getenv("GGML_CUDA_FATTN_Q5G6_COMPACT");
            }
            // default ON since the M1 gate (2026-10-07, +2.33% G at 100K); =0 restores the generic tile
            return e == nullptr || strcmp(e, "0") != 0;
        }();
        const ggml_tensor * K = dst->src[1];
        const ggml_tensor * mask = dst->src[3];
        if (compact && ggml_cuda_info().devices[ctx.device].cc == 860 &&
                Q->ne[1] == 5 && Q->ne[3] == 1 && Q->ne[2] == 6*K->ne[2] &&
                K->ne[3] == 1 && dst->src[2]->ne[3] == 1 &&
                mask != nullptr && mask->ne[1] >= 5 && mask->ne[2] == 1 && mask->ne[3] == 1) {
            ggml_cuda_fattn_path_note("q5g6_compact32", dst, 8);
            ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 4, 8, type_K, type_V, true>(ctx, dst);
            return;
        }
    }
#endif
    // Other widths, types, and layouts keep the existing tile dispatch.
    // GGML_Q8_TURBO3_MMA_NCOLS1_MIN pads single queries into the (2,8) instance. Default 2 since P5b:
    // the (1,8) instance runs 84 blocks at 4% occupancy (883 us at 100K under ncu) while the padded
    // (2,8) route runs 252 blocks (390 vs 700 us/launch in test-backend-ops perf). Set =1 to disable.
    static const int ncols1_min = [] { const char * e = getenv("GGML_Q8_TURBO3_MMA_NCOLS1_MIN"); const int v = e ? atoi(e) : 2; return (v == 1 || v == 2 || v == 4) ? v : 2; }();
    if (Q->ne[1] <= 1 && ncols1_min == 1) { ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 1, 8, type_K, type_V>(ctx, dst); return; }
    if (Q->ne[1] <= 2 && ncols1_min <= 2) { ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 2, 8, type_K, type_V>(ctx, dst); return; }
    if constexpr (DKQ == 256 && DV == 256 && ((type_K == GGML_TYPE_Q8_0 && (type_V == GGML_TYPE_TURBO3_0 || type_V == GGML_TYPE_Q8_0)) ||
                                             (type_K == GGML_TYPE_TQ6_0 && type_V == GGML_TYPE_TURBO3_0) ||
                                             (type_K == GGML_TYPE_TQ5_0 && type_V == GGML_TYPE_TURBO3_0) ||
                                             ((type_K == GGML_TYPE_TQ5_0 || type_K == GGML_TYPE_TQ6_0 || type_K == GGML_TYPE_Q8_0) && type_V == GGML_TYPE_TURBO4_0) ||
                                             ((type_K == GGML_TYPE_TQ6_0 || type_K == GGML_TYPE_TQ5_0) && type_V == GGML_TYPE_TQ5_0) ||
                                             (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q5_1))) {
        if (Q->ne[1] > 4) {
            ggml_cuda_fattn_path_note("turbo_fused_8x8", dst, 8);
            ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 8, 8, type_K, type_V>(ctx, dst);
            return;
        }
    }
    ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 4, 8, type_K, type_V>(ctx, dst); // Q->ne[1] in {3,4}
}
template <int DKQ, int DV, ggml_type type_K, ggml_type type_V>
static void ggml_cuda_flash_attn_ext_mma_turbo_dispatch_ncols1_4(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0]; // ncols2 == 4: (2,4),(4,4)
    if (Q->ne[1] <= 2) { ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 2, 4, type_K, type_V>(ctx, dst); return; }
    ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 4, 4, type_K, type_V>(ctx, dst); // Q->ne[1] in {3,4}
}

static bool ggml_cuda_flash_attn_ext_mma_turbo_use_gqa(const ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Mirror the f16 use_gqa_opt computation. Quantized tensors are skipped in the nb%16 loop.
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    return use_gqa_opt;
}

template <int DKQ, int DV, ggml_type type_K, ggml_type type_V>
static void ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const bool use_gqa_opt = ggml_cuda_flash_attn_ext_mma_turbo_use_gqa(dst);

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    if (use_gqa_opt && gqa_ratio > 4) {                                  // ncols2 = 8
        ggml_cuda_flash_attn_ext_mma_turbo_dispatch_ncols1_8<DKQ, DV, type_K, type_V>(ctx, dst);
        return;
    }
    if (use_gqa_opt && gqa_ratio > 2) {                                  // ncols2 = 4
        ggml_cuda_flash_attn_ext_mma_turbo_dispatch_ncols1_4<DKQ, DV, type_K, type_V>(ctx, dst);
        return;
    }
    if (use_gqa_opt && gqa_ratio > 1) {                                  // ncols2 = 2 -> (4,2)
        ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 4, 2, type_K, type_V>(ctx, dst);
        return;
    }
    ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 8, 1, type_K, type_V>(ctx, dst); // ncols2 = 1 -> (8,1)
}

// Env latch for the fused turbo MMA decode path. DEFAULT ON.
//
// The MMA path is correctness-validated (coherent output, KLD == VEC baseline 0.008396)
// and faster than VEC at every depth (beats rival "buun"), BUT it is NOT bit/token-identical
// to the VEC reference: MMA and VEC accumulate the P·V (VKQ) reduction in f16 with different
// reduction trees (tensor-core fragment order vs per-thread VEC order), so a near-tie greedy
// token can flip (~1 in ~25 tokens on a hard tie). This is the same irreducible f16-order
// difference that exists between the base f16-MMA and f16-VEC kernels — not a regression — but
// it fails strict token-identity. GGML_TURBO_MMA_FUSED=0 is the VEC kill-switch for anyone who
// needs that identity guarantee back.
static bool ggml_cuda_turbo_mma_fused() {
    static const bool v = []{
        const char * s = getenv("GGML_TURBO_MMA_FUSED");
        return !(s && s[0] == '0');  // default ON (faster GQA-packed MMA, quality-neutral); GGML_TURBO_MMA_FUSED=0 = VEC kill-switch
    }();
    return v;
}

// Fused Q8_0-K / TURBO3-V MMA path. Default ON since P5b (2026-09-09): D7/D7b census and the
// P5a temp-1.0 ABBA showed it beats the vector and generic-MMA routes at every verify width
// 1..5 for D=256. Set GGML_Q8_TURBO3_MMA_FUSED=0 to fall back to the pre-P5b routing.
static bool ggml_cuda_q8_turbo3_mma_fused() {
    static const bool value = [] {
        const char * env = getenv("GGML_Q8_TURBO3_MMA_FUSED");
        return env == nullptr || env[0] != '0';
    }();
    return value;
}

// smallest query width routed to the q8_0/turbo3 MMA path. Default 1 since P5b: width-1
// fused MMA is 797 us vs 1418 us for the vector kernel at 100K (D7b), width-2 434 us vs the
// generic MMA + f16 temporaries. GGML_Q8_TURBO3_MMA_MIN_Q=3 restores the pre-P5b routing.
static int ggml_cuda_q8_turbo3_mma_min_q() {
    static const int value = [] {
        const char * env = getenv("GGML_Q8_TURBO3_MMA_MIN_Q");
        const int v = env ? atoi(env) : 1;
        return (v >= 1 && v <= 5) ? v : 1;
    }();
    return value;
}

// largest query width routed to the fused q8_0-K MMA paths. Default 8: the (8,8) instance is a full eight-row
// tile, so widths 6..8 (MTP depth 5..7, n-gram drafts up to 7, DFlash block_size 8 verify) use it instead of
// falling to the generic mma_f16 dequant path (kv=100352 width 8: 1434 -> 429 us per layer, widths 1..5 unchanged,
// test-backend-ops 1158/1158). GGML_Q8_TURBO3_MMA_MAX_Q=5 restores the previous routing (valid 5..8).
static int ggml_cuda_q8_turbo3_mma_max_q() {
    static const int value = [] {
        const char * env = getenv("GGML_Q8_TURBO3_MMA_MAX_Q");
        const int v = env ? atoi(env) : 8;
        return (v >= 5 && v <= 8) ? v : 8;
    }();
    return value;
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst, bool preserve_cand = false) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst, preserve_cand);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst, preserve_cand);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst, preserve_cand);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst, preserve_cand);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst, preserve_cand);
            break;
        case 192: {
            // MiMo-V2.5 / V2.5-Pro / V2-Flash: gqa_ratio is 8 (SWA) or 16 (full attn)
            GGML_ASSERT(V->ne[0] == 128);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));
            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);
            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128, 16>(ctx, dst, preserve_cand);
            } else {
                GGML_ASSERT(gqa_ratio % 8 == 0);
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128,  8>(ctx, dst, preserve_cand);
            }
        } break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst, preserve_cand);
            break;
        case 320:
            // For Mistral Small 4, go straight to the ncols1 switch (ncols2=32-only build).
            GGML_ASSERT(V->ne[0] == 256);
            {
                float max_bias = 0.0f;
                memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

                const bool use_gqa_opt = mask && max_bias == 0.0f;
                GGML_ASSERT(use_gqa_opt);
                GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
                const int gqa_ratio = Q->ne[2] / K->ne[2];
                GGML_ASSERT(gqa_ratio % 32 == 0);

                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<320, 256, 32>(ctx, dst, preserve_cand);
            }
            break;
        case 512:
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<512, 512>(ctx, dst, preserve_cand);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio == 20 && GGML_CUDA_CC_IS_NVIDIA(cc)) { // GLM 4.7 Flash
                if (cc >= GGML_CUDA_CC_DGX_SPARK) {
                    if (Q->ne[1] <= 8) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst, preserve_cand);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst, preserve_cand);
                    break;
                }
                if (cc >= GGML_CUDA_CC_BLACKWELL) {
                    if (Q->ne[1] <= 4 && K->ne[1] >= 65536) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst, preserve_cand);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst, preserve_cand);
                    break;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 4) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst, preserve_cand);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst, preserve_cand);
                    break;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    if (Q->ne[1] <= 4) {
                        if (K->ne[1] <= 16384) {
                            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst, preserve_cand);
                            break;
                        }
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst, preserve_cand);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst, preserve_cand);
                    break;
                }
                // Volta:
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst, preserve_cand);
            } else if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst, preserve_cand);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst, preserve_cand);
            }
        } break;
        case 640: {
            // Padded turbo KV cache for GLM-4.7 Flash (K head_dim=576 zero-padded to 640).
            // D=640 shared memory (Q storage = ncols*(DKQ/2+4)*4) exceeds hardware limit at ncols1>=4.
            // Cap at ncols1=2 (ncols=32): Q=32*324*4=41KB + KV≈37KB = ~78KB total.
            GGML_ASSERT(V->ne[0] == 512);
            if (Q->ne[1] <= 1) {
                ggml_cuda_flash_attn_ext_mma_f16_case<640, 512, 1, 16>(ctx, dst, preserve_cand);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_case<640, 512, 2, 16>(ctx, dst, preserve_cand);
            }
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

#define FATTN_VEC_CASE(D, type_K_case, type_V_case)                                                                                \
    if constexpr (GGML_CUDA_FA_##type_K_case##_##type_V_case) {                                                                    \
        const bool type_K_okay = type_K == GGML_TYPE_##type_K_case || (type_K == GGML_TYPE_F32 && GGML_TYPE_##type_K_case == GGML_TYPE_F16); \
        const bool type_V_okay = type_V == GGML_TYPE_##type_V_case || (type_V == GGML_TYPE_F32 && GGML_TYPE_##type_V_case == GGML_TYPE_F16); \
        if (head_size == (D) && type_K_okay && type_V_okay) {                                                                      \
            return ggml_cuda_flash_attn_ext_vec_case<D, GGML_TYPE_##type_K_case, GGML_TYPE_##type_V_case>;                         \
        }                                                                                                                          \
    }                                                                                                                              \

#define FATTN_VEC_CASES_ALL_D(type_K_case, type_V_case) \
    FATTN_VEC_CASE( 64, type_K_case, type_V_case)       \
    FATTN_VEC_CASE(128, type_K_case, type_V_case)       \
    FATTN_VEC_CASE(256, type_K_case, type_V_case)       \

typedef void (* fattn_vec_case_t)(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Vector kernel for the given head size and K/V types, nullptr if its template instance was not compiled:
static fattn_vec_case_t ggml_cuda_get_fattn_vec_case(const int64_t head_size, const ggml_type type_K, const ggml_type type_V) {
    FATTN_VEC_CASES_ALL_D(F16,  F16)
    FATTN_VEC_CASES_ALL_D(Q4_0, F16)
    FATTN_VEC_CASES_ALL_D(Q4_1, F16)
    FATTN_VEC_CASES_ALL_D(Q5_0, F16)
    FATTN_VEC_CASES_ALL_D(Q5_1, F16)
    FATTN_VEC_CASES_ALL_D(Q8_0, F16)
    FATTN_VEC_CASES_ALL_D(BF16, F16)

    FATTN_VEC_CASES_ALL_D(F16,  Q4_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q4_0)

    FATTN_VEC_CASES_ALL_D(F16,  Q4_1)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(BF16, Q4_1)

    FATTN_VEC_CASES_ALL_D(F16,  Q5_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q5_0)

    FATTN_VEC_CASES_ALL_D(F16,  Q5_1)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(BF16, Q5_1)

    FATTN_VEC_CASES_ALL_D(F16,  Q8_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q8_0)

    FATTN_VEC_CASES_ALL_D(F16,  BF16)
    FATTN_VEC_CASES_ALL_D(Q4_0, BF16)
    FATTN_VEC_CASES_ALL_D(Q4_1, BF16)
    FATTN_VEC_CASES_ALL_D(Q5_0, BF16)
    FATTN_VEC_CASES_ALL_D(Q5_1, BF16)
    FATTN_VEC_CASES_ALL_D(Q8_0, BF16)
    FATTN_VEC_CASES_ALL_D(BF16, BF16)

    // TurboQuant KV cache types (fork-only, always compiled - see ggml_cuda_fattn_vec_instances)
    FATTN_VEC_CASES_ALL_D(TURBO3_0, TURBO3_0)
    FATTN_VEC_CASES_ALL_D(TURBO3_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q8_0,     TURBO3_0)
    FATTN_VEC_CASES_ALL_D(F16,      TURBO3_0)
    FATTN_VEC_CASES_ALL_D(TURBO3_0, F16)

    FATTN_VEC_CASES_ALL_D(TURBO2_0, TURBO2_0)
    FATTN_VEC_CASES_ALL_D(TURBO2_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q8_0,     TURBO2_0)
    FATTN_VEC_CASES_ALL_D(F16,      TURBO2_0)
    FATTN_VEC_CASES_ALL_D(TURBO2_0, F16)
    FATTN_VEC_CASES_ALL_D(TURBO3_0, TURBO2_0)
    FATTN_VEC_CASES_ALL_D(TURBO2_0, TURBO3_0)

    FATTN_VEC_CASES_ALL_D(TURBO4_0, TURBO4_0)
    FATTN_VEC_CASES_ALL_D(TURBO4_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q8_0,     TURBO4_0)
    FATTN_VEC_CASES_ALL_D(F16,      TURBO4_0)
    FATTN_VEC_CASES_ALL_D(TURBO4_0, F16)
    FATTN_VEC_CASES_ALL_D(TURBO4_0, TURBO3_0)
    FATTN_VEC_CASES_ALL_D(TURBO3_0, TURBO4_0)
    FATTN_VEC_CASES_ALL_D(TURBO4_0, TURBO2_0)
    FATTN_VEC_CASES_ALL_D(TURBO2_0, TURBO4_0)

    // tq6 blocks hold 128 values: head dim 64 has no instance (see the cmake FA_TQ6_COMBINATIONS list)
    FATTN_VEC_CASE(128, TQ6_0,    TQ6_0)
    FATTN_VEC_CASE(256, TQ6_0,    TQ6_0)
    FATTN_VEC_CASE(128, TQ6_0,    TURBO3_0)
    FATTN_VEC_CASE(256, TQ6_0,    TURBO3_0)
    FATTN_VEC_CASE(128, TQ6_0,    Q8_0)
    FATTN_VEC_CASE(256, TQ6_0,    Q8_0)
    FATTN_VEC_CASE(128, Q8_0,     TQ6_0)
    FATTN_VEC_CASE(256, Q8_0,     TQ6_0)
    FATTN_VEC_CASE(128, TQ6_0,    F16)
    FATTN_VEC_CASE(256, TQ6_0,    F16)
    FATTN_VEC_CASE(128, F16,      TQ6_0)
    FATTN_VEC_CASE(256, F16,      TQ6_0)

    // tq5 shares the tq6 geometry (128 values per block)
    FATTN_VEC_CASE(128, TQ5_0,    TQ5_0)
    FATTN_VEC_CASE(256, TQ5_0,    TQ5_0)
    FATTN_VEC_CASE(128, TQ5_0,    TURBO3_0)
    FATTN_VEC_CASE(256, TQ5_0,    TURBO3_0)
    FATTN_VEC_CASE(128, TQ5_0,    Q8_0)
    FATTN_VEC_CASE(256, TQ5_0,    Q8_0)
    FATTN_VEC_CASE(128, Q8_0,     TQ5_0)
    FATTN_VEC_CASE(256, Q8_0,     TQ5_0)
    FATTN_VEC_CASE(128, TQ5_0,    F16)
    FATTN_VEC_CASE(256, TQ5_0,    F16)
    FATTN_VEC_CASE(128, F16,      TQ5_0)
    FATTN_VEC_CASE(256, F16,      TQ5_0)

    return nullptr;
}

#define FATTN_VEC_CASE_D512(type_K, type_V)                                                                          \
    {                                                                                                                \
        const bool type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16);   \
        const bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16);   \
        if (Q->ne[0] == 512 && type_K_okay && type_V_okay) {                                                        \
            ggml_cuda_flash_attn_ext_vec_case_d512<type_K, type_V>(ctx, dst);                                       \
            return;                                                                                                  \
        }                                                                                                            \
    }                                                                                                                \

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

#ifdef GGML_USE_HIP
    FATTN_VEC_CASE_D512(GGML_TYPE_Q8_0, GGML_TYPE_F16)
    FATTN_VEC_CASE_D512(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE_D512(GGML_TYPE_Q8_0, GGML_TYPE_BF16)
    FATTN_VEC_CASE_D512(GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASE_D512(GGML_TYPE_Q8_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASE_D512(GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0)
#endif

    fattn_vec_case_t vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type);
    if (vec_case == nullptr) {
        static bool warned = false;
        if (!warned) {
            GGML_LOG_WARN("%s: no FlashAttention vector kernel compiled for K/V types %s-%s, converting K and V to f16 instead (slow). "
                "Add \"%s-%s\" to GGML_CUDA_FA_QUANTS to compile it.\n",
                __func__, ggml_type_name(K->type), ggml_type_name(V->type), ggml_type_name(K->type), ggml_type_name(V->type));
            warned = true;
        }
        vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], GGML_TYPE_F16, GGML_TYPE_F16);
    }
    GGML_ASSERT(vec_case != nullptr);
    vec_case(ctx, dst);
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE    =   0,
    BEST_FATTN_KERNEL_TILE    = 200,
    BEST_FATTN_KERNEL_VEC     = 100,
    BEST_FATTN_KERNEL_MMA_F16 = 400,
};

// K/V types for which there is a vector kernel template instance, other kernels convert these to f16:
static bool ggml_cuda_fattn_kv_type_supported(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return true;
        case GGML_TYPE_TURBO2_0:
        case GGML_TYPE_TURBO3_0:
        case GGML_TYPE_TURBO4_0:
            // turbo KV types; head-dim geometry is validated separately in
            // ggml_cuda_get_best_fattn_kernel (multiples of 64 only)
            return true;
        case GGML_TYPE_TQ6_0:
        case GGML_TYPE_TQ5_0:
            // tq6/tq5 KV types; head dim must be a multiple of 128 (block size), checked below
            return true;
        default:
            return false;
    }
}

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 192:
            if (V->ne[0] != 128 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 8 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 320:
            if (V->ne[0] != 256 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 32 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 640:
#ifdef GGML_USE_HIP
            // The D=640 tile kernel exceeds HIP's local memory limit and is not compiled:
            // at ncols=32 the fp16 path needs 67584 B against a 65536 B limit. D=576 needs
            // 63488 B at the same config and is compiled, so it is handled above.
            return BEST_FATTN_KERNEL_NONE;
#endif
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    if (!ggml_cuda_fattn_kv_type_supported(K->type) || !ggml_cuda_fattn_kv_type_supported(V->type)) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // turbo VEC/MMA kernels are instantiated for head dims that are multiples of 64
    {
        auto is_turbo = [](ggml_type t) {
            return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0;
        };
        if ((is_turbo(K->type) && K->ne[0] % 64 != 0) ||
            (is_turbo(V->type) && V->ne[0] % 64 != 0)) {
            return BEST_FATTN_KERNEL_NONE;
        }
        // tq6 packs 128 values per block and is only instantiated for head dims 128 and 256
        if (((K->type == GGML_TYPE_TQ6_0 || K->type == GGML_TYPE_TQ5_0) && K->ne[0] % 128 != 0) ||
            ((V->type == GGML_TYPE_TQ6_0 || V->type == GGML_TYPE_TQ5_0) && V->ne[0] % 128 != 0)) {
            return BEST_FATTN_KERNEL_NONE;
        }
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    // 192 satisfies % 64 == 0 but has no vec instance (DKQ != DV); force it onto the MMA path.
#ifdef GGML_USE_HIP
    // D=512 VEC is decode-only (ncols=1) with K=q8_0; turbo K types are register-unsafe at D=512.
    const bool d512_v_supported = V->type == GGML_TYPE_F16 || V->type == GGML_TYPE_Q8_0 ||
        V->type == GGML_TYPE_BF16 || V->type == GGML_TYPE_TURBO2_0 ||
        V->type == GGML_TYPE_TURBO3_0 || V->type == GGML_TYPE_TURBO4_0;
    const bool d512_vec_safe = Q->ne[0] == 512 && Q->ne[1] == 1 &&
        K->type == GGML_TYPE_Q8_0 && d512_v_supported;
    const bool can_use_vector_kernel = (Q->ne[0] <= 256 || d512_vec_safe)
                                    && Q->ne[0] % 64 == 0
                                    && Q->ne[0] != 192
                                    && K->ne[1] % FATTN_KQ_STRIDE == 0;
#else
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0;
#endif

#ifdef GGML_USE_HIP
    // Prefer inline-dequant VEC for small quantized-KV batches. D=512 is limited
    // to the decode-only q8_0 K instances above; larger batches use TILE/MMA.
    if ((ggml_is_quantized(K->type) || ggml_is_quantized(V->type)) && can_use_vector_kernel && Q->ne[1] <= 8) {
        return BEST_FATTN_KERNEL_VEC;
    }
#endif // GGML_USE_HIP

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                // the sparse gather exists only in the MMA kernel: (DKQ, DV, 1, 8) with GQA > 4
                const bool sparse_decode = gqa_opt_applies && gqa_ratio > 4 &&
                    ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(K->ne[0], V->ne[0], 1, 8) &&
                    ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1, 8);
                if (!sparse_decode && cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 &&
                        !(gqa_ratio > 4 && (Q->ne[0] >= 256 || K->ne[1] >= 8192))) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 2) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                } else {
                    if (Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    const int ncols2_max = Q->ne[0] == 320 ? 32 : ((Q->ne[0] == 576 || Q->ne[0] == 640 || Q->ne[0] == 192) ? 16 : 8);
    int gqa_ratio_eff = 1;
    while (max_bias == 0.0f && gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
        gqa_ratio_eff *= 2;
    }

    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 16) {
            return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // TQ: RDNA4 fast path for TurboQuant cache types — prefer VEC for quantized K/V at small q-cols
    if (amd_wmma_available(cc) && GGML_CUDA_CC_IS_RDNA4(cc) && gqa_opt_applies && Q->ne[0] <= 128 && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                if (Q->ne[1] == 1) {
                    if (!gqa_opt_applies) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            } else {
                if (Q->ne[1] <= 2) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        }
        int gqa_ratio_eff_rdna4 = 1;
        const int ncols2_max_rdna4 = (Q->ne[0] == 576 || Q->ne[0] == 640) ? 16 : 8;
        while (gqa_ratio % (2*gqa_ratio_eff_rdna4) == 0 && gqa_ratio_eff_rdna4 < ncols2_max_rdna4) {
            gqa_ratio_eff_rdna4 *= 2;
        }
        if (Q->ne[1] * gqa_ratio_eff_rdna4 <= 8) {
            return BEST_FATTN_KERNEL_TILE;
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // AMD MFMA needs a certain minimum batch size to outscale the tile kernel for large head sizes.
    if (amd_mfma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if ((Q->ne[0] <= 64 && Q->ne[1] * gqa_ratio_eff > 8)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 128 && Q->ne[1] * gqa_ratio_eff > 16)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 256 && Q->ne[1] * gqa_ratio_eff > 64)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if (Q->ne[0] > 256 && gqa_opt_applies && Q->ne[1] * gqa_ratio_eff > 128) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
    }

    // AMD WMMA is faster than the tile kernel if the wide tiles with high arithmetic intensity can be utilized.
    if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72 &&
            Q->ne[1] * gqa_ratio_eff > (Q->ne[0] <= 128 ? 8 : 16)) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

// The fused packed-KV routes (q8_0 K over turbo3 / q8_0 V, and the turbo / tq gate) run before the generic selector
// and read K/V in place: launch_fattn gets need_f16_K = need_f16_V = false. This one function decides the route for
// execution (ctx != nullptr: launch) and for ggml_cuda_flash_attn_ext_get_alloc_size (ctx == nullptr: only report),
// so the scratch reserved behind dst follows the kernel that runs (#64). Returns true when a fused route applies.
static bool ggml_cuda_flash_attn_ext_fused(ggml_backend_cuda_context * ctx, ggml_tensor * dst, const int device) {
#define FATTN_FUSED_NOTE(...) do { if (ctx != nullptr) { ggml_cuda_fattn_path_note(__VA_ARGS__); } } while (0)
#define FATTN_FUSED_LAUNCH(DKQ_, DV_, TK_, TV_) \
    do { if (ctx != nullptr) { ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<DKQ_, DV_, TK_, TV_>(*ctx, dst); } return true; } while (0)

    // KVarN region attention: its q8_0 / tq6_0 staging rows would otherwise match the q8/q8 and turbo gates below
    if (dst->src[6] != nullptr) {
        return false;
    }

    // Qwen3.8 verification fast path: Q8 K and Turbo3 V are decoded directly
    // into the Stream-K MMA tile. This removes full-cache FP16 conversion and
    // shares each compressed tile across the packed MTP query rows.
    {
        const ggml_tensor * Q = dst->src[0];
        const ggml_tensor * K = dst->src[1];
        const ggml_tensor * V = dst->src[2];
        const int cc = ggml_cuda_info().devices[device].cc;
        if (ggml_cuda_q8_turbo3_mma_fused() && K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_TURBO3_0 &&
                Q->ne[0] == 256 && V->ne[0] == 256 && Q->ne[1] >= ggml_cuda_q8_turbo3_mma_min_q() && Q->ne[1] <= ggml_cuda_q8_turbo3_mma_max_q() && turing_mma_available(cc)) {
            FATTN_FUSED_NOTE("q8_turbo3_fused", dst, -1);
            FATTN_FUSED_LAUNCH(256, 256, GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0);
        }
        // Same fused path for a q8_0 K / q8_0 V cache (e.g. the MTP draft cache): identical staging and
        // K decode, V decoded with the q8_0 tile loader instead of turbo3. Same routing knobs.
        if (ggml_cuda_q8_turbo3_mma_fused() && K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0 &&
                Q->ne[0] == 256 && V->ne[0] == 256 && Q->ne[1] >= ggml_cuda_q8_turbo3_mma_min_q() && Q->ne[1] <= ggml_cuda_q8_turbo3_mma_max_q() && turing_mma_available(cc)) {
            FATTN_FUSED_NOTE("q8_q8_fused", dst, -1);
            FATTN_FUSED_LAUNCH(256, 256, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);
        }
    }

    // Fused turbo MMA decode gate (DEFAULT ON, see ggml_cuda_turbo_mma_fused; GGML_TURBO_MMA_FUSED=0 disables).
    // Routes turbo2/3/4 K==V, D in {128,256}, decode (Q->ne[1] <= 4) onto the GQA-packed
    // MMA path (KV read once per head-group instead of per query head). Q is ALREADY
    // graph-rotated (src/llama-graph.cpp) and the FA output is inverse-rotated there, so this
    // path does NO inline FWHT and NO src swap. GGML_TURBO_MMA_FUSED=0 falls straight through
    // to the original VEC dispatch (kill-switch).
    {
        const ggml_tensor * Q = dst->src[0];
        const ggml_tensor * K = dst->src[1];
        const ggml_tensor * V = dst->src[2];
        const int cc = ggml_cuda_info().devices[device].cc;
        const bool turbo_matched = (K->type == V->type &&
            (K->type == GGML_TYPE_TURBO4_0 || K->type == GGML_TYPE_TURBO3_0 || K->type == GGML_TYPE_TURBO2_0 ||
             K->type == GGML_TYPE_TQ6_0 || K->type == GGML_TYPE_TQ5_0)) ||
            // asymmetric tq6/tq5 K over a turbo3 V: the pair they are meant for, K and V decoded by
            // their own tile loaders into the same GQA-packed tile.
            ((K->type == GGML_TYPE_TQ6_0 || K->type == GGML_TYPE_TQ5_0) && V->type == GGML_TYPE_TURBO3_0) ||
            // turbo4 V under a q8_0/tq6_0/tq5_0 K (D=256 only): unstaged turbo4 V tile loader, K loader as above
            ((K->type == GGML_TYPE_TQ6_0 || K->type == GGML_TYPE_TQ5_0 || K->type == GGML_TYPE_Q8_0) && V->type == GGML_TYPE_TURBO4_0 && Q->ne[0] == 256) ||
            // tq6_0 K over a tq5_0 V (D=256 only): both tiles staged, same loaders as the matched tq6/tq5 pairs
            (K->type == GGML_TYPE_TQ6_0 && V->type == GGML_TYPE_TQ5_0 && Q->ne[0] == 256);
        // the tq6_0/tq5_0 K / turbo3_0 V pairs at D=256 have an (8,8) instance, so MTP verify widths 5..8 stay fused.
        // FA55: matched tq5_0/tq5_0 at D=256 takes the compiled (8,8) route when the GQA-packed dispatch picks ncols2 = 8
        // (gqa > 4); other layouts keep their existing fallback.
        const bool tq5_wide = K->type == GGML_TYPE_TQ5_0 && V->type == GGML_TYPE_TQ5_0 && Q->ne[0] == 256 &&
            Q->ne[2] / K->ne[2] > 4 && ggml_cuda_flash_attn_ext_mma_turbo_use_gqa(dst);
        int turbo_max_q = (((K->type == GGML_TYPE_TQ6_0 || K->type == GGML_TYPE_TQ5_0 || K->type == GGML_TYPE_Q8_0) &&
                                  (V->type == GGML_TYPE_TURBO3_0 || V->type == GGML_TYPE_TURBO4_0) && Q->ne[0] == 256) ||
                                 (K->type == GGML_TYPE_TQ6_0 && V->type == GGML_TYPE_TQ5_0 && Q->ne[0] == 256) || tq5_wide) ? 8 : 4;
        // SLOWKV (GGML_SLOWKV, default off): widths 5-8 of turbo4/turbo4 and tq6_0/tq6_0 take the fused compact (4,8)
        // tile (width 5) or the fused (8,8) tile instead of the generic f16 route; only when the ncols1 == 8 dispatch
        // will pick one of those tiles.
        if (turbo_max_q == 4 && Q->ne[1] > 4 && Q->ne[1] <= 8 &&
                (ggml_cuda_fattn_slowkv_compact_applies(dst, device) || ggml_cuda_fattn_slowkv_wide_applies(dst, device))) {
            turbo_max_q = (int) Q->ne[1];
        }
        // SLOWKV q8_0 K / q5_1 V: fused at widths 1-8 with the q5_1 V tile loader (generic route without the switch)
        if (ggml_cuda_turbo_mma_fused() && turing_mma_available(cc) && ggml_cuda_fattn_slowkv_q5_1_applies(dst, device)) {
            FATTN_FUSED_NOTE("slowkv_q5_1", dst, -1);
            FATTN_FUSED_LAUNCH(256, 256, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1);
        }
        if (ggml_cuda_turbo_mma_fused() && turbo_matched
                && Q->ne[1] <= turbo_max_q && V->ne[0] == Q->ne[0] && turing_mma_available(cc)) {
            FATTN_FUSED_NOTE("turbo_fused_gate", dst, -1);
            if (K->type == GGML_TYPE_TQ6_0 && V->type == GGML_TYPE_TURBO3_0) {
                if (Q->ne[0] == 128) { FATTN_FUSED_LAUNCH(128, 128, GGML_TYPE_TQ6_0, GGML_TYPE_TURBO3_0); }
                if (Q->ne[0] == 256) { FATTN_FUSED_LAUNCH(256, 256, GGML_TYPE_TQ6_0, GGML_TYPE_TURBO3_0); }
            }
            if (V->type == GGML_TYPE_TURBO4_0 && Q->ne[0] == 256) {
                if (K->type == GGML_TYPE_TQ5_0) { FATTN_FUSED_LAUNCH(256, 256, GGML_TYPE_TQ5_0, GGML_TYPE_TURBO4_0); }
                if (K->type == GGML_TYPE_TQ6_0) { FATTN_FUSED_LAUNCH(256, 256, GGML_TYPE_TQ6_0, GGML_TYPE_TURBO4_0); }
                if (K->type == GGML_TYPE_Q8_0)  { FATTN_FUSED_LAUNCH(256, 256, GGML_TYPE_Q8_0,  GGML_TYPE_TURBO4_0); }
            }
            if (K->type == GGML_TYPE_TQ6_0 && V->type == GGML_TYPE_TQ5_0 && Q->ne[0] == 256) {
                FATTN_FUSED_LAUNCH(256, 256, GGML_TYPE_TQ6_0, GGML_TYPE_TQ5_0);
            }
            if (K->type == GGML_TYPE_TQ5_0 && V->type == GGML_TYPE_TURBO3_0) {
                if (Q->ne[0] == 128) { FATTN_FUSED_LAUNCH(128, 128, GGML_TYPE_TQ5_0, GGML_TYPE_TURBO3_0); }
                if (Q->ne[0] == 256) { FATTN_FUSED_LAUNCH(256, 256, GGML_TYPE_TQ5_0, GGML_TYPE_TURBO3_0); }
            }
            if (Q->ne[0] == 128) {
                switch (K->type) {
                    case GGML_TYPE_TURBO4_0: FATTN_FUSED_LAUNCH(128, 128, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0);
                    case GGML_TYPE_TURBO3_0: FATTN_FUSED_LAUNCH(128, 128, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0);
                    case GGML_TYPE_TURBO2_0: FATTN_FUSED_LAUNCH(128, 128, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0);
                    case GGML_TYPE_TQ6_0:    FATTN_FUSED_LAUNCH(128, 128, GGML_TYPE_TQ6_0,    GGML_TYPE_TQ6_0);
                    case GGML_TYPE_TQ5_0:    FATTN_FUSED_LAUNCH(128, 128, GGML_TYPE_TQ5_0,    GGML_TYPE_TQ5_0);
                    default: break;
                }
            }
            if (Q->ne[0] == 256) {
                switch (K->type) {
                    case GGML_TYPE_TURBO4_0: FATTN_FUSED_LAUNCH(256, 256, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0);
                    case GGML_TYPE_TURBO3_0: FATTN_FUSED_LAUNCH(256, 256, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0);
                    case GGML_TYPE_TQ6_0:    FATTN_FUSED_LAUNCH(256, 256, GGML_TYPE_TQ6_0,    GGML_TYPE_TQ6_0);
                    case GGML_TYPE_TQ5_0:    FATTN_FUSED_LAUNCH(256, 256, GGML_TYPE_TQ5_0,    GGML_TYPE_TQ5_0);
                    // turbo2 + head_dim 256: intentionally NO fused case (routes to VEC via
                    // default below). At 2-bit KV the fused path's GQA-pack saving is tiny while the
                    // dequant/no-pipeline overhead is unchanged, so it is neutral on high-BW GPUs and
                    // regresses ~1-2.5% on bandwidth-limited ones (tester @everson: Gemma-12B / RTX
                    // 5060 Ti). VEC == baseline there. turbo2 + hd128 keeps fused (a +6.6..+69% depth
                    // win on dense models); turbo3/turbo4 stay fused at both head dims.
                    default: break;
                }
            }
        }
    }


    return false;
#undef FATTN_FUSED_LAUNCH
#undef FATTN_FUSED_NOTE
}

static const char * ggml_cuda_fattn_kernel_name(const best_fattn_kernel kernel) {
    switch (kernel) {
        case BEST_FATTN_KERNEL_TILE:    return "tile";
        case BEST_FATTN_KERNEL_VEC:     return "vec";
        case BEST_FATTN_KERNEL_MMA_F16: return "mma_f16";
        case BEST_FATTN_KERNEL_NONE:    break;
    }
    return "none";
}

// GGML_CUDA_FATTN_ALLOC_ROUTE=0 restores the generic-selector sizing: f16 K+V copies for TILE/MMA (and VEC without a
// type instance) even when a fused route runs and never touches them. Default 1 (#64).
static bool ggml_cuda_fattn_alloc_route() {
    static const bool v = [] { const char * e = getenv("GGML_CUDA_FATTN_ALLOC_ROUTE"); return !(e && e[0] == '0'); }();
    return v;
}

// GGML_CUDA_FATTN_ALLOC_LOG=1: one line per new (K type, V type, n_q, log2 n_kv, route) with the scratch reserved
// behind dst and what the generic selector would reserve, to size the gap at 32K / 100K (#64).
static bool ggml_cuda_fattn_alloc_log() {
    static const bool v = [] { const char * e = getenv("GGML_CUDA_FATTN_ALLOC_LOG"); return e && e[0] == '1'; }();
    return v;
}

// ---------------------------------------------------------------------------------------------------------------------
// [#22/#29] Bounded f16 prefill. GGML_CUDA_PREFILL_KV_MIB=<MiB> (1..16384) sets the workspace budget; unset or empty
// = the default budget of 256 MiB; 0 or off = off, and off runs none of the code below (the plan returns before reading
// the tensors), so sizing and launch are exactly the #64 tree's. The default: at 256 MiB prefill measured 1.5% slower
// than unbounded (4.8-5.1% at 1 MiB), decode unchanged, output identical, and ATX IQ4_XS tq5_0/turbo4 at ctx 262,144
// fits under 23,552 MiB (bounded-prefill/STATUS.md, 2026-09-25). Shapes whose full f16 K+V copies fit the budget keep
// the unbounded route, so the default changes nothing below that size (4 KV heads at D 256: n_kv <= 65,536).
//
// The generic MMA route converts the whole quantized K and V cache to f16 before the kernel (launch_fattn,
// fattn-common.cuh), and ggml_cuda_flash_attn_ext_get_alloc_size reserves both copies behind dst in the compute buffer:
// 2 * n_kv * n_head_kv * D * 2 bytes, 1 GiB for 4 KV heads at D 256 and n_kv 262,144, reserved at load for the
// worst-case prefill graph. The bounded plan converts one group of whole KV heads at a time into a workspace of
// heads * (f16 K + f16 V + that group's f32 output) and runs the ordinary f16 MMA kernel per group, then scatters the
// group's output rows into dst. The attention math per head is the f16 MMA kernel's; only the grid (fewer heads per
// launch) differs, which can change the stream-k partition and so the f32 reduction order at some shapes.
//
// #29: the workspace never drops below one complete GQA head group (one KV head's f16 K+V plus the outputs of its gqa
// query heads). When the budget is below that floor at the current n_kv, the floor is used (WARN once, with sizes);
// the full-copy path stays only for shapes the plan excludes. Token-axis stripes are a separate item.
//
// One function, ggml_cuda_fattn_bounded_prefill_plan, decides eligibility and sizes for the allocator
// (ggml_cuda_flash_attn_ext_get_alloc_size), the executor (ggml_cuda_flash_attn_ext) and the CUDA graph compatibility
// check (ggml_cuda_flash_attn_ext_bounded_prefill_applies, called from ggml-cuda.cu), so they cannot disagree.
struct ggml_cuda_fattn_bounded_plan {
    int    heads      = 0;     // KV heads per group; 0 = the plan does not apply
    int    n_head_kv  = 0;
    int    gqa        = 0;     // query heads per KV head
    size_t kv_bytes   = 0;     // f16 bytes of one group's K (and, separately, V): heads * n_kv * D * 2
    size_t out_bytes  = 0;     // f32 bytes of one group's output: heads * gqa * n_q * D * 4
    size_t offset     = 0;     // workspace offset behind dst->data: ggml_nbytes(dst) padded to 128
    size_t workspace  = 0;     // 2 * kv_bytes + out_bytes
    size_t reserve    = 0;     // bytes reserved behind dst: max(workspace, budget), see below
    size_t budget     = 0;     // GGML_CUDA_PREFILL_KV_MIB in bytes (256 MiB when unset)
    size_t floor_bytes = 0;     // one KV head's f16 K+V plus its GQA group's output at this n_kv
    size_t full       = 0;     // the full f16 K+V copies the unbounded route reserves
    bool   floor_used = false; // budget < floor: one head per group
};

static size_t ggml_cuda_fattn_prefill_budget() {
    static const size_t budget = [] {
        constexpr size_t default_mib = 256;
        const char * s = getenv("GGML_CUDA_PREFILL_KV_MIB");
        if (s == nullptr || *s == '\0') {
            return default_mib << 20;
        }
        if (strcmp(s, "off") == 0) {
            return size_t(0);
        }
        size_t n = 0;
        for (const char * c = s; *c; ++c) {
            if (*c < '0' || *c > '9' || n > 16384) {
                GGML_LOG_WARN("fattn bounded prefill: GGML_CUDA_PREFILL_KV_MIB=\"%s\" is not an integer in 0..16384; "
                              "bounded prefill stays off\n", s);
                return size_t(0);
            }
            n = n*10 + size_t(*c - '0');
        }
        if (n > 16384) {
            GGML_LOG_WARN("fattn bounded prefill: GGML_CUDA_PREFILL_KV_MIB=%zu is above 16384; bounded prefill stays off\n", n);
            return size_t(0);
        }
        return n << 20;
    }();
    return budget;
}

static ggml_cuda_fattn_bounded_plan ggml_cuda_fattn_bounded_prefill_plan(const int device, const ggml_tensor * dst) {
    ggml_cuda_fattn_bounded_plan p;

    const size_t budget = ggml_cuda_fattn_prefill_budget();
    if (budget == 0) {
        return p;
    }
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    if (Q == nullptr || K == nullptr || V == nullptr) {
        return p;
    }
    // any further source (hybrid / KVarN descriptors and the like) is not an ordinary quantized cache: excluded
    for (int i = 5; i < GGML_MAX_SRC; ++i) {
        if (dst->src[i] != nullptr) {
            return p;
        }
    }

    // Ampere only (the measured target; Ada and newer and every AMD device keep the #64 route)
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_AMPERE || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
        return p;
    }

    // D 256 only, batch 1, widths above 8: decode and MTP / n-gram verify widths 1-8 never take the plan (they keep
    // their routes and their CUDA graphs); every wider batch that reaches the f16 MMA kernel does, so that no runtime
    // shape smaller than the reserved worst case falls back to the full copy (see `reserve`)
    if (Q->type != GGML_TYPE_F32 || Q->nb[0] != sizeof(float) || Q->ne[0] != 256 || K->ne[0] != 256 || V->ne[0] != 256) {
        return p;
    }
    if (Q->ne[1] <= 8 || Q->ne[1] > 65536 || Q->ne[2] < 2 || Q->ne[2] > 256 || Q->ne[3] != 1) {
        return p;
    }
    if (K->ne[1] < 1 || K->ne[1] > (int64_t(1) << 24) || K->ne[1] != V->ne[1]) {
        return p;
    }
    // at least two KV heads (one head cannot be split), matching K/V head counts, integral GQA
    if (K->ne[2] < 2 || K->ne[2] != V->ne[2] || K->ne[2] > Q->ne[2] || Q->ne[2] % K->ne[2] != 0 ||
            Q->ne[2] / K->ne[2] > 32 || K->ne[3] != 1 || V->ne[3] != 1) {
        return p;
    }
    // contiguous f32 output [D, n_head, n_q, 1]: the per-group scatter below writes whole D-rows into it
    if (dst->type != GGML_TYPE_F32 || dst->ne[0] != 256 || dst->ne[1] != Q->ne[2] || dst->ne[2] != Q->ne[1] ||
            dst->ne[3] != 1 || !ggml_is_contiguous(dst)) {
        return p;
    }
    // shared K/V views (MLA) are excluded: the full path converts that tensor once for both
    if (K == V || V->view_src == K || K->view_src == V ||
            (V->view_src != nullptr && V->view_src == K->view_src && V->view_offs == K->view_offs)) {
        return p;
    }
    // ordinary quantized cache types with a strided f16 converter (ggml_get_to_fp16_nc_cuda), block-contiguous rows
    for (const ggml_tensor * t : {K, V}) {
        switch (t->type) {
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_TURBO2_0:
            case GGML_TYPE_TURBO3_0:
            case GGML_TYPE_TURBO4_0:
            case GGML_TYPE_TQ5_0:
            case GGML_TYPE_TQ6_0:
                break;
            default:
                return p;
        }
        const size_t ts = ggml_type_size(t->type);
        if (t->ne[0] % ggml_blck_size(t->type) != 0 || t->nb[0] != ts || t->nb[1] % ts != 0 || t->nb[2] % ts != 0 ||
                t->nb[3] % ts != 0) {
            return p;
        }
    }
    // ALiBi slopes use the global query head index; renumbered heads would get the wrong slope
    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    if (max_bias != 0.0f) {
        return p;
    }
    // one mask shared by every head (per-head masks excluded)
    if (mask != nullptr && (mask->type != GGML_TYPE_F16 || mask->ne[0] < K->ne[1] || mask->ne[1] < Q->ne[1] ||
            mask->ne[2] != 1 || mask->ne[3] != 1)) {
        return p;
    }
    // sinks: one f32 per query head, contiguous, so a group's sinks are a pointer offset
    if (sinks != nullptr && (sinks->type != GGML_TYPE_F32 || !ggml_is_contiguous(sinks) || sinks->ne[0] != Q->ne[2] ||
            ggml_nelements(sinks) != Q->ne[2])) {
        return p;
    }

    // the route this op takes without the plan: not a fused packed-KV route, and the generic selector's f16 MMA kernel
    // (the only consumer the per-group launch below calls)
    if (ggml_cuda_flash_attn_ext_fused(nullptr, const_cast<ggml_tensor *>(dst), device)) {
        return p;
    }
    if (ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_MMA_F16) {
        return p;
    }

    const int64_t n_head_kv    = K->ne[2];
    const int64_t gqa          = Q->ne[2] / n_head_kv;
    const size_t  kv_per_head  = size_t(K->ne[1]) * 256 * sizeof(half);
    const size_t  out_per_head = size_t(Q->ne[1]) * size_t(gqa) * 256 * sizeof(float);

    p.budget = budget;
    p.full   = 2 * kv_per_head * size_t(n_head_kv);
    p.floor_bytes  = 2 * kv_per_head + out_per_head;
    // the full copies fit the budget, or one group is no smaller than the full copies: unchanged route
    if (p.full <= budget || p.floor_bytes >= p.full) {
        return p;
    }

    size_t heads = budget / p.floor_bytes;
    if (heads == 0) {
        heads        = 1; // #29: never below one complete GQA head group, never back to the full copy
        p.floor_used = true;
    }
    heads = std::min(heads, size_t(n_head_kv));

    p.heads     = int(heads);
    p.n_head_kv = int(n_head_kv);
    p.gqa       = int(gqa);
    p.kv_bytes  = kv_per_head * heads;
    p.out_bytes = out_per_head * heads;
    p.offset    = GGML_PAD(ggml_nbytes(dst), 128);
    p.workspace = 2 * p.kv_bytes + p.out_bytes;
    // The compute buffer is sized once, from the worst-case graph (n_ubatch queries, full context), and every runtime
    // shape must fit what that shape reserved. Smaller runtime shapes can take more heads per group (a smaller floor) or
    // the full copies (when they fit the budget): both are <= budget. Reserving max(workspace, budget) at every shape
    // therefore keeps every runtime reservation <= the worst-case one; the extra over `workspace` is < one floor.
    p.reserve   = std::max(p.workspace, budget);

    if (p.floor_used) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) {
            GGML_LOG_WARN("fattn bounded prefill: GGML_CUDA_PREFILL_KV_MIB=%zu is below the one-group floor at n_kv=%lld "
                          "(f16 K+V of 1 KV head %.1f MiB + output of its %lld query heads at n_q=%lld %.1f MiB = %.1f MiB); "
                          "using the floor: 1 of %lld KV heads per group, workspace %.1f MiB instead of the %.1f MiB "
                          "full f16 copies (logged once)\n",
                          budget >> 20, (long long) K->ne[1], 2 * kv_per_head / 1048576.0, (long long) gqa,
                          (long long) Q->ne[1], out_per_head / 1048576.0, p.floor_bytes / 1048576.0, (long long) n_head_kv,
                          p.workspace / 1048576.0, p.full / 1048576.0);
        }
    }
    return p;
}

bool ggml_cuda_flash_attn_ext_bounded_prefill_applies(int device, const ggml_tensor * dst) {
    return ggml_cuda_fattn_bounded_prefill_plan(device, dst).heads > 0;
}

// Scatter one group's contiguous output [D, heads*gqa, n_q] into dst [D, n_head, n_q] at query head `first`.
static __global__ void ggml_cuda_fattn_bounded_prefill_scatter(
        const float * src, float * dst, const int64_t n, const int group_heads,
        const int first, const int all_heads) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int64_t row = i / 256;            // (token, head within the group)
    const int64_t tok = row / group_heads;
    const int64_t h   = row % group_heads;
    dst[(tok*all_heads + first + h)*256 + i % 256] = src[i];
}

static void ggml_cuda_flash_attn_ext_bounded_prefill(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_fattn_bounded_plan & p) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    cudaStream_t stream = ctx.stream();

    // No allocation inside CUDA graph capture. ggml_cuda_graph_check_compability (ggml-cuda.cu) marks every cgraph that
    // holds a node this plan applies to as incompatible, so such graphs always run eagerly and this assert holds by
    // construction; it is here so that a future change to the capture rules fails loudly instead of recording a pool
    // allocation into a graph.
    {
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
        GGML_ASSERT(capture_status == cudaStreamCaptureStatusNone && "bounded prefill must not run inside CUDA graph capture");
    }

    // Workspace: the region ggml_cuda_flash_attn_ext_get_alloc_size reserved behind dst from the same plan, or the pool
    // when dst was not allocated through the buffer type (a view, or no buffer).
    ggml_cuda_pool_alloc<char> ws_pool(ctx.pool());
    char * ws = nullptr;
    const bool reserved = dst->buffer != nullptr && dst->view_src == nullptr && (uintptr_t) dst->data % 128 == 0 &&
        p.offset + p.workspace <= ggml_backend_buffer_get_alloc_size(dst->buffer, dst);
    if (reserved) {
        ws = (char *) dst->data + p.offset;
    } else {
        ws = ws_pool.alloc(p.workspace);
    }
    half  * K_ws = (half  *)  ws;
    half  * V_ws = (half  *) (ws + p.kv_bytes);
    float * O_ws = (float *) (ws + 2*p.kv_bytes);

    {
        // fallback ledger (GGML_LEDGER=1): the same f16_convert family as launch_fattn, with the bounded scratch
        const uint64_t id = ggml_cuda_ledger_mix(ggml_cuda_ledger_mix(ggml_cuda_ledger_mix(ggml_cuda_ledger_mix(ggml_cuda_ledger_mix(
            ggml_cuda_ledger_mix(0x22b0dedull, (uint64_t) K->type), (uint64_t) V->type), (uint64_t) ggml_cuda_ledger_width_bucket(Q->ne[1])),
            (uint64_t) reserved + 2), (uint64_t) p.heads), (uint64_t) p.n_head_kv);
        ggml_cuda_ledger_count("cuda.fattn", id, [&](char * buf, size_t size) {
            char w[16];
            ggml_cuda_ledger_width_str(Q->ne[1], w, sizeof(w));
            snprintf(buf, size, "f16_convert K=%s V=%s n_q=%s scratch=bounded-%s heads=%d/%d", ggml_type_name(K->type),
                ggml_type_name(V->type), w, reserved ? "reserved" : "pool", p.heads, p.n_head_kv);
        });
    }

    // Groups in ascending KV-head order on one stream: deterministic, and the workspace is reused group after group
    // (stream order serialises each group's conversion after the previous group's kernel and scatter).
    for (int first = 0; first < p.n_head_kv; first += p.heads) {
        const int heads = std::min(p.heads, p.n_head_kv - first);

        ggml_tensor q = *Q;
        ggml_tensor k = *K;
        ggml_tensor v = *V;
        ggml_tensor out = *dst;
        ggml_tensor sinks;

        q.ne[2] = int64_t(heads) * p.gqa;
        q.data  = (char *) Q->data + size_t(first) * p.gqa * Q->nb[2];

        // Converter: ggml_get_to_fp16_nc_cuda (convert.cu:934) -> dequantize_block_cuda (convert.cu:417) -> kernel
        // dequantize_block (convert.cu:10). A head slice of the KV cache view is not contiguous (the cache interleaves
        // heads per token: nb[1] = n_head_kv rows, nb[2] = one row), so this is always the strided converter with
        // block-unit strides (nb / type size), never the contiguous converter on a head slice. The output is canonical
        // contiguous f16 [D, n_kv, heads].
        ggml_tensor * kv_t[2]  = { &k, &v };
        half        * kv_ws[2] = { K_ws, V_ws };
        for (int j = 0; j < 2; ++j) {
            ggml_tensor * t = kv_t[j];
            const size_t ts = ggml_type_size(t->type);
            const to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(t->type);
            GGML_ASSERT(to_fp16 != nullptr);
            to_fp16((const char *) t->data + size_t(first) * t->nb[2], kv_ws[j],
                t->ne[0], t->ne[1], heads, 1, int64_t(t->nb[1] / ts), int64_t(t->nb[2] / ts), int64_t(t->nb[3] / ts), stream);
            CUDA_CHECK(cudaGetLastError());

            t->type  = GGML_TYPE_F16;
            t->ne[2] = heads;
            t->nb[0] = sizeof(half);
            for (int d = 1; d < GGML_MAX_DIMS; ++d) {
                t->nb[d] = t->nb[d - 1] * t->ne[d - 1];
            }
            t->data      = kv_ws[j];
            t->buffer    = nullptr;
            t->view_src  = nullptr;
            t->view_offs = 0;
        }

        out.src[0] = &q;
        out.src[1] = &k;
        out.src[2] = &v;
        out.ne[1]  = q.ne[2];
        out.nb[0]  = sizeof(float);
        for (int d = 1; d < GGML_MAX_DIMS; ++d) {
            out.nb[d] = out.nb[d - 1] * out.ne[d - 1];
        }
        out.data      = O_ws;
        out.buffer    = nullptr; // launch_fattn: no reserved-region lookup (K/V are f16 already, nothing converts)
        out.view_src  = nullptr;
        out.view_offs = 0;
        if (dst->src[4] != nullptr) {
            sinks       = *dst->src[4];
            sinks.ne[0] = q.ne[2];
            for (int d = 1; d < GGML_MAX_DIMS; ++d) {
                sinks.nb[d] = sinks.nb[d - 1] * sinks.ne[d - 1];
            }
            sinks.data  = (char *) dst->src[4]->data + size_t(first) * p.gqa * sizeof(float);
            out.src[4]  = &sinks;
        }

        // Consumer: ggml_cuda_flash_attn_ext_mma_f16 (this file) -> ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>
        // -> ggml_cuda_flash_attn_ext_mma_f16_case (fattn-mma-f16.cuh) -> launch_fattn (fattn-common.cuh), the same kernel
        // the unbounded route runs, with the same gqa ratio (so the same ncols2), on `heads` KV heads.
        ggml_cuda_flash_attn_ext_mma_f16(ctx, &out, ggml_cuda_fattn_mma_cand_pair(K->type, V->type));

        const int64_t n = ggml_nelements(&out);
        const ggml_cuda_kernel_launch_params launch(dim3((unsigned int) ((n + 255) / 256)), dim3(256), 0, stream);
        ggml_cuda_kernel_launch(ggml_cuda_fattn_bounded_prefill_scatter, launch, (const float *) O_ws, (float *) dst->data,
            n, heads * p.gqa, first * p.gqa, (int) Q->ne[2]);
        CUDA_CHECK(cudaGetLastError());
    }
}

// [I8QK] opt-in int8-QK prefill route (GGML_CUDA_FA_I8QK=1, fattn-i8qk.cu). Takes only ops that would run the f16 MMA
// kernel (generic or bounded prefill route) and reuses the reservation the allocator made for that route: the f16 K
// region holds the int8 K, the f16 V region first stages the f16 K, then holds the f16 V. No new reservation.
static bool ggml_cuda_flash_attn_ext_i8qk_route(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;
    if (!ggml_cuda_fattn_i8qk_applies(cc, dst)) {
        return false;
    }
    if (ggml_cuda_flash_attn_ext_fused(nullptr, dst, device) ||
            ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_MMA_F16) {
        return false;
    }
    const ggml_tensor * K = dst->src[1];

    const ggml_cuda_fattn_bounded_plan p = ggml_cuda_fattn_bounded_prefill_plan(device, dst);
    if (p.heads > 0) {
        // same workspace and capture rule as ggml_cuda_flash_attn_ext_bounded_prefill
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &capture_status));
        GGML_ASSERT(capture_status == cudaStreamCaptureStatusNone && "bounded prefill must not run inside CUDA graph capture");

        ggml_cuda_pool_alloc<char> ws_pool(ctx.pool());
        char * ws = nullptr;
        const bool reserved = dst->buffer != nullptr && dst->view_src == nullptr && (uintptr_t) dst->data % 128 == 0 &&
            p.offset + p.workspace <= ggml_backend_buffer_get_alloc_size(dst->buffer, dst);
        ws = reserved ? (char *) dst->data + p.offset : ws_pool.alloc(p.workspace);

        ggml_cuda_fattn_path_note("i8qk_bounded", dst, 0);
        for (int first = 0; first < p.n_head_kv; first += p.heads) {
            const int heads = std::min(p.heads, p.n_head_kv - first);
            ggml_cuda_flash_attn_ext_i8qk(ctx, dst, first, heads, ws, (half *) (ws + p.kv_bytes));
        }
        return true;
    }

    // generic route: the f16 K/V copies reserved behind dst (both caches are quantized, see the gate); without that
    // reservation the op keeps the f16 route rather than taking new VRAM from the pool
    if (dst->buffer == nullptr || dst->view_src != nullptr) {
        return false;
    }
    const ggml_cuda_flash_attn_ext_f16_extra_data want = ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, true, true);
    if (want.K == 0 || want.V == 0 || want.end - (uintptr_t) dst->data > ggml_backend_buffer_get_alloc_size(dst->buffer, dst)) {
        return false;
    }
    void * ws_k = (void *) want.K;
    half * ws_v = (half *) want.V;
    ggml_cuda_fattn_path_note("i8qk", dst, 0);
    ggml_cuda_flash_attn_ext_i8qk(ctx, dst, 0, (int) K->ne[2], ws_k, ws_v);
    return true;
}

static size_t ggml_cuda_fattn_generic_alloc_size(const int device, const ggml_tensor * dst, best_fattn_kernel * kernel_out) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    // KVarN region attention (src[6]) runs its own MMA path with no f16 K/V copies behind dst
    const best_fattn_kernel kernel = dst->src[6] != nullptr ? BEST_FATTN_KERNEL_NONE : ggml_cuda_get_best_fattn_kernel(device, dst);
    if (kernel_out) {
        *kernel_out = kernel;
    }

    bool need_f16_K = false;
    bool need_f16_V = false;

    switch (kernel) {
        case BEST_FATTN_KERNEL_TILE:
        case BEST_FATTN_KERNEL_MMA_F16:
            need_f16_K = true;
            need_f16_V = true;
            break;
        case BEST_FATTN_KERNEL_VEC: {
            const bool f16_fallback = ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type) == nullptr;
            need_f16_K = K->type == GGML_TYPE_F32 || f16_fallback;
            need_f16_V = V->type == GGML_TYPE_F32 || f16_fallback;
        } break;
        case BEST_FATTN_KERNEL_NONE:
            break;
    }

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, need_f16_K, need_f16_V);

    return f16_extra.end - (uintptr_t) dst->data;
}

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);

    // dry run: nothing launches and dst is not modified
    const bool fused = ggml_cuda_flash_attn_ext_fused(nullptr, const_cast<ggml_tensor *>(dst), device);

    best_fattn_kernel kernel = BEST_FATTN_KERNEL_NONE;
    const size_t size_generic = ggml_cuda_fattn_generic_alloc_size(device, dst, &kernel);
    const size_t size_fused   = ggml_nbytes(dst); // no f16 K/V copies
    // [#22/#29] the same plan the executor runs (ggml_cuda_fattn_bounded_prefill_plan); budget 0 = never applies
    const ggml_cuda_fattn_bounded_plan bounded = fused ? ggml_cuda_fattn_bounded_plan() :
        ggml_cuda_fattn_bounded_prefill_plan(device, dst);
    const size_t size_bounded = bounded.offset + bounded.reserve;
    const size_t size         = fused && ggml_cuda_fattn_alloc_route() ? size_fused :
                                bounded.heads > 0 ? size_bounded : size_generic;

    if (ggml_cuda_fattn_alloc_log()) {
        static std::mutex mtx;
        static std::map<uint64_t, bool> seen;
        int log2_kv = 0;
        while ((int64_t(1) << (log2_kv + 1)) <= K->ne[1]) {
            log2_kv++;
        }
        const uint64_t key = ggml_cuda_ledger_mix(ggml_cuda_ledger_mix(ggml_cuda_ledger_mix(ggml_cuda_ledger_mix(
            (uint64_t) K->type, (uint64_t) V->type), (uint64_t) Q->ne[1]), (uint64_t) log2_kv), (uint64_t) fused * 2 + (uint64_t) kernel);
        const uint64_t key_bp = bounded.heads > 0 ? ggml_cuda_ledger_mix(key, (uint64_t) bounded.heads) : key;
        std::lock_guard<std::mutex> lock(mtx);
        if (bounded.heads > 0 && seen.emplace(key_bp, true).second) {
            GGML_LOG_WARN("fattn alloc: dev %d K=%s V=%s n_q=%lld n_kv=%lld route=bounded_mma_f16 reserve %.1f MiB behind dst "
                          "(%d of %d KV heads per group: f16 K+V %.1f MiB + group output %.1f MiB; budget %zu MiB, "
                          "one-group floor %.1f MiB%s; generic selector: %s, %.1f MiB; saved %.1f MiB)\n",
                device, ggml_type_name(K->type), ggml_type_name(V->type), (long long) Q->ne[1], (long long) K->ne[1],
                (size - ggml_nbytes(dst)) / 1048576.0, bounded.heads, bounded.n_head_kv, 2 * bounded.kv_bytes / 1048576.0,
                bounded.out_bytes / 1048576.0, bounded.budget >> 20, bounded.floor_bytes / 1048576.0,
                bounded.floor_used ? " used" : "", ggml_cuda_fattn_kernel_name(kernel),
                (size_generic - ggml_nbytes(dst)) / 1048576.0, ((double) size_generic - (double) size) / 1048576.0);
        } else if (bounded.heads == 0 && seen.emplace(key, true).second) {
            // WARN, not INFO: ggml INFO maps to trace verbosity (4), above the server's default of 3, so INFO never prints
            GGML_LOG_WARN("fattn alloc: dev %d K=%s V=%s n_q=%lld n_kv=%lld route=%s reserve %.1f MiB behind dst "
                          "(generic selector: %s, %.1f MiB; saved %.1f MiB)\n",
                device, ggml_type_name(K->type), ggml_type_name(V->type), (long long) Q->ne[1], (long long) K->ne[1],
                fused ? "fused" : ggml_cuda_fattn_kernel_name(kernel),
                (size - ggml_nbytes(dst)) / 1048576.0, ggml_cuda_fattn_kernel_name(kernel),
                (size_generic - ggml_nbytes(dst)) / 1048576.0, (size_generic - size) / 1048576.0);
        }
    }

    return size;
}

void ggml_cuda_flash_attn_kvarn_lowbits(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// [#139] same predicate as the low-bit branch of ggml_cuda_flash_attn_ext below
static bool ggml_cuda_fattn_kvarn_is_lowbits(const ggml_tensor * dst) {
    return ggml_get_op_params_i32(dst,5) != ((4 << 8) | 4) || (ggml_get_op_params_i32(dst,7) == GGML_TYPE_I16 && ggml_kvarn::trellis3::tokens4());
}

// [#139] fork-only: the KVarN paths that apply the fused Q / output rotation inside their kernels. Must mirror the
// dispatch above: the low-bit tile kernel (Ampere config, nbatch_combine == DV/2) and width kernel, and the 4/4
// stream decode kernel (widths <= GGML_KVARN_DIRECT_STREAM_MAX). Everything else runs the separate-pass fallback.
static bool ggml_cuda_fattn_kvarn_rot_in_kernel(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    // GGML_KVARN_FUSED_ROT_SEPARATE=1: every path takes the separate passes (A/B of the in-kernel rotation)
    static const bool separate = getenv("GGML_KVARN_FUSED_ROT_SEPARATE") != nullptr && atoi(getenv("GGML_KVARN_FUSED_ROT_SEPARATE")) != 0;
    if (separate || !ggml_cuda_fattn_kvarn_rot_q_aligned(Q) || ggml_get_op_params_i32(dst, 7) == GGML_TYPE_TURBO4_0) {
        return false;
    }
    if (ggml_cuda_fattn_kvarn_is_lowbits(dst)) {
        return true; // fattn-kvarn-lowbits.cu rotates in-kernel or takes the separate passes itself
    }
    GGML_UNUSED(ctx);
    // 4/4: the stream decode kernel (fattn-kvarn-stream.cuh) through ggml_cuda_flash_attn_ext_kvarn_direct_impl
    static const int stream_max = getenv("GGML_KVARN_DIRECT_STREAM_MAX") != nullptr
                                ? atoi(getenv("GGML_KVARN_DIRECT_STREAM_MAX")) : 8; // same default as fattn-kvarn-direct.cuh
    return Q->ne[1] <= 8 && Q->ne[1] <= stream_max && Q->ne[1] < 128 && ggml_cuda_flash_attn_ext_kvarn_direct_supported(dst);
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);

    // KVarN region-aware attention (sink/ring f16 rows + sealed 4-bit body records): dedicated MMA path.
    if (dst->src[6] != nullptr) {
        if (ggml_cuda_fattn_kvarn_rot(dst) && !ggml_cuda_fattn_kvarn_rot_in_kernel(ctx, dst)) {
            // separate passes around this same dispatch, on a copy of the node without the flag
            ggml_cuda_flash_attn_ext_kvarn_rot_unfused(ctx, dst, ggml_cuda_flash_attn_ext);
            return;
        }
        // 4/4 token-axis trellis4 body (I16 under an explicit nonzero GGML_KVARN_TRELLIS_TOKENS; the low-bit token-axis default does not apply to 4/4): the low-bit tile loader
        if (ggml_get_op_params_i32(dst,5) != ((4 << 8) | 4) || (ggml_get_op_params_i32(dst,7) == GGML_TYPE_I16 && ggml_kvarn::trellis3::tokens4())) {
            ggml_cuda_fattn_path_note("kvarn_lowerbits", dst, -1);
            ggml_cuda_flash_attn_kvarn_lowbits(ctx,dst);
            return;
        }
        ggml_cuda_fattn_path_note("kvarn", dst, -1);
        ggml_cuda_flash_attn_ext_mma_kvarn_switch_ncols2<256, 256>(ctx, dst);
        return;
    }

    if (ggml_cuda_flash_attn_ext_fused(&ctx, dst, ggml_cuda_get_device())) {
        return;
    }

    if (ggml_cuda_flash_attn_ext_i8qk_route(ctx, dst)) {
        return;
    }

    // [#22/#29] bounded f16 prefill: the plan the allocator sized this op's reservation with
    {
        const ggml_cuda_fattn_bounded_plan bounded = ggml_cuda_fattn_bounded_prefill_plan(ggml_cuda_get_device(), dst);
        if (bounded.heads > 0) {
            ggml_cuda_fattn_path_note("mma_f16_bounded", dst, 0);
            ggml_cuda_flash_attn_ext_bounded_prefill(ctx, dst, bounded);
            return;
        }
    }

    switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_fattn_path_note("tile", dst, 0);
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_fattn_path_note("vec", dst, 0);
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
    }
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    if (dst->src[6] != NULL) {
        // KVarN region attention: only the dedicated MMA path (fattn-mma-kvarn.cuh) serves it
        const int cc = ggml_cuda_info().devices[device].cc;
        return ggml_cuda_flash_attn_ext_kvarn_supported(cc, dst);
    }
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}
