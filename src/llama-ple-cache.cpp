#include "llama-ple-cache.h"
#include "llama-impl.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

#ifndef _WIN32
#include <sys/mman.h>
#include <unistd.h>
#endif

std::unique_ptr<llama_ple_cache> llama_ple_cache::load(const ggml_tensor * tensor, const char * profile, bool pin) {
    if (!tensor || !tensor->data || !tensor->buffer || !ggml_backend_buffer_is_host(tensor->buffer) ||
            !ggml_is_contiguous(tensor) || tensor->ne[1] <= 0 || tensor->ne[1] > INT32_MAX ||
            tensor->ne[2] != 1 || tensor->ne[3] != 1) {
        throw std::runtime_error("PLE RAM cache requires a contiguous host table");
    }
    std::ifstream input(profile);
    std::string magic;
    int64_t type, width, full_rows, row_bytes, n_rows;
    if (!(input >> magic >> type >> width >> full_rows >> row_bytes >> n_rows) || magic != "LLAMA_PLE_ROWS_V1" ||
            type != tensor->type || width != tensor->ne[0] || full_rows != tensor->ne[1] ||
            row_bytes <= 0 || (uint64_t) row_bytes != tensor->nb[1] || n_rows <= 0 || n_rows > full_rows ||
            (uint64_t) n_rows > SIZE_MAX / (uint64_t) row_bytes) {
        throw std::runtime_error("invalid or incompatible PLE row profile");
    }
    std::unique_ptr<llama_ple_cache> cache(new llama_ple_cache());
    const auto rows_start = input.tellg();
    input.seekg(0, std::ios::end);
    const auto profile_end = input.tellg();
    if (rows_start < 0 || profile_end < rows_start || (uint64_t) n_rows > (uint64_t) (profile_end - rows_start) / 2) {
        throw std::runtime_error("truncated PLE profile");
    }
    input.seekg(rows_start);
    const size_t bytes = (size_t) n_rows * (size_t) row_bytes;
    const size_t available = llama_moe_arena::available_memory();
    const size_t reserve = 7ull << 30;
    if (available && (available <= reserve || bytes > available - reserve ||
            (uint64_t) n_rows * sizeof(int32_t) > available - reserve - bytes)) {
        throw std::runtime_error("PLE row cache would cross the seven GiB RAM floor");
    }
    cache->full_rows = full_rows;
    if (const char * path = getenv("LLAMA_PLE_MISS_PROFILE"); path && *path) {
        cache->miss_profile = path;
    }
    cache->rows.reserve(n_rows);
    for (int64_t i = 0; i < n_rows; ++i) {
        int64_t row;
        if (!(input >> row) || row < 0 || row >= full_rows || (!cache->rows.empty() && row <= cache->rows.back())) {
            throw std::runtime_error("PLE profile rows must be sorted, distinct and in range");
        }
        cache->rows.push_back((int32_t) row);
    }
    input >> std::ws;
    if (!input.eof()) {
        throw std::runtime_error("unexpected data after PLE profile rows");
    }
    cache->storage.reset(new llama_moe_arena(bytes));
    const char * source = (const char *) tensor->data;
    for (size_t i = 0; i < cache->rows.size(); ++i) {
        if (i % 4096 == 0) {
            const size_t left = llama_moe_arena::available_memory();
            if (left && left < reserve) {
                throw std::runtime_error("PLE row loading crossed the seven GiB RAM floor");
            }
        }
        memcpy((char *) cache->storage->data + i * row_bytes, source + (size_t) cache->rows[i] * row_bytes, row_bytes);
    }
    if (pin) {
        if (llama_moe_arena::pin_supported()) {
            cache->storage->pin();
        } else {
#ifndef _WIN32
            if (mlock(cache->storage->data, bytes) != 0) {
                throw std::runtime_error("cannot lock PLE row cache in RAM");
            }
#else
            throw std::runtime_error("PLE RAM cache requires host memory locking");
#endif
        }
    }
#ifndef _WIN32
    // Source pages are no longer needed. Keep neighboring tensors' boundary pages intact.
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    const uintptr_t start = (uintptr_t) source;
    const size_t source_bytes = ggml_nbytes(tensor);
    const size_t skip = (page - start % page) % page;
    if (source_bytes > skip) {
        madvise((void *) (start + skip), (source_bytes - skip) / page * page, MADV_DONTNEED);
    }
#endif
    cache->context.reset(ggml_init({ggml_tensor_overhead(), nullptr, true}));
    if (!cache->context) { throw std::bad_alloc(); }
    cache->tensor = ggml_new_tensor_2d(cache->context.get(), tensor->type, width, n_rows);
    ggml_set_name(cache->tensor, "ple_ram");
    if (ggml_backend_tensor_alloc(cache->storage->buffer.get(), cache->tensor, cache->storage->data) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("cannot bind compact PLE rows");
    }
    LLAMA_LOG_INFO("%s: PLE RAM rows=%lld/%lld bytes=%zu original_bytes=%llu; startup only, misses stop without disk reads\n",
            __func__, (long long) n_rows, (long long) full_rows, bytes, (unsigned long long) full_rows * row_bytes);
    return cache;
}

void llama_ple_cache::save_miss_profile(const int32_t * indices, size_t count) const {
    if (miss_profile.empty()) { return; }
    std::lock_guard<std::mutex> lock(miss_mutex);
    auto expanded = rows;
    for (size_t i = 0; i < count; ++i) {
        if (indices[i] >= 0 && indices[i] < full_rows) { expanded.push_back(indices[i]); }
    }
    std::sort(expanded.begin(), expanded.end());
    expanded.erase(std::unique(expanded.begin(), expanded.end()), expanded.end());
    std::ofstream output(miss_profile);
    output << "LLAMA_PLE_ROWS_V1 " << int(tensor->type) << ' ' << tensor->ne[0] << ' ' << full_rows << ' '
           << tensor->nb[1] << ' ' << expanded.size() << '\n';
    for (int32_t row : expanded) { output << row << '\n'; }
    output.close();
    if (!output) {
        LLAMA_LOG_ERROR("%s: cannot save PLE miss profile to %s\n", __func__, miss_profile.c_str());
    } else {
        LLAMA_LOG_INFO("%s: saved %zu startup rows to %s; includes this failed batch, not future coverage\n",
                __func__, expanded.size(), miss_profile.c_str());
    }
}

bool llama_ple_cache::remap(int32_t * indices, size_t count) const {
    std::vector<int32_t> mapped(count);
    uint64_t missing = 0;
    for (size_t i = 0; i < count; ++i) {
        const auto it = std::lower_bound(rows.begin(), rows.end(), indices[i]);
        if (it == rows.end() || *it != indices[i]) {
            ++missing;
        } else {
            mapped[i] = (int32_t) (it - rows.begin());
        }
    }
    lookups.fetch_add(count, std::memory_order_relaxed);
    misses.fetch_add(missing, std::memory_order_relaxed);
    if (missing) {
        save_miss_profile(indices, count);
        LLAMA_LOG_ERROR("%s: PLE RAM cache misses=%llu/%zu cumulative_misses=%llu; no weight reads attempted\n",
                __func__, (unsigned long long) missing, count, (unsigned long long) misses.load());
        return false;
    }
    LLAMA_LOG_DEBUG("%s: PLE RAM lookups=%zu misses=0 cumulative_lookups=%llu cumulative_misses=%llu\n",
            __func__, count, (unsigned long long) lookups.load(), (unsigned long long) misses.load());
    std::copy(mapped.begin(), mapped.end(), indices);
    return true;
}
