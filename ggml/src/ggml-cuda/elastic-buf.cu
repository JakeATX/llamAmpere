#include "elastic-buf.cuh"

#include <algorithm>
#include <vector>

// fork (SURGE-cache, elastic KV): see elastic-buf.cuh. One physical allocation per page, so any page can be given back.

#if defined(GGML_USE_VMM) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

struct ggml_cuda_elastic {
    int         device   = 0;
    int         physical = 0;
    CUdeviceptr base     = 0;
    size_t      size     = 0; // reserved bytes, a multiple of page
    size_t      page     = 0;
    uint8_t     value    = 0; // fill of a newly mapped page
    std::vector<uint8_t> mapped;
    size_t      n_mapped = 0;
};

// map the n unmapped pages from page i: one physical allocation per page (any page can be given back alone), then one
// access call and one fill for the whole run (per page, the four calls took ~0.13 ms a 2 MiB page: ~0.3 s to map the
// 4.6 GiB a prompt lent again). The range of one access call may span many allocations once all of them are mapped.
// On a failure the pages of this run are unmapped again and false is returned.
static bool elastic_map_run(ggml_cuda_elastic * e, size_t i, size_t n) {
    CUmemAllocationProp prop = {};
    prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = e->physical;
    size_t k = 0;
    for (; k < n; ++k) {
        CUmemGenericAllocationHandle handle;
        if (cuMemCreate(&handle, e->page, &prop, 0) != CUDA_SUCCESS) {
            break;
        }
        const CUdeviceptr ptr = e->base + (i + k)*e->page;
        if (cuMemMap(ptr, e->page, 0, handle, 0) != CUDA_SUCCESS) {
            cuMemRelease(handle);
            break;
        }
        // the mapping keeps the memory alive; the handle is not needed after it
        CU_CHECK(cuMemRelease(handle));
    }
    const CUdeviceptr ptr = e->base + i*e->page;
    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id   = e->physical;
    access.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (k < n || cuMemSetAccess(ptr, n*e->page, &access, 1) != CUDA_SUCCESS) {
        for (size_t q = 0; q < k; ++q) {
            CU_CHECK(cuMemUnmap(ptr + q*e->page, e->page));
        }
        return false;
    }
    CUDA_CHECK(cudaMemset((void *) ptr, e->value, n*e->page));
    for (size_t q = 0; q < n; ++q) {
        e->mapped[i + q] = 1;
    }
    e->n_mapped += n;
    return true;
}

static void elastic_unmap_page(ggml_cuda_elastic * e, size_t i) {
    CU_CHECK(cuMemUnmap(e->base + i*e->page, e->page));
    e->mapped[i] = 0;
    e->n_mapped--;
}

ggml_cuda_elastic * ggml_cuda_elastic_create(int device, int physical_device, size_t size, void ** base) {
    CUmemAllocationProp prop = {};
    prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = physical_device;
    size_t page = 0;
    if (cuMemGetAllocationGranularity(&page, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS || page == 0) {
        return nullptr;
    }
    auto * e = new ggml_cuda_elastic;
    e->device   = device;
    e->physical = physical_device;
    e->page     = page;
    e->size     = std::max(page, (size + page - 1)/page*page);
    if (cuMemAddressReserve(&e->base, e->size, 0, 0, 0) != CUDA_SUCCESS) {
        delete e;
        return nullptr;
    }
    e->mapped.assign(e->size/page, 0);
    ggml_cuda_set_device(device);
    if (!elastic_map_run(e, 0, e->mapped.size())) {
        ggml_cuda_elastic_destroy(e);
        return nullptr;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    *base = (void *) e->base;
    return e;
}

void ggml_cuda_elastic_destroy(ggml_cuda_elastic * e) {
    if (e == nullptr) {
        return;
    }
    ggml_cuda_set_device(e->device);
    CUDA_CHECK(cudaDeviceSynchronize());
    for (size_t i = 0; i < e->mapped.size(); ++i) {
        if (e->mapped[i]) {
            elastic_unmap_page(e, i);
        }
    }
    CU_CHECK(cuMemAddressFree(e->base, e->size));
    delete e;
}

void ggml_cuda_elastic_clear(ggml_cuda_elastic * e, uint8_t value) {
    ggml_cuda_set_device(e->device);
    e->value = value;
    for (size_t i = 0; i < e->mapped.size();) {
        if (!e->mapped[i]) {
            i++;
            continue;
        }
        size_t n = 1;
        while (i + n < e->mapped.size() && e->mapped[i + n]) {
            n++;
        }
        CUDA_CHECK(cudaMemset((void *) (e->base + i*e->page), value, n*e->page));
        i += n;
    }
}

bool ggml_cuda_elastic_set_ranges(ggml_cuda_elastic * e, const size_t * off, const size_t * len, int n) {
    const size_t n_pages = e->mapped.size();
    std::vector<uint8_t> want(n_pages, 0);
    for (int i = 0; i < n; ++i) {
        if (len[i] == 0) {
            continue;
        }
        const size_t first = off[i]/e->page;
        const size_t last  = std::min(n_pages - 1, (off[i] + len[i] - 1)/e->page);
        for (size_t p = first; p <= last; ++p) {
            want[p] = 1;
        }
    }
    ggml_cuda_set_device(e->device);
    // the pages a kernel in flight may still read go only after the device is idle
    bool sync = false;
    std::vector<std::pair<size_t, size_t>> added; // runs of newly mapped pages
    for (size_t p = 0; p < n_pages;) {
        if (!want[p] || e->mapped[p]) {
            p++;
            continue;
        }
        size_t n = 1;
        while (p + n < n_pages && want[p + n] && !e->mapped[p + n]) {
            n++;
        }
        if (!elastic_map_run(e, p, n)) {
            // undo: back to the old set
            for (const auto & r : added) {
                if (!sync) {
                    CUDA_CHECK(cudaDeviceSynchronize());
                    sync = true;
                }
                for (size_t q = r.first; q < r.first + r.second; ++q) {
                    elastic_unmap_page(e, q);
                }
            }
            (void) cudaGetLastError();
            return false;
        }
        added.push_back({ p, n });
        p += n;
    }
    for (size_t p = 0; p < n_pages; ++p) {
        if (!want[p] && e->mapped[p]) {
            if (!sync) {
                CUDA_CHECK(cudaDeviceSynchronize());
                sync = true;
            }
            elastic_unmap_page(e, p);
        }
    }
    if (!added.empty()) {
        // the fills run on the legacy stream, which the backend's non-blocking streams do not wait for
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    return true;
}

size_t ggml_cuda_elastic_mapped(const ggml_cuda_elastic * e) {
    return e->n_mapped*e->page;
}

size_t ggml_cuda_elastic_page(const ggml_cuda_elastic * e) {
    return e->page;
}

#else // no VMM: the elastic buffer type is not offered

struct ggml_cuda_elastic {};

ggml_cuda_elastic * ggml_cuda_elastic_create(int, int, size_t, void **) { return nullptr; }
void   ggml_cuda_elastic_destroy(ggml_cuda_elastic *) {}
void   ggml_cuda_elastic_clear(ggml_cuda_elastic *, uint8_t) {}
bool   ggml_cuda_elastic_set_ranges(ggml_cuda_elastic *, const size_t *, const size_t *, int) { return false; }
size_t ggml_cuda_elastic_mapped(const ggml_cuda_elastic *) { return 0; }
size_t ggml_cuda_elastic_page(const ggml_cuda_elastic *) { return 0; }

#endif
