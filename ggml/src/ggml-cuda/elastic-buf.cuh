#pragma once

// fork (SURGE-cache, elastic KV): a device buffer whose virtual range is reserved once and whose physical pages can be
// mapped and unmapped while the buffer lives (CUDA VMM). The KV cache keeps its sealed-record pools in one so the pages
// past the live context can be given back to the driver (and taken by the MoE expert cache), then mapped again when the
// context grows. Unmapped pages read as the last clear value once mapped again. Opt-in: LLAMA_ELASTIC_KV=1.

#include "common.cuh"

struct ggml_cuda_elastic;

// reserve `size` bytes of virtual memory on `device` and map all of it; nullptr if VMM is not available or out of memory
ggml_cuda_elastic * ggml_cuda_elastic_create(int device, int physical_device, size_t size, void ** base);
void   ggml_cuda_elastic_destroy(ggml_cuda_elastic * e);
// memset the mapped pages; pages mapped later are filled with `value` too
void   ggml_cuda_elastic_clear(ggml_cuda_elastic * e, uint8_t value);
// map exactly the pages that intersect the n ranges [off[i], off[i] + len[i]) (offsets from the base), unmap the rest.
// The device is synchronized before any page is unmapped. On a failed map nothing changes and false is returned.
bool   ggml_cuda_elastic_set_ranges(ggml_cuda_elastic * e, const size_t * off, const size_t * len, int n);
size_t ggml_cuda_elastic_mapped(const ggml_cuda_elastic * e);
size_t ggml_cuda_elastic_page(const ggml_cuda_elastic * e);
