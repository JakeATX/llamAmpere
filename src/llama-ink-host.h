#pragma once

// fork (INKMISS): where the host-resident experts of a MoE model sit in RAM (Inkling 2 on 8/10/12 GB cards).
// All of it is off by default; with none of the env vars set nothing here runs and the experts stay in the model
// file pages, as before.
//
//  LLAMA_INK_HOST_ARENA=1  (#104) copy the host experts into one anonymous arena and point the tensors at it. Each layer
//                          is laid out like an expert slot of the prompt streamer (same offsets, zeroed tails), so one
//                          copy can upload a whole layer (LLAMA_INK_STREAM_DIRECT). Linux only. The file pages of the
//                          experts are dropped from the process after the copy.
//  LLAMA_INK_HOST_THP=0|1|2 (POSTREV 8) with the arena: back it with 2 MiB transparent huge pages. 1 (default) only when
//                          free 2 MiB blocks cover half of it (compaction on a fragmented box can stall the start), 2 always,
//                          0 never. Fewer page faults at start and fewer TLB misses for CPU-computed experts.
//  LLAMA_INK_HOST_PIN=1    (#104) page-lock the host experts for the GPU (the arena, else the file pages in place, read
//                          only): expert uploads (MoE cache misses, prompt streamer) become DMA from pinned memory instead of
//                          staged pageable copies, and the host thread no longer blocks on them.

#include "ggml-backend.h"

#include <cstddef>

struct llama_model;

// once per model, before the MoE cache and the prompt streamer read the experts; buft_dev: the GPU buffer type the
// streamer allocates its slots from (for the arena layout), may be nullptr. No-op unless one of the env vars is set.
void llama_ink_host_prepare(const llama_model & model, ggml_backend_buffer_type_t buft_dev);

// whether [ptr, ptr + size) is host expert memory that llama_ink_host_prepare page-locked
bool llama_ink_host_pinned(const void * ptr, size_t size);

// bytes from ptr (the first expert tensor of a layer in the arena) to the end of that layer's last zeroed tail, laid out
// as a streamer slot for the given buffer type; 0 if ptr does not start such a layer
size_t llama_ink_host_layer_span(const void * ptr, ggml_backend_buffer_type_t buft_dev);
