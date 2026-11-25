#pragma once

// HOSTOVL host-side submit helpers.
//
// A graph whose only CPU work is the token-embedding lookup (a few rows of GET_ROWS on a host-mapped tok_embd,
// plus views/copies of the ids) still enters the CPU backend with the full batch thread count. With OpenMP that is a
// parallel region whose end barrier waits until every team thread has been scheduled; on a pinned CPU set shared
// with the CUDA driver and server threads, that wait measured 9-14 ms per call in streaks (DSpark draft decode,
// Inkling 24ink 100K: 13.1 of 16.5 ms GPU idle per round). Running such a graph's CPU part on one thread removes it.

#include "ggml.h"
#include "ggml-backend.h"

// LLAMA_HOST_LITE1=1 enables the single-thread submit for lite CPU parts (=0 kill switch)
bool llama_host_lite1_enabled();
// LLAMA_HOST_LITE1_ROWS=N: the row bound (default 64; bytes bound N * 64 KiB)
int64_t llama_host_lite1_rows();

// true when every node the scheduler placed on `cpu` is a light op (views, copies, GET_ROWS, scale) over at most
// `max_rows` rows and `max_rows` * 64 KiB; a graph with no CPU nodes is lite too
bool llama_host_cpu_part_is_lite(ggml_backend_sched_t sched, ggml_backend_t cpu, ggml_cgraph * gf, int64_t max_rows = 64);

// LLAMA_DSPARK_EMBD_DEV=1 (default off): a DFlash/DSpark draft context copies the target's token embeddings to the
// draft's output device once, so its per-round lookup runs on the card (no CPU split). Costs n_vocab rows of VRAM
// (Inkling q8_0: 201024 x 2304 = 492 MB) that the memory fit does not see.
bool llama_dspark_embd_dev_enabled();

#include "ggml-cpp.h"   // ggml_context_ptr, ggml_backend_buffer_ptr

// returns the device copy (owned by ctx_out/buf_out) or nullptr (src null, src already on a device, no device, or the
// allocation failed: the draft then keeps the target's tensor)
ggml_tensor * llama_dspark_embd_dev_copy(const ggml_tensor * src, ggml_backend_dev_t dev, ggml_context_ptr & ctx_out,
        ggml_backend_buffer_ptr & buf_out);
