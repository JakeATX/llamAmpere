#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <vector>

struct llama_model;

// keeps recently and repeatedly used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    // cpu_share is the share of the misses computed on the CPU instead of uploaded
    llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share);
    ~llama_moe_cache();

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    // cpu_part: the graph computes the misses in the CPU map of the layer (see get_cpu_map)
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used, bool cpu_part) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // the experts of layer il computed on the CPU: I32 [1, n_expert] in host memory, e for these experts, -1 for the others
    // nullptr if no misses of layer il are computed on the CPU for ubatches of n_tokens tokens
    ggml_tensor * get_cpu_map(int32_t il, int64_t n_tokens) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    // fork (INKSTREAM): free the device banks for a prompt that streams its experts (returns the bytes freed); the slot
    // maps and the LRU stay, and no graph may use the cache until resume(). resume() allocates the banks again and
    // uploads the experts the slots held (same experts in the same slots). false: no VRAM, still suspended.
    size_t suspend();
    bool   resume();
    bool   suspended() const;
    // fork (CACHELOAN): while suspended, rebuild the banks for `size` bytes (keeps the hottest experts; resume()
    // uploads them). false: not possible here (several devices, CPU share, trace), nothing changed. size(): bank bytes.
    bool   resize(size_t size);
    size_t size() const;

    // fork (SURGE-PROMPT): whether the scheduler offloads large ops on host weights (cparams.op_offload). Only read by the
    // few-row CPU tier (LLAMA_MOE_CACHE_FEWROW_CPU): with it off, the CPU part of a prompt chunk stays on the CPU.
    void   set_op_offload(bool on);

    // fork (INKCACHEUX): see llama-moe-cache-policy.h. window_end(): a decode step was submitted (GS8,
    // LLAMA_MOE_CACHE_BOUNDARY=1; boundary() says whether it is on). request_end(): a request finished
    // (LLAMA_MOE_CACHE_REQ_STATS line, LLAMA_MOE_CACHE_HEAT periodic save). Both are no-ops with the flags unset.
    void   window_end();
    bool   boundary() const;
    void   request_end();
    // fork (ONEGRAPH, LLAMA_MOE_ONEGRAPH=1, llama-moe-onegraph.h): the GPU-side lookup of layer il for a decode ubatch
    // (<= 8 rows), nullptr when it is off or does not apply; og_prepare() runs before every graph of the context.
    ggml_tensor * og_slots(ggml_context * ctx, ggml_tensor * ids, int32_t il, int64_t n_tokens, int64_t n_expert_used) const;
    void          og_prepare();

    // fork (SPLITCAL): boot timings of the CPU expert path and an upload for LLAMA_MOE_CACHE_SPLIT_CAL and
    // LLAMA_MOE_CACHE_FEWROW_CPU=auto (see llama-moe-cache-policy.h); a no-op with both unset
    void   calibrate(int32_t n_threads, int32_t n_threads_batch);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;

// fork (INKSMALL2): host expert bytes the MoE cache could hold (looped layers once); 0 = nothing to cache
size_t llama_moe_cache_host_expert_bytes(const llama_model & model);
