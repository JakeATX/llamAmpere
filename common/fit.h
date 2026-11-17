#pragma once

#include "ggml.h"
#include "llama.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

// fork: optional host memory, planned after required model/context allocations.
enum common_fit_ram_consumer {
    COMMON_FIT_RAM_EXPERTS,
    COMMON_FIT_RAM_STAGING,
    COMMON_FIT_RAM_PREFETCH,
    COMMON_FIT_RAM_PROMPT,
    COMMON_FIT_RAM_NGRAM,
    COMMON_FIT_RAM_EVAL,
    COMMON_FIT_RAM_COUNT,
};

struct common_fit_ram_request {
    uint64_t bytes = 0; // zero keeps an unused consumer off
    bool explicit_size = false;
};

struct common_fit_ram_result {
    bool floor_met = false;
    uint64_t available = 0;
    uint64_t floor = 5ull << 30;
    uint64_t automatic_budget = 0;
    std::array<uint64_t, COMMON_FIT_RAM_COUNT> bytes = {};
};

common_fit_ram_result common_fit_ram_plan(uint64_t available,
        const std::array<common_fit_ram_request, COMMON_FIT_RAM_COUNT> & requests);

size_t common_fit_vram_floor(bool display_attached);
size_t common_fit_vram_floor_detect(const std::string & drm_root, bool display_session);
struct common_params;
std::vector<size_t> common_fit_vram_targets(const common_params & params, size_t floor);

uint64_t common_fit_eval_cache_ram_bytes(uint64_t available, bool fit);
uint64_t common_fit_eval_cache_load_bytes(uint64_t available, bool fit, uint64_t n_tokens);

struct common_params_speculative_ngram_cache;
void common_fit_ngram_cache_load_limits(uint64_t available, bool fit, common_params_speculative_ngram_cache & params, uint32_t n_seq = 1);

bool common_fit_eval_cache_save_fits(uint64_t available, bool fit, uint64_t n_tokens, uint64_t spec_bytes);

// Resolve the server prompt cache only after model, contexts and draft have loaded.
// Positive/zero limits and the explicit half-free setting keep their meaning.
int32_t common_fit_prompt_cache_ram_mib(uint64_t total, uint64_t available,
        int32_t requested_mib, bool fit, std::string & why, uint64_t explicit_free = 0);

struct common_moe_cache_params;

const char * common_moe_cache_tensor_override_pattern();

struct common_moe_cache_fit_device_input {
    int physical_device = -1;
    int compute_capability = 0;
    int64_t free_bytes = 0;
    size_t used_bytes = 0;
    size_t margin_bytes = 0;
};

struct common_moe_cache_fit_shape_input {
    enum ggml_type type = GGML_TYPE_COUNT;
    size_t expert_size = 0;
    size_t tensor_bytes = 0;
    size_t scratch_bytes = 0;
    size_t pool_bytes = 0;
    bool cacheable = false;
};

struct common_moe_cache_fit_device {
    int physical_device = -1;
    int compute_capability = 0;
    int64_t free_bytes = 0;
    int64_t used_bytes = 0;
    size_t reserve_bytes = 0;
    size_t cache_bytes = 0;
};

struct common_moe_cache_fit_result {
    bool feasible = false;
    std::string reason;
    std::vector<common_moe_cache_fit_device> devices;
    size_t expert_bytes = 0;
    size_t cache_bytes = 0;
    size_t minimum_device_bytes = 0;
};

common_moe_cache_fit_result common_moe_cache_plan_fit(
        const std::vector<common_moe_cache_fit_device_input> & devices,
        const std::vector<common_moe_cache_fit_shape_input> & shapes,
        size_t reserve_bytes,
        size_t budget_bytes,
        int min_devices,
        size_t minimum_slab_bytes = 0);

enum common_params_fit_status {
    COMMON_PARAMS_FIT_STATUS_SUCCESS = 0, // found allocations that are projected to fit
    COMMON_PARAMS_FIT_STATUS_FAILURE = 1, // could not find allocations that are projected to fit
    COMMON_PARAMS_FIT_STATUS_ERROR   = 2, // a hard error occurred, e.g. because no model could be found at the specified path
    COMMON_PARAMS_FIT_STATUS_NO_ROOM = 3, // no allocation can fit, even with every layer's weights in system memory
};

// fork: classify measured allocations before treating requested margins as a proven fit.
common_params_fit_status common_fit_vram_margin_status(int64_t free_bytes, size_t used_bytes, size_t margin_bytes);
int64_t common_fit_vram_margin_surplus(int64_t free_bytes, const std::vector<int64_t> & margins);
struct common_fit_vram_layer_result {
    common_params_fit_status status;
    uint32_t step;
};
common_fit_vram_layer_result common_fit_vram_layer_step(uint32_t delta, int64_t low, int64_t high, int64_t target, int64_t free_bytes);

// fork: require a proven fit with at least `floor` (never below 300 MiB) left on every device.
bool common_fit_verify_compact_safe(bool complete, const std::vector<int64_t> & headroom, const std::vector<int64_t> & margins, int64_t floor = 300ll * 1024 * 1024);

// a second model that shares the devices of the main model, e.g. a draft model
//   - its context follows the context of the main model, so its memory is measured again whenever that context changes
//   - shares_model tells the fit that the weights are already counted in the main model, as for an MTP context
//   - shares_compute tells the fit that the context borrows the compute buffers of the main context, as an MTP context
//     does on a single device (LLAMA_SHARED_COMPUTE); it then adds its KV cache and only the growth of the shared
//     compute buffer beyond the main context's own need
struct common_fit_extra_model {
    std::vector<std::string> paths_model;
    llama_model_params * mparams;
    llama_context_params * cparams;
    bool shares_model;
    bool shares_compute;
};

// memory of the extra model on one device, in bytes
struct common_fit_extra_memory {
    size_t model;
    size_t context;
    size_t compute;
};

// memory the extra model adds to one device at context size n_ctx, from a measurement at n_ctx_measured:
//   - the weights do not depend on the context, the KV cache grows linearly with it (rounded up)
//   - a context that shares compute buffers grows the main context's buffer (compute_main) to the larger of the two
//     graphs (ggml_gallocr donor rule), so it adds max(0, measured compute - compute_main)
common_fit_extra_memory common_fit_extra_memory_at(
        const common_fit_extra_memory & measured,
                             uint32_t   n_ctx_measured,
                             uint32_t   n_ctx,
                               size_t   compute_main,
                                 bool   shares_compute);

// fits mparams and cparams to free device memory (assumes system memory is unlimited)
//   - returns true if the parameters could be successfully modified to fit device memory
//   - this function is NOT thread safe because it modifies the global llama logger state
//   - only parameters that have the same value as in llama_default_model_params are modified
//     with the exception of the context size which may be reduced when unset or bounded by n_ctx_preset_max
common_params_fit_status common_fit_params(
     const std::vector<std::string> & paths_model,
                 llama_model_params * mparams,
               llama_context_params * cparams,
                              float * tensor_split,          // writable buffer for tensor split, needs at least llama_max_devices elements
   llama_model_tensor_buft_override * tensor_buft_overrides, // writable buffer for overrides, needs at least llama_max_tensor_buft_overrides elements
           common_moe_cache_params * moe_cache,
                             size_t * margins,               // margins of memory to leave per device in bytes
                           uint32_t   n_ctx_min,             // minimum context size to set when trying to reduce memory use
      const common_fit_extra_model * extra,                  // model to fit alongside the main one, nullptr if there is none
                     ggml_log_level   log_level,             // minimum log level to print during fitting, lower levels go to debug log
                               bool * margins_verified = nullptr,
                           uint32_t   n_ctx_preset_max = 0);

// print estimated memory to stdout
void common_fit_print(
     const std::vector<std::string> & paths_model,
                 llama_model_params * mparams,
               llama_context_params * cparams);

void common_memory_breakdown_print(const llama_context * ctx);

// one row of the memory breakdown: a device, the host or another buffer type
struct common_memory_breakdown_row {
    enum kind_t { DEVICE, HOST, BUFFER_TYPE };

    kind_t      kind = DEVICE;
    std::string name;        // device or buffer type name, "Host" for the host row
    std::string description; // device description, empty otherwise
    int64_t     total = 0;   // device memory in bytes, 0 unless kind == DEVICE
    int64_t     free  = 0;   // free device memory in bytes, 0 unless kind == DEVICE
    size_t      model   = 0; // bytes held by this context, by kind of buffer
    size_t      context = 0;
    size_t      compute = 0;
};

// memory a model needs at the min and max context --fit may pick
struct common_memory_reports {
    uint32_t n_ctx_train = 0;
    uint32_t n_ctx_max   = 0; // the largest context that fits the free device memory right now
    uint32_t n_ctx_min   = 0; // --fit-ctx (the smallest context --fit may pick), with the placement set in params
    std::vector<common_memory_breakdown_row> info_max; // breakdown at n_ctx_max
    std::vector<common_memory_breakdown_row> info_min; // breakdown at n_ctx_min
};

// rows of common_memory_breakdown_print: model devices first, then host
std::vector<common_memory_breakdown_row> common_memory_breakdown_rows(const llama_context * ctx);

// load model + context with no_alloc and return the memory breakdown, params are used as-is
// n_ctx_train and n_ctx are optional outputs; throws on failure
std::vector<common_memory_breakdown_row> common_fit_measure(
     const std::vector<std::string> & paths_model,
           const llama_model_params * mparams,
         const llama_context_params * cparams,
                           uint32_t * n_ctx_train,
                           uint32_t * n_ctx,
                     ggml_log_level   log_level);

struct common_params;

// measure min and max without allocating, params may be modified as with --fit; throws on failure
common_memory_reports common_measure_memory(common_params & params);

struct common_device_memory_data {
    int64_t total;
    int64_t free;
    size_t  model;
    size_t  context;
    size_t  compute;
};

using common_device_memory_data_vec = std::vector<common_device_memory_data>;

// Load a model + context with no_alloc and return the per-device memory breakdown.
common_device_memory_data_vec common_get_device_memory_data(
     const std::vector<std::string> & paths_model,
           const llama_model_params * mparams,
         const llama_context_params * cparams,
    std::vector<ggml_backend_dev_t> & devs,
                           uint32_t & hp_ngl,
                           uint32_t & hp_ngl_skip,
                           uint32_t & hp_n_ctx_train,
                           uint32_t & hp_n_expert,
                     ggml_log_level   log_level);

// Load a parent model + context with no_alloc, then return the per-device memory breakdown of a child model + context.
common_device_memory_data_vec common_get_device_memory_data_with_parent(
     const std::vector<std::string> & paths_model,
           const llama_model_params * mparams,
         const llama_context_params * cparams,
     const std::vector<std::string> & paths_parent,
           const llama_model_params * mparams_parent,
         const llama_context_params * cparams_parent,
    std::vector<ggml_backend_dev_t> & devs,
                           uint32_t & hp_ngl,
                           uint32_t & hp_ngl_skip,
                           uint32_t & hp_n_ctx_train,
                           uint32_t & hp_n_expert,
                     ggml_log_level   log_level);
