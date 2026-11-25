#pragma once

#include "common.h"

#include <string>
#include <vector>

// fork DEFPATH: per-arch best-config reference (common/arch-defaults.cpp, one JSON row per model family).
//
// A new model or a new release must not inherit slow defaults without anyone seeing it. The table lists, per
// general.architecture, the measured best settings: drafter and its KV, ub ladder ceiling, lean prompt, KV type,
// the engine's default-ON switches and the flags still under gate. At boot:
//   - common_arch_defaults_apply() sets the table's env values the user did not set (LLAMA_LEAN_PROMPT, ...);
//   - common_arch_defaults_report() prints one WARN block: every lever, on/off/overridden, and why;
//   - an arch with no row gets a WARN that every lever is at the engine default.
// The gate script scripts/defpath/defpath_gate.py reads the same JSON from this file (between the ARCHJSON markers)
// and compares the default run with the best config row.
//
// fork DEFSYNC: this table is the ONE source of per-arch default settings. The 0.7.5 Inkling env defaults that sat in
// llama_model_load (src/llama.cpp) are its Inkling row's apply_env now; the switches that are default-ON in code are
// listed in env_on so the report shows them. Frontends that load or fit a model without common_init (llama-bench,
// batched-bench, llama-fit-params) call common_arch_defaults_apply() themselves, so they get the whole row (on a
// 24 GB card with Inkling that includes LLAMA_LEAN_PROMPT=0, as in the server). A program that links libllama alone
// (examples, bindings, tests) gets the engine defaults: every apply_env value unset.
//
// Kill switches: LLAMA_ARCH_DEFAULTS=0 (table applies nothing: no env, no auto drafter, no table ub ceiling),
// LLAMA_ENGAGED_REPORT=0 (no report block). LLAMA_UBDEFAULT=0 is UBDEFAULT's own switch (see llama-context.cpp).

// false with LLAMA_ARCH_DEFAULTS=0
bool common_arch_defaults_enabled();

// general.architecture of the GGUF header (cached per path); empty if unreadable
std::string common_arch_of_model(const std::string & path);

// smallest total memory (MiB) of the GPUs this run uses (params.devices, else all GPU devices); 0 = no GPU
size_t common_arch_min_gpu_total_mib(const common_params & params);

// the 24 GB-class auto ub ceiling for this arch (UBDEFAULT; 4096 for an arch without a row), and whether it only
// applies when -c is given. LLAMA_UB_CEILING still wins (caller).
int32_t common_arch_ub_ceiling_24g(const std::string & arch, bool * needs_ctx);

// the table's -b for a 24 GB-class card (0 = UBDEFAULT rule)
int32_t common_arch_batch_24g(const std::string & arch);

// set the env values of the arch row that are not set yet (before the model load); remembers which were the user's
void common_arch_defaults_apply(const common_params & params);

// the speculative row of the arch (drafter file beside the model): fields for spec-defaults.cpp
struct common_arch_spec_default {
    std::string type;       // --spec-type, e.g. "draft-dspark"
    std::string model;      // draft GGUF file name beside the target model (the published name)
    std::vector<std::string> model_alt; // older names of the same file, tried after `model`
    int32_t     ngl = -1;   // --spec-draft-ngl
    int32_t     ubatch = 0; // --spec-draft-ubatch (0 = keep)
    std::string type_k;     // explicit draft KV types ("" = engine resolver)
    std::string type_v;
    size_t      min_gpu_mib = 0; // only on cards at least this large (small cards: the drafter costs expert cache)
    // preset rows (applied by arg.cpp for that card preset at one slot, in place of the preset's own drafter):
    std::string preset;     // "" = arch row for any run; else only under this --preset
    std::string kill_env;   // <env>=0 keeps the preset's own drafter
    std::string sampling;   // --spec-draft-sampling ("" = keep)
    int32_t     n_max = -1; // --spec-draft-n-max (-1 = keep)
    int32_t     n_min = -1; // --spec-draft-n-min (-1 = keep)
    float       p_min = -1.0f; // --spec-draft-p-min (< 0 = keep)
    int32_t     mb_mtp = -1;   // --spec-mb-mtp: 1 on, 0 off, -1 keep
    std::string fit;        // --fit ("" = keep)
    std::string why;
};
bool common_arch_spec_default_get(const std::string & arch, common_arch_spec_default & out);
// true when the row's kill switch env is set to 0
bool common_arch_spec_default_killed(const common_arch_spec_default & row);
// the row's drafter file beside the model: `model` first, then each of `model_alt`; "" = none of them is there
std::string common_arch_spec_default_file(const common_arch_spec_default & row, const std::string & model_path);

// one WARN block after the main context is created
void common_arch_defaults_report(const common_params & params, const llama_model * model, const llama_context * ctx);
