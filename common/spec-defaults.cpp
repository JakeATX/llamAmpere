#include "spec-defaults.h"
#include "speculative-inkdepth.h"

#include "arch-defaults.h"

#include "common.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "log.h"
#include "speculative.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

// one row per model family, keyed by general.architecture and whether the model has an MTP head
static const common_speculative_family_default common_speculative_family_defaults[] = {
    // Qwen3.8 (qwen35) with the built-in MTP head: the production settings
    // (--spec-type draft-mtp-adaptive --spec-draft-n-max 4 --spec-draft-n-min-adaptive 3 --spec-draft-p-min 0,
    //  vocab map auto). The draft KV types stay unset on purpose: the resolver in common_base_params_to_speculative
    //  picks DSJ (draft on the trunk's SJ-KVaRN regions + the RM24H shortlist) or tq5_0/turbo4 over a SJ-KVaRN trunk;
    //  DSJ needs -ctkd/-ctvd unset (an explicit draft type or vocab map turned it off in the 11-19 loss). The report
    //  (common/arch-defaults.cpp) prints the resolved draft KV.
    // Drafters that ship as a separate file beside the model (Inkling 2 DSpark) are rows of the arch table
    // (common/arch-defaults.cpp, "spec"), applied by common_speculative_apply_file_default below.
    { "qwen35", true, "MTP drafter", COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, 4, 3, 0.0f, "auto" },
};

const common_speculative_family_default * common_speculative_family_default_find(const std::string & arch, uint32_t n_nextn) {
    for (const auto & row : common_speculative_family_defaults) {
        if (arch == row.arch && (n_nextn > 0) == row.mtp_head) {
            return &row;
        }
    }
    return nullptr;
}

std::string common_speculative_family_defaults_str() {
    std::string result;
    for (const auto & row : common_speculative_family_defaults) {
        if (!result.empty()) {
            result += ", ";
        }
        result += string_format("%s%s: %s", row.arch, row.mtp_head ? " with an MTP head" : "",
                common_speculative_type_to_str(row.type).c_str());
    }
    return result;
}

// the drafting was chosen explicitly (or inferred from a draft model), the family default must not apply
static bool common_speculative_family_default_allowed(const common_params_speculative & spec) {
    if (spec.user_set & COMMON_PARAMS_SPECULATIVE_USER_TYPE) {
        return false;
    }
    if (spec.types != std::vector<enum common_speculative_type>{ COMMON_SPECULATIVE_TYPE_NONE }) {
        return false;
    }
    return !spec.has_dft() && !spec.draft.eagle3 && !spec.draft.dflash;
}

const common_speculative_family_default * common_speculative_apply_family_default(
        common_params_speculative & spec, const std::string & arch, uint32_t n_nextn) {
    if (!common_speculative_family_default_allowed(spec)) {
        return nullptr;
    }

    const common_speculative_family_default * row = common_speculative_family_default_find(arch, n_nextn);
    if (row == nullptr) {
        return nullptr;
    }

    const bool user_n_max          = spec.user_set & COMMON_PARAMS_SPECULATIVE_USER_DRAFT_N_MAX;
    const bool user_n_min_adaptive = spec.user_set & COMMON_PARAMS_SPECULATIVE_USER_DRAFT_N_MIN_ADAPTIVE;

    int32_t n_max = user_n_max ? spec.draft.n_max : row->n_max;
    if (!user_n_max && user_n_min_adaptive) {
        n_max = std::max(n_max, spec.draft.n_min_adaptive);
    }
    if (n_max < 1) {
        // an explicit --spec-draft-n-max 0 asks for no drafting
        return nullptr;
    }

    // same list as --spec-type <type> gives: the handler appends to the default { none }
    spec.types.push_back(row->type);

    spec.draft.n_max          = n_max;
    spec.draft.n_min_adaptive = user_n_min_adaptive ? spec.draft.n_min_adaptive : std::min(row->n_min_adaptive, n_max);

    if (!(spec.user_set & COMMON_PARAMS_SPECULATIVE_USER_DRAFT_P_MIN)) {
        spec.draft.p_min = row->p_min;
    }
    if (!(spec.user_set & COMMON_PARAMS_SPECULATIVE_USER_DRAFT_VOCAB_MAP)) {
        spec.draft.vocab_map = row->vocab_map;
    }

    return row;
}

static uint32_t spec_gguf_get_uint(const gguf_context * ctx, const std::string & key) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0) {
        return 0;
    }
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(ctx, id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx, id);
        case GGUF_TYPE_INT32:  return (uint32_t) std::max(0, gguf_get_val_i32(ctx, id));
        default:               return 0;
    }
}

// general.architecture and the number of MTP (nextn) layers of a GGUF model, header only (no tensor data)
// n_nextn is 0 when the MTP block is declared but its tensors are not in the model files (e.g. a head shipped as a
// separate sidecar), so that the family default never asks the loader for tensors that are not there
static bool common_speculative_read_arch_nextn(const std::string & path, std::string & arch, uint32_t & n_nextn) {
    const struct gguf_init_params gguf_params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ nullptr,
    };

    gguf_context_ptr ctx(gguf_init_from_file(path.c_str(), gguf_params));
    if (!ctx) {
        return false;
    }

    const int64_t arch_id = gguf_find_key(ctx.get(), "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(ctx.get(), arch_id) != GGUF_TYPE_STRING) {
        return false;
    }

    arch    = gguf_get_val_str(ctx.get(), arch_id);
    n_nextn = spec_gguf_get_uint(ctx.get(), arch + ".nextn_predict_layers");
    if (n_nextn == 0) {
        return true;
    }

    // the MTP blocks are the last ones: look for the eh_proj of the last block, in every split
    const uint32_t n_block = spec_gguf_get_uint(ctx.get(), arch + ".block_count");
    const std::string name = "blk." + std::to_string(n_block > 0 ? n_block - 1 : 0) + ".nextn.eh_proj.weight";

    bool found = gguf_find_tensor(ctx.get(), name.c_str()) >= 0;

    const uint32_t n_split = spec_gguf_get_uint(ctx.get(), "split.count");
    if (!found && n_split > 1) {
        std::vector<char> prefix(path.size() + 64);
        if (llama_split_prefix(prefix.data(), prefix.size(), path.c_str(), 0, (int32_t) n_split) > 0) {
            std::vector<char> split_path(path.size() + 64);
            for (uint32_t i = 1; i < n_split && !found; ++i) {
                if (llama_split_path(split_path.data(), split_path.size(), prefix.data(), (int32_t) i, (int32_t) n_split) <= 0) {
                    break;
                }
                gguf_context_ptr ctx_split(gguf_init_from_file(split_path.data(), gguf_params));
                found = ctx_split && gguf_find_tensor(ctx_split.get(), name.c_str()) >= 0;
            }
        }
    }

    if (!found) {
        LOG_INF("speculative: %s declares %u MTP layer(s) but the model files hold no MTP tensors ('%s')\n",
                arch.c_str(), n_nextn, name.c_str());
        n_nextn = 0;
    }

    return true;
}

void common_speculative_apply_mtp_sampling_default(common_params & params) {
    const auto & types = params.speculative.types;
    const bool mtp = std::find(types.begin(), types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != types.end() ||
                     std::find(types.begin(), types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE) != types.end();
    // INKSPEC2: the DSpark drafter samples its draft (q), so the same mode-biased accept applies to it (as with MTP)
    const bool dspark = std::find(types.begin(), types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK) != types.end();
    if ((mtp || dspark) && !params.sampling.spec_lossy_explicit && !params.sampling.spec_lossy) {
        params.sampling.spec_lossy = true;
        LOG_INF("FYI: mode-biased MTP (--spec-mb-mtp) is on by default with %s drafting. "
                "To turn it off: --no-spec-mb-mtp\n", mtp ? "MTP" : "DSpark");
    }
}

void common_speculative_apply_dspark_default(common_params & params) {
    auto & spec = params.speculative;
    if (spec.draft.dspark_adaptive || spec.draft.dspark_n_max_prev > 0 ||
            std::find(spec.types.begin(), spec.types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK) == spec.types.end()) {
        return;
    }
    // INKSPEC2 (3090 screen 11-22): with mb on, the Inkling DSpark drafter is fastest at fixed depth 4 (adaptive 3..5
    // and fixed 5/6 were slower). LLAMA_DSPARK_MB_DEPTH=N picks the depth (0 = keep n_max as before, the kill switch),
    // LLAMA_DSPARK_ADAPT=1 drafts at the adaptive depth 3..5 instead. A draft GGUF without the Markov head (not the
    // Inkling DSpark) keeps its depth.
    const char * ed = getenv("LLAMA_DSPARK_MB_DEPTH");
    const int    depth = ed ? atoi(ed) : 4;
    const char * ea = getenv("LLAMA_DSPARK_ADAPT");
    const bool   adapt = ea && atoi(ea) != 0;
    if (!spec.draft.mparams.path.empty()) {
        const struct gguf_init_params gp = { /* .no_alloc = */ true, /* .ctx = */ nullptr };
        gguf_context_ptr ctx(gguf_init_from_file(spec.draft.mparams.path.c_str(), gp));
        if (ctx && gguf_find_key(ctx.get(), "dflash.dspark.markov_rank") < 0) {
            return;
        }
    }
    if (params.sampling.spec_lossy &&
            !(spec.user_set & (COMMON_PARAMS_SPECULATIVE_USER_DRAFT_N_MAX | COMMON_PARAMS_SPECULATIVE_USER_DRAFT_N_MIN_ADAPTIVE))) {
        if (adapt) {
            spec.draft.dspark_adaptive   = true;
            spec.draft.dspark_n_max_prev = spec.draft.n_max;
            spec.draft.n_max             = 5;
            spec.draft.n_min_adaptive    = 3;
        } else if (depth > 0) {
            spec.draft.dspark_n_max_prev = spec.draft.n_max;
            spec.draft.n_max             = depth;
            // INKDEPTH (LLAMA_INKDEPTH=0 = off): walk up to LLAMA_INKDEPTH_MAX slots and keep the best tokens/ms prefix
            // per round; `depth` stays the fixed depth for rounds where several sequences draft
            if (common_inkdepth::cfg().on && common_inkdepth::cfg().max_depth > depth) {
                spec.draft.dspark_inkdepth_legacy = depth;
                spec.draft.n_max                  = common_inkdepth::cfg().max_depth;
            }
        }
    } else if (spec.user_set & COMMON_PARAMS_SPECULATIVE_USER_DRAFT_N_MIN_ADAPTIVE) {
        spec.draft.dspark_adaptive = true; // an explicit adaptive floor asks for the adaptive depth
    }
}

// fork DEFPATH (SPECROW): an arch whose drafter ships as a file beside the model (the arch table's "spec" row, e.g.
// inkling2: DSpark, -ngld 99 -ubd 64, draft KV F16 explicit). Only when no drafting was chosen, the file is there, the
// whole model is on the GPU and the smallest used GPU is at least min_gpu_mib (on small cards the drafter's VRAM
// comes out of the expert cache and measured slower). LLAMA_ARCH_DEFAULTS=0 or --spec-type none: no drafter.
static ggml_type spec_kv_type_from_name(const std::string & name) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * n = ggml_type_name((ggml_type) t);
        if (n && name == n) {
            return (ggml_type) t;
        }
    }
    return GGML_TYPE_COUNT;
}

static void common_speculative_apply_file_default(common_params & params, const std::string & arch) {
    auto & spec = params.speculative;
    common_arch_spec_default row;
    if (!common_arch_spec_default_get(arch, row) || !row.preset.empty()) {
        return; // a preset row is applied by arg.cpp under its card preset only
    }
    const auto type = common_speculative_type_from_name(row.type);
    if (type == COMMON_SPECULATIVE_TYPE_COUNT) {
        LOG_WRN("speculative: arch table drafter type '%s' for %s is unknown, no drafter\n", row.type.c_str(), arch.c_str());
        return;
    }
    const std::string file = common_arch_spec_default_file(row, params.model.path);
    const size_t min_mib = common_arch_min_gpu_total_mib(params);
    const char * why_not = nullptr;
    if (file.empty()) {
        why_not = "the drafter file is not beside the model";
    } else if (params.n_gpu_layers == 0 || params.n_cpu_moe_auto ||
            (!params.tensor_buft_overrides.empty() && params.tensor_buft_overrides[0].pattern != nullptr)) {
        // (arg.cpp pads tensor_buft_overrides with null entries for the fit: only a real pattern counts)
        why_not = "the model is not fully on the GPU (-ngl 0, --cpu-moe or -ot)";
    } else if (min_mib < row.min_gpu_mib) {
        why_not = "the card is below the drafter's card line";
    }
    if (why_not) {
        LOG_INF("speculative: %s drafter %s not applied: %s (%s)\n", arch.c_str(), row.model.c_str(), why_not, row.why.c_str());
        return;
    }
    spec.draft.mparams.path    = file;
    spec.draft.mparams.hf_file = file;
    spec.types.push_back(type);
    if (row.ngl != -1 && spec.draft.n_gpu_layers == -1) {
        spec.draft.n_gpu_layers = row.ngl;
    }
    if (row.ubatch > 0 && spec.draft.n_ubatch < 0) {
        spec.draft.n_ubatch = row.ubatch;
    }
    // explicit draft KV (not inherited from the trunk); LLAMA_DSPARK_DRAFT_SJKVARN=1 keeps its opt-in path
    const char * dsj = getenv("LLAMA_DSPARK_DRAFT_SJKVARN");
    if (!(dsj && *dsj && atoi(dsj) != 0) && !spec.draft.sj_kvarn_explicit) {
        if (!row.type_k.empty() && spec.draft.cache_type_k == GGML_TYPE_COUNT) {
            spec.draft.cache_type_k = spec_kv_type_from_name(row.type_k);
        }
        if (!row.type_v.empty() && spec.draft.cache_type_v == GGML_TYPE_COUNT) {
            spec.draft.cache_type_v = spec_kv_type_from_name(row.type_v);
        }
    }
    common_speculative_apply_mtp_sampling_default(params);
    common_speculative_apply_dspark_default(params);
    LOG_INF("speculative: %s on by default for %s (arch table): %s, -ngld %d -ubd %d, draft KV %s/%s, n-max %d; "
            "--spec-type none or LLAMA_ARCH_DEFAULTS=0 disables\n", file.c_str(), arch.c_str(), row.type.c_str(),
            spec.draft.n_gpu_layers, spec.draft.n_ubatch,
            spec.draft.cache_type_k == GGML_TYPE_COUNT ? "trunk" : ggml_type_name(spec.draft.cache_type_k),
            spec.draft.cache_type_v == GGML_TYPE_COUNT ? "trunk" : ggml_type_name(spec.draft.cache_type_v), spec.draft.n_max);
}

// A card preset whose arch row names a file drafter (24dense: DSpine b16 beside the model) swaps the preset's own
// drafter (the built-in MTP head) for it at one slot. Called by arg.cpp after the preset values are in place; `given`
// tells which flags the user set (flag or env). Any explicit drafter choice, <kill_env>=0, LLAMA_ARCH_DEFAULTS=0, a
// missing file or a card below the row's line keeps the preset's drafter. Explicit --fit, -ctkd/-ctvd and
// --[no-]spec-mb-mtp are kept.
bool common_speculative_apply_preset_file_default(common_params & params, const std::function<bool(const std::string &)> & given) {
    auto & spec = params.speculative;
    if (params.card_preset.empty() || params.n_parallel != 1 || params.model.path.empty()) {
        return false;
    }
    for (const char * key : { "--spec-type", "--spec-draft-model", "--spec-draft-hf", "--spec-draft-n-max", "--spec-draft-n-min",
                              "--spec-draft-n-min-adaptive", "--spec-draft-p-min", "--spec-draft-sampling", "--spec-chain" }) {
        if (given(key)) {
            return false; // the user tuned the drafting: keep it
        }
    }
    if (spec.has_dft() || spec.draft.eagle3 || spec.draft.dflash) {
        return false;
    }
    std::string arch;
    uint32_t n_nextn = 0;
    if (!common_speculative_read_arch_nextn(params.model.path, arch, n_nextn)) {
        return false;
    }
    common_arch_spec_default row;
    if (!common_arch_spec_default_get(arch, row) || row.preset != params.card_preset) {
        return false;
    }
    const auto type = common_speculative_type_from_name(row.type);
    if (type == COMMON_SPECULATIVE_TYPE_COUNT) {
        LOG_WRN("speculative: arch table drafter type '%s' for %s is unknown, preset drafter kept\n", row.type.c_str(), arch.c_str());
        return false;
    }
    const std::string file = common_arch_spec_default_file(row, params.model.path);
    const char * why_not = nullptr;
    if (common_arch_spec_default_killed(row)) {
        why_not = "turned off by env";
    } else if (file.empty()) {
        why_not = "the drafter file is not beside the model";
    } else if (common_arch_min_gpu_total_mib(params) < row.min_gpu_mib) {
        why_not = "the card is below the drafter's card line";
    }
    if (why_not) {
        LOG_INF("card preset %s: drafter %s not applied: %s; the preset's own drafter is used\n",
                params.card_preset.c_str(), row.model.c_str(), why_not);
        return false;
    }
    spec.types                 = { type };
    spec.draft.mparams.path    = file;
    spec.draft.mparams.hf_file = file;
    if (row.ngl != -1 && !given("--spec-draft-ngl")) {
        spec.draft.n_gpu_layers = row.ngl;
    }
    if (row.n_max >= 0) {
        spec.draft.n_max = row.n_max;
    }
    if (row.n_min >= 0) {
        spec.draft.n_min = row.n_min;
    }
    if (row.p_min >= 0.0f) {
        spec.draft.p_min = row.p_min;
    }
    if (!row.sampling.empty()) {
        spec.draft.probabilistic = row.sampling == "probabilistic";
    }
    if (row.mb_mtp != -1 && !params.sampling.spec_lossy_explicit) {
        params.sampling.spec_lossy = row.mb_mtp == 1;
    }
    if (!row.type_k.empty() && !given("--spec-draft-type-k")) {
        spec.draft.cache_type_k = spec_kv_type_from_name(row.type_k);
    }
    if (!row.type_v.empty() && !given("--spec-draft-type-v")) {
        spec.draft.cache_type_v = spec_kv_type_from_name(row.type_v);
    }
    if (!row.fit.empty() && !given("--fit")) {
        params.fit_params = row.fit != "off";
    }
    LOG_INF("card preset %s: %s on by default (arch table): %s, n-max %d, n-min %d, p-min %g, sampling %s, mb %s, "
            "draft KV %s/%s, fit %s; %s=0 keeps the MTP head\n", params.card_preset.c_str(), file.c_str(), row.type.c_str(),
            spec.draft.n_max, spec.draft.n_min, (double) spec.draft.p_min, spec.draft.probabilistic ? "probabilistic" : "greedy",
            params.sampling.spec_lossy ? "on" : "off",
            spec.draft.cache_type_k == GGML_TYPE_COUNT ? "trunk" : ggml_type_name(spec.draft.cache_type_k),
            spec.draft.cache_type_v == GGML_TYPE_COUNT ? "trunk" : ggml_type_name(spec.draft.cache_type_v),
            params.fit_params ? "on" : "off", row.kill_env.c_str());
    return true;
}

bool common_speculative_apply_model_default(common_params & params) {
    auto & spec = params.speculative;

    // no I/O when the user already chose the drafting
    if (params.model.path.empty() || !common_speculative_family_default_allowed(spec)) {
        return true;
    }

    std::string arch;
    uint32_t n_nextn = 0;
    if (!common_speculative_read_arch_nextn(params.model.path, arch, n_nextn)) {
        return true; // the model load reports an unreadable file
    }

    const common_speculative_family_default * row = common_speculative_apply_family_default(spec, arch, n_nextn);
    if (row == nullptr) {
        common_speculative_apply_file_default(params, arch);
        return true;
    }

    const bool adaptive = row->type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE;
    if (adaptive && (spec.draft.n_min_adaptive < 1 || spec.draft.n_min_adaptive > spec.draft.n_max)) {
        LOG_ERR("error: --spec-draft-n-min-adaptive must be in [1, --spec-draft-n-max]\n");
        return false;
    }

    common_speculative_apply_mtp_sampling_default(params);

    const std::string n_min_adaptive = adaptive ? string_format(", n-min-adaptive %d", spec.draft.n_min_adaptive) : "";

    LOG_INF("speculative: %s on by default for %s (nextn=%u): %s, n-max %d%s, p-min %g, vocab map %s; --spec-type none disables\n",
            row->drafter, arch.c_str(), n_nextn,
            common_speculative_type_to_str(row->type).c_str(),
            spec.draft.n_max, n_min_adaptive.c_str(),
            (double) spec.draft.p_min,
            spec.draft.vocab_map.c_str());

    return true;
}
