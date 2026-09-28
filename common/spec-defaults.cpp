#include "spec-defaults.h"

#include "common.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "log.h"
#include "speculative.h"

#include <algorithm>
#include <string>
#include <vector>

// one row per model family, keyed by general.architecture and whether the model has an MTP head
static const common_speculative_family_default common_speculative_family_defaults[] = {
    // Qwen3.8 (qwen35) with the built-in MTP head: the production settings, fixed draft depth 4
    // (--spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0, vocab map auto); n-min-adaptive is the
    // --spec-draft-n-min-adaptive default and only matters for draft-mtp-adaptive, which stays available as an
    // explicit --spec-type; the draft KV cache types are left unset and inherit the trunk -ctk/-ctv
    { "qwen35", true, "MTP drafter", COMMON_SPECULATIVE_TYPE_DRAFT_MTP, 4, 3, 0.0f, "auto" },
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
        return true;
    }

    const bool adaptive = row->type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE;
    if (adaptive && (spec.draft.n_min_adaptive < 1 || spec.draft.n_min_adaptive > spec.draft.n_max)) {
        LOG_ERR("error: --spec-draft-n-min-adaptive must be in [1, --spec-draft-n-max]\n");
        return false;
    }

    const std::string n_min_adaptive = adaptive ? string_format(", n-min-adaptive %d", spec.draft.n_min_adaptive) : "";

    LOG_INF("speculative: %s on by default for %s (nextn=%u): %s, n-max %d%s, p-min %g, vocab map %s; --spec-type none disables\n",
            row->drafter, arch.c_str(), n_nextn,
            common_speculative_type_to_str(row->type).c_str(),
            spec.draft.n_max, n_min_adaptive.c_str(),
            (double) spec.draft.p_min,
            spec.draft.vocab_map.c_str());

    return true;
}
