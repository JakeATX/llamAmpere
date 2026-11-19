#include "arg.h"
#include "preset.h"
#include "peg-parser.h"
#include "log.h"
#include "download.h"

#include <fstream>
#include <sstream>
#include <filesystem>
#include <regex>

// fork: runtime card profiles. Values stay here so each QC change has one table to update.
const common_card_preset & common_card_preset_get(const std::string & name, bool headless) {
    static const std::vector<common_card_preset> headless_profiles = [] {
        // Timed ub QC: V07/ASYMCTX/QC_READY_asymctx_24gb.md; ceiling stays bounded by -ub.
        common_card_preset large {
            "24dense", "2026-11-17.3", "best known/recheck before ship", "", {
                {"--ctx-size", "262144"}, {"--parallel", "1"}, {"--batch-size", "4096"}, {"--fit", "on"},
                {"--ubatch-size", "1024"}, {"--gpu-layers", "99"}, {"--flash-attn", "on"},
                {"--cache-type-k", "sjkvarn4"}, {"--cache-type-v", "sjkvarn4"},
                {"--spec-type", "draft-mtp-adaptive"},
                {"--spec-draft-n-min-adaptive", "4"}, {"--spec-draft-n-max", "5"}, {"--spec-draft-n-min", "4"},
                {"--spec-draft-p-min", "0"}, {"--spec-draft-window", "4096"},
                {"--spec-draft-vocab-map", "auto"}, {"--gdn-replay", "1"}, {"--kv-unified", "1"}
            }, {
                {"LLAMA_GDN_REPLAY_LANES", "1"}, {"LLAMA_DEEPCTX_LADDER", "1"}
            }
        };
        // Small-card profiles keep packed masks and one sequence.
        // TODO: multi-user for small cards (future).
        common_card_preset small {
            "10dense", "2026-11-17.3", "best known/recheck before ship", "", {
                {"--ctx-size", "212992"}, {"--parallel", "1"}, {"--batch-size", "4096"},
                {"--ubatch-size", "512"}, {"--gpu-layers", "99"}, {"--flash-attn", "on"},
                {"--fit", "on"}, {"--cache-type-k", "sjkvarn4"}, {"--cache-type-v", "sjkvarn4"},
                {"--sjkvarn-body-type", "sjkvarn4"}, {"--cache-type-s", "q8_0"},
                {"--cache-ram", "0"}, {"--no-cache-disk", "1"},
                {"--override-tensor", "token_embd.weight=CPU"}, {"--threads", "6"},
                {"--load-mode", "none"}, {"--spec-type", "draft-mtp"},
                {"--spec-draft-n-max", "4"}, {"--spec-draft-p-min", "0"},
                {"--spec-draft-vocab-map", "auto:65536"}, {"--spec-draft-window", "2048"},
                {"--spec-draft-type-k", "q8_0"}, {"--spec-draft-type-v", "q8_0"},
                {"--sjkvarn-sink", "128"}, {"--sjkvarn-sink-type", "f16"},
                {"--sjkvarn-staging-type", "tq6_0"}, {"--sjkvarn-tail", "4096"},
                {"--sjkvarn-tail-max", "8192"}
            }, {
                {"LLAMA_DEEPCTX_LADDER", "1"},
                {"LLAMA_MTP_DRAFT_COMPUTE_LEAN", "1"}, {"LLAMA_KQ_MASK_PACKED", "1"},
                {"LLAMA_DRAFT_VOCAB_COMPACT", "0"}, {"LLAMA_LSA_MODE", "key"},
                {"LLAMA_LSA_MAP", "@model/lsa-qwen38-27b-d2048.gguf"},
                {"LLAMA_LSA_IDX_CACHE", "1"}, {"LLAMA_LSA_PACKED", "1"},
                {"LLAMA_LSA_KEEP", "51072"}, {"LLAMA_LSA_RESIDENT_MIB", "300.0"},
                {"LLAMA_LSA_IDX_Q8", "1"}, {"LLAMA_LSA_BLOCK", "128"},
                {"LLAMA_LSA_BLOCK_TAU", "2"}, {"LLAMA_LSA_SJ_SKIP", "1"},
                {"LLAMA_LSA_REFINE", "128"}, {"LLAMA_LSA_REFINE_MULT", "2"},
                {"LLAMA_LSA_SHARED", "1"}, {"LLAMA_LSA_FUSED_SEL", "1"},
                {"LLAMA_LSA_COMP", "0"}, {"LLAMA_LSA_GRAPH_REUSE", "0"},
                {"GGML_SJKVARN_TILE_SKIP", "1"},
                {"GGML_CUDA_LSA_RESIDENT_STAGE", "0"}, {"LLAMA_LSA_KEEP_FRAC", "0.195"},
                {"LLAMA_LSA_REFINE_SH", "1"}
            }
        };
        auto medium = small;
        medium.name = "12dense";
        medium.options.front().second = "262144";
        medium.env.emplace_back("GGML_CUDA_GRAPH_MAX_ROWS", "16");
        // 8 GB loop check: V07/PRESETS/SOL_HANDOFF.md, owner order 2026-11-17T04:03Z.
        // 0.7.1: the drafter is the mtp-dm.gguf overlay (127 MB sidecar beside the model, owner option B 2026-11-19)
        // instead of the file's own MTP layer; mb-MTP stays on as in the other profiles (owner small-card design:
        // LSA + MTP). With the overlay, 204800 peaks at 7,607 MiB nvidia-smi on a real RTX 3070 (headless line 7,692);
        // the in-file MTP layer peaks at 7,819 there (over), so LLAMA_MODEL_OVERLAY_PATCH= (empty) needs -c 114688.
        // Draft window 20480 and vocab map auto:65536 stay (small-card memory ruling, V07/PRESETS/SOL_HANDOFF.md).
        // V07/REL071/STATUS.md.
        common_card_preset tiny {
            "8dense", "2026-11-17.4", "best known/recheck before ship", "", {
                {"--ctx-size", "204800"}, {"--parallel", "1"}, {"--batch-size", "4096"},
                {"--ubatch-size", "256"}, {"--gpu-layers", "99"}, {"--flash-attn", "on"},
                {"--fit", "on"}, {"--cache-ram", "0"}, {"--no-cache-disk", "1"},
                {"--cache-type-k", "sjkvarn3"}, {"--cache-type-v", "sjkvarn3"},
                {"--sjkvarn-body-type", "sjkvarn4t"}, {"--cache-type-s", "q8_0"},
                {"--override-tensor", "token_embd.weight=CPU"}, {"--load-mode", "none"},
                {"--spec-type", "draft-mtp"}, {"--spec-draft-n-max", "3"},
                {"--spec-n-rs-seq", "3"}, {"--spec-draft-p-min", "0"},
                {"--spec-draft-vocab-map", "auto:65536"},
                {"--spec-draft-window", "20480"}, {"--spec-draft-n-ubatch", "8"},
                {"--spec-draft-type-k", "q8_0"}, {"--spec-draft-type-v", "q8_0"}
            }, {
                {"LLAMA_DEEPCTX_LADDER", "1"},
                {"LLAMA_LSA_MODE", "key"}, {"LLAMA_LSA_MAP", "@model/lsa-qwen38-27b-d2048.gguf"},
                {"LLAMA_LSA_KEEP", "16384"}, {"LLAMA_LSA_SHARED", "1"},
                {"LLAMA_LSA_IDX_CACHE", "1"}, {"LLAMA_LSA_PACKED", "1"},
                {"LLAMA_LSA_IDX_Q8", "1"}, {"LLAMA_LSA_IDX_Q4", "1"},
                {"LLAMA_LSA_BLOCK", "128"}, {"LLAMA_LSA_BLOCK_TAU", "2"},
                {"LLAMA_LSA_SJ_SKIP", "1"}, {"LLAMA_LSA_REFINE", "128"},
                {"LLAMA_LSA_REFINE_SH", "1"},
                {"LLAMA_LSA_REFINE_MULT", "2"}, {"LLAMA_LSA_RING_ROWS", "512"},
                {"LLAMA_LSA_MEAN_CHUNKS", "1"}, {"LLAMA_RESERVE_VERIFY_ROWS", "7"},
                {"GGML_CUDA_DISABLE_GRAPHS", "1"}, {"GGML_ALLOC_GROW_PCT", "1"},
                {"LLAMA_LSA_FUSED_SEL", "1"}, {"LLAMA_LSA_COMP", "2"},
                {"LLAMA_LSA_COMP_LB", "1"}, {"LLAMA_LSA_GRAPH_REUSE", "0"},
                {"LLAMA_LSA_RESIDENT_MIB", "65.7"}, {"GGML_CUDA_LSA_RESIDENT_STAGE", "256"},
                {"LLAMA_KQ_MASK_PACKED", "1"}, {"GGML_SJKVARN_TILE_SKIP", "1"},
                {"GGML_SJKVARN_TRELLIS_PREFILL_MIB", "16"}, {"GGML_SJKVARN_TRELLIS_WIDTH", "1"},
                {"GGML_SJKVARN_TRELLIS_WIDTH_LSE", "1"},
                {"LLAMA_MTP_DRAFT_COMPUTE_LEAN", "0"}, {"LLAMA_DRAFT_VOCAB_FULL_HEAD", "1"},
                {"LLAMA_MODEL_OVERLAY_PATCH", "@model/mtp-dm.gguf"}, {"GGML_CUDA_GRAPH_MAX_ROWS", "16"}
            }
        };
        tiny.fit_margin_mib = 500;
        std::vector<common_card_preset> cards {large, medium, small, tiny};
        for (auto & card : cards) {
            card.version = card.name == "8dense" ? "2026-11-19.1" : "2026-11-17.6";
            card.options.insert(card.options.end(), {
                {"--temp", "1.0"}, {"--top-k", "20"}, {"--top-p", "0.95"}, {"--min-p", "0"}
            });
        }
        return cards;
    }();
    // Headed fit evidence: V07/SMALLUB/RESULT.md; owner order 2026-11-17T04:48Z.
    static const std::vector<common_card_preset> headed_profiles = [] {
        auto cards = headless_profiles;
        for (auto & card : cards) {
            if (card.name == "10dense") {
                card.options.front().second = "180224";
            } else if (card.name == "12dense") {
                card.options.front().second = "196608";
                card.env.emplace_back("GGML_CUDA_DISABLE_GRAPHS", "1");
            }
        }
        return cards;
    }();
    headless = headless || string_ends_with(name, "-headless");
    const auto & profiles = headless ? headless_profiles : headed_profiles;
    const std::string base_name = string_ends_with(name, "-headless") ? name.substr(0, name.size() - 9) : name;
    for (const auto & profile : profiles) {
        if (profile.name == base_name) {
            return profile;
        }
    }
    throw std::invalid_argument("unknown card preset: expected 24dense, 12dense, 10dense, or 8dense");
}

std::string common_card_preset_apply_env(const common_params & params) {
    std::string applied;
    if (params.card_preset.empty()) {
        return applied;
    }
    const auto & card = common_card_preset_get(params.card_preset, params.card_preset_headless);
    for (const auto & kv : card.env) {
        if (!params.card_preset_kv_default && (kv.first.compare(0, 10, "LLAMA_LSA_") == 0 ||
            kv.first.compare(0, 14, "GGML_CUDA_LSA_") == 0 || kv.first == "GGML_SJKVARN_TILE_SKIP")) {
            continue;
        }
        if (!std::getenv(kv.first.c_str())) {
            auto model_path = [&](const std::string & value) {
                return value.compare(0, 7, "@model/") == 0 ?
                    (std::filesystem::path(params.model.path).parent_path() / value.substr(7)).string() : value;
            };
            std::string value;
            if (kv.first == "LLAMA_MODEL_OVERLAY_PATCH") {
                std::istringstream paths(kv.second);
                std::string path;
                while (std::getline(paths, path, ',')) {
                    if (!value.empty()) {
                        value += ",";
                    }
                    value += model_path(path);
                }
                // a missing profile sidecar stops here with the fix, not later as "failed to load overlay patch"
                std::error_code ec;
                for (size_t b = 0, e = 0; b <= value.size(); b = e + 1) {
                    e = value.find(',', b);
                    e = e == std::string::npos ? value.size() : e;
                    const std::string file = value.substr(b, e - b);
                    if (!file.empty() && !std::filesystem::is_regular_file(file, ec)) {
                        throw std::runtime_error("card preset " + card.name + " needs " + file + (ec ? " (" + ec.message() + ")" : std::string()) +
                            " beside the model (download it with the release), or set LLAMA_MODEL_OVERLAY_PATCH= (empty)"
                            " to use the model file's own MTP layer, with -c 114688 on an 8 GB card");
                    }
                }
            } else {
                value = model_path(kv.second);
            }
            common_set_env(kv.first, value);
        }
        const char * value = std::getenv(kv.first.c_str());
        if (!value) {
            throw std::runtime_error("cannot set card preset environment: " + kv.first);
        }
        applied += " " + kv.first + "=" + value;
    }
    return applied;
}

static std::string rm_leading_dashes(const std::string & str) {
    size_t pos = 0;
    while (pos < str.size() && str[pos] == '-') {
        ++pos;
    }
    return str.substr(pos);
}

static std::string canonical_tag(const std::string & tag) {
    static const std::regex re_tag("[-.]([A-Z0-9_]+)$", std::regex::icase);
    std::smatch m;
    if (std::regex_search(tag, m, re_tag)) {
        std::string canon = m[1].str();
        for (char & c : canon) {
            c = (char) std::toupper((unsigned char) c);
        }
        return canon;
    }
    std::string upper = tag;
    for (char & c : upper) {
        c = (char) std::toupper((unsigned char) c);
    }
    return upper;
}

std::vector<std::string> common_preset::to_args(const std::string & bin_path) const {
    std::vector<std::string> args;

    if (!bin_path.empty()) {
        args.push_back(bin_path);
    }

    for (const auto & [opt, value] : options) {
        if (opt.is_preset_only) {
            continue; // skip preset-only options (they are not CLI args)
        }

        // use the last arg as the main arg (i.e. --long-form)
        args.push_back(opt.args.back());

        // handle value(s)
        if (opt.value_hint == nullptr && opt.value_hint_2 == nullptr) {
            // flag option, no value
            if (common_arg_utils::is_falsey(value)) {
                // use negative arg if available
                if (!opt.args_neg.empty()) {
                    args.back() = opt.args_neg.back();
                } else {
                    // otherwise, skip the flag
                    // TODO: maybe throw an error instead?
                    args.pop_back();
                }
            }
        }
        if (opt.value_hint != nullptr) {
            // single value
            args.push_back(value);
        }
        if (opt.value_hint != nullptr && opt.value_hint_2 != nullptr) {
            throw std::runtime_error(string_format(
                "common_preset::to_args(): option '%s' has two values, which is not supported yet",
                opt.args.back()
            ));
        }
    }

    return args;
}

std::string common_preset::to_ini() const {
    std::ostringstream ss;

    ss << "[" << name << "]\n";
    for (const auto & [opt, value] : options) {
        auto espaced_value = value;
        string_replace_all(espaced_value, "\n", "\\\n");
        ss << rm_leading_dashes(opt.args.back()) << " = ";
        ss << espaced_value << "\n";
    }
    ss << "\n";

    return ss.str();
}

void common_preset::set_option(const common_preset_context & ctx, const std::string & env, const std::string & value) {
    // try if option exists, update it
    for (auto & [opt, val] : options) {
        if (opt.env && env == opt.env) {
            val = value;
            return;
        }
    }
    // if option does not exist, we need to add it
    if (ctx.key_to_opt.find(env) == ctx.key_to_opt.end()) {
        throw std::runtime_error(string_format(
            "%s: option with env '%s' not found in ctx_params",
            __func__, env.c_str()
        ));
    }
    options[ctx.key_to_opt.at(env)] = value;
}

void common_preset::unset_option(const std::string & env) {
    for (auto it = options.begin(); it != options.end(); ) {
        const common_arg & opt = it->first;
        if (opt.env && env == opt.env) {
            it = options.erase(it);
            return;
        } else {
            ++it;
        }
    }
}

bool common_preset::get_option(const std::string & env, std::string & value) const {
    for (const auto & [opt, val] : options) {
        if (opt.env && env == opt.env) {
            value = val;
            return true;
        }
    }
    return false;
}

void common_preset::merge(const common_preset & other) {
    for (const auto & [opt, val] : other.options) {
        options[opt] = val; // overwrite existing options
    }
}

void common_preset::apply_to_params(common_params & params, const std::set<std::string> & handled_keys) const {
    for (const auto & [opt, val] : options) {
        if (!handled_keys.empty()) {
            if (!opt.env || handled_keys.find(opt.env) == handled_keys.end()) {
                continue;
            }
        }
        // apply each option to params
        if (opt.handler_string) {
            opt.handler_string(params, val);
        } else if (opt.handler_int) {
            opt.handler_int(params, std::stoi(val));
        } else if (opt.handler_bool) {
            opt.handler_bool(params, common_arg_utils::is_truthy(val));
        } else if (opt.handler_str_str) {
            // not supported yet
            throw std::runtime_error(string_format(
                "%s: option with two values is not supported yet",
                __func__
            ));
        } else if (opt.handler_void) {
            opt.handler_void(params);
        } else {
            GGML_ABORT("unknown handler type");
        }
    }
}

static std::map<std::string, std::map<std::string, std::string>> parse_ini_from_file(const std::filesystem::path & path) {
    std::map<std::string, std::map<std::string, std::string>> parsed;

    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("preset file does not exist: " + path.string());
    }

    std::ifstream file(path);
    if (!file.good()) {
        throw std::runtime_error("failed to open server preset file: " + path.string());
    }

    std::string contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    static const auto parser = build_peg_parser([](auto & p) {
        // newline ::= "\r\n" / "\n" / "\r"
        auto newline = p.rule("newline", p.literal("\r\n") | p.literal("\n") | p.literal("\r"));

        // ws ::= [ \t]*
        auto ws = p.rule("ws", p.chars("[ \t]", 0, -1));

        // comment ::= [;#] (!newline .)*
        auto comment = p.rule("comment", p.chars("[;#]", 1, 1) + p.zero_or_more(p.negate(newline) + p.any()));

        // eol ::= ws comment? (newline / EOF)
        auto eol = p.rule("eol", ws + p.optional(comment) + (newline | p.end()));

        // ident ::= [a-zA-Z_] [a-zA-Z0-9_.-]*
        auto ident = p.rule("ident", p.chars("[a-zA-Z_]", 1, 1) + p.chars("[a-zA-Z0-9_.-]", 0, -1));

        // value ::= (!eol-start .)*
        auto eol_start = p.rule("eol-start", ws + (p.chars("[;#]", 1, 1) | newline | p.end()));
        auto value = p.rule("value", p.zero_or_more(p.negate(eol_start) + p.any()));

        // header-line ::= "[" ws ident ws "]" eol
        auto header_line = p.rule("header-line", "[" + ws + p.tag("section-name", p.chars("[^]]")) + ws + "]" + eol);

        // kv-line ::= ident ws "=" ws value eol
        auto kv_line = p.rule("kv-line", p.tag("key", ident) + ws + "=" + ws + p.tag("value", value) + eol);

        // comment-line ::= ws comment (newline / EOF)
        auto comment_line = p.rule("comment-line", ws + comment + (newline | p.end()));

        // blank-line ::= ws (newline / EOF)
        auto blank_line = p.rule("blank-line", ws + (newline | p.end()));

        // line ::= header-line / kv-line / comment-line / blank-line
        auto line = p.rule("line", header_line | kv_line | comment_line | blank_line);

        // ini ::= line* EOF
        auto ini = p.rule("ini", p.zero_or_more(line) + p.end());

        return ini;
    });

    common_peg_parse_context ctx(contents);
    const auto result = parser.parse(ctx);
    if (!result.success()) {
        throw std::runtime_error("failed to parse server config file: " + path.string());
    }

    std::string current_section = COMMON_PRESET_DEFAULT_NAME;
    std::string current_key;

    ctx.ast.visit(result, [&](const auto & node) {
        if (node.tag == "section-name") {
            const std::string section = std::string(node.text);
            current_section = section;
            parsed[current_section] = {};
        } else if (node.tag == "key") {
            const std::string key = std::string(node.text);
            current_key = key;
        } else if (node.tag == "value" && !current_key.empty() && !current_section.empty()) {
            parsed[current_section][current_key] = std::string(node.text);
            current_key.clear();
        }
    });

    return parsed;
}

static std::map<std::string, common_arg> get_map_key_opt(common_params_context & ctx_params) {
    std::map<std::string, common_arg> mapping;
    for (const auto & opt : ctx_params.options) {
        for (const auto & env : opt.get_env()) {
            mapping[env] = opt;
        }
        for (const auto & arg : opt.get_args()) {
            mapping[rm_leading_dashes(arg)] = opt;
        }
    }
    return mapping;
}

static bool is_bool_arg(const common_arg & arg) {
    return !arg.args_neg.empty();
}

static std::string parse_bool_arg(const common_arg & arg, const std::string & key, const std::string & value) {
    // if this is a negated arg, we need to reverse the value
    for (const auto & neg_arg : arg.args_neg) {
        if (rm_leading_dashes(neg_arg) == key) {
            return common_arg_utils::is_truthy(value) ? "false" : "true";
        }
    }
    // otherwise, not negated
    return value;
}

common_preset_context::common_preset_context(llama_example ex)
        : ctx_params(common_params_parser_init(default_params, ex)) {
    common_params_add_preset_options(ctx_params.options);
    key_to_opt = get_map_key_opt(ctx_params);
}

common_presets common_preset_context::load_from_ini(const std::filesystem::path & path, common_preset & global) const {
    common_presets out;
    auto ini_data = parse_ini_from_file(path);

    for (auto section : ini_data) {
        common_preset preset;
        std::string section_name = section.first.empty() ? std::string(COMMON_PRESET_DEFAULT_NAME) : section.first;
        if (section_name != "*" && section_name != COMMON_PRESET_DEFAULT_NAME) {
            auto colon_idx = section_name.rfind(':');
            if (colon_idx != std::string::npos) {
                std::string tag       = section_name.substr(colon_idx + 1);
                std::string canon_tag = canonical_tag(tag);
                if (canon_tag != tag) {
                    section_name = section_name.substr(0, colon_idx + 1) + canon_tag;
                }
            }
        }
        preset.name = section_name;
        LOG_DBG("loading preset: %s\n", preset.name.c_str());
        for (const auto & [key, value] : section.second) {
            if (key == "version") {
                // skip version key (reserved for future use)
                continue;
            }

            LOG_DBG("option: %s = %s\n", key.c_str(), value.c_str());
            if (filter_allowed_keys && allowed_keys.find(key) == allowed_keys.end()) {
                throw std::runtime_error(string_format(
                    "option '%s' is not allowed in remote presets",
                    key.c_str()
                ));
            }
            if (key_to_opt.find(key) != key_to_opt.end()) {
                const auto & opt = key_to_opt.at(key);
                if (is_bool_arg(opt)) {
                    preset.options[opt] = parse_bool_arg(opt, key, value);
                } else {
                    preset.options[opt] = value;
                }
                LOG_DBG("accepted option: %s = %s\n", key.c_str(), preset.options[opt].c_str());
            } else if (ignore_unknown_keys) {
                LOG_WRN("ignoring option '%s' from %s: not supported by this program\n", key.c_str(), path.string().c_str());
            } else {
                throw std::runtime_error(string_format(
                    "option '%s' not recognized in preset '%s'",
                    key.c_str(), preset.name.c_str()
                ));
            }
        }

        if (preset.name == COMMON_PRESET_DEFAULT_NAME && preset.options.empty()) {
            continue;
        }

        if (preset.name == "*") {
            // handle global preset
            global = preset;
        } else {
            out[preset.name] = preset;
        }
    }

    return out;
}

common_presets common_preset_context::load_from_cache() const {
    common_presets out;

    auto cached_models = common_list_cached_models();
    for (const auto & model : cached_models) {
        common_preset preset;
        preset.name = model.to_string();
        preset.set_option(*this, "LLAMA_ARG_HF_REPO", model.to_string());
        out[preset.name] = preset;
    }

    return out;
}

struct local_model {
    std::string name;
    std::string path;
    std::string path_mmproj;
    std::string path_draft;
};

// TODO @ngxson: handle "eagle3-" when it's supported by common_speculative_types_from_gguf()
static const char * draft_prefixes[] = { "mtp-", "dspark-", "dflash-" };

static bool is_mmproj_file(const std::string & fname) {
    return fname.find("mmproj") != std::string::npos;
}

static bool is_draft_file(const std::string & fname) {
    for (const auto & prefix : draft_prefixes) {
        if (fname.rfind(prefix, 0) == 0) {
            return true;
        }
    }
    return false;
}

common_presets common_preset_context::load_from_models_dir(const std::string & models_dir) const {
    const std::filesystem::path dir = models_dir;
    if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
        throw std::runtime_error(string_format("error: '%s' does not exist or is not a directory\n", models_dir.c_str()));
    }

    std::vector<local_model> models;
    auto scan_subdir = [&models](const std::filesystem::path & subdir_path, const std::string & name) {
        std::filesystem::path model_file;
        std::filesystem::path first_shard_file;
        std::filesystem::path mmproj_file;
        std::filesystem::path draft_file;
        std::error_code ec;
        for (const auto & entry : std::filesystem::directory_iterator(subdir_path)) {
            if (!entry.is_regular_file(ec)) {
                continue;
            }
            const std::string fname = entry.path().filename().string();
            if (!string_ends_with(fname, ".gguf")) {
                continue;
            }
            if (is_mmproj_file(fname)) {
                mmproj_file = entry.path();
            } else if (is_draft_file(fname)) {
                if (draft_file.empty()) {
                    draft_file = entry.path(); // first sidecar found wins
                }
            } else if (fname.find("-00001-of-") != std::string::npos) {
                first_shard_file = entry.path();
            } else {
                model_file = entry.path();
            }
        }
        // single file model
        const std::filesystem::path & path = first_shard_file.empty() ? model_file : first_shard_file;
        if (!path.empty()) {
            models.push_back({
                /* name        */ name,
                /* path        */ path.string(),
                /* path_mmproj */ mmproj_file.string(), // can be empty
                /* path_draft  */ draft_file.string()   // can be empty
            });
        }
    };

    for (const auto & entry : std::filesystem::directory_iterator(dir)) {
        std::error_code ec;
        if (entry.is_directory(ec)) {
            scan_subdir(entry.path(), entry.path().filename().string());
            continue;
        }
        if (!entry.is_regular_file(ec)) {
            continue;
        }
        const std::string fname = entry.path().filename().string();
        if (!string_ends_with(fname, ".gguf")) {
            continue;
        }
        if (is_mmproj_file(fname) || is_draft_file(fname)) {
            continue; // companion file, cannot be loaded as a model on its own
        }
        // single file model
        std::string name = fname;
        string_replace_all(name, ".gguf", "");
        models.push_back({
            /* name        */ name,
            /* path        */ entry.path().string(),
            /* path_mmproj */ "",
            /* path_draft  */ ""
        });
    }

    // convert local models to presets
    common_presets out;
    for (const auto & model : models) {
        common_preset preset;
        preset.name = model.name;
        preset.set_option(*this, "LLAMA_ARG_MODEL", model.path);
        if (!model.path_mmproj.empty()) {
            preset.set_option(*this, "LLAMA_ARG_MMPROJ", model.path_mmproj);
        }
        if (!model.path_draft.empty()) {
            preset.set_option(*this, "LLAMA_ARG_SPEC_DRAFT_MODEL", model.path_draft);
        }
        out[preset.name] = preset;
    }

    return out;
}

common_preset common_preset_context::load_from_args(int argc, char ** argv) const {
    common_preset preset;
    preset.name = COMMON_PRESET_DEFAULT_NAME;

    bool ok = common_params_to_map(argc, argv, ctx_params.ex, preset.options);
    if (!ok) {
        throw std::runtime_error("failed to parse CLI arguments into preset");
    }

    return preset;
}

common_presets common_preset_context::cascade(const common_presets & base, const common_presets & added) const {
    common_presets out = base; // copy
    for (const auto & [name, preset_added] : added) {
        if (out.find(name) != out.end()) {
            // if exists, merge
            common_preset & target = out[name];
            target.merge(preset_added);
        } else {
            // otherwise, add directly
            out[name] = preset_added;
        }
    }
    return out;
}

common_presets common_preset_context::cascade(const common_preset & base, const common_presets & presets) const {
    common_presets out;
    for (const auto & [name, preset] : presets) {
        common_preset tmp = base; // copy
        tmp.name = name;
        tmp.merge(preset);
        out[name] = std::move(tmp);
    }
    return out;
}
