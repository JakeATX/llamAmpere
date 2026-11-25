// fork DEFPATH: per-arch best-config reference, boot-time apply and engaged-path report (see arch-defaults.h)

#include "arch-defaults.h"

#include "common.h"
#include "log.h"
#include "speculative.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <set>

using json = nlohmann::ordered_json;

// The table. Machine-readable: scripts/defpath/defpath_gate.py extracts the text between the ARCHJSON markers.
// Per row:
//   arch            general.architecture values the row covers
//   ub              24 GB-class auto ub ceiling (UBDEFAULT), needs_ctx: only when -c is given, batch_24g: -b (0 = rule)
//   lean_prompt     per card class ("24g" = smallest used GPU >= 23,552 MiB, "small" = below): on | off | auto
//                   (auto = llama-context's free-MiB rule: MoE + < 14,336 MiB free after the weights)
//   kv              the proven trunk KV pair (report only: the engine's own arch default sets it)
//   spec            drafter beside the model, applied by spec-defaults.cpp when no drafting was chosen
//                   (model = the published file name, model_alt = older names of the same file, tried in order)
//   env_on          engine switches that are ON by default (unset = on, "off" value = the kill switch); report only
//   apply_env       values the table sets when unset (before the model load)
//   watch           flags that are OFF by default and still under gate (report only)
//   best            best config per card class for the gate: preset + extra args/env, the evidence and the file
static const char * k_arch_defaults_json = R"ARCHJSON(
{
  "version": "2026-11-24.3",
  "protocol": {"chat": true, "temp": 1.0, "npred": 5120, "fill": 100000, "seeds": 6, "order": "ABBA",
               "fail_if_default_slower_pct": 1.0},
  "rows": [
    {
      "name": "Inkling 2 Mini",
      "arch": ["inkling2", "inkling2_loop"],
      "ub": {"ladder": true, "ceiling_24g": 1024, "needs_ctx": true, "batch_24g": 2048,
             "why": "UBDEFAULT sweep 3090 4/4 c131072: 1024 best at 32K/100K (SWA rel-bias work grows with ub); INKCOMP F262: -ub 1024 -b 2048 +3% decode, +131K max ctx"},
      "lean_prompt": {"24g": "off", "small": "auto",
             "why": "free-MiB rule was tuned on Flash-Next; NQ 24 GB (7,667 MiB free) took it silently. DEFPATH smoke 11-24 3090 300 W, NQ defaults, 16K fill, 5,120-tok chat temp 1, ABBA 2+2: LEAN=0 prefill 1993 vs 1949 (+2.3%), decode 97.9 vs 97.7"},
      "kv": {"type_k": "sjkvarn4", "type_v": "sjkvarn4", "why": "Jake 11-22: Inkling ships SJ-KVaRN 4/4 (LLAMA_INK_KV44 engine default)"},
      "spec": {"type": "draft-dspark", "model": "Inkling-2-Mini-ATX-Drafter-Q8_0.gguf",
               "model_alt": ["Inkling-2-Mini-DSpark-Q8_0-fold.gguf"], "ngl": 99, "ubatch": 64,
               "type_k": "f16", "type_v": "f16", "min_gpu_mib": 23552,
               "why": "24ink preset drafter; draft KV F16 explicit (DSJ opt-in measured -3.0% at 100K, 11-22 gate); small cards: drafter -12..-29% at 100K (its VRAM is expert cache)"},
      "parallel": 1,
      "env_on": [
        {"env": "GGML_CUDA_INK_SJ_D128",     "off": "0", "why": "packed D128 SJ-KVaRN FA route"},
        {"env": "LLAMA_INKKV_FAR_PACKED",    "off": "0", "why": "one-bit packed far mask (prompt)"},
        {"env": "LLAMA_INKKV_QPACK",         "off": "0", "why": "packed Q for the INKKV far route"},
        {"env": "GGML_CUDA_INKGLUE",         "off": "0", "why": "fused glue kernels"},
        {"env": "LLAMA_INKLING2_GATE_FUSED", "off": "0", "why": "fused gate"},
        {"env": "LLAMA_DSPARK_FAST_GRAPH",   "off": "0", "why": "DSpark fast draft graph"},
        {"env": "LLAMA_DSPARK_MASK_GPU",     "off": "0", "why": "DSpark mask on the GPU"},
        {"env": "LLAMA_INK_KV44",            "off": "0", "why": "4/4 KV default for Inkling"},
        {"env": "LLAMA_INK_CACHE_LOAN",      "off": "0", "why": "small cards: CACHELOAN +30% decode 10ink"},
        {"env": "LLAMA_DSPARK_MB_DEPTH",     "off": "0", "why": "mb DSpark fixed depth 4 (INKSPEC2)"},
        {"env": "GGML_CUDA_NK_PF_MM",        "off": "0", "why": "NVFP4 MoE prefill GEMM (NEONKERN: HOME 100K ABBA prefill +10.2%, byte-exact)"},
        {"env": "GGML_CUDA_NK_Q8_0_TC",      "off": "0", "why": "Q8_0 MoE tensor-core path (NEONKERN: 100K ABBA -2.77% ms/round, E21 |dKLD| <= 1e-6)"},
        {"env": "GGML_CUDA_NK_DSPARK_HEAD",  "off": "0", "why": "DSpark head kernel (NEONKERN: 100K ABBA -1.36% ms/round, identical)"},
        {"env": "GGML_CUDA_DSPARK_HEAD",     "off": "0", "why": "DSpark head fusion"},
        {"env": "GGML_CUDA_PREFILL_FUSE_GLU_ACTQ",  "off": "0", "why": "prefill glue: GLU + activation quant fused (0.7.5)"},
        {"env": "GGML_CUDA_PREFILL_FUSE_NORM_ACTQ", "off": "0", "why": "prefill glue: norm + activation quant fused (0.7.5)"},
        {"env": "GGML_CUDA_FUSE_RELMERGE",   "off": "0", "why": "FUSE4: HOME 3090 Ti 100K ABBA x6 +0.87% decode (all four), byte-identical"},
        {"env": "GGML_CUDA_FUSE_KVCONV",     "off": "0", "why": "FUSE4"},
        {"env": "GGML_CUDA_FUSE_MIXNORMV",   "off": "0", "why": "FUSE4"},
        {"env": "GGML_CUDA_FUSE_QKVNORM",    "off": "0", "why": "FUSE4"}
      ],
      "apply_env": [
        {"env": "LLAMA_NK_DEV_MASK",    "value": "1",    "why": "device-side DSpark / SJ-KVaRN far masks (0.7.5 Inkling default)"},
        {"env": "LLAMA_NK_KV_PAD",      "value": "4096", "why": "coarser deep-context n_kv padding step (0.7.5 Inkling default)"},
        {"env": "GGML_CUDA_Q8TC",       "value": "2",    "why": "tensor-core dense Q8_0 matvec T=4..8, hi/lo activations; REL075 E21 ub5 dKLD -4.4e-5 (mode 1 +1.05e-4 fails: opt-in)"},
        {"env": "GGML_CUDA_Q8TC_M1",    "value": "g",    "why": "Q8TC-PT: grouped launches run mode 1 under mode 2; HOME 3090 Ti 350 W NQ 100K ABBA x6 -1.07% ms/round (6/6 seeds), kld4x paired dKLD -0.04e-4 +/- 0.50e-4; off: GGML_CUDA_Q8TC_M1=0"},
        {"env": "GGML_CUDA_INKGQA_FAR", "value": "1",    "why": "GQA far-attention prefill kernel (SURGE-FA, 0.7.5 default)"},
        {"env": "GGML_CUDA_INKGQA_SWA", "value": "1",    "why": "same kernel family on the SWA layers (with FAR+RELF +34.5% 100K prefill; E21 = FAR+RELF)"},
        {"env": "LLAMA_INKFA_RELF",     "value": "1",    "why": "relative-bias factor form, no [n_head, extent, T] table (0.7.5 default)"}
      ],
      "watch": [
        {"env": "LLAMA_DSPARK_DRAFT_SJKVARN", "why": "measured slower (-3.0% at 100K): keep off"},
        {"env": "LLAMA_HEAD2",                "why": "SURGE-HEAD: not a speedup on Inkling: keep off"},
        {"env": "LLAMA_ELASTIC_KV",           "why": "SURGE-cache: elastic KV, gate pending"}
      ],
      "best": {
        "24g":   {"preset": "24ink", "args": "", "env": "", "file": "NQ",
                  "evidence": "REL074/INKCOMP 11-24: 24ink -b 2048 -ub 1024 + DSpark, NQ 100K 3090"},
        "small": {"preset": "12ink", "args": "", "env": "", "file": "S34d",
                  "evidence": "INKSMALL2/INKSTREAM 11-24 (8/10/12ink by card)"}
      }
    },
    {
      "name": "Qwen3.8 27B (24dense)",
      "arch": ["qwen35"],
      "ub": {"ladder": true, "ceiling_24g": 2048, "needs_ctx": false, "batch_24g": 0,
             "why": "UBDEFAULT: 4096 vs 2048 no prefill gain, 16-token replies -26%/-29% -> 2048"},
      "lean_prompt": {"24g": "auto", "small": "auto", "why": "dense: the MoE-only rule never fires"},
      "kv": {"type_k": "sjkvarn4", "type_v": "sjkvarn4", "why": "24dense preset"},
      "spec": {"type": "draft-dflash", "model": "Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf", "model_alt": [],
               "preset": "24dense", "kill_env": "LLAMA_DSPINE_DEFAULT", "min_gpu_mib": 23552,
               "type_k": "tq5_0", "type_v": "turbo4", "sampling": "probabilistic",
               "n_max": 7, "n_min": 0, "p_min": 0.0, "mb_mtp": true, "fit": "off",
               "why": "DSpine b16 beside the model, one slot: +8.7% tok/s vs the 0.7.5 MTP default at 98.8K fill (3090 Ti 350 W, P76, chat temp 1, 5,120 tok, 12 seed pairs, 9 faster); LLAMA_DSPINE_DEFAULT=0 keeps the MTP head"},
      "spec_note": "built-in MTP head: spec-defaults.cpp row qwen35 (draft-mtp-adaptive); draft KV left to the resolver on purpose (DSJ + RM24H need unset -ctkd/-ctvd; 11-19 loss came from overriding them); a prompt with images runs that request without the drafter (M-RoPE gap, 0.7.5 server fix)",
      "parallel": 1,
      "env_on": [],
      "apply_env": [],
      "watch": [
        {"env": "GGML_SJKVARN_TILE_SKIP", "why": "SJ-KVaRN tile skip: off in code; the small-card LSA presets set =1 (common/preset.cpp)"}
      ],
      "best": {
        "24g":   {"preset": "24dense", "args": "", "env": "", "file": "P76",
                  "evidence": "REL07x speed gates, P76 24dense one slot (ub 2048, DSJ + RM24H; MTP adaptive 3..7, or DSpine b16 n-max 7 when its file is beside the model)"},
        "small": {"preset": "12dense", "args": "", "env": "", "file": "P76",
                  "evidence": "V07 small-card presets (LSA + MTP)"}
      }
    },
    {
      "name": "Qwen3.8 Flash-Next",
      "arch": ["qwen35moe"],
      "ub": {"ladder": true, "ceiling_24g": 4096, "needs_ctx": false, "batch_24g": 0,
             "why": "UBDEFAULT generic 24 GB ceiling (no Flash-Next sweep yet)"},
      "lean_prompt": {"24g": "auto", "small": "auto",
             "why": "the free-MiB rule is the Flash-Next 12 GB pick (24 GB: prompt 0.980x, writing 0.984x)"},
      "kv": {"type_k": "sjkvarn4", "type_v": "sjkvarn4", "why": "AI4 ship config"},
      "spec": null,
      "spec_note": "MTP head ships as a -md file: no automatic drafter yet (needs a file-name rule)",
      "parallel": 0,
      "env_on": [
        {"env": "LLAMA_SMALL_CARD_AUTO", "off": "0", "why": "small-card prompt plan (host carry + async staging)"},
        {"env": "LLAMA_UB_SMALL_CARD_CAP", "off": "0", "why": "-ub 6144 cap below 14,336 MiB free"}
      ],
      "apply_env": [],
      "watch": [],
      "best": {
        "24g": null,
        "small": null
      },
      "best_note": "not frozen yet: take the 0.7 AI4 release gate config before the next Flash-Next release"
    }
  ]
}
)ARCHJSON";

bool common_arch_defaults_enabled() {
    static const bool on = [] {
        const char * e = getenv("LLAMA_ARCH_DEFAULTS");
        return !(e && e[0] == '0');
    }();
    return on;
}

static const json & arch_table() {
    static const json t = [] {
        try {
            return json::parse(k_arch_defaults_json);
        } catch (const std::exception & e) {
            LOG_ERR("arch-defaults: table does not parse: %s\n", e.what());
            return json::object();
        }
    }();
    return t;
}

static const json * arch_row(const std::string & arch) {
    const json & t = arch_table();
    if (arch.empty() || !t.contains("rows")) {
        return nullptr;
    }
    for (const auto & row : t["rows"]) {
        for (const auto & a : row["arch"]) {
            if (a.get<std::string>() == arch) {
                return &row;
            }
        }
    }
    return nullptr;
}

std::string common_arch_of_model(const std::string & path) {
    static std::mutex mu;
    static std::map<std::string, std::string> cache;
    if (path.empty()) {
        return {};
    }
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(path);
    if (it != cache.end()) {
        return it->second;
    }
    std::string arch = common_get_gguf_info(path).arch;
    cache[path] = arch;
    return arch;
}

size_t common_arch_min_gpu_total_mib(const common_params & params) {
    size_t min_total = 0;
    auto consider = [&](ggml_backend_dev_t dev) {
        if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            return;
        }
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(dev, &free, &total);
        min_total = min_total ? std::min(min_total, total) : total;
    };
    if (!params.devices.empty()) {
        for (auto * dev : params.devices) {
            consider(dev);
        }
    } else {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            consider(ggml_backend_dev_get(i));
        }
    }
    return min_total >> 20;
}

static const char * card_class(const common_params & params) {
    return common_arch_min_gpu_total_mib(params) >= 23552 ? "24g" : "small";
}

int32_t common_arch_ub_ceiling_24g(const std::string & arch, bool * needs_ctx) {
    const json * row = arch_row(arch);
    if (needs_ctx) {
        *needs_ctx = row ? (*row)["ub"].value("needs_ctx", false) : false;
    }
    return row ? (*row)["ub"].value("ceiling_24g", 4096) : 4096;
}

int32_t common_arch_batch_24g(const std::string & arch) {
    const json * row = arch_row(arch);
    return row ? (*row)["ub"].value("batch_24g", 0) : 0;
}

// env names that were set before the table applied anything (user, launcher or card preset)
static std::set<std::string> & env_preset() {
    static std::set<std::string> s;
    return s;
}
static std::set<std::string> & env_by_table() {
    static std::set<std::string> s;
    return s;
}

static void note_env(const std::string & name) {
    if (getenv(name.c_str())) {
        env_preset().insert(name);
    }
}

void common_arch_defaults_apply(const common_params & params) {
    static bool done = false;
    if (done) {
        return; // the first init is the main model (a draft init must not re-decide)
    }
    done = true;
    const std::string arch = common_arch_of_model(params.model.path);
    note_env("LLAMA_LEAN_PROMPT");
    const json * row = arch_row(arch);
    if (row) {
        for (const char * list : {"env_on", "apply_env", "watch"}) {
            for (const auto & e : (*row)[list]) {
                note_env(e["env"].get<std::string>());
            }
        }
    }
    if (!row || !common_arch_defaults_enabled()) {
        return;
    }
    const std::string lean = (*row)["lean_prompt"].value(card_class(params), "auto");
    if ((lean == "on" || lean == "off") && !getenv("LLAMA_LEAN_PROMPT")) {
        common_set_env("LLAMA_LEAN_PROMPT", lean == "on" ? "1" : "0");
        env_by_table().insert("LLAMA_LEAN_PROMPT");
    }
    for (const auto & e : (*row)["apply_env"]) {
        const std::string name = e["env"].get<std::string>();
        if (!getenv(name.c_str())) {
            common_set_env(name, e["value"].get<std::string>());
            env_by_table().insert(name);
        }
    }
}

bool common_arch_spec_default_get(const std::string & arch, common_arch_spec_default & out) {
    const json * row = arch_row(arch);
    if (!row || !common_arch_defaults_enabled() || !row->contains("spec") || (*row)["spec"].is_null()) {
        return false;
    }
    const json & s = (*row)["spec"];
    out.type        = s.value("type", "");
    out.model       = s.value("model", "");
    out.model_alt   = s.value("model_alt", std::vector<std::string>());
    out.ngl         = s.value("ngl", -1);
    out.ubatch      = s.value("ubatch", 0);
    out.type_k      = s.value("type_k", "");
    out.type_v      = s.value("type_v", "");
    out.min_gpu_mib = s.value("min_gpu_mib", (size_t) 0);
    out.why         = s.value("why", "");
    out.preset      = s.value("preset", "");
    out.kill_env    = s.value("kill_env", "");
    out.sampling    = s.value("sampling", "");
    out.n_max       = s.value("n_max", -1);
    out.n_min       = s.value("n_min", -1);
    out.p_min       = s.value("p_min", -1.0f);
    out.mb_mtp      = s.contains("mb_mtp") ? (s["mb_mtp"].get<bool>() ? 1 : 0) : -1;
    out.fit         = s.value("fit", "");
    return !out.type.empty() && !out.model.empty();
}

bool common_arch_spec_default_killed(const common_arch_spec_default & row) {
    const char * v = row.kill_env.empty() ? nullptr : getenv(row.kill_env.c_str());
    return v && strcmp(v, "0") == 0;
}

// the spec row that is the table's pick for this run: a preset row counts only under its preset, at one slot, with its
// kill switch unset and its file beside the model (else the arch's spec_note drafter is the pick)
static const json * spec_row_for_run(const json * row, const common_params & params) {
    if (!row || !row->contains("spec") || (*row)["spec"].is_null()) {
        return nullptr;
    }
    const json & s = (*row)["spec"];
    const std::string preset = s.value("preset", "");
    if (!preset.empty()) {
        common_arch_spec_default d;
        d.model     = s.value("model", "");
        d.model_alt = s.value("model_alt", std::vector<std::string>());
        d.kill_env  = s.value("kill_env", "");
        if (!common_arch_defaults_enabled() || preset != params.card_preset || params.n_parallel != 1 || common_arch_spec_default_killed(d) ||
                common_arch_spec_default_file(d, params.model.path).empty()) {
            return nullptr;
        }
    }
    return &s;
}

std::string common_arch_spec_default_file(const common_arch_spec_default & row, const std::string & model_path) {
    if (model_path.empty()) {
        return "";
    }
    const auto dir = std::filesystem::path(model_path).parent_path();
    std::vector<std::string> names = { row.model };
    names.insert(names.end(), row.model_alt.begin(), row.model_alt.end());
    std::error_code ec;
    for (const auto & name : names) {
        const std::string file = (dir / name).string();
        if (!name.empty() && std::filesystem::is_regular_file(file, ec)) {
            return file;
        }
    }
    return "";
}

// -------------------------------------------------------------------------------------------------------------------
// report

static std::string kv_name(ggml_type t, uint32_t sj_bits) {
    if (sj_bits > 0) {
        return "sjkvarn" + std::to_string(sj_bits);
    }
    return t == GGML_TYPE_COUNT ? "unset" : ggml_type_name(t);
}

void common_arch_defaults_report(const common_params & params, const llama_model * model, const llama_context * ctx) {
    static bool done = false;
    const char * off = getenv("LLAMA_ENGAGED_REPORT");
    if (done || (off && off[0] == '0') || !model || !ctx) {
        return;
    }
    done = true;
    char buf[128] = {0};
    std::string arch = llama_model_meta_val_str(model, "general.architecture", buf, sizeof(buf)) > 0 ? buf : "";
    const json * row = arch_row(arch);
    const std::string cls = card_class(params);
    const bool table_on = common_arch_defaults_enabled();
    std::vector<std::string> lines;
    std::vector<std::string> differ;
    // state: ON / OFF / OVERRIDE / N/A / MISMATCH (a table name set at runtime by code other than the table)
    auto add = [&](const char * state, const std::string & lever, const std::string & value, const std::string & why) {
        lines.push_back(string_format("  [%-8s] %-26s %-28s %s", state, lever.c_str(), value.c_str(), why.c_str()));
    };
    auto src_env = [&](const std::string & name) -> std::string {
        if (env_by_table().count(name)) return "table";
        if (env_preset().count(name)) return "set before boot";
        return getenv(name.c_str()) ? "engine" : "default";
    };

    // ub ladder (UBDEFAULT)
    {
        const char * ub = getenv("LLAMA_UBDEFAULT");
        const char * ladder = getenv("LLAMA_DEEPCTX_LADDER");
        const bool ubdef = !(ub && ub[0] == '0');
        const bool on = ubdef ? (!ladder || !ladder[0] || strcmp(ladder, "0") != 0) : (ladder && strcmp(ladder, "1") == 0);
        add(on ? "ON" : "OFF", "ub ladder", on ? "dynamic" : "fixed",
            !ubdef ? "LLAMA_UBDEFAULT=0 (pre-UBDEFAULT rule)" : (on ? "UBDEFAULT default (LLAMA_DEEPCTX_LADDER=0 off)" : "LLAMA_DEEPCTX_LADDER=0"));
        if (!on) differ.push_back("ub ladder");
    }
    // ub ceiling + batch
    {
        bool needs_ctx = false;
        const int32_t best = common_arch_ub_ceiling_24g(arch, &needs_ctx);
        const int32_t best_b = common_arch_batch_24g(arch);
        const uint32_t ub = llama_n_ubatch(ctx), nb = llama_n_batch(ctx);
        const std::string v = string_format("-ub %u -b %u", ub, nb);
        if (cls != "24g") {
            add("N/A", "ub ceiling (24 GB)", v, "card below 23,552 MiB: preset/-ub value");
        } else {
            const bool same = (int32_t) ub == best && (best_b == 0 || (int32_t) nb == best_b);
            const std::string why = string_format("best -ub %d%s%s (%s)", best, best_b ? string_format(" -b %d", best_b).c_str() : "",
                    needs_ctx ? ", needs -c" : "", params.n_ubatch_auto ? "auto" : "-ub/-b or preset given");
            add(same ? "ON" : (params.n_ubatch_auto ? "OFF" : "OVERRIDE"), "ub ceiling (24 GB)", v, why);
            if (!same) differ.push_back("ub ceiling");
        }
    }
    // trunk KV
    {
        const std::string v = kv_name(params.cache_type_k, params.sj_kvarn_bits_k) + "/" + kv_name(params.cache_type_v, params.sj_kvarn_bits_v);
        if (row) {
            const std::string best = (*row)["kv"].value("type_k", "") + "/" + (*row)["kv"].value("type_v", "");
            const bool same = v == best;
            add(same ? "ON" : "OVERRIDE", "KV cache", v, same ? (*row)["kv"].value("why", "") : "best " + best);
            if (!same) differ.push_back("KV");
        } else {
            add("N/A", "KV cache", v, "no table row");
        }
    }
    // flash attention
    add(params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED ? "OFF" : "ON", "flash attention",
        llama_flash_attn_type_name(params.flash_attn_type), "FA routes need -fa on/auto");
    // drafter
    {
        std::string types;
        for (auto t : params.speculative.types) {
            if (t == COMMON_SPECULATIVE_TYPE_NONE) continue;
            types += (types.empty() ? "" : ",") + common_speculative_type_to_str(t);
        }
        const std::string dft = params.speculative.has_dft() ? std::filesystem::path(params.speculative.draft.mparams.path).filename().string() : "";
        const std::string v = types.empty() ? "none" : types + (dft.empty() ? "" : " " + dft);
        std::string best_type;
        const json * srow = spec_row_for_run(row, params);
        if (srow) {
            const size_t min_mib = srow->value("min_gpu_mib", (size_t) 0);
            best_type = common_arch_min_gpu_total_mib(params) >= min_mib ? srow->value("type", "") :
                        srow->value("preset", "").empty() ? "none" : "";
        }
        if (best_type.empty()) {
            add(types.empty() ? "OFF" : "ON", "drafter", v, row ? row->value("spec_note", "no drafter row") : "no table row");
        } else {
            const bool same = best_type == "none" ? types.empty() : types.find(best_type) != std::string::npos;
            add(same ? (types.empty() ? "OFF" : "ON") : "OVERRIDE", "drafter", v, same ? srow->value("why", "") : "best " + best_type);
            if (!same) differ.push_back("drafter");
        }
        if (!types.empty()) {
            const auto & d = params.speculative.draft;
            // an unset draft type of a file drafter (not MTP) inherits the trunk's plain type (F16 under -ctk sjkvarnN)
            bool mtp = false;
            for (auto t : params.speculative.types) {
                mtp = mtp || t == COMMON_SPECULATIVE_TYPE_DRAFT_MTP || t == COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE;
            }
            auto dname = [&](ggml_type t, ggml_type trunk) -> std::string {
                if (t != GGML_TYPE_COUNT) return ggml_type_name(t);
                return params.speculative.has_dft() && !mtp ? ggml_type_name(trunk) : "resolver";
            };
            const std::string dk = dname(d.cache_type_k, params.cache_type_k) + "/" + dname(d.cache_type_v, params.cache_type_v);
            std::string best_dk;
            if (srow && !best_type.empty() && !srow->value("type_k", "").empty()) {
                best_dk = srow->value("type_k", "") + "/" + srow->value("type_v", "");
            }
            if (best_dk.empty()) {
                add("ON", "draft KV", dk, "resolver: DSJ / tq5_0+turbo4 for MTP, else trunk type");
            } else {
                const bool same = dk == best_dk;
                add(same ? "ON" : "OVERRIDE", "draft KV", dk, same ? "explicit table value" : "best " + best_dk);
                if (!same) differ.push_back("draft KV");
            }
        }
    }
    // lean prompt (LEANAUTO)
    {
        const char * e = getenv("LLAMA_LEAN_PROMPT");
        const bool on = e && atoi(e) != 0;
        const std::string best = row ? (*row)["lean_prompt"].value(cls, "auto") : "auto";
        const std::string src = src_env("LLAMA_LEAN_PROMPT");
        std::string why = src == "engine" ? "free-MiB rule (< 14,336 MiB free after weights, MoE)" : src;
        why += ", best " + best + " on " + cls;
        const bool same = best == "auto" || (best == "on") == on;
        add(same ? (on ? "ON" : "OFF") : "OVERRIDE", "lean prompt", on ? "1" : "0", why);
        if (!same) differ.push_back("lean prompt");
    }
    // parallel
    if (row && (*row).value("parallel", 0) > 0) {
        const int best = (*row).value("parallel", 0);
        const bool same = params.n_parallel == best;
        add(same ? "ON" : "OVERRIDE", "parallel", std::to_string(params.n_parallel), string_format("best -np %d (one user)", best));
        if (!same) differ.push_back("parallel");
    }
    // engine switches and gated flags
    if (row) {
        // fork DEFSYNC: a value that some code other than this table set at runtime ("engine") is a second, hidden
        // source of defaults (the 11-19 trap, and the 0.7.5 llama_model_load stopgap): the row's status disagrees with
        // the runtime default, so it prints MISMATCH and counts as a differing lever
        auto mismatch = [&](const std::string & name, const char * listed) {
            add("MISMATCH", name, getenv(name.c_str()), string_format("listed as %s here, but set at runtime by other code (a hidden "
                "default): move the value into this row's apply_env and drop the other setter", listed));
            differ.push_back("status " + name);
        };
        for (const auto & e : (*row)["env_on"]) {
            const std::string name = e["env"].get<std::string>();
            const char * v = getenv(name.c_str());
            if (v && src_env(name) == "engine") { mismatch(name, "default-ON in code"); continue; }
            const bool killed = v && e.value("off", "0") == v;
            add(killed ? "OFF" : "ON", name, v ? v : "unset", (killed ? "kill switch set (" + src_env(name) + "); " : "") + e.value("why", ""));
            if (killed) differ.push_back(name);
        }
        for (const auto & e : (*row)["apply_env"]) {
            const std::string name = e["env"].get<std::string>();
            const char * v = getenv(name.c_str());
            if (v && src_env(name) == "engine") { mismatch(name, "a table value"); continue; }
            const bool same = v && e.value("value", "") == v;
            add(same ? "ON" : "OVERRIDE", name, v ? v : "unset",
                src_env(name) + "; " + (same ? "" : "best " + e.value("value", "") + "; ") + e.value("why", ""));
            if (!same) differ.push_back(name);
        }
        for (const auto & e : (*row)["watch"]) {
            const std::string name = e["env"].get<std::string>();
            const char * v = getenv(name.c_str());
            const bool on = v && v[0] && strcmp(v, "0") != 0;
            if (v && src_env(name) == "engine") { mismatch(name, "gated, default off"); continue; }
            add(on ? "ON" : "OFF", name, v ? v : "unset", std::string(on ? src_env(name) + "; " : "gated, default off; ") + e.value("why", ""));
        }
    }

    LOG_WRN("engaged path (DEFPATH table %s, arch %s, card %s, %s; LLAMA_ENGAGED_REPORT=0 hides this):\n",
            arch_table().value("version", "?").c_str(), arch.empty() ? "?" : arch.c_str(), cls.c_str(),
            !row ? "NO ROW" : (table_on ? "table on" : "table off: LLAMA_ARCH_DEFAULTS=0"));
    for (const auto & l : lines) {
        LOG_WRN("%s\n", l.c_str());
    }
    if (!row) {
        LOG_WRN("  no arch-defaults row for '%s': every lever is at the engine default, none is checked against a best "
                "config; add a row to common/arch-defaults.cpp after its first speed gate\n", arch.c_str());
    } else if (differ.empty()) {
        LOG_WRN("  all levers match the %s best config (%s)\n", cls.c_str(), row->value("name", "").c_str());
    } else {
        std::string d;
        for (const auto & x : differ) d += (d.empty() ? "" : ", ") + x;
        LOG_WRN("  %zu lever(s) differ from the %s best config: %s\n", differ.size(), cls.c_str(), d.c_str());
    }
}
