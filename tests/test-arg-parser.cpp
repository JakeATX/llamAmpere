#include "arg.h"
#include "common.h"
#include "download.h"
#include "llama.h"
#include "speculative.h"
#include "spec-defaults.h"
#include "preset.h"
#include "fit.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <unordered_set>

#undef NDEBUG
#include <cassert>

static void test(void) {
    common_params params;
    {
        const size_t MiB = 1024 * 1024;
        assert(common_fit_preset_target(9873 * MiB, 9542 * MiB, 1024 * MiB) == 693 * MiB);
        assert(common_fit_preset_target(9873 * MiB, 9542 * MiB, 500 * MiB) == 169 * MiB);
        assert(common_fit_preset_target(9873 * MiB, 9873 * MiB, 1024 * MiB) == 1024 * MiB);
        assert(common_fit_preset_target(8192 * MiB, 7000 * MiB, 500 * MiB) == 0);
        assert(common_fit_preset_target(0, 0, 500 * MiB) == 500 * MiB);
    }

    auto assert_output_limits = [](int32_t n_batch, int32_t n_parallel, int32_t n_draft,
                                   int32_t total, int32_t per_seq) {
        const auto limits = common_speculative_get_output_limits(n_batch, n_parallel, n_draft);
        assert(limits.total == total);
        assert(limits.per_seq == per_seq);
    };

    assert_output_limits(16, 2,  3, 8, 4);
    assert_output_limits(16, 2, -1, 2, 1);
    assert_output_limits( 6, 2,  3, 6, 4);
    assert_output_limits( 2, 1,  3, 2, 2);
    {
        // -b 1 with MTP: the context raises n_batch to n_rs_seq + 2, and the output rows must follow it
        // (v0.6.3.6 sized them from -b and aborted on the first verify: n_outputs_max <= cparams.n_outputs_max)
        common_params b1;
        b1.n_batch = b1.n_ubatch = 1;
        b1.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        b1.speculative.draft.n_max = 4;
        assert(common_context_n_batch(b1) == 6);
        assert(common_context_params_to_llama(b1).n_batch == 6);
        assert_output_limits(common_context_n_batch(b1), 1, 4, 5, 5);
        b1.speculative.types.clear();
        assert(common_context_n_batch(b1) == 1);
    }
    assert_output_limits(
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max());

    {
        common_params_speculative spec;
        spec.synth_len = 3.4;

        auto assert_invalid = [](const common_params_speculative & value, int32_t n_max) {
            try {
                common_speculative_synth_rates_resolve(&value, n_max);
                assert(false);
            } catch (const std::invalid_argument &) {
            }
        };

        const auto rates = common_speculative_synth_rates_resolve(&spec, 4);
        assert(rates.size() == 4);
        assert(std::abs(rates[0] - 0.80581) < 1e-5);
        assert(std::abs(rates[1] - 0.64933) < 1e-5);
        assert(std::abs(rates[2] - 0.52323) < 1e-5);
        assert(std::abs(rates[3] - 0.42163) < 1e-5);
        assert(std::abs(1.0 + rates[0] + rates[1] + rates[2] + rates[3] - 3.4) < 1e-8);

        spec.synth_len = 1.0;
        assert(common_speculative_synth_rates_resolve(&spec, 4) == std::vector<double>({0.0, 0.0, 0.0, 0.0}));

        spec.synth_len = 5.0;
        assert(common_speculative_synth_rates_resolve(&spec, 4) == std::vector<double>({1.0, 1.0, 1.0, 1.0}));

        spec.synth_len = 5.1;
        assert_invalid(spec, 4);

        spec.synth_len = std::numeric_limits<double>::quiet_NaN();
        assert_invalid(spec, 4);

        spec.synth_len = 0.0;
        assert_invalid(spec, 4);

        spec.synth_len = -1.0;
        spec.synth_rates = {0.8, 0.6, 0.4};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, 0.6, 0.4, 0.2};
        assert(common_speculative_synth_rates_resolve(&spec, 4) == spec.synth_rates);

        spec.synth_rates = {0.8, 0.9, 0.4, 0.2};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, std::numeric_limits<double>::quiet_NaN(), 0.4, 0.2};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, 0.6, 0.4, -0.2};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, 0.6, 0.4, 0.2};
        spec.synth_len = 3.0;
        assert_invalid(spec, 4);
    }

    {
        common_params base;
        base.n_parallel = 4;
        base.n_outputs_max_per_seq = 8;
        base.n_ubatch = 512;

        {
            const auto draft = common_base_params_to_speculative(base);
            assert(draft.n_outputs_max == 4);
            assert(draft.n_outputs_max_per_seq == 1);
            assert(draft.n_ubatch == 512); // inherits the target by default
        }

        base.speculative.draft.n_ubatch = 8;
        {
            const auto draft = common_base_params_to_speculative(base);
            assert(draft.n_ubatch == 8);
        }
    }

#ifndef _WIN32
    {
        const char * saved_env = std::getenv("LLAMA_MTP_DRAFT_COMPUTE_LEAN");
        const bool had_env = saved_env != nullptr;
        const std::string saved = saved_env ? saved_env : "";
        common_params base;
        base.n_parallel = 1;
        base.n_batch = 4096;
        base.n_ubatch = 1024;
        base.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE };
        unsetenv("LLAMA_MTP_DRAFT_COMPUTE_LEAN");
        assert(common_base_params_to_speculative(base).n_ubatch == 1024);
        setenv("LLAMA_MTP_DRAFT_COMPUTE_LEAN", "0", 1);
        assert(common_base_params_to_speculative(base).n_ubatch == 1024);
        setenv("LLAMA_MTP_DRAFT_COMPUTE_LEAN", "1", 1);
        const auto draft = common_base_params_to_speculative(base);
        assert(draft.n_batch == 4096);
        assert(draft.n_ubatch == 64);
        assert(common_context_params_to_llama(draft).n_ubatch == 64);
        for (int width = 1; width <= 8; ++width) {
            base.n_ubatch = width;
            assert(common_base_params_to_speculative(base).n_ubatch == width);
        }
        base.n_ubatch = 0;
        assert(common_base_params_to_speculative(base).n_ubatch == 64);
        base.n_ubatch = 1024;
        // an explicit smaller --spec-draft-n-ubatch stays; a larger one is still capped at 64
        base.speculative.draft.n_ubatch = 8;
        assert(common_base_params_to_speculative(base).n_ubatch == 8);
        base.speculative.draft.n_ubatch = 512;
        assert(common_base_params_to_speculative(base).n_ubatch == 64);
        base.speculative.draft.n_ubatch = -1;
        base.n_parallel = 2;
        assert(common_base_params_to_speculative(base).n_ubatch == 1024);
        base.n_parallel = 1;
        base.speculative.types = { COMMON_SPECULATIVE_TYPE_NONE };
        assert(common_base_params_to_speculative(base).n_ubatch == 1024);
        if (had_env) {
            setenv("LLAMA_MTP_DRAFT_COMPUTE_LEAN", saved.c_str(), 1);
        } else {
            unsetenv("LLAMA_MTP_DRAFT_COMPUTE_LEAN");
        }
    }
#endif

    printf("test-arg-parser: make sure there is no duplicated arguments in any examples\n\n");
    {
        const auto ctx = common_params_parser_init(params, LLAMA_EXAMPLE_SERVER);
        bool found = false;
        for (const auto & opt : ctx.options) {
            if (std::find(opt.args.begin(), opt.args.end(), std::string("--moe-skip-tail")) != opt.args.end()) {
                assert(opt.help.empty());
                assert(opt.env && std::string(opt.env) == "LLAMA_ARG_MOE_SKIP_TAIL");
                found = true;
            }
        }
        assert(found);
    }
    for (int ex = 0; ex < LLAMA_EXAMPLE_COUNT; ex++) {
        try {
            auto ctx_arg = common_params_parser_init(params, (enum llama_example)ex);
            common_params_add_preset_options(ctx_arg.options);
            std::unordered_set<std::string> seen_args;
            std::unordered_set<std::string> seen_env_vars;
            for (const auto & opt : ctx_arg.options) {
                // check for args duplications
                for (const auto & arg : opt.get_args()) {
                    if (seen_args.find(arg) == seen_args.end()) {
                        seen_args.insert(arg);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same argument: %s", arg.c_str());
                        exit(1);
                    }
                }
                // check for env var duplications
                for (const auto & env : opt.get_env()) {
                    if (seen_env_vars.find(env) == seen_env_vars.end()) {
                        seen_env_vars.insert(env);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same env var: %s", env.c_str());
                        exit(1);
                    }
                }

                // exclude spec args from this check
                // ref: https://github.com/ggml-org/llama.cpp/pull/22397
                const bool skip = opt.is_spec;

                // ensure shorter argument precedes longer argument
                if (!skip && opt.args.size() > 1) {
                    const std::string first(opt.args.front());
                    const std::string last(opt.args.back());

                    if (first.length() > last.length()) {
                        fprintf(stderr, "test-arg-parser: shorter argument should come before longer one: %s, %s\n",
                                first.c_str(), last.c_str());
                        assert(false);
                    }
                }

                // same check for negated arguments
                if (opt.args_neg.size() > 1) {
                    const std::string first(opt.args_neg.front());
                    const std::string last(opt.args_neg.back());

                    if (first.length() > last.length()) {
                        fprintf(stderr, "test-arg-parser: shorter negated argument should come before longer one: %s, %s\n",
                                first.c_str(), last.c_str());
                        assert(false);
                    }
                }
            }
        } catch (std::exception & e) {
            printf("%s\n", e.what());
            assert(false);
        }
    }

    auto list_str_to_char = [](std::vector<std::string> & argv) -> std::vector<char *> {
        std::vector<char *> res;
        for (auto & arg : argv) {
            res.push_back(const_cast<char *>(arg.data()));
        }
        return res;
    };

    std::vector<std::string> argv;

    {
        auto parse_card = [&](std::vector<std::string> args, common_params & out, llama_example ex = LLAMA_EXAMPLE_SERVER) {
            args.insert(args.begin(), "test-arg-parser");
            return common_params_parse(args.size(), list_str_to_char(args).data(), out, ex);
        };
        for (const std::string name : {"24dense", "12dense", "10dense", "8dense"}) {
            for (const auto & args : std::vector<std::vector<std::string>>{
                    {"--preset", name}, {name}, {"--preset", name, "headless"},
                    {name, "headless"}, {name + "-headless"}, {"--preset", name + "-headless"},
                    {name, "headed"}, {"--preset", name, "headed"}}) {
                common_params selected;
                assert(parse_card(args, selected));
                const bool headless = name == "8dense" || args.back() == "headless" || string_ends_with(args.back(), "-headless");
                assert(selected.card_preset == name && selected.card_preset_ctx_default && selected.fit_params);
                assert(selected.card_preset_headless == headless);
                const int expected_ctx = name == "8dense" ? 204800 : name == "10dense" ?
                        (headless ? 212992 : 180224) : name == "12dense" ? (headless ? 262144 : 196608) : 262144;
                assert(selected.n_ctx == expected_ctx);
                if (name != "24dense") {
                    assert(selected.speculative.draft.vocab_map == "auto:65536");
                    assert(selected.speculative.user_set & COMMON_PARAMS_SPECULATIVE_USER_DRAFT_VOCAB_MAP);
                }
                assert(selected.fit_params_target_explicit);
                assert(selected.card_preset_fit_target);
                assert(selected.sampling.temp == 1.0f && selected.sampling.top_k == 20);
                assert(selected.sampling.top_p == 0.95f && selected.sampling.min_p == 0.0f);
                assert(selected.sampling.spec_lossy);
                for (size_t target : selected.fit_params_target) {
                    assert(target == size_t(headless ? 500 : 1024) * 1024 * 1024);
                }
                for (bool preset_first : {false, true}) {
                    std::vector<std::string> combined = {"-c", "8192", "-fitt", "768", "-fit", "off",
                            "--temperature", "0.7", "--top-k", "32", "--top-p", "0.9", "--min-p", "0.1", "--spec-mb-mtp",
                            "--spec-draft-vocab-map", "none"};
                    combined.insert(preset_first ? combined.begin() : combined.end(), args.begin(), args.end());
                    common_params overridden;
                    assert(parse_card(combined, overridden));
                    assert(overridden.n_ctx == 8192 && !overridden.card_preset_ctx_default && !overridden.fit_params);
                    assert(overridden.fit_params_target.front() == size_t(768) * 1024 * 1024);
                    assert(!overridden.card_preset_fit_target && overridden.fit_params_target_explicit);
                    assert(overridden.sampling.temp == 0.7f && overridden.sampling.top_k == 32);
                    assert(overridden.sampling.top_p == 0.9f && overridden.sampling.min_p == 0.1f);
                    assert(overridden.sampling.spec_lossy);
                    assert(overridden.speculative.draft.vocab_map == "none");
                }
            }
        }
        common_params no_card;
        assert(!parse_card({"headless"}, no_card));
        assert(!parse_card({"headed"}, no_card));
        common_params plain_fit;
        assert(parse_card({"-fitt", "1024"}, plain_fit));
        assert(!plain_fit.card_preset_fit_target && plain_fit.fit_params_target_explicit);
        const char * fit_env = std::getenv("LLAMA_ARG_FIT_TARGET");
        const bool had_fit_env = fit_env != nullptr;
        const std::string saved_fit_env = fit_env ? fit_env : "";
        common_params reused_fit;
        assert(parse_card({"10dense", "headless"}, reused_fit));
        assert(reused_fit.card_preset_fit_target);
        reused_fit.n_gpu_layers = -1; // A CPU-only parse sets this to 0; parser init requires a negative default.
        assert(parse_card({"-fitt", "500"}, reused_fit));
        assert(!reused_fit.card_preset_fit_target && reused_fit.fit_params_target.front() == size_t(500) * 1024 * 1024);
        common_params reused_env_fit;
        assert(parse_card({"10dense", "headless"}, reused_env_fit));
        assert(reused_env_fit.card_preset_fit_target);
        setenv("LLAMA_ARG_FIT_TARGET", "500", 1);
        common_params env_fit;
        assert(parse_card({"10dense"}, env_fit));
        assert(!env_fit.card_preset_fit_target && env_fit.fit_params_target.front() == size_t(500) * 1024 * 1024);
        reused_env_fit.n_gpu_layers = -1;
        assert(parse_card({}, reused_env_fit));
        assert(!reused_env_fit.card_preset_fit_target && reused_env_fit.fit_params_target.front() == size_t(500) * 1024 * 1024);
        if (had_fit_env) {
            setenv("LLAMA_ARG_FIT_TARGET", saved_fit_env.c_str(), 1);
        } else {
            unsetenv("LLAMA_ARG_FIT_TARGET");
        }
        common_params cli_headless;
        assert(parse_card({"8dense-headless", "-m", "model_file.gguf"}, cli_headless, LLAMA_EXAMPLE_CLI));
        assert(cli_headless.fit_params_target.front() == size_t(500) * 1024 * 1024);
        common_params full_context;
        assert(parse_card({"--preset", "8dense", "-c", "0"}, full_context));
        assert(full_context.n_ctx == 0 && !full_context.card_preset_ctx_default);
        for (const auto & name : {"24dense", "12dense", "10dense", "8dense"}) {
            common_params selected;
            assert(parse_card({"--preset", name}, selected));
            common_params bare;
            assert(parse_card({name}, bare));
            assert(selected.card_preset == name && bare.card_preset == name);
            assert(selected.n_ctx == bare.n_ctx && selected.n_parallel == 1);
            assert(!selected.card_preset_flags.empty());
            for (const auto & entry : common_card_preset_get(name).env) {
                assert(entry.first != "LLAMA_SPEC_COPY");
            }
            const uint32_t bits = selected.card_preset == "8dense" ? 3 : 4;
            assert(selected.cache_type_k == GGML_TYPE_F16 && selected.cache_type_v == GGML_TYPE_F16);
            assert(selected.sj_kvarn_bits_k == bits && selected.sj_kvarn_bits_v == bits);
            assert(bare.sj_kvarn_bits_k == bits && bare.sj_kvarn_bits_v == bits);
            if (selected.card_preset == "24dense") {
                assert(selected.cache_type_s == GGML_TYPE_F32);
            }
            for (bool preset_first : {false, true}) {
                std::vector<std::string> args = {"-c", "8192", "-ub", "128", "--spec-type", "none", "--no-spec-mb-mtp"};
                args.insert(preset_first ? args.begin() : args.end(), {"--preset", name});
                common_params overrides;
                assert(parse_card(args, overrides));
                assert(overrides.n_ctx == 8192 && overrides.n_ubatch == 128);
                assert(!overrides.sampling.spec_lossy);
                assert(std::find(overrides.speculative.types.begin(), overrides.speculative.types.end(),
                                 COMMON_SPECULATIVE_TYPE_DRAFT_MTP) == overrides.speculative.types.end());
                assert(std::find(overrides.speculative.types.begin(), overrides.speculative.types.end(),
                                 COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE) == overrides.speculative.types.end());
            }
        }
        for (const auto & name : {"12dense", "10dense", "8dense"}) {
            for (int slots : {2, 3, 4, -1}) {
                common_params invalid;
                assert(!parse_card({"-np", std::to_string(slots), "--preset", name}, invalid));
                assert(invalid.card_preset.empty());
            }
        }
        for (int slots : {2, 3, 4}) {
            common_params multi;
            assert(parse_card({"-np", std::to_string(slots), "--preset", "24dense"}, multi));
            assert(multi.n_parallel == slots && multi.kv_unified && multi.speculative.draft.n_min == 4);
            assert(multi.speculative.draft.attn_window == 4096 && multi.gdn_replay);
            assert(multi.speculative.draft.n_max == 5 && multi.speculative.draft.n_min_adaptive == 4);
            assert(multi.n_ubatch == 1024);
            assert(!(multi.speculative.user_set & COMMON_PARAMS_SPECULATIVE_USER_DRAFT_VOCAB_MAP));
        }
        {
            common_params one;
            assert(parse_card({"--preset", "24dense"}, one));
            assert(one.n_parallel == 1 && one.speculative.draft.attn_window == 0 && !one.gdn_replay);
            assert(one.speculative.draft.vocab_map == "auto" &&
                   !(one.speculative.user_set & COMMON_PARAMS_SPECULATIVE_USER_DRAFT_VOCAB_MAP));
            assert(one.speculative.draft.n_max == 7 && one.speculative.draft.n_min_adaptive == 3);
            assert(one.n_batch == 4096 && one.n_ubatch == 2048);
            common_params ub_kept;
            assert(parse_card({"--preset", "24dense", "-ub", "1024"}, ub_kept));
            assert(ub_kept.n_ubatch == 1024);
            common_params multi_large;
            assert(parse_card({"-np", "2", "-b", "8192", "--preset", "24dense"}, multi_large));
            assert(multi_large.n_ubatch == 1024);
            setenv("LLAMA_ARG_UBATCH", "1024", 1);
            common_params ub_env;
            assert(parse_card({"--preset", "24dense"}, ub_env));
            unsetenv("LLAMA_ARG_UBATCH");
            assert(ub_env.n_ubatch == 1024);
            common_params kept;
            assert(parse_card({"--preset", "24dense", "--spec-draft-window", "4096", "--gdn-replay"}, kept));
            assert(kept.speculative.draft.attn_window == 4096 && kept.gdn_replay);
        }
        common_params depth;
        assert(parse_card({"--spec-draft-n-max", "3", "--preset", "24dense"}, depth));
        assert(depth.speculative.draft.n_max == 3 && depth.speculative.draft.n_min_adaptive == 3);
        common_params depth_max;
        assert(parse_card({"--preset", "24dense", "--spec-draft-n-max", "6"}, depth_max));
        assert(depth_max.speculative.draft.n_max == 6 && depth_max.speculative.draft.n_min_adaptive == 4);
        common_params fixed_depth;
        assert(parse_card({"--preset", "24dense", "--spec-type", "draft-mtp"}, fixed_depth));
        assert(fixed_depth.speculative.draft.n_max == 5);
        for (bool preset_first : {false, true}) {
            std::vector<std::string> args = {"--spec-draft-n-min-adaptive", "6"};
            args.insert(preset_first ? args.begin() : args.end(), {"--preset", "24dense"});
            common_params minimum;
            assert(parse_card(args, minimum));
            assert(minimum.speculative.draft.n_min_adaptive == 6 && minimum.speculative.draft.n_max == 6);
        }
        common_params conflict;
        assert(!parse_card({"--preset", "24dense", "--spec-draft-n-min-adaptive", "6",
                           "--spec-draft-n-max", "5"}, conflict));
        common_params batch;
        assert(parse_card({"--preset", "24dense", "-b", "128"}, batch));
        assert(batch.n_batch == 128 && batch.n_ubatch == 128);
        common_params large_batch;
        assert(parse_card({"-b", "8192", "--preset", "24dense"}, large_batch));
        assert(large_batch.n_batch == 8192 && large_batch.n_ubatch == 2048);
        for (const auto & name : {"10dense", "12dense"}) {
            common_params plain_mtp;
            assert(parse_card({"--preset", name, "--spec-draft-n-min-adaptive", "6"}, plain_mtp));
            assert(plain_mtp.speculative.draft.n_max == 4 && plain_mtp.speculative.draft.n_min_adaptive == 6);
        }
        common_params min_depth;
        assert(parse_card({"--preset", "24dense", "--spec-draft-n-max", "3"}, min_depth));
        assert(min_depth.speculative.draft.n_min == 3);
        common_params explicit_min;
        assert(parse_card({"--spec-draft-n-min", "1", "--preset", "24dense"}, explicit_min));
        assert(explicit_min.speculative.draft.n_min == 1);
        common_params kv;
        assert(parse_card({"-ctk", "q8_0", "-ctv", "q8_0", "--preset", "10dense"}, kv));
        assert(kv.cache_type_k == GGML_TYPE_Q8_0 && kv.cache_type_v == GGML_TYPE_Q8_0 && !kv.card_preset_kv_default);
        common_params asym;
        assert(parse_card({"--slot-ctx", "262144,0,0", "--preset", "24dense", "-np", "3", "-c", "352256"}, asym));
        assert(asym.slot_ctx == std::vector<int32_t>({262144, 0, 0}) && asym.n_ctx == 352256);
        common_params derived;
        assert(parse_card({"--preset", "24dense", "--slot-ctx", "65536,32768,32768", "-np", "3"}, derived));
        assert(derived.n_ctx == 0);
        common_params per_slot;
        assert(parse_card({"--kv-unified-per-slot", "65536", "--preset", "24dense", "-np", "4"}, per_slot));
        assert(per_slot.n_ctx == 0 && per_slot.kv_unified_per_slot == 65536);
        for (const auto & literal : {"24dense", "--preset", "8dense", "headless", "headed"}) {
            common_params prompt;
            assert(parse_card({"-m", "model_file.gguf", "-p", literal}, prompt, LLAMA_EXAMPLE_CLI));
            assert(prompt.prompt == literal && prompt.card_preset.empty());
            common_params with_preset;
            assert(parse_card({"--preset", "24dense", "-m", "model_file.gguf", "-p", literal}, with_preset, LLAMA_EXAMPLE_CLI));
            assert(with_preset.prompt == literal && with_preset.card_preset == "24dense");
        }
        common_params adapter;
        assert(parse_card({"--lora-scaled", "24dense:0.5", "--preset", "24dense"}, adapter));
        assert(adapter.lora_adapters.size() == 1);
        assert(adapter.lora_adapters.front().path == "24dense" && adapter.lora_adapters.front().scale == 0.5f);
        common_params invalid;
        assert(!parse_card({"--preset", "unknown"}, invalid));
        assert(!parse_card({"--preset"}, invalid));
        assert(!parse_card({"24dense", "--preset", "10dense"}, invalid));
        common_params tiny;
        assert(parse_card({"--preset", "8dense"}, tiny));
        assert(tiny.n_ctx == 204800 && tiny.n_ubatch == 512);
        assert(tiny.cache_type_k == GGML_TYPE_F16 && tiny.cache_type_v == GGML_TYPE_F16);
        assert(tiny.sj_kvarn_bits_k == 3 && tiny.sj_kvarn_bits_v == 3);
        assert(tiny.sj_kvarn_body_type == GGML_TYPE_I16);
        assert(tiny.speculative.draft.n_max == 3 && tiny.speculative.draft.n_ubatch == 8);
        assert(tiny.speculative.n_rs_seq_override == 3);
        // 8dense keeps the RAM + disk prompt cache defaults and turns warm-prefill on; the kill switches still win
        assert(tiny.warm_prefill && tiny.cache_disk && tiny.cache_ram_mib == COMMON_CACHE_RAM_AUTO && !tiny.cache_ram_explicit);
        common_params tiny_off;
        assert(parse_card({"--preset", "8dense", "--no-warm-prefill", "--cache-ram", "0", "--no-cache-disk"}, tiny_off));
        assert(!tiny_off.warm_prefill && !tiny_off.cache_disk && tiny_off.cache_ram_mib == 0);
        common_params tiny_off_first;
        assert(parse_card({"--no-warm-prefill", "8dense"}, tiny_off_first) && !tiny_off_first.warm_prefill);
#ifndef _WIN32
        const char * old_ctx = std::getenv("LLAMA_ARG_CTX_SIZE");
        const bool had_ctx = old_ctx != nullptr;
        const std::string saved_ctx = old_ctx ? old_ctx : "";
        setenv("LLAMA_ARG_CTX_SIZE", "16384", 1);
        common_params env;
        assert(parse_card({"--preset", "24dense"}, env) && env.n_ctx == 16384);
        common_params cli;
        assert(parse_card({"-c", "8192", "--preset", "24dense"}, cli) && cli.n_ctx == 8192);
        if (had_ctx) {
            setenv("LLAMA_ARG_CTX_SIZE", saved_ctx.c_str(), 1);
        } else {
            unsetenv("LLAMA_ARG_CTX_SIZE");
        }

        std::set<std::string> env_keys;
        std::map<std::string, std::string> saved_env;
        for (const auto & name : {"24dense", "12dense", "10dense", "8dense"}) {
            for (const auto & entry : common_card_preset_get(name).env) {
                env_keys.insert(entry.first);
            }
        }
        for (const auto & key : env_keys) {
            if (const char * value = std::getenv(key.c_str())) {
                saved_env[key] = value;
            }
            unsetenv(key.c_str());
        }
        common_params no_preset;
        assert(common_card_preset_apply_env(no_preset).empty());
        // 8dense loads the mtp-dm.gguf overlay from the model directory: give it a model dir that has one.
        const auto tiny_dir = std::filesystem::temp_directory_path() /
                ("test-arg-parser-8dense-" + std::to_string(std::random_device{}()));
        std::filesystem::create_directories(tiny_dir);
        const std::string tiny_model = (tiny_dir / "AI2M10-iw.gguf").string();
        const std::string tiny_overlay = (tiny_dir / "mtp-dm.gguf").string();
        std::ofstream(tiny_model).put('x');
        std::ofstream(tiny_overlay).put('x');
        for (bool headless : {false, true}) {
            common_params medium_env;
            assert(parse_card({"12dense", headless ? "headless" : "headed"}, medium_env));
            common_card_preset_apply_env(medium_env);
            assert(std::getenv("GGML_CUDA_DISABLE_GRAPHS") == nullptr);
            assert(std::string(std::getenv("GGML_CUDA_GRAPH_MAX_ROWS")) == "16");
            for (const auto & key : env_keys) {
                unsetenv(key.c_str());
            }
        }
        for (const auto & name : {"24dense", "12dense", "10dense", "8dense"}) {
            common_params dynamic;
            assert(parse_card({"--preset", name, "-m", tiny_model}, dynamic));
            const auto enabled = common_card_preset_apply_env(dynamic);
            assert(enabled.find(" LLAMA_DEEPCTX_LADDER=1") != std::string::npos);
            setenv("LLAMA_DEEPCTX_LADDER", "0", 1);
            const auto disabled = common_card_preset_apply_env(dynamic);
            assert(disabled.find(" LLAMA_DEEPCTX_LADDER=0") != std::string::npos);
            for (const auto & key : env_keys) {
                unsetenv(key.c_str());
            }
        }
        common_params small_env;
        assert(parse_card({"--preset", "10dense", "-m", "/models/public-AI2.gguf"}, small_env));
        setenv("LLAMA_LSA_RESIDENT_MIB", "123", 1);
        const auto applied = common_card_preset_apply_env(small_env);
        assert(std::string(std::getenv("LLAMA_LSA_MAP")) == "/models/lsa-qwen38-27b-d2048.gguf");
        assert(std::string(std::getenv("LLAMA_LSA_RESIDENT_MIB")) == "123");
        assert(std::string(std::getenv("LLAMA_LSA_MODE")) == "key");
        assert(applied.find(" LLAMA_LSA_RESIDENT_MIB=123") != std::string::npos);
        for (const auto & key : env_keys) {
            unsetenv(key.c_str());
        }
        common_params explicit_env;
        assert(parse_card({"--attn-sparse", "none", "--attn-sparse-map", "/explicit/map.gguf",
                           "--preset", "10dense"}, explicit_env));
        common_card_preset_apply_env(explicit_env);
        assert(std::string(std::getenv("LLAMA_LSA_MODE")) == "none");
        assert(std::string(std::getenv("LLAMA_LSA_MAP")) == "/explicit/map.gguf");
        for (const auto & key : env_keys) {
            unsetenv(key.c_str());
        }
        common_card_preset_apply_env(kv);
        for (const auto & key : env_keys) {
            if (key.compare(0, 10, "LLAMA_LSA_") == 0 || key == "GGML_CUDA_LSA_RESIDENT_STAGE" || key == "GGML_SJKVARN_TILE_SKIP") {
                assert(!std::getenv(key.c_str()));
            }
            unsetenv(key.c_str());
        }
        // 8dense loads the mtp-dm.gguf drafter overlay beside the model and no D1 head; a missing overlay stops with a hint.
        const std::string saved_overlay = std::getenv("LLAMA_MODEL_OVERLAY_PATCH") ? std::getenv("LLAMA_MODEL_OVERLAY_PATCH") : "";
        const std::string saved_d1 = std::getenv("LLAMA_D1_HEAD") ? std::getenv("LLAMA_D1_HEAD") : "";
        unsetenv("LLAMA_MODEL_OVERLAY_PATCH");
        unsetenv("LLAMA_D1_HEAD");
        common_params tiny_missing;
        assert(parse_card({"8dense", "-m", "/models/tiny-missing/AI2M10-iw.gguf"}, tiny_missing));
        bool missing_threw = false;
        try {
            common_card_preset_apply_env(tiny_missing);
        } catch (const std::runtime_error & e) {
            missing_threw = std::string(e.what()).find("/models/tiny-missing/mtp-dm.gguf") != std::string::npos;
        }
        assert(missing_threw);
        for (const auto & key : env_keys) {
            unsetenv(key.c_str());
        }
        common_params tiny_env;
        assert(parse_card({"8dense", "-m", tiny_model}, tiny_env));
        const auto tiny_applied = common_card_preset_apply_env(tiny_env);
        assert(std::string(std::getenv("LLAMA_MODEL_OVERLAY_PATCH")) == tiny_overlay && !std::getenv("LLAMA_D1_HEAD"));
        assert(tiny_applied.find(" LLAMA_MODEL_OVERLAY_PATCH=" + tiny_overlay) != std::string::npos);
        assert(tiny_applied.find("D1_HEAD") == std::string::npos);
        assert(tiny_env.sampling.spec_lossy);
        assert(std::string(std::getenv("LLAMA_LSA_MAP")) == (tiny_dir / "lsa-qwen38-27b-d2048.gguf").string());
        assert(std::string(std::getenv("LLAMA_LSA_MEAN_CHUNKS")) == "1");
        assert(std::string(std::getenv("LLAMA_RESERVE_VERIFY_ROWS")) == "7");
        assert(!std::getenv("GGML_CUDA_DISABLE_GRAPHS"));
        assert(std::string(std::getenv("GGML_ALLOC_GROW_PCT")) == "1");
        for (const auto & key : env_keys) {
            unsetenv(key.c_str());
        }
        // shell values win; an empty overlay value selects the file's own MTP layer and skips the file check
        setenv("LLAMA_MODEL_OVERLAY_PATCH", "", 1);
        setenv("LLAMA_LSA_MEAN_CHUNKS", "0", 1);
        setenv("LLAMA_RESERVE_VERIFY_ROWS", "9", 1);
        setenv("GGML_CUDA_DISABLE_GRAPHS", "0", 1);
        setenv("GGML_ALLOC_GROW_PCT", "5", 1);
        common_card_preset_apply_env(tiny_missing);
        assert(std::string(std::getenv("LLAMA_MODEL_OVERLAY_PATCH")).empty() && !std::getenv("LLAMA_D1_HEAD"));
        assert(std::string(std::getenv("LLAMA_LSA_MEAN_CHUNKS")) == "0");
        assert(std::string(std::getenv("LLAMA_RESERVE_VERIFY_ROWS")) == "9");
        assert(std::string(std::getenv("GGML_CUDA_DISABLE_GRAPHS")) == "0");
        assert(std::string(std::getenv("GGML_ALLOC_GROW_PCT")) == "5");
        for (const auto & key : env_keys) {
            unsetenv(key.c_str());
        }
        std::filesystem::remove_all(tiny_dir);
        common_params large_env;
        assert(parse_card({"--preset", "24dense"}, large_env));
        common_card_preset_apply_env(large_env);
        assert(std::string(std::getenv("LLAMA_GDN_REPLAY_LANES")) == "1");
        for (const auto & key : env_keys) {
            if (saved_env.count(key)) {
                setenv(key.c_str(), saved_env.at(key).c_str(), 1);
            } else {
                unsetenv(key.c_str());
            }
        }
        if (!saved_overlay.empty()) {
            setenv("LLAMA_MODEL_OVERLAY_PATCH", saved_overlay.c_str(), 1);
        }
        if (!saved_d1.empty()) {
            setenv("LLAMA_D1_HEAD", saved_d1.c_str(), 1);
        }
#endif
    }

    printf("test-arg-parser: test invalid usage\n\n");

    {
        // --moe-skip-tail: 0..9, default 0 (exact)
        common_params defaults;
        assert(defaults.moe_cache.skip_tail == 0);
        std::vector<std::string> args_skip = {"binary_name", "--moe-skip-tail", "7"};
        common_params skip;
        assert(common_params_parse(args_skip.size(), list_str_to_char(args_skip).data(), skip, LLAMA_EXAMPLE_SERVER));
        assert(skip.moe_cache.skip_tail == 7);
        assert(common_context_params_to_llama(skip).moe_skip_tail == 7);
        for (const char * value : {"10", "-1", "x"}) {
            args_skip = {"binary_name", "--moe-skip-tail", value};
            common_params invalid;
            assert(!common_params_parse(args_skip.size(), list_str_to_char(args_skip).data(), invalid, LLAMA_EXAMPLE_SERVER));
        }
        assert(llama_context_default_params().moe_skip_tail == 0);
    }

    {
        // The budget has its own enable; no implicit route-rank cutoff.
        common_params defaults;
        assert(defaults.moe_cache.skip_budget == 0.0f);
        assert(defaults.moe_cache.skip_rank_floor == -1);
        unsetenv("GGML_MOE_SKIP_RANK_FLOOR");
        unsetenv("GGML_MOE_SKIP_POLICY");
        unsetenv("GGML_MOE_SKIP_THRESHOLD");
        assert(common_context_params_to_llama(defaults).moe_skip_tail == 0);
        assert(!common_context_params_to_llama(defaults).moe_skip_mass);
        assert(getenv("GGML_MOE_SKIP_POLICY") == nullptr);
        assert(getenv("GGML_MOE_SKIP_RANK_FLOOR") == nullptr);
        std::vector<std::string> args_budget = {"binary_name", "--moe-skip-budget", "0.2"};
        common_params budget;
        assert(common_params_parse(args_budget.size(), list_str_to_char(args_budget).data(), budget, LLAMA_EXAMPLE_SERVER));
        assert(budget.moe_cache.skip_budget == 0.2f);
        assert(common_context_params_to_llama(budget).moe_skip_tail == 0);
        assert(common_context_params_to_llama(budget).moe_skip_mass);
        assert(std::string(getenv("GGML_MOE_SKIP_POLICY")) == "cumulative");
        assert(std::string(getenv("GGML_MOE_SKIP_THRESHOLD")) == "0.2");
        args_budget = {"binary_name", "--moe-skip-budget", "0.15", "--moe-skip-tail", "8"};
        common_params with_tail;
        assert(common_params_parse(args_budget.size(), list_str_to_char(args_budget).data(), with_tail, LLAMA_EXAMPLE_SERVER));
        assert(common_context_params_to_llama(with_tail).moe_skip_tail == 8);
        assert(common_context_params_to_llama(with_tail).moe_skip_mass);
        assert(std::string(getenv("GGML_MOE_SKIP_THRESHOLD")) == "0.15");
        for (const char * value : {"0", "1", "5", "10"}) {
            args_budget = {"binary_name", "--moe-skip-budget", "0.15", "--moe-skip-rank-floor", value};
            common_params floor;
            assert(common_params_parse(args_budget.size(), list_str_to_char(args_budget).data(), floor, LLAMA_EXAMPLE_SERVER));
            assert(floor.moe_cache.skip_rank_floor == std::stoi(value));
            assert(common_context_params_to_llama(floor).moe_skip_tail == 0);
            assert(common_context_params_to_llama(floor).moe_skip_mass);
            assert(std::string(getenv("GGML_MOE_SKIP_RANK_FLOOR")) == value);
        }
        for (const char * value : {"-1", "11", "x"}) {
            args_budget = {"binary_name", "--moe-skip-rank-floor", value};
            common_params invalid;
            assert(!common_params_parse(args_budget.size(), list_str_to_char(args_budget).data(), invalid, LLAMA_EXAMPLE_SERVER));
        }
        for (bool tail_first : {false, true}) {
            args_budget = tail_first ? std::vector<std::string>{"binary_name", "--moe-skip-tail", "0", "--moe-skip-budget", "0.15"}
                                     : std::vector<std::string>{"binary_name", "--moe-skip-budget", "0.15", "--moe-skip-tail", "0"};
            common_params exact;
            assert(common_params_parse(args_budget.size(), list_str_to_char(args_budget).data(), exact, LLAMA_EXAMPLE_SERVER));
            assert(common_context_params_to_llama(exact).moe_skip_tail == 0);
            assert(common_context_params_to_llama(exact).moe_skip_mass);
        }
        unsetenv("GGML_MOE_SKIP_RANK_FLOOR");
        setenv("GGML_MOE_SKIP_POLICY", "threshold", 1);
        bool threw = false;
        try { common_context_params_to_llama(budget); } catch (const std::invalid_argument &) { threw = true; }
        assert(threw);
        unsetenv("GGML_MOE_SKIP_POLICY");
        unsetenv("GGML_MOE_SKIP_THRESHOLD");
        for (const char * value : {"0.3", "-0.1", "x", "0.2x", "nan"}) {
            args_budget = {"binary_name", "--moe-skip-budget", value};
            common_params invalid;
            assert(!common_params_parse(args_budget.size(), list_str_to_char(args_budget).data(), invalid, LLAMA_EXAMPLE_SERVER));
        }
    }

    {
        auto parse_skip = [&](std::vector<std::string> args) {
            common_params parsed;
            assert(common_params_parse(args.size(), list_str_to_char(args).data(), parsed, LLAMA_EXAMPLE_SERVER));
            return parsed;
        };
        auto preset = parse_skip({"binary_name", "--moe-skip"});
        assert(preset.moe_cache.skip_tail == 0);
        assert(preset.moe_cache.skip_budget == 0.15f && preset.moe_cache.skip_rank_floor == 5);
        assert(common_context_params_to_llama(preset).moe_skip_mass);
        // Omitted value must not eat the next option; mapping is used by presets too.
        auto next = parse_skip({"binary_name", "--moe-skip", "--ctx-size", "4096"});
        assert(next.n_ctx == 4096 && next.moe_cache.skip_budget == 0.15f);
        std::vector<std::string> args = {"binary_name", "--moe-skip", "--ctx-size", "4096"};
        std::map<common_arg, std::string> mapped;
        assert(common_params_to_map(args.size(), list_str_to_char(args).data(), LLAMA_EXAMPLE_SERVER, mapped));
        for (const auto & entry : mapped) {
            if (std::string(entry.first.args.front()) == "--moe-skip") { assert(entry.second == "on"); }
        }
        for (const char * name : {"--moe-skip-budget", "--moe-skip-rank-floor", "--moe-skip-tail"}) {
            const std::string value = std::string(name) == "--moe-skip-budget" ? "0.1" : "8";
            for (bool first : {false, true}) {
                auto overridden = parse_skip(first ? std::vector<std::string>{"binary_name", name, value, "--moe-skip"}
                                                   : std::vector<std::string>{"binary_name", "--moe-skip", name, value});
                assert(overridden.moe_cache.skip_budget == (std::string(name) == "--moe-skip-budget" ? 0.1f : 0.15f));
                assert(overridden.moe_cache.skip_rank_floor == (std::string(name) == "--moe-skip-rank-floor" ? 8 : 5));
                assert(overridden.moe_cache.skip_tail == (std::string(name) == "--moe-skip-tail" ? 8 : 0));
            }
        }
        // An explicit zero budget overrides the preset too.
        for (bool first : {false, true}) {
            auto zero = parse_skip(first ? std::vector<std::string>{"binary_name", "--moe-skip-budget", "0", "--moe-skip"}
                                         : std::vector<std::string>{"binary_name", "--moe-skip", "--moe-skip-budget", "0"});
            assert(!common_context_params_to_llama(zero).moe_skip_mass);
        }
        auto off = parse_skip({"binary_name", "--moe-skip", "off"});
        assert(off.moe_cache.skip_budget == 0 && off.moe_cache.skip_rank_floor == 0);
        assert(!common_context_params_to_llama(off).moe_skip_mass && off.moe_cache.skip_tail == 0);
        auto legacy_off = parse_skip({"binary_name", "--moe-skip-tail", "7", "--moe-skip", "off"});
        assert(legacy_off.moe_cache.skip_tail == 0);
        setenv("LLAMA_ARG_MOE_SKIP", "on", 1);
        assert(parse_skip({"binary_name"}).moe_cache.skip_budget == 0.15f);
        assert(parse_skip({"binary_name", "--moe-skip", "off"}).moe_cache.skip_budget == 0);
        setenv("LLAMA_ARG_MOE_SKIP", "off", 1);
        assert(parse_skip({"binary_name"}).moe_cache.skip_budget == 0);
        assert(parse_skip({"binary_name", "--moe-skip"}).moe_cache.skip_budget == 0.15f);
        unsetenv("LLAMA_ARG_MOE_SKIP");
        args = {"binary_name", "--moe-skip", "bad"};
        common_params invalid;
        assert(!common_params_parse(args.size(), list_str_to_char(args).data(), invalid, LLAMA_EXAMPLE_SERVER));
        unsetenv("GGML_MOE_SKIP_POLICY");
        unsetenv("GGML_MOE_SKIP_THRESHOLD");
        unsetenv("GGML_MOE_SKIP_RANK_FLOOR");
    }

#ifndef _WIN32
    {
        const char * prev = getenv("LLAMA_MOE_CPU_SHARE");
        const bool had_prev = prev != nullptr;
        const std::string saved = prev ? prev : "";
        auto parse_share = [&list_str_to_char](std::vector<std::string> args) {
            common_params p;
            return common_params_parse(args.size(), list_str_to_char(args).data(), p, LLAMA_EXAMPLE_SERVER);
        };
        unsetenv("LLAMA_MOE_CPU_SHARE");
        assert(parse_share({"binary_name"}));
        assert(getenv("LLAMA_MOE_CPU_SHARE") == nullptr);
        for (const char * value : {"0", "0.6", "0.8", "1"}) {
            assert(parse_share({"binary_name", "--moe-cpu-share", value}));
            assert(std::string(getenv("LLAMA_MOE_CPU_SHARE")) == value);
        }
        setenv("LLAMA_MOE_CPU_SHARE", "0.8", 1);
        assert(parse_share({"binary_name"}));
        assert(parse_share({"binary_name", "--moe-cpu-share", "0.6"}));
        assert(std::string(getenv("LLAMA_MOE_CPU_SHARE")) == "0.6");
        for (const char * value : {"-0.1", "1.1", "nan", "inf", "0.6junk", ""}) {
            unsetenv("LLAMA_MOE_CPU_SHARE");
            assert(!parse_share({"binary_name", "--moe-cpu-share", value}));
            setenv("LLAMA_MOE_CPU_SHARE", value, 1);
            if (*value) { assert(!parse_share({"binary_name"})); }
        }
        if (had_prev) { setenv("LLAMA_MOE_CPU_SHARE", saved.c_str(), 1); }
        else { unsetenv("LLAMA_MOE_CPU_SHARE"); }
    }
#endif

    // missing value
    argv = {"binary_name", "-m"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (int)
    argv = {"binary_name", "-ngl", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (enum)
    argv = {"binary_name", "-sm", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    {
        common_params penalty_params;
        assert(penalty_params.sampling.penalty_last_n == 64);
        assert(penalty_params.sampling.dry_penalty_last_n == 64);

        argv = {"binary_name", "--repeat-last-n", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--dry-penalty-last-n", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "0"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "nan"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "inf"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "-inf"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        const char * penalty_options[] = {"--frequency-penalty", "--presence-penalty"};
        const char * nonfinite_values[] = {"nan", "inf", "-inf"};
        for (const char * option : penalty_options) {
            for (const char * value : nonfinite_values) {
                argv = {"binary_name", option, value};
                assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));
            }
        }
    }

    // non-existence arg in specific example (--draft cannot be used outside llama-speculative)
    argv = {"binary_name", "--draft", "123"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_EMBEDDING));

    argv = {"binary_name", "-lm", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    printf("test-arg-parser: test valid usage\n\n");

    argv = {"binary_name", "-m", "model_file.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "model_file.gguf");

    {
        auto parse_shards = [&list_str_to_char](std::vector<std::string> args) {
            common_params shard_params;
            args.insert(args.begin(), {"binary_name", "-m", "model_file.gguf"});
            assert(common_params_parse(args.size(), list_str_to_char(args).data(), shard_params, LLAMA_EXAMPLE_COMMON));
            auto paths = shard_params.model.get_paths();
            assert(paths.front() == "model_file.gguf");
            return std::vector<std::string>(paths.begin() + 1, paths.end());
        };
        assert(parse_shards({"--shard", "a.gguf"}) == std::vector<std::string>({"a.gguf"}));
        const std::vector<std::string> two = {"a.gguf", "b.gguf"};
        assert(parse_shards({"--shard", "a.gguf", "--shard", "b.gguf"}) == two);
        assert(parse_shards({"--shard", "a.gguf,b.gguf"}) == two);
        const std::vector<std::string> three = {"a.gguf", "b.gguf", "c.gguf"};
        assert(parse_shards({"--shard", "a.gguf,b.gguf", "--shard", "c.gguf"}) == three);
        assert(parse_shards({"--shard", "a.gguf", "--shard", "b.gguf", "--shard", "c.gguf"}) == three);
        assert(parse_shards({"--ple", "p.gguf"}) == parse_shards({"--shard", "p.gguf"}));
        assert(parse_shards({"--shard", "model_file.gguf,a.gguf,b.gguf,a.gguf"}) == two);
    }

    argv = {"binary_name", "-t", "1234"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.cpuparams.n_threads == 1234);

    argv = {"binary_name", "--verbose"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.verbosity > 1);

    argv = {"binary_name", "-m", "abc.gguf", "--predict", "6789", "--batch-size", "9090"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "abc.gguf");
    assert(params.n_predict == 6789);
    assert(params.n_batch == 9090);

    printf("test-arg-parser: test --cache-disk-min-free units\n\n");
    {
        struct mf_case { const char * v; bool ok; int32_t mib; float pct; };
        const mf_case cases[] = {
            { "2048",   true,  2048,  -1.0f }, // plain number = MiB (unchanged meaning)
            { "-1",     true,  -1,    -1.0f },
            { "0",      true,  0,     -1.0f },
            { "20G",    true,  20480, -1.0f },
            { "1.5GiB", true,  1536,  -1.0f },
            { "512M",   true,  512,   -1.0f },
            { "5%",     true,  -1,    5.0f  },
            { "0%",     true,  -1,    0.0f  },
            { "101%",   false, 0,     0.0f  },
            { "-2",     false, 0,     0.0f  },
            { "-1G",    false, 0,     0.0f  },
            { "abc",    false, 0,     0.0f  },
            { "10X",    false, 0,     0.0f  },
        };
        for (const auto & c : cases) {
            common_params mf;
            argv = {"binary_name", "--cache-disk-min-free", c.v};
            const bool ok = common_params_parse(argv.size(), list_str_to_char(argv).data(), mf, LLAMA_EXAMPLE_SERVER);
            if (ok != c.ok || (ok && (mf.cache_disk_min_free_mib != c.mib || mf.cache_disk_min_free_pct != c.pct))) {
                fprintf(stderr, "--cache-disk-min-free %s: ok=%d mib=%d pct=%g\n", c.v, ok, mf.cache_disk_min_free_mib, mf.cache_disk_min_free_pct);
                assert(false);
            }
        }
        // the later of two settings wins, whichever unit
        common_params mf;
        argv = {"binary_name", "--cache-disk-min-free", "5%", "--cache-disk-min-free", "100"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), mf, LLAMA_EXAMPLE_SERVER));
        assert(mf.cache_disk_min_free_mib == 100 && mf.cache_disk_min_free_pct < 0.0f);
    }

    printf("test-arg-parser: test KV cache type names and aliases\n\n");
    {
        const std::vector<std::pair<std::string, ggml_type>> names = {
            // ggml type names
            {"f16",      GGML_TYPE_F16},      {"q8_0",     GGML_TYPE_Q8_0},
            {"turbo2",   GGML_TYPE_TURBO2_0}, {"turbo3",   GGML_TYPE_TURBO3_0},
            {"turbo4",   GGML_TYPE_TURBO4_0}, {"tq5_0",    GGML_TYPE_TQ5_0},
            {"tq6_0",    GGML_TYPE_TQ6_0},
            // aliases
            {"tq2",      GGML_TYPE_TURBO2_0},
            {"tq3",      GGML_TYPE_TURBO3_0}, {"tq3_0",    GGML_TYPE_TURBO3_0},
            {"tq4",      GGML_TYPE_TURBO4_0}, {"tq4_0",    GGML_TYPE_TURBO4_0},
            {"turbo5",   GGML_TYPE_TQ5_0},    {"tq5",      GGML_TYPE_TQ5_0},
            {"turbo6",   GGML_TYPE_TQ6_0},    {"tq6",      GGML_TYPE_TQ6_0},
            // case-insensitive
            {"Turbo5",   GGML_TYPE_TQ5_0},    {"TQ4",      GGML_TYPE_TURBO4_0},
        };
        for (const auto & [name, type] : names) {
            assert(common_kv_cache_type_from_name(name) == type);

            common_params kv_params;
            argv = {"binary_name", "-ctk", name, "-ctv", name, "-ctkd", name, "-ctvd", name};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), kv_params, LLAMA_EXAMPLE_SERVER));
            assert(kv_params.cache_type_k == type);
            assert(kv_params.cache_type_v == type);
            assert(kv_params.speculative.draft.cache_type_k == type);
            assert(kv_params.speculative.draft.cache_type_v == type);
        }

        // unknown names are rejected; tq2_0 is ggml's ternary weight type, and the
        // TurboQuant weight formats and a turbo5_0/turbo6_0 spelling are not cache types
        for (const std::string name : {"hello", "tq2_0", "tq1_0", "tq3_1s", "tq4_1s", "turbo5_0", "turbo6_0", "turbo7", "tq7"}) {
            assert(common_kv_cache_type_from_name(name) == GGML_TYPE_COUNT);

            common_params kv_params;
            argv = {"binary_name", "-ctk", name};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), kv_params, LLAMA_EXAMPLE_COMMON));
        }
    }

    // --draft cannot be used outside llama-speculative
    argv = {"binary_name", "--spec-draft-n-max", "123"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SPECULATIVE));
    assert(params.speculative.draft.n_max == 123);

    // speculative draft defaults and adaptive floor
    argv = {"binary_name", "-m", "model_file.gguf"};
    common_params spec_defaults;
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), spec_defaults, LLAMA_EXAMPLE_SPECULATIVE));
    assert(spec_defaults.speculative.draft.n_max == 3);
    assert(spec_defaults.speculative.draft.n_min_adaptive == 3);

    for (const std::string type : {"draft-mtp", "draft-mtp-adaptive"}) {
        for (const std::string mode : {"default", "--spec-mb-mtp", "--no-spec-mb-mtp"}) {
            argv = {"binary_name", "--spec-type", type};
            if (mode != "default") { argv.push_back(mode); }
            common_params mb;
            assert(common_params_parse(argv.size(), list_str_to_char(argv).data(), mb, LLAMA_EXAMPLE_SERVER));
            const bool enabled = mode != "--no-spec-mb-mtp";
            assert(mb.sampling.spec_lossy == enabled);
            assert(mb.speculative.draft.n_max == (enabled ? 5 : 3));
            argv.insert(argv.end(), {"--spec-draft-n-max", "4"});
            common_params explicit_depth;
            assert(common_params_parse(argv.size(), list_str_to_char(argv).data(), explicit_depth, LLAMA_EXAMPLE_SERVER));
            assert(explicit_depth.speculative.draft.n_max == 4);
        }
    }

    // MTP drafter attention window: off by default, the default context params carry no window,
    // and the cparams helper leaves them untouched when the flags are absent
    assert(spec_defaults.speculative.draft.attn_window == 0);
    assert(spec_defaults.speculative.draft.attn_sink == -1);
    {
        llama_context_params cp = llama_context_default_params();
        assert(cp.draft_attn_window == 0 && cp.draft_attn_sink == 0);
        common_speculative_mtp_cparams(spec_defaults, cp);
        assert(cp.draft_attn_window == 0 && cp.draft_attn_sink == 0);
    }
    // DSPARKWIN: the DSpark drafter window is off by default, separate from the MTP window, and its helper only
    // writes the draft context params when it is set
    assert(spec_defaults.speculative.draft.dspark_window == 0);
    assert(spec_defaults.speculative.draft.dspark_window_sink == -1);
    {
        llama_context_params cp = llama_context_default_params();
        common_speculative_dspark_cparams(spec_defaults, cp);
        assert(cp.draft_attn_window == 0 && cp.draft_attn_sink == 0);
    }
    for (const auto & ex : {LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}) {
        argv = {"binary_name", "-m", "model_file.gguf", "--spec-dspark-window", "8192"};
        common_params dw;
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), dw, ex));
        assert(dw.speculative.draft.dspark_window == 8192 && dw.speculative.draft.attn_window == 0);
        llama_context_params cp = llama_context_default_params();
        common_speculative_mtp_cparams(dw, cp);
        assert(cp.draft_attn_window == 0); // the MTP helper ignores the DSpark window
        common_speculative_dspark_cparams(dw, cp);
        assert(cp.draft_attn_window == 8192 && cp.draft_attn_sink == 128);

        argv = {"binary_name", "-m", "model_file.gguf", "--spec-dspark-window", "4096", "--spec-dspark-window-sink", "0"};
        common_params dws;
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), dws, ex));
        cp = llama_context_default_params();
        common_speculative_dspark_cparams(dws, cp);
        assert(cp.draft_attn_window == 4096 && cp.draft_attn_sink == 0);
    }
    for (const auto & ex : {LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}) {
        argv = {"binary_name", "-m", "model_file.gguf", "--spec-draft-window", "8192"};
        common_params window_params;
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), window_params, ex));
        assert(window_params.speculative.draft.attn_window == 8192);
        llama_context_params cp = llama_context_default_params();
        common_speculative_mtp_cparams(window_params, cp);
        assert(cp.draft_attn_window == 8192 && cp.draft_attn_sink == 128); // default sink

        argv = {"binary_name", "-m", "model_file.gguf", "--spec-draft-window", "4096", "--spec-draft-window-sink", "0"};
        common_params sink_params;
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), sink_params, ex));
        cp = llama_context_default_params();
        common_speculative_mtp_cparams(sink_params, cp);
        assert(cp.draft_attn_window == 4096 && cp.draft_attn_sink == 0);

        // a negative window is clamped to 0 (off)
        argv = {"binary_name", "-m", "model_file.gguf", "--spec-draft-window", "-5"};
        common_params neg_params;
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), neg_params, ex));
        assert(neg_params.speculative.draft.attn_window == 0);
    }

    argv = {"binary_name", "-m", "model_file.gguf", "--spec-draft-n-min-adaptive", "5"};
    common_params adaptive_params;
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), adaptive_params, LLAMA_EXAMPLE_SPECULATIVE));
    assert(adaptive_params.speculative.draft.n_min_adaptive == 5);

    argv = {"binary_name", "-m", "model_file.gguf", "--spec-chain", "8"};
    common_params chain_params;
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), chain_params, LLAMA_EXAMPLE_SPECULATIVE));
    assert(chain_params.speculative.draft.chain);
    assert(chain_params.speculative.draft.n_max == 8);

    chain_params.n_batch = 8;
    chain_params.n_ubatch = 8;
    chain_params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    const llama_context_params chain_ctx_params = common_context_params_to_llama(chain_params);
    assert(chain_ctx_params.n_rs_seq == 8);
    assert(chain_ctx_params.n_batch == 10);
    assert(chain_ctx_params.n_ubatch == 10);

    argv = {"binary_name", "-m", "model_file.gguf", "--spec-chain", "0"};
    common_params no_chain_params;
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), no_chain_params, LLAMA_EXAMPLE_SPECULATIVE));
    assert(!no_chain_params.speculative.draft.chain);
    assert(no_chain_params.speculative.draft.n_max == 3);

    // draft KV cache types inherit the main -ctk/-ctv unless --spec-draft-type-k/-v is given (GGML_TYPE_COUNT = not set),
    // K and V independently; an explicit f16 stays explicit
    argv = {"binary_name", "-m", "model_file.gguf", "-ctk", "q8_0", "-ctv", "q4_0"};
    common_params dkv_inherit;
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), dkv_inherit, LLAMA_EXAMPLE_SPECULATIVE));
    assert(dkv_inherit.speculative.draft.cache_type_k == GGML_TYPE_COUNT);
    assert(dkv_inherit.speculative.draft.cache_type_v == GGML_TYPE_COUNT);
    {
        const auto draft = common_base_params_to_speculative(dkv_inherit);
        assert(draft.cache_type_k == GGML_TYPE_Q8_0);
        assert(draft.cache_type_v == GGML_TYPE_Q4_0);
    }

    argv = {"binary_name", "-m", "model_file.gguf", "-ctk", "q8_0", "-ctv", "q4_0", "-ctkd", "f16"};
    common_params dkv_k_only;
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), dkv_k_only, LLAMA_EXAMPLE_SPECULATIVE));
    assert(dkv_k_only.speculative.draft.cache_type_k == GGML_TYPE_F16);
    assert(dkv_k_only.speculative.draft.cache_type_v == GGML_TYPE_COUNT);
    {
        const auto draft = common_base_params_to_speculative(dkv_k_only);
        assert(draft.cache_type_k == GGML_TYPE_F16);
        assert(draft.cache_type_v == GGML_TYPE_Q4_0);
    }

    argv = {"binary_name", "-m", "model_file.gguf", "-ctk", "q8_0", "-ctv", "q4_0",
            "--spec-draft-type-k", "q5_0", "--spec-draft-type-v", "f16"};
    common_params dkv_both;
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), dkv_both, LLAMA_EXAMPLE_SPECULATIVE));
    assert(dkv_both.speculative.draft.cache_type_k == GGML_TYPE_Q5_0);
    assert(dkv_both.speculative.draft.cache_type_v == GGML_TYPE_F16);
    {
        const auto draft = common_base_params_to_speculative(dkv_both);
        assert(draft.cache_type_k == GGML_TYPE_Q5_0);
        assert(draft.cache_type_v == GGML_TYPE_F16);
    }

    // a SJ-KVaRN trunk cannot be inherited: unset draft types default to tq5_0 K / turbo4 V, explicit ones win
    argv = {"binary_name", "-m", "model_file.gguf", "-ctk", "sjkvarn4", "-ctv", "sjkvarn4"};
    common_params dkv_sj_kvarn;
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), dkv_sj_kvarn, LLAMA_EXAMPLE_SPECULATIVE));
    {
        const auto draft = common_base_params_to_speculative(dkv_sj_kvarn);
        assert(draft.cache_type_k == COMMON_SJKVARN_MTP_DRAFT_KV[0]);
        assert(draft.cache_type_v == COMMON_SJKVARN_MTP_DRAFT_KV[1]);
        assert(draft.speculative.draft.cache_type_sj_kvarn_default);
        assert(draft.sj_kvarn_bits_k == 0 && draft.sj_kvarn_bits_v == 0);
    }
    argv = {"binary_name", "-m", "model_file.gguf", "-ctk", "sjkvarn4", "-ctv", "sjkvarn4", "-ctkd", "f16", "-ctvd", "tq6_0"};
    common_params dkv_sj_kvarn_pin;
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), dkv_sj_kvarn_pin, LLAMA_EXAMPLE_SPECULATIVE));
    {
        const auto draft = common_base_params_to_speculative(dkv_sj_kvarn_pin);
        assert(draft.cache_type_k == GGML_TYPE_F16);
        assert(draft.cache_type_v == GGML_TYPE_TQ6_0);
    }
    {
        const auto draft = common_base_params_to_speculative(dkv_inherit);
        assert(!draft.speculative.draft.cache_type_sj_kvarn_default);
    }
    // one side explicit, the other takes the SJ-KVaRN default
    argv = {"binary_name", "-m", "model_file.gguf", "-ctk", "sjkvarn4", "-ctv", "sjkvarn4", "-ctkd", "q8_0"};
    common_params dkv_sj_kvarn_k;
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), dkv_sj_kvarn_k, LLAMA_EXAMPLE_SPECULATIVE));
    {
        const auto draft = common_base_params_to_speculative(dkv_sj_kvarn_k);
        assert(draft.cache_type_k == GGML_TYPE_Q8_0);
        assert(draft.cache_type_v == COMMON_SJKVARN_MTP_DRAFT_KV[1]);
    }
    // a separate draft model with no MTP type keeps the plain cache the SJ-KVaRN trunk flags leave behind
    argv = {"binary_name", "-m", "model_file.gguf", "-md", "draft.gguf", "-ctk", "sjkvarn4", "-ctv", "sjkvarn4"};
    common_params dkv_sj_kvarn_md;
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), dkv_sj_kvarn_md, LLAMA_EXAMPLE_SPECULATIVE));
    {
        const auto draft = common_base_params_to_speculative(dkv_sj_kvarn_md);
        assert(draft.cache_type_k == dkv_sj_kvarn_md.cache_type_k);
        assert(draft.cache_type_v == dkv_sj_kvarn_md.cache_type_v);
        assert(!draft.speculative.draft.cache_type_sj_kvarn_default);
    }
    // a separate-file MTP head (ctx_type MTP) is an MTP drafter: unset types take tq5_0/turbo4
    argv = {"binary_name", "-m", "model_file.gguf", "-md", "draft.gguf", "--spec-type", "draft-mtp-adaptive",
            "-ctk", "sjkvarn4", "-ctv", "sjkvarn4"};
    common_params dkv_sj_kvarn_md_mtp;
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), dkv_sj_kvarn_md_mtp, LLAMA_EXAMPLE_SERVER));
    {
        const auto draft = common_base_params_to_speculative(dkv_sj_kvarn_md_mtp);
        assert(draft.cache_type_k == COMMON_SJKVARN_MTP_DRAFT_KV[0]);
        assert(draft.cache_type_v == COMMON_SJKVARN_MTP_DRAFT_KV[1]);
        assert(draft.speculative.draft.cache_type_sj_kvarn_default);
        assert(draft.sj_kvarn_bits_k == 0 && draft.sj_kvarn_bits_v == 0);
    }
    // explicit f16 still wins on that separate MTP file
    argv = {"binary_name", "-m", "model_file.gguf", "-md", "draft.gguf", "--spec-type", "draft-mtp-adaptive",
            "-ctk", "sjkvarn4", "-ctv", "sjkvarn4", "-ctkd", "f16", "-ctvd", "f16"};
    common_params dkv_sj_kvarn_md_mtp_f16;
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), dkv_sj_kvarn_md_mtp_f16, LLAMA_EXAMPLE_SERVER));
    {
        const auto draft = common_base_params_to_speculative(dkv_sj_kvarn_md_mtp_f16);
        assert(draft.cache_type_k == GGML_TYPE_F16);
        assert(draft.cache_type_v == GGML_TYPE_F16);
        assert(!draft.speculative.draft.cache_type_sj_kvarn_default);
    }
    // EAGLE3 is a separate draft but not MTP, so it keeps the plain SJ-KVaRN cache
    argv = {"binary_name", "-m", "model_file.gguf", "-md", "draft.gguf", "--spec-type", "draft-eagle3",
            "-ctk", "sjkvarn4", "-ctv", "sjkvarn4"};
    common_params dkv_sj_kvarn_eagle;
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), dkv_sj_kvarn_eagle, LLAMA_EXAMPLE_SERVER));
    {
        const auto draft = common_base_params_to_speculative(dkv_sj_kvarn_eagle);
        assert(draft.cache_type_k == dkv_sj_kvarn_eagle.cache_type_k);
        assert(draft.cache_type_v == dkv_sj_kvarn_eagle.cache_type_v);
        assert(draft.cache_type_k == GGML_TYPE_F16);
        assert(!draft.speculative.draft.cache_type_sj_kvarn_default);
    }
    // a quantized trunk stays quantized for a separate MTP file; it is not left at f16
    argv = {"binary_name", "-m", "model_file.gguf", "-md", "draft.gguf", "--spec-type", "draft-mtp",
            "-ctk", "q8_0", "-ctv", "q8_0"};
    common_params dkv_q8_md_mtp;
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), dkv_q8_md_mtp, LLAMA_EXAMPLE_SERVER));
    {
        const auto draft = common_base_params_to_speculative(dkv_q8_md_mtp);
        assert(draft.cache_type_k == GGML_TYPE_Q8_0);
        assert(draft.cache_type_v == GGML_TYPE_Q8_0);
        assert(!draft.speculative.draft.cache_type_sj_kvarn_default);
    }

    argv = {"binary_name", "-m", "model_file.gguf", "--spec-type", "draft-mtp-adaptive",
            "--spec-draft-sjkvarn", "-ctk", "sjkvarn3", "-ctv", "sjkvarn2", "--sjkvarn-body-type", "sjkvarn4t",
            "--sjkvarn-staging-type", "tq6_0", "--sjkvarn-sink-type", "f16", "--sjkvarn-tail", "4096",
            "--sjkvarn-tail-max", "8192", "--sjkvarn-flush-chunk", "4", "--parallel", "1"};
    common_params dkv_sj_kvarn_inh;
    assert(common_params_parse(argv.size(), list_str_to_char(argv).data(), dkv_sj_kvarn_inh, LLAMA_EXAMPLE_SERVER));
    {
        const auto draft = common_base_params_to_speculative(dkv_sj_kvarn_inh);
        const auto cparams = common_context_params_to_llama(draft);
        assert(cparams.sj_kvarn_bits_k == 3 && cparams.sj_kvarn_bits_v == 2);
        assert(cparams.sj_kvarn_body_type == GGML_TYPE_I16);
        assert(cparams.sj_kvarn_staging_type == GGML_TYPE_TQ6_0);
        assert(cparams.sj_kvarn_sink_type == GGML_TYPE_F16);
        assert(cparams.sj_kvarn_tail == 4096 && cparams.sj_kvarn_tail_max == 8192);
        assert(cparams.sj_kvarn_flush_chunk == 4 && cparams.sj_kvarn_edge_layers == 0);
        assert(cparams.type_k == GGML_TYPE_TQ6_0 && cparams.type_v == GGML_TYPE_TQ6_0);

        auto invalid = [](const common_params & p) {
            bool rejected = false;
            try {
                common_base_params_to_speculative(p);
            } catch (const std::invalid_argument &) {
                rejected = true;
            }
            assert(rejected);
        };
        auto bad = dkv_sj_kvarn_inh;
        bad.speculative.draft.cache_type_k = GGML_TYPE_Q8_0;
        invalid(bad);
        bad = dkv_sj_kvarn_inh;
        bad.speculative.draft.cache_type_v = GGML_TYPE_F16;
        invalid(bad);
        bad = dkv_sj_kvarn_inh;
        bad.n_parallel = 2;
        invalid(bad);
        bad = dkv_sj_kvarn_inh;
        bad.sj_kvarn_bits_k = 0;
        invalid(bad);
        bad = dkv_sj_kvarn_inh;
        bad.speculative.types = { COMMON_SPECULATIVE_TYPE_NONE };
        invalid(bad);
        bad = dkv_sj_kvarn_inh;
        bad.speculative.draft.mparams.path = "draft.gguf";
        invalid(bad);
        bad = dkv_sj_kvarn_inh;
        bad.sj_kvarn_tail = 0;
        invalid(bad);
        for (int width = 1; width <= 8; ++width) {
            auto p = dkv_sj_kvarn_inh;
            p.n_ctx = 262144;
            p.speculative.draft.n_max = width;
            assert(common_context_params_to_llama(common_base_params_to_speculative(p)).sj_kvarn_tail == 4096);
        }
        bad = dkv_sj_kvarn_inh;
        bad.speculative.draft.sj_kvarn = false;
        const auto plain = common_base_params_to_speculative(bad);
        assert(plain.sj_kvarn_bits_k == 0 && plain.sj_kvarn_bits_v == 0);
        assert(plain.sj_kvarn_tail_max == 0 && plain.sj_kvarn_flush_chunk == 0);
        assert(plain.cache_type_k == COMMON_SJKVARN_MTP_DRAFT_KV[0] && plain.cache_type_v == COMMON_SJKVARN_MTP_DRAFT_KV[1]);
        assert(plain.speculative.draft.cache_type_sj_kvarn_default);
        assert(!draft.speculative.draft.cache_type_sj_kvarn_default);
    }

    {
        auto automatic = dkv_sj_kvarn_inh;
        automatic.speculative.draft.sj_kvarn = false;
        automatic.speculative.draft.sj_kvarn_explicit = false;
        automatic.speculative.draft.attn_window = 0;
        assert(common_speculative_draft_sj_kvarn_error(automatic) == nullptr);
        const auto draft = common_base_params_to_speculative(automatic);
        assert(draft.speculative.draft.sj_kvarn);
        assert(draft.sj_kvarn_bits_k == automatic.sj_kvarn_bits_k);
        assert(draft.sj_kvarn_bits_v == automatic.sj_kvarn_bits_v);
        assert(draft.cache_type_k == automatic.sj_kvarn_staging_type);
        assert(draft.cache_type_v == automatic.sj_kvarn_staging_type);
        assert(draft.speculative.draft.vocab_map == "auto:rm24h" && draft.speculative.draft.vocab_hot == 0);

        auto window = automatic;
        window.speculative.draft.attn_window = 4096;
        window.speculative.draft.attn_sink = 64;
        for (bool explicit_choice : { false, true }) {
            window.speculative.draft.sj_kvarn_explicit = explicit_choice;
            assert(common_speculative_draft_sj_kvarn_error(window) != nullptr);
            const auto fallback = common_base_params_to_speculative(window);
            assert(!fallback.speculative.draft.sj_kvarn);
            assert(fallback.sj_kvarn_bits_k == 0 && fallback.sj_kvarn_bits_v == 0);
            assert(fallback.cache_type_k == COMMON_SJKVARN_MTP_DRAFT_KV[0]);
            assert(fallback.cache_type_v == COMMON_SJKVARN_MTP_DRAFT_KV[1]);
            assert(fallback.speculative.draft.cache_type_sj_kvarn_default);
            assert(fallback.speculative.draft.vocab_map == "auto");
            auto cparams = common_context_params_to_llama(fallback);
            common_speculative_mtp_cparams(fallback, cparams);
            assert(cparams.sj_kvarn_bits_k == 0 && cparams.sj_kvarn_bits_v == 0);
            assert(cparams.draft_attn_window == 4096 && cparams.draft_attn_sink == 64);
        }
        window.speculative.draft.sj_kvarn = true;
        bool rejected = false;
        try {
            common_base_params_to_speculative(window);
        } catch (const std::invalid_argument & e) {
            rejected = true;
            assert(std::string(e.what()).find("--spec-draft-window") != std::string::npos);
        }
        assert(rejected);

        for (int mode = 0; mode < 9; ++mode) {
            auto unsupported = automatic;
            switch (mode) {
                case 0: unsupported.speculative.types = { COMMON_SPECULATIVE_TYPE_NONE }; break;
                case 1: unsupported.speculative.draft.mparams.path = "draft.gguf"; break;
                case 2: unsupported.n_parallel = 2; break;
                case 3: unsupported.sj_kvarn_bits_k = 0; break;
                case 4: unsupported.sj_kvarn_bits_v = 0; break;
                case 5: unsupported.speculative.draft.cache_type_k = GGML_TYPE_F16; break;
                case 6: unsupported.speculative.draft.cache_type_v = GGML_TYPE_Q8_0; break;
                case 7: unsupported.sj_kvarn_tail = unsupported.speculative.draft.n_max; break;
                case 8: unsupported.speculative.draft.n_max = -1; break;
            }
            assert(common_speculative_draft_sj_kvarn_error(unsupported) != nullptr);
            const auto fallback = common_base_params_to_speculative(unsupported);
            assert(!fallback.speculative.draft.sj_kvarn && fallback.speculative.draft.vocab_map == "auto");
        }
        for (int mode = 0; mode < 4; ++mode) {
            auto overridden = automatic;
            switch (mode) {
                case 0: overridden.speculative.user_set |= COMMON_PARAMS_SPECULATIVE_USER_DRAFT_VOCAB_MAP; break;
                case 1: overridden.speculative.user_set |= COMMON_PARAMS_SPECULATIVE_USER_DRAFT_VOCAB_HOT; break;
                case 2: overridden.speculative.draft.vocab_map = "custom.txt"; break;
                case 3: overridden.speculative.draft.vocab_hot = 17; break;
            }
            const auto kept = common_base_params_to_speculative(overridden);
            assert(kept.speculative.draft.vocab_map == overridden.speculative.draft.vocab_map);
            assert(kept.speculative.draft.vocab_hot == overridden.speculative.draft.vocab_hot);
        }
        automatic.sj_kvarn_tail = automatic.speculative.draft.n_max + 1;
        assert(common_speculative_draft_sj_kvarn_error(automatic) == nullptr);
        automatic.speculative.draft.sj_kvarn_explicit = true;
        const auto disabled = common_base_params_to_speculative(automatic);
        assert(!disabled.speculative.draft.sj_kvarn);
        assert(disabled.sj_kvarn_bits_k == 0 && disabled.sj_kvarn_bits_v == 0);
        assert(disabled.cache_type_k == COMMON_SJKVARN_MTP_DRAFT_KV[0]);
        assert(disabled.cache_type_v == COMMON_SJKVARN_MTP_DRAFT_KV[1]);
        assert(disabled.speculative.draft.vocab_map == "auto");

        for (bool enabled : { false, true }) {
            argv = {"binary_name", enabled ? "--no-spec-draft-sjkvarn" : "--spec-draft-sjkvarn",
                    enabled ? "--spec-draft-sjkvarn" : "--no-spec-draft-sjkvarn",
                    "--spec-draft-vocab-map", "auto", "--spec-draft-vocab-hot", "0"};
            common_params parsed;
            assert(common_params_parse(argv.size(), list_str_to_char(argv).data(), parsed, LLAMA_EXAMPLE_SERVER));
            assert(parsed.speculative.draft.sj_kvarn == enabled);
            assert(parsed.speculative.draft.sj_kvarn_explicit);
            assert(parsed.speculative.draft.vocab_map == "auto" && parsed.speculative.draft.vocab_hot == 0);
            assert(parsed.speculative.user_set & COMMON_PARAMS_SPECULATIVE_USER_DRAFT_VOCAB_MAP);
            assert(parsed.speculative.user_set & COMMON_PARAMS_SPECULATIVE_USER_DRAFT_VOCAB_HOT);
        }
    }

    // n-gram drafters request recurrent-state snapshots for in-place rollback (draft width, capped at 8);
    // --spec-n-rs-seq overrides, 0 = checkpoint restore on every partial acceptance
    argv = {"binary_name", "-m", "model_file.gguf", "--spec-type", "ngram-cache", "--spec-ngram-cache-n-max", "7"};
    common_params ngram_rs_params;
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), ngram_rs_params, LLAMA_EXAMPLE_SPECULATIVE));
    assert(ngram_rs_params.speculative.need_n_rs_seq() == 7);
    ngram_rs_params.speculative.ngram_cache.n_max = 0;
    assert(ngram_rs_params.speculative.need_n_rs_seq() == 8);   // the drafter's built-in default width
    ngram_rs_params.speculative.ngram_cache.n_max = 32;
    assert(ngram_rs_params.speculative.need_n_rs_seq() == 8);   // capped
    ngram_rs_params.n_batch = 4;
    ngram_rs_params.n_ubatch = 4;
    const llama_context_params ngram_rs_ctx_params = common_context_params_to_llama(ngram_rs_params);
    assert(ngram_rs_ctx_params.n_rs_seq == 8);
    assert(ngram_rs_ctx_params.n_batch == 10);
    assert(ngram_rs_ctx_params.n_ubatch == 10);

    argv = {"binary_name", "-m", "model_file.gguf", "--spec-type", "ngram-cache", "--spec-n-rs-seq", "0"};
    common_params ngram_rs0_params;
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), ngram_rs0_params, LLAMA_EXAMPLE_SPECULATIVE));
    assert(ngram_rs0_params.speculative.need_n_rs_seq() == 0);

    argv = {"binary_name", "-m", "model_file.gguf", "--spec-type", "draft-mtp", "--spec-n-rs-seq", "-2"};
    common_params ngram_rs_bad_params;
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), ngram_rs_bad_params, LLAMA_EXAMPLE_SPECULATIVE));

    // the adaptive MTP type parses to the dedicated enum value
    argv = {"binary_name", "-m", "model_file.gguf", "--spec-type", "draft-mtp-adaptive"};
    common_params spec_params;
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), spec_params, LLAMA_EXAMPLE_SPECULATIVE));
    assert(std::find(spec_params.speculative.types.begin(), spec_params.speculative.types.end(),
                     COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE) != spec_params.speculative.types.end());

    argv = {"binary_name", "-m", "model_file.gguf", "--spec-type", "draft-mtp-adaptive", "--spec-draft-n-max", "2"};
    common_params invalid_adaptive_params;
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), invalid_adaptive_params, LLAMA_EXAMPLE_SPECULATIVE));

    // per-family speculative defaults (spec-defaults.h): qwen35 with an MTP head gets the adaptive MTP drafter
    // with the production settings unless --spec-type was given; explicit --spec-draft-* values are kept
    {
        const std::vector<enum common_speculative_type> types_none     = { COMMON_SPECULATIVE_TYPE_NONE };
        const std::vector<enum common_speculative_type> types_adaptive = { COMMON_SPECULATIVE_TYPE_NONE, COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE };

        assert(common_speculative_family_default_find("qwen35", 1) != nullptr);
        assert(common_speculative_family_default_find("qwen35", 0) == nullptr);
        assert(common_speculative_family_default_find("llama",  1) == nullptr);
        assert(!common_speculative_family_defaults_str().empty());

        // qwen35 + nextn=1 -> the production settings, the same parameters as the explicit flags
        argv = {"binary_name", "-m", "model_file.gguf"};
        common_params auto_params;
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), auto_params, LLAMA_EXAMPLE_SERVER));
        assert(auto_params.speculative.user_set == 0);
        assert(common_speculative_apply_family_default(auto_params.speculative, "qwen35", 1) != nullptr);
        assert(auto_params.speculative.types == types_adaptive);
        assert(auto_params.speculative.draft.n_max == 4);
        assert(auto_params.speculative.draft.n_min_adaptive == 3);
        assert(auto_params.speculative.draft.p_min == 0.0f);
        assert(auto_params.speculative.draft.vocab_map == "auto");
        assert(auto_params.speculative.draft.cache_type_k == GGML_TYPE_COUNT);
        assert(auto_params.speculative.draft.cache_type_v == GGML_TYPE_COUNT);

        argv = {"binary_name", "-m", "model_file.gguf", "--spec-type", "draft-mtp-adaptive", "--spec-draft-n-max", "4",
                "--spec-draft-n-min-adaptive", "3", "--spec-draft-p-min", "0"};
        common_params explicit_params;
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), explicit_params, LLAMA_EXAMPLE_SERVER));
        assert(common_speculative_apply_family_default(explicit_params.speculative, "qwen35", 1) == nullptr);
        assert(explicit_params.speculative.types                == auto_params.speculative.types);
        assert(explicit_params.speculative.draft.n_max          == auto_params.speculative.draft.n_max);
        assert(explicit_params.speculative.draft.n_min          == auto_params.speculative.draft.n_min);
        assert(explicit_params.speculative.draft.n_min_adaptive == auto_params.speculative.draft.n_min_adaptive);
        assert(explicit_params.speculative.draft.p_min          == auto_params.speculative.draft.p_min);
        assert(explicit_params.speculative.draft.vocab_map      == auto_params.speculative.draft.vocab_map);
        assert(explicit_params.speculative.draft.chain          == auto_params.speculative.draft.chain);
        assert(explicit_params.speculative.need_n_rs_seq()      == auto_params.speculative.need_n_rs_seq());

        // qwen35 without an MTP head, and other architectures -> none
        argv = {"binary_name", "-m", "model_file.gguf"};
        common_params no_head_params;
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), no_head_params, LLAMA_EXAMPLE_SERVER));
        assert(common_speculative_apply_family_default(no_head_params.speculative, "qwen35", 0) == nullptr);
        assert(no_head_params.speculative.types == types_none);
        assert(common_speculative_apply_family_default(no_head_params.speculative, "llama", 1) == nullptr);
        assert(no_head_params.speculative.types == types_none);
        assert(no_head_params.speculative.draft.n_max == 3);

        // an explicit --spec-type none turns the default off
        argv = {"binary_name", "-m", "model_file.gguf", "--spec-type", "none"};
        common_params none_params;
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), none_params, LLAMA_EXAMPLE_SERVER));
        assert(none_params.speculative.user_set & COMMON_PARAMS_SPECULATIVE_USER_TYPE);
        assert(common_speculative_apply_family_default(none_params.speculative, "qwen35", 1) == nullptr);
        assert(std::find(none_params.speculative.types.begin(), none_params.speculative.types.end(),
                         COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE) == none_params.speculative.types.end());

        // an explicit --spec-draft-n-max without --spec-type -> the auto type with that n-max (the floor follows it)
        argv = {"binary_name", "-m", "model_file.gguf", "--spec-draft-n-max", "2"};
        common_params n_max_params;
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), n_max_params, LLAMA_EXAMPLE_SERVER));
        assert(n_max_params.speculative.user_set == COMMON_PARAMS_SPECULATIVE_USER_DRAFT_N_MAX);
        assert(common_speculative_apply_family_default(n_max_params.speculative, "qwen35", 1) != nullptr);
        assert(n_max_params.speculative.types == types_adaptive);
        assert(n_max_params.speculative.draft.n_max == 2);
        assert(n_max_params.speculative.draft.n_min_adaptive == 2);

        // other explicit draft values are kept on top of the default
        argv = {"binary_name", "-m", "model_file.gguf", "--spec-draft-p-min", "0.5", "--spec-draft-vocab-map", "none",
                "--spec-draft-n-min-adaptive", "5"};
        common_params kept_params;
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), kept_params, LLAMA_EXAMPLE_SERVER));
        assert(common_speculative_apply_family_default(kept_params.speculative, "qwen35", 1) != nullptr);
        assert(kept_params.speculative.draft.p_min == 0.5f);
        assert(kept_params.speculative.draft.vocab_map == "none");
        assert(kept_params.speculative.draft.n_min_adaptive == 5);
        assert(kept_params.speculative.draft.n_max == 5);

        // an explicit --spec-draft-n-max 0 asks for no drafting
        argv = {"binary_name", "-m", "model_file.gguf", "--spec-draft-n-max", "0"};
        common_params n_max0_params;
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), n_max0_params, LLAMA_EXAMPLE_SERVER));
        assert(common_speculative_apply_family_default(n_max0_params.speculative, "qwen35", 1) == nullptr);
        assert(n_max0_params.speculative.types == types_none);

        // a separate draft model turns the default off
        argv = {"binary_name", "-m", "model_file.gguf", "-md", "draft_file.gguf"};
        common_params draft_model_params;
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), draft_model_params, LLAMA_EXAMPLE_SERVER));
        assert(common_speculative_apply_family_default(draft_model_params.speculative, "qwen35", 1) == nullptr);
        assert(draft_model_params.speculative.types == types_none);
    }

    {
        common_params synth_params;
        argv = {"binary_name", "--spec-synth-len", "3.4"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), synth_params, LLAMA_EXAMPLE_SERVER));
        assert(synth_params.speculative.synth_len == 3.4);
    }

    {
        common_params synth_params;
        argv = {"binary_name", "--spec-synth-rates", "0.8,0.6,0.2"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), synth_params, LLAMA_EXAMPLE_SERVER));
        assert(synth_params.speculative.synth_rates == std::vector<double>({0.8, 0.6, 0.2}));
    }

    {
        common_params synth_params;
        argv = {"binary_name", "--spec-synth-len", "3.4x"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), synth_params, LLAMA_EXAMPLE_SERVER));
    }

    argv = {"binary_name", "-lm", "none"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_NONE);

    argv = {"binary_name", "-lm", "mmap"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP);

    argv = {"binary_name", "-lm", "mlock"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MLOCK);

    argv = {"binary_name", "-lm", "mmap+mlock"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK);

    argv = {"binary_name", "-lm", "dio"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_DIRECT_IO);

    // multi-value args (CSV)
    argv = {"binary_name", "--lora", "file1.gguf,\"file2,2.gguf\",\"file3\"\"3\"\".gguf\",file4\".gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.lora_adapters.size() == 4);
    assert(params.lora_adapters[0].path == "file1.gguf");
    assert(params.lora_adapters[1].path == "file2,2.gguf");
    assert(params.lora_adapters[2].path == "file3\"3\".gguf");
    assert(params.lora_adapters[3].path == "file4\".gguf");

    printf("test-arg-parser: test MoE cache and repack modes\n\n");

    {
        common_params mode_params;
        assert(mode_params.moe_cache.mode == COMMON_MOE_CACHE_MODE_AUTO);
        assert(mode_params.moe_cache.mode_explicit == false);
        assert(common_context_params_to_llama(mode_params).moe_cache_mode == LLAMA_MOE_CACHE_MODE_UNSPECIFIED);
    }

    const std::vector<std::vector<std::string>> invalid_moe_cache_args = {
        {"binary_name", "--moe-cache"},
        {"binary_name", "--moe-cache", "invalid"},
        {"binary_name", "--moe-cache", "-1"},
        {"binary_name", "--moe-cache", "1048577"},
        {"binary_name", "--moe-cache", "256x"},
        {"binary_name", "--moe-cache", "999999999999999999999999"},
    };
    for (auto invalid_argv : invalid_moe_cache_args) {
        common_params mode_params;
        assert(false == common_params_parse(
                invalid_argv.size(), list_str_to_char(invalid_argv).data(),
                mode_params, LLAMA_EXAMPLE_COMMON));
    }

    {
        common_params mode_params;
        argv = {"binary_name", "-m", "model.gguf", "--moe-cache", "auto"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), mode_params, LLAMA_EXAMPLE_COMMON));
        assert(mode_params.moe_cache.mode == COMMON_MOE_CACHE_MODE_AUTO);
        assert(mode_params.moe_cache.budget_mib == 0);
        assert(mode_params.moe_cache.mode_explicit == true);
        assert(mode_params.no_extra_bufts == false);
        const llama_context_params cparams = common_context_params_to_llama(mode_params);
        assert(cparams.moe_cache_mode == LLAMA_MOE_CACHE_MODE_AUTO);
        assert(cparams.moe_cache_budget_mib == 0);
    }

    {
        common_params mode_params;
        argv = {"binary_name", "-m", "model.gguf", "--moe-cache", "on", "--repack"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), mode_params, LLAMA_EXAMPLE_COMMON));
        assert(mode_params.moe_cache.mode == COMMON_MOE_CACHE_MODE_ON);
        assert(mode_params.moe_cache.budget_mib == 0);
        assert(mode_params.no_extra_bufts == true);
        assert(common_context_params_to_llama(mode_params).moe_cache_mode == LLAMA_MOE_CACHE_MODE_ON);
    }

    {
        common_params mode_params;
        argv = {"binary_name", "-m", "model.gguf", "--repack", "--moe-cache", "256"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), mode_params, LLAMA_EXAMPLE_COMMON));
        assert(mode_params.moe_cache.mode == COMMON_MOE_CACHE_MODE_ON);
        assert(mode_params.moe_cache.budget_mib == 256);
        assert(mode_params.no_extra_bufts == true);
        const llama_context_params cparams = common_context_params_to_llama(mode_params);
        assert(cparams.moe_cache_mode == LLAMA_MOE_CACHE_MODE_ON);
        assert(cparams.moe_cache_budget_mib == 256);
    }

    {
        common_params mode_params;
        argv = {"binary_name", "-m", "model.gguf", "--moe-cache", "1048576"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), mode_params, LLAMA_EXAMPLE_COMMON));
        assert(mode_params.moe_cache.mode == COMMON_MOE_CACHE_MODE_ON);
        assert(mode_params.moe_cache.budget_mib == 1048576);
        assert(mode_params.no_extra_bufts == true);
        assert(common_context_params_to_llama(mode_params).moe_cache_budget_mib == 1048576);
    }

    {
        common_params mode_params;
        argv = {"binary_name", "-m", "model.gguf", "--moe-cache", "256", "--moe-cache", "auto"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), mode_params, LLAMA_EXAMPLE_COMMON));
        assert(mode_params.moe_cache.mode == COMMON_MOE_CACHE_MODE_AUTO);
        assert(mode_params.moe_cache.budget_mib == 0);
        assert(mode_params.no_extra_bufts == false);
        assert(common_context_params_to_llama(mode_params).moe_cache_mode == LLAMA_MOE_CACHE_MODE_AUTO);
    }

    {
        common_params mode_params;
        argv = {"binary_name", "-m", "model.gguf", "--moe-cache", "off", "--repack"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), mode_params, LLAMA_EXAMPLE_COMMON));
        assert(mode_params.moe_cache.mode == COMMON_MOE_CACHE_MODE_OFF);
        assert(mode_params.moe_cache.budget_mib == 0);
        assert(mode_params.no_extra_bufts == false);
        assert(common_context_params_to_llama(mode_params).moe_cache_mode == LLAMA_MOE_CACHE_MODE_OFF);
    }

    {
        common_params mode_params;
        argv = {"binary_name", "-m", "model.gguf", "--moe-cache", "0"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), mode_params, LLAMA_EXAMPLE_COMMON));
        assert(mode_params.moe_cache.mode == COMMON_MOE_CACHE_MODE_OFF);
        assert(mode_params.moe_cache.budget_mib == 0);
        assert(mode_params.no_extra_bufts == false);
        assert(common_context_params_to_llama(mode_params).moe_cache_mode == LLAMA_MOE_CACHE_MODE_OFF);
    }

// skip this part on windows, because setenv is not supported
#ifdef _WIN32
    printf("test-arg-parser: skip on windows build\n");
#else
    printf("test-arg-parser: test environment variables (valid + invalid usages)\n\n");

    for (const char * value : { "0", "1" }) {
        setenv("LLAMA_ARG_SPEC_DRAFT_SJKVARN", value, true);
        setenv("LLAMA_ARG_SPEC_DRAFT_VOCAB_MAP", "auto:24576", true);
        setenv("LLAMA_ARG_SPEC_DRAFT_VOCAB_HOT", "2048", true);
        common_params from_env;
        argv = {"binary_name"};
        assert(common_params_parse(argv.size(), list_str_to_char(argv).data(), from_env, LLAMA_EXAMPLE_SERVER));
        assert(from_env.speculative.draft.sj_kvarn == (value[0] == '1'));
        assert(from_env.speculative.draft.sj_kvarn_explicit);
        assert(from_env.speculative.draft.vocab_map == "auto:24576" && from_env.speculative.draft.vocab_hot == 2048);
        assert(from_env.speculative.user_set & COMMON_PARAMS_SPECULATIVE_USER_DRAFT_VOCAB_MAP);
        assert(from_env.speculative.user_set & COMMON_PARAMS_SPECULATIVE_USER_DRAFT_VOCAB_HOT);
        common_params from_cli;
        argv = {"binary_name", value[0] == '1' ? "--no-spec-draft-sjkvarn" : "--spec-draft-sjkvarn",
                "--spec-draft-vocab-map", "auto", "--spec-draft-vocab-hot", "0"};
        assert(common_params_parse(argv.size(), list_str_to_char(argv).data(), from_cli, LLAMA_EXAMPLE_SERVER));
        assert(from_cli.speculative.draft.sj_kvarn == (value[0] == '0'));
        assert(from_cli.speculative.draft.sj_kvarn_explicit);
        assert(from_cli.speculative.draft.vocab_map == "auto" && from_cli.speculative.draft.vocab_hot == 0);
    }
    unsetenv("LLAMA_ARG_SPEC_DRAFT_SJKVARN");
    unsetenv("LLAMA_ARG_SPEC_DRAFT_VOCAB_MAP");
    unsetenv("LLAMA_ARG_SPEC_DRAFT_VOCAB_HOT");
    setenv("LLAMA_ARG_SPEC_DRAFT_SJKVARN", "1", true);
    setenv("LLAMA_ARG_NO_SPEC_DRAFT_SJKVARN", "1", true);
    {
        common_params negated;
        argv = {"binary_name"};
        assert(common_params_parse(argv.size(), list_str_to_char(argv).data(), negated, LLAMA_EXAMPLE_SERVER));
        assert(negated.speculative.draft.sj_kvarn_explicit && !negated.speculative.draft.sj_kvarn);
        argv = {"binary_name", "--spec-draft-sjkvarn"};
        assert(common_params_parse(argv.size(), list_str_to_char(argv).data(), negated, LLAMA_EXAMPLE_SERVER));
        assert(negated.speculative.draft.sj_kvarn_explicit && negated.speculative.draft.sj_kvarn);
    }
    unsetenv("LLAMA_ARG_SPEC_DRAFT_SJKVARN");
    unsetenv("LLAMA_ARG_NO_SPEC_DRAFT_SJKVARN");

    setenv("LLAMA_ARG_THREADS", "blah", true);
    argv = {"binary_name"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "blah.gguf");
    assert(params.cpuparams.n_threads == 1010);

    setenv("LLAMA_ARG_LOAD_MODE", "blah", true);
    argv = {"binary_name"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    setenv("LLAMA_ARG_LOAD_MODE", "mmap", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP);

    setenv("LLAMA_ARG_LOAD_MODE", "mlock", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MLOCK);

    unsetenv("LLAMA_ARG_LOAD_MODE");
    setenv("LLAMA_ARG_MOE_CACHE", "invalid", true);
    argv = {"binary_name"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    setenv("LLAMA_ARG_MOE_CACHE", "on", true);
    {
        common_params mode_params;
        argv = {"binary_name"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), mode_params, LLAMA_EXAMPLE_COMMON));
        assert(mode_params.moe_cache.mode == COMMON_MOE_CACHE_MODE_ON);
        assert(mode_params.moe_cache.mode_explicit == true);
        assert(mode_params.no_extra_bufts == true);
    }

    {
        common_params mode_params;
        argv = {"binary_name", "--moe-cache", "off"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), mode_params, LLAMA_EXAMPLE_COMMON));
        assert(mode_params.moe_cache.mode == COMMON_MOE_CACHE_MODE_OFF);
        assert(mode_params.moe_cache.mode_explicit == true);
        assert(mode_params.no_extra_bufts == false);
        assert(common_context_params_to_llama(mode_params).moe_cache_mode == LLAMA_MOE_CACHE_MODE_OFF);
    }
    unsetenv("LLAMA_ARG_MOE_CACHE");

    setenv("LLAMA_ARG_LOAD_MODE", "mmap+mlock", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK);

    setenv("LLAMA_ARG_LOAD_MODE", "dio", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_DIRECT_IO);

    printf("test-arg-parser: test negated environment variables\n\n");

    setenv("LLAMA_ARG_LOAD_MODE", "none", true);
    setenv("LLAMA_ARG_NO_PERF", "1", true); // legacy format
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_NONE);
    assert(params.no_perf == true);

    printf("test-arg-parser: test environment variables being overwritten\n\n");

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name", "-m", "overwritten.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "overwritten.gguf");
    assert(params.cpuparams.n_threads == 1010);
#endif // _WIN32

    {
        printf("test-arg-parser: test --kv-stream-arena-mib\n\n");
        common_params kv_stream_params;
        assert(kv_stream_params.kv_stream_arena_mib == 0);

        argv = {"binary_name", "-m", "model.gguf", "--kv-stream-arena-mib", "4096"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), kv_stream_params, LLAMA_EXAMPLE_COMMON));
        assert(kv_stream_params.kv_stream_arena_mib == 4096);

        // legacy alias must set the same field
        common_params kv_stream_alias_params;
        argv = {"binary_name", "-m", "model.gguf", "--kv-stream-stage-mib", "2048"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), kv_stream_alias_params, LLAMA_EXAMPLE_COMMON));
        assert(kv_stream_alias_params.kv_stream_arena_mib == 2048);

        // negative values are rejected
        argv = {"binary_name", "--kv-stream-arena-mib", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), kv_stream_params, LLAMA_EXAMPLE_COMMON));
        argv = {"binary_name", "--kv-stream-stage-mib", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), kv_stream_params, LLAMA_EXAMPLE_COMMON));

        // 0 explicitly disables it and is not rejected
        common_params kv_stream_disabled_params;
        argv = {"binary_name", "-m", "model.gguf", "--kv-stream-arena-mib", "0"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), kv_stream_disabled_params, LLAMA_EXAMPLE_COMMON));
        assert(kv_stream_disabled_params.kv_stream_arena_mib == 0);
    }

    printf("test-arg-parser: test download functions\n\n");
    const char * GOOD_URL = "http://ggml.ai/";
    const char * BAD_URL  = "http://ggml.ai/404";

    {
        printf("test-arg-parser: test good URL\n\n");
        auto res = common_remote_get_content(GOOD_URL, {});
        assert(res.first == 200);
        assert(res.second.size() > 0);
        std::string str(res.second.data(), res.second.size());
        assert(str.find("llama.cpp") != std::string::npos);
    }

    {
        printf("test-arg-parser: test bad URL\n\n");
        auto res = common_remote_get_content(BAD_URL, {});
        assert(res.first == 404);
    }

    {
        printf("test-arg-parser: test max size error\n");
        common_remote_params params;
        params.max_size = 1;
        try {
            common_remote_get_content(GOOD_URL, params);
            assert(false && "it should throw an error");
        } catch (std::exception & e) {
            printf("  expected error: %s\n\n", e.what());
        }
    }

    printf("test-arg-parser: all tests OK\n\n");
}

int main(void) {
    try {
        test();
    } catch (std::exception & e) {
        fprintf(stderr, "test-arg-parser: exception: %s\n", e.what());
        return 1;
    }
    return 0;
}
