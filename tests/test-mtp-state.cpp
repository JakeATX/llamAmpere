// Exercise real MTP cache restoration with a generated CPU model, split prefill, deferred rows, and adaptive draft widths.
// Restore into a fresh driver and a different destination sequence.
// --save-model PATH exports the fixture for other state tests.

#include "common.h"
#include "log.h"
#include "speculative.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

#include "../src/llama-arch.h"
#include "../src/llama-model.h"
#include "../src/llama-model-saver.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr uint32_t n_vocab = 32, n_embd = 256;

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static void tensor_data(ggml_tensor * tensor, void *) {
    std::mt19937 rng(std::hash<std::string>()(tensor->name));
    std::normal_distribution<float> dist(0.0f, 0.1f);
    require(tensor->type == GGML_TYPE_F32, "fixture tensor type");
    std::vector<float> data(ggml_nelements(tensor));
    for (auto & value : data) {
        value = dist(rng);
    }
    ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
}

static llama_model * make_model() {
    gguf_context * meta = gguf_init_empty();
    llama_model_saver ms(LLM_ARCH_QWEN35, meta);
    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE, llm_arch_name(LLM_ARCH_QWEN35));
    ms.add_kv(LLM_KV_VOCAB_SIZE, n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH, uint32_t(2048));
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH, n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT, uint32_t(3));
    ms.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS, uint32_t(1));
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, uint32_t(128));
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, uint32_t(1));
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, uint32_t(1));
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({64, 64, 64, 64}));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(2));
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL, uint32_t(4));
    ms.add_kv(LLM_KV_SSM_INNER_SIZE, uint32_t(128));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE, uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK, uint32_t(1));
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT, uint32_t(1));
    ms.add_kv(LLM_KV_TOKENIZER_MODEL, "test");
    std::vector<std::string> tokens;
    for (uint32_t i = 0; i < n_vocab; ++i) {
        tokens.push_back("tok_" + std::to_string(i));
    }
    ms.add_kv(LLM_KV_TOKENIZER_LIST, tokens);
    ms.add_kv(LLM_KV_TOKENIZER_SCORES, std::vector<float>(n_vocab, 0.0f));
    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t devices[] = { nullptr };
    mp.devices = devices;
    mp.load_mtp = true;
    llama_model * model = llama_model_init_from_user(meta, tensor_data, nullptr, mp);
    gguf_free(meta);
    require(model != nullptr, "create fixture model");
    return model;
}

struct cache_config {
    const char * name;
    ggml_type k, v;
    uint32_t sj_k = 0, sj_v = 0;
    ggml_type body = GGML_TYPE_F32;
};

struct contexts {
    llama_context * tgt = nullptr;
    llama_context * dft = nullptr;
    common_speculative * spec = nullptr;

    contexts(llama_model * model, const cache_config & config, uint32_t batch_size = 128,
            common_speculative_type type = COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, uint32_t ubatch_size = 0) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 2048;
        cp.n_batch = batch_size;
        cp.n_ubatch = ubatch_size ? ubatch_size : batch_size;
        cp.n_seq_max = 2;
        cp.kv_unified = false;
        cp.n_rs_seq = 4;
        cp.n_threads = cp.n_threads_batch = 2;
        cp.offload_kqv = false;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.type_k = config.k;
        cp.type_v = config.v;
        if (config.sj_k) {
            cp.sj_kvarn_bits_k = config.sj_k;
            cp.sj_kvarn_bits_v = config.sj_v;
            cp.sj_kvarn_body_type = config.body;
            cp.sj_kvarn_staging_type = GGML_TYPE_TQ6_0;
            cp.sj_kvarn_sink_type = GGML_TYPE_F16;
            cp.sj_kvarn_sink = 128;
            cp.sj_kvarn_tail = 128;
            cp.sj_kvarn_tail_max = 256;
        }
        tgt = llama_init_from_model(model, cp);
        require(tgt != nullptr, "create target context");
        cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        cp.ctx_other = tgt;
        cp.n_rs_seq = 0;
        cp.sj_kvarn_bits_k = cp.sj_kvarn_bits_v = 0;
        if (config.sj_k) {
            cp.type_k = COMMON_SJKVARN_MTP_DRAFT_KV[0];
            cp.type_v = COMMON_SJKVARN_MTP_DRAFT_KV[1];
        }
        dft = llama_init_from_model(model, cp);
        require(dft != nullptr, "create draft context");
        common_params_speculative params;
        params.types = { type };
        params.draft.ctx_tgt = tgt;
        params.draft.ctx_dft = dft;
        params.draft.n_max = 4;
        params.draft.n_min_adaptive = 3;
        params.draft.p_min = 0.0f;
        params.draft.chain = true;
        spec = common_speculative_init(params, 2);
        require(spec != nullptr, "create MTP driver");
    }

    ~contexts() {
        common_speculative_free(spec);
        llama_free(dft);
        llama_free(tgt);
    }
};

static llama_token token_at(llama_pos pos) {
    return (pos * 7 + 3) % n_vocab;
}

static void process(contexts & ctx, const llama_tokens & tokens, llama_pos pos, llama_seq_id seq) {
    common_batch batch(ctx.tgt);
    for (llama_token token : tokens) {
        batch.add(token, pos++, seq, true);
    }
    require(llama_process(ctx.tgt, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "target decode");
    require(common_speculative_process(ctx.spec, batch), "MTP process");
}

static void prefill(contexts & ctx, llama_pos from, llama_pos to, llama_seq_id seq) {
    while (from < to) {
        llama_tokens tokens;
        const llama_pos end = std::min(from + (llama_pos) llama_n_batch(ctx.tgt), to);
        for (llama_pos pos = from; pos < end; ++pos) {
            tokens.push_back(token_at(pos));
        }
        process(ctx, tokens, from, seq);
        from = end;
    }
}

struct snapshot {
    std::vector<uint8_t> tgt, dft, spec;
};

static std::vector<uint8_t> save(llama_context * ctx, llama_seq_id seq) {
    const auto flags = LLAMA_STATE_SEQ_FLAGS_KEEP_LAYOUT;
    std::vector<uint8_t> state(llama_state_seq_get_size_ext(ctx, seq, flags));
    require(!state.empty(), "sequence state exists");
    require(llama_state_seq_get_data_ext(ctx, state.data(), state.size(), seq, flags) == state.size(), "save sequence state");
    return state;
}

static snapshot save(contexts & ctx, llama_seq_id seq) {
    snapshot state { save(ctx.tgt, seq), save(ctx.dft, seq), {} };
    require(common_speculative_get_state(ctx.spec, seq, state.spec) && !state.spec.empty(), "MTP snapshot contains carry and adaptive state");
    return state;
}

static void check_rejected_state(contexts & ctx, llama_seq_id seq, const std::vector<uint8_t> & valid) {
    for (int test = 0; test < 3; ++test) {
        auto invalid = valid;
        if (test == 0) {
            invalid.clear();
        } else if (test == 1) {
            invalid.pop_back();
        } else {
            invalid.front() ^= 0xff;
        }
        require(!common_speculative_restore_state(ctx.spec, seq, invalid), "invalid MTP state is rejected");
        std::vector<uint8_t> after;
        require(common_speculative_get_state(ctx.spec, seq, after) && after == valid, "invalid MTP state leaves sequence unchanged");
    }
}

static void restore(contexts & ctx, const snapshot & state, llama_seq_id seq) {
    const auto flags = LLAMA_STATE_SEQ_FLAGS_KEEP_LAYOUT;
    require(llama_state_seq_set_data_ext(ctx.tgt, state.tgt.data(), state.tgt.size(), seq, flags) == state.tgt.size(), "restore target state");
    require(llama_state_seq_set_data_ext(ctx.dft, state.dft.data(), state.dft.size(), seq, flags) == state.dft.size(), "restore draft state");
    require(common_speculative_restore_state(ctx.spec, seq, state.spec), "restore MTP host state");
    std::vector<uint8_t> round_trip;
    require(common_speculative_get_state(ctx.spec, seq, round_trip) && round_trip == state.spec, "MTP snapshot round trip across sequences");
    check_rejected_state(ctx, seq, round_trip);
}

struct schedule {
    std::vector<size_t> widths;
    std::vector<llama_tokens> tokens;

    bool operator==(const schedule & other) const {
        return widths == other.widths && tokens == other.tokens;
    }
};

static void require_schedule(const schedule & actual, const schedule & expected, const char * message) {
    if (actual == expected) {
        return;
    }
    fprintf(stderr, "%s: reference widths", message);
    for (size_t width : expected.widths) {
        fprintf(stderr, " %zu", width);
    }
    fprintf(stderr, "; restored widths");
    for (size_t width : actual.widths) {
        fprintf(stderr, " %zu", width);
    }
    fprintf(stderr, "\n");
    for (size_t round = 0; round < std::min(actual.tokens.size(), expected.tokens.size()); ++round) {
        const auto & got = actual.tokens[round];
        const auto & ref = expected.tokens[round];
        if (got != ref) {
            fprintf(stderr, "first differing draft round %zu: reference", round);
            for (llama_token token : ref) {
                fprintf(stderr, " %d", token);
            }
            fprintf(stderr, "; restored");
            for (llama_token token : got) {
                fprintf(stderr, " %d", token);
            }
            fprintf(stderr, "\n");
            break;
        }
    }
    require(false, message);
}

static schedule generate(contexts & ctx, llama_seq_id seq, llama_pos & pos, int first, int count) {
    schedule result;
    common_params_sampling sampling;
    sampling.temp = 0.0f;
    for (int round = first; round < first + count; ++round) {
        llama_tokens prompt, draft;
        auto & dp = common_speculative_get_draft_params(ctx.spec, seq);
        dp = {};
        dp.drafting = true;
        dp.pos0 = pos;
        dp.id_last = token_at(pos);
        dp.prompt = &prompt;
        dp.result = &draft;
        dp.sampling = &sampling;
        common_speculative_draft(ctx.spec);
        require(!draft.empty(), "MTP drafted tokens");
        result.widths.push_back(draft.size());
        result.tokens.push_back(draft);
        llama_tokens verify { dp.id_last };
        verify.insert(verify.end(), draft.begin(), draft.end());
        process(ctx, verify, pos, seq);
        const int accepted = round % 7 == 6 ? 1 : (int) draft.size();
        common_speculative_accept(ctx.spec, seq, accepted);
        pos += accepted + 1;
        require(llama_memory_seq_rm(llama_get_memory(ctx.tgt), seq, pos, -1), "target rollback");
        require(llama_memory_seq_rm(llama_get_memory(ctx.dft), seq, pos, -1), "draft rollback");
    }
    return result;
}

static void run_prefill_identity(llama_model * model, const cache_config & config) {
    struct env_restore {
        std::string saved = common_get_env("GGML_MTP_PREFILL_OVERLAP");
        ~env_restore() { common_set_env("GGML_MTP_PREFILL_OVERLAP", saved); }
    } env;

    common_set_env("GGML_MTP_PREFILL_OVERLAP", "");
    contexts reference(model, config, 128, COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, 32);
    common_set_env("GGML_MTP_PREFILL_OVERLAP", "1");
    contexts coalesced(model, config, 128, COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, 32);

    auto check_state = [&](llama_seq_id seq) {
        const snapshot a = save(reference, seq);
        const snapshot b = save(coalesced, seq);
        require(a.tgt == b.tgt, "overlap on/off target state is bit-identical");
        require(a.dft == b.dft, "overlap on/off drafter KV is bit-identical");
        require(a.spec == b.spec, "overlap on/off carry and deferred state is bit-identical");
    };
    auto process_pair = [&](common_batch & batch) {
        for (contexts * ctx : { &reference, &coalesced }) {
            common_batch input(ctx->tgt);
            input.tokens = batch.tokens;
            require(llama_process(ctx->tgt, LLAMA_PROCESS_TYPE_DECODE, input.get()) == 0, "identity target decode");
            require(common_speculative_process(ctx->spec, input), "identity MTP process");
        }
        if (std::any_of(batch.tokens.begin(), batch.tokens.end(), [](const common_batch::token & t) { return t.output; })) {
            require(std::memcmp(llama_get_logits_ith(reference.tgt, -1), llama_get_logits_ith(coalesced.tgt, -1),
                        n_vocab * sizeof(float)) == 0, "overlap on/off target logits are bit-identical");
        }
    };

    llama_pos pos[2] = {};
    // Cross target/draft ubatches and the deferred-row limit, including one-token prompt tails.
    for (int width : { 128, 67, 3, 1, 127 }) {
        for (llama_seq_id seq = 0; seq < 2; ++seq) {
            common_batch batch(reference.tgt);
            for (int i = 0; i < width; ++i) {
                batch.add(token_at(pos[seq]), pos[seq], seq, width != 128 && i == width - 1);
                ++pos[seq];
            }
            process_pair(batch);
            check_state(0);
            check_state(1);
        }
    }

    // Unequal sequence spans exercise dense-row reordering with sparse logits.
    common_batch batch(reference.tgt);
    for (llama_seq_id seq : { 1, 0 }) {
        const int width = seq == 1 ? 17 : 80;
        for (int i = 0; i < width; ++i) {
            batch.add(token_at(pos[seq]), pos[seq], seq, i == width - 1);
            ++pos[seq];
        }
    }
    process_pair(batch);
    check_state(0);
    check_state(1);

    const snapshot cached = save(coalesced, 0);
    common_set_env("GGML_MTP_PREFILL_OVERLAP", "0");
    contexts resumed(model, config, 128, COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, 32);
    restore(resumed, cached, 1);
    common_speculative_begin(resumed.spec, 1, llama_tokens(pos[0], 0));
    llama_pos pos_restored = pos[0];
    const schedule restored = generate(resumed, 1, pos_restored, 0, 8);
    common_set_env("GGML_MTP_PREFILL_OVERLAP", "1");
    contexts resumed_on(model, config, 128, COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, 32);
    restore(resumed_on, save(reference, 0), 1);
    common_speculative_begin(resumed_on.spec, 1, llama_tokens(pos[0], 0));
    pos_restored = pos[0];
    require_schedule(generate(resumed_on, 1, pos_restored, 0, 8), restored, "off prompt snapshot restores with switch on into another sequence");
    for (llama_seq_id seq = 0; seq < 2; ++seq) {
        common_speculative_begin(reference.spec, seq, llama_tokens(pos[seq], 0));
        common_speculative_begin(coalesced.spec, seq, llama_tokens(pos[seq], 0));
        llama_pos pos_ref = pos[seq], pos_got = pos[seq];
        const schedule expected = generate(reference, seq, pos_ref, 0, 8);
        require_schedule(generate(coalesced, seq, pos_got, 0, 8), expected, "overlap on/off draft tokens and widths");
        if (seq == 0) {
            require_schedule(restored, expected, "overlap prompt snapshot restores with switch off into another sequence");
        }
        check_state(seq);
    }
    printf("PASS %s: overlap on/off bit-identical target, draft KV, carry, deferred rows, logits and 16 draft rounds; cache restore in both directions\n", config.name);
}

static void run_case(llama_model * model, const cache_config & config, bool deferred) {
    contexts reference(model, config);
    const llama_pos boundary = 512 + (deferred ? 3 : 0);
    const llama_pos prompt_end = boundary + (deferred ? 5 : 128);
    prefill(reference, 0, boundary, 0);
    const snapshot prefix = save(reference, 0);
    prefill(reference, boundary, prompt_end, 0);
    const snapshot prompt = save(reference, 0);
    common_speculative_begin(reference.spec, 0, llama_tokens(prompt_end, 0));
    llama_pos pos_ref = prompt_end;
    const schedule expected = generate(reference, 0, pos_ref, 0, 8);
    const snapshot middle = save(reference, 0);
    const llama_pos middle_pos = pos_ref;
    const schedule expected_tail = generate(reference, 0, pos_ref, 8, 8);
    require(std::find(expected.widths.begin(), expected.widths.end(), 3) != expected.widths.end() &&
            std::find(expected.widths.begin(), expected.widths.end(), 4) != expected.widths.end(), "schedule exercises adaptive climb");

    contexts resumed(model, config);
    std::vector<uint8_t> cold;
    require(common_speculative_get_state(resumed.spec, 0, cold), "save empty sequence state");
    prefill(resumed, 0, 259, 0);
    std::vector<uint8_t> other_before, other_after;
    require(common_speculative_get_state(resumed.spec, 0, other_before), "save other sequence carry");
    restore(resumed, prefix, 1);
    require(common_speculative_get_state(resumed.spec, 0, other_after) && other_before == other_after, "restore preserves another sequence carry and deferred rows");
    // Clear the other sequence after checking isolation, so its deferred rows cannot change the draft batch shape.
    require(llama_memory_seq_rm(llama_get_memory(resumed.tgt), 0, -1, -1), "clear other target sequence");
    require(llama_memory_seq_rm(llama_get_memory(resumed.dft), 0, -1, -1), "clear other draft sequence");
    require(common_speculative_restore_state(resumed.spec, 0, cold), "restore empty sequence state");
    prefill(resumed, boundary, prompt_end, 1);
    common_speculative_begin(resumed.spec, 1, llama_tokens(prompt_end, 0));
    llama_pos pos_got = prompt_end;
    require_schedule(generate(resumed, 1, pos_got, 0, 8), expected, "prefill restore draft tokens and round widths");

    contexts cached_prompt(model, config);
    restore(cached_prompt, prompt, 1);
    common_speculative_begin(cached_prompt.spec, 1, llama_tokens(prompt_end, 0));
    pos_got = prompt_end;
    require_schedule(generate(cached_prompt, 1, pos_got, 0, 8), expected, "complete prompt restore draft tokens and round widths");

    contexts continued(model, config);
    restore(continued, middle, 1);
    pos_got = middle_pos;
    require_schedule(generate(continued, 1, pos_got, 8, 8), expected_tail, "generation restore draft tokens and adaptive round widths");
    printf("PASS %s: %s, fresh driver and sequence 0 -> 1, 24 matching draft rounds\n", config.name, deferred ? "deferred carry" : "batch carry");
}

static void run_deferred_overflow(llama_model * model, const cache_config & config) {
    contexts reference(model, config, 64);
    prefill(reference, 0, 124, 0);
    require(llama_memory_seq_pos_max(llama_get_memory(reference.dft), 0) == 63, "sixty real rows remain deferred");
    const snapshot prefix = save(reference, 0);
    prefill(reference, 124, 188, 0);
    common_speculative_begin(reference.spec, 0, llama_tokens(188, 0));
    llama_pos pos = 188;
    const schedule expected = generate(reference, 0, pos, 0, 8);

    contexts resumed(model, config, 64);
    std::vector<uint8_t> cold;
    require(common_speculative_get_state(resumed.spec, 0, cold), "save empty overflow sequence state");
    restore(resumed, prefix, 0);
    restore(resumed, prefix, 1);
    // Each snapshot fits the decode capacity; together their 120 deferred rows need more than one flush batch.
    prefill(resumed, 124, 188, 0);
    require(llama_memory_seq_pos_max(llama_get_memory(resumed.dft), 0) == 187, "overflow flush retains first sequence rows");
    require(llama_memory_seq_pos_max(llama_get_memory(resumed.dft), 1) == 123, "overflow flush retains all other sequence rows");
    prefill(resumed, 124, 188, 1);
    common_speculative_begin(resumed.spec, 0, llama_tokens(188, 0));
    pos = 188;
    require_schedule(generate(resumed, 0, pos, 0, 8), expected, "overflow restore first sequence draft tokens and round widths");

    require(llama_memory_seq_rm(llama_get_memory(resumed.tgt), 0, -1, -1), "clear first overflow target sequence");
    require(llama_memory_seq_rm(llama_get_memory(resumed.dft), 0, -1, -1), "clear first overflow draft sequence");
    require(common_speculative_restore_state(resumed.spec, 0, cold), "restore empty overflow sequence state");
    common_speculative_begin(resumed.spec, 1, llama_tokens(188, 0));
    pos = 188;
    require_schedule(generate(resumed, 1, pos, 0, 8), expected, "overflow restore other sequence draft tokens and round widths");

    common_params_sampling sampling;
    sampling.temp = 0.0f;
    llama_tokens draft_prompt, expected_drafts[2], drafts[2];
    auto prepare = [&](contexts & ctx, llama_seq_id seq, llama_tokens & result) {
        auto & dp = common_speculative_get_draft_params(ctx.spec, seq);
        dp = {};
        dp.drafting = true;
        dp.pos0 = 124;
        dp.id_last = token_at(124);
        dp.prompt = &draft_prompt;
        dp.result = &result;
        dp.sampling = &sampling;
    };
    contexts fresh(model, config, 64);
    for (llama_seq_id seq = 0; seq < 2; ++seq) {
        prefill(fresh, 0, 124, seq);
        common_speculative_begin(fresh.spec, seq, llama_tokens(124, 0));
        prepare(fresh, seq, expected_drafts[seq]);
    }
    common_speculative_draft(fresh.spec);
    for (llama_seq_id seq = 0; seq < 2; ++seq) {
        require(expected_drafts[seq].size() == 3, "fresh simultaneous sequences draft at adaptive floor");
    }

    contexts simultaneous(model, config, 64);
    for (llama_seq_id seq = 0; seq < 2; ++seq) {
        restore(simultaneous, prefix, seq);
        common_speculative_begin(simultaneous.spec, seq, llama_tokens(124, 0));
        prepare(simultaneous, seq, drafts[seq]);
    }
    common_speculative_draft(simultaneous.spec);
    for (llama_seq_id seq = 0; seq < 2; ++seq) {
        require_schedule({ { drafts[seq].size() }, { drafts[seq] } },
                         { { expected_drafts[seq].size() }, { expected_drafts[seq] } },
                         "simultaneous restored deferred drafts match fresh tokens and width");
    }
    printf("PASS %s: two restored 60-row deferred states with a 64-row decode capacity, 16 matching draft rounds plus two simultaneous drafts\n", config.name);
}

static void run_eagle_boundary_restore(llama_model * model) {
    // Only exercise EAGLE's boundary serializer with the hybrid fixture; no EAGLE graph is decoded.
    model->target_layer_ids = { 0, 1, 2 };
    contexts ctx(model, { "EAGLE boundary", GGML_TYPE_Q8_0, GGML_TYPE_Q8_0 }, 128, COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3);
    std::vector<uint8_t> got;
    require(!common_speculative_get_state(ctx.spec, 1, got), "fresh EAGLE sequence has no boundary state");
    const llama_pos pos = 123;
    std::vector<float> carry(n_embd);
    for (uint32_t i = 0; i < n_embd; ++i) {
        carry[i] = (float) i / n_embd;
    }
    std::vector<uint8_t> state(sizeof(pos) + carry.size() * sizeof(float));
    std::memcpy(state.data(), &pos, sizeof(pos));
    std::memcpy(state.data() + sizeof(pos), carry.data(), carry.size() * sizeof(float));
    require(common_speculative_restore_state(ctx.spec, 1, state), "restore EAGLE boundary into a fresh driver");
    require(common_speculative_get_state(ctx.spec, 1, got) && got == state, "EAGLE boundary restore round trip");
    require(!common_speculative_get_state(ctx.spec, 0, got), "EAGLE boundary restore leaves another sequence cold");
    auto truncated = state;
    truncated.pop_back();
    require(!common_speculative_restore_state(ctx.spec, 1, truncated), "truncated EAGLE boundary is rejected");
    require(!common_speculative_restore_state(ctx.spec, 1, {}), "missing EAGLE boundary cannot replace a saved boundary");
    require(common_speculative_get_state(ctx.spec, 1, got) && got == state, "invalid EAGLE boundary leaves state unchanged");
    printf("PASS EAGLE: boundary state restores into a fresh driver; invalid snapshots preserve it\n");
}

int main(int argc, char ** argv) {
    common_log_set_verbosity_thold(LOG_LEVEL_ERROR);
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            fputs(text, stderr);
        }
    }, nullptr);
    llama_backend_init();
    int failed = 0;
    llama_model * model = nullptr;
    try {
        model = make_model();
        if (argc == 3 && std::strcmp(argv[1], "--save-model") == 0) {
            llama_model_saver saver(model);
            saver.add_kv_from_model();
            saver.add_tensors_from_model();
            // Include the MTP block in the fixture's recurrent-layer metadata.
            const bool recurrent[] = { true, false, false };
            gguf_set_arr_data(saver.gguf_ctx, saver.llm_kv(LLM_KV_ATTENTION_RECURRENT_LAYERS).c_str(), GGUF_TYPE_BOOL, recurrent, 3);
            saver.save(argv[2]);
            llama_model_free(model);
            llama_backend_free();
            return 0;
        }
        const cache_config configs[] = {
            { "q8_0/q8_0", GGML_TYPE_Q8_0, GGML_TYPE_Q8_0 },
            { "tq5_0/turbo4", GGML_TYPE_TQ5_0, GGML_TYPE_TURBO4_0 },
            { "SJ-KVaRN 4/4", GGML_TYPE_F16, GGML_TYPE_F16, 4, 4 },
            { "SJ-KVaRN 3/3t", GGML_TYPE_F16, GGML_TYPE_F16, 3, 3, GGML_TYPE_I16 },
            { "SJ-KVaRN 3/2t", GGML_TYPE_F16, GGML_TYPE_F16, 3, 2, GGML_TYPE_I16 },
        };
        for (const auto & config : configs) {
            try {
                run_prefill_identity(model, config);
            } catch (const std::exception & error) {
                fprintf(stderr, "FAIL %s (prefill identity): %s\n", config.name, error.what());
                ++failed;
            }
            for (bool deferred : { false, true }) {
                try {
                    run_case(model, config, deferred);
                } catch (const std::exception & error) {
                    fprintf(stderr, "FAIL %s (%s): %s\n", config.name, deferred ? "deferred" : "batch", error.what());
                    ++failed;
                }
            }
            try {
                run_deferred_overflow(model, config);
            } catch (const std::exception & error) {
                fprintf(stderr, "FAIL %s (deferred overflow): %s\n", config.name, error.what());
                ++failed;
            }
        }
        try {
            run_eagle_boundary_restore(model);
        } catch (const std::exception & error) {
            fprintf(stderr, "FAIL EAGLE boundary restore: %s\n", error.what());
            ++failed;
        }
    } catch (const std::exception & error) {
        fprintf(stderr, "FAIL fixture: %s\n", error.what());
        ++failed;
    }
    llama_model_free(model);
    llama_backend_free();
    return failed ? 1 : 0;
}
