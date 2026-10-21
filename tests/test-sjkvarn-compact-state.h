#pragma once

#include "ggml-cpp.h"

#include <algorithm>
#include <memory>

// The small model supplies real cache/state bytes. This graph supplies the
// 24Q/2KV sparse geometry that the compact kernel accepts.
struct compact_state_probe {
    static constexpr int positions = 12288;
    static constexpr int bound = 512;

    using count_t = int64_t (*)(ggml_backend_t, const char *);
    using reset_t = bool (*)(ggml_backend_t);

    llama_context * lctx;
    llama_kv_cache * cache;
    ggml_backend_t backend;
    int width;
    count_t count;
    reset_t reset;
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * q;
    ggml_tensor * mask;
    ggml_tensor * desc;
    ggml_tensor * actual;
    ggml_tensor * reference;
    ggml_cgraph * graph_actual;
    ggml_cgraph * graph_reference;
    std::vector<float> previous_q;
    std::vector<ggml_fp16_t> previous_mask;
    std::vector<int32_t> previous_desc;
    int input_updates = 0;
    int descriptor_updates = 0;
    int graph_replays = 0;
    int64_t dispatches = 0;

    compact_state_probe(llama_context * lctx, int width) : lctx(lctx), cache(cache_of(lctx)), width(width) {
        lctx->synchronize();
        auto * body = cache->get_sj_kvarn_body(0);
        require(body && body->buffer && !ggml_backend_buffer_is_host(body->buffer), "compact state body is on the GPU");
        backend = lctx->sj_kvarn_backend(body->buffer);
        require(backend != nullptr, "compact state GPU backend");
        auto * dev = ggml_backend_get_device(backend);
        require(dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU, "compact state uses a GPU backend");
        auto * reg = ggml_backend_dev_backend_reg(dev);
        count = (count_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_fusion_count");
        reset = (reset_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_graph_reset");
        require(count && reset && count(backend, "sj_compact") >= 0, "compact state requires the CUDA sj_compact counter and graph reset API");

        ggml_init_params ip = { 64*ggml_tensor_overhead() + 2*ggml_graph_overhead(), nullptr, true };
        ctx.reset(ggml_init(ip));
        require(ctx != nullptr, "compact state graph context");
        q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 256, width, 24, 1);
        mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, positions, width, 1, 1);
        desc = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, GGML_SJKVARN_DESC_N_ENTRIES);
        ggml_set_input(q);
        ggml_set_input(mask);
        ggml_set_input(desc);
        llama_kv_cache::slot_info si;
        si.s0 = si.s1 = 0;
        si.strm = { 0 };
        si.idxs = { { 0 } };
        auto * k = ggml_permute(ctx.get(), cache->get_k(ctx.get(), 0, 0, si), 0, 2, 1, 3);
        auto * v = ggml_permute(ctx.get(), cache->get_v(ctx.get(), 0, 0, si), 0, 2, 1, 3);
        require(k->ne[0] == 256 && k->ne[2] == 2 && v->ne[0] == 256 && v->ne[2] == 2, "compact state has D256/2KV geometry");
        auto attention = [&](int n_kv_max) {
            // Separate first nodes keep the control/candidate CUDA graph keys distinct.
            auto * qq = ggml_cont(ctx.get(), q);
            auto * out = ggml_flash_attn_ext(ctx.get(), qq, k, v, mask, 1.0f/16.0f, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_sj_kvarn(out, body, desc, 4, 4, positions);
            ggml_flash_attn_ext_set_n_kv_max(out, n_kv_max);
            ggml_prec_set_acc(out, GGML_PREC_F32);
            return out;
        };
        reference = attention(0); // bound=0 refuses compact dispatch.
        actual = attention(bound);
        graph_reference = ggml_new_graph(ctx.get());
        graph_actual = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph_reference, reference);
        ggml_build_forward_expand(graph_actual, actual);
        require(ggml_graph_n_nodes(graph_reference) > 0 && ggml_graph_n_nodes(graph_actual) > 0 &&
                ggml_graph_nodes(graph_reference)[0] != ggml_graph_nodes(graph_actual)[0], "compact/control graph keys differ");
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        require(buffer != nullptr, "compact state GPU graph buffer");
    }

    ~compact_state_probe() {
        // Captured graphs reference this probe's input and output buffers.
        ggml_backend_synchronize(backend);
        reset(backend);
    }

    std::vector<float> run(int epoch, const char * phase) {
        lctx->synchronize();
        const int n = (int) llama_memory_seq_pos_max(llama_get_memory(lctx), 0) + 1;
        const int qpos = n - width - epoch%3;
        auto d = descriptor(cache, qpos);
        const int sink = d[GGML_SJKVARN_DESC_S], body_end = d[GGML_SJKVARN_DESC_B];
        require(d[GGML_SJKVARN_DESC_N] == n && d[GGML_SJKVARN_DESC_G] == 128 &&
                d[GGML_SJKVARN_DESC_D] == 256 && d[GGML_SJKVARN_DESC_HKV] == 2 &&
                d[GGML_SJKVARN_DESC_BODY_TYPE] == GGML_TYPE_F32 && d[GGML_SJKVARN_DESC_TABLE] == 0,
                "compact state descriptor has scalar single-stream D256/G128/2KV geometry");
        require(n < positions && qpos >= body_end && body_end > sink && n > sink + d[GGML_SJKVARN_DESC_CAP],
                "compact state has sealed groups, wrapped ring and a causal query tail");
        std::mt19937 rng(20261020u + 37u*epoch + 1009u*width);
        std::vector<float> query(ggml_nelements(q));
        for (auto & x : query) { x = float(int(rng()%65536) - 32768)/131072.0f; }
        std::vector<ggml_fp16_t> masks((size_t) positions*width);
        for (int j = 0; j < width; ++j) {
            int selected = 0;
            const int pos = qpos + j;
            const int window = (17*epoch + 8*j)%508;
            for (int p = 0; p < positions; ++p) {
                const bool body_live = p >= sink && p < body_end && epoch%3 != 1 &&
                    ((p - sink)%512 >= window && (p - sink)%512 < window + 4);
                const bool live = p < n && p <= pos && (p < sink || p >= std::max(body_end, pos - 127) || body_live);
                masks[(size_t) j*positions + p] = ggml_fp32_to_fp16(live ? -0.125f*((p + 3*j + epoch)%9) : -INFINITY);
                selected += live;
            }
            require(selected > 0 && selected <= bound, "compact state sparse bound is valid");
        }
        if (!previous_q.empty()) {
            require(query != previous_q && masks != previous_mask, "compact state replay changes Q and masks");
            ++input_updates;
            descriptor_updates += d != previous_desc;
        }
        previous_q = query;
        previous_mask = masks;
        previous_desc = d;
        ggml_backend_tensor_set(q, query.data(), 0, ggml_nbytes(q));
        ggml_backend_tensor_set(mask, masks.data(), 0, ggml_nbytes(mask));
        ggml_backend_tensor_set(desc, d.data(), 0, ggml_nbytes(desc));

        const int64_t before = count(backend, "sj_compact");
        require(ggml_backend_graph_compute(backend, graph_reference) == GGML_STATUS_SUCCESS, "compact state ordinary attention compute");
        require(count(backend, "sj_compact") == before, "compact state ordinary control refuses compact dispatch");
        require(ggml_backend_graph_compute(backend, graph_actual) == GGML_STATUS_SUCCESS, "compact state candidate attention compute");
        const int64_t after = count(backend, "sj_compact");
        require(after >= before, "compact dispatch counter is monotonic");
        dispatches += after - before;
        require(dispatches > 0, "compact state candidate actually dispatched");
        // Eligibility depends only on fixed tensor geometry and op parameters.
        // Thus, once witnessed, this host dispatch can disappear only on a CUDA
        // graph replay. Direct execution and graph capture both increment it.
        graph_replays += after == before;
        std::vector<float> output(ggml_nelements(actual)), control(output.size());
        ggml_backend_tensor_get(actual, output.data(), 0, ggml_nbytes(actual));
        ggml_backend_tensor_get(reference, control.data(), 0, ggml_nbytes(reference));
        double difference = 0.0, norm = 0.0;
        for (size_t i = 0; i < output.size(); ++i) {
            require(std::isfinite(output[i]) && std::isfinite(control[i]), "compact state attention outputs are finite");
            const double delta = (double) output[i] - control[i];
            difference += delta*delta;
            norm += (double) control[i]*control[i];
        }
        const double relative_l2_squared = difference/std::max(norm, 1e-30);
        require(relative_l2_squared <= 1e-4, "compact state attention matches ordinary control");
        printf("compact state: phase=%s T=%d epoch=%d B=%d N=%d dispatches=%lld graph_replays=%d relative_l2_squared=%g\n",
                phase, width, epoch, body_end, n, (long long) dispatches, graph_replays, relative_l2_squared);
        return output;
    }
};

static void compact_state_live_equal(llama_context * a, llama_context * b) {
    const auto x = snapshot(a), y = snapshot(b);
    require(x.B == y.B && x.N == y.N && x.sink_k == y.sink_k && x.sink_v == y.sink_v &&
            x.f16_k == y.f16_k && x.f16_v == y.f16_v && x.tail_k == y.tail_k && x.tail_v == y.tail_v,
            "compact state sink/ring live bytes match the accepted-only oracle");
    auto * kv = cache_of(a);
    const uint32_t groups = (kv->get_size() - kv->get_sj_kvarn_sink() - kv->get_sj_kvarn().tail)/128;
    require(records(a, groups) == records(b, groups), "compact state sealed records match the oracle");
}

static void compact_state_case(llama_model * model, const sj_cfg & cfg, const char * tmp_dir) {
    const char * mode = getenv("GGML_SJKVARN_COMPACT");
    require(mode && (!strcmp(mode, "1") || !strcmp(mode, "2")), "--compact requires GGML_SJKVARN_COMPACT=1 or 2 before startup");
    require(getenv("SJ_TEST_GPU") != nullptr, "--compact requires SJ_TEST_GPU=1");
    require(getenv("GGML_CUDA_DISABLE_GRAPHS") == nullptr, "compact state replay requires CUDA graphs enabled");
    llama_context_ptr source(make_ctx(model, cfg, 1, false, compact_state_probe::positions));
    llama_context_ptr restored(make_ctx(model, cfg, 1, false, compact_state_probe::positions));
    llama_context_ptr oracle(make_ctx(model, cfg, 1, false, compact_state_probe::positions));
    require(source && restored && oracle, "compact state contexts");
    const llama_pos initial = 8193;
    decode(source.get(), 0, initial, 128);
    const auto saved = seq_state(source.get());
    const std::string path = std::string(tmp_dir) + "/sjkvarn-compact-state.bin";
    std::vector<llama_token> saved_tokens;
    for (llama_pos p = 0; p < initial; ++p) { saved_tokens.push_back(token_at(p, 0)); }
    require(llama_state_seq_save_file(source.get(), path.c_str(), 0, saved_tokens.data(), saved_tokens.size()) > 0,
            "compact state save file");
    decode(restored.get(), 0, 129, 128, 9);
    std::vector<llama_token> loaded_tokens(saved_tokens.size());
    size_t loaded = 0;
    require(llama_state_seq_load_file(restored.get(), path.c_str(), 0, loaded_tokens.data(), loaded_tokens.size(), &loaded) > 0 &&
            loaded == saved_tokens.size() && loaded_tokens == saved_tokens && seq_state(restored.get()) == saved,
            "compact state file restores exact tokens and state bytes");
    compact_state_live_equal(source.get(), restored.get());

    std::vector<std::unique_ptr<compact_state_probe>> from, into, ordinary;
    for (int width : {1, 4, 5}) {
        from.emplace_back(new compact_state_probe(source.get(), width));
        into.emplace_back(new compact_state_probe(restored.get(), width));
    }
    auto compare = [&](int epoch, const char * phase) {
        for (size_t i = 0; i < from.size(); ++i) {
            require(same_logits(from[i]->run(epoch, phase), into[i]->run(epoch, phase)),
                    "compact state source/restored attention is bit-identical");
        }
    };
    for (int epoch = 0; epoch < 4; ++epoch) { compare(epoch, "restore-replay"); }

    const uint32_t before_seal = cache_of(source.get())->get_sj_kvarn_sealed_end();
    const llama_pos continued = initial + 129;
    require(same_logits(decode(source.get(), initial, continued, 128, 5), decode(restored.get(), initial, continued, 128, 5)),
            "compact state restored model continuation logits are bit-identical");
    require(cache_of(source.get())->get_sj_kvarn_sealed_end() > before_seal, "compact state continuation seals another group");
    compact_state_live_equal(source.get(), restored.get());
    for (int epoch = 4; epoch < 8; ++epoch) { compare(epoch, "continuation-replay"); }

    const auto checkpoint = seq_state(source.get());
    require(llama_state_seq_set_data(oracle.get(), checkpoint.data(), checkpoint.size(), 0) == checkpoint.size(),
            "compact state accepted-only oracle checkpoint restore");
    for (int width : {1, 4, 5}) { ordinary.emplace_back(new compact_state_probe(oracle.get(), width)); }
    const uint32_t body_before_draft = cache_of(source.get())->get_sj_kvarn_sealed_end();
    // Tokenwise cache writes keep the accepted-only oracle's arithmetic equal.
    // The attention probes still run T1/T4/T5 before and after the rollback.
    decode(source.get(), continued, continued + 5, 1, 7);
    decode(restored.get(), continued, continued + 5, 1, 7);
    compare(8, "speculative-tail");
    require(cache_of(source.get())->get_sj_kvarn_sealed_end() == body_before_draft, "compact state speculative tail did not seal");
    const llama_pos accepted = continued + 2;
    require(llama_memory_seq_rm(llama_get_memory(source.get()), 0, accepted, -1) &&
            llama_memory_seq_rm(llama_get_memory(restored.get()), 0, accepted, -1), "compact state rejects three speculative rows");
    decode(oracle.get(), continued, accepted, 1, 7);
    compact_state_live_equal(source.get(), oracle.get());
    for (int epoch = 9; epoch < 13; ++epoch) {
        for (size_t i = 0; i < from.size(); ++i) {
            const auto result = from[i]->run(epoch, "rollback-replay");
            require(same_logits(result, into[i]->run(epoch, "rollback-replay")),
                    "compact state source/restored rollback attention is bit-identical");
            require(same_logits(result, ordinary[i]->run(epoch, "rollback-oracle")),
                    "compact state rollback attention is bit-identical to the accepted-only oracle");
        }
    }
    const auto expected = decode(oracle.get(), accepted, accepted + 17, 1, 11);
    require(same_logits(decode(source.get(), accepted, accepted + 17, 1, 11), expected) &&
            same_logits(decode(restored.get(), accepted, accepted + 17, 1, 11), expected),
            "compact state continuation after rollback matches the accepted-only oracle");
    compact_state_live_equal(source.get(), oracle.get());
    compare(13, "rollback-continuation");
    for (const auto & probe : from) {
        require(probe->dispatches > 0 && probe->graph_replays > 0 && probe->input_updates >= 12 && probe->descriptor_updates >= 12,
                "compact state observed dispatch, graph replay, changed inputs and changed descriptors at every width");
    }
    require(std::remove(path.c_str()) == 0, "remove compact state file");
    printf("OK: compact state T1/T4/T5, changed-input/descriptor replay, file restore, sealing continuation and 2/5 rollback\n");
}
