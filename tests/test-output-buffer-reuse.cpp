#include "llama.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

static void require(bool value, const char * message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s CPU_MODEL.gguf\n", argv[0]);
        return 1;
    }
    try {
        llama_backend_init();
        ggml_backend_dev_t devices[] = {ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr};
        require(devices[0] != nullptr, "CPU backend unavailable");
        auto mp = llama_model_default_params();
        mp.devices = devices;
        mp.n_gpu_layers = 0;
        using model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
        model_ptr model(llama_model_load_from_file(argv[1], mp), llama_model_free);
        require(model != nullptr, "model load failed");

        using sampler_ptr = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;
        std::vector<sampler_ptr> samplers;

        auto cp = llama_context_default_params();
        cp.n_ctx = 512;
        cp.n_batch = 32;
        cp.n_ubatch = 32;
        cp.n_outputs_max = 32;
        cp.n_seq_max = 4;
        cp.kv_unified = true;
        cp.n_threads = 2;
        cp.n_threads_batch = 2;
        cp.offload_kqv = false;
        using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;
        context_ptr ctx(llama_init_from_model(model.get(), cp), llama_free);
        require(ctx != nullptr, "context creation failed");
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        require(n_vocab > 32, "fixture vocabulary too small");

        int position = 0;
        uintptr_t stable_logits = 0;
        int checks = 0;
        auto decode = [&](llama_sampler * sampler, int rows, const char * name) {
            require(llama_set_sampler(ctx.get(), 0, sampler), "sampler configuration failed");
            llama_batch batch = llama_batch_init(rows, 0, 1);
            batch.n_tokens = rows;
            for (int i = 0; i < rows; ++i) {
                batch.token[i] = 16 + (position + i) % 16;
                batch.pos[i] = position + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = true;
            }
            const int rc = llama_decode(ctx.get(), batch);
            llama_batch_free(batch);
            require(rc == 0, "decode failed");
            float * logits = llama_get_logits(ctx.get());
            require(logits != nullptr, "logits storage missing");
            const uintptr_t address = reinterpret_cast<uintptr_t>(logits);
            if (!stable_logits) {
                stable_logits = address;
            }
            std::printf("%s: rows=%d, logits buffer %s\n", name, rows, address == stable_logits ? "stable" : "MOVED");
            require(address == stable_logits, "sampler switch reallocated the logits buffer");
            if (sampler) {
                const llama_token token = llama_get_sampled_token_ith(ctx.get(), 0);
                require(token >= 0 && token < n_vocab, "backend did not produce a valid sampled token");
            } else {
                for (int i = 0; i < rows * n_vocab; ++i) {
                    require(std::isfinite(logits[i]), "nonfinite CPU logits");
                }
            }
            position += rows;
            ++checks;
        };
        decode(nullptr, 16, "initial CPU wide output");
        for (int repeat = 0; repeat < 2; ++repeat) {
            // A backend chain can only be initialized once, so each attachment uses a fresh chain.
            samplers.emplace_back(llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free);
            llama_sampler_chain_add(samplers.back().get(), llama_sampler_init_greedy());
            decode(samplers.back().get(), 1, "compact greedy");
            samplers.emplace_back(llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free);
            llama_sampler_chain_add(samplers.back().get(), llama_sampler_init_top_k(20));
            llama_sampler_chain_add(samplers.back().get(), llama_sampler_init_temp(1.0f));
            llama_sampler_chain_add(samplers.back().get(), llama_sampler_init_dist(6100));
            decode(samplers.back().get(), 1, "sampled output");
            decode(nullptr, 16, "CPU wide output");
            decode(nullptr, 1, "CPU narrow output");
            decode(nullptr, 16, "CPU wide output again");
        }
        require(llama_set_sampler(ctx.get(), 0, nullptr), "sampler cleanup failed");
        std::printf("PASS: %d decodes, stable logits storage across sampler and row-count changes\n", checks);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 2;
    }
    return 0;
}
