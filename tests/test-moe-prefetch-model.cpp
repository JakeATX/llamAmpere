#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMPLETION)) {
        return 1;
    }
    llama_backend_init();
    llama_numa_init(params.numa);
    auto init = common_init_from_params(params);
    auto * ctx = init->context();
    if (!ctx) {
        return 2;
    }
    const char * input_path = std::getenv("FN_GOLDEN_IDS");
    const char * output_path = std::getenv("FN_GOLDEN_LOGITS");
    if (!input_path || !output_path) {
        return 3;
    }
    FILE * input = std::fopen(input_path, "rb");
    const char * prompt_env = std::getenv("FN_GOLDEN_PROMPT");
    const int n_prompt = prompt_env ? std::atoi(prompt_env) : 2048;
    if (n_prompt < 1 || n_prompt > 65536) {
        return 11;
    }
    std::vector<llama_token> tokens((size_t) n_prompt + 32);
    if (!input || std::fread(tokens.data(), sizeof(llama_token), tokens.size(), input) != tokens.size()) {
        return 4;
    }
    std::fclose(input);
    llama_set_decode_phase(ctx, LLAMA_DECODE_PHASE_PROMPT);
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), n_prompt)) != 0) {
        return 5;
    }
    llama_set_decode_phase(ctx, LLAMA_DECODE_PHASE_GENERATION);
    FILE * output = std::fopen(output_path, "wb");
    if (!output) {
        return 6;
    }
    const int32_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(init->model()));
    constexpr int widths[] = {1, 4, 5, 8};
    constexpr int steps_per_width = 8;
    const int32_t rows = steps_per_width * (1 + 4 + 5 + 8);
    std::fwrite(&vocab, sizeof(vocab), 1, output);
    std::fwrite(&rows, sizeof(rows), 1, output);
    for (int step = 0; step < 4 * steps_per_width; ++step) {
        const int batch = widths[step / steps_per_width];
        // Fixed input IDs make flag comparisons independent of sampling drift.
        auto input_batch = llama_batch_get_one(tokens.data() + n_prompt + step % 16, batch);
        std::vector<int8_t> all_outputs(batch, 1);
        input_batch.logits = all_outputs.data();
        if (llama_decode(ctx, input_batch) != 0) {
            return 7;
        }
        for (int row = 0; row < batch; ++row) {
            const float * logits = llama_get_logits_ith(ctx, row);
            if (!logits || std::fwrite(logits, sizeof(float), vocab, output) != (size_t) vocab) {
                return 8;
            }
        }
    }
    std::fclose(output);
    const char * token_path = std::getenv("FN_GOLDEN_TOKENS");
    llama_token next = 0;
    if (token_path) {
        const float * logits = llama_get_logits_ith(ctx, widths[3] - 1);
        if (!logits) {
            return 12;
        }
        next = llama_token(std::max_element(logits, logits + vocab) - logits);
    }
    const size_t bytes = llama_state_get_size(ctx);
    std::vector<uint8_t> state(bytes);
    if (llama_state_get_data(ctx, state.data(), bytes) != bytes ||
            llama_state_set_data(ctx, state.data(), bytes) != bytes) {
        return 9;
    }
    std::vector<uint8_t> restored(llama_state_get_size(ctx));
    if (llama_state_get_data(ctx, restored.data(), restored.size()) != restored.size() || state != restored) {
        return 10;
    }
    if (token_path) {
        FILE * generated = std::fopen(token_path, "wb");
        if (!generated) {
            return 13;
        }
        for (int step = 0; step < 32; ++step) {
            if (std::fwrite(&next, sizeof(next), 1, generated) != 1 ||
                    llama_decode(ctx, llama_batch_get_one(&next, 1)) != 0) {
                std::fclose(generated);
                return 14;
            }
            const float * logits = llama_get_logits_ith(ctx, 0);
            if (!logits) {
                std::fclose(generated);
                return 15;
            }
            next = llama_token(std::max_element(logits, logits + vocab) - logits);
        }
        std::fclose(generated);
        std::puts("prefetch-golden: generated_tokens=32 greedy=1 after_restore=1");
    }
    std::printf("prefetch-golden: rows=%d widths=1,4,5,8 vocab=%d saved-cache-bytes=%zu roundtrip=exact\n", rows, vocab, bytes);
    return 0;
}
