#include "arg.h"
#include "common.h"
#include "llama.h"

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
    std::vector<llama_token> tokens(2080);
    if (!input || std::fread(tokens.data(), sizeof(llama_token), tokens.size(), input) != tokens.size()) {
        return 4;
    }
    std::fclose(input);
    llama_set_decode_phase(ctx, LLAMA_DECODE_PHASE_PROMPT);
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), 2048)) != 0) {
        return 5;
    }
    llama_set_decode_phase(ctx, LLAMA_DECODE_PHASE_GENERATION);
    FILE * output = std::fopen(output_path, "wb");
    if (!output) {
        return 6;
    }
    const int32_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(init->model()));
    const int32_t rows = 32;
    std::fwrite(&vocab, sizeof(vocab), 1, output);
    std::fwrite(&rows, sizeof(rows), 1, output);
    for (int step = 0; step < rows; ++step) {
        const int batch = step < 16 ? 1 : 4;
        // Fixed input IDs make flag comparisons independent of sampling drift.
        if (llama_decode(ctx, llama_batch_get_one(tokens.data() + 2048 + step % 16, batch)) != 0) {
            return 7;
        }
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits || std::fwrite(logits, sizeof(float), vocab, output) != (size_t) vocab) {
            return 8;
        }
    }
    std::fclose(output);
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
    std::printf("prefetch-golden: rows=%d vocab=%d saved-cache-bytes=%zu roundtrip=exact\n", rows, vocab, bytes);
    return 0;
}
