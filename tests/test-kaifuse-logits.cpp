// [KAIFUSE] Full-logit dump for the R1 identity check on a real model (needs -m, so not a ctest entry).
// Prefills a fixed token sequence in 64-token ubatches, then teacher-forces it at decode widths 1, 2, 3, 4, 5 and 8
// (the MTP verify widths). Writes one line per logit row to OUT: width, position, FNV-1a 64 of the row's float32 bytes,
// and the argmax. Run once with LLAMA_KAIFUSE=0 and once with the default, then compare the two files (cmp): equal
// hashes = bit-identical logits.
// usage: test-kaifuse-logits MODEL OUT [n_tokens=256]
#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s MODEL OUT [n_tokens]\n", argv[0]);
        return 2;
    }
    const int n_tok = argc > 3 ? std::atoi(argv[3]) : 256;
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // a deterministic token walk over ordinary ids (no special tokens)
    std::vector<llama_token> toks(n_tok);
    uint32_t r = 20261121u;
    for (int i = 0; i < n_tok; ++i) {
        r = r*1664525u + 1013904223u;
        toks[i] = (llama_token) (1000 + (r >> 8) % 60000);
    }
    FILE * out = std::fopen(argv[2], "w");
    if (!out) {
        return 1;
    }
    for (int width : { 64, 1, 2, 3, 4, 5, 8 }) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 4096;
        cp.n_batch = 512;
        cp.n_ubatch = width == 64 ? 64 : 512;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) {
            return 1;
        }
        // half the sequence as a 64-row prefill (the cached prep covers it too), then the rest at this width
        const int pre = n_tok/2;
        llama_batch b = llama_batch_init(std::max(pre, 64), 0, 1);
        int pos = 0;
        auto run = [&](int n, bool all_logits) {
            b.n_tokens = n;
            for (int i = 0; i < n; ++i) {
                b.token[i] = toks[pos + i];
                b.pos[i] = pos + i;
                b.n_seq_id[i] = 1;
                b.seq_id[i][0] = 0;
                b.logits[i] = all_logits || i == n - 1;
            }
            if (llama_decode(ctx, b) != 0) {
                std::fprintf(stderr, "decode failed at %d\n", pos);
                std::exit(1);
            }
            for (int i = 0; i < n; ++i) {
                if (b.logits[i]) {
                    const float * l = llama_get_logits_ith(ctx, i);
                    const unsigned char * p = (const unsigned char *) l;
                    uint64_t h = 1469598103934665603ULL;
                    for (size_t k = 0; k < (size_t) n_vocab*sizeof(float); ++k) {
                        h = (h ^ p[k])*1099511628211ULL;
                    }
                    int am = 0;
                    for (int k = 1; k < n_vocab; ++k) {
                        am = l[k] > l[am] ? k : am;
                    }
                    std::fprintf(out, "%d %d %016llx %d\n", width, pos + i, (unsigned long long) h, am);
                }
            }
            pos += n;
        };
        run(pre, false);
        const int step = width == 64 ? 1 : width;
        while (pos + step <= n_tok) {
            run(step, true);
        }
        llama_batch_free(b);
        llama_free(ctx);
        std::fprintf(stderr, "width %d: done\n", width);
    }
    std::fclose(out);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
