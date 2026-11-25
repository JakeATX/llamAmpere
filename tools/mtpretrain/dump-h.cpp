// MTPRETRAIN: dump the trunk's h_nextn rows (final hidden state after output_norm, the MTP head's "h" input)
// for every token of a list of sequences, as bf16. Offline data tool for retraining the MTP draft head on the
// quantized model's own states; it does not change any runtime path.
//
// usage: llama-mtp-dump-h -m MODEL [usual common flags: -c, -b, -ub, -fa, -ctk, -ctv, -ngl ...] \
//            --mtp-in SEQ.bin --mtp-out H.bin
// SEQ.bin: int32 n_seq, then per sequence: int32 len, int32 tokens[len]
// H.bin:   per sequence, len rows of n_embd bf16 (row p = h of token p), sequences back to back.
//          H.bin.idx: text, one line per sequence: "seq_index row_offset len"
// Every sequence starts from an empty cache (memory cleared), decoded in chunks of n_batch tokens.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static uint16_t f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) { return 0x7fc0; }  // NaN
    const uint32_t rounding = 0x7fffu + ((u >> 16) & 1u);         // round to nearest even
    return (uint16_t) ((u + rounding) >> 16);
}

int main(int argc, char ** argv) {
    std::string in_path, out_path;
    std::vector<char *> args;
    for (int i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], "--mtp-in") == 0 && i + 1 < argc) { in_path = argv[++i]; continue; }
        if (std::strcmp(argv[i], "--mtp-out") == 0 && i + 1 < argc) { out_path = argv[++i]; continue; }
        args.push_back(argv[i]);
    }
    if (in_path.empty() || out_path.empty()) {
        fprintf(stderr, "usage: %s -m MODEL [common flags] --mtp-in SEQ.bin --mtp-out H.bin\n", argv[0]);
        return 2;
    }
    common_params params;
    if (!common_params_parse((int) args.size(), args.data(), params, LLAMA_EXAMPLE_COMMON)) { return 2; }
    params.warmup = false;

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    auto init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    if (!model || !ctx) { fprintf(stderr, "dump-h: model/context init failed\n"); return 1; }
    const int n_embd = llama_model_n_embd(model);
    const int n_ctx = (int) llama_n_ctx(ctx);
    const int n_batch = (int) llama_n_batch(ctx);
    llama_set_embeddings_nextn(ctx, true, /*masked=*/ false);

    FILE * fin = fopen(in_path.c_str(), "rb");
    if (!fin) { fprintf(stderr, "dump-h: cannot open %s\n", in_path.c_str()); return 1; }
    int32_t n_seq = 0;
    if (fread(&n_seq, 4, 1, fin) != 1 || n_seq < 0) { fprintf(stderr, "dump-h: bad header\n"); return 1; }
    FILE * fout = fopen(out_path.c_str(), "wb");
    FILE * fidx = fopen((out_path + ".idx").c_str(), "w");
    if (!fout || !fidx) { fprintf(stderr, "dump-h: cannot write %s\n", out_path.c_str()); return 1; }

    std::vector<llama_token> toks;
    std::vector<uint16_t> row(n_embd);
    int64_t row_off = 0;
    const int64_t t0 = ggml_time_us();
    int64_t n_tok_total = 0;
    for (int s = 0; s < n_seq; ++s) {
        int32_t len = 0;
        if (fread(&len, 4, 1, fin) != 1 || len <= 0) { fprintf(stderr, "dump-h: bad length at seq %d\n", s); return 1; }
        toks.resize(len);
        if (fread(toks.data(), 4, len, fin) != (size_t) len) { fprintf(stderr, "dump-h: short read at seq %d\n", s); return 1; }
        if (len > n_ctx) { fprintf(stderr, "dump-h: seq %d has %d tokens > n_ctx %d\n", s, len, n_ctx); return 1; }
        llama_memory_clear(llama_get_memory(ctx), true);
        for (int p0 = 0; p0 < len; p0 += n_batch) {
            const int n = std::min(n_batch, len - p0);
            llama_batch batch = llama_batch_init(n, 0, 1);
            for (int j = 0; j < n; ++j) {
                batch.token[j] = toks[p0 + j];
                batch.pos[j] = p0 + j;
                batch.n_seq_id[j] = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j] = j == n - 1;  // one output row keeps the head cost small; h_nextn is unmasked
            }
            batch.n_tokens = n;
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "dump-h: decode failed at seq %d pos %d\n", s, p0); return 1; }
            for (int j = 0; j < n; ++j) {
                const float * h = llama_get_embeddings_nextn_ith(ctx, j);
                if (!h) { fprintf(stderr, "dump-h: no h_nextn row %d (seq %d pos %d)\n", j, s, p0 + j); return 1; }
                for (int k = 0; k < n_embd; ++k) {
                    if (!std::isfinite(h[k])) { fprintf(stderr, "dump-h: non-finite h at seq %d pos %d\n", s, p0 + j); return 1; }
                    row[k] = f32_to_bf16(h[k]);
                }
                fwrite(row.data(), 2, n_embd, fout);
            }
            llama_batch_free(batch);
        }
        fprintf(fidx, "%d %lld %d\n", s, (long long) row_off, len);
        row_off += len;
        n_tok_total += len;
        if ((s + 1) % 50 == 0 || s + 1 == n_seq) {
            const double dt = (ggml_time_us() - t0) / 1e6;
            fprintf(stderr, "dump-h: %d/%d seqs, %lld tokens, %.1f tok/s\n", s + 1, n_seq, (long long) n_tok_total, n_tok_total / dt);
            fflush(fout); fflush(fidx);
        }
    }
    fclose(fin); fclose(fout); fclose(fidx);
    llama_backend_free();
    return 0;
}
