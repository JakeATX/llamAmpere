// Built-in draft vocabulary shortlists (--spec-draft-vocab-map auto).
// Without arguments: checks the embedded bitmaps against their headers and the "auto[:N]" parser.
// With GGUF paths: loads each vocabulary only, computes the tokenizer fingerprint and reports which built-in
// list `auto` picks; `--expect NAME` makes a mismatch fail (use it to prove a new quant maps to its family).
#include "../src/llama-mtp-vocab-builtin.h"
#include "../src/llama-ext.h"
#include "llama.h"
#include "ggml.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>

static void check_parse() {
    int64_t n = -1;
    GGML_ASSERT(llama_mtp_vocab_parse_auto("auto", n) && n == 0);
    GGML_ASSERT(llama_mtp_vocab_parse_auto("auto:32768", n) && n == 32768);
    GGML_ASSERT(!llama_mtp_vocab_parse_auto("auto:", n));
    GGML_ASSERT(!llama_mtp_vocab_parse_auto("auto:0", n));
    GGML_ASSERT(!llama_mtp_vocab_parse_auto("auto:12x", n));
    GGML_ASSERT(!llama_mtp_vocab_parse_auto("automatic", n));
    GGML_ASSERT(!llama_mtp_vocab_parse_auto("docs/mtp-vocab/atx_65536.txt", n));
    GGML_ASSERT(!llama_mtp_vocab_parse_auto(nullptr, n));
}

static void check_builtins() {
    int n_lists = 0;
    for (const llama_mtp_vocab_builtin * b = llama_mtp_vocab_builtins(); b->name != nullptr; ++b, ++n_lists) {
        const auto ids = llama_mtp_vocab_builtin_ids(*b);
        GGML_ASSERT((int64_t) ids.size() == b->n_sel);
        for (size_t i = 1; i < ids.size(); ++i) {
            GGML_ASSERT(ids[i - 1] < ids[i]);
        }
        GGML_ASSERT(ids.front() >= 0 && ids.back() < b->n_vocab);
        // find() returns this entry when asked for its exact size
        GGML_ASSERT(llama_mtp_vocab_builtin_find(b->tok_fp, b->n_vocab, b->arch, b->n_sel) == b);
        // a different architecture, vocabulary size or fingerprint never matches
        GGML_ASSERT(llama_mtp_vocab_builtin_find(b->tok_fp, b->n_vocab, "llama", 0) == nullptr || std::strcmp(b->arch, "llama") == 0);
        GGML_ASSERT(llama_mtp_vocab_builtin_find(b->tok_fp, b->n_vocab + 1, b->arch, 0) == nullptr);
        GGML_ASSERT(llama_mtp_vocab_builtin_find(b->tok_fp ^ 1, b->n_vocab, b->arch, 0) == nullptr);
        GGML_ASSERT(llama_mtp_vocab_builtin_find(b->tok_fp, b->n_vocab, b->arch, b->n_sel + 1) == nullptr);
        printf("built-in %-20s %-12s %-8s fp %016" PRIx64 " %6lld of %lld tokens, ids %d..%d\n", b->name, b->family, b->arch,
               b->tok_fp, (long long) b->n_sel, (long long) b->n_vocab, ids.front(), ids.back());
    }
    GGML_ASSERT(n_lists > 0);

    // the preferred size is 65,536 when the family has it
    const llama_mtp_vocab_builtin * b0 = llama_mtp_vocab_builtins();
    const llama_mtp_vocab_builtin * pick = llama_mtp_vocab_builtin_find(b0->tok_fp, b0->n_vocab, b0->arch, 0);
    GGML_ASSERT(pick != nullptr);
    bool has_64k = false;
    for (const llama_mtp_vocab_builtin * b = b0; b->name != nullptr; ++b) {
        has_64k |= b->tok_fp == b0->tok_fp && b->n_vocab == b0->n_vocab && std::strcmp(b->arch, b0->arch) == 0 && b->n_sel == 65536;
    }
    GGML_ASSERT(!has_64k || pick->n_sel == 65536);
}

int main(int argc, char ** argv) {
    check_parse();
    check_builtins();

    const char * expect = nullptr;
    int n_fail = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--expect") == 0 && i + 1 < argc) {
            expect = argv[++i];
            continue;
        }
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.vocab_only = true;
        llama_model * model = llama_model_load_from_file(argv[i], mp);
        if (model == nullptr) {
            fprintf(stderr, "cannot load %s\n", argv[i]);
            return 1;
        }
        const uint64_t fp = llama_model_tokenizer_fingerprint(model);
        char arch[64] = {};
        llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
        const int64_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        const llama_mtp_vocab_builtin * b = llama_mtp_vocab_builtin_find(fp, n_vocab, arch, 0);
        printf("%s: fp %016" PRIx64 " arch %s n_vocab %lld -> %s\n", argv[i], fp, arch, (long long) n_vocab, b ? b->name : "(none)");
        if (expect && (b == nullptr || std::strcmp(b->name, expect) != 0)) {
            fprintf(stderr, "expected %s\n", expect);
            n_fail++;
        }
        llama_model_free(model);
    }
    if (n_fail == 0) {
        printf("OK\n");
    }
    return n_fail == 0 ? 0 : 1;
}
