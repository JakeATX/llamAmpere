#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Built-in draft vocabulary shortlists, selected with `--spec-draft-vocab-map auto`.
//
// A list is ranked on one base model's generations, but the draft head it indexes is a row set of the
// tokenizer, so every quantization and fine-tune of that base shares it. The match is on the tokenizer
// itself (fingerprint of every token string and every BPE merge, in order) plus the architecture, never
// on the file name or `general.name`: quantizers rename models freely, and a different model class with
// the same tokenizer was not ranked by this list.
//
// Fingerprint: FNV-1a 64 over each token text followed by a NUL byte (ids 0..n_vocab-1), one 0x01 byte,
// then each merge ("left right", rank order, empty string for a rank with no merge) followed by a NUL.
// scripts/gen-mtp-vocab-builtin.py computes the same value from the GGUF arrays.

struct llama_fnv1a64 {
    uint64_t h = 1469598103934665603ULL;

    void add(const void * data, size_t n) {
        const auto * p = (const uint8_t *) data;
        for (size_t i = 0; i < n; ++i) {
            h ^= p[i];
            h *= 1099511628211ULL;
        }
    }
    void add_str0(const std::string & s) {
        add(s.data(), s.size());
        const uint8_t z = 0;
        add(&z, 1);
    }
};

struct llama_mtp_vocab_builtin {
    const char *         name;    // e.g. "qwen3.8-27b-65536"
    const char *         family;  // base model the ranking was built on
    const char *         arch;    // llm_arch name the list applies to
    uint64_t             tok_fp;  // tokenizer fingerprint (see above)
    int64_t              n_vocab;
    int64_t              n_sel;   // number of set bits
    const char * const * hex;     // n_vocab-bit bitmap, bit i of byte i/8 = token i, hex chunks, nullptr-terminated
};

// all built-in lists, terminated by an entry whose name is nullptr
const llama_mtp_vocab_builtin * llama_mtp_vocab_builtins();

// ascending token ids of a built-in list; throws if the embedded bitmap is malformed
std::vector<int32_t> llama_mtp_vocab_builtin_ids(const llama_mtp_vocab_builtin & b);

// Pick the built-in list for a tokenizer and architecture. `size` > 0 asks for that exact shortlist size;
// 0 takes the family's preferred size (65,536 when present, else the largest). nullptr when nothing matches.
const llama_mtp_vocab_builtin * llama_mtp_vocab_builtin_find(uint64_t tok_fp, int64_t n_vocab, const char * arch, int64_t size);

// Parse a `--spec-draft-vocab-map` value: "auto" or "auto:N" -> true with N (0 for "auto").
bool llama_mtp_vocab_parse_auto(const char * value, int64_t & size);
