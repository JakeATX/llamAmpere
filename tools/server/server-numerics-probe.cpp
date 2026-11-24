#include "server-numerics-probe.h"

#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

bool server_numerics_probe_enabled() {
    const char * e = getenv("LLAMA_PROMPT_CACHE_NUMERICS_PROBE");
    return e != nullptr && e[0] == '1' && e[1] == '\0';
}

static uint64_t fnv1a(uint64_t h, const void * data, size_t n) {
    const auto * p = (const uint8_t *) data;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

static constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ull;
static constexpr int      PROBE_TOPK = 32;

// one probe pass: clear seq 0, decode the prompt, copy the last-token logits, clear seq 0 again
static bool probe_once(llama_context * ctx, const llama_tokens & toks, int n_vocab, std::vector<float> & logits, std::string & err) {
    llama_memory_t mem = llama_get_memory(ctx);
    if (mem == nullptr) {
        err = "context has no memory";
        return false;
    }
    llama_memory_seq_rm(mem, 0, -1, -1);

    llama_batch batch = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        common_batch_add(batch, toks[i], (llama_pos) i, { 0 }, i + 1 == toks.size());
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        llama_memory_seq_rm(mem, 0, -1, -1);
        err = "probe decode failed (rc " + std::to_string(rc) + ")";
        return false;
    }
    const float * l = llama_get_logits_ith(ctx, -1);
    if (l == nullptr) {
        llama_memory_seq_rm(mem, 0, -1, -1);
        err = "probe produced no logits";
        return false;
    }
    logits.assign(l, l + n_vocab);
    llama_memory_seq_rm(mem, 0, -1, -1);
    return true;
}

static uint64_t rank_hash(const std::vector<float> & logits) {
    const int k = std::min<int>(PROBE_TOPK, (int) logits.size());
    std::vector<int32_t> idx(logits.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int32_t a, int32_t b) {
        return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
    });
    return fnv1a(FNV_OFFSET, idx.data(), (size_t) k*sizeof(int32_t));
}

bool server_numerics_probe(llama_context * ctx, std::string & out, std::string & err) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (llama_pooling_type(ctx) != LLAMA_POOLING_TYPE_NONE) {
        err = "pooled (embedding) context";
        return false;
    }
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // fixed text: words, digits and punctuation, so the probe crosses several embedding rows and positions
    static const char * probe_text = "Numerics probe: The quick brown fox jumps over the lazy dog, 0123456789 times.";
    llama_tokens toks = common_tokenize(vocab, probe_text, true, true);
    const size_t n_max = std::min<size_t>(llama_n_batch(ctx), llama_n_ubatch(ctx));
    if (toks.size() > n_max) {
        toks.resize(n_max);
    }
    if (toks.empty()) {
        err = "probe text tokenized to nothing";
        return false;
    }

    std::vector<float> a, b;
    if (!probe_once(ctx, toks, n_vocab, a, err) || !probe_once(ctx, toks, n_vocab, b, err)) {
        return false;
    }

    const uint64_t tok_h = fnv1a(FNV_OFFSET, toks.data(), toks.size()*sizeof(llama_token));
    if (std::memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0) {
        out = string_format("bits:%016" PRIx64 ":%016" PRIx64, tok_h, fnv1a(FNV_OFFSET, a.data(), a.size()*sizeof(float)));
        return true;
    }
    const uint64_t ra = rank_hash(a);
    if (ra == rank_hash(b)) {
        LOG_WRN("prompt cache numerics probe: logits not bit-repeatable run to run; keying on the top-%d order only\n", PROBE_TOPK);
        out = string_format("rank:%016" PRIx64 ":%016" PRIx64, tok_h, ra);
        return true;
    }
    LOG_WRN("%s", "prompt cache numerics probe: top-token order not repeatable run to run; disk entries from other processes will not be restored\n");
    out = string_format("unstable:%lld", (long long) std::chrono::steady_clock::now().time_since_epoch().count());
    return true;
}
