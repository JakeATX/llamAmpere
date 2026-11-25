// INKSPEC2 DSPARKTRAIN: offline data + parity tool for retraining the Inkling2 DSpark drafter. Pure logging: it calls
// the same public API the server's DSpark path uses (common/speculative.cpp, common_speculative_impl_draft_dflash) and
// changes no runtime path.
//
// dump mode (target only):
//   llama-dspark-dump -m TARGET [common flags] --ds-in SEQ.bin --ds-out PREFIX [--ds-topk 64] [--ds-dtype bf16|f16|f32]
//   For every token of every sequence: the target features the server injects into the drafter, i.e. the rows of
//   llama_get_embeddings_layer_inp(ctx_tgt, id) for the drafter's target_layer_ids (read from --ds-layers or -md),
//   concatenated k-major like batch_inject ([k*n_embd + e]); and for rows >= n_prompt-1 the target's top-K logits.
//   PREFIX.feat : per token n_layers*n_embd values (dtype --ds-dtype), sequences back to back
//   PREFIX.top  : per token row >= n_prompt-1: K int32 ids (descending logit) then K float32 logits
//   PREFIX.idx  : text "seq feat_row_off len n_prompt top_row_off"
//   PREFIX.eog  : the target vocab's end-of-generation ids, one per line
//
// parity mode (target + drafter, as the server pairs them):
//   llama-dspark-dump -m TARGET -md DRAFTER [flags] --ds-in SEQ.bin --ds-out PREFIX --ds-parity N [--ds-walk 4]
//                     [--ds-sample-k 20] [--ds-seed 1]
//   For the first sequence: decodes the target, injects the features into the drafter in increasing position order
//   (process()), and at N anchors (spread over the reply) decodes the DSpark block [anchor, mask x7] at the anchor's
//   position with the committed context feature of pos-1 (draft()), twice:
//     A: argmax walk, logits out (LLAMA_DSPARK_IDS=0 path): full drafter logits of slots 1..walk
//     B: sampled walk (INKSPEC default): Gumbel noise from --ds-seed, rows [state, id, K ids, K logits]
//   then removes the block from the drafter cache (seq_rm, as the server does after drafting).
//   PREFIX.par.feat : f32 features of the whole sequence (the exact inputs)
//   PREFIX.par.bin  : header + per anchor: pos, noise[walk*K], A logits [walk][n_vocab], A ids [walk],
//                     B ids [walk], B cand ids [walk][K], B cand logits [walk][K], f_rnn[pos-1][n_embd]
//
// SEQ.bin: int32 n_seq, then per sequence: int32 len, int32 n_prompt, int32 tokens[len]
// Every sequence starts from an empty cache; the target decodes it in chunks of n_batch tokens.

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "speculative.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

static uint16_t f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) { return 0x7fc0; }
    const uint32_t rounding = 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) ((u + rounding) >> 16);
}

static uint16_t f32_to_f16(float f) {
    // round to nearest even, IEEE half
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t  exp  = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (((x >> 23) & 0xff) == 0xff) { return (uint16_t) (sign | 0x7c00u | (mant ? 0x200u : 0)); }
    if (exp >= 31) { return (uint16_t) (sign | 0x7c00u); }
    if (exp <= 0) {
        if (exp < -10) { return (uint16_t) sign; }
        mant |= 0x800000u;
        const int shift = 14 - exp;
        uint32_t h = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1u))) { h++; }
        return (uint16_t) (sign | h);
    }
    uint32_t h = ((uint32_t) exp << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) { h++; }
    return (uint16_t) (sign | h);
}

struct seq_t {
    std::vector<llama_token> toks;
    int32_t n_prompt = 0;
};

static bool read_seqs(const std::string & path, std::vector<seq_t> & out) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { return false; }
    int32_t n = 0;
    if (fread(&n, 4, 1, f) != 1 || n < 0) { fclose(f); return false; }
    out.resize(n);
    for (auto & s : out) {
        int32_t len = 0;
        if (fread(&len, 4, 1, f) != 1 || len <= 0 || fread(&s.n_prompt, 4, 1, f) != 1 || s.n_prompt <= 0 || s.n_prompt > len) {
            fclose(f); return false;
        }
        s.toks.resize(len);
        if (fread(s.toks.data(), 4, len, f) != (size_t) len) { fclose(f); return false; }
    }
    fclose(f);
    return true;
}

// top-K of a logit row, descending, ties by lower id first (as a stable host sort)
static void topk_row(const float * row, int n_vocab, int k, int32_t * ids, float * vals, std::vector<int32_t> & scratch) {
    scratch.resize(n_vocab);
    for (int i = 0; i < n_vocab; ++i) { scratch[i] = i; }
    auto cmp = [row](int32_t a, int32_t b) { return row[a] > row[b] || (row[a] == row[b] && a < b); };
    std::partial_sort(scratch.begin(), scratch.begin() + k, scratch.end(), cmp);
    for (int i = 0; i < k; ++i) { ids[i] = scratch[i]; vals[i] = row[scratch[i]]; }
}

// decode a whole sequence on the target in chunks; features[pos][k*n_embd + e] (f32), logits top-K for rows >= row0
static bool decode_target(llama_context * ctx, const seq_t & s, const std::vector<int32_t> & lids, int n_embd, int n_vocab,
                          int topk, int n_threads, std::vector<float> & feat, std::vector<int32_t> & top_ids,
                          std::vector<float> & top_vals) {
    const int len = (int) s.toks.size();
    const int n_batch = (int) llama_n_batch(ctx);
    const int nl = (int) lids.size();
    const int row0 = std::max(0, s.n_prompt - 1);
    feat.assign((size_t) len * nl * n_embd, 0.0f);
    if (topk > 0) {
        top_ids.assign((size_t) (len - row0) * topk, 0);
        top_vals.assign((size_t) (len - row0) * topk, 0.0f);
    }
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    for (int p0 = 0; p0 < len; p0 += n_batch) {
        const int n = std::min(n_batch, len - p0);
        for (int j = 0; j < n; ++j) {
            batch.token[j] = s.toks[p0 + j];
            batch.pos[j] = p0 + j;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j] = topk > 0 ? (p0 + j >= row0) : (j == n - 1);
        }
        batch.n_tokens = n;
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "dspark-dump: target decode failed at pos %d\n", p0); llama_batch_free(batch); return false; }
        for (int k = 0; k < nl; ++k) {
            const float * L = llama_get_embeddings_layer_inp(ctx, (uint32_t) lids[k]);
            if (!L) { fprintf(stderr, "dspark-dump: target layer %d input not extracted\n", lids[k]); llama_batch_free(batch); return false; }
            for (int j = 0; j < n; ++j) {
                const float * src = L + (size_t) j * n_embd;
                float * dst = feat.data() + ((size_t) (p0 + j) * nl + k) * n_embd;
                for (int e = 0; e < n_embd; ++e) {
                    if (!std::isfinite(src[e])) { fprintf(stderr, "dspark-dump: non-finite feature at pos %d\n", p0 + j); llama_batch_free(batch); return false; }
                }
                std::memcpy(dst, src, (size_t) n_embd * sizeof(float));
            }
        }
        if (topk > 0) {
            std::vector<int> rows;
            for (int j = 0; j < n; ++j) { if (p0 + j >= row0) { rows.push_back(j); } }
            std::vector<const float *> lrow(rows.size());
            for (size_t r = 0; r < rows.size(); ++r) {
                lrow[r] = llama_get_logits_ith(ctx, rows[r]);
                if (!lrow[r]) { fprintf(stderr, "dspark-dump: no logits row %d\n", rows[r]); llama_batch_free(batch); return false; }
            }
            std::vector<std::thread> th;
            const int nt = std::max(1, std::min(n_threads, (int) rows.size()));
            for (int t = 0; t < nt; ++t) {
                th.emplace_back([&, t]() {
                    std::vector<int32_t> scratch;
                    for (size_t r = t; r < rows.size(); r += nt) {
                        const size_t o = (size_t) (p0 + rows[r] - row0) * topk;
                        topk_row(lrow[r], n_vocab, topk, top_ids.data() + o, top_vals.data() + o, scratch);
                    }
                });
            }
            for (auto & x : th) { x.join(); }
        }
    }
    llama_batch_free(batch);
    return true;
}

int main(int argc, char ** argv) {
    std::string in_path, out_path, dtype = "bf16", layers_arg;
    int topk = 64, n_parity = 0, walk = 4, sample_k = 20, max_seq = -1;
    uint32_t seed = 1;
    std::vector<char *> args;
    for (int i = 0; i < argc; ++i) {
        const auto is = [&](const char * k) { return std::strcmp(argv[i], k) == 0 && i + 1 < argc; };
        if (is("--ds-in"))       { in_path = argv[++i]; continue; }
        if (is("--ds-out"))      { out_path = argv[++i]; continue; }
        if (is("--ds-topk"))     { topk = atoi(argv[++i]); continue; }
        if (is("--ds-dtype"))    { dtype = argv[++i]; continue; }
        if (is("--ds-layers"))   { layers_arg = argv[++i]; continue; }
        if (is("--ds-parity"))   { n_parity = atoi(argv[++i]); continue; }
        if (is("--ds-walk"))     { walk = atoi(argv[++i]); continue; }
        if (is("--ds-sample-k")) { sample_k = atoi(argv[++i]); continue; }
        if (is("--ds-seed"))     { seed = (uint32_t) atoll(argv[++i]); continue; }
        if (is("--ds-max-seq"))  { max_seq = atoi(argv[++i]); continue; }
        args.push_back(argv[i]);
    }
    if (in_path.empty() || out_path.empty() || (dtype != "bf16" && dtype != "f16" && dtype != "f32") || topk < 0 || topk > 1024) {
        fprintf(stderr, "usage: %s -m TARGET [-md DRAFTER] [common flags] --ds-in SEQ.bin --ds-out PREFIX [--ds-topk 64] "
                        "[--ds-dtype bf16|f16|f32] [--ds-layers 5,23,41,48] [--ds-parity N --ds-walk 4 --ds-sample-k 20 --ds-seed 1]\n", argv[0]);
        return 2;
    }
    common_params params;
    if (!common_params_parse((int) args.size(), args.data(), params, LLAMA_EXAMPLE_SPECULATIVE)) { return 2; }
    params.warmup = false;
    params.n_outputs_max = 0;
    const std::string draft_path = params.speculative.draft.mparams.path;
    if (n_parity > 0 && draft_path.empty()) { fprintf(stderr, "dspark-dump: --ds-parity needs -md DRAFTER\n"); return 2; }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    auto init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    if (!model || !ctx) { fprintf(stderr, "dspark-dump: target init failed\n"); return 1; }
    const int n_embd = llama_model_n_embd(model);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int n_threads = std::max(1, (int) std::thread::hardware_concurrency() / 2);

    // drafter (parity), created like common_speculative_init_result: ctx_other = target, n_rs_seq 0, block-sized ubatch
    llama_model * model_dft = nullptr;
    llama_context * ctx_dft = nullptr;
    std::vector<int32_t> lids;
    if (!draft_path.empty()) {
        common_params params_dft = common_base_params_to_speculative(params);
        auto mparams = common_model_params_to_llama(params_dft);
        model_dft = llama_model_load_from_file(draft_path.c_str(), mparams);
        if (!model_dft) { fprintf(stderr, "dspark-dump: cannot load drafter %s\n", draft_path.c_str()); return 1; }
        const int32_t * ids = llama_model_target_layer_ids(model_dft);
        for (uint32_t k = 0; k < llama_model_target_layer_ids_n(model_dft); ++k) { lids.push_back(ids[k]); }
    }
    if (!layers_arg.empty()) {
        std::vector<int32_t> l2;
        for (size_t p = 0; p < layers_arg.size();) {
            size_t q = layers_arg.find(',', p);
            if (q == std::string::npos) { q = layers_arg.size(); }
            l2.push_back(atoi(layers_arg.substr(p, q - p).c_str()));
            p = q + 1;
        }
        if (!lids.empty() && l2 != lids) { fprintf(stderr, "dspark-dump: --ds-layers differs from the drafter's target layers\n"); return 2; }
        lids = l2;
    }
    if (lids.empty()) { lids = {5, 23, 41, 48}; }
    for (int32_t id : lids) {
        if (id < 0 || id > llama_model_n_layer(model)) { fprintf(stderr, "dspark-dump: bad target layer %d\n", id); return 2; }
        llama_set_embeddings_layer_inp(ctx, (uint32_t) id, true);
    }
    const int nl = (int) lids.size();
    const int n_embd_enc = nl * n_embd;
    fprintf(stderr, "dspark-dump: n_embd %d, n_vocab %d, target layers", n_embd, n_vocab);
    for (int32_t id : lids) { fprintf(stderr, " %d", id); }
    fprintf(stderr, " (target n_layer %d)\n", llama_model_n_layer(model));

    std::vector<seq_t> seqs;
    if (!read_seqs(in_path, seqs)) { fprintf(stderr, "dspark-dump: cannot read %s\n", in_path.c_str()); return 1; }
    if (max_seq >= 0 && (int) seqs.size() > max_seq) { seqs.resize(max_seq); }
    const int n_ctx = (int) llama_n_ctx(ctx);
    for (size_t s = 0; s < seqs.size(); ++s) {
        if ((int) seqs[s].toks.size() > n_ctx) { fprintf(stderr, "dspark-dump: seq %zu has %zu tokens > n_ctx %d\n", s, seqs[s].toks.size(), n_ctx); return 1; }
    }

    std::vector<float> feat, top_vals;
    std::vector<int32_t> top_ids;

    if (n_parity <= 0) {
        FILE * ff = fopen((out_path + ".feat").c_str(), "wb");
        FILE * ft = fopen((out_path + ".top").c_str(), "wb");
        FILE * fi = fopen((out_path + ".idx").c_str(), "w");
        if (!ff || !ft || !fi) { fprintf(stderr, "dspark-dump: cannot write %s.*\n", out_path.c_str()); return 1; }
        fprintf(fi, "# dtype %s n_layers %d n_embd %d topk %d layers", dtype.c_str(), nl, n_embd, topk);
        for (int32_t id : lids) { fprintf(fi, " %d", id); }
        fprintf(fi, "\n");
        int64_t feat_off = 0, top_off = 0, n_tok = 0;
        const int64_t t0 = ggml_time_us();
        std::vector<uint16_t> h16;
        for (size_t s = 0; s < seqs.size(); ++s) {
            if (!decode_target(ctx, seqs[s], lids, n_embd, n_vocab, topk, n_threads, feat, top_ids, top_vals)) { return 1; }
            const int len = (int) seqs[s].toks.size();
            if (dtype == "f32") {
                fwrite(feat.data(), 4, feat.size(), ff);
            } else {
                h16.resize(feat.size());
                for (size_t i = 0; i < feat.size(); ++i) { h16[i] = dtype == "bf16" ? f32_to_bf16(feat[i]) : f32_to_f16(feat[i]); }
                fwrite(h16.data(), 2, h16.size(), ff);
            }
            const int n_top = topk > 0 ? len - std::max(0, seqs[s].n_prompt - 1) : 0;
            for (int r = 0; r < n_top; ++r) {
                fwrite(top_ids.data() + (size_t) r * topk, 4, topk, ft);
                fwrite(top_vals.data() + (size_t) r * topk, 4, topk, ft);
            }
            fprintf(fi, "%zu %lld %d %d %lld\n", s, (long long) feat_off, len, seqs[s].n_prompt, (long long) top_off);
            feat_off += len; top_off += n_top; n_tok += len;
            if ((s + 1) % 20 == 0 || s + 1 == seqs.size()) {
                const double dt = (ggml_time_us() - t0) / 1e6;
                fprintf(stderr, "dspark-dump: %zu/%zu seqs, %lld tokens, %.1f tok/s\n", s + 1, seqs.size(), (long long) n_tok, n_tok / dt);
                fflush(ff); fflush(ft); fflush(fi);
            }
        }
        fclose(ff); fclose(ft); fclose(fi);
        {
            // the target's end-of-generation ids (the mb rescue excludes them): PREFIX.eog, one id per line
            FILE * fe = fopen((out_path + ".eog").c_str(), "w");
            const llama_vocab * vocab = llama_model_get_vocab(model);
            for (llama_token t = 0; t < n_vocab; ++t) {
                if (llama_vocab_is_eog(vocab, t)) { fprintf(fe, "%d\n", t); }
            }
            fclose(fe);
        }
        llama_backend_free();
        return 0;
    }

    // ---- parity mode
    {
        auto cparams = common_context_params_to_llama(common_base_params_to_speculative(params));
        cparams.ctx_other = ctx;
        cparams.n_rs_seq = 0;
        cparams.n_ctx = llama_n_ctx(ctx);
        cparams.n_ubatch = std::max<uint32_t>(cparams.n_ubatch, 8);
        cparams.n_batch = std::max<uint32_t>(cparams.n_batch, 8);
        cparams.n_outputs_max = 0;
        cparams.n_outputs_max_per_seq = 0;
        cparams.moe_cache_size = 0;
        ctx_dft = llama_init_from_model(model_dft, cparams);
        if (!ctx_dft) { fprintf(stderr, "dspark-dump: drafter context failed\n"); return 1; }
    }
    char buf[64] = {};
    int block = 8;
    if (llama_model_meta_val_str(model_dft, "dflash.block_size", buf, sizeof(buf)) >= 0) { block = atoi(buf); }
    int rank = 0;
    if (llama_model_meta_val_str(model_dft, "dflash.dspark.markov_rank", buf, sizeof(buf)) >= 0) { rank = atoi(buf); }
    const int n_embd_dec = llama_model_n_embd(model_dft);
    if (rank <= 0 || n_embd_dec != n_embd || walk < 1 || walk >= block || rank + 1 + 2 * sample_k > n_embd_dec) {
        fprintf(stderr, "dspark-dump: not an Inkling2 DSpark drafter or bad walk/K\n"); return 2;
    }
    const llama_token mask_id = llama_vocab_mask(llama_model_get_vocab(model_dft));
    llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ false);
    llama_set_causal_attn(ctx_dft, false);

    const seq_t & s = seqs[0];
    const int len = (int) s.toks.size();
    if (!decode_target(ctx, s, lids, n_embd, n_vocab, 0, n_threads, feat, top_ids, top_vals)) { return 1; }
    {
        FILE * f = fopen((out_path + ".par.feat").c_str(), "wb");
        fwrite(feat.data(), 4, feat.size(), f);
        fclose(f);
    }
    // anchors spread evenly over the reply: a in [n_prompt, len - block]
    std::vector<int> anchors;
    const int lo = s.n_prompt, hi = len - block;
    if (hi < lo) { fprintf(stderr, "dspark-dump: reply too short for parity\n"); return 1; }
    for (int i = 0; i < n_parity; ++i) {
        const int a = lo + (int) ((int64_t) (hi - lo) * i / std::max(1, n_parity - 1));
        if (anchors.empty() || a > anchors.back()) { anchors.push_back(a); }
    }
    FILE * fp = fopen((out_path + ".par.bin").c_str(), "wb");
    const int32_t hdr[8] = { (int32_t) anchors.size(), walk, sample_k, n_vocab, n_embd, rank, block, len };
    fwrite(hdr, 4, 8, fp);
    std::vector<float> f_rnn((size_t) len * n_embd, 0.0f);
    llama_batch inj = llama_batch_init(llama_n_ubatch(ctx_dft), n_embd_enc, 1);
    const int n_ub = (int) llama_n_ubatch(ctx_dft);
    int injected = 0; // positions [0, injected) are in the drafter cache
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    common_batch brnn(ctx_dft);
    for (int a : anchors) {
        // process(): inject [injected, a) in chunks of n_ubatch, keep every row's f_rnn
        while (injected < a) {
            const int n = std::min(n_ub, a - injected);
            inj.n_tokens = n;
            for (int i = 0; i < n; ++i) {
                std::memcpy(inj.embd + (size_t) i * n_embd_enc, feat.data() + (size_t) (injected + i) * n_embd_enc, (size_t) n_embd_enc * sizeof(float));
                inj.pos[i] = injected + i;
                inj.n_seq_id[i] = 1;
                inj.seq_id[i][0] = 0;
                inj.logits[i] = false;
            }
            if (llama_decode(ctx_dft, inj) != 0) { fprintf(stderr, "dspark-dump: drafter injection failed\n"); return 1; }
            const float * h = llama_get_embeddings_nextn(ctx_dft);
            if (!h) { fprintf(stderr, "dspark-dump: no drafter h_nextn after injection\n"); return 1; }
            std::memcpy(f_rnn.data() + (size_t) injected * n_embd, h, (size_t) n * n_embd * sizeof(float));
            injected += n;
        }
        const float * ctxf = f_rnn.data() + (size_t) (a - 1) * n_embd;
        const auto run_block = [&]() -> bool {
            brnn.clear();
            for (int i = 0; i < block; ++i) {
                const int idx = brnn.add(i == 0 ? s.toks[a] : mask_id, a + i, 0, true);
                brnn.set_embd_state(idx, { ctxf, 1, (size_t) n_embd });
            }
            llama_set_dspark_walk(ctx_dft, walk);
            const int rc = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, brnn.get());
            return rc == 0;
        };
        // noise for B, layout [K, n_blocks = 1, walk]
        std::vector<float> noise((size_t) sample_k * walk);
        for (int w = 0; w < walk; ++w) {
            for (int k = 0; k < sample_k; ++k) {
                double u = std::min(std::max(uni(rng), 1e-12), 1.0 - 1e-12);
                noise[(size_t) k + (size_t) sample_k * w] = (float) -std::log(-std::log(u));
            }
        }
        // A: argmax walk with logits
        llama_set_dspark_ids_out(ctx_dft, false);
        llama_set_dspark_sample(ctx_dft, 0, 1.0f, nullptr, 0);
        if (!run_block()) { fprintf(stderr, "dspark-dump: draft A failed at %d\n", a); return 1; }
        std::vector<float> A((size_t) walk * n_vocab);
        std::vector<int32_t> Aid(walk);
        for (int i = 1; i <= walk; ++i) {
            const float * l = llama_get_logits_ith(ctx_dft, i);
            if (!l) { fprintf(stderr, "dspark-dump: no drafter logits slot %d\n", i); return 1; }
            std::memcpy(A.data() + (size_t) (i - 1) * n_vocab, l, (size_t) n_vocab * sizeof(float));
            Aid[i - 1] = (int32_t) (std::max_element(l, l + n_vocab) - l);
        }
        llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, a, -1);
        // B: sampled walk
        llama_set_dspark_ids_out(ctx_dft, true);
        llama_set_dspark_sample(ctx_dft, sample_k, 1.0f, noise.data(), noise.size());
        if (!run_block()) { fprintf(stderr, "dspark-dump: draft B failed at %d\n", a); return 1; }
        std::vector<int32_t> Bid(walk), Bc((size_t) walk * sample_k);
        std::vector<float> Bl((size_t) walk * sample_k);
        const float * h = llama_get_embeddings_nextn(ctx_dft);
        for (int i = 1; i <= walk; ++i) {
            const float * row = h + (size_t) i * n_embd + rank;
            Bid[i - 1] = (int32_t) row[0];
            for (int k = 0; k < sample_k; ++k) {
                Bc[(size_t) (i - 1) * sample_k + k] = (int32_t) row[1 + k];
                Bl[(size_t) (i - 1) * sample_k + k] = row[1 + sample_k + k];
            }
        }
        llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, a, -1);
        llama_set_dspark_sample(ctx_dft, 0, 1.0f, nullptr, 0);
        const int32_t pa = a;
        fwrite(&pa, 4, 1, fp);
        fwrite(noise.data(), 4, noise.size(), fp);
        fwrite(A.data(), 4, A.size(), fp);
        fwrite(Aid.data(), 4, Aid.size(), fp);
        fwrite(Bid.data(), 4, Bid.size(), fp);
        fwrite(Bc.data(), 4, Bc.size(), fp);
        fwrite(Bl.data(), 4, Bl.size(), fp);
        fwrite(ctxf, 4, n_embd, fp);
        fprintf(stderr, "dspark-dump: parity anchor %d: A ids %d %d %d %d, B ids %d %d %d %d\n", a,
                Aid[0], walk > 1 ? Aid[1] : -1, walk > 2 ? Aid[2] : -1, walk > 3 ? Aid[3] : -1,
                Bid[0], walk > 1 ? Bid[1] : -1, walk > 2 ? Bid[2] : -1, walk > 3 ? Bid[3] : -1);
    }
    fclose(fp);
    llama_batch_free(inj);
    llama_free(ctx_dft);
    llama_model_free(model_dft);
    llama_backend_free();
    fprintf(stderr, "dspark-dump: parity done, %zu anchors\n", anchors.size());
    return 0;
}
