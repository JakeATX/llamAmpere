// llama-kld-depth: position-resolved KL divergence of one KV-cache configuration against a base.
//
// The stock `llama-perplexity --kl-divergence` scores only the second half of every chunk and stores a
// dense uint16 log-prob record per position, which at 150K tokens x 248K vocabulary is tens of GB. This
// tool feeds ONE long sequence (the file, truncated to -c tokens) through the model in -b sized batches,
// scores every KLD_STRIDE-th position as it goes, and buckets the results by absolute position so the
// error of a quantized cache can be read off at 10K / 25K / 50K / ... depth.
//
// Base records are sparse: for each scored position the tokens whose base log-prob is above -16 (the same
// cutoff the perplexity tool applies inside its KLD sum), plus the base NLL of the actual next token and the
// base argmax. This truncated sum follows the perplexity cutoff, but is not full-vocabulary KL.
// KLD_FULL_VOCAB=1 stores dense double log-probabilities without a cutoff.
//
// All ordinary flags are the perplexity tool's (-m -f -c -b -ub -t -tb -ngl -fa -ctk -ctv ...). Extras via env:
//   KLD_MODE=write|read   write the base file (default read)
//   KLD_BASE=<path>       base record file (required)
//   KLD_STRIDE=<n>        score every n-th position (default 1); must match between write and read
//   KLD_OUT=<csv>         per-position CSV (read mode)
//   KLD_BINS=a,b,c,...    bin upper edges in tokens (default 10240,25600,51200,76800,102400,128000,153600)
//   KLD_KVARN_IDLE_EVERY=n compress adaptive KVarN after every n input tokens (default 0/off)
//   KLD_FULL_VOCAB=1     dense double log-probabilities, exact token/config validation
//   KLD_CONTRACT=<text>   required in full mode; caller model/config identity (exclude cache type)
//   KLD_MAX_TOKENS=<n>    use at most n tokens of the file (default: the context size)
//   KLD_INPUT_IDS=<path>  read the sequence as raw int32 token IDs instead of tokenizing -f (exact replay of a
//                         recorded trajectory, e.g. tokens.i32)
//   KLD_SCORE_MASK=<path> one uint8 (0/1) per input token: position i is scored only if mask[i] = 1 (position i
//                         predicts token i+1, so a trajectory's score_mask.u8 selects the generated region);
//                         must have the same length as the input before KLD_MAX_TOKENS truncation

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <cerrno>
#include <climits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

static const char * KLD_MAGIC = "KLDDPTH1";
static const float  KLD_LOGP_CUTOFF = -16.0f;

struct base_header {
    char    magic[8];
    int32_t n_vocab;
    int32_t stride;
    int32_t n_tokens;
    int32_t n_records;
};

struct sparse_record {
    int32_t pos;        // position whose NEXT token is predicted
    int32_t tok;        // that next token
    float   nll;        // -log p_base(tok)
    int32_t top;        // base argmax
    std::vector<int32_t> ids;
    std::vector<float>   logp;
    std::vector<double> full_logp;
};

struct pos_result {
    int32_t pos;
    int32_t tok;
    float   kld;
    float   p_diff;     // p_q(tok) - p_base(tok)
    float   nll;        // -log p_q(tok)
    float   nll_base;
    int     same_top;
};

static int integer_env(const char * name, int fallback, int minimum = 1) {
    const char * value = getenv(name);
    if (!value) return fallback;
    char * end = nullptr;
    errno = 0;
    const long result = strtol(value, &end, 10);
    if (errno || !*value || *end || result < minimum || result > INT_MAX) {
        LOG_ERR("%s must be an integer in [%d, %d]\n", name, minimum, INT_MAX);
        exit(1);
    }
    return (int) result;
}

static std::vector<int> parse_bins(const char * s) {
    std::vector<int> bins;
    if (!s || !*s) {
        bins = {10240, 25600, 51200, 76800, 102400, 128000, 153600};
        return bins;
    }
    std::string str(s);
    size_t start = 0;
    while (start < str.size()) {
        size_t comma = str.find(',', start);
        if (comma == std::string::npos) comma = str.size();
        bins.push_back(atoi(str.substr(start, comma - start).c_str()));
        start = comma + 1;
    }
    return bins;
}

// log-softmax pieces for one row
static void row_max_lse(const float * logits, int n_vocab, float & max_logit, int & imax, double & lse, bool full = false) {
    max_logit = logits[0];
    imax = 0;
    for (int i = 0; i < n_vocab; ++i) {
        if (!std::isfinite(logits[i])) {
            max_logit = NAN; lse = NAN; return;
        }
        if (logits[i] > max_logit) { max_logit = logits[i]; imax = i; }
    }
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_exp += full ? exp((double) logits[i] - max_logit) : expf(logits[i] - max_logit);
    }
    lse = log(sum_exp);
}

static void make_record(const float * logits, int n_vocab, int32_t pos, int32_t tok, sparse_record & rec, bool full) {
    float max_logit; int imax; double lse;
    row_max_lse(logits, n_vocab, max_logit, imax, lse, full);
    const float shift = (float) (max_logit + lse);
    rec.pos = pos;
    rec.tok = tok;
    rec.nll = shift - logits[tok];
    rec.top = imax;
    if (full) {
        const double shift_full = (double) max_logit + lse;
        rec.nll = shift_full - logits[tok];
        rec.full_logp.resize(n_vocab);
        for (int i = 0; i < n_vocab; ++i) rec.full_logp[i] = (double) logits[i] - shift_full;
        return;
    }
    rec.ids.clear();
    rec.logp.clear();
    for (int i = 0; i < n_vocab; ++i) {
        const float lp = logits[i] - shift;
        if (lp > KLD_LOGP_CUTOFF) {
            rec.ids.push_back(i);
            rec.logp.push_back(lp);
        }
    }
}

static pos_result score_record(const float * logits, int n_vocab, const sparse_record & rec, bool full) {
    float max_logit; int imax; double lse;
    row_max_lse(logits, n_vocab, max_logit, imax, lse, full);
    const float shift = (float) (max_logit + lse);
    double kld = 0.0;
    for (size_t k = 0; k < rec.ids.size(); ++k) {
        const float lp_base = rec.logp[k];
        const float lp_q    = logits[rec.ids[k]] - shift;
        kld += expf(lp_base) * (double) (lp_base - lp_q);
    }
    if (full) {
        const double shift_full = (double) max_logit + lse;
        for (int i = 0; i < n_vocab; ++i) {
            const double lp = rec.full_logp[i];
            kld += exp(lp) * (lp - ((double) logits[i] - shift_full));
        }
    }
    pos_result r;
    r.pos      = rec.pos;
    r.tok      = rec.tok;
    r.kld      = (float) kld;
    r.nll      = (full ? (double) max_logit + lse : shift) - logits[rec.tok];
    r.nll_base = rec.nll;
    r.p_diff   = expf(-r.nll) - expf(-r.nll_base);
    r.same_top = imax == rec.top ? 1 : 0;
    return r;
}

static void write_record(std::ofstream & out, const sparse_record & rec, bool full) {
    const int32_t n = full ? (int32_t) rec.full_logp.size() : (int32_t) rec.ids.size();
    out.write((const char *) &rec.pos, sizeof(rec.pos));
    out.write((const char *) &rec.tok, sizeof(rec.tok));
    out.write((const char *) &rec.nll, sizeof(rec.nll));
    out.write((const char *) &rec.top, sizeof(rec.top));
    out.write((const char *) &n, sizeof(n));
    if (full) {
        out.write((const char *) rec.full_logp.data(), n * sizeof(double));
        return;
    }
    out.write((const char *) rec.ids.data(),  n * sizeof(int32_t));
    out.write((const char *) rec.logp.data(), n * sizeof(float));
}

static bool read_record(std::ifstream & in, sparse_record & rec, int n_vocab, bool full) {
    int32_t n = 0;
    if (!in.read((char *) &rec.pos, sizeof(rec.pos))) return false;
    in.read((char *) &rec.tok, sizeof(rec.tok));
    in.read((char *) &rec.nll, sizeof(rec.nll));
    in.read((char *) &rec.top, sizeof(rec.top));
    in.read((char *) &n, sizeof(n));
    if (!in || n < 0 || n > n_vocab || rec.tok < 0 || rec.tok >= n_vocab ||
        rec.top < 0 || rec.top >= n_vocab || !std::isfinite(rec.nll)) return false;
    if (full) {
        if (n != n_vocab) return false;
        rec.full_logp.resize(n);
        in.read((char *) rec.full_logp.data(), n * sizeof(double));
        for (double lp : rec.full_logp) if (!std::isfinite(lp) || lp > 0) return false;
        return (bool) in;
    }
    rec.ids.resize(n);
    rec.logp.resize(n);
    in.read((char *) rec.ids.data(),  n * sizeof(int32_t));
    in.read((char *) rec.logp.data(), n * sizeof(float));
    for (int i = 0; i < n; ++i) {
        if (rec.ids[i] < 0 || rec.ids[i] >= n_vocab || !std::isfinite(rec.logp[i]) ||
            (i > 0 && rec.ids[i] <= rec.ids[i - 1])) return false;
    }
    return (bool) in;
}

static double quantile(std::vector<float> & v, double q) {
    if (v.empty()) return 0.0;
    const size_t k = std::min(v.size() - 1, (size_t) (q * (v.size() - 1) + 0.5));
    return v[k];
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_ctx   = 153600;
    params.n_batch = 4096;
    params.n_ubatch = 1024;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 1;
    }

    common_init();

    const char * env_mode   = getenv("KLD_MODE");
    const char * env_base   = getenv("KLD_BASE");
    const char * env_out    = getenv("KLD_OUT");
    const char * env_bins   = getenv("KLD_BINS");

    const bool full = getenv("KLD_FULL_VOCAB") && strcmp(getenv("KLD_FULL_VOCAB"), "1") == 0;
    const char * contract_env = getenv("KLD_CONTRACT");
    const std::string contract = contract_env ? contract_env : "";
    if (full && (contract.empty() || contract.size() > 65536)) {
        LOG_ERR("full mode requires KLD_CONTRACT of 1..65536 bytes\n"); return 1;
    }
    if (env_mode && strcmp(env_mode, "write") && strcmp(env_mode, "read")) {
        LOG_ERR("KLD_MODE must be write or read\n"); return 1;
    }
    const char * magic = full ? "KLDDPTH2" : KLD_MAGIC;
    const bool write_mode = env_mode && strcmp(env_mode, "write") == 0;
    if (!env_base || !*env_base) {
        LOG_ERR("KLD_BASE is required\n");
        return 1;
    }
    const int stride = integer_env("KLD_STRIDE", 1);
    const std::vector<int> bins = parse_bins(env_bins);
    const int idle_every = integer_env("KLD_KVARN_IDLE_EVERY", 0, 0);
    if (idle_every && (params.n_batch <= 0 || idle_every % params.n_batch != 0)) {
        LOG_ERR("KLD_KVARN_IDLE_EVERY must be a multiple of batch size\n"); return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (!model || !ctx) {
        LOG_ERR("failed to load model / create context\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_ctx   = llama_n_ctx(ctx);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens;
    if (const char * input_ids = getenv("KLD_INPUT_IDS")) {
        std::ifstream input(input_ids, std::ios::binary | std::ios::ate);
        const auto bytes = input ? (long long) input.tellg() : -1;
        if (bytes < 2*(long long) sizeof(llama_token) || bytes > 64ll*1024*1024 || bytes % sizeof(llama_token) != 0) {
            LOG_ERR("invalid input token file %s\n", input_ids); return 1;
        }
        tokens.resize((size_t) bytes / sizeof(llama_token));
        input.seekg(0);
        input.read((char *) tokens.data(), bytes);
        if (!input || std::any_of(tokens.begin(), tokens.end(), [n_vocab](llama_token t) { return t < 0 || t >= n_vocab; })) {
            LOG_ERR("invalid input token IDs in %s\n", input_ids); return 1;
        }
    } else {
        tokens = common_tokenize(ctx, params.prompt, add_bos, true);
    }
    std::vector<unsigned char> score_mask(tokens.size(), 1);
    if (const char * mask_path = getenv("KLD_SCORE_MASK")) {
        std::ifstream mask_in(mask_path, std::ios::binary | std::ios::ate);
        if (!mask_in || (long long) mask_in.tellg() != (long long) tokens.size()) {
            LOG_ERR("score mask %s length does not match the %zu input tokens\n", mask_path, tokens.size()); return 1;
        }
        mask_in.seekg(0);
        mask_in.read((char *) score_mask.data(), (std::streamsize) score_mask.size());
        if (!mask_in || std::any_of(score_mask.begin(), score_mask.end(), [](unsigned char x) { return x > 1; })) {
            LOG_ERR("invalid score mask %s\n", mask_path); return 1;
        }
    }
    int n_tokens = (int) tokens.size();
    const int max_tokens = integer_env("KLD_MAX_TOKENS", n_ctx);
    if (n_tokens > std::min(n_ctx, max_tokens)) {
        n_tokens = std::min(n_ctx, max_tokens);
        tokens.resize(n_tokens);
        score_mask.resize(n_tokens);
    }
    LOG_INF("%s: n_vocab=%d n_ctx=%d file tokens used=%d stride=%d mode=%s base=%s\n", __func__,
            n_vocab, n_ctx, n_tokens, stride, write_mode ? "write" : "read", env_base);
    if (n_tokens < 2) {
        LOG_ERR("not enough tokens\n");
        return 1;
    }

    std::ofstream base_out;
    std::ifstream base_in;
    base_header hdr;
    if (write_mode) {
        base_out.open(env_base, std::ios::binary);
        if (!base_out) { LOG_ERR("cannot open %s for writing\n", env_base); return 1; }
        memset(&hdr, 0, sizeof(hdr));
        memcpy(hdr.magic, magic, 8);
        hdr.n_vocab = n_vocab; hdr.stride = stride; hdr.n_tokens = n_tokens; hdr.n_records = 0;
        base_out.write((const char *) &hdr, sizeof(hdr));
    } else {
        base_in.open(env_base, std::ios::binary);
        if (!base_in) { LOG_ERR("cannot open %s\n", env_base); return 1; }
        base_in.read((char *) &hdr, sizeof(hdr));
        if (!base_in || memcmp(hdr.magic, magic, 8) != 0) { LOG_ERR("bad base file\n"); return 1; }
        if (hdr.n_vocab != n_vocab || hdr.stride != stride || hdr.n_tokens != n_tokens) {
            LOG_ERR("base mismatch: n_vocab %d/%d stride %d/%d n_tokens %d/%d\n",
                    hdr.n_vocab, n_vocab, hdr.stride, stride, hdr.n_tokens, n_tokens);
            return 1;
        }
    }

    int32_t expected_records = 0;
    for (int i = 0; i + 1 < n_tokens; ++i) { expected_records += (i % stride == 0 && score_mask[i]); }
    if (!write_mode && hdr.n_records != expected_records) {
        LOG_ERR("base record count mismatch or incomplete file\n"); return 1;
    }
    if (full) {
        const int32_t config[4] = {(int32_t) llama_n_batch(ctx), (int32_t) llama_n_ubatch(ctx),
                                  n_ctx, (int32_t) contract.size()};
        if (write_mode) {
            base_out.write((const char *) config, sizeof(config));
            base_out.write(contract.data(), contract.size());
            base_out.write((const char *) tokens.data(), tokens.size() * sizeof(llama_token));
        } else {
            int32_t saved[4];
            base_in.read((char *) saved, sizeof(saved));
            if (!base_in || memcmp(config, saved, sizeof(config))) {
                LOG_ERR("base batch/ubatch/context/contract length mismatch\n"); return 1;
            }
            std::string saved_contract(contract.size(), '\0');
            std::vector<llama_token> saved_tokens(tokens.size());
            base_in.read(&saved_contract[0], saved_contract.size());
            base_in.read((char *) saved_tokens.data(), saved_tokens.size() * sizeof(llama_token));
            if (!base_in || saved_contract != contract || saved_tokens != tokens) {
                LOG_ERR("base model/config contract or exact token stream mismatch\n"); return 1;
            }
        }
    }

    const int n_batch = params.n_batch;
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    llama_memory_clear(llama_get_memory(ctx), true);

    const int n_threads = std::max(1, params.cpuparams.n_threads > 0 ? params.cpuparams.n_threads : 8);
    std::vector<pos_result> results;
    int32_t n_records = 0;
    const auto t_start = ggml_time_us();

    for (int start = 0; start < n_tokens; start += n_batch) {
        const int end = std::min(n_tokens, start + n_batch);
        common_batch_clear(batch);
        std::vector<int> scored_idx;   // index within the batch
        for (int i = start; i < end; ++i) {
            const bool want = (i % stride == 0) && (i + 1 < n_tokens) && score_mask[i];
            common_batch_add(batch, tokens[i], i, {0}, want);
            if (want) scored_idx.push_back(i - start);
        }
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("llama_decode failed at start=%d\n", start);
            return 1;
        }
        // gather logits pointers first (they stay valid until the next decode)
        const int n_scored = (int) scored_idx.size();
        if (write_mode) {
            std::vector<sparse_record> recs(n_scored);
            std::mutex m; int counter = 0;
            auto work = [&]() {
                while (true) {
                    int k;
                    { std::lock_guard<std::mutex> lock(m); k = counter++; }
                    if (k >= n_scored) break;
                    const int bi = scored_idx[k];
                    const float * logits = llama_get_logits_ith(ctx, bi);
                    make_record(logits, n_vocab, start + bi, tokens[start + bi + 1], recs[k], full);
                }
            };
            std::vector<std::thread> th;
            for (int t = 1; t < n_threads; ++t) th.emplace_back(work);
            work();
            for (auto & t : th) t.join();
            for (auto & r : recs) {
                if (!std::isfinite(r.nll) || (full && !std::all_of(r.full_logp.begin(), r.full_logp.end(),
                        [](double lp) { return std::isfinite(lp) && lp <= 0; }))) {
                    LOG_ERR("non-finite reference distribution at pos=%d\n", r.pos); return 1;
                }
                write_record(base_out, r, full);
                ++n_records;
            }
        } else {
            std::vector<sparse_record> recs(n_scored);
            for (int k = 0; k < n_scored; ++k) {
                if (!read_record(base_in, recs[k], n_vocab, full)) { LOG_ERR("base file ended early at record %d\n", n_records + k); return 1; }
                if (recs[k].pos != start + scored_idx[k] || recs[k].tok != tokens[start + scored_idx[k] + 1]) {
                    LOG_ERR("base record pos %d != %d\n", recs[k].pos, start + scored_idx[k]); return 1;
                }
            }
            std::vector<pos_result> out(n_scored);
            std::mutex m; int counter = 0;
            auto work = [&]() {
                while (true) {
                    int k;
                    { std::lock_guard<std::mutex> lock(m); k = counter++; }
                    if (k >= n_scored) break;
                    const float * logits = llama_get_logits_ith(ctx, scored_idx[k]);
                    out[k] = score_record(logits, n_vocab, recs[k], full);
                }
            };
            std::vector<std::thread> th;
            for (int t = 1; t < n_threads; ++t) th.emplace_back(work);
            work();
            for (auto & t : th) t.join();
            for (const auto & r : out) {
                if (!std::isfinite(r.kld) || !std::isfinite(r.nll)) {
                    LOG_ERR("non-finite candidate distribution at pos=%d\n", r.pos); return 1;
                }
            }
            results.insert(results.end(), out.begin(), out.end());
            n_records += n_scored;
        }
        if (idle_every && end % idle_every == 0) {
            // All logits have been consumed. Teacher forcing has no speculative suffix or checkpoints.
            const int32_t before = llama_kvarn_sealed_end(ctx);
            const int32_t status = llama_kvarn_compress_idle(ctx, 0, end);
            const int32_t after = llama_kvarn_sealed_end(ctx);
            LOG_INF("kld_idle: accepted_end=%d status=%d sealed_before=%d sealed_after=%d\n",
                    end, status, before, after);
            if (status < 0) { LOG_ERR("KVarN idle compression failed\n"); return 1; }
        }
        if (write_mode && !base_out) { LOG_ERR("base write failed\n"); return 1; }
        const double el = (ggml_time_us() - t_start) / 1e6;
        LOG_INF("progress: %d / %d tokens, %d records, %.1f s (%.0f tok/s)\n", end, n_tokens, n_records, el, end / el);
    }

    if (write_mode) {
        hdr.n_records = n_records;
        base_out.seekp(0);
        base_out.write((const char *) &hdr, sizeof(hdr));
        base_out.close();
        if (!base_out) { LOG_ERR("base finalization failed\n"); return 1; }
        LOG_INF("wrote %d records to %s\n", n_records, env_base);
    } else {
        if (base_in.peek() != std::ifstream::traits_type::eof()) {
            LOG_ERR("unexpected trailing base records\n"); return 1;
        }
        if (env_out && *env_out) {
            std::ofstream csv(env_out);
            csv << std::setprecision(9);
            csv << "pos,tok,kld,p_diff,nll_q,nll_base,same_top\n";
            for (const auto & r : results) {
                csv << r.pos << "," << r.tok << "," << r.kld << "," << r.p_diff << "," << r.nll << "," << r.nll_base << "," << r.same_top << "\n";
            }
            csv.close();
            if (!csv) { LOG_ERR("CSV write failed\n"); return 1; }
        }
        // per-bin summary
        printf("\n== KLD by position bin (records every %d tokens, %d total) ==\n", stride, (int) results.size());
        printf("%-16s %8s %12s %10s %10s %10s %10s %10s %10s %8s %9s %9s %9s\n",
               "bin", "n", "mean_kld", "sem", "median", "p90", "p99", "p99.9", "max", "top1%", "nll_base", "nll_q", "ppl_ratio");
        int lo = 0;
        std::vector<std::pair<int,int>> ranges;
        for (int hi : bins) { ranges.push_back({lo, hi}); lo = hi; }
        ranges.push_back({0, 1 << 30});
        for (size_t b = 0; b < ranges.size(); ++b) {
            const bool all = b + 1 == ranges.size();
            std::vector<float> k; double sk = 0, sk2 = 0, snb = 0, snq = 0; long same = 0;
            for (const auto & r : results) {
                if (r.pos < ranges[b].first || r.pos >= ranges[b].second) continue;
                k.push_back(r.kld); sk += r.kld; sk2 += (double) r.kld * r.kld; snb += r.nll_base; snq += r.nll; same += r.same_top;
            }
            if (k.empty()) continue;
            std::sort(k.begin(), k.end());
            const double n = (double) k.size();
            const double mean = sk / n;
            const double var  = std::max(0.0, sk2 / n - mean * mean);
            char name[32];
            if (all) snprintf(name, sizeof(name), "ALL");
            else     snprintf(name, sizeof(name), "%d-%d", ranges[b].first, ranges[b].second);
            printf("%-16s %8zu %12.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.4f %8.3f %9.4f %9.4f %9.4f\n",
                   name, k.size(), mean, sqrt(var / n), quantile(k, 0.5), quantile(k, 0.9), quantile(k, 0.99), quantile(k, 0.999), (double) k.back(),
                   100.0 * same / n, snb / n, snq / n, exp(snq / n - snb / n));
        }
    }

    llama_batch_free(batch);
    llama_perf_context_print(ctx);
    llama_backend_free();
    return 0;
}
