// INKLSA feasibility probe: one long prefill of the prompt file, and for the listed cache slots (layer visits) the
// attention inputs go to disk, so attention concentration can be measured offline with the exact model logits.
//
//   K, V   : every token, f16, <dir>/K.L<il>.f16 / V.L<il>.f16, row = token, [n_head_kv*head_dim]
//   Q, R, S: sampled tokens only (pos % stride == stride - 1), f32, <dir>/Q.L<il>.f32 [n_head*head_dim],
//            R.L<il>.f32 [n_rel*n_head] (relative-bias query) and S.L<il>.f32 [n_head] (per-query sink logit)
//
// Env: INKDUMP_DIR (required), INKDUMP_LAYERS (default 5,11,17,23,29,35,41,47), INKDUMP_QSTRIDE (default 256),
//      INKDUMP_N (truncate the prompt to this many tokens, 0 = all).
// The Q rows already carry the log position scale (tau); K is post-clip, V post-norm (what the cache stores).

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

struct dump_state {
    std::string dir;
    std::set<int> layers;
    int64_t stride = 256;
    std::map<std::string, int64_t> seen;   // tokens seen per tensor name
    std::map<std::string, FILE *> files;
    std::vector<float> buf;
    std::vector<ggml_fp16_t> hbuf;
};

static bool parse_name(const char * name, std::string & kind, int & il) {
    const char * dash = strrchr(name, '-');
    if (!dash) {
        return false;
    }
    kind.assign(name, dash - name);
    if (kind != "Qcur" && kind != "Kcur" && kind != "Vcur" && kind != "Rcur" && kind != "attn_sinks") {
        return false;
    }
    // exact "<kind>-<il>" only: views of these tensors inherit names like "Kcur-11 (permuted)"
    if (!dash[1]) {
        return false;
    }
    for (const char * c = dash + 1; *c; ++c) {
        if (*c < '0' || *c > '9') {
            return false;
        }
    }
    il = atoi(dash + 1);
    return true;
}

static FILE * get_file(dump_state & st, const std::string & fname) {
    auto it = st.files.find(fname);
    if (it != st.files.end()) {
        return it->second;
    }
    FILE * f = fopen((st.dir + "/" + fname).c_str(), "wb");
    GGML_ASSERT(f);
    st.files[fname] = f;
    return f;
}

static bool dump_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto & st = *(dump_state *) user_data;
    std::string kind;
    int il = -1;
    if (!parse_name(t->name, kind, il) || !st.layers.count(il)) {
        return ask ? false : true;
    }
    if (ask) {
        return true;
    }
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    // rows = tokens: last non-1 dim
    const int64_t n_tok = kind == "attn_sinks" ? t->ne[1] : t->ne[2];
    const int64_t row   = ggml_nelements(t) / n_tok;
    st.buf.resize(ggml_nelements(t));
    if (ggml_is_contiguous(t)) {
        ggml_backend_tensor_get(t, st.buf.data(), 0, ggml_nbytes(t));
    } else {
        // strided view: fetch the spanned bytes, then pack in ne order
        std::vector<uint8_t> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        size_t o = 0;
        for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
        for (int64_t i1 = 0; i1 < t->ne[1]; ++i1)
        for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
            memcpy(&st.buf[o++], raw.data() + i0*t->nb[0] + i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3], sizeof(float));
        }
    }

    const std::string key = kind + "-" + std::to_string(il);
    int64_t & off = st.seen[key];
    const std::string sl = ".L" + std::to_string(il);
    if (kind == "Kcur" || kind == "Vcur") {
        st.hbuf.resize(st.buf.size());
        ggml_fp32_to_fp16_row(st.buf.data(), st.hbuf.data(), st.buf.size());
        FILE * f = get_file(st, std::string(kind == "Kcur" ? "K" : "V") + sl + ".f16");
        fwrite(st.hbuf.data(), sizeof(ggml_fp16_t), st.hbuf.size(), f);
    } else {
        const char * p = kind == "Qcur" ? "Q" : kind == "Rcur" ? "R" : "S";
        FILE * f = get_file(st, std::string(p) + sl + ".f32");
        FILE * fp = kind == "Qcur" ? get_file(st, "pos" + sl + ".i64") : nullptr;
        for (int64_t i = 0; i < n_tok; ++i) {
            const int64_t pos = off + i;
            // stride samples, plus a run of 64 consecutive rows before every 32K boundary (pick stability)
            if (pos % st.stride == st.stride - 1 || pos % 32768 >= 32768 - 64) {
                fwrite(st.buf.data() + i*row, sizeof(float), row, f);
                if (fp) {
                    fwrite(&pos, sizeof(pos), 1, fp);
                }
            }
        }
    }
    off += n_tok;
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    dump_state st;
    const char * dir = getenv("INKDUMP_DIR");
    if (!dir) {
        LOG_ERR("INKDUMP_DIR not set\n");
        return 1;
    }
    st.dir = dir;
    {
        const char * ls = getenv("INKDUMP_LAYERS");
        std::stringstream ss(ls ? ls : "5,11,17,23,29,35,41,47");
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            st.layers.insert(atoi(tok.c_str()));
        }
    }
    if (const char * s = getenv("INKDUMP_QSTRIDE")) {
        st.stride = atoll(s);
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval = dump_cb;
    params.cb_eval_user_data = &st;
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    if (const char * s = getenv("INKDUMP_N")) {
        const size_t n = atoll(s);
        if (n > 0 && tokens.size() > n) {
            tokens.resize(n);
        }
    }
    LOG_INF("inklsa-dump: %zu tokens, stride %lld, %zu layers -> %s\n", tokens.size(), (long long) st.stride,
            st.layers.size(), st.dir.c_str());

    const int64_t t0 = ggml_time_us();
    const size_t nb = llama_n_batch(ctx);
    for (size_t i = 0; i < tokens.size(); i += nb) {
        const size_t n = std::min(nb, tokens.size() - i);
        llama_batch batch = llama_batch_get_one(tokens.data() + i, n);
        if (llama_decode(ctx, batch)) {
            LOG_ERR("decode failed at %zu\n", i);
            return 1;
        }
        LOG_INF("inklsa-dump: %zu / %zu tokens, %.1f s\n", i + n, tokens.size(), (ggml_time_us() - t0)/1e6);
    }
    for (auto & kv : st.files) {
        fclose(kv.second);
    }
    FILE * f = fopen((st.dir + "/meta.txt").c_str(), "w");
    fprintf(f, "n_tokens %zu\nstride %lld\n", tokens.size(), (long long) st.stride);
    for (auto & kv : st.seen) {
        fprintf(f, "seen %s %lld\n", kv.first.c_str(), (long long) kv.second);
    }
    fclose(f);

    llama_perf_context_print(ctx);
    llama_backend_free();
    return 0;
}
