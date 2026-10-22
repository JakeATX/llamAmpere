// Model-level check for the hyper-connection mix paths (LLAMA_HC_FUSED): run one batch of the prompt (at most 8
// tokens, the fused path's range) and write the hc_* tensors, the layer outputs and the final norm to a directory.
// Run it once per path and compare the two directories with compare.py next to this file; the first tensor that
// moves beyond the threshold names the layer and the stage that went wrong.
//
//   LLAMA_HC_FUSED=0 llama-hc-fused-check -m model.gguf -p "..." --dump-dir /tmp/base ...
//   LLAMA_HC_FUSED=1 llama-hc-fused-check -m model.gguf -p "..." --dump-dir /tmp/hcf  ...
//   python3 compare.py /tmp/base /tmp/hcf
//
// Only the last token asks for logits (as in llama-perplexity's first half and in plain generation), so the graph
// also narrows the last layer to the output rows: the glue between that crop and the fused mix is covered too.

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
#include <string>
#include <vector>

struct dump_state {
    std::string dir;
    FILE *      index = nullptr;
    int         seq   = 0;
    bool        done  = false;  // dump the first graph only
};

static bool wanted(const char * name) {
    return strncmp(name, "hc_", 3) == 0 || strncmp(name, "l_last", 6) == 0 || strncmp(name, "ffn_out", 7) == 0 ||
        strcmp(name, "result_norm") == 0 || strcmp(name, "result_output") == 0;
}

static bool dump_cb(ggml_tensor * t, bool ask, void * ud) {
    auto * st = (dump_state *) ud;
    if (st->done || !wanted(t->name)) {
        return false;
    }
    if (ask) {
        return true;
    }
    if (t->type != GGML_TYPE_F32) {
        return true;
    }
    // read the whole span the tensor covers, then gather it in logical order (views may be strided)
    const size_t span = ggml_nbytes(t);
    std::vector<uint8_t> raw(span);
    ggml_backend_tensor_get(t, raw.data(), 0, span);
    std::vector<float> out;
    out.reserve(ggml_nelements(t));
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
                    const size_t off = i0*t->nb[0] + i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3];
                    float v;
                    memcpy(&v, raw.data() + off, sizeof(v));
                    out.push_back(v);
                }
            }
        }
    }
    const std::string file = st->dir + "/" + std::to_string(st->seq) + ".f32";
    FILE * f = fopen(file.c_str(), "wb");
    if (f) {
        fwrite(out.data(), sizeof(float), out.size(), f);
        fclose(f);
    }
    fprintf(st->index, "%d %s %s %lld %lld %lld %lld\n", st->seq, t->name, ggml_op_desc(t),
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    st->seq++;
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // --dump-dir is ours; everything else goes to the common parser
    dump_state st;
    std::vector<char *> args;
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "--dump-dir") == 0 && i + 1 < argc) {
            st.dir = argv[++i];
            continue;
        }
        args.push_back(argv[i]);
    }
    if (st.dir.empty()) {
        fprintf(stderr, "usage: %s --dump-dir DIR -m MODEL -p PROMPT [common options]\n", argv[0]);
        return 1;
    }

    common_params params;
    common_init();
    if (!common_params_parse((int) args.size(), args.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    st.index = fopen((st.dir + "/index.txt").c_str(), "w");
    if (!st.index) {
        fprintf(stderr, "cannot write %s/index.txt\n", st.dir.c_str());
        return 1;
    }

    params.cb_eval           = dump_cb;
    params.cb_eval_user_data = &st;
    params.warmup            = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    if (tokens.empty() || tokens.size() > 8) {
        LOG_ERR("%s : the prompt must be 1..8 tokens (got %zu)\n", __func__, tokens.size());
        return 1;
    }

    // HC_CHECK_ALL_OUT=1: every token asks for logits (no last-layer crop)
    const bool all_out = getenv("HC_CHECK_ALL_OUT") && atoi(getenv("HC_CHECK_ALL_OUT")) != 0;
    common_batch batch(ctx);
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.add(tokens[i], (llama_pos) i, 0, all_out || i + 1 == tokens.size());
    }
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        LOG_ERR("%s : failed to eval\n", __func__);
        return 1;
    }
    st.done = true;
    fclose(st.index);

    // top-1 of the last token, for a quick look
    const float * logits = llama_get_logits_ith(ctx, -1);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    int best = 0;
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    printf("tokens %zu, tensors dumped %d, top-1 %d (%.4f)\n", tokens.size(), st.seq, best, logits[best]);

    llama_backend_free();
    return 0;
}
