// Host-KV KQ mask regression (v0.6.3): with every KV layer in host memory (-ngl 0) the attention reads the host mask
// directly, so LLAMA_KQ_MASK_DEVICE's deferred fill must not be used. usage: test-kq-mask-host MODEL.gguf OUT.bin
// Runs 8 single-token decodes and one 8-token batch with flash attention on, writes the raw f32 logits to OUT.bin and
// returns 5 if any logit is non-finite. The gate runs it twice (default env, LLAMA_KQ_MASK_DEVICE=0; the env is read
// once per process) and byte-compares the two files.
#include "llama.h"
#include <cmath>
#include <cstdio>
#include <vector>

static bool run(llama_model * m, bool one_by_one, std::vector<float> & out) {
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(m));
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 64; cp.n_batch = 16; cp.n_ubatch = 16;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context * c = llama_init_from_model(m, cp);
    if (!c) { fprintf(stderr, "ctx failed\n"); return false; }
    std::vector<llama_token> tok = {3, 17, 5, 42, 9, 30, 11, 2};
    bool ok = true;
    if (one_by_one) {
        for (size_t i = 0; ok && i < tok.size(); ++i) {
            ok = llama_decode(c, llama_batch_get_one(&tok[i], 1)) == 0;
            if (ok) { const float * l = llama_get_logits(c); out.insert(out.end(), l, l + nv); }
        }
    } else {
        ok = llama_decode(c, llama_batch_get_one(tok.data(), (int) tok.size())) == 0;
        if (ok) { const float * l = llama_get_logits(c); out.insert(out.end(), l, l + nv); }
    }
    llama_free(c);
    return ok;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s MODEL.gguf OUT.bin\n", argv[0]); return 2; }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * m = llama_model_load_from_file(argv[1], mp);
    if (!m) { fprintf(stderr, "load failed %s\n", argv[1]); return 3; }
    std::vector<float> all;
    if (!run(m, true, all) || !run(m, false, all) || all.empty()) { fprintf(stderr, "decode failed\n"); return 3; }
    size_t bad = 0;
    for (float v : all) bad += !std::isfinite(v);
    FILE * f = fopen(argv[2], "wb");
    if (!f || fwrite(all.data(), sizeof(float), all.size(), f) != all.size()) return 4;
    fclose(f);
    printf("kq-mask-host: %zu logits, %zu non-finite -> %s\n", all.size(), bad, bad ? "FAIL" : "ok");
    llama_model_free(m);
    llama_backend_free();
    return bad ? 5 : 0;
}
