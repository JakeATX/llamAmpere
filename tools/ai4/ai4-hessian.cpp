// llama-ai4-hessian (PRIVATE measurement tool, not for releases): full input Hessians H = sum_t x_t x_t^T of the
// projection weights, collected on the GPU with cuBLAS SSYRK from the eval callback (the activations never leave the
// card). One H per input group, shared by the weights that read the same activation:
//   ssm_in   = attn_qkv, attn_gate (GDN in-proj)      attn_in  = attn_q, attn_k, attn_v
//   ffn_in   = ffn_gate, ffn_up                         ffn_down = ffn_down
//   attn_out = attn_output                              ssm_out  = ssm_out
// usage: llama-ai4-hessian -m M.gguf --ids T.i32 --nctx 2048 --layers A-B --out-dir DIR [--groups g1,g2] [--max-windows N]
// T.i32: raw int32 token ids, consumed in windows of --nctx tokens (each window its own sequence, memory cleared).
// Output per group: DIR/blk.L.<group>.hess = header {char[4] "AIQH", int32 K, double ntok} + lower triangle packed
// row by row (row i holds H[i][0..i]), float32, normalized by ntok.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>

struct hgroup {
    int64_t K = 0;
    float * dH = nullptr;
    double ntok = 0;
};

static std::map<std::string, hgroup> g_groups;   // key "blk.L.group"
static int g_l0 = 0, g_l1 = -1;
static std::set<int> g_lset;   // optional explicit layer list
static std::set<std::string> g_want;
static cublasHandle_t g_cublas;
static float * g_scratch = nullptr;
static size_t g_scratch_n = 0;
static size_t g_vram_budget = 0, g_vram_used = 0;
static bool g_fail = false;

static const char * lead_group(const char * w) {
    // only the first-computed member of each group is collected (one SYRK per activation)
    static const std::map<std::string, const char *> m = {
        {"attn_qkv", "ssm_in"}, {"attn_q", "attn_in"}, {"ffn_gate", "ffn_in"},
        {"ffn_down", "ffn_down"}, {"attn_output", "attn_out"}, {"ssm_out", "ssm_out"},
    };
    auto it = m.find(w);
    return it == m.end() ? nullptr : it->second;
}

static bool parse_w(const char * name, int & L, std::string & w) {
    char buf[128];
    if (sscanf(name, "blk.%d.%127[^.].weight", &L, buf) != 2) return false;
    w = buf;
    return true;
}

static bool cb(struct ggml_tensor * t, bool ask, void *) {
    if (t->op != GGML_OP_MUL_MAT) return false;
    const ggml_tensor * w = t->src[0];
    const ggml_tensor * x = t->src[1];
    if (!w || !x || x->type != GGML_TYPE_F32) return false;
    int L; std::string wn;
    if (!parse_w(w->name, L, wn) || L < g_l0 || L > g_l1 || (!g_lset.empty() && !g_lset.count(L))) return false;
    const char * g = lead_group(wn.c_str());
    if (!g || (!g_want.empty() && !g_want.count(g))) return false;
    if (ask) return true;

    const int64_t K = x->ne[0];
    const int64_t T = x->ne[1] * x->ne[2] * x->ne[3];
    if (x->nb[0] != sizeof(float) || x->nb[1] != (size_t) K * sizeof(float) || !ggml_is_contiguous(x)) {
        fprintf(stderr, "non-contiguous activation for %s, skipped\n", w->name);
        return true;
    }
    const std::string key = "blk." + std::to_string(L) + "." + g;
    hgroup & h = g_groups[key];
    if (!h.dH) {
        const size_t bytes = (size_t) K * K * sizeof(float);
        if (g_vram_used + bytes > g_vram_budget) {
            fprintf(stderr, "VRAM budget exceeded at %s (%zu MiB used)\n", key.c_str(), g_vram_used >> 20);
            g_fail = true;
            return false;
        }
        if (cudaMalloc(&h.dH, bytes) != cudaSuccess) { fprintf(stderr, "cudaMalloc H %s failed\n", key.c_str()); g_fail = true; return false; }
        cudaMemset(h.dH, 0, bytes);
        g_vram_used += bytes;
        h.K = K;
    }
    if (h.K != K) { fprintf(stderr, "K mismatch %s\n", key.c_str()); g_fail = true; return false; }
    const float * dx = (const float *) x->data;
    cudaDeviceSynchronize();
    if (ggml_backend_buffer_is_host(x->buffer)) {
        const size_t n = (size_t) K * T;
        if (n > g_scratch_n) { cudaFree(g_scratch); cudaMalloc(&g_scratch, n * sizeof(float)); g_scratch_n = n; }
        cudaMemcpy(g_scratch, x->data, n * sizeof(float), cudaMemcpyHostToDevice);
        dx = g_scratch;
    }
    const float one = 1.0f;
    // column-major: X is K x T (ld K); H(lower) += X X^T
    if (cublasSsyrk(g_cublas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, (int) K, (int) T, &one, dx, (int) K, &one, h.dH, (int) K)
            != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "syrk failed %s\n", key.c_str()); g_fail = true;
    }
    cudaDeviceSynchronize();
    h.ntok += (double) T;
    return true;
}

int main(int argc, char ** argv) {
    std::string fm, fids, odir;
    int nctx = 2048, maxw = -1;
    double budget_gib = 8.0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); } return std::string(argv[++i]); };
        if (a == "-m") fm = next();
        else if (a == "--ids") fids = next();
        else if (a == "--nctx") nctx = atoi(next().c_str());
        else if (a == "--layers") {
            std::string s = next();
            if (s.find(',') != std::string::npos) {
                size_t p = 0; g_l0 = 1 << 30; g_l1 = -1;
                while (p <= s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); const int v = atoi(s.substr(p, q - p).c_str()); g_lset.insert(v); g_l0 = std::min(g_l0, v); g_l1 = std::max(g_l1, v); p = q + 1; }
            } else if (sscanf(s.c_str(), "%d-%d", &g_l0, &g_l1) != 2) { g_l0 = g_l1 = atoi(s.c_str()); }
        }
        else if (a == "--out-dir") odir = next();
        else if (a == "--groups") { std::string s = next(); size_t p = 0; while (p <= s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); g_want.insert(s.substr(p, q - p)); p = q + 1; } }
        else if (a == "--max-windows") maxw = atoi(next().c_str());
        else if (a == "--vram-gib") budget_gib = atof(next().c_str());
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (fm.empty() || fids.empty() || odir.empty() || g_l1 < g_l0) {
        fprintf(stderr, "usage: %s -m M.gguf --ids T.i32 --layers A-B --out-dir DIR [--nctx 2048] [--groups ...] [--max-windows N] [--vram-gib 8]\n", argv[0]);
        return 2;
    }
    g_vram_budget = (size_t) (budget_gib * 1024.0 * 1024.0 * 1024.0);
    std::vector<int32_t> ids;
    {
        FILE * f = fopen(fids.c_str(), "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", fids.c_str()); return 1; }
        fseek(f, 0, SEEK_END); long n = ftell(f) / 4; fseek(f, 0, SEEK_SET);
        ids.resize(n);
        if (fread(ids.data(), 4, n, f) != (size_t) n) { fprintf(stderr, "short read\n"); return 1; }
        fclose(f);
    }
    int nw = (int) (ids.size() / nctx);
    if (maxw > 0 && maxw < nw) nw = maxw;

    cudaSetDevice(0);
    if (cublasCreate(&g_cublas) != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasCreate failed\n"); return 1; }
    cublasSetMathMode(g_cublas, CUBLAS_DEFAULT_MATH);   // full fp32, no TF32

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(fm.c_str(), mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = nctx; cp.n_batch = nctx; cp.n_ubatch = 512; cp.n_seq_max = 1;
    cp.cb_eval = cb; cp.cb_eval_user_data = nullptr;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 1; }

    const auto t0 = std::chrono::steady_clock::now();
    for (int w = 0; w < nw && !g_fail; ++w) {
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_batch b = llama_batch_get_one(ids.data() + (size_t) w * nctx, nctx);
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed at window %d\n", w); return 1; }
        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "window %d/%d  %.0f s  groups %zu  vram %zu MiB\n", w + 1, nw, el, g_groups.size(), g_vram_used >> 20);
    }
    if (g_fail) return 1;

    std::vector<float> h, pk;
    for (auto & kv : g_groups) {
        hgroup & g = kv.second;
        const int64_t K = g.K;
        h.resize((size_t) K * K);
        cudaMemcpy(h.data(), g.dH, h.size() * sizeof(float), cudaMemcpyDeviceToHost);
        pk.resize((size_t) K * (K + 1) / 2);
        const double inv = 1.0 / g.ntok;
        size_t o = 0;
        // column-major lower: element (i, j), i >= j at [i + j*K]; write row i: H[i][0..i]
        for (int64_t i = 0; i < K; ++i) for (int64_t j = 0; j <= i; ++j) pk[o++] = (float) (h[(size_t) i + (size_t) j * K] * inv);
        const std::string fn = odir + "/" + kv.first + ".hess";
        const std::string tmp = fn + ".tmp";
        FILE * f = fopen(tmp.c_str(), "wb");
        if (!f) { fprintf(stderr, "cannot write %s\n", tmp.c_str()); return 1; }
        int32_t K32 = (int32_t) K;
        fwrite("AIQH", 1, 4, f); fwrite(&K32, 4, 1, f); fwrite(&g.ntok, 8, 1, f);
        fwrite(pk.data(), sizeof(float), pk.size(), f);
        if (fclose(f) != 0 || rename(tmp.c_str(), fn.c_str()) != 0) { fprintf(stderr, "write %s failed\n", fn.c_str()); return 1; }
        fprintf(stderr, "wrote %s K=%lld ntok=%.0f\n", fn.c_str(), (long long) K, g.ntok);
        cudaFree(g.dH);
    }
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
