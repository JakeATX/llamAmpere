// UPSYNC 0.7 guards (llamAmpere-local type IDs and FLASH_ATTN_EXT src slots):
//  1. a weights GGUF with a GGML type 43 tensor (TURBO2_0 here, F8_E4M3 upstream) is refused at load
//  2. file_type 42 (MOSTLY_Q8_CR here, MOSTLY_F8_E4M3 upstream) without Q8_CR tensors is refused at load
//  3. FLASH_ATTN_EXT slots: kv_rows -> src[9]; SJ-KVaRN body/desc -> src[5]/src[6]; QSA rows src[8] untouched
//  4. kv_rows + SJ-KVaRN on one node is refused (ggml.c assert, and the CPU backend at compute time)
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <sys/wait.h>

static int         g_fail = 0;
static std::string g_log;

#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } else { printf("ok   " __VA_ARGS__); putchar('\n'); } } while (0)

static void log_cb(ggml_log_level, const char * text, void *) { g_log += text; }

static std::string write_gguf(const char * name, ggml_type type, int32_t file_type) {
    std::string path = std::string(P_tmpdir) + "/" + name + "-" + std::to_string(getpid()) + ".gguf";
    ggml_init_params ip = { 16*ggml_tensor_overhead() + 4096, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * t = ggml_new_tensor_2d(ctx, type, 128, 1);
    ggml_set_name(t, "token_embd.weight");
    memset(t->data, 0, ggml_nbytes(t));
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "llama");
    if (file_type >= 0) {
        gguf_set_val_u32(g, "general.file_type", (uint32_t) file_type);
    }
    gguf_add_tensor(g, t);
    gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    return path;
}

static bool load_refused(const std::string & path, const char * want) {
    g_log.clear();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * m = llama_model_load_from_file(path.c_str(), mp);
    const bool refused = m == nullptr && g_log.find(want) != std::string::npos;
    if (m) {
        llama_model_free(m);
    }
    if (!refused) {
        fprintf(stderr, "---- load log for %s ----\n%s----\n", path.c_str(), g_log.c_str());
    }
    unlink(path.c_str());
    return refused;
}

// runs fn in a child; true when the child dies abnormally (abort / non-zero exit)
template <typename F> static bool dies(F fn) {
    fflush(stdout); fflush(stderr);
    pid_t pid = fork();
    if (pid == 0) {
        freopen("/dev/null", "w", stderr);
        fn();
        _exit(0);
    }
    int st = 0;
    waitpid(pid, &st, 0);
    return !(WIFEXITED(st) && WEXITSTATUS(st) == 0);
}

struct fa_inputs {
    ggml_tensor * q, * k, * v, * mask, * rows, * body, * desc;
};

static fa_inputs make_inputs(ggml_context * ctx) {
    const int D = 64, n_q = 1, H = 1, n_kv = 64, ne3 = 1;
    fa_inputs in;
    in.q    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, n_q, H, ne3);
    in.k    = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, n_kv, H, 1);
    in.v    = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, n_kv, H, 1);
    in.mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, n_q, 1, ne3);
    in.rows = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_kv, ne3);
    in.body = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, 256);
    in.desc = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, GGML_SJKVARN_DESC_N_ENTRIES);
    return in;
}

static ggml_context * new_ctx(bool no_alloc) {
    ggml_init_params ip = { 64*ggml_tensor_overhead() + ggml_graph_overhead() + (no_alloc ? 0 : 1u << 20), nullptr, no_alloc };
    return ggml_init(ip);
}

int main() {
    llama_log_set(log_cb, nullptr);
    llama_backend_init();

    // 1. type 43 tensor in a weights file
    CHECK(load_refused(write_gguf("upsync-t43", GGML_TYPE_TURBO2_0, -1), "TURBO2_0"),
          "weights GGUF with a GGML type 43 tensor is refused (TURBO2_0 here / F8_E4M3 upstream)");
    // 2. file_type 42 without Q8_CR tensors
    CHECK(load_refused(write_gguf("upsync-ft42", GGML_TYPE_F32, 42), "Q8_CR"),
          "file_type 42 (MOSTLY_Q8_CR here / MOSTLY_F8_E4M3 upstream) without Q8_CR tensors is refused");
    CHECK(GGML_TYPE_F8_E4M3 == 60 && GGML_TYPE_TURBO2_0 == 43 && GGML_FTYPE_MOSTLY_F8_E4M3 == 32 &&
          LLAMA_FTYPE_MOSTLY_F8_E4M3 == 53 && LLAMA_FTYPE_MOSTLY_Q8_CR == 42,
          "local IDs: F8_E4M3 type 60 / ggml ftype 32 / llama ftype 53; TURBO2_0 43; MOSTLY_Q8_CR 42");

    // 3. FA src slots
    {
        ggml_context * ctx = new_ctx(true);
        fa_inputs in = make_inputs(ctx);
        ggml_tensor * fr = ggml_flash_attn_ext_rows(ctx, in.q, in.k, in.v, in.mask, in.rows, 0.125f, 0.0f, 0.0f);
        CHECK(fr->src[9] == in.rows && fr->src[5] == nullptr && fr->src[6] == nullptr && !ggml_flash_attn_ext_is_sj_kvarn(fr),
              "ggml_flash_attn_ext_rows puts kv_rows in src[9], leaves src[5]/src[6] empty");

        ggml_tensor * fs = ggml_flash_attn_ext(ctx, in.q, in.k, in.v, in.mask, 0.125f, 0.0f, 0.0f);
        ggml_tensor * sel = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 64);
        fs->src[8] = sel;
        ggml_flash_attn_ext_set_sj_kvarn(fs, in.body, in.desc, 4, 4, 64);
        CHECK(fs->src[5] == in.body && fs->src[6] == in.desc && fs->src[8] == sel && fs->src[9] == nullptr &&
              ggml_flash_attn_ext_is_sj_kvarn(fs),
              "SJ-KVaRN keeps body in src[5], desc in src[6], QSA rows in src[8]; src[9] stays empty");
        ggml_free(ctx);
    }

    // 4a. kv_rows + SJ-KVaRN refused when building the node
    CHECK(dies([] {
        ggml_context * ctx = new_ctx(true);
        fa_inputs in = make_inputs(ctx);
        ggml_tensor * fr = ggml_flash_attn_ext_rows(ctx, in.q, in.k, in.v, in.mask, in.rows, 0.125f, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_sj_kvarn(fr, in.body, in.desc, 4, 4, 64);
    }), "ggml_flash_attn_ext_set_sj_kvarn on a kv_rows node aborts");

    // 4b. a hand-built node with both is refused by the CPU backend at compute time
    CHECK(dies([] {
        ggml_context * ctx = new_ctx(false);
        fa_inputs in = make_inputs(ctx);
        for (ggml_tensor * t : { in.q, in.k, in.v, in.mask, in.body, in.desc }) {
            memset(t->data, 0, ggml_nbytes(t));
        }
        for (int i = 0; i < 64; ++i) {
            ((int32_t *) in.rows->data)[i] = i;
        }
        ggml_tensor * fr = ggml_flash_attn_ext_rows(ctx, in.q, in.k, in.v, in.mask, in.rows, 0.125f, 0.0f, 0.0f);
        fr->src[5] = in.body;
        fr->src[6] = in.desc;
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, fr);
        ggml_graph_compute_with_ctx(ctx, gf, 1);
    }), "CPU FLASH_ATTN_EXT with kv_rows (src[9]) and SJ-KVaRN (src[6]) aborts");

    // control: a plain kv_rows node computes on the CPU
    CHECK(!dies([] {
        ggml_context * ctx = new_ctx(false);
        fa_inputs in = make_inputs(ctx);
        for (ggml_tensor * t : { in.q, in.k, in.v, in.mask }) {
            memset(t->data, 0, ggml_nbytes(t));
        }
        for (int i = 0; i < 64; ++i) {
            ((int32_t *) in.rows->data)[i] = i;
        }
        ggml_tensor * fr = ggml_flash_attn_ext_rows(ctx, in.q, in.k, in.v, in.mask, in.rows, 0.125f, 0.0f, 0.0f);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, fr);
        ggml_graph_compute_with_ctx(ctx, gf, 1);
    }), "control: plain kv_rows FLASH_ATTN_EXT computes on the CPU");

    llama_backend_free();
    printf("%s (%d failure(s))\n", g_fail ? "FAILED" : "PASSED", g_fail);
    return g_fail ? 1 : 0;
}
