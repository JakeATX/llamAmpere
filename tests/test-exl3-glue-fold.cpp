// [#136] EXL3 glue folding (GGML_CUDA_EXL3_GLUE_FOLD=1) must be bit-identical to the unfolded path.
//
// Each case builds one decoder-layer-shaped graph the way llama build_lora_mm / qwen35 do for EXL3 weights (trellis
// weights with suh/svh on src[2]/src[3]):
//   h1  = rms_norm(x) * g1                      (also read by a side output, like the GDN alpha/beta projections)
//   p   = W_in h1                               EXL3
//   o   = W_o silu(p)                           EXL3
//   r1  = o + x                                 residual ADD (operand order per case)
//   h2  = rms_norm(r1) * g2
//   FFN: fused [gate | up] = W_gu h2, a = swiglu(gu) (or swiglu_swapped)           -- a --fuse GGUF
//        or gate = W_g h2, up = W_u h2, a = swiglu_split(gate, up)                  -- separate tensors (#74 bridge)
//   out = W_d a + r1                            residual ADD
// It runs on a GPU backend twice, in forked children, with GGML_CUDA_EXL3_GLUE_FOLD=0 and =1 (read once per process),
// and compares out and the side output byte for byte. The trellis words are random: every 16-bit code decodes to a
// codebook value, so any bit pattern is a valid EXL3 tensor. Widths 1..16 are the split-glue GEMV range (the folds
// must fire), width 20 takes the GEMM path (the folds must decline). Needs a GPU backend; skips (exit 0) without one.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

struct test_case_def {
    int64_t   n_embd;
    int64_t   n_in;    // W_in outputs (= W_o inputs)
    int64_t   n_ff;
    int64_t   T;
    ggml_type type;
    bool      fused_gu;
    bool      swapped;   // fused: swiglu_swapped
    bool      res_first; // ADD(residual, mm) instead of ADD(mm, residual)
};

static std::vector<test_case_def> cases() {
    std::vector<test_case_def> out;
    for (ggml_type type : { GGML_TYPE_EXL3_2, GGML_TYPE_EXL3_3, GGML_TYPE_EXL3_4, GGML_TYPE_EXL3_5 }) {
        for (int64_t T : { 1, 2, 3, 4, 5, 8, 12, 16, 20 }) {
            out.push_back({ 512, 1024, 1536, T, type, true, false, false });
        }
        out.push_back({ 512, 1024, 1536, 4, type, true,  true,  false });
        out.push_back({ 512, 1024, 1536, 4, type, true,  false, true  });
        out.push_back({ 512, 1024, 1536, 1, type, false, false, false });
        out.push_back({ 512, 1024, 1536, 5, type, false, false, true  });
    }
    for (ggml_type type : { GGML_TYPE_EXL3_2, GGML_TYPE_EXL3_4 }) {
        for (int64_t T : { 1, 4, 8 }) {
            out.push_back({ 5120, 16384, 17408, T, type, true, false, false });
        }
    }
    return out;
}

static constexpr int64_t max_fold_T = 16; // EXL3_GEMV_MAX_T

struct counters { int64_t norm, add, gu, bridge; };

static size_t case_out_bytes(const test_case_def & c) {
    return (size_t) (2 * c.n_embd * c.T) * sizeof(float);   // out + side
}

static void run_case(ggml_backend_t backend, const test_case_def & c, std::vector<uint8_t> & blob) {
    ggml_init_params pw = { ggml_tensor_overhead() * 32, nullptr, true };
    ggml_context * ctx_w = ggml_init(pw);
    ggml_tensor * x    = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, c.n_embd, c.T);
    ggml_tensor * g1   = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, c.n_embd);
    ggml_tensor * g2   = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, c.n_embd);
    ggml_tensor * w_in = ggml_new_tensor_2d(ctx_w, c.type, c.n_embd, c.n_in);
    ggml_tensor * w_o  = ggml_new_tensor_2d(ctx_w, c.type, c.n_in, c.n_embd);
    ggml_tensor * w_gu = ggml_new_tensor_2d(ctx_w, c.type, c.n_embd, 2 * c.n_ff);
    ggml_tensor * w_g  = ggml_new_tensor_2d(ctx_w, c.type, c.n_embd, c.n_ff);
    ggml_tensor * w_u  = ggml_new_tensor_2d(ctx_w, c.type, c.n_embd, c.n_ff);
    ggml_tensor * w_d  = ggml_new_tensor_2d(ctx_w, c.type, c.n_ff, c.n_embd);
    ggml_tensor * ws[] = { w_in, w_o, w_gu, w_g, w_u, w_d };
    ggml_tensor * suh[6];
    ggml_tensor * svh[6];
    for (int k = 0; k < 6; ++k) {
        suh[k] = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, ws[k]->ne[0]);
        svh[k] = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, ws[k]->ne[1]);
    }
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::mt19937 rng(13600 + 31*c.n_embd + c.T + 131*c.type + (c.fused_gu ? 7 : 0) + (c.swapped ? 11 : 0) + (c.res_first ? 13 : 0));
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.5f, 1.5f);
    {
        std::vector<float> v(c.n_embd * c.T);
        for (float & f : v) {
            f = nd(rng);
        }
        ggml_backend_tensor_set(x, v.data(), 0, ggml_nbytes(x));
    }
    for (ggml_tensor * g : { g1, g2 }) {
        std::vector<float> v(c.n_embd);
        for (float & f : v) {
            f = 0.05f * ud(rng);
        }
        ggml_backend_tensor_set(g, v.data(), 0, ggml_nbytes(g));
    }
    for (ggml_tensor * w : ws) {
        std::vector<uint32_t> words(ggml_nbytes(w) / sizeof(uint32_t));
        for (uint32_t & u : words) {
            u = (uint32_t) rng();
        }
        ggml_backend_tensor_set(w, words.data(), 0, ggml_nbytes(w));
    }
    for (int k = 0; k < 6; ++k) {
        for (ggml_tensor * s : { suh[k], svh[k] }) {
            std::vector<float> v(s->ne[0]);
            for (float & f : v) {
                f = (rng() & 1 ? 1.0f : -1.0f) * ud(rng) * (s == svh[k] ? 0.05f : 1.0f);
            }
            ggml_backend_tensor_set(s, v.data(), 0, ggml_nbytes(s));
        }
    }

    ggml_init_params pg = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(pg);
    auto mm = [&](int k, ggml_tensor * in) {
        ggml_tensor * r = ggml_mul_mat(ctx, ws[k], in);
        r->src[2] = suh[k];
        r->src[3] = svh[k];
        return r;
    };
    auto add = [&](ggml_tensor * m, ggml_tensor * res) {
        return c.res_first ? ggml_add(ctx, res, m) : ggml_add(ctx, m, res);
    };
    const float eps = 1e-6f;
    ggml_tensor * h1   = ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), g1);
    ggml_tensor * p    = mm(0, h1);
    ggml_tensor * o    = mm(1, ggml_silu(ctx, p));
    ggml_tensor * r1   = add(o, x);
    ggml_tensor * h2   = ggml_mul(ctx, ggml_rms_norm(ctx, r1, eps), g2);
    ggml_tensor * a;
    if (c.fused_gu) {
        ggml_tensor * gu = mm(2, h2);
        a = c.swapped ? ggml_swiglu_swapped(ctx, gu) : ggml_swiglu(ctx, gu);
    } else {
        a = ggml_swiglu_split(ctx, mm(3, h2), mm(4, h2));
    }
    ggml_tensor * out  = add(mm(5, a), r1);
    ggml_tensor * side = ggml_scale(ctx, h1, 0.5f);
    ggml_set_output(out);
    ggml_set_output(side);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_build_forward_expand(gf, side);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    const size_t off = blob.size();
    blob.resize(off + ggml_nbytes(out) + ggml_nbytes(side));
    ggml_backend_tensor_get(out,  blob.data() + off, 0, ggml_nbytes(out));
    ggml_backend_tensor_get(side, blob.data() + off + ggml_nbytes(out), 0, ggml_nbytes(side));

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx_w);
}

static counters read_counters(ggml_backend_t backend) {
    typedef int64_t (*fusion_count_t)(ggml_backend_t, const char *);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    fusion_count_t fc = reg ? (fusion_count_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_fusion_count") : nullptr;
    if (!fc) {
        return { -1, -1, -1, -1 };
    }
    return { fc(backend, "exl3_fold_norm"), fc(backend, "exl3_fold_add"), fc(backend, "exl3_fold_gu"), fc(backend, "exl3_ffn_bridge") };
}

static bool run_all(std::vector<uint8_t> & blob) {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (dev == nullptr) {
        return false;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    GGML_ASSERT(backend);
    for (const test_case_def & c : cases()) {
        run_case(backend, c, blob);
    }
    const counters cnt = read_counters(backend);
    const size_t off = blob.size();
    blob.resize(off + sizeof(cnt));
    memcpy(blob.data() + off, &cnt, sizeof(cnt));
    ggml_backend_free(backend);
    return true;
}

#ifndef _WIN32
static std::vector<uint8_t> run_child(bool fold) {
    int fd[2];
    GGML_ASSERT(pipe(fd) == 0);
    const pid_t pid = fork();
    GGML_ASSERT(pid >= 0);
    if (pid == 0) {
        close(fd[0]);
        setenv("GGML_CUDA_EXL3_GLUE_FOLD", fold ? "1" : "0", 1);
        std::vector<uint8_t> blob;
        run_all(blob);
        size_t done = 0;
        while (done < blob.size()) {
            const ssize_t n = write(fd[1], blob.data() + done, blob.size() - done);
            if (n <= 0) {
                _exit(2);
            }
            done += (size_t) n;
        }
        close(fd[1]);
        _exit(0);
    }
    close(fd[1]);
    std::vector<uint8_t> blob;
    uint8_t tmp[1 << 16];
    ssize_t n;
    while ((n = read(fd[0], tmp, sizeof(tmp))) > 0) {
        blob.insert(blob.end(), tmp, tmp + n);
    }
    close(fd[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    GGML_ASSERT(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    return blob;
}
#endif

int main() {
#ifdef _WIN32
    printf("needs fork(), skipping\n");
    return 0;
#else
    const std::vector<uint8_t> ref = run_child(false);
    const std::vector<uint8_t> fol = run_child(true);
    if (ref.empty() && fol.empty()) {
        printf("no GPU backend, skipping\n");
        return 0;
    }
    constexpr size_t tail = sizeof(counters);
    GGML_ASSERT(ref.size() == fol.size() && ref.size() >= tail);
    counters cr, cf;
    memcpy(&cr, ref.data() + ref.size() - tail, tail);
    memcpy(&cf, fol.data() + fol.size() - tail, tail);

    const std::vector<test_case_def> cs = cases();
    int     n_fail = 0;
    size_t  off    = 0;
    int64_t want_norm = 0, want_add = 0, want_gu = 0;
    for (const test_case_def & c : cs) {
        const size_t sz = case_out_bytes(c);
        if (memcmp(ref.data() + off, fol.data() + off, sz) != 0) {
            size_t n_diff = 0;
            for (size_t j = 0; j < sz; j += sizeof(float)) {
                n_diff += memcmp(ref.data() + off + j, fol.data() + off + j, sizeof(float)) != 0;
            }
            printf("%s %lld/%lld/%lld T %2lld %s%s%s: differs in %zu of %zu values\n", ggml_type_name(c.type),
                   (long long) c.n_embd, (long long) c.n_in, (long long) c.n_ff, (long long) c.T,
                   c.fused_gu ? "fused-gu" : "gate+up", c.swapped ? " swapped" : "", c.res_first ? " res-first" : "",
                   n_diff, sz / sizeof(float));
            n_fail++;
        }
        off += sz;
        if (c.T <= max_fold_T) {
            want_norm += 2;
            want_add  += 2;
            want_gu   += c.fused_gu;
        }
    }
    GGML_ASSERT(off + tail == ref.size());

    printf("fold=0: norm %lld add %lld gu %lld bridge %lld\n", (long long) cr.norm, (long long) cr.add, (long long) cr.gu, (long long) cr.bridge);
    printf("fold=1: norm %lld add %lld gu %lld bridge %lld (expected norm %lld add %lld gu %lld)\n", (long long) cf.norm,
           (long long) cf.add, (long long) cf.gu, (long long) cf.bridge, (long long) want_norm, (long long) want_add, (long long) want_gu);
    if (cf.norm >= 0) {
        if (cr.norm != 0 || cr.add != 0 || cr.gu != 0) {
            printf("folds fired with GGML_CUDA_EXL3_GLUE_FOLD=0\n");
            n_fail++;
        }
        if (cf.norm != want_norm || cf.add != want_add || cf.gu != want_gu) {
            printf("fold counters differ from the expected counts\n");
            n_fail++;
        }
    }
    if (n_fail) {
        printf("FAILED (%d)\n", n_fail);
        return 1;
    }
    printf("OK: %zu cases bit-identical\n", cs.size());
    return 0;
#endif
}
