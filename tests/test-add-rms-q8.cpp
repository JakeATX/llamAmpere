// [#46] Fused ADD + RMS_NORM + MUL with q8_1 cache prefill (GGML_CUDA_ADD_RMS_Q8) must be bit-identical to the
// unfused path.
//
// Each case builds the pre-norm residual block of a decoder layer, s = a + b, y = rms_norm(s) * gamma, then two
// quantized matvecs w1 * y and w2 * y, and runs it on a GPU backend twice, in forked children: once with the fusion
// off and once with it on (the env var is read once per process). The residual s, the normed y and both matvec
// outputs are compared byte for byte. The matvecs read y through the q8_1 cache, so they also cover the prefilled
// quantization (the IQ4_XS-swizzled layout and the standard one, a consumer of another weight type re-quantizes).
// n_embd 256 runs 256-thread blocks, 2304 and 5120 1024-thread blocks, and 2304 pads its q8_1 rows to 2560. Every
// case with rows > 1 has an all-zero row (b = -a), the amax == 0 branch of the quantization.
// The fusion counters must stay at 0 with the fusion off; with it on every case must fuse, and every case with
// rows <= 4 must prefill (those route to MMVQ on every architecture). Needs a GPU backend; skips (exit 0) without one.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
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
    int64_t   rows;
    ggml_type t1;
    ggml_type t2;
};

static std::vector<test_case_def> cases() {
    std::vector<test_case_def> out;
    for (int64_t n_embd : { 256, 2304, 5120 }) {
        for (int64_t rows = 1; rows <= 8; ++rows) {
            out.push_back({ n_embd, rows, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_XS });
            out.push_back({ n_embd, rows, GGML_TYPE_Q4_K,   GGML_TYPE_IQ4_XS });
            out.push_back({ n_embd, rows, GGML_TYPE_Q8_0,   GGML_TYPE_Q8_0   });
        }
    }
    return out;
}

static constexpr int64_t n_out = 512; // matvec output rows

static void append(std::vector<uint8_t> & blob, ggml_tensor * t) {
    const size_t off = blob.size();
    blob.resize(off + ggml_nbytes(t));
    ggml_backend_tensor_get(t, blob.data() + off, 0, ggml_nbytes(t));
}

static void run_case(ggml_backend_t backend, const test_case_def & c, std::vector<uint8_t> & blob) {
    // leaves in their own buffer, like model weights (a compute-buffer weight never routes to MMVQ's cache)
    ggml_init_params pw = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_context * ctx_w = ggml_init(pw);
    ggml_tensor * a     = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, c.n_embd, c.rows);
    ggml_tensor * b     = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, c.n_embd, c.rows);
    ggml_tensor * gamma = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, c.n_embd);
    ggml_tensor * w1    = ggml_new_tensor_2d(ctx_w, c.t1, c.n_embd, n_out);
    ggml_tensor * w2    = ggml_new_tensor_2d(ctx_w, c.t2, c.n_embd, n_out);
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::mt19937 rng(1000 + 17*c.n_embd + c.rows + 131*c.t1);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> va(c.n_embd * c.rows), vb(c.n_embd * c.rows), vg(c.n_embd);
    for (size_t i = 0; i < va.size(); ++i) {
        va[i] = nd(rng) * 4.0f;
        vb[i] = nd(rng);
    }
    if (c.rows > 1) {
        for (int64_t i = 0; i < c.n_embd; ++i) {
            vb[c.n_embd + i] = -va[c.n_embd + i]; // row 1 sums to exactly zero
        }
    }
    for (float & g : vg) {
        g = 1.0f + 0.25f * nd(rng);
    }
    ggml_backend_tensor_set(a,     va.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b,     vb.data(), 0, ggml_nbytes(b));
    ggml_backend_tensor_set(gamma, vg.data(), 0, ggml_nbytes(gamma));
    for (ggml_tensor * w : { w1, w2 }) {
        std::vector<float> f(c.n_embd * n_out);
        for (float & x : f) {
            x = nd(rng) * 0.05f;
        }
        std::vector<uint8_t> q(ggml_nbytes(w));
        ggml_quantize_chunk(w->type, f.data(), q.data(), 0, n_out, c.n_embd, nullptr);
        ggml_backend_tensor_set(w, q.data(), 0, q.size());
    }

    ggml_init_params pg = { ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(pg);
    ggml_tensor * s   = ggml_add(ctx, a, b);
    ggml_tensor * y   = ggml_mul(ctx, ggml_rms_norm(ctx, s, 1e-6f), gamma);
    ggml_tensor * mm1 = ggml_mul_mat(ctx, w1, y);
    ggml_tensor * mm2 = ggml_mul_mat(ctx, w2, y);
    ggml_tensor * out = ggml_add(ctx, mm1, mm2);
    for (ggml_tensor * t : { s, y, mm1, mm2, out }) {
        ggml_set_output(t);
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    for (ggml_tensor * t : { s, y, mm1, mm2 }) {
        append(blob, t);
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx_w);
}

// counters: add_rms, add_rms_q8, q8_cache_hits (-1 when the backend has no counters)
static void read_counters(ggml_backend_t backend, int64_t cnt[3]) {
    typedef int64_t (*fusion_count_t)(ggml_backend_t, const char *);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    fusion_count_t fc = reg ? (fusion_count_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_fusion_count") : nullptr;
    const char * names[3] = { "add_rms", "add_rms_q8", "q8_cache_hits" };
    for (int i = 0; i < 3; ++i) {
        cnt[i] = fc ? fc(backend, names[i]) : -1;
    }
}

// runs every case; the blob ends with the three counters. Returns false without a GPU backend.
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
    int64_t cnt[3];
    read_counters(backend, cnt);
    const size_t off = blob.size();
    blob.resize(off + sizeof(cnt));
    memcpy(blob.data() + off, cnt, sizeof(cnt));
    ggml_backend_free(backend);
    return true;
}

#ifndef _WIN32
// runs run_all() in a child with the fusion on or off and returns its blob (empty: no GPU backend)
static std::vector<uint8_t> run_child(bool fused) {
    int fd[2];
    GGML_ASSERT(pipe(fd) == 0);
    const pid_t pid = fork();
    GGML_ASSERT(pid >= 0);
    if (pid == 0) {
        close(fd[0]);
        if (fused) {
            setenv("GGML_CUDA_ADD_RMS_Q8", "1", 1);
        } else {
            unsetenv("GGML_CUDA_ADD_RMS_Q8");
        }
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
    const std::vector<uint8_t> fus = run_child(true);
    if (ref.empty() && fus.empty()) {
        printf("no GPU backend, skipping\n");
        return 0;
    }
    constexpr size_t tail = 3 * sizeof(int64_t);
    GGML_ASSERT(ref.size() == fus.size() && ref.size() >= tail);

    int64_t cnt_ref[3], cnt_fus[3];
    memcpy(cnt_ref, ref.data() + ref.size() - tail, tail);
    memcpy(cnt_fus, fus.data() + fus.size() - tail, tail);

    const std::vector<test_case_def> cs = cases();
    int n_fail = 0;
    size_t off = 0;
    for (const test_case_def & c : cs) {
        const size_t act = c.n_embd * c.rows * sizeof(float);
        const size_t mm  = n_out    * c.rows * sizeof(float);
        const size_t sizes[4] = { act, act, mm, mm };
        const char * names[4] = { "residual", "normed", "w1*y", "w2*y" };
        for (int i = 0; i < 4; ++i) {
            if (memcmp(ref.data() + off, fus.data() + off, sizes[i]) != 0) {
                size_t n_diff = 0;
                for (size_t j = 0; j < sizes[i]; j += sizeof(float)) {
                    n_diff += memcmp(ref.data() + off + j, fus.data() + off + j, sizeof(float)) != 0;
                }
                printf("n_embd %5lld rows %lld %-6s/%-6s: %-8s differs in %zu of %zu values\n", (long long) c.n_embd,
                       (long long) c.rows, ggml_type_name(c.t1), ggml_type_name(c.t2), names[i], n_diff, sizes[i] / sizeof(float));
                n_fail++;
            }
            off += sizes[i];
        }
    }
    GGML_ASSERT(off + tail == ref.size());

    int64_t min_q8 = 0;
    for (const test_case_def & c : cs) {
        min_q8 += c.rows <= 4;
    }
    printf("unfused: add_rms %lld add_rms_q8 %lld q8_cache_hits %lld\n",
           (long long) cnt_ref[0], (long long) cnt_ref[1], (long long) cnt_ref[2]);
    printf("fused:   add_rms %lld add_rms_q8 %lld q8_cache_hits %lld (cases %zu, rows <= 4: %lld)\n",
           (long long) cnt_fus[0], (long long) cnt_fus[1], (long long) cnt_fus[2], cs.size(), (long long) min_q8);
    if (cnt_fus[0] >= 0) {
        if (cnt_ref[0] != 0 || cnt_ref[1] != 0) {
            printf("fusion fired with GGML_CUDA_ADD_RMS_Q8 unset\n");
            n_fail++;
        }
        if (cnt_fus[0] != (int64_t) cs.size()) {
            printf("fusion fired in %lld of %zu cases\n", (long long) cnt_fus[0], cs.size());
            n_fail++;
        }
        if (cnt_fus[1] < min_q8) {
            printf("q8 prefill in %lld cases, expected at least %lld\n", (long long) cnt_fus[1], (long long) min_q8);
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
