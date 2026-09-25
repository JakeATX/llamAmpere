// [#74] EXL3 FFN light bridge (GGML_CUDA_EXL3_FFN_BRIDGE) must be bit-identical to the unfused path.
//
// Each case builds the dense FFN of a decoder layer the way llama build_ffn + build_lora_mm do for EXL3 weights:
// gate = W_gate x and up = W_up x (EXL3 trellis weights, suh/svh on src[2]/src[3]), h = swiglu_split(gate, up),
// out = W_down h. It runs on a GPU backend twice, in forked children: once with the bridge off and once with it on
// (the env var is read once per process), and the FFN outputs are compared byte for byte. The trellis words are
// random: every 16-bit code decodes to a codebook value, so any bit pattern is a valid EXL3 tensor.
// Cases cover 2..5 bits, widths 1..16 (the GEMV range) plus width 20 (the bridge must decline and fall back), both
// graph orders of the gate and up nodes, and the 27B FFN shape (5120 -> 17408 -> 5120).
// The fusion counter must stay at 0 with the bridge off; with it on every case of width <= 16 must fuse. Needs a
// GPU backend; skips (exit 0) without one.

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
    int64_t   n_ff;
    int64_t   T;
    ggml_type type;
    bool      up_first; // graph order: up node before the gate node
};

static std::vector<test_case_def> cases() {
    std::vector<test_case_def> out;
    for (ggml_type type : { GGML_TYPE_EXL3_2, GGML_TYPE_EXL3_3, GGML_TYPE_EXL3_4, GGML_TYPE_EXL3_5 }) {
        for (int64_t T : { 1, 2, 3, 4, 5, 8, 12, 16, 20 }) {
            out.push_back({ 512, 1536, T, type, false });
        }
        out.push_back({ 512, 1536, 4, type, true });
    }
    for (ggml_type type : { GGML_TYPE_EXL3_3, GGML_TYPE_EXL3_4 }) {
        for (int64_t T : { 1, 4, 8 }) {
            out.push_back({ 5120, 17408, T, type, false });
        }
    }
    return out;
}

static constexpr int64_t max_bridge_T = 16; // EXL3_GEMV_MAX_T

static void run_case(ggml_backend_t backend, const test_case_def & c, std::vector<uint8_t> & blob) {
    ggml_init_params pw = { ggml_tensor_overhead() * 16, nullptr, true };
    ggml_context * ctx_w = ggml_init(pw);
    ggml_tensor * x     = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, c.n_embd, c.T);
    ggml_tensor * w_g   = ggml_new_tensor_2d(ctx_w, c.type, c.n_embd, c.n_ff);
    ggml_tensor * w_u   = ggml_new_tensor_2d(ctx_w, c.type, c.n_embd, c.n_ff);
    ggml_tensor * w_d   = ggml_new_tensor_2d(ctx_w, c.type, c.n_ff, c.n_embd);
    ggml_tensor * suh_g = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, c.n_embd);
    ggml_tensor * svh_g = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, c.n_ff);
    ggml_tensor * suh_u = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, c.n_embd);
    ggml_tensor * svh_u = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, c.n_ff);
    ggml_tensor * suh_d = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, c.n_ff);
    ggml_tensor * svh_d = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, c.n_embd);
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::mt19937 rng(7400 + 31*c.n_embd + c.T + 131*c.type + (c.up_first ? 7 : 0));
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.5f, 1.5f);
    {
        std::vector<float> v(c.n_embd * c.T);
        for (float & f : v) {
            f = 0.05f * nd(rng);
        }
        ggml_backend_tensor_set(x, v.data(), 0, ggml_nbytes(x));
    }
    for (ggml_tensor * w : { w_g, w_u, w_d }) {
        std::vector<uint32_t> words(ggml_nbytes(w) / sizeof(uint32_t));
        for (uint32_t & u : words) {
            u = (uint32_t) rng();
        }
        ggml_backend_tensor_set(w, words.data(), 0, ggml_nbytes(w));
    }
    for (ggml_tensor * s : { suh_g, svh_g, suh_u, svh_u, suh_d, svh_d }) {
        std::vector<float> v(s->ne[0]);
        for (float & f : v) {
            f = (rng() & 1 ? 1.0f : -1.0f) * ud(rng);
        }
        ggml_backend_tensor_set(s, v.data(), 0, ggml_nbytes(s));
    }

    ggml_init_params pg = { ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(pg);
    ggml_tensor * gate = ggml_mul_mat(ctx, w_g, x);
    gate->src[2] = suh_g;
    gate->src[3] = svh_g;
    ggml_tensor * up = ggml_mul_mat(ctx, w_u, x);
    up->src[2] = suh_u;
    up->src[3] = svh_u;
    ggml_tensor * h = ggml_swiglu_split(ctx, gate, up);
    ggml_tensor * out = ggml_mul_mat(ctx, w_d, h);
    out->src[2] = suh_d;
    out->src[3] = svh_d;
    ggml_set_output(out);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    if (c.up_first) {
        ggml_build_forward_expand(gf, up);
    }
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    const size_t off = blob.size();
    blob.resize(off + ggml_nbytes(out));
    ggml_backend_tensor_get(out, blob.data() + off, 0, ggml_nbytes(out));

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx_w);
}

// the bridge counter (-1 when the backend has no counters)
static int64_t read_counter(ggml_backend_t backend) {
    typedef int64_t (*fusion_count_t)(ggml_backend_t, const char *);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    fusion_count_t fc = reg ? (fusion_count_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_fusion_count") : nullptr;
    return fc ? fc(backend, "exl3_ffn_bridge") : -1;
}

// runs every case; the blob ends with the counter. Returns false without a GPU backend.
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
    const int64_t cnt = read_counter(backend);
    const size_t off = blob.size();
    blob.resize(off + sizeof(cnt));
    memcpy(blob.data() + off, &cnt, sizeof(cnt));
    ggml_backend_free(backend);
    return true;
}

#ifndef _WIN32
// runs run_all() in a child with the bridge on or off and returns its blob (empty: no GPU backend)
static std::vector<uint8_t> run_child(bool fused) {
    int fd[2];
    GGML_ASSERT(pipe(fd) == 0);
    const pid_t pid = fork();
    GGML_ASSERT(pid >= 0);
    if (pid == 0) {
        close(fd[0]);
        if (fused) {
            setenv("GGML_CUDA_EXL3_FFN_BRIDGE", "1", 1);
        } else {
            unsetenv("GGML_CUDA_EXL3_FFN_BRIDGE");
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
    constexpr size_t tail = sizeof(int64_t);
    GGML_ASSERT(ref.size() == fus.size() && ref.size() >= tail);

    int64_t cnt_ref, cnt_fus;
    memcpy(&cnt_ref, ref.data() + ref.size() - tail, tail);
    memcpy(&cnt_fus, fus.data() + fus.size() - tail, tail);

    const std::vector<test_case_def> cs = cases();
    int     n_fail   = 0;
    int64_t n_bridge = 0;
    size_t  off      = 0;
    for (const test_case_def & c : cs) {
        const size_t sz = c.n_embd * c.T * sizeof(float);
        if (memcmp(ref.data() + off, fus.data() + off, sz) != 0) {
            size_t n_diff = 0;
            for (size_t j = 0; j < sz; j += sizeof(float)) {
                n_diff += memcmp(ref.data() + off + j, fus.data() + off + j, sizeof(float)) != 0;
            }
            printf("%s %lld -> %lld T %2lld%s: output differs in %zu of %zu values\n", ggml_type_name(c.type),
                   (long long) c.n_embd, (long long) c.n_ff, (long long) c.T, c.up_first ? " up-first" : "",
                   n_diff, sz / sizeof(float));
            n_fail++;
        }
        off += sz;
        n_bridge += c.T <= max_bridge_T;
    }
    GGML_ASSERT(off + tail == ref.size());

    printf("unfused: exl3_ffn_bridge %lld\n", (long long) cnt_ref);
    printf("fused:   exl3_ffn_bridge %lld (cases %zu, width <= %lld: %lld)\n", (long long) cnt_fus, cs.size(),
           (long long) max_bridge_T, (long long) n_bridge);
    if (cnt_fus >= 0) {
        if (cnt_ref != 0) {
            printf("bridge fired with GGML_CUDA_EXL3_FFN_BRIDGE unset\n");
            n_fail++;
        }
        if (cnt_fus != n_bridge) {
            printf("bridge fired in %lld cases, expected %lld\n", (long long) cnt_fus, (long long) n_bridge);
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
