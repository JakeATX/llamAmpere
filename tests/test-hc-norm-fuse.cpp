// [HCPN] The Flash-Next (qwen4exp) hyper-connection norm fusions must be bit-identical to the unfused graph:
//   GGML_CUDA_HC_POST_NORM=1  hc_post fused with the next RMS_NORM -> MUL(gamma) [-> CPY to BF16]
//   GGML_CUDA_HC_PRE_XN=1     the gated hc_pre rebuilds xn from the norm input, the f32 xn is not written
//
// Each case builds one hyper-connection step the way qwen4exp does with LLAMA_HC_BF16_ONCE=1:
//   H     = hc_post(block_out, residual, 2*sigmoid(inject0/hc))   (post_in; else H = a + b, as after a PLE block)
//   xn    = rms_norm(H) * gamma, xb = bf16(xn) (cast)
//   lo    = silu(w_down*xb / hc), gate = w_up*lo, inject = w_inject*xb
//   mixed = hc_pre_gated(xn, gate, 1/hc)
//   H2    = hc_post(mixed, H, 2*sigmoid(inject/hc))
// with the graph allocator reusing buffers (the inputs are freed after their last reader, so xn or xb usually lands
// partly over a dead hc_post input and HC_POST_NORM must refuse), with the inputs in their own buffer (ext_in), or
// through ggml_backend_sched as llama.cpp runs it (sched: the hc_post inputs are GPU intermediates; the CUDA
// graph_optimize keeps them allocated through the fused run when HC_POST_NORM is on, so it must fuse), and runs it on
// a GPU backend in forked children, one per environment (each variable is read once per process): plain,
// GGML_CUDA_NORM_CAST, GGML_CUDA_DISABLE_FUSION, and the two fusions alone, together and with NORM_CAST. Every output
// is hashed in 16 chunks; every run must match the plain run byte for byte, and a second compute of the same graph
// (inputs set again, as llama.cpp does) must match the first. Split cases compute the graph as two views cut right
// after the hc_pre, as the scheduler runs its splits, so the only later reader of H is in the second view. Shapes are Flash-Next's (n_embd 2560, hc 4) at 12288,
// 1734, 512, 64, 33 and 16 tokens, plus n_embd 768 (the 256-thread norm) and 5120 (over the fused kernel's 4096
// columns, both fusions must refuse). The fusion counters must advance exactly where each fusion applies and stay at
// 0 where a guard has to refuse it (xn is a graph output, nothing reads H after the hc_pre, or the allocator put an
// output partly over an hc_post input). Needs a GPU backend; skips (exit 0) without one.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-impl.h" // ggml_graph_view

#include <array>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

static constexpr int64_t hc      = 4;
static constexpr int64_t hc_lr   = 64;   // the hc_down rank
static constexpr int     n_chunk = 16;   // hashes per output

struct hc_case {
    int64_t n_embd;
    int64_t n_tokens;
    bool    post_in; // the norm input is an hc_post output (else an ADD, as after a PLE block)
    bool    cast;    // the BF16 copy of xn feeds both BF16 matmuls (LLAMA_HC_BF16_ONCE, 32+ tokens)
    bool    h_out;   // H is a graph output (else only the next hc_post's residual keeps it alive)
    bool    xn_out;  // xn is a graph output: HC_PRE_XN must refuse
    bool    x_dead;  // the next hc_post adds another residual, nothing reads H after the hc_pre: HC_PRE_XN must refuse
    bool    ext_in = false; // the inputs live in their own buffer, so the allocator cannot reuse them
    bool    split  = false; // computed as two graph views cut right after the hc_pre
    bool    sched  = false; // allocated and computed by ggml_backend_sched (GPU + CPU), the inputs copied by a SCALE(1)

    bool fits() const { return n_embd <= 4096; }
    bool expect_post() const { return post_in && fits() && !x_dead; } // x_dead: the norm may run in place on H
    bool expect_xn()   const { return cast && fits() && !xn_out && !x_dead; }
    bool forbid_xn()   const { return !expect_xn(); }

    std::string name() const {
        char buf[160];
        snprintf(buf, sizeof(buf), "n_embd %4lld T %5lld %s%s%s%s%s%s%s%s", (long long) n_embd, (long long) n_tokens,
                 post_in ? "hc_post" : "add    ", cast ? " cast" : "", h_out ? " h_out" : "", xn_out ? " xn_out" : "",
                 x_dead ? " x_dead" : "", ext_in ? " ext_in" : "", split ? " split" : "", sched ? " sched" : "");
        return buf;
    }
};

static std::vector<hc_case> cases() {
    std::vector<hc_case> out;
    // the production shapes: the main pattern, H live only through the next residual, and the PLE-like ADD input
    for (int64_t T : { 12288, 1734, 512 }) {
        out.push_back({ 2560, T, true,  true, true,  false, false });
        out.push_back({ 2560, T, true,  true, false, false, false });
        out.push_back({ 2560, T, false, true, false, false, false });
        out.push_back({ 2560, T, true,  true, false, false, false, true });
        out.push_back({ 2560, T, true,  true, false, false, false, false, true });
        out.push_back({ 2560, T, true,  true, false, false, false, false, false, true });
    }
    // every variant at small shapes, both norm block sizes
    for (int64_t n_embd : { 2560, 768 }) {
        for (int64_t T : { 64, 33 }) {
            out.push_back({ n_embd, T, true,  true,  true,  false, false });
            out.push_back({ n_embd, T, true,  true,  false, false, false });
            out.push_back({ n_embd, T, true,  false, true,  false, false });
            out.push_back({ n_embd, T, false, true,  false, false, false });
            out.push_back({ n_embd, T, false, false, true,  false, false });
            out.push_back({ n_embd, T, true,  true,  true,  true,  false });
            out.push_back({ n_embd, T, true,  true,  false, false, true  });
            out.push_back({ n_embd, T, true,  true,  true,  false, false, true });
            out.push_back({ n_embd, T, true,  false, false, false, false, true });
            out.push_back({ n_embd, T, true,  true,  false, false, false, false, true });
            out.push_back({ n_embd, T, true,  true,  true,  false, false, false, true });
            out.push_back({ n_embd, T, true,  true,  false, false, true,  false, true });
            out.push_back({ n_embd, T, false, true,  false, false, false, false, true });
            out.push_back({ n_embd, T, true,  true,  false, false, false, false, false, true });
            out.push_back({ n_embd, T, true,  true,  true,  false, false, false, false, true });
            out.push_back({ n_embd, T, true,  false, true,  false, false, false, false, true });
            out.push_back({ n_embd, T, false, true,  false, false, false, false, false, true });
        }
    }
    // under 32 tokens qwen4exp has no BF16 copy
    out.push_back({ 2560, 16, true, false, true,  false, false });
    out.push_back({ 2560, 16, true, false, false, false, false });
    // over the fused kernel's column limit: both refuse
    out.push_back({ 5120, 64, true,  true, true,  false, false });
    out.push_back({ 5120, 64, false, true, false, false, false });
    return out;
}

struct env_config {
    const char * name;
    bool post_norm;
    bool pre_xn;
    bool norm_cast;
    bool no_fusion;
};

static const env_config configs[] = {
    { "plain",                   false, false, false, false },
    { "NORM_CAST",               false, false, true,  false },
    { "DISABLE_FUSION",          false, false, false, true  },
    { "HC_POST_NORM",            true,  false, false, false },
    { "HC_POST_NORM+NORM_CAST",  true,  false, true,  false },
    { "HC_PRE_XN",               false, true,  false, false },
    { "HC_PRE_XN+NORM_CAST",     false, true,  true,  false },
    { "both",                    true,  true,  false, false },
    { "both+NORM_CAST",          true,  true,  true,  false },
};
static constexpr int n_config = (int) (sizeof(configs) / sizeof(configs[0]));

// counters: hc_post_norm, hc_pre_xn (-1 when the backend has no counters)
static void read_counters(ggml_backend_t backend, int64_t cnt[2]) {
    typedef int64_t (*fusion_count_t)(ggml_backend_t, const char *);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    fusion_count_t fc = reg ? (fusion_count_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_fusion_count") : nullptr;
    cnt[0] = fc ? fc(backend, "hc_post_norm") : -1;
    cnt[1] = fc ? fc(backend, "hc_pre_xn")    : -1;
}

static uint64_t splitmix64(uint64_t & s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// uniform in [lo, hi), from the tensor's name and the case so every child sets the same values
static void fill(ggml_tensor * t, const hc_case & c, float lo, float hi) {
    uint64_t s = 0x1234567ull ^ (uint64_t) c.n_embd * 1000003ull ^ (uint64_t) c.n_tokens * 7919ull;
    for (const char * p = ggml_get_name(t); *p; ++p) {
        s = (s ^ (uint8_t) *p) * 1099511628211ull;
    }
    const int64_t n = ggml_nelements(t);
    std::vector<float> f(n);
    for (int64_t i = 0; i < n; ++i) {
        f[i] = lo + (hi - lo) * (float) ((splitmix64(s) >> 40) * (1.0 / 16777216.0));
    }
    // the first n_embd values of each H input are 0, so stream 0 of token 0 of H is exactly zero (the rsqrt(eps) row)
    for (const char * zero_head : { "block_out", "residual", "add_a", "add_b" }) {
        if (strcmp(ggml_get_name(t), zero_head) == 0) {
            for (int64_t i = 0; i < c.n_embd; ++i) {
                f[i] = 0.0f;
            }
        }
    }
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, f.data(), 0, ggml_nbytes(t));
    } else {
        GGML_ASSERT(t->type == GGML_TYPE_BF16);
        std::vector<ggml_bf16_t> b(n);
        ggml_fp32_to_bf16_row(f.data(), b.data(), n);
        ggml_backend_tensor_set(t, b.data(), 0, ggml_nbytes(t));
    }
}

static void hash_tensor(ggml_tensor * t, std::vector<uint8_t> & scratch, std::vector<uint64_t> & out) {
    const size_t nb = ggml_nbytes(t);
    scratch.resize(nb);
    ggml_backend_tensor_get(t, scratch.data(), 0, nb);
    for (int k = 0; k < n_chunk; ++k) {
        const size_t b0 = nb * k / n_chunk;
        const size_t b1 = nb * (k + 1) / n_chunk;
        uint64_t h = 1469598103934665603ull ^ (uint64_t) (b1 - b0);
        size_t i = b0;
        for (; i + 8 <= b1; i += 8) {
            uint64_t w;
            memcpy(&w, scratch.data() + i, 8);
            h = (h ^ w) * 1099511628211ull;
            h ^= h >> 29;
        }
        for (; i < b1; ++i) {
            h = (h ^ scratch[i]) * 1099511628211ull;
        }
        out.push_back(h);
    }
}

static bool overlaps(const ggml_tensor * a, const ggml_tensor * b) {
    const char * a0 = (const char *) a->data;
    const char * b0 = (const char *) b->data;
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}

// where the allocator put xn (the MUL output) and xb relative to the hc_post inputs
enum {
    ALIAS_XN_ON_RESIDUAL = 1, // xn exactly on the residual: allowed, HC_POST_NORM still fuses
    ALIAS_PARTIAL        = 2, // any other overlap with an hc_post input: HC_POST_NORM must refuse
};

// per case: [n_out][n_chunk] hashes for compute 1, the same for compute 2, the counter deltas of each compute, and
// the ALIAS_* flags
struct case_result {
    std::vector<std::string> names;
    std::vector<uint64_t>    h1;
    std::vector<uint64_t>    h2;
    int64_t                  d_post[2] = { 0, 0 };
    int64_t                  d_xn[2]   = { 0, 0 };
    uint64_t                 alias     = 0;
};

static case_result run_case(ggml_backend_t backend, ggml_backend_t backend_cpu, const hc_case & c,
                            std::vector<uint8_t> & scratch) {
    const int64_t n_embd = c.n_embd;
    const int64_t T      = c.n_tokens;
    const int64_t hc_dim = n_embd * hc;

    // weights in their own buffer, as in the model
    ggml_init_params pw = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_context * ctx_w = ggml_init(pw);
    ggml_tensor * gamma    = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32,  n_embd, hc);
    ggml_tensor * w_down   = ggml_new_tensor_2d(ctx_w, GGML_TYPE_BF16, hc_dim, hc_lr);
    ggml_tensor * w_up     = ggml_new_tensor_2d(ctx_w, GGML_TYPE_BF16, hc_lr,  hc_dim);
    ggml_tensor * w_inject = ggml_new_tensor_2d(ctx_w, GGML_TYPE_BF16, hc_dim, hc);
    ggml_set_name(gamma, "gamma");
    ggml_set_name(w_down, "w_down");
    ggml_set_name(w_up, "w_up");
    ggml_set_name(w_inject, "w_inject");
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    fill(gamma, c, 0.5f, 1.5f);
    fill(w_down, c, -0.05f, 0.05f);
    fill(w_up, c, -0.05f, 0.05f);
    fill(w_inject, c, -0.05f, 0.05f);

    ggml_init_params pg = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(pg);
    ggml_init_params pi = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_context * ctx_in = ggml_init(pi);
    ggml_context * ci = c.ext_in || c.sched ? ctx_in : ctx;

    // inputs
    ggml_tensor * block_out = ggml_new_tensor_2d(ci, GGML_TYPE_F32, n_embd, T);
    ggml_tensor * residual  = ggml_new_tensor_3d(ci, GGML_TYPE_F32, n_embd, hc, T);
    ggml_tensor * inject0   = ggml_new_tensor_2d(ci, GGML_TYPE_F32, hc, T);
    ggml_tensor * add_a     = ggml_new_tensor_3d(ci, GGML_TYPE_F32, n_embd, hc, T);
    ggml_tensor * add_b     = ggml_new_tensor_3d(ci, GGML_TYPE_F32, n_embd, hc, T);
    ggml_tensor * other     = ggml_new_tensor_3d(ci, GGML_TYPE_F32, n_embd, hc, T);
    ggml_backend_buffer_t buf_in = c.ext_in || c.sched ? ggml_backend_alloc_ctx_tensors(ctx_in, backend) : nullptr;
    ggml_set_name(block_out, "block_out");
    ggml_set_name(residual, "residual");
    ggml_set_name(inject0, "inject0");
    ggml_set_name(add_a, "add_a");
    ggml_set_name(add_b, "add_b");
    ggml_set_name(other, "other");
    std::vector<ggml_tensor *> inputs;
    if (c.post_in) {
        inputs = { block_out, residual, inject0 };
    } else {
        inputs = { add_a, add_b };
    }
    if (c.x_dead) {
        inputs.push_back(other);
    }
    for (ggml_tensor * t : inputs) {
        ggml_set_input(t);
    }

    const float inv_hc = 1.0f / (float) hc;

    // sched: the hc_post inputs are intermediates in the compute buffer, as the block output and the previous
    // residual stream are in the model (x*1 is exact)
    ggml_tensor * hp_x   = c.sched ? ggml_scale(ctx, block_out, 1.0f) : block_out;
    ggml_tensor * hp_res = c.sched ? ggml_scale(ctx, residual,  1.0f) : residual;

    ggml_tensor * H  = nullptr;
    ggml_tensor * w0 = nullptr;
    if (c.post_in) {
        w0 = ggml_scale(ctx, ggml_sigmoid(ctx, ggml_scale(ctx, inject0, inv_hc)), 2.0f);
        H  = ggml_dsv4_hc_post(ctx, hp_x, hp_res, w0, nullptr);
    } else {
        H = ggml_add(ctx, add_a, add_b);
    }
    ggml_set_name(H, "H");

    ggml_tensor * xn  = ggml_mul(ctx, ggml_rms_norm(ctx, H, 1e-6f), gamma);
    ggml_set_name(xn, "xn");
    ggml_tensor * xn2 = ggml_reshape_2d(ctx, xn, hc_dim, T);

    ggml_tensor * lo_mm = ggml_mul_mat(ctx, w_down, xn2);
    ggml_tensor * lo    = ggml_silu(ctx, ggml_scale(ctx, lo_mm, inv_hc));
    ggml_tensor * gate  = ggml_mul_mat(ctx, w_up, lo);
    ggml_set_name(lo_mm, "lo_mm");
    ggml_set_name(gate, "gate");

    ggml_tensor * mixed = ggml_dsv4_hc_pre_gated(ctx,
            ggml_reshape_3d(ctx, xn2,  n_embd, hc, T),
            ggml_reshape_3d(ctx, gate, n_embd, hc, T), inv_hc);
    ggml_set_name(mixed, "mixed");

    ggml_tensor * inject = ggml_mul_mat(ctx, w_inject, xn2);
    ggml_set_name(inject, "inject");

    ggml_tensor * xb = nullptr;
    if (c.cast) {
        xb = ggml_cast(ctx, xn2, GGML_TYPE_BF16);
        ggml_set_name(xb, "xb");
        lo_mm->src[1]  = xb;
        inject->src[1] = xb;
    }

    ggml_tensor * w2 = ggml_scale(ctx, ggml_sigmoid(ctx, ggml_scale(ctx, inject, inv_hc)), 2.0f);
    ggml_tensor * H2 = ggml_dsv4_hc_post(ctx, mixed, c.x_dead ? other : H, w2, nullptr);
    ggml_set_name(H2, "H2");

    std::vector<ggml_tensor *> outs;
    if (c.h_out) {
        outs.push_back(H);
    }
    if (c.xn_out) {
        outs.push_back(xn);
    }
    if (xb != nullptr) {
        outs.push_back(xb);
    }
    for (ggml_tensor * t : { lo_mm, gate, inject, mixed, H2 }) {
        outs.push_back(t);
    }
    for (ggml_tensor * t : outs) {
        ggml_set_output(t);
    }

    // the model's order: the mixed stream is expanded first (the block reads it), then the next residual
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, mixed);
    ggml_build_forward_expand(gf, H2);

    ggml_gallocr_t       galloc = nullptr;
    ggml_backend_sched_t sched  = nullptr;
    if (c.sched) {
        ggml_backend_t backends[2] = { backend, backend_cpu };
        sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, gf));
        GGML_ASSERT(ggml_backend_sched_get_n_splits(sched) == 1);
    } else {
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));
    }
    const int n_nodes = ggml_graph_n_nodes(gf);
    int k_split = -1;
    if (c.split) {
        for (int i = 0; i < n_nodes; ++i) {
            if (ggml_graph_node(gf, i) == mixed) {
                k_split = i + 1;
            }
        }
        GGML_ASSERT(k_split > 0 && k_split < n_nodes);
    }
    const auto compute = [&]() {
        if (sched != nullptr) {
            GGML_ASSERT(ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS);
            return;
        }
        if (k_split < 0) {
            GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
            return;
        }
        ggml_cgraph g1 = ggml_graph_view(gf, 0, k_split);
        ggml_cgraph g2 = ggml_graph_view(gf, k_split, n_nodes);
        GGML_ASSERT(ggml_backend_graph_compute(backend, &g1) == GGML_STATUS_SUCCESS);
        GGML_ASSERT(ggml_backend_graph_compute(backend, &g2) == GGML_STATUS_SUCCESS);
    };
    // the inputs may share memory with later nodes, so they are set before every compute, as llama.cpp does
    const auto set_inputs = [&]() {
        for (ggml_tensor * t : inputs) {
            const char * n = ggml_get_name(t);
            const bool weights_like = strcmp(n, "inject0") == 0;
            fill(t, c, weights_like ? -4.0f : -1.0f, weights_like ? 4.0f : 1.0f);
        }
    };

    case_result r;
    for (ggml_tensor * t : outs) {
        r.names.push_back(ggml_get_name(t));
    }
    if (c.post_in) {
        for (const ggml_tensor * in : { hp_x, hp_res, w0 }) {
            for (const ggml_tensor * o : { xn, xb }) {
                if (o == nullptr || !overlaps(o, in)) {
                    continue;
                }
                const bool exact = o == xn && in == hp_res && o->data == in->data;
                r.alias |= exact ? ALIAS_XN_ON_RESIDUAL : ALIAS_PARTIAL;
            }
        }
    }

    int64_t cnt[3][2];
    read_counters(backend, cnt[0]);
    set_inputs();
    compute();
    read_counters(backend, cnt[1]);
    for (ggml_tensor * t : outs) {
        hash_tensor(t, scratch, r.h1);
    }
    set_inputs();
    compute();
    read_counters(backend, cnt[2]);
    for (ggml_tensor * t : outs) {
        hash_tensor(t, scratch, r.h2);
    }
    for (int k = 0; k < 2; ++k) {
        r.d_post[k] = cnt[0][0] < 0 ? -1 : cnt[k + 1][0] - cnt[k][0];
        r.d_xn[k]   = cnt[0][1] < 0 ? -1 : cnt[k + 1][1] - cnt[k][1];
    }

    if (sched != nullptr) {
        ggml_backend_sched_free(sched);
    }
    if (galloc != nullptr) {
        ggml_gallocr_free(galloc);
    }
    if (buf_in != nullptr) {
        ggml_backend_buffer_free(buf_in);
    }
    ggml_free(ctx_in);
    ggml_free(ctx);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx_w);
    return r;
}

static void put_u64(std::vector<uint8_t> & blob, uint64_t v) {
    const size_t off = blob.size();
    blob.resize(off + 8);
    memcpy(blob.data() + off, &v, 8);
}

static uint64_t get_u64(const std::vector<uint8_t> & blob, size_t & off) {
    GGML_ASSERT(off + 8 <= blob.size());
    uint64_t v;
    memcpy(&v, blob.data() + off, 8);
    off += 8;
    return v;
}

// runs every case; returns false without a GPU backend
static bool run_all(std::vector<uint8_t> & blob) {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (dev == nullptr) {
        return false;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    GGML_ASSERT(backend);
    ggml_backend_t backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    GGML_ASSERT(backend_cpu);
    std::vector<uint8_t> scratch;
    for (const hc_case & c : cases()) {
        const case_result r = run_case(backend, backend_cpu, c, scratch);
        put_u64(blob, r.names.size());
        for (const uint64_t h : r.h1) {
            put_u64(blob, h);
        }
        for (const uint64_t h : r.h2) {
            put_u64(blob, h);
        }
        for (int k = 0; k < 2; ++k) {
            put_u64(blob, (uint64_t) r.d_post[k]);
            put_u64(blob, (uint64_t) r.d_xn[k]);
        }
        put_u64(blob, r.alias);
    }
    ggml_backend_free(backend_cpu);
    ggml_backend_free(backend);
    return true;
}

#ifndef _WIN32
static std::vector<uint8_t> run_child(const env_config & e) {
    int fd[2];
    GGML_ASSERT(pipe(fd) == 0);
    fflush(stdout);
    const pid_t pid = fork();
    GGML_ASSERT(pid >= 0);
    if (pid == 0) {
        close(fd[0]);
        for (const char * v : { "GGML_CUDA_HC_POST_NORM", "GGML_CUDA_HC_PRE_XN", "GGML_CUDA_NORM_CAST",
                                "GGML_CUDA_DISABLE_FUSION", "GGML_CUDA_ADD_RMS_Q8" }) {
            unsetenv(v);
        }
        if (e.post_norm) { setenv("GGML_CUDA_HC_POST_NORM",   "1", 1); }
        if (e.pre_xn)    { setenv("GGML_CUDA_HC_PRE_XN",      "1", 1); }
        if (e.norm_cast) { setenv("GGML_CUDA_NORM_CAST",      "1", 1); }
        if (e.no_fusion) { setenv("GGML_CUDA_DISABLE_FUSION", "1", 1); }
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
    const std::vector<hc_case> cs = cases();
    // names of each case's outputs, in the order run_case hashes them
    const auto out_names = [](const hc_case & c) {
        std::vector<std::string> v;
        if (c.h_out)  { v.push_back("H"); }
        if (c.xn_out) { v.push_back("xn"); }
        if (c.cast)   { v.push_back("xb"); }
        for (const char * n : { "lo_mm", "gate", "inject", "mixed", "H2" }) {
            v.push_back(n);
        }
        return v;
    };

    std::vector<std::vector<uint8_t>> blobs(n_config);
    for (int k = 0; k < n_config; ++k) {
        blobs[k] = run_child(configs[k]);
        if (blobs[k].empty()) {
            printf("no GPU backend, skipping\n");
            return 0;
        }
    }

    int n_fail = 0;
    std::vector<size_t> off(n_config, 0);
    int64_t tot_post[n_config] = {};
    int64_t tot_xn[n_config]   = {};
    int n_alias[3] = {}; // cases with xn exactly on the residual, partly over an input, neither
    for (const hc_case & c : cs) {
        const std::vector<std::string> names = out_names(c);
        std::vector<std::vector<uint64_t>> h1(n_config), h2(n_config);
        std::vector<std::array<int64_t, 2>> d_post(n_config), d_xn(n_config);
        std::vector<uint64_t> alias(n_config);
        for (int k = 0; k < n_config; ++k) {
            const uint64_t n_out = get_u64(blobs[k], off[k]);
            GGML_ASSERT(n_out == names.size());
            for (size_t i = 0; i < n_out * n_chunk; ++i) {
                h1[k].push_back(get_u64(blobs[k], off[k]));
            }
            for (size_t i = 0; i < n_out * n_chunk; ++i) {
                h2[k].push_back(get_u64(blobs[k], off[k]));
            }
            for (int j = 0; j < 2; ++j) {
                d_post[k][j] = (int64_t) get_u64(blobs[k], off[k]);
                d_xn[k][j]   = (int64_t) get_u64(blobs[k], off[k]);
                tot_post[k] += d_post[k][j] > 0 ? d_post[k][j] : 0;
                tot_xn[k]   += d_xn[k][j]   > 0 ? d_xn[k][j]   : 0;
            }
            alias[k] = get_u64(blobs[k], off[k]);
            // sched: with HC_POST_NORM the CUDA graph_optimize keeps the hc_post inputs allocated through the fused
            // run, so nothing may overlap them; every other allocation is the plain run's
            const bool kept = c.sched && configs[k].post_norm && c.post_in && c.fits();
            if (alias[k] != alias[0] && !kept) {
                printf("%s: %-22s allocation differs from plain\n", c.name().c_str(), configs[k].name);
                n_fail++;
            }
            if (kept && alias[k] != 0) {
                printf("%s: %-22s xn or xb overlaps an hc_post input that is kept allocated\n", c.name().c_str(),
                       configs[k].name);
                n_fail++;
            }
        }
        const bool partial0 = (alias[0] & ALIAS_PARTIAL) != 0;
        n_alias[partial0 ? 1 : (alias[0] & ALIAS_XN_ON_RESIDUAL) ? 0 : 2]++;

        std::string line;
        for (int k = 0; k < n_config; ++k) {
            const env_config & e = configs[k];
            for (size_t o = 0; o < names.size(); ++o) {
                int n_diff = 0;
                int n_rep  = 0;
                for (int j = 0; j < n_chunk; ++j) {
                    n_diff += h1[k][o*n_chunk + j] != h1[0][o*n_chunk + j];
                    n_rep  += h2[k][o*n_chunk + j] != h1[k][o*n_chunk + j];
                }
                if (n_diff) {
                    printf("%s: %-22s %-6s differs from plain in %d of %d chunks\n", c.name().c_str(), e.name,
                           names[o].c_str(), n_diff, n_chunk);
                    n_fail++;
                }
                if (n_rep) {
                    printf("%s: %-22s %-6s second compute differs in %d of %d chunks\n", c.name().c_str(), e.name,
                           names[o].c_str(), n_rep, n_chunk);
                    n_fail++;
                }
            }
            if (d_post[k][0] < 0) {
                continue; // no counters on this backend
            }
            // the first compute runs the fusion code; the second may replay a captured CUDA graph (0) or not (1)
            const bool partial   = (alias[k] & ALIAS_PARTIAL) != 0;
            const bool want_post = e.post_norm && c.expect_post() && !partial;
            const bool want_xn   = e.pre_xn && c.expect_xn();
            const bool may_post  = e.post_norm && c.post_in && c.fits() && !partial; // x_dead: allowed either way
            const auto check = [&](const char * what, bool want, bool may, const std::array<int64_t, 2> & d) {
                const bool ok = want ? (d[0] == 1 && (d[1] == 0 || d[1] == 1)) :
                                may  ? (d[0] == d[1] || d[1] == 0) && d[0] <= 1 && d[1] <= 1 :
                                       (d[0] == 0 && d[1] == 0);
                if (!ok) {
                    printf("%s: %-22s %s fired %lld + %lld times, expected %s\n", c.name().c_str(), e.name, what,
                           (long long) d[0], (long long) d[1], want ? "1 + (0 or 1)" : may ? "0 or 1" : "0 + 0");
                    n_fail++;
                }
            };
            check("hc_post_norm", want_post, may_post, d_post[k]);
            check("hc_pre_xn",    want_xn,   want_xn,  d_xn[k]);
            if (k > 0) {
                line += " ";
            }
            line += std::to_string(d_post[k][0]) + "/" + std::to_string(d_xn[k][0]);
        }
        printf("%-56s %s outputs %zu, fused post/xn per config: %s\n", c.name().c_str(),
               partial0 ? "xn|xb over an input  " : (alias[0] & ALIAS_XN_ON_RESIDUAL) ? "xn on the residual   " :
                                                                                       "no overlap           ",
               names.size(), line.c_str());
    }
    for (int k = 0; k < n_config; ++k) {
        GGML_ASSERT(off[k] == blobs[k].size());
        printf("config %-22s hc_post_norm %4lld  hc_pre_xn %4lld\n", configs[k].name, (long long) tot_post[k],
               (long long) tot_xn[k]);
    }
    printf("allocation: xn on the residual in %d cases, xn or xb partly over an hc_post input in %d, no overlap in %d\n",
           n_alias[0], n_alias[1], n_alias[2]);
    if (n_fail) {
        printf("FAILED (%d)\n", n_fail);
        return 1;
    }
    printf("OK: %zu cases x %d configurations bit-identical to the plain graph\n", cs.size(), n_config);
    return 0;
#endif
}
