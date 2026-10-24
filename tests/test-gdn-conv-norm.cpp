#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

// Convolution/normalization test: each off/on case gets a new backend/process because
// the toggle and launch marker are static. The marker proves host dispatch,
// while numerical outputs prove completed computation; it is not a device PC.
struct test_case {
    int tokens, sequences, slots = 5, rollback = 0, guard = 0;
    bool in_place = false;
};

static std::vector<test_case> cases() {
    std::vector<test_case> out;
    out.push_back({1, 1, 5, 0, 4});         // in-place Q norm changes the retained mixed output
    out.push_back({1, 1, 5, 0, 5});         // in-place K norm changes the retained mixed output
    for (int t : {1, 2, 3, 4, 5, 6, 7, 8}) {
        for (int s : {1, 2}) {
            out.push_back({t, s});
        }
    }
    out.push_back({3, 1, 5, 2});             // rollback source plane
    out.push_back({8, 2, 8});                // all eight snapshot slots
    out.push_back({1, 1, 1, 0, 0, true});   // snapshot replaces its source
    out.push_back({9, 1});                   // unsupported reference width
    out.push_back({3, 1, 5, 0, 1});         // raw convolution is retained
    out.push_back({3, 1, 5, 0, 2});         // Q view has an external reader
    out.push_back({3, 1, 5, 0, 3});         // differing Q/K epsilon
    return out;
}

static void append(std::vector<float> & values, ggml_tensor * t) {
    GGML_ASSERT(ggml_is_contiguous(t));
    const size_t start = values.size();
    values.resize(start + ggml_nelements(t));
    ggml_backend_tensor_get(t, values.data() + start, 0, ggml_nbytes(t));
}

static std::vector<float> compute(ggml_backend_t backend, const test_case & c) {
    constexpr int dim = 128, heads = 16, vheads = 48, channels = dim*(2*heads + vheads), cells = 4;
    const int rows = c.slots*cells, written = std::min(c.tokens, c.slots);
    ggml_context * ctx = ggml_init({ggml_tensor_overhead()*96 + ggml_graph_overhead(), nullptr, true});
    GGML_ASSERT(ctx);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3*channels, rows);
    ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, c.sequences);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, channels, c.tokens, c.sequences);
    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, channels);
    for (ggml_tensor * leaf : {cache, ids, input, weight}) {
        ggml_set_input(leaf);
        ggml_set_output(leaf); // allocator must not turn positives into overlap fallbacks
    }
    ggml_tensor * plane = ggml_view_2d(ctx, cache, 3*channels, cells, cache->nb[1],
                                      c.rollback*cells*cache->nb[1]);
    ggml_tensor * keep = ggml_reshape_3d(ctx, ggml_get_rows(ctx, plane, ids), 3, channels, c.sequences);
    ggml_tensor * concat = ggml_concat(ctx, keep, ggml_transpose(ctx, input), 0);
    // The production CONT3+CPY window form, newest history in slot zero.
    for (int slot = 0; slot < written; ++slot) {
        ggml_tensor * win = ggml_view_3d(ctx, concat, 3, channels, c.sequences,
                                        concat->nb[1], concat->nb[2], (c.tokens-slot)*sizeof(float));
        ggml_tensor * dst = ggml_view_2d(ctx, cache, 3*channels, c.sequences, cache->nb[1], slot*cells*cache->nb[1]);
        ggml_build_forward_expand(graph, ggml_cpy(ctx, ggml_cont(ctx, win), dst));
    }
    ggml_tensor * conv = ggml_ssm_conv(ctx, concat, weight);
    ggml_tensor * mixed = ggml_silu(ctx, conv);
    const size_t stride_token = channels*sizeof(float), stride_seq = stride_token*c.tokens;
    ggml_tensor * qview = ggml_view_4d(ctx, mixed, dim, heads, c.tokens, c.sequences,
                                      dim*sizeof(float), stride_token, stride_seq, 0);
    ggml_tensor * kview = ggml_view_4d(ctx, mixed, dim, heads, c.tokens, c.sequences,
                                      dim*sizeof(float), stride_token, stride_seq, dim*heads*sizeof(float));
    ggml_tensor * qnorm = c.guard == 4 ? ggml_rms_norm_inplace(ctx, qview, 1e-6f/dim) : ggml_rms_norm(ctx, qview, 1e-6f/dim);
    ggml_tensor * knorm = c.guard == 5 ? ggml_rms_norm_inplace(ctx, kview, 1e-6f/dim) : ggml_rms_norm(ctx, kview, (c.guard == 3 ? 2e-6f : 1e-6f)/dim);
    if (c.guard == 4 || c.guard == 5) {
        GGML_ASSERT(c.tokens == 1 && c.sequences == 1);
        ggml_tensor * norm = c.guard == 4 ? qnorm : knorm;
        // Pack unused singleton strides for SCALE without changing any address or alias.
        norm->nb[2] = norm->nb[1]*norm->ne[1];
        norm->nb[3] = norm->nb[2]*norm->ne[2];
    }
    ggml_tensor * q = ggml_scale(ctx, qnorm, 1.0f/sqrtf(dim));
    ggml_tensor * k = ggml_scale(ctx, knorm, 1.0f/sqrtf(dim));
    for (ggml_tensor * t : {mixed, q, k}) {
        ggml_set_output(t);
    }
    if (c.guard == 1) { ggml_set_output(conv); }
    if (c.guard == 2) { ggml_set_output(qview); }
    ggml_build_forward_expand(graph, q);
    ggml_build_forward_expand(graph, k);
    const ggml_op expected[] = {GGML_OP_SSM_CONV, GGML_OP_UNARY, GGML_OP_VIEW, GGML_OP_RMS_NORM,
                               GGML_OP_SCALE, GGML_OP_VIEW, GGML_OP_RMS_NORM, GGML_OP_SCALE};
    int start = -1;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        if (ggml_graph_node(graph, i) == conv) { start = i; }
    }
    GGML_ASSERT(start >= 0 && start + 7 < ggml_graph_n_nodes(graph));
    for (int i = 0; i < 8; ++i) { GGML_ASSERT(ggml_graph_node(graph, start+i)->op == expected[i]); }
    // The production scheduler applies the CUDA allocation dependencies before
    // allocation; direct gallocr use can recycle the ring's hidden operands.
    ggml_backend_t cpu = ggml_backend_dev_init(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr);
    GGML_ASSERT(cpu);
    ggml_backend_t backends[] = {backend, cpu};
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    GGML_ASSERT(sched);
    for (ggml_tensor * leaf : {cache, ids, input, weight}) {
        ggml_backend_sched_set_tensor_backend(sched, leaf, backend);
    }
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_backend_sched_set_tensor_backend(sched, ggml_graph_node(graph, i), backend);
    }
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
    GGML_ASSERT(ggml_backend_sched_get_tensor_backend(sched, conv) == backend);
    std::vector<float> output;
    // Repeat the same allocated graph with changed data; reset history identically
    // in each arm. This checks replay, not full server cache-save/restore.
    for (int replay = 0; replay < 2; ++replay) {
        std::mt19937 rng(9127 + c.tokens*31 + c.sequences + replay*1009);
        std::uniform_real_distribution<float> random(-2.0f, 2.0f);
        std::vector<float> history(ggml_nelements(cache)), x(ggml_nelements(input)), w(ggml_nelements(weight));
        for (auto * values : {&history, &x, &w}) {
            for (float & f : *values) { f = random(rng); }
        }
        for (size_t j = 0; j < x.size(); ++j) {
            if (j % 37 == 0) { x[j] = 0; }
            if (j % 41 == 0) { x[j] *= 30; }
            if (j % 43 == 0) { x[j] *= 1e-12f; }
        }
        std::vector<int32_t> source(c.sequences);
        for (int s = 0; s < c.sequences; ++s) { source[s] = c.in_place ? s : 2+s; }
        ggml_backend_tensor_set(cache, history.data(), 0, ggml_nbytes(cache));
        ggml_backend_tensor_set(ids, source.data(), 0, ggml_nbytes(ids));
        ggml_backend_tensor_set(input, x.data(), 0, ggml_nbytes(input));
        ggml_backend_tensor_set(weight, w.data(), 0, ggml_nbytes(weight));
        std::vector<float> expected_cache = history;
        for (int slot = 0; slot < written; ++slot) {
            for (int s = 0; s < c.sequences; ++s) {
                for (int channel = 0; channel < channels; ++channel) {
                    for (int a = 0; a < 3; ++a) {
                        const int col = c.tokens-slot+a;
                        const float value = col < 3 ? history[((c.rollback*cells+source[s])*channels+channel)*3+col] :
                            x[((s*c.tokens+col-3)*channels)+channel];
                        expected_cache[((slot*cells+s)*channels+channel)*3+a] = value;
                    }
                }
            }
        }
        GGML_ASSERT(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
        std::vector<float> actual_cache(history.size());
        ggml_backend_tensor_get(cache, actual_cache.data(), 0, ggml_nbytes(cache));
        GGML_ASSERT(memcmp(actual_cache.data(), expected_cache.data(), ggml_nbytes(cache)) == 0);
        append(output, mixed);
        append(output, q);
        append(output, k);
        append(output, cache);
        if (c.guard == 1) { append(output, conv); }
    }
    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
    ggml_free(ctx);
    return output;
}

#ifndef _WIN32
static std::vector<float> child(const test_case & c, int enabled) {
    int pipefd[2];
    GGML_ASSERT(pipe(pipefd) == 0);
    FILE * logs = tmpfile();
    GGML_ASSERT(logs);
    const pid_t pid = fork();
    GGML_ASSERT(pid >= 0);
    if (pid == 0) {
        close(pipefd[0]);
        GGML_ASSERT(dup2(fileno(logs), STDERR_FILENO) >= 0);
        if (enabled < 0) {
            unsetenv("GGML_CUDA_GDN_CONV_NORM");
        } else {
            setenv("GGML_CUDA_GDN_CONV_NORM", enabled ? "1" : "0", 1);
        }
        setenv("GGML_CUDA_GDN_QK_NORM", "0", 1);
        setenv("GGML_CUDA_DISABLE_FUSION", "0", 1);
        setenv("GGML_VL_SF_CONVRING", "1", 1);
        setenv("GGML_CUDA_CONV_RING_PREFILL", "1", 1);
        ggml_backend_load_all();
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!dev) { _exit(77); }
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        GGML_ASSERT(backend);
        const auto values = compute(backend, c);
        ggml_backend_free(backend);
        const char * bytes = (const char *) values.data();
        size_t remaining = values.size()*sizeof(float);
        while (remaining) {
            const ssize_t n = write(pipefd[1], bytes, remaining);
            if (n <= 0) { _exit(2); }
            bytes += n; remaining -= n;
        }
        close(pipefd[1]);
        fflush(stderr);
        _exit(0);
    }
    close(pipefd[1]);
    std::vector<char> bytes;
    char buffer[65536];
    ssize_t n;
    while ((n = read(pipefd[0], buffer, sizeof(buffer))) > 0) { bytes.insert(bytes.end(), buffer, buffer+n); }
    close(pipefd[0]);
    int status;
    GGML_ASSERT(waitpid(pid, &status, 0) == pid);
    rewind(logs);
    std::string diagnostic;
    while ((n = fread(buffer, 1, sizeof(buffer), logs)) > 0) { diagnostic.append(buffer, n); }
    fclose(logs);
    fprintf(stderr, "GDN child T=%d S=%d guard=%d enabled=%d\n%s", c.tokens, c.sequences, c.guard, enabled, diagnostic.c_str());
    if (WIFEXITED(status) && WEXITSTATUS(status) == 77) { std::exit(77); }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) { GGML_ABORT("GDN child failed"); }
    const std::string marker = "[gdn-conv-norm] convolution/SiLU/QK normalization enabled";
    const bool fired = diagnostic.find(marker) != std::string::npos;
    GGML_ASSERT(fired == (enabled != 0 && c.tokens <= 8 && c.guard == 0));
    GGML_ASSERT(!bytes.empty() && bytes.size()%sizeof(float) == 0);
    std::vector<float> values(bytes.size()/sizeof(float));
    memcpy(values.data(), bytes.data(), bytes.size());
    return values;
}
#endif

int main() {
#ifdef _WIN32
    return 77;
#else
    int failures = 0;
    for (const auto & c : cases()) {
        const auto off = child(c, 0), on = child(c, 1), defaults = child(c, -1);
        GGML_ASSERT(off.size() == on.size());
        GGML_ASSERT(on.size() == defaults.size());
        GGML_ASSERT(memcmp(on.data(), defaults.data(), on.size()*sizeof(float)) == 0);
        size_t different = 0, bad = 0;
        float max_abs = 0, max_rel = 0;
        for (size_t i = 0; i < off.size(); ++i) {
            different += memcmp(&off[i], &on[i], sizeof(float)) != 0;
            const float error = fabsf(off[i]-on[i]);
            max_abs = std::max(max_abs, error);
            max_rel = std::max(max_rel, error/std::max(1e-6f, fabsf(off[i])));
            bad += !std::isfinite(off[i]) || !std::isfinite(on[i]) || error > 2e-5f + 2e-4f*fabsf(off[i]);
        }
        printf("GDN conv norm T=%d S=%d K=%d rollback=%d guard=%d in_place=%d differing=%zu bad=%zu max_abs=%g max_rel=%g\n",
               c.tokens, c.sequences, c.slots, c.rollback, c.guard, c.in_place, different, bad, max_abs, max_rel);
        failures += bad != 0 || ((c.tokens > 8 || c.guard != 0) && different != 0);
    }
    printf("GDN conv norm cases=%zu failures=%d (valid math, not full-model identity/KLD)\n", cases().size(), failures);
    return failures ? 1 : 0;
#endif
}
