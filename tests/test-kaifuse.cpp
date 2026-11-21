// [KAIFUSE] Exactness of the two KAIFUSE changes on the CUDA backend.
//
// R1 (shared AI+ activation prep): graphs where one activation feeds several AI+ / AIF matmuls (the GDN qkv + z,
// attention q/k/v and FFN gate + up shapes, same type and mixed types) at decode and verify widths. Run in two child
// processes, LLAMA_KAIFUSE=0 (every matmul preps its own input, as before) and the default (one prep per input group);
// every output must be bit-identical, the default must report prep cache hits and the kill switch none, and
// default preps + hits must equal the kill switch's preps. Each graph runs on the graph allocator and is computed
// several times with new inputs (CUDA graph capture and replay, new eval epochs); one case interleaves five
// activations so cache entries are evicted within an eval.
//
// R2 (draft shortlist on a fractional AI+ head): MUL_MAT_ID over one-row experts of an AIF head (the graph the draft
// shortlist builds), with one id list broadcast to every output (row stride 0) and with one list per output, against
// the full-head MUL_MAT gathered on the host: bit-identical.
//
// Exit 0 = pass, 1 = fail, 77 = no CUDA device.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cinttypes>
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

// bytes of one 16-row weight block (fp16 row scales first, then code words) and the K it covers
static size_t block_bytes(ggml_type t) {
    switch (t) {
        case GGML_TYPE_AIF25_U:      return 672;
        case GGML_TYPE_AIF25_U2:     return 3232;
        case GGML_TYPE_AIF25_U2_512: return 2592;
        case GGML_TYPE_AIF175_U2:    return 1824;
        default: {
            const int v2 = ggml_ai4v2_chunks(t);
            return (size_t) (32 + (v2 ? v2 : 1)*256*ggml_ai4_rate(t));
        }
    }
}

static void fill_weights(ggml_tensor * w, std::mt19937 & rng) {
    std::uniform_real_distribution<float> scale(0.002f, 0.02f);
    std::vector<uint8_t> buf(ggml_nbytes(w));
    const size_t bb = block_bytes(w->type);
    for (size_t b = 0; b + bb <= buf.size(); b += bb) {
        ggml_fp16_t * d = (ggml_fp16_t *) (buf.data() + b);
        for (int i = 0; i < 16; ++i) {
            d[i] = ggml_fp32_to_fp16(scale(rng));
        }
        uint32_t * c = (uint32_t *) (buf.data() + b + 32);
        for (size_t i = 0; i < (bb - 32)/4; ++i) {
            c[i] = (uint32_t) rng();
        }
    }
    ggml_backend_tensor_set(w, buf.data(), 0, buf.size());
}

static void fill_floats(ggml_tensor * x, std::mt19937 & rng) {
    std::uniform_real_distribution<float> act(-1.0f, 1.0f);
    std::vector<float> v(ggml_nelements(x));
    for (float & f : v) {
        f = act(rng);
    }
    ggml_backend_tensor_set(x, v.data(), 0, ggml_nbytes(x));
}

static int64_t counter(ggml_backend_t backend, const char * name) {
    typedef int64_t (*fusion_count_t)(ggml_backend_t, const char *);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    fusion_count_t fc = reg ? (fusion_count_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_fusion_count") : nullptr;
    return fc ? fc(backend, name) : -1;
}

struct group_case {
    std::vector<ggml_type> types;   // one matmul per entry on each activation
    std::vector<int64_t>   rows;    // their output rows
    int64_t                k;
    int64_t                n;
    int                    n_act = 1; // > 1: that many activations, their consumers interleaved (cache eviction in one eval)
    std::vector<int>       order = {}; // optional consumer order (activation index per matmul, types used round-robin)
};

static constexpr int N_EVALS = 4;   // repeated evals of one graph with new inputs: CUDA graph capture + replay, epochs

// One graph on the graph allocator: x0_a -> scale (a computed node, as a norm output is) -> every matmul of the group,
// for each activation a. Computed N_EVALS times with new x0 contents each time; appends every eval's outputs.
// GROUP_UNSUPPORTED (nothing written) when the backend does not run one of the matmuls; GROUP_FAILED on an error.
enum group_result { GROUP_RAN, GROUP_UNSUPPORTED, GROUP_FAILED };

static group_result run_group(ggml_backend_t backend, const group_case & c, uint32_t seed, std::vector<uint8_t> & blob) {
    ggml_init_params wp = { 64*ggml_tensor_overhead(), nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    ggml_init_params gp = { 256*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * gctx = ggml_init(gp);
    std::vector<ggml_tensor *> ws, x0s, xs, ys;
    for (size_t i = 0; i < c.types.size(); ++i) {
        ws.push_back(ggml_new_tensor_2d(wctx, c.types[i], c.k, c.rows[i]));
    }
    for (int a = 0; a < c.n_act; ++a) {
        x0s.push_back(ggml_new_tensor_2d(gctx, GGML_TYPE_F32, c.k, c.n));
        ggml_set_input(x0s.back());
        xs.push_back(ggml_scale(gctx, x0s.back(), 0.75f + 0.125f*a));
    }
    ggml_cgraph * gf = ggml_new_graph(gctx);
    if (c.order.empty()) {
        for (size_t i = 0; i < ws.size(); ++i) {           // consumer i of every activation, then consumer i + 1, ...
            for (int a = 0; a < c.n_act; ++a) {
                ys.push_back(ggml_mul_mat(gctx, ws[i], xs[a]));
                ggml_set_output(ys.back());
                ggml_build_forward_expand(gf, ys.back());
            }
        }
    } else {
        for (size_t i = 0; i < c.order.size(); ++i) {
            ys.push_back(ggml_mul_mat(gctx, ws[i % ws.size()], xs[c.order[i]]));
            ggml_set_output(ys.back());
            ggml_build_forward_expand(gf, ys.back());
        }
    }
    bool ok = true;
    for (ggml_tensor * y : ys) {
        ok = ok && ggml_backend_supports_op(backend, y);
    }
    if (!ok) {
        ggml_free(gctx);
        ggml_free(wctx);
        return GROUP_UNSUPPORTED;
    }
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
    ggml_gallocr_t galloc = nullptr;
    if (ok && wbuf) {
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ok = ggml_gallocr_alloc_graph(galloc, gf);
    }
    if (ok && wbuf) {
        std::mt19937 rng(seed);
        for (ggml_tensor * w : ws) {
            fill_weights(w, rng);
        }
        for (int e = 0; e < N_EVALS && ok; ++e) {
            for (ggml_tensor * x0 : x0s) {
                fill_floats(x0, rng);
            }
            ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
            ggml_backend_synchronize(backend);
            for (ggml_tensor * y : ys) {
                const size_t off = blob.size();
                blob.resize(off + ggml_nbytes(y));
                ggml_backend_tensor_get(y, blob.data() + off, 0, ggml_nbytes(y));
            }
        }
    }
    ok = ok && wbuf != nullptr;
    if (!ok) {
        std::fprintf(stderr, "R1 case (K=%" PRId64 ", N=%" PRId64 ", %zu matmuls x %d activations) failed to allocate or compute\n",
                     c.k, c.n, c.types.size(), c.n_act);
    }
    if (galloc) {
        ggml_gallocr_free(galloc);
    }
    if (wbuf) {
        ggml_backend_buffer_free(wbuf);
    }
    ggml_free(gctx);
    ggml_free(wctx);
    return ok ? GROUP_RAN : GROUP_FAILED;
}

static std::vector<group_case> group_cases() {
    const ggml_type F175 = GGML_TYPE_AIF175_U2, F25 = GGML_TYPE_AIF25_U2, A5 = GGML_TYPE_AIT5_U, A2 = GGML_TYPE_AIT2_U2;
    std::vector<group_case> cs;
    for (int64_t n : { 1, 2, 4, 5, 8, 12, 33 }) {
        cs.push_back({ { F175, F175 }, { 10240, 6144 }, 5120, n });           // GDN qkv + z (8 GB file: 126 / 126)
        cs.push_back({ { F25, F25, F25 }, { 12288, 1024, 1024 }, 5120, n });  // attention q / k / v
        cs.push_back({ { F175, F175, F175 }, { 1024, 1024, 512 }, 5120, n }); // gate / up (rows cut for size)
        cs.push_back({ { A5, A5 }, { 10240, 6144 }, 5120, n });               // P76 qkv + z (114 / 114)
        cs.push_back({ { A2, A2 }, { 2048, 1024 }, 5120, n });                // 122 / 122
        cs.push_back({ { F25, A2, F175, A5, A2 }, { 1024, 512, 512, 512, 256 }, 5120, n }); // mixed layouts on one input
        cs.push_back({ { F175 }, { 5120 }, 17408, n });                        // down: a single consumer
        cs.push_back({ { F175, F25 }, { 1024, 512 }, 5120, n, 5 });            // 5 activations in a cycle: evict on every miss
        cs.push_back({ { F175, F175, F175 }, { 512, 512, 512 }, 5120, n, 4, { 0, 1, 2, 3, 0, 3, 1, 2, 0, 3, 1 } });
                                                                               // 4 activations: hits next to evictions
    }
    return cs;
}

// R2: head [k, rows] of type t, n_out outputs, n_sel shortlisted ids (broadcast or per output). Returns false on a bit
// difference between the one-row-expert MUL_MAT_ID and the full-head MUL_MAT gathered on the host.
static bool run_rows(ggml_backend_t backend, ggml_type t, int64_t k, int64_t rows, int64_t n_out, int64_t n_sel,
                     bool broadcast, uint32_t seed, int & n_checked) {
    ggml_init_params ip = { 32*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * head = ggml_new_tensor_2d(ctx, t, k, rows);
    ggml_tensor * x    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n_out);
    ggml_tensor * ids  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_sel, broadcast ? 1 : n_out);
    ggml_tensor * full = ggml_mul_mat(ctx, head, x);                                   // [rows, n_out]
    ggml_tensor * as   = ggml_reshape_3d(ctx, head, k, 1, rows);
    ggml_tensor * ids_v = ids;
    if (broadcast) {
        ids_v = ggml_view_2d(ctx, ids, n_sel, 1, ids->nb[1], 0);
        ids_v->ne[1] = n_out;   // the draft graph's broadcast view (llama-graph.cpp build_draft_vocab_logits)
        ids_v->nb[1] = 0;
    }
    ggml_tensor * sel = ggml_mul_mat_id(ctx, as, ggml_reshape_3d(ctx, x, k, 1, n_out), ids_v); // [1, n_sel, n_out]
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, full);
    ggml_build_forward_expand(gf, sel);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buf);
    std::mt19937 rng(seed);
    fill_weights(head, rng);
    fill_floats(x, rng);
    std::vector<int32_t> idv(ggml_nelements(ids));
    // clustered low ids (as the shortlists are) plus scattered ones and repeats
    std::uniform_int_distribution<int32_t> low(0, (int32_t) rows/4), any(0, (int32_t) rows - 1);
    for (size_t i = 0; i < idv.size(); ++i) {
        idv[i] = (i % 3 == 0) ? any(rng) : low(rng);
    }
    ggml_backend_tensor_set(ids, idv.data(), 0, ggml_nbytes(ids));
    bool ok = ggml_backend_supports_op(backend, sel) && ggml_backend_supports_op(backend, full);
    if (!ok) {
        std::fprintf(stderr, "R2 %s: MUL_MAT_ID over one-row experts not supported\n", ggml_type_name(t));
    } else {
        ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
        ggml_backend_synchronize(backend);
        std::vector<float> yf(ggml_nelements(full)), ys(ggml_nelements(sel));
        ggml_backend_tensor_get(full, yf.data(), 0, ggml_nbytes(full));
        ggml_backend_tensor_get(sel, ys.data(), 0, ggml_nbytes(sel));
        int64_t bad = 0;
        for (int64_t o = 0; o < n_out; ++o) {
            for (int64_t j = 0; j < n_sel; ++j) {
                const int32_t id = idv[(broadcast ? 0 : o)*n_sel + j];
                const float a = yf[o*rows + id], b = ys[o*n_sel + j];
                if (std::memcmp(&a, &b, sizeof(float)) != 0) {
                    ++bad;
                }
            }
        }
        ++n_checked;
        if (bad != 0) {
            std::fprintf(stderr, "R2 %s K=%" PRId64 " rows=%" PRId64 " n_out=%" PRId64 " n_sel=%" PRId64 " broadcast=%d: %" PRId64 " logits differ\n",
                         ggml_type_name(t), k, rows, n_out, n_sel, (int) broadcast, bad);
            ok = false;
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

// every R1 case (blob + the two counters at the end) and, when rows is set, every R2 case
static int run_all(std::vector<uint8_t> & blob, bool rows) {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("CUDA0");
    if (!dev) {
        return 77;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    GGML_ASSERT(backend);
    uint32_t seed = 20261121;
    int n_ran = 0, n_skip = 0, n_fail = 0;
    for (const group_case & c : group_cases()) {
        // a skipped case writes nothing, in both children alike (same device, same support answers)
        switch (run_group(backend, c, seed++, blob)) {
            case GROUP_RAN:         ++n_ran;  break;
            case GROUP_UNSUPPORTED: ++n_skip; break;
            case GROUP_FAILED:      ++n_fail; break;
        }
    }
    const int64_t replays = counter(backend, "cuda_graph_replays");
    std::fprintf(stderr, "R1: %d cases ran, %d skipped (not supported on this device), %d failed; %" PRId64 " CUDA graph replays\n",
                 n_ran, n_skip, n_fail, replays);
    if (n_fail != 0) {
        ggml_backend_free(backend);
        return 1;
    }
    if (n_ran == 0) {
        ggml_backend_free(backend);
        return 77;
    }
    const int64_t cnt[3] = { counter(backend, "ai_prep_runs"), counter(backend, "ai_prep_hits"), replays };
    const size_t off = blob.size();
    blob.resize(off + sizeof(cnt));
    std::memcpy(blob.data() + off, cnt, sizeof(cnt));
    int rc = 0;
    if (rows) {
        int n_checked = 0;
        for (ggml_type t : { GGML_TYPE_AIF25_U2, GGML_TYPE_AIF175_U2, GGML_TYPE_AIF25_U2_512, GGML_TYPE_AIF25_U }) {
            for (int64_t n_out : { 1, 2, 3, 4, 8 }) {
                for (bool bc : { true, false }) {
                    if (!run_rows(backend, t, 5120, 8192, n_out, 777, bc, seed++, n_checked)) {
                        rc = 1;
                    }
                }
            }
        }
        // AIF25_U at K = 1,920 (K % 256 != 0: the unshifted prep layout)
        for (int64_t n_out : { 1, 3, 8 }) {
            for (bool bc : { true, false }) {
                if (!run_rows(backend, GGML_TYPE_AIF25_U, 1920, 4096, n_out, 300, bc, seed++, n_checked)) {
                    rc = 1;
                }
            }
        }
        // the real head shape (248,320 rows) with the RM24H list size, broadcast and per output
        for (bool bc : { true, false }) {
            if (!run_rows(backend, GGML_TYPE_AIF25_U2, 5120, 248320, 4, 24576, bc, seed++, n_checked)) {
                rc = 1;
            }
        }
        std::fprintf(stderr, "R2: %d shortlist cases checked against the full head: %s\n", n_checked, rc ? "FAIL" : "bit-identical");
    }
    ggml_backend_free(backend);
    return rc;
}

#ifndef _WIN32
static int run_child(bool kaifuse, std::vector<uint8_t> & blob) {
    int fd[2];
    GGML_ASSERT(pipe(fd) == 0);
    const pid_t pid = fork();
    GGML_ASSERT(pid >= 0);
    if (pid == 0) {
        close(fd[0]);
        setenv("LLAMA_KAIFUSE", kaifuse ? "1" : "0", 1);
        std::vector<uint8_t> b;
        const int rc = run_all(b, kaifuse);
        size_t done = 0;
        while (done < b.size()) {
            const ssize_t n = write(fd[1], b.data() + done, b.size() - done);
            if (n <= 0) {
                _exit(1);
            }
            done += (size_t) n;
        }
        close(fd[1]);
        _exit(rc);
    }
    close(fd[1]);
    uint8_t tmp[65536];
    ssize_t n;
    while ((n = read(fd[0], tmp, sizeof(tmp))) > 0) {
        blob.insert(blob.end(), tmp, tmp + n);
    }
    close(fd[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}
#endif

int main() {
#ifdef _WIN32
    std::fprintf(stderr, "test-kaifuse needs fork(); not run\n");
    return 77;
#else
    std::vector<uint8_t> ref, on;
    const int rc_ref = run_child(false, ref);
    if (rc_ref == 77) {
        std::fprintf(stderr, "no CUDA device / AI+ kernels; not run\n");
        return 77;
    }
    const int rc_on = run_child(true, on);
    if (rc_ref != 0 || rc_on != 0 || ref.size() != on.size() || ref.size() < 3*sizeof(int64_t)) {
        std::fprintf(stderr, "test-kaifuse: child failed (rc %d / %d, %zu / %zu bytes): FAIL\n", rc_ref, rc_on, ref.size(), on.size());
        return 1;
    }
    int64_t c_ref[3], c_on[3];
    std::memcpy(c_ref, ref.data() + ref.size() - sizeof(c_ref), sizeof(c_ref));
    std::memcpy(c_on, on.data() + on.size() - sizeof(c_on), sizeof(c_on));
    const size_t n_out = ref.size() - sizeof(c_ref);
    size_t diff = 0;
    for (size_t i = 0; i < n_out; i += sizeof(float)) {
        diff += std::memcmp(ref.data() + i, on.data() + i, sizeof(float)) != 0;
    }
    std::fprintf(stderr, "R1: %zu outputs; kill switch preps %" PRId64 " hits %" PRId64 "; default preps %" PRId64 " hits %" PRId64
                 "; CUDA graph replays %" PRId64 " / %" PRId64 "; %zu outputs differ\n",
                 n_out/sizeof(float), c_ref[0], c_ref[1], c_on[0], c_on[1], c_ref[2], c_on[2], diff);
    if (c_on[2] <= 0) {
        std::fprintf(stderr, "note: no CUDA graph replay happened (graphs off or not captured): replay not covered here\n");
    }
    // every prep the kill switch launches is either launched or served from the cache by the default
    const bool ok = diff == 0 && c_ref[1] == 0 && c_on[1] > 0 && c_on[0] + c_on[1] == c_ref[0];
    std::fprintf(stderr, "test-kaifuse: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
#endif
}
