// KVarN fused write rotation (#139, GGML_KVARN_FUSED_ROT): on every backend, the fused SET_ROWS
// (ggml_set_rows_kvarn_rot) must store exactly the bytes of the unfused pair
// ggml_set_rows_tq6_rotated(dst, ggml_turbo_wht(src, 0, group, NULL), idx), f16 sink mirror included.
//
//   test-kvarn-fused-rot          all backends, built-in shapes; exit 1 on any byte difference

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct shape {
    int hkv;        // heads per row (row = hkv*256 floats)
    int n_tok;      // rows written
    int n_rows;     // ring rows in dst
    int group;      // 256 or 128
    bool sink;      // f16 sink mirror (op_params 2/3)
    bool strided;   // source rows with a padded token stride (non-contiguous view)
    bool i32;       // I32 row indices instead of I64
};

static bool run_shape(ggml_backend_t backend, const shape & s, uint32_t seed) {
    const int64_t ne0 = (int64_t) s.hkv*256;
    const int sink_rows = 64;
    const size_t packed_row = ggml_row_size(GGML_TYPE_TQ6_0, ne0);
    const int64_t rows = s.n_rows + (s.sink ? ((int64_t) sink_rows*ne0*2 + packed_row - 1)/packed_row : 0);
    const int64_t pad = s.strided ? 256 : 0;

    ggml_init_params ip = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * dst_a = ggml_new_tensor_2d(ctx, GGML_TYPE_TQ6_0, ne0, rows);
    ggml_tensor * dst_b = ggml_new_tensor_2d(ctx, GGML_TYPE_TQ6_0, ne0, rows);
    ggml_tensor * src_m = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0 + pad, s.n_tok);
    ggml_tensor * idx   = ggml_new_tensor_1d(ctx, s.i32 ? GGML_TYPE_I32 : GGML_TYPE_I64, s.n_tok);
    ggml_tensor * src   = ggml_view_2d(ctx, src_m, ne0, s.n_tok, src_m->nb[1], 0);

    // unfused: the graph's ggml_cont + GGML_OP_TURBO_WHT, then the TQ6 write in the rotated basis
    ggml_tensor * rot = ggml_turbo_wht(ctx, ggml_cont(ctx, src), 0, s.group, nullptr);
    ggml_tensor * wa  = ggml_set_rows_tq6_rotated(ctx, dst_a, rot, idx);
    ggml_tensor * wb  = ggml_set_rows_kvarn_rot(ctx, dst_b, src, idx, s.group);
    for (ggml_tensor * w : { wa, wb }) {
        if (s.sink) {
            w->op_params[2] = sink_rows;
            w->op_params[3] = s.n_rows;
        }
        int32_t g128 = 128;
        memcpy(w->op_params, &g128, sizeof(g128));  // what cpy_k/cpy_v store for TQ6
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, wa);
    ggml_build_forward_expand(gf, wb);

    if (!ggml_backend_supports_op(backend, wb) || !ggml_backend_supports_op(backend, rot)) {
        printf("  %s: fused op not supported, skipped\n", ggml_backend_name(backend));
        ggml_free(ctx);
        return true;
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    std::vector<float> h_src((size_t) (ne0 + pad)*s.n_tok);
    for (int t = 0; t < s.n_tok; ++t) {
        // per-row scale spread (K rows carry outlier channels), plus an all-zero row for the norm-0 path
        const float scale = t == 1 ? 0.0f : std::pow(10.0f, uni(rng)*2.0f);
        for (int64_t i = 0; i < ne0 + pad; ++i) {
            float v = uni(rng)*scale;
            if (i % 61 == 7) { v *= 30.0f; }
            h_src[(size_t) t*(ne0 + pad) + i] = v;
        }
    }
    ggml_backend_tensor_set(src_m, h_src.data(), 0, h_src.size()*sizeof(float));

    // distinct target rows, some of them in [0, sink_rows) so the sink mirror is exercised
    std::vector<int64_t> perm(s.n_rows);
    for (int i = 0; i < s.n_rows; ++i) { perm[i] = i; }
    std::shuffle(perm.begin(), perm.end(), rng);
    if (s.i32) {
        std::vector<int32_t> h(s.n_tok);
        for (int i = 0; i < s.n_tok; ++i) { h[i] = (int32_t) perm[i]; }
        ggml_backend_tensor_set(idx, h.data(), 0, h.size()*sizeof(int32_t));
    } else {
        ggml_backend_tensor_set(idx, perm.data(), 0, (size_t) s.n_tok*sizeof(int64_t));
    }

    std::vector<uint8_t> fill(ggml_nbytes(dst_a));
    for (auto & b : fill) { b = (uint8_t) rng(); }
    ggml_backend_tensor_set(dst_a, fill.data(), 0, fill.size());
    ggml_backend_tensor_set(dst_b, fill.data(), 0, fill.size());

    ggml_backend_graph_compute(backend, gf);

    std::vector<uint8_t> a(fill.size()), b(fill.size());
    ggml_backend_tensor_get(dst_a, a.data(), 0, a.size());
    ggml_backend_tensor_get(dst_b, b.data(), 0, b.size());

    size_t n_diff = 0, first = 0, n_changed = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) { if (n_diff++ == 0) { first = i; } }
        if (a[i] != fill[i]) { n_changed++; }
    }
    printf("  %-6s hkv=%d n_tok=%4d group=%d sink=%d strided=%d idx=%s: %s (%zu bytes written, %zu differ%s)\n",
            ggml_backend_name(backend), s.hkv, s.n_tok, s.group, s.sink, s.strided, s.i32 ? "i32" : "i64",
            n_diff == 0 && n_changed > 0 ? "OK" : "FAIL", n_changed, n_diff,
            n_diff ? (", first at byte " + std::to_string(first)).c_str() : "");

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return n_diff == 0 && n_changed > 0;
}

int main() {
    ggml_backend_load_all();

    std::vector<shape> shapes;
    for (int group : { 256, 128 }) {
        for (int n_tok : { 1, 2, 5, 8, 37, 512 }) {
            for (bool sink : { false, true }) {
                shapes.push_back({ 4, n_tok, 1024, group, sink, false, false });
            }
        }
        shapes.push_back({ 1, 3, 512, group, true, true, true });
        shapes.push_back({ 4, 5, 512, group, false, true, false });
        shapes.push_back({ 2, 64, 512, group, true, false, true });
    }

    bool ok = true;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) { continue; }
        ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
        if (!backend) { continue; }
        printf("backend %s\n", ggml_backend_name(backend));
        uint32_t seed = 1;
        for (const auto & s : shapes) {
            ok = run_shape(backend, s, seed++) && ok;
        }
        ggml_backend_free(backend);
    }
    printf("%s\n", ok ? "ALL OK" : "FAILED");
    return ok ? 0 : 1;
}
