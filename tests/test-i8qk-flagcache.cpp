// Shared-mask cache check: two attention layers, changing masks across graph evaluations and fresh-vs-warm output identity.
// Run FLAGCACHE=0/1 in separate processes and compare --dump files. --packed uses I32 masks;
// --write-mask overwrites one mask between ordered consumers and checks against separate-mask storage.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static constexpr int64_t D      = 256;
static constexpr int64_t HKV    = 4;
static constexpr int64_t GQA    = 6;
static constexpr int64_t CAP    = 4096;
static constexpr int64_t NQ     = 256;
static constexpr ggml_type TK   = GGML_TYPE_TQ5_0;
static constexpr ggml_type TV   = GGML_TYPE_TURBO4_0;

static bool packed_mask = false;
static bool write_mask = false;

struct cache_pair {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
};

static cache_pair make_cache(ggml_backend_t be, const std::vector<uint8_t> & kb, const std::vector<uint8_t> & vb) {
    cache_pair c;
    ggml_init_params ip = { 4*ggml_tensor_overhead(), nullptr, true };
    c.ctx = ggml_init(ip);
    c.k = ggml_new_tensor_2d(c.ctx, TK, D*HKV, CAP);
    c.v = ggml_new_tensor_2d(c.ctx, TV, D*HKV, CAP);
    c.buf = ggml_backend_alloc_ctx_tensors(c.ctx, be);
    ggml_backend_tensor_set(c.k, kb.data(), 0, kb.size());
    ggml_backend_tensor_set(c.v, vb.data(), 0, vb.size());
    return c;
}

static void quantize_rows(ggml_type t, const std::vector<float> & src, std::vector<uint8_t> & dst, int64_t r0, int64_t r1) {
    const size_t rs = ggml_row_size(t, D*HKV);
    ggml_quantize_chunk(t, src.data() + r0*D*HKV, dst.data() + r0*rs, 0, r1 - r0, D*HKV, nullptr);
}

struct run_io {
    std::vector<float> q1, q2;
    std::vector<ggml_fp16_t> mask;
};

// two attention ops (q1, q2) over the first n_kv cells of cache c, sharing one mask; returns both outputs concatenated.
// masks: one or more mask contents, evaluated one after the other on the same graph (only the last output is returned)
static std::vector<float> run(ggml_backend_t be, const cache_pair & c, int64_t n_kv, const run_io & io,
                              const std::vector<std::vector<ggml_fp16_t>> & masks, bool write_reference = false) {
    ggml_init_params ip = { 32*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, NQ, HKV*GQA);
    ggml_tensor * q2 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, NQ, HKV*GQA);
    ggml_tensor * m  = ggml_new_tensor_2d(ctx, packed_mask ? GGML_TYPE_I32 : GGML_TYPE_F16, packed_mask ? n_kv/32 : n_kv, NQ);
    if (packed_mask) {
        m->op_params[0] = int32_t(n_kv);
    }
    ggml_tensor * mask_after = write_mask ? ggml_dup_tensor(ctx, m) : nullptr;
    if (packed_mask && mask_after) {
        mask_after->op_params[0] = int32_t(n_kv);
    }
    ggml_tensor * k  = ggml_view_3d(ctx, c.k, D, n_kv, HKV, c.k->nb[1], ggml_row_size(TK, D), 0);
    ggml_tensor * v  = ggml_view_3d(ctx, c.v, D, n_kv, HKV, c.v->nb[1], ggml_row_size(TV, D), 0);
    ggml_tensor * o1 = ggml_flash_attn_ext(ctx, q1, k, v, m, 1.0f/16.0f, 0.0f, 0.0f);
    ggml_tensor * q2_ready = q2;
    if (write_mask && !write_reference) {
        // Explicit graph dependencies: attention1 -> overwrite -> attention2.
        // Both attention nodes still use exactly the same mask tensor pointer.
        ggml_tensor * ordered = ggml_view_tensor(ctx, mask_after);
        ordered->src[1] = o1;
        ggml_tensor * overwrite = ggml_cpy(ctx, ordered, m);
        q2_ready = ggml_view_tensor(ctx, q2);
        q2_ready->src[1] = overwrite;
    }
    ggml_tensor * o2 = ggml_flash_attn_ext(ctx, q2_ready, k, v,
            write_mask && write_reference ? mask_after : m, 1.0f/16.0f, 0.0f, 0.0f);
    ggml_prec_set_acc(o1, GGML_PREC_F32);
    ggml_prec_set_acc(o2, GGML_PREC_F32);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, o1);
    ggml_build_forward_expand(gf, o2);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(q1, io.q1.data(), 0, ggml_nbytes(q1));
    ggml_backend_tensor_set(q2, io.q2.data(), 0, ggml_nbytes(q2));
    for (const auto & mk : masks) {
        // mask rows are n_kv wide: copy the first n_kv columns of each CAP-wide host row
        std::vector<ggml_fp16_t> mm((size_t) n_kv*NQ);
        for (int64_t r = 0; r < NQ; ++r) {
            memcpy(mm.data() + r*n_kv, mk.data() + r*CAP, n_kv*sizeof(ggml_fp16_t));
        }
        auto upload_mask = [&](ggml_tensor * dst, const std::vector<ggml_fp16_t> & values) {
            if (packed_mask) {
                std::vector<uint32_t> words(size_t(n_kv/32)*NQ, 0);
                for (int64_t r = 0; r < NQ; ++r) {
                    for (int64_t j = 0; j < n_kv; ++j) {
                        if (values[r*n_kv + j] == ggml_fp32_to_fp16(0.0f)) {
                            words[r*(n_kv/32) + j/32] |= uint32_t(1) << (j%32);
                        }
                    }
                }
                ggml_backend_tensor_set(dst, words.data(), 0, ggml_nbytes(dst));
            } else {
                ggml_backend_tensor_set(dst, values.data(), 0, ggml_nbytes(dst));
            }
        };
        upload_mask(m, mm);
        if (write_mask) {
            // Toggle one entire KV tile for the first query tile. This changes
            // classification, rather than merely changing a mixed tile's bits.
            for (int64_t r = 0; r < 128; ++r) {
                for (int64_t j = 0; j < 64; ++j) {
                    auto & value = mm[r*n_kv + j];
                    value = ggml_fp32_to_fp16(value == ggml_fp32_to_fp16(0.0f) ? -INFINITY : 0.0f);
                }
            }
            upload_mask(mask_after, mm);
        }
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "graph compute failed\n");
            exit(1);
        }
    }
    std::vector<float> out(ggml_nelements(o1) + ggml_nelements(o2));
    ggml_backend_tensor_get(o1, out.data(), 0, ggml_nbytes(o1));
    ggml_backend_tensor_get(o2, out.data() + ggml_nelements(o1), 0, ggml_nbytes(o2));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

// causal mask for queries at the last NQ positions of n_kv cells; extra: also mask a block of old cells per query block
static std::vector<ggml_fp16_t> causal_mask(int64_t n_kv, int64_t hole0, int64_t hole1) {
    std::vector<ggml_fp16_t> mk((size_t) CAP*NQ);
    for (int64_t r = 0; r < NQ; ++r) {
        const int64_t pos = n_kv - NQ + r;
        for (int64_t j = 0; j < CAP; ++j) {
            const bool vis = j <= pos && !(j >= hole0 && j < hole1 && r < NQ/2);
            mk[r*CAP + j] = ggml_fp32_to_fp16(vis ? 0.0f : -INFINITY);
        }
    }
    return mk;
}

int main(int argc, char ** argv) {
#ifdef _WIN32
    _putenv_s("GGML_CUDA_FA_I8QK", "1");
#else
    setenv("GGML_CUDA_FA_I8QK", "1", 0);
#endif
    const char * dump = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            dump = argv[++i];
        } else if (strcmp(argv[i], "--packed") == 0) {
            packed_mask = true;
        } else if (strcmp(argv[i], "--write-mask") == 0) {
            write_mask = true;
        }
    }
    ggml_backend_load_all();
    ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!be) {
        fprintf(stderr, "no GPU backend\n");
        return 1;
    }
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> kf((size_t) CAP*D*HKV), vf((size_t) CAP*D*HKV);
    std::vector<float> bias(D*HKV);
    for (auto & b : bias) { b = 3.0f*nd(rng); }
    for (size_t i = 0; i < kf.size(); ++i) { kf[i] = nd(rng) + bias[i % (D*HKV)]; vf[i] = nd(rng); }
    std::vector<uint8_t> kb(ggml_row_size(TK, D*HKV)*CAP), vb(ggml_row_size(TV, D*HKV)*CAP);
    quantize_rows(TK, kf, kb, 0, CAP);
    quantize_rows(TV, vf, vb, 0, CAP);

    run_io io;
    io.q1.resize((size_t) D*NQ*HKV*GQA);
    io.q2.resize(io.q1.size());
    for (auto & x : io.q1) { x = nd(rng); }
    for (auto & x : io.q2) { x = 2.0f*nd(rng); }

    cache_pair warm = make_cache(be, kb, vb);
    std::vector<cache_pair> colds;
    FILE * fd = dump ? fopen(dump, "wb") : nullptr;
    if (dump && !fd) {
        fprintf(stderr, "cannot open dump file\n");
        return 1;
    }

    int n_fail = 0, n_step = 0;
    auto rewrite = [&](int64_t t0, int64_t t1, float amp) {
        for (int64_t t = t0; t < t1; ++t) {
            for (int64_t i = 0; i < D*HKV; ++i) {
                kf[t*D*HKV + i] = amp*nd(rng) + bias[i];
            }
        }
        quantize_rows(TK, kf, kb, t0, t1);
        const size_t rs = ggml_row_size(TK, D*HKV);
        ggml_backend_tensor_set(warm.k, kb.data() + t0*rs, t0*rs, (t1 - t0)*rs);
    };
    auto step = [&](const char * what, int64_t n_kv, const std::vector<std::vector<ggml_fp16_t>> & masks) {
        const std::vector<float> a = run(be, warm, n_kv, io, masks);
        colds.push_back(make_cache(be, kb, vb));
        const std::vector<float> b = run(be, colds.back(), n_kv, io, {masks.back()});
        ++n_step;
        size_t nd_ = 0; double maxd = 0.0; bool finite = true;
        for (size_t i = 0; i < a.size(); ++i) {
            if (memcmp(&a[i], &b[i], sizeof(float)) != 0) { ++nd_; maxd = std::max(maxd, (double) fabsf(a[i] - b[i])); }
            finite = finite && std::isfinite(a[i]);
        }
        bool ok = nd_ == 0 && finite;
        if (write_mask) {
            const auto reference = run(be, colds.back(), n_kv, io, {masks.back()}, true);
            const bool write_ok = reference.size() == a.size() &&
                    memcmp(reference.data(), a.data(), a.size()*sizeof(float)) == 0;
            printf("    ordered mask overwrite vs separate-mask reference: %s\n", write_ok ? "IDENTICAL" : "DIFFER");
            ok = ok && write_ok;
        }
        n_fail += !ok;
        printf("  step %d %-44s n_kv %5lld: warm vs cold %s (%zu differ, max |d| %.3g)%s\n", n_step, what, (long long) n_kv,
               ok ? "IDENTICAL" : "DIFFER", nd_, maxd, finite ? "" : " NON-FINITE");
        if (fd && fwrite(a.data(), sizeof(float), a.size(), fd) != a.size()) {
            fprintf(stderr, "cannot write dump file\n");
            ++n_fail;
        }
    };

    step("first launch",                         1024, {causal_mask(1024, 0, 0)});
    step("context grows",                        2048, {causal_mask(2048, 0, 0)});
    step("same launch again",                    2048, {causal_mask(2048, 0, 0)});
    rewrite(1100, 1111, 1.0f);
    step("old cells 1100-1110 rewritten",        2048, {causal_mask(2048, 0, 0)});
    rewrite(5, 6, 4.0f);
    step("cell 5 (inside the mean prefix) rewritten", 1536, {causal_mask(1536, 0, 0)});
    step("regrow past tiles of the old mean",    2048, {causal_mask(2048, 0, 0)});
    rewrite(0, 64, 1.0f);
    rewrite(3000, 3100, 1.0f);
    step("full context, prefix tile rewritten",  4096, {causal_mask(4096, 0, 0)});
    step("mask changes between evaluations",     4096, {causal_mask(4096, 0, 0), causal_mask(4096, 512, 1536)});
    step("shrink",                               1024, {causal_mask(1024, 128, 512)});

    if (fd && fclose(fd) != 0) { ++n_fail; }
    for (auto & c : colds) { ggml_backend_buffer_free(c.buf); ggml_free(c.ctx); }
    ggml_backend_buffer_free(warm.buf);
    ggml_free(warm.ctx);
    ggml_backend_free(be);
    printf("%s: %d/%d steps identical\n", n_fail ? "FAIL" : "OK", n_step - n_fail, n_step);
    return n_fail ? 1 : 0;
}
