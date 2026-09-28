// [#81] Compact resident draft head (src/llama-draft-vocab-compact.h) for the MTP draft vocabulary shortlist.
//
// A synthetic EXL3 head W[K, N] (random trellis words: every 16-bit code decodes to a codebook value, so any bit
// pattern is a valid EXL3 tensor; random suh/svh with random signs; the Sylvester 128x128 Hadamard / sqrt(128), which
// equals the exl3_had128.weight of the 27B EXL3 4.0 bpw file to 1.5e-9) and a shuffled shortlist of n_sel ids that
// includes id 0, id N-1 and one whole 128-row output block.
//
// (a) domain: the builder's f32 rows equal an independent double-precision reference
//       ref[j] = svh[j] * sum_r had[j % 128][r] * W_trellis[128 * (j / 128) + r]
//     (the effective head row in the rotated input domain) within max|diff| / max|ref| <= 1e-5.
// (b) quantization: the compact rows of each type, dequantized back (to_float), against ref: max abs and
//     NMSE = sum(diff^2) / sum(ref^2); bounds q8_0 <= 1e-4, q6_K <= 2e-3, iq4_xs <= 1.5e-2, and strictly
//     q8_0 < q6_K < iq4_xs on the EXL3 heads (q8_0 must be the tightest).
// (c) logits: on every available backend (CPU always, the GPU when present), widths T = 1..8, the full EXL3 head
//     y = mul_mat(W, x) with suh/svh on src[2]/src[3] (the production fused path) gathered at the selected ids,
//     against mul_mat(compact, x_rot) with x_rot = H128(suh * x) built from generic ops exactly as the llama graph
//     builds it. Reported: rel = max|diff| / max|y_sel| and NMSE over the selected logits. Bounds (rel / NMSE):
//     f32 5e-3 / 1e-5, q8_0 3e-2 / 3e-4, q6_K 6e-2 / 3e-3, iq4_xs 0.2 / 2e-2. The q8_0..iq4_xs bounds include the
//     backend's own activation quantization (q8_1 / q8_K) of x_rot, and the EXL3 kernel's fp16 activations.
//     Negative control: the raw trellis rows (no output Hadamard, no svh) on the same x_rot must MISS (rel >= 0.5),
//     which shows the output-side mixing is required and that (c) detects a wrong domain.
// (d) a plain row-addressable head (Q6_K, the LLAMA_DRAFT_VOCAB_COMPACT=1 path): f32 compact rows are byte-equal
//     to to_float of the head rows, and (b) is repeated against those rows. (c) is repeated with x_rot = x against
//     an exact host reference, as (a) does for EXL3: y_ref[i] = sum_k to_float(head row ids[i])[k] * x[k] accumulated
//     in double, with the same per-type bounds as above. The backend's plain-head mul_mat(head, x) cannot be that
//     reference: it quantizes x (q8_K on CPU, q8_1 on CUDA) and carries ~1e-2 relative error, more than the f32
//     bound allows (the EXL3 reference path keeps x in f32/fp16, so the EXL3 heads keep it as their reference).
// (c2) plain heads only, so the backend full-head path stays exercised: the same compact logits against
//     mul_mat(head, x) on the backend gathered at the selected ids. Bound = the (c) bound of the compact type widened
//     to at least rel 2e-2 / NMSE 2e-4 for the reference's own activation quantization (measured for the byte-exact
//     f32 compact rows on the 3090 Ti box: rel 5.0e-3..8.3e-3, NMSE 2.5e-5..4.9e-5, CPU and CUDA, T = 1..8).
// Every measured value is printed; the process exits 1 on any bound miss.

#include "../src/llama-draft-vocab-compact.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int g_fail = 0;

static void check(bool ok, const char * what) {
    if (!ok) {
        g_fail++;
        printf("  FAIL: %s\n", what);
    }
}

struct head_case {
    std::string          name;
    ggml_type            type;     // EXL3_b or a plain type
    int64_t              K, N;
    std::vector<uint8_t> data;     // head bytes
    std::vector<float>   suh, svh, had;   // EXL3 only
    std::vector<int32_t> ids;
    std::vector<float>   ref;      // [n_sel][K] reference effective rows
    std::vector<float>   trellis;  // EXL3 only: [n_sel][K] raw trellis rows (negative control)
};

static std::vector<float> sylvester128() {
    std::vector<float> h(128 * 128);
    const float s = 1.0f / sqrtf(128.0f);
    for (int i = 0; i < 128; ++i) {
        for (int r = 0; r < 128; ++r) {
            h[i * 128 + r] = (__builtin_popcount(i & r) & 1) ? -s : s;
        }
    }
    return h;
}

static std::vector<int32_t> make_ids(int64_t N, int64_t n_sel, std::mt19937 & rng) {
    std::vector<int32_t> all(N);
    for (int64_t i = 0; i < N; ++i) all[i] = (int32_t) i;
    std::shuffle(all.begin(), all.end(), rng);
    std::vector<uint8_t> in(N, 0);
    std::vector<int32_t> ids;
    auto add = [&](int32_t id) { if (!in[id]) { in[id] = 1; ids.push_back(id); } };
    add(0);
    add((int32_t) N - 1);
    for (int32_t r = 0; r < 128; ++r) add(5 * 128 + r); // one whole output block
    for (int64_t i = 0; (int64_t) ids.size() < n_sel; ++i) add(all[i]);
    std::shuffle(ids.begin(), ids.end(), rng);
    return ids;
}

static head_case make_exl3(int bits, int64_t K, int64_t N, int64_t n_sel, uint32_t seed) {
    head_case h;
    h.type = (ggml_type) (GGML_TYPE_EXL3_2 + bits - 2);
    GGML_ASSERT(ggml_exl3_bits(h.type) == bits);
    h.name = std::string("exl3_") + std::to_string(bits);
    h.K = K; h.N = N;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> ud(0.5f, 1.5f);
    const size_t nbytes = (size_t) K * N * bits / 8;
    GGML_ASSERT(nbytes % 4 == 0);
    h.data.resize(nbytes);
    for (size_t i = 0; i < nbytes; i += 4) {
        const uint32_t u = (uint32_t) rng();
        memcpy(h.data.data() + i, &u, 4);
    }
    h.suh.resize(K); h.svh.resize(N);
    for (float & f : h.suh) f = (rng() & 1 ? 1.0f : -1.0f) * ud(rng) * 0.02f;
    for (float & f : h.svh) f = (rng() & 1 ? 1.0f : -1.0f) * ud(rng);
    h.had = sylvester128();
    h.ids = make_ids(N, n_sel, rng);

    // independent reference: whole-tensor decode, then the output Hadamard + svh in double
    std::vector<float> W((size_t) N * K);
    for (int64_t g = 0; g < N / 16; ++g) {
        ggml_exl3_dequantize_row_group(h.data.data(), K, N, bits, g, W.data() + (size_t) g * 16 * K);
    }
    h.ref.resize((size_t) n_sel * K);
    h.trellis.resize((size_t) n_sel * K);
    std::vector<double> acc(K);
    for (int64_t i = 0; i < n_sel; ++i) {
        const int64_t j = h.ids[i];
        std::fill(acc.begin(), acc.end(), 0.0);
        for (int r = 0; r < 128; ++r) {
            const double hr = h.had[(j % 128) * 128 + r];
            const float * w = W.data() + (size_t) ((j / 128) * 128 + r) * K;
            for (int64_t k = 0; k < K; ++k) acc[k] += hr * w[k];
        }
        for (int64_t k = 0; k < K; ++k) {
            h.ref[(size_t) i * K + k]     = (float) (acc[k] * h.svh[j]);
            h.trellis[(size_t) i * K + k] = W[(size_t) j * K + k];
        }
    }
    return h;
}

static head_case make_plain(ggml_type type, int64_t K, int64_t N, int64_t n_sel, uint32_t seed) {
    head_case h;
    h.type = type;
    h.name = std::string("plain_") + ggml_type_name(type);
    h.K = K; h.N = N;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> f((size_t) K * N);
    for (float & v : f) v = nd(rng);
    h.data.resize(ggml_row_size(type, K) * N);
    ggml_quantize_chunk(type, f.data(), h.data.data(), 0, N, K, nullptr);
    h.ids = make_ids(N, n_sel, rng);
    h.ref.resize((size_t) n_sel * K);
    const size_t rs = ggml_row_size(type, K);
    for (int64_t i = 0; i < n_sel; ++i) {
        ggml_get_type_traits(type)->to_float(h.data.data() + (size_t) h.ids[i] * rs, h.ref.data() + (size_t) i * K, K);
    }
    return h;
}

static std::vector<uint8_t> build(const head_case & h, ggml_type qt) {
    llama_draft_compact_src src;
    src.data = h.data.data();
    src.type = h.type;
    src.K    = h.K;
    src.N    = h.N;
    if (!h.svh.empty()) {
        src.svh = h.svh.data();
        src.had = h.had.data();
    }
    const int64_t n_sel = (int64_t) h.ids.size();
    std::vector<uint8_t> out(ggml_row_size(qt, h.K) * n_sel);
    GGML_ASSERT(llama_draft_compact_build(src, h.ids.data(), n_sel, qt, out.data(), 4));
    return out;
}

static std::vector<float> to_f32(const std::vector<uint8_t> & q, ggml_type qt, int64_t K, int64_t n) {
    std::vector<float> f((size_t) K * n);
    if (qt == GGML_TYPE_F32) {
        memcpy(f.data(), q.data(), f.size() * sizeof(float));
    } else {
        ggml_get_type_traits(qt)->to_float(q.data(), f.data(), K * n);
    }
    return f;
}

struct err { double rel_max, nmse, max_abs; };

static err compare(const float * a, const float * ref, size_t n, size_t stride_a = 1, size_t stride_r = 1) {
    double md = 0, mr = 0, sd = 0, sr = 0;
    for (size_t i = 0; i < n; ++i) {
        const double d = (double) a[i * stride_a] - ref[i * stride_r];
        md = std::max(md, fabs(d));
        mr = std::max(mr, fabs((double) ref[i * stride_r]));
        sd += d * d;
        sr += (double) ref[i * stride_r] * ref[i * stride_r];
    }
    return { mr > 0 ? md / mr : md, sr > 0 ? sd / sr : sd, md };
}

static const ggml_type QTYPES[] = { GGML_TYPE_Q8_0, GGML_TYPE_Q6_K, GGML_TYPE_IQ4_XS };
static double nmse_bound(ggml_type t) { return t == GGML_TYPE_Q8_0 ? 1e-4 : t == GGML_TYPE_Q6_K ? 2e-3 : 1.5e-2; }
static double logit_rel_bound(ggml_type t) {
    return t == GGML_TYPE_F32 ? 5e-3 : t == GGML_TYPE_Q8_0 ? 3e-2 : t == GGML_TYPE_Q6_K ? 6e-2 : 0.2;
}
static double logit_nmse_bound(ggml_type t) {
    return t == GGML_TYPE_F32 ? 1e-5 : t == GGML_TYPE_Q8_0 ? 3e-4 : t == GGML_TYPE_Q6_K ? 3e-3 : 2e-2;
}
// (c2) plain heads vs the backend's full-head mul_mat, whose x is quantized (q8_K CPU / q8_1 CUDA)
static double backend_rel_bound(ggml_type t)  { return std::max(logit_rel_bound(t),  2e-2); }
static double backend_nmse_bound(ggml_type t) { return std::max(logit_nmse_bound(t), 2e-4); }

// (c)/(d) logits on one backend: every compact type (+ the negative control for EXL3) at widths 1..8
static void run_logits(ggml_backend_t be, const head_case & h,
        const std::vector<std::pair<ggml_type, std::vector<uint8_t>>> & compact) {
    const bool exl3 = ggml_exl3_bits(h.type) != 0;
    const int64_t K = h.K, N = h.N, n_sel = (int64_t) h.ids.size();
    const char * bname = ggml_backend_name(be);
    std::mt19937 rng(911 + (uint32_t) h.type);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    for (int64_t T = 1; T <= 8; ++T) {
        const int n_c = (int) compact.size() + (exl3 ? 1 : 0);
        ggml_init_params pw = { ggml_tensor_overhead() * (8 + n_c), nullptr, true };
        ggml_context * cw = ggml_init(pw);
        ggml_tensor * w   = ggml_new_tensor_2d(cw, h.type, K, N);
        ggml_tensor * x   = ggml_new_tensor_2d(cw, GGML_TYPE_F32, K, T);
        ggml_tensor * suh = exl3 ? ggml_new_tensor_1d(cw, GGML_TYPE_F32, K) : nullptr;
        ggml_tensor * svh = exl3 ? ggml_new_tensor_1d(cw, GGML_TYPE_F32, N) : nullptr;
        ggml_tensor * had = exl3 ? ggml_new_tensor_2d(cw, GGML_TYPE_F32, 128, 128) : nullptr;
        std::vector<ggml_tensor *> ct;
        for (const auto & c : compact) ct.push_back(ggml_new_tensor_2d(cw, c.first, K, n_sel));
        ggml_tensor * neg = exl3 ? ggml_new_tensor_2d(cw, GGML_TYPE_F32, K, n_sel) : nullptr;
        ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors(cw, be);
        GGML_ASSERT(bw);
        ggml_backend_buffer_set_usage(bw, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_set(w, h.data.data(), 0, ggml_nbytes(w));
        std::vector<float> xv((size_t) K * T);
        for (float & f : xv) f = nd(rng);
        ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
        if (exl3) {
            ggml_backend_tensor_set(suh, h.suh.data(), 0, ggml_nbytes(suh));
            ggml_backend_tensor_set(svh, h.svh.data(), 0, ggml_nbytes(svh));
            ggml_backend_tensor_set(had, h.had.data(), 0, ggml_nbytes(had));
            ggml_backend_tensor_set(neg, h.trellis.data(), 0, ggml_nbytes(neg));
        }
        for (size_t i = 0; i < compact.size(); ++i) {
            ggml_backend_tensor_set(ct[i], compact[i].second.data(), 0, ggml_nbytes(ct[i]));
        }

        ggml_init_params pg = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
        ggml_context * cg = ggml_init(pg);
        ggml_tensor * y_full = ggml_mul_mat(cg, w, x);
        ggml_tensor * xr = x;
        if (exl3) {
            y_full->src[2] = suh;
            y_full->src[3] = svh;
            // llama-graph.cpp build_draft_vocab_compact: x_rot = H128(suh * x) via build_exl3_had128
            xr = ggml_mul(cg, x, suh);
            ggml_tensor * t3 = ggml_reshape_3d(cg, xr, 128, K / 128, T);
            t3 = ggml_mul_mat(cg, had, t3);
            xr = ggml_reshape_2d(cg, t3, K, T);
        }
        ggml_set_output(y_full);
        ggml_cgraph * gf = ggml_new_graph(cg);
        ggml_build_forward_expand(gf, y_full);
        std::vector<ggml_tensor *> yc;
        for (ggml_tensor * c : ct) {
            yc.push_back(ggml_mul_mat(cg, c, xr));
            ggml_set_output(yc.back());
            ggml_build_forward_expand(gf, yc.back());
        }
        ggml_tensor * yneg = nullptr;
        if (exl3) {
            yneg = ggml_mul_mat(cg, neg, xr);
            ggml_set_output(yneg);
            ggml_build_forward_expand(gf, yneg);
        }
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        GGML_ASSERT(ggml_gallocr_alloc_graph(ga, gf));
        GGML_ASSERT(ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS);

        std::vector<float> yf((size_t) N * T);
        ggml_backend_tensor_get(y_full, yf.data(), 0, ggml_nbytes(y_full));
        std::vector<float> ysel((size_t) n_sel * T);
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t i = 0; i < n_sel; ++i) {
                ysel[(size_t) t * n_sel + i] = yf[(size_t) t * N + h.ids[i]];
            }
        }
        // plain heads: the (c) reference is exact, the dequantized head rows (h.ref) times the f32 x in double
        std::vector<float> yref;
        if (!exl3) {
            yref.resize((size_t) n_sel * T);
            for (int64_t t = 0; t < T; ++t) {
                const float * xt = xv.data() + (size_t) t * K;
                for (int64_t i = 0; i < n_sel; ++i) {
                    const float * r = h.ref.data() + (size_t) i * K;
                    double s = 0.0;
                    for (int64_t k = 0; k < K; ++k) s += (double) r[k] * (double) xt[k];
                    yref[(size_t) t * n_sel + i] = (float) s;
                }
            }
        }
        const std::vector<float> & yc_ref = exl3 ? ysel : yref;
        std::vector<float> yv((size_t) n_sel * T);
        for (size_t i = 0; i < compact.size(); ++i) {
            ggml_backend_tensor_get(yc[i], yv.data(), 0, ggml_nbytes(yc[i]));
            const err e = compare(yv.data(), yc_ref.data(), yv.size());
            const ggml_type qt = compact[i].first;
            const bool ok = e.rel_max <= logit_rel_bound(qt) && e.nmse <= logit_nmse_bound(qt) && std::isfinite(e.nmse);
            printf("  (c) %-6s %-10s T=%lld %-7s rel %.3e (<= %.0e)  nmse %.3e (<= %.0e)%s  %s\n", bname, h.name.c_str(),
                    (long long) T, ggml_type_name(qt), e.rel_max, logit_rel_bound(qt), e.nmse, logit_nmse_bound(qt),
                    exl3 ? "" : "  vs f64 host reference", ok ? "ok" : "FAIL");
            check(ok, "logits bound");
            if (!exl3) {
                const err e2 = compare(yv.data(), ysel.data(), yv.size());
                const bool ok2 = e2.rel_max <= backend_rel_bound(qt) && e2.nmse <= backend_nmse_bound(qt) &&
                        std::isfinite(e2.nmse);
                printf("  (c2) %-6s %-10s T=%lld %-7s rel %.3e (<= %.0e)  nmse %.3e (<= %.0e)  vs backend full-head mul_mat  %s\n",
                        bname, h.name.c_str(), (long long) T, ggml_type_name(qt), e2.rel_max, backend_rel_bound(qt),
                        e2.nmse, backend_nmse_bound(qt), ok2 ? "ok" : "FAIL");
                check(ok2, "(c2) backend full-head logits bound");
            }
        }
        if (exl3) {
            ggml_backend_tensor_get(yneg, yv.data(), 0, ggml_nbytes(yneg));
            const err e = compare(yv.data(), ysel.data(), yv.size());
            const bool ok = e.rel_max >= 0.5;
            printf("  (c) %-6s %-10s T=%lld neg-ctl rel %.3e (>= 0.5 expected: raw trellis rows)  %s\n", bname,
                    h.name.c_str(), (long long) T, e.rel_max, ok ? "ok" : "FAIL");
            check(ok, "negative control");
        }
        ggml_gallocr_free(ga);
        ggml_free(cg);
        ggml_backend_buffer_free(bw);
        ggml_free(cw);
    }
}

static void run_head(const head_case & h, const std::vector<ggml_backend_t> & backends) {
    const int64_t K = h.K, n_sel = (int64_t) h.ids.size();
    const bool exl3 = ggml_exl3_bits(h.type) != 0;
    printf("head %s: K %lld, N %lld, n_sel %lld\n", h.name.c_str(), (long long) K, (long long) h.N, (long long) n_sel);

    std::vector<std::pair<ggml_type, std::vector<uint8_t>>> compact;
    {
        auto f = build(h, GGML_TYPE_F32);
        if (exl3) {
            const err e = compare((const float *) f.data(), h.ref.data(), h.ref.size());
            const bool ok = e.rel_max <= 1e-5;
            printf("  (a) f32 rows vs double reference: max|diff| %.3e, rel %.3e (<= 1e-5)  %s\n", e.max_abs, e.rel_max, ok ? "ok" : "FAIL");
            check(ok, "(a) domain");
        } else {
            const bool ok = memcmp(f.data(), h.ref.data(), f.size()) == 0;
            printf("  (d) f32 rows byte-equal to to_float of the head rows: %s\n", ok ? "ok" : "FAIL");
            check(ok, "(d) plain rows");
        }
        compact.emplace_back(GGML_TYPE_F32, std::move(f));
    }
    double prev = -1.0;
    for (ggml_type qt : QTYPES) {
        auto q = build(h, qt);
        const auto f = to_f32(q, qt, K, n_sel);
        const err e = compare(f.data(), h.ref.data(), h.ref.size());
        // the ordering is required on the EXL3 heads; a plain head re-quantized to its own grid can be near exact
        const bool ok = e.nmse <= nmse_bound(qt) && (!exl3 || e.nmse > prev);
        printf("  (b) %-7s %6.1f KiB  max abs %.3e  NMSE %.3e (<= %.1e%s)  %s\n", ggml_type_name(qt),
                q.size() / 1024.0, e.max_abs, e.nmse, nmse_bound(qt), exl3 ? ", > previous type" : "", ok ? "ok" : "FAIL");
        check(ok, "(b) quantization");
        prev = e.nmse;
        compact.emplace_back(qt, std::move(q));
    }
    for (ggml_backend_t be : backends) {
        run_logits(be, h, compact);
    }
}

int main() {
    ggml_backend_load_all();
    std::vector<ggml_backend_t> backends;
    if (ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) {
        backends.push_back(ggml_backend_dev_init(cpu, nullptr));
    }
    if (ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) {
        backends.push_back(ggml_backend_dev_init(gpu, nullptr));
    }
    GGML_ASSERT(!backends.empty());
    printf("backends:");
    for (ggml_backend_t be : backends) printf(" %s", ggml_backend_name(be));
    printf("\n");

    // K = 1024 (8 input Hadamard blocks, 4 IQ4_XS/Q6_K super-blocks), N = 2048 (16 output blocks)
    run_head(make_exl3(6, 1024, 2048, 333, 8101), backends); // the 27B head is EXL3_6
    run_head(make_exl3(3, 1024, 2048, 333, 8102), backends);
    run_head(make_plain(GGML_TYPE_Q6_K, 1024, 2048, 333, 8103), backends);

    for (ggml_backend_t be : backends) ggml_backend_free(be);
    printf("%s: %d failing checks\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
