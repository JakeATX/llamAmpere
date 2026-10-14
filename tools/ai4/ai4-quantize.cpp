// llama-ai4-quantize: build an AI+ GGUF (AI4 = Ampere INT4 trellis, formerly D2) from a quantized template GGUF (the
// mixture), the BF16 source and an imatrix.
//
// Selection (which tensors become AI+ and which AI+ type):
//   - default (no --tensor-type / --tensor-list): every template tensor of type --replace (default iq4_xs) that is not
//     token_embd / output and fits --replace-type (ai4: ne0 % 1024 == 0; ai4u: ne0 % 128 == 0; ne1 % 16 == 0) becomes
//     --replace-type (default ai4, the original behaviour). 2-D tensors and 3-D expert stacks both qualify.
//   - --tensor-type REGEX=TYPE (repeatable, first match wins) and/or --tensor-list FILE (lines "name TYPE"; exact names,
//     checked before the regexes): exactly the matching tensors become TYPE, whatever their template type (Q6_K, Q8_0,
//     IQ4_XS, ...). TYPE = ai4 | ai4u | keep. Every other tensor keeps its template type and bytes, so AI+ tensors sit next
//     to Q6_K / Q8_0 / K-quant tensors in one file. Rates k = 2, 3, 4, 5, 6 (ai2u / ai3u / ai4u / ai5u / ai6u) are implemented.
// Selected tensors are re-encoded from the BF16 source on the GPU (tools/ai4/ai4-encode.cu) with the imatrix-driven block
// LDLQ of ggml-ai4.h; expert stacks (ne2 = n_expert) are encoded expert by expert with that expert's imatrix row.
//
// --selftest: encode the first 32 rows of three tensors on the GPU and on the CPU reference (ggml-ai4.h) and compare
// the imatrix-weighted error; also checks the CUDA backend's reconstruct kernel against the CPU decoder. Exit 1 on
// failure.
//
// usage: llama-ai4-quantize --template T.gguf --bf16 shard1.gguf [--bf16 shard2.gguf ...] --imatrix I.gguf
//                          (--out OUT.gguf | --selftest) [--replace iq4_xs] [--replace-type ai4|ai4u]
//                          [--tensor-type REGEX=TYPE ...] [--tensor-list FILE] [--max-tensors N]
//
// The output is written to OUT.gguf.tmp and renamed to OUT.gguf only after the last tensor is written, so an
// interrupted run never leaves a complete-looking file. general.file_type is set to LLAMA_FTYPE_MOSTLY_AI4.
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "imatrix-loader.h"
#include "ggml-ai4.h"
#include "ai4-encode.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

struct src_tensor {
    FILE *    f = nullptr;
    size_t    off = 0;
    ggml_type type = GGML_TYPE_COUNT;
    int64_t   ne[4] = { 1, 1, 1, 1 };
};

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static bool read_at(FILE * f, size_t off, void * dst, size_t n) {
    if (fseeko(f, (off_t) off, SEEK_SET) != 0) {
        return false;
    }
    return fread(dst, 1, n, f) == n;
}

// rows [r0, r0 + nr) of a 2-D source tensor as f32
static bool load_rows_f32(const src_tensor & s, int64_t r0, int64_t nr, std::vector<float> & out) {
    const int64_t K = s.ne[0];
    out.resize((size_t) K * nr);
    const size_t es = ggml_type_size(s.type);
    if (s.type != GGML_TYPE_BF16 && s.type != GGML_TYPE_F16 && s.type != GGML_TYPE_F32) {
        return false;
    }
    std::vector<uint8_t> raw((size_t) K * nr * es);
    if (!read_at(s.f, s.off + (size_t) r0 * K * es, raw.data(), raw.size())) {
        return false;
    }
    if (s.type == GGML_TYPE_BF16) {
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) raw.data(), out.data(), K * nr);
    } else if (s.type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw.data(), out.data(), K * nr);
    } else {
        memcpy(out.data(), raw.data(), raw.size());
    }
    return true;
}

// LDLQ factors for all rb-column rotation blocks of a tensor (rb = 256 for AI4, ai4u_rot_size(K) for AI4_U): U (for the
// CPU reference, [K/rb][rb][rb]) and G (for the GPU encoder, [K/rb][rb/8][8][rb])
static void build_ldlq(const std::vector<float> & lam, int64_t K, std::vector<float> & Uall, std::vector<float> & G,
                       std::vector<uint8_t> & ok, const int rb = AI4_HB) {
    const int64_t NB = K / rb;
    const int NCS = rb / AI4_TC;
    Uall.assign((size_t) NB * rb * rb, 0.0f);
    G.assign((size_t) NB * NCS * 8 * rb, 0.0f);
    ok.assign(NB, 0);
    const int nth = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (int t = 0; t < nth; ++t) {
        th.emplace_back([&, t]() {
            for (int64_t b = t; b < NB; b += nth) {
                float * U = Uall.data() + (size_t) b * rb * rb;
                ok[b] = (uint8_t) ai4_ldlq_prep_n(lam.data() + b * rb, U, rb);
                if (!ok[b]) {
                    continue;
                }
                for (int c = 0; c < NCS; ++c) {
                    const int c0 = c * AI4_TC;
                    double Ub[AI4_TC * AI4_TC], Ui[AI4_TC * AI4_TC];
                    for (int a = 0; a < AI4_TC; ++a) {
                        for (int bb = 0; bb < AI4_TC; ++bb) {
                            Ub[a * AI4_TC + bb] = U[(c0 + a) * rb + c0 + bb];
                        }
                    }
                    ai4_inv_upper8(Ub, Ui);
                    float * g = G.data() + (((size_t) b * NCS + c) * 8) * rb;
                    for (int a = 0; a < AI4_TC; ++a) {
                        for (int k = 0; k < rb; ++k) {
                            double s = 0.0;
                            for (int bb = a; bb < AI4_TC; ++bb) {   // Ui is upper triangular
                                s += Ui[a * AI4_TC + bb] * U[(c0 + bb) * rb + k];
                            }
                            g[a * rb + k] = (float) s;
                        }
                    }
                }
            }
        });
    }
    for (auto & x : th) {
        x.join();
    }
}

// --scale-mode (private A/B control, do not ship): per-128-group scale table [M][K/128] handed to the encoder in place
// of each group's own RMS. mode 1 "rank1": two-way additive least squares of log(per-rb-block row RMS), i.e.
// fit[r][b] = rowmean[r] + colmean[b] - grandmean; mode 2 "row": one scale per row (the whole-row RMS); mode 3 "span":
// one scale per row and 1024 K (that span's RMS), the scale granularity of the AIk_S types. The AI+ rotation
// is orthogonal inside each rb-column block, so the per-block RMS of the original-basis row is exactly the per-block RMS
// the encoder sees in the rotated basis; only the split into the two 128-groups of a 256 block is lost.
static void build_sfix(const float * W, int64_t K, int64_t M, int rb, int mode, std::vector<float> & sfix) {
    const int64_t NB = K / rb, NG = K / 128, gpb = rb / 128;
    std::vector<double> S((size_t) M * NB, 0.0);
    for (int64_t r = 0; r < M; ++r) {
        for (int64_t b = 0; b < NB; ++b) {
            double ss = 0.0;
            for (int64_t i = 0; i < rb; ++i) { const double v = W[(size_t) r * K + b * rb + i]; ss += v * v; }
            S[(size_t) r * NB + b] = sqrt(ss / (double) rb);
        }
    }
    sfix.assign((size_t) M * NG, 0.0f);
    if (mode == 3) {
        // one scale per row and 1024 K: the RMS over that span's rb blocks (a 1024 span is a whole number of them)
        const int64_t bps = 1024 / rb, nsp = K / 1024;
        for (int64_t r = 0; r < M; ++r) {
            for (int64_t sp = 0; sp < nsp; ++sp) {
                double ss = 0.0;
                for (int64_t b = 0; b < bps; ++b) { const double v = S[(size_t) r * NB + sp * bps + b]; ss += v * v; }
                const float v = (float) sqrt(ss / (double) bps);
                for (int64_t g = 0; g < 1024 / 128; ++g) sfix[(size_t) r * NG + sp * 8 + g] = v;
            }
        }
        return;
    }
    if (mode == 2) {
        for (int64_t r = 0; r < M; ++r) {
            double ss = 0.0;
            for (int64_t b = 0; b < NB; ++b) { const double v = S[(size_t) r * NB + b]; ss += v * v; }
            const float v = (float) sqrt(ss / (double) NB);
            for (int64_t g = 0; g < NG; ++g) sfix[(size_t) r * NG + g] = v;
        }
        return;
    }
    std::vector<double> L((size_t) M * NB), rm((size_t) M, 0.0), cm((size_t) NB, 0.0);
    for (size_t i = 0; i < L.size(); ++i) L[i] = log(S[i] > 1e-30 ? S[i] : 1e-30);
    for (int64_t r = 0; r < M; ++r) { double a = 0; for (int64_t b = 0; b < NB; ++b) a += L[(size_t) r * NB + b]; rm[r] = a / (double) NB; }
    for (int64_t b = 0; b < NB; ++b) { double a = 0; for (int64_t r = 0; r < M; ++r) a += L[(size_t) r * NB + b]; cm[b] = a / (double) M; }
    double gm = 0.0;
    for (int64_t r = 0; r < M; ++r) gm += rm[r];
    gm /= (double) M;
    for (int64_t r = 0; r < M; ++r) {
        for (int64_t b = 0; b < NB; ++b) {
            const float v = (float) exp(rm[r] + cm[b] - gm);
            for (int64_t g = 0; g < gpb; ++g) sfix[(size_t) r * NG + b * gpb + g] = v;
        }
    }
}

static void werr(const float * w, const float * r, const float * lam, int64_t K, int64_t n, double & rel, double & wrel) {
    double su = 0, nu = 0, sw = 0, nw = 0;
    for (int64_t i = 0; i < n; ++i) {
        const double e = (double) w[i] - r[i], l = lam ? lam[i % K] : 1.0;
        su += e * e; nu += (double) w[i] * w[i]; sw += l * e * e; nw += l * (double) w[i] * w[i];
    }
    rel = su / nu; wrel = sw / nw;
}

// AIQ (private): full input Hessians from llama-ai4-hessian, one file per input group (DIR/blk.L.<group>.hess)
static std::string aiq_group_of(const std::string & name) {
    int L; char w[128];
    if (sscanf(name.c_str(), "blk.%d.%127[^.].weight", &L, w) != 2) return "";
    static const std::map<std::string, std::string> m = {
        {"attn_qkv", "ssm_in"}, {"attn_gate", "ssm_in"}, {"attn_q", "attn_in"}, {"attn_k", "attn_in"}, {"attn_v", "attn_in"},
        {"ffn_gate", "ffn_in"}, {"ffn_up", "ffn_in"}, {"ffn_down", "ffn_down"}, {"attn_output", "attn_out"}, {"ssm_out", "ssm_out"},
    };
    auto it = m.find(w);
    if (it == m.end()) return "";
    return "blk." + std::to_string(L) + "." + it->second;
}

static bool aiq_load_hess(const std::string & fn, int64_t K, std::vector<float> & H) {
    FILE * f = fopen(fn.c_str(), "rb");
    if (!f) return false;
    char mg[4]; int32_t k32 = 0; double ntok = 0;
    if (fread(mg, 1, 4, f) != 4 || memcmp(mg, "AIQH", 4) != 0 || fread(&k32, 4, 1, f) != 1 || fread(&ntok, 8, 1, f) != 1 || k32 != K) {
        fclose(f); return false;
    }
    std::vector<float> pk((size_t) K * (K + 1) / 2);
    const bool ok = fread(pk.data(), sizeof(float), pk.size(), f) == pk.size();
    fclose(f);
    if (!ok) return false;
    H.resize((size_t) K * K);
    size_t o = 0;
    for (int64_t i = 0; i < K; ++i) for (int64_t j = 0; j <= i; ++j) { H[(size_t) i * K + j] = pk[o]; H[(size_t) j * K + i] = pk[o]; ++o; }
    return true;
}

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s --template T.gguf --bf16 S1.gguf [--bf16 S2.gguf ...] --imatrix I.gguf (--out O.gguf | --selftest)\n"
        "          [--replace iq4_xs] [--replace-type ai4|ai4u] [--tensor-type REGEX=TYPE ...] [--tensor-list FILE]\n"
        "          [--max-tensors N]\n"
        "\n"
        "Builds an AI+ GGUF on the GPU (CUDA device 0). Selected tensors are re-encoded from the BF16 source with the\n"
        "imatrix; every other tensor and all metadata are copied from the template.\n"
        "\n"
        "  --template T.gguf     the mixture to start from, normally an IQ4_XS quant of the same model made with the same imatrix\n"
        "  --bf16 S.gguf         BF16 (or F16/F32) source GGUF; repeat once per shard\n"
        "  --imatrix I.gguf      importance matrix (GGUF or legacy .dat); tensors without an entry are encoded without LDLQ\n"
        "  --out O.gguf          output path (written as O.gguf.tmp, renamed when complete)\n"
        "  --selftest            encode 32 rows of up to three selected tensors on GPU and CPU and compare; writes nothing\n"
        "  --replace TYPE        default selection: template tensor type to replace (default iq4_xs)\n"
        "  --replace-type TYPE   default selection: AI+ type to write, ai4 (1024 layout, K %% 1024) or ai4u (unit-128, K %% 128)\n"
        "  --tensor-type R=TYPE  explicit selection: tensors whose name matches regex R (std::regex_search) become TYPE\n"
        "                        (ai4 | ai4u | ai2u | ai3u | ai5u | ai6u | ai6uo | ai5un | keep); repeatable, first match\n"
        "                        wins; disables the default selection\n"
        "  --tensor-list FILE    explicit selection: lines \"tensor_name TYPE\" ('#' comments); exact names, checked first\n"
        "  --max-tensors N       replace at most N tensors (for tests)\n"
        "  AIQ (private): --hessian-dir D (full-K LDLQ with D/blk.L.<group>.hess, ai4 1024 layout, dense tensors only),\n"
        "  --hess-mode full|blockdiag|diag|imatrix, --damp X (default 0.01), --only REGEX (= --tensor-type ^REGEX$=ai4),\n"
        "  --overlay-only (write only the replaced tensors: a small GGUF for LLAMA_OVERLAY_GGUF), --werr (print the\n"
        "  H-weighted error of each tensor that has a Hessian), --rate-check (selftest: rate-k kernel at k=4 == AI4_U encoder),\n"
        "  --smul X / --smul5 X (scale seed multiplier override; the default is the type's ggml_ai4_scale_mul)\n"
        "  --hess-unit (EXLGAP, opt-in): with --hessian-dir, ai4u / ai5u tensors (K %% 256) also use the full-K LDLQ\n"
        "              (--hess-mode / --damp apply); without it they keep the imatrix path\n"
        "  --selftest-all / --selftest-max N: check every selected tensor (default: one per type / K class, 4 max).\n"
        "  --scale-mode group|rank1|row|span (private A/B control): where the fp16 per-128-group scale comes from. group\n"
        "    (default) = that group's own RMS, today's bytes; rank1 = rank-1 additive fit of log per-block row RMS;\n"
        "    row = one scale per row; span = one scale per row and 1024 K. Needs a unit-128 or span AI+ type; not\n"
        "    supported on the full-K (--hess-unit) paths. The span types (ai2s..ai6s) require row or span.\n"
        "  --repack-s IN.gguf --out OUT.gguf: repack an AIk_U file or overlay into the scale-free AIk_S span layout\n"
        "    (no encode; needs no --template / --bf16 / --imatrix). Fails unless every 1024 span of every tensor has one\n"
        "    scale per row, and checks that the dequantized floats are bit-identical to the AIk_U source.\n"
        "  --repack-verify-all: check every slab of every tensor in --repack-s (default: the first and last slab).\n"
        "  --refit (EXLGAP, opt-in, full-K paths only): after encoding, rescale each row's fp16 group scales by the\n"
        "              closed-form factor that minimizes the Hessian-weighted row error (trellis unchanged)\n", argv0);
}

// AI+ target types this tool writes. ai6uo / ai5un are private A/B controls for the constant search, do not ship.
static bool parse_ai_type(const std::string & s, ggml_type & t, bool & keep) {
    keep = false;
    if (s == "ai4") { t = GGML_TYPE_AI4; return true; }
    if (s == "ai4u" || s == "ai4_u") { t = GGML_TYPE_AI4_U; return true; }
    if (s == "ai3u" || s == "ai3_u") { t = GGML_TYPE_AI3_U; return true; }
    if (s == "ai2u" || s == "ai2_u") { t = GGML_TYPE_AI2_U; return true; }
    if (s == "ai5u" || s == "ai5_u") { t = GGML_TYPE_AI5_U; return true; }
    if (s == "ai6u" || s == "ai6_u") { t = GGML_TYPE_AI6_U; return true; }
    if (s == "ai6uo") { t = GGML_TYPE_AI6_UO; return true; }
    if (s == "ai5un") { t = GGML_TYPE_AI5_UN; return true; }
    if (s == "ai2s" || s == "ai2_s") { t = GGML_TYPE_AI2_S; return true; }
    if (s == "ai3s" || s == "ai3_s") { t = GGML_TYPE_AI3_S; return true; }
    if (s == "ai4s" || s == "ai4_s") { t = GGML_TYPE_AI4_S; return true; }
    if (s == "ai5s" || s == "ai5_s") { t = GGML_TYPE_AI5_S; return true; }
    if (s == "ai6s" || s == "ai6_s") { t = GGML_TYPE_AI6_S; return true; }
    if (s == "keep") { keep = true; t = GGML_TYPE_COUNT; return true; }
    fprintf(stderr, "unknown AI+ type %s (ai4 | ai4u | ai2u | ai3u | ai5u | ai6u | ai6uo | ai5un | ai2s..ai6s | keep)\n", s.c_str());
    return false;
}

// trellis bits per weight of an AI+ type
static int ai_kbits(ggml_type t) {
    const int r = ggml_ai4_rate(t);
    return r ? r : 4;
}

// bytes per 16-row slab and 128 K: the AIk_U unit (32 + 256k) or, for the span types, a span's eighth (4 + 256k), so
// (M/16)*(K/128)*ai_unit_bytes(t) is the tensor size in both layouts
static int64_t ai_unit_bytes(ggml_type t) {
    return ggml_ai4_is_span(t) ? AI4S_SPAN_BYTES(ai_kbits(t))/AI4S_UNITS_PER_SPAN : AI4U_UNIT_BYTES(ai_kbits(t));
}

static int64_t ai_kmod(ggml_type t) {
    return ggml_ai4_is_unit(t) ? 128 : 1024;
}

// the AIk_S type an AIk_U tensor repacks into (GGML_TYPE_COUNT: not a repackable type)
static ggml_type ai_unit_to_span(ggml_type t) {
    switch (t) {
        case GGML_TYPE_AI2_U: return GGML_TYPE_AI2_S;
        case GGML_TYPE_AI3_U: return GGML_TYPE_AI3_S;
        case GGML_TYPE_AI4_U: return GGML_TYPE_AI4_S;
        case GGML_TYPE_AI5_U: return GGML_TYPE_AI5_S;
        case GGML_TYPE_AI6_U: return GGML_TYPE_AI6_S;
        default:              return GGML_TYPE_COUNT;
    }
}

// --repack-s: rewrite an AIk_U file (or overlay) with its AIk_S twin, dropping the 7 redundant scale headers of every
// 1024 span. No encode and no GPU: the trellis bytes are copied. Fails if any span's 8 unit headers are not identical
// (that tensor was encoded with per-128-group scales), and checks that the dequantized floats are bit-identical to the
// AIk_U source on the first and last slab of every tensor (verify_all: every slab).
// a failed repack must not leave a half-written OUT.gguf.tmp behind
static int rp_fail(FILE * fo, const std::string & f_tmp) {
    fclose(fo);
    remove(f_tmp.c_str());
    return 1;
}

static int repack_s(const std::string & f_in, const std::string & f_out, bool verify_all) {
    ggml_context * cm = nullptr;
    gguf_init_params p = { /*no_alloc =*/ true, /*ctx =*/ &cm };
    gguf_context * in = gguf_init_from_file(f_in.c_str(), p);
    if (!in) { fprintf(stderr, "cannot open %s\n", f_in.c_str()); return 1; }
    FILE * fi = fopen(f_in.c_str(), "rb");
    if (!fi) { fprintf(stderr, "cannot open %s\n", f_in.c_str()); return 1; }
    const int64_t nt = gguf_get_n_tensors(in);
    const size_t tdoff = gguf_get_data_offset(in);
    gguf_context * out = gguf_init_empty();
    gguf_set_kv(out, in);
    std::vector<ggml_type> nty((size_t) nt, GGML_TYPE_COUNT);
    int n_rep = 0;
    for (int64_t i = 0; i < nt; ++i) {
        const char * name = gguf_get_tensor_name(in, i);
        ggml_tensor * t = ggml_get_tensor(cm, name);
        gguf_add_tensor(out, t);
        const ggml_type st = ai_unit_to_span(t->type);
        if (st == GGML_TYPE_COUNT) {
            continue;
        }
        if (t->ne[0] % AI4S_SPAN_K != 0 || t->ne[1] % AI4_TR != 0) {
            fprintf(stderr, "%s: %s needs K %% 1024 == 0 and M %% 16 == 0, got %lld x %lld\n", name, ggml_type_name(t->type),
                    (long long) t->ne[0], (long long) t->ne[1]);
            return 1;
        }
        nty[i] = st;
        gguf_set_tensor_type(out, name, st);
        n_rep++;
    }
    if (n_rep == 0) { fprintf(stderr, "%s has no AIk_U tensors to repack\n", f_in.c_str()); return 1; }
    const std::string f_tmp = f_out + ".tmp";
    FILE * fo = fopen(f_tmp.c_str(), "wb");
    if (!fo) { fprintf(stderr, "cannot write %s\n", f_tmp.c_str()); return 1; }
    const size_t meta = gguf_get_meta_size(out), align = gguf_get_alignment(out);
    std::vector<uint8_t> zeros(std::max(meta, align), 0);
    fwrite(zeros.data(), 1, meta, fo);
    std::vector<uint8_t> bu, bs;
    std::vector<float> du, ds;
    double t0 = now_s();
    int64_t n_ver = 0;
    for (int64_t i = 0; i < nt; ++i) {
        const char * name = gguf_get_tensor_name(in, i);
        const ggml_tensor * t = ggml_get_tensor(cm, name);
        const size_t szu = gguf_get_tensor_size(in, i);
        bu.resize(szu);
        if (!read_at(fi, tdoff + gguf_get_tensor_offset(in, i), bu.data(), szu)) { fprintf(stderr, "read %s failed\n", name); return rp_fail(fo, f_tmp); }
        size_t szo = szu;
        const uint8_t * src = bu.data();
        if (nty[i] != GGML_TYPE_COUNT) {
            const int      k  = ggml_ai4_rate(t->type);
            const uint32_t C  = ggml_ai4_trellis_mul(t->type);
            const int64_t  K  = t->ne[0], M = t->ne[1], E = t->ne[2]*t->ne[3];
            const size_t   sb = (size_t) ai4x_slab_bytes_k(K, k), sbS = (size_t) ai4s_slab_bytes_k(K, k);
            const int64_t  ns = M / AI4_TR;
            szo = (size_t) ns * E * sbS;
            bs.resize(szo);
            du.resize((size_t) AI4_TR * K);
            ds.resize((size_t) AI4_TR * K);
            for (int64_t e = 0; e < E; ++e) {
                for (int64_t sl = 0; sl < ns; ++sl) {
                    const uint8_t * su = bu.data() + ((size_t) e*ns + sl)*sb;
                    uint8_t * sd = bs.data() + ((size_t) e*ns + sl)*sbS;
                    if (!ai4s_pack_slab(k, su, sd, K)) {
                        fprintf(stderr, "%s expert %lld slab %lld: the scale is not constant inside a 1024 span "
                                "(this tensor is not row/span scaled)\n", name, (long long) e, (long long) sl);
                        return rp_fail(fo, f_tmp);
                    }
                    if (verify_all || sl == 0 || sl == ns - 1) {
                        aik_dequantize_slab_c(k, C, su, du.data(), K);
                        ai4s_dequantize_slab_c(k, C, sd, ds.data(), K);
                        if (memcmp(du.data(), ds.data(), du.size()*sizeof(float)) != 0) {
                            fprintf(stderr, "%s expert %lld slab %lld: the span decode differs from the unit decode\n",
                                    name, (long long) e, (long long) sl);
                            return rp_fail(fo, f_tmp);
                        }
                        n_ver++;
                    }
                }
            }
            src = bs.data();
            fprintf(stderr, "repack %-32s %-5s -> %-5s %6lld x %-6lld x %-4lld  %.3f -> %.3f GiB\n", name,
                    ggml_type_name(t->type), ggml_type_name(nty[i]), (long long) K, (long long) M, (long long) E,
                    szu/1073741824.0, szo/1073741824.0);
        }
        if (szo != gguf_get_tensor_size(out, i)) { fprintf(stderr, "size mismatch %s: %zu vs %zu\n", name, szo, gguf_get_tensor_size(out, i)); return rp_fail(fo, f_tmp); }
        fwrite(src, 1, szo, fo);
        fwrite(zeros.data(), 1, GGML_PAD(szo, align) - szo, fo);
    }
    std::vector<uint8_t> mbuf(meta);
    gguf_get_meta_data(out, mbuf.data());
    fseeko(fo, 0, SEEK_SET);
    fwrite(mbuf.data(), 1, meta, fo);
    if (fclose(fo) != 0) { fprintf(stderr, "close %s failed\n", f_tmp.c_str()); remove(f_tmp.c_str()); return 1; }
    if (rename(f_tmp.c_str(), f_out.c_str()) != 0) { fprintf(stderr, "rename %s -> %s failed\n", f_tmp.c_str(), f_out.c_str()); remove(f_tmp.c_str()); return 1; }
    fprintf(stderr, "wrote %s: %d tensors repacked to the span layout, %lld slabs verified bit-identical, %.0f s\n",
            f_out.c_str(), n_rep, (long long) n_ver, now_s() - t0);
    return 0;
}

int main(int argc, char ** argv) {
    std::string f_tmpl, f_imat, f_out, replace = "iq4_xs", replace_type = "ai4", f_list;
    std::vector<std::string> f_bf16;
    std::vector<std::pair<std::regex, std::string>> rules;
    std::vector<std::string> rule_src;
    bool selftest = false;
    int max_tensors = -1;
    std::string hess_dir, hess_mode = "full";
    double damp = 0.01;
    bool overlay_only = false, want_werr = false, rate_check = false;
    float smul_k5 = -1.0f;   // --smul5 / --smul: override the type's scale seed multiplier
    bool hess_unit = false, refit = false;   // EXLGAP --hess-unit, --refit
    int scale_mode = 0;   // --scale-mode: 0 group (default), 1 rank1, 2 row, 3 span
    const char * scale_mode_name = "group";
    std::string repack_in;       // --repack-s IN.gguf: AIk_U -> AIk_S, no encode
    bool repack_verify_all = false;
    bool selftest_all = false;   // --selftest-all: check every selected tensor, not one per (type, K class)
    int selftest_max = 4;        // --selftest-max N
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); } return argv[++i]; };
        if (a == "-h" || a == "--help") { print_usage(argv[0]); return 0; }
        else if (a == "--template") f_tmpl = next();
        else if (a == "--bf16") f_bf16.push_back(next());
        else if (a == "--imatrix") f_imat = next();
        else if (a == "--out") f_out = next();
        else if (a == "--replace") replace = next();
        else if (a == "--replace-type") replace_type = next();
        else if (a == "--tensor-list") f_list = next();
        else if (a == "--tensor-type") {
            const std::string v = next();
            const size_t eq = v.rfind('=');
            if (eq == std::string::npos || eq == 0) { fprintf(stderr, "--tensor-type wants REGEX=TYPE, got %s\n", v.c_str()); return 2; }
            try {
                rules.emplace_back(std::regex(v.substr(0, eq)), v.substr(eq + 1));
            } catch (const std::regex_error & e) {
                fprintf(stderr, "bad regex in --tensor-type %s: %s\n", v.c_str(), e.what());
                return 2;
            }
            rule_src.push_back(v);
        }
        else if (a == "--selftest") selftest = true;
        else if (a == "--max-tensors") max_tensors = atoi(next().c_str());
        else if (a == "--hessian-dir") hess_dir = next();
        else if (a == "--hess-mode") hess_mode = next();
        else if (a == "--damp") damp = atof(next().c_str());
        else if (a == "--only") { const std::string v = "^" + next() + "$"; rules.emplace_back(std::regex(v), "ai4"); rule_src.push_back(v + "=ai4"); }
        else if (a == "--overlay-only") overlay_only = true;
        else if (a == "--werr") want_werr = true;
        else if (a == "--rate-check") rate_check = true;
        else if (a == "--smul5" || a == "--smul") smul_k5 = (float) atof(next().c_str());
        else if (a == "--hess-unit") hess_unit = true;
        else if (a == "--refit") refit = true;
        else if (a == "--repack-s") repack_in = next();
        else if (a == "--repack-verify-all") repack_verify_all = true;
        else if (a == "--selftest-all") { selftest_all = true; selftest_max = 1 << 20; }
        else if (a == "--selftest-max") selftest_max = atoi(next().c_str());
        else if (a == "--scale-mode") {
            const std::string v = next();
            scale_mode = v == "group" ? 0 : v == "rank1" ? 1 : v == "row" ? 2 : v == "span" ? 3 : -1;
            if (scale_mode < 0) { fprintf(stderr, "--scale-mode wants group | rank1 | row | span, got %s\n", v.c_str()); return 2; }
            scale_mode_name = scale_mode == 0 ? "group" : scale_mode == 1 ? "rank1" : scale_mode == 2 ? "row" : "span";
        }
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); print_usage(argv[0]); return 2; }
    }
    if (!repack_in.empty()) {
        if (f_out.empty()) { fprintf(stderr, "--repack-s needs --out\n"); return 2; }
        std::error_code ec;
        if (std::filesystem::equivalent(repack_in, f_out, ec)) { fprintf(stderr, "--out %s is the input\n", f_out.c_str()); return 2; }
        return repack_s(repack_in, f_out, repack_verify_all);
    }
    if (f_tmpl.empty() || f_bf16.empty() || f_imat.empty() || (f_out.empty() && !selftest)) {
        print_usage(argv[0]);
        return 2;
    }
    if (!f_out.empty()) {
        std::error_code ec;
        for (const std::string & in : f_bf16) {
            if (std::filesystem::equivalent(in, f_out, ec)) { fprintf(stderr, "--out %s is an input file\n", f_out.c_str()); return 2; }
        }
        if (std::filesystem::equivalent(f_tmpl, f_out, ec) || std::filesystem::equivalent(f_imat, f_out, ec)) {
            fprintf(stderr, "--out %s is an input file\n", f_out.c_str());
            return 2;
        }
    }
    ggml_type rtype = GGML_TYPE_COUNT;
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * n = ggml_type_name((ggml_type) t);
        if (n && replace == n) rtype = (ggml_type) t;
    }
    if (rtype == GGML_TYPE_COUNT) { fprintf(stderr, "unknown --replace type %s\n", replace.c_str()); return 2; }
    ggml_type def_type;
    {
        bool keep;
        if (!parse_ai_type(replace_type, def_type, keep) || keep) { fprintf(stderr, "bad --replace-type %s\n", replace_type.c_str()); return 2; }
    }
    for (const auto & r : rules) {
        ggml_type t; bool keep;
        if (!parse_ai_type(r.second, t, keep)) return 2;
    }
    std::map<std::string, std::string> list_sel;
    if (!f_list.empty()) {
        std::ifstream lf(f_list);
        if (!lf) { fprintf(stderr, "cannot read --tensor-list %s\n", f_list.c_str()); return 2; }
        std::string line;
        while (std::getline(lf, line)) {
            const size_t h = line.find('#');
            if (h != std::string::npos) line.resize(h);
            std::istringstream ls(line);
            std::string n, ty;
            if (!(ls >> n)) continue;
            if (!(ls >> ty)) { fprintf(stderr, "--tensor-list line without a type: %s\n", line.c_str()); return 2; }
            ggml_type t; bool keep;
            if (!parse_ai_type(ty, t, keep)) return 2;
            list_sel[n] = ty;
        }
    }
    const bool explicit_sel = !rules.empty() || !f_list.empty();

    // template
    ggml_context * ctx_meta = nullptr;
    gguf_init_params gp = { /*no_alloc =*/ true, /*ctx =*/ &ctx_meta };
    gguf_context * tmpl = gguf_init_from_file(f_tmpl.c_str(), gp);
    if (!tmpl) { fprintf(stderr, "cannot open template %s\n", f_tmpl.c_str()); return 1; }
    FILE * ftmpl = fopen(f_tmpl.c_str(), "rb");

    // BF16 source shards
    std::map<std::string, src_tensor> src;
    for (const auto & fb : f_bf16) {
        ggml_context * cm = nullptr;
        gguf_init_params p2 = { true, &cm };
        gguf_context * g = gguf_init_from_file(fb.c_str(), p2);
        if (!g) { fprintf(stderr, "cannot open %s\n", fb.c_str()); return 1; }
        FILE * f = fopen(fb.c_str(), "rb");
        const size_t doff = gguf_get_data_offset(g);
        for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
            const char * name = gguf_get_tensor_name(g, i);
            const ggml_tensor * t = ggml_get_tensor(cm, name);
            src_tensor s;
            s.f = f; s.off = doff + gguf_get_tensor_offset(g, i); s.type = t->type;
            for (int d = 0; d < 4; ++d) s.ne[d] = t->ne[d];
            src[name] = s;
        }
    }
    fprintf(stderr, "source: %zu tensors from %zu file(s)\n", src.size(), f_bf16.size());

    // imatrix: per-column mean of x^2; expert stacks have one row (and one count) per expert
    common_imatrix im;
    if (!common_imatrix_load(f_imat, im)) { fprintf(stderr, "cannot load imatrix %s\n", f_imat.c_str()); return 1; }
    auto get_lam = [&](const std::string & name, int64_t K, int64_t e, int64_t n_e, std::vector<float> & lam) -> bool {
        auto it = im.entries.find(name);
        if (it == im.entries.end() || (int64_t) it->second.sums.size() != K * n_e) return false;
        lam.resize(K);
        const auto & en = it->second;
        const double cnt = en.counts.empty() ? 0.0 : (double) en.counts[(int64_t) en.counts.size() == n_e ? e : 0];
        const float * sm = en.sums.data() + e * K;
        if (!im.is_legacy && !(cnt > 0) && n_e > 1) {
            return false;   // expert never routed in the imatrix run: no information, encode without LDLQ
        }
        for (int64_t k = 0; k < K; ++k) {   // n_e == 1: the original AI4 rule (uniform lambda when the count is 0)
            lam[k] = im.is_legacy ? (cnt > 0 ? (float) (sm[k] / cnt) : sm[k]) : (cnt > 0 ? (float) (sm[k] / cnt) : 1.0f);
        }
        return true;
    };

    // selection
    const int64_t nt = gguf_get_n_tensors(tmpl);
    std::vector<ggml_type> repl(nt, GGML_TYPE_COUNT);   // COUNT = keep
    int n_repl = 0;
    std::map<std::string, int> n_by_type;
    for (int64_t i = 0; i < nt; ++i) {
        const char * name = gguf_get_tensor_name(tmpl, i);
        const ggml_tensor * t = ggml_get_tensor(ctx_meta, name);
        ggml_type want = GGML_TYPE_COUNT;
        bool chosen = false;
        if (explicit_sel) {
            std::string ty;
            auto li = list_sel.find(name);
            if (li != list_sel.end()) {
                ty = li->second;
            } else {
                for (const auto & r : rules) {
                    if (std::regex_search(std::string(name), r.first)) { ty = r.second; break; }
                }
            }
            if (!ty.empty()) {
                bool keep;
                parse_ai_type(ty, want, keep);
                chosen = !keep;
            }
        } else if (t->type == rtype && strcmp(name, "token_embd.weight") != 0 && strcmp(name, "output.weight") != 0) {
            want = def_type;
            chosen = true;
        }
        if (!chosen) {
            continue;
        }
        auto it = src.find(name);
        const bool shape_ok = it != src.end() && it->second.ne[0] == t->ne[0] && it->second.ne[1] == t->ne[1]
            && it->second.ne[2] == t->ne[2] && t->ne[3] == 1;
        const bool fits = t->ne[0] % ai_kmod(want) == 0 && t->ne[1] % 16 == 0;
        if (!shape_ok || !fits) {
            fprintf(stderr, "keep %s (%s %lld x %lld x %lld): %s for %s\n", name, ggml_type_name(t->type), (long long) t->ne[0],
                    (long long) t->ne[1], (long long) t->ne[2], !shape_ok ? "no matching BF16 source" : "shape does not tile",
                    ggml_type_name(want));
            continue;
        }
        if (max_tensors >= 0 && n_repl >= max_tensors) {
            continue;
        }
        repl[i] = want;
        n_repl++;
        n_by_type[std::string(ggml_type_name(t->type)) + " -> " + ggml_type_name(want)]++;
    }
    fprintf(stderr, "%d of %lld tensors -> AI+\n", n_repl, (long long) nt);
    for (const auto & kv : n_by_type) {
        fprintf(stderr, "  %4d  %s\n", kv.second, kv.first.c_str());
    }
    // lrms / smul / the trellis constant are per AI+ type (ggml_ai4_trellis_mul / ggml_ai4_scale_mul)
    auto ai_mul  = [](ggml_type t) { return ggml_ai4_trellis_mul(t); };
    auto ai_lrms = [](ggml_type t) { return aik_level_rms_c(ggml_ai4_trellis_mul(t)); };
    auto ai_smul = [&](ggml_type t) { return smul_k5 > 0.0f ? smul_k5 : ggml_ai4_scale_mul(t); };

    // encode rows [r0, r0 + nr) of one expert's matrix; returns encoded bytes in q
    auto encode_mat = [&](const std::string & name, const src_tensor & s, ggml_type ty, int64_t e, int64_t nr,
                          std::vector<uint8_t> & q, bool & has_im, float & ms, std::vector<float> * Wout,
                          std::vector<float> * lam_out) -> int {
        const int64_t K = s.ne[0], M = s.ne[1], E = s.ne[2];
        const int unit = ty != GGML_TYPE_AI4;   // the unit-based encoder (the span types encode as AIk_U, then pack)
        const bool span = ggml_ai4_is_span(ty);
        const int kbits = ai_kbits(ty);
        const uint32_t lmul = ai_mul(ty);
        const float lrms = ai_lrms(ty), smul = ai_smul(ty);
        const int rb = unit ? ai4u_rot_size(K) : AI4_HB;
        std::vector<float> W, lam, Uall, G;
        std::vector<uint8_t> ok;
        if (!load_rows_f32(s, e * M, nr, W)) { fprintf(stderr, "read source %s failed\n", name.c_str()); return 1; }
        std::vector<float> Hf;
        const std::string grp = aiq_group_of(name);
        const bool has_hf = !hess_dir.empty() && E == 1 && !grp.empty() && aiq_load_hess(hess_dir + "/" + grp + ".hess", K, Hf);
        if (!hess_dir.empty() && !has_hf) { fprintf(stderr, "ERROR: no Hessian for %s (%s)\n", name.c_str(), grp.c_str()); return 1; }
        if (span && K % AI4S_SPAN_K != 0) {
            fprintf(stderr, "%s: %s needs K %% 1024 == 0, got K = %lld\n", name.c_str(), ggml_type_name(ty), (long long) K);
            return 1;
        }
        if (span && has_hf && hess_mode != "imatrix") {
            fprintf(stderr, "%s: the span types take the imatrix path only (--hessian-dir with --hess-mode imatrix)\n", name.c_str());
            return 1;
        }
        if (has_hf && !unit && hess_mode != "imatrix") {   // AIQ full-K LDLQ (ai4 1024 layout)
            const int hmode = hess_mode == "blockdiag" ? 1 : hess_mode == "diag" ? 2 : 0;
            double we = -1.0;
            q.assign((size_t) (nr / 16) * (K / 128) * AI4U_UNIT_BYTES4, 0);
            double wp = -1.0;
            const int rc = ai4_cuda_encode_fullk_rate(W.data(), Hf.data(), K, nr, hmode, damp, lrms, smul, 0, lmul, refit ? 1 : 0,
                                                      q.data(), &ms, want_werr ? &we : nullptr, want_werr && refit ? &wp : nullptr);
            if (want_werr && refit) fprintf(stderr, "werr %s before refit: %.6e\n", name.c_str(), wp);
            if (rc) { fprintf(stderr, "encode %s failed (%d)\n", name.c_str(), rc); return 1; }
            if (want_werr) fprintf(stderr, "werr %s mode %s damp %g: %.6e\n", name.c_str(), hess_mode.c_str(), damp, we);
            has_im = true;
            if (Wout) *Wout = std::move(W);
            if (lam_out) *lam_out = std::vector<float>();
            return 0;
        }
        if (has_hf && unit && hess_unit && hess_mode != "imatrix" && K % 256 == 0) {   // EXLGAP: full-K LDLQ, AI4_U / AI5_U
            const int hmode = hess_mode == "blockdiag" ? 1 : hess_mode == "diag" ? 2 : 0;
            double we = -1.0, wp = -1.0;
            q.assign((size_t) (nr / 16) * (K / 128) * ai_unit_bytes(ty), 0);
            const int rc = ai4_cuda_encode_fullk_rate(W.data(), Hf.data(), K, nr, hmode, damp, lrms,
                                                      smul, kbits, lmul, refit ? 1 : 0, q.data(), &ms,
                                                      want_werr ? &we : nullptr, want_werr && refit ? &wp : nullptr);
            if (rc) { fprintf(stderr, "encode %s failed (%d)\n", name.c_str(), rc); return 1; }
            if (want_werr && refit) fprintf(stderr, "werr %s before refit: %.6e\n", name.c_str(), wp);
            if (want_werr) fprintf(stderr, "werr %s k%d mode %s damp %g: %.6e\n", name.c_str(), kbits, hess_mode.c_str(), damp, we);
            has_im = true;
            if (Wout) *Wout = std::move(W);
            if (lam_out) *lam_out = std::vector<float>();
            return 0;
        }
        has_im = get_lam(name, K, e, E, lam);
        if (has_im) {
            build_ldlq(lam, K, Uall, G, ok, rb);
        }
        // the span types encode into AIk_U bytes (with a span-constant scale table) and are packed below
        const size_t sbU = (size_t) (K / 128) * AI4U_UNIT_BYTES(kbits);
        std::vector<uint8_t> qu;
        if (span) {
            qu.assign((size_t) (nr / 16) * sbU, 0);
        } else {
            q.assign((size_t) (nr / 16) * (K / 128) * ai_unit_bytes(ty), 0);
        }
        uint8_t * qenc = span ? qu.data() : q.data();
        std::vector<float> sfix;
        if (scale_mode != 0) {
            if (!unit) { fprintf(stderr, "--scale-mode needs a unit-128 or span AI+ type, got %s\n", ggml_type_name(ty)); return 1; }
            build_sfix(W.data(), K, nr, rb, scale_mode, sfix);
        } else if (span) {
            fprintf(stderr, "%s needs --scale-mode row or span (a per-128-group scale cannot be stored)\n", ggml_type_name(ty));
            return 1;
        }
        const int rc = (kbits != 4 || lmul != AI4_MUL || !sfix.empty())
            ? ai4_cuda_encode_rate(W.data(), K, nr, has_im ? G.data() : nullptr, has_im ? ok.data() : nullptr, lrms, smul, kbits,
                                   lmul, qenc, &ms, sfix.empty() ? nullptr : sfix.data())
            : ai4_cuda_encode_layout(W.data(), K, nr, has_im ? G.data() : nullptr, has_im ? ok.data() : nullptr, lrms,
                                     smul, unit, qenc, &ms);
        if (rc) { fprintf(stderr, "encode %s failed (%d)\n", name.c_str(), rc); return 1; }
        if (want_werr && has_hf) {
            std::vector<float> Wh((size_t) K * nr);
            double we = -1.0;
            if ((unit ? ai4_cuda_reconstruct_rate(qenc, K, nr, kbits, lmul, 0, Wh.data()) : ai4_cuda_reconstruct_layout(qenc, K, nr, unit, Wh.data())) == 0 && ai4_cuda_werr(W.data(), Wh.data(), Hf.data(), K, nr, &we) == 0) {
                fprintf(stderr, "werr %s mode imatrix: %.6e\n", name.c_str(), we);
            }
        }
        if (span) {
            const size_t sbS = (size_t) ai4s_slab_bytes_k(K, kbits);
            q.assign((size_t) (nr / 16) * sbS, 0);
            for (int64_t sl = 0; sl < nr / 16; ++sl) {
                if (!ai4s_pack_slab(kbits, qu.data() + (size_t) sl * sbU, q.data() + (size_t) sl * sbS, K)) {
                    fprintf(stderr, "pack %s slab %lld: the scale is not constant inside a 1024 span (--scale-mode %s)\n",
                            name.c_str(), (long long) sl, scale_mode_name);
                    return 1;
                }
            }
        }
        if (Wout) *Wout = std::move(W);
        if (lam_out) *lam_out = has_im ? lam : std::vector<float>();
        return 0;
    };

    if (selftest) {
        std::vector<int64_t> pick;
        const char * want_names[3] = { "blk.20.attn_qkv.weight", "blk.39.attn_output.weight", "blk.5.ffn_down.weight" };
        if (selftest_all) {
            for (int64_t i = 0; i < nt; ++i) {
                if (repl[i] != GGML_TYPE_COUNT) pick.push_back(i);
            }
        } else {
        for (const char * n : want_names) {
            const int64_t id = gguf_find_tensor(tmpl, n);
            if (id >= 0 && repl[id] != GGML_TYPE_COUNT) pick.push_back(id);
        }
        // otherwise: the first selected tensor of each (type, K % 256 == 0) class, so the H128 path is covered
        if (pick.empty()) {
            std::map<std::pair<int, int>, int64_t> first;
            for (int64_t i = 0; i < nt; ++i) {
                if (repl[i] == GGML_TYPE_COUNT) continue;
                const ggml_tensor * t = ggml_get_tensor(ctx_meta, gguf_get_tensor_name(tmpl, i));
                const auto key = std::make_pair((int) repl[i], (int) (t->ne[0] % 256 == 0) * 2 + (t->ne[2] > 1));
                if (!first.count(key)) first[key] = i;
            }
            for (const auto & kv : first) pick.push_back(kv.second);
        }
        }
        int fail = 0, done = 0;
        for (const int64_t id : pick) {
            const std::string name = gguf_get_tensor_name(tmpl, id);
            const src_tensor & s = src[name];
            const ggml_type ty = repl[id];
            const int unit = ty != GGML_TYPE_AI4;
            const bool span = ggml_ai4_is_span(ty);
            const int kbits = ai_kbits(ty);
            const uint32_t lmul = ai_mul(ty);
            const float lrms = ai_lrms(ty), smul = ai_smul(ty);
            const int64_t K = s.ne[0], R = 32;
            const int64_t E = s.ne[2];
            int64_t e = E > 1 ? E / 2 : 0;   // a middle expert of a stack (the next one with imatrix data if it has none)
            std::vector<uint8_t> qg;
            std::vector<float> W, lam;
            bool has_im = false;
            float ms = 0;
            for (int64_t t = 0; t < E && !has_im; ++t) {
                e = (E / 2 + t) % E;
                if (encode_mat(name, s, ty, e, R, qg, has_im, ms, &W, &lam)) return 1;
            }
            if (!has_im) { fprintf(stderr, "selftest: no imatrix for %s (expert %lld), skipped\n", name.c_str(), (long long) e); continue; }
            std::vector<float> Uall, G;
            std::vector<uint8_t> ok;
            const int rb = unit ? ai4u_rot_size(K) : AI4_HB;
            build_ldlq(lam, K, Uall, G, ok, rb);
            const size_t sb = (size_t) ai4x_slab_bytes_k(K, kbits);          // AIk_U / AI4 bytes per slab
            const size_t sbS = (size_t) ai4s_slab_bytes_k(K, kbits);         // AIk_S bytes per slab
            // the span types are compared in the unit layout (unpacked), plus the two pack-identity checks below
            std::vector<uint8_t> qgu;
            if (span) {
                qgu.assign((size_t) (R / 16) * sb, 0);
                for (int sl = 0; sl < R / 16; ++sl) {
                    ai4s_unpack_slab(kbits, qg.data() + (size_t) sl * sbS, qgu.data() + (size_t) sl * sb, K);
                }
            }
            const uint8_t * qgU = span ? qgu.data() : qg.data();
            std::vector<float> sfix;
            if (scale_mode != 0) {
                build_sfix(W.data(), K, R, rb, scale_mode, sfix);
            }
            std::vector<uint8_t> qc((size_t) (R / 16) * sb);
            double t0 = now_s();
            std::vector<std::thread> th;
            for (int sl = 0; sl < R / 16; ++sl) {
                th.emplace_back([&, sl]() {
                    ai4_work * ws = ai4_work_new();
                    const float * dfix = sfix.empty() ? nullptr : sfix.data() + (size_t) sl * 16 * (K / 128);
                    if (kbits != 4 || lmul != AI4_MUL || dfix) {
                        aik_quantize_slab_rb_cf(ws, kbits, lmul, W.data() + (size_t) sl * 16 * K, K, lam.data(), Uall.data(), ok.data(),
                                               qc.data() + sl * sb, ai4u_rot_size(K), smul, dfix);
                    } else if (unit) {
                        ai4u_quantize_slab(ws, W.data() + (size_t) sl * 16 * K, K, lam.data(), Uall.data(), ok.data(), qc.data() + sl * sb);
                    } else {
                        ai4_quantize_slab(ws, W.data() + (size_t) sl * 16 * K, K, lam.data(), Uall.data(), ok.data(), qc.data() + sl * sb);
                    }
                    free(ws);
                });
            }
            for (auto & x : th) x.join();
            const double cpu_s = now_s() - t0;
            std::vector<float> rg((size_t) R * K), rc((size_t) R * K), rk((size_t) R * K);
            for (int sl = 0; sl < R / 16; ++sl) {
                ai4x_dequantize_slab_kc(unit, kbits, lmul, qgU + sl * sb, rg.data() + (size_t) sl * 16 * K, K);
                ai4x_dequantize_slab_kc(unit, kbits, lmul, qc.data() + sl * sb, rc.data() + (size_t) sl * 16 * K, K);
            }
            if (unit ? ai4_cuda_reconstruct_rate(qgU, K, R, kbits, lmul, 0, rk.data())
                     : ai4_cuda_reconstruct_layout(qgU, K, R, unit, rk.data())) return 1;
            if (span) {
                // 1. the span decoder must give exactly the floats the unpacked unit bytes give
                std::vector<float> rs((size_t) R * K);
                for (int sl = 0; sl < R / 16; ++sl) {
                    ai4s_dequantize_slab_c(kbits, lmul, qg.data() + (size_t) sl * sbS, rs.data() + (size_t) sl * 16 * K, K);
                }
                const bool dsame = memcmp(rs.data(), rg.data(), rs.size() * sizeof(float)) == 0;
                // 2. packing the CPU encoder's unit bytes must reproduce the file's span bytes (encoder == U + repack)
                std::vector<uint8_t> qp((size_t) (R / 16) * sbS, 0), qpc((size_t) (R / 16) * sbS, 0);
                bool pok = true;
                for (int sl = 0; sl < R / 16; ++sl) {
                    pok &= ai4s_pack_slab(kbits, qgu.data() + (size_t) sl * sb, qp.data() + (size_t) sl * sbS, K) != 0;
                    pok &= ai4s_pack_slab(kbits, qc.data()  + (size_t) sl * sb, qpc.data() + (size_t) sl * sbS, K) != 0;
                }
                const bool psame = pok && memcmp(qp.data(), qg.data(), qp.size()) == 0;
                printf("selftest %-28s span: decode == unpacked unit decode %s, repack(unpack) == file %s, cpu encode packs %s\n",
                       name.c_str(), dsame ? "YES" : "NO", psame ? "YES" : "NO", pok ? "YES" : "NO");
                fail |= !dsame || !psame;
            }
            if (kbits == 4 && unit && lmul == AI4_MUL && rate_check && scale_mode == 0) {
                // the generalized rate-k kernel at k = 4 must reproduce the AI4_U encoder byte for byte
                std::vector<uint8_t> q4(qg.size());
                float ms4 = 0;
                if (ai4_cuda_encode_rate(W.data(), K, R, G.data(), ok.data(), lrms, smul, 4, lmul, q4.data(), &ms4)) return 1;
                const bool same = memcmp(q4.data(), qg.data(), qg.size()) == 0;
                printf("selftest %-28s rate-k kernel at k=4 (scale-mode group) vs AI4_U encoder: %s\n", name.c_str(),
                       same ? "IDENTICAL" : "DIFFERENT");
                fail |= !same;
            }
            double rel_g, wrel_g, rel_c, wrel_c, ms2 = 0, wms = 0;
            werr(W.data(), rg.data(), lam.data(), K, R * K, rel_g, wrel_g);
            werr(W.data(), rc.data(), lam.data(), K, R * K, rel_c, wrel_c);
            for (int64_t i = 0; i < R * K; ++i) {
                ms2 = std::max(ms2, (double) fabsf(rk[i] - rg[i]));
                wms += (double) W[i] * W[i];
            }
            const double rmsw = sqrt(wms / (R * K));
            const double ratio = wrel_g / wrel_c;
            const bool pass = ratio <= 1.03 && ms2 <= 1e-4 * rmsw;
            printf("selftest %-28s %-5s scale %-5s K=%-6lld e=%-4lld gpu wrel %.4e rel %.4e | cpu wrel %.4e rel %.4e | gpu/cpu %.4f | "
                   "reconstruct max|diff|/rms %.2e | gpu %.1f ms, cpu %.1f s | %s\n", name.c_str(), ggml_type_name(ty), scale_mode_name,
                   (long long) K, (long long) e, wrel_g, rel_g, wrel_c, rel_c, ratio, ms2 / rmsw, ms, cpu_s, pass ? "PASS" : "FAIL");
            fail |= !pass;
            done++;
            if (done >= selftest_max) break;
        }
        printf("selftest %s (%d tensors)\n", (fail || done == 0) ? "FAIL" : "PASS", done);
        return (fail || done == 0) ? 1 : 0;
    }

    // output
    gguf_context * out = gguf_init_empty();
    gguf_set_kv(out, tmpl);
    if (n_repl > 0) {
        gguf_set_val_u32(out, "general.file_type", LLAMA_FTYPE_MOSTLY_AI4);
    }
    gguf_set_val_u32(out, "ai4.format_version", 2);
    gguf_set_val_str(out, "ai4.note", "AI+ (AI4 = Ampere INT4) trellis (ggml-ai4.h); selected tensors replaced from BF16 with imatrix block LDLQ");
    if (explicit_sel) {
        std::string sel;
        for (const auto & r : rule_src) sel += r + ";";
        if (!f_list.empty()) sel += "list:" + std::to_string(list_sel.size());
        gguf_set_val_str(out, "ai4.selection", sel.c_str());
    }
    std::vector<int64_t> oid(nt, -1);
    int64_t n_out = 0;
    for (int64_t i = 0; i < nt; ++i) {
        const char * name = gguf_get_tensor_name(tmpl, i);
        if (overlay_only && repl[i] == GGML_TYPE_COUNT) continue;
        gguf_add_tensor(out, ggml_get_tensor(ctx_meta, name));
        oid[i] = n_out++;
        if (repl[i] != GGML_TYPE_COUNT) {
            gguf_set_tensor_type(out, name, repl[i]);
        }
    }
    const std::string f_tmp = f_out + ".tmp";
    FILE * fo = fopen(f_tmp.c_str(), "wb");
    if (!fo) { fprintf(stderr, "cannot write %s\n", f_tmp.c_str()); return 1; }
    const size_t meta = gguf_get_meta_size(out);
    const size_t align = gguf_get_alignment(out);
    std::vector<uint8_t> zeros(std::max(meta, align), 0);
    fwrite(zeros.data(), 1, meta, fo);
    const size_t tdoff = gguf_get_data_offset(tmpl);
    double t_start = now_s(), gpu_ms_total = 0;
    int n_done = 0;
    std::vector<uint8_t> buf, qe;
    for (int64_t i = 0; i < nt; ++i) {
        const char * name = gguf_get_tensor_name(tmpl, i);
        size_t sz;
        if (oid[i] < 0) continue;
        if (repl[i] == GGML_TYPE_COUNT) {
            sz = gguf_get_tensor_size(tmpl, i);
            buf.resize(sz);
            if (!read_at(ftmpl, tdoff + gguf_get_tensor_offset(tmpl, i), buf.data(), sz)) { fprintf(stderr, "read %s failed\n", name); return 1; }
        } else {
            const src_tensor & s = src[name];
            const int64_t K = s.ne[0], M = s.ne[1], E = s.ne[2];
            const size_t per = (size_t) (M / 16) * (K / 128) * ai_unit_bytes(repl[i]);
            sz = per * E;
            buf.resize(sz);
            int n_im = 0;
            float ms_t = 0;
            for (int64_t e = 0; e < E; ++e) {
                bool has_im = false;
                float ms = 0;
                if (encode_mat(name, s, repl[i], e, M, qe, has_im, ms, nullptr, nullptr)) return 1;
                memcpy(buf.data() + e * per, qe.data(), per);
                n_im += has_im;
                ms_t += ms;
            }
            if (n_im < E) {
                fprintf(stderr, "WARNING: %s: %lld of %lld matrices without imatrix (encoded without LDLQ)\n", name,
                        (long long) (E - n_im), (long long) E);
            }
            gpu_ms_total += ms_t;
            n_done++;
            const double el = now_s() - t_start;
            fprintf(stderr, "[%d/%d] %-32s %-5s %6lld x %-6lld x %-4lld ldlq %lld/%lld  gpu %.1f s  elapsed %.0f s  eta %.0f s\n",
                    n_done, n_repl, name, ggml_type_name(repl[i]), (long long) K, (long long) M, (long long) E, (long long) n_im,
                    (long long) E, ms_t / 1e3, el, el / n_done * (n_repl - n_done));
        }
        if (sz != gguf_get_tensor_size(out, oid[i])) { fprintf(stderr, "size mismatch %s\n", name); return 1; }
        fwrite(buf.data(), 1, sz, fo);
        const size_t pad = GGML_PAD(sz, align) - sz;
        fwrite(zeros.data(), 1, pad, fo);
    }
    std::vector<uint8_t> mbuf(meta);
    gguf_get_meta_data(out, mbuf.data());
    fseeko(fo, 0, SEEK_SET);
    fwrite(mbuf.data(), 1, meta, fo);
    if (fclose(fo) != 0) { fprintf(stderr, "close %s failed\n", f_tmp.c_str()); return 1; }
    if (rename(f_tmp.c_str(), f_out.c_str()) != 0) { fprintf(stderr, "rename %s -> %s failed\n", f_tmp.c_str(), f_out.c_str()); return 1; }
    fprintf(stderr, "wrote %s: %d tensors AI+, gpu %.0f s, wall %.0f s\n", f_out.c_str(), n_done, gpu_ms_total / 1e3, now_s() - t_start);
    return 0;
}
