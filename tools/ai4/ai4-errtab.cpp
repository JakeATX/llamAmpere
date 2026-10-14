// llama-ai4-errtab (AIQ3, private): per-tensor x per-tier measured error tables for the AI+ byte-budget allocator
// (tools/ai4/aiq_alloc.py), plus a CPU reference tier encoder for small models and tests. CPU only, no GPU.
//
// Table mode: every tensor of each candidate GGUF (one GGUF per tier: a full encode, an overlay from
// llama-ai4-quantize --overlay-only, or a template) that also exists in the BF16 source is decoded on the CPU and compared
// with the BF16 weights. One TSV row per (tensor, candidate):
//   tensor  tag  type  bytes  n_weights  sse  ref  wsse  wref  hsse
//     sse  = sum (W - What)^2                   ref  = sum W^2
//     wsse = sum_k lambda_k (W - What)^2        wref = sum_k lambda_k W^2   (lambda = imatrix mean x_k^2 of that input
//            column: the diagonal-Hessian estimate of the output error energy ||dW x||^2 summed over calibration tokens)
//     hsse = sum_rows dw^T H dw with the full input Hessian of llama-ai4-hessian (--hessian-dir), -1 when there is none
//   Tensors without an imatrix entry get wsse = wref = -1. Types: AI4, AI4_U, AI5_U (ggml-ai4.h slab decoders) and every
//   ggml type with a row to_float (Q*_K, IQ*, Q8_0, F16, ...). EXL3 tensors need their suh/svh side vectors and are skipped.
//
// CPU encode mode (--cpu-encode TYPE): writes a GGUF in which the source tensors matching --re (default: every 2-D
// weight that tiles) are encoded to TYPE on the CPU (ai4u / ai5u with the ggml-ai4.h reference encoder and imatrix block
// LDLQ; any other ggml type through ggml_quantize_chunk); every other tensor and all metadata are copied. Meant for tiny
// models (unit tests) and small tensors; real models use the GPU encoder in llama-ai4-quantize.
//
// usage: llama-ai4-errtab --bf16 S.gguf [--bf16 ...] [--imatrix I.gguf] --cand TAG=F.gguf [--cand ...] [--re REGEX]
//                         [--hessian-dir D] [--threads N] --out T.tsv
//        llama-ai4-errtab --cpu-encode TYPE --bf16 S.gguf [--imatrix I.gguf] [--re REGEX] [--overlay-only] [--threads N] --out O.gguf
#include "ggml.h"
#include "gguf.h"
#include "imatrix-loader.h"
#include "ggml-ai4.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <regex>
#include <string>
#include <thread>
#include <vector>

struct gfile {
    std::string     path;
    gguf_context *  g   = nullptr;
    ggml_context *  ctx = nullptr;
    FILE *          f   = nullptr;
    size_t          doff = 0;
};

static bool open_gguf(const std::string & p, gfile & out) {
    out.path = p;
    gguf_init_params gp = { /*no_alloc =*/ true, /*ctx =*/ &out.ctx };
    out.g = gguf_init_from_file(p.c_str(), gp);
    if (!out.g) return false;
    out.f = fopen(p.c_str(), "rb");
    out.doff = gguf_get_data_offset(out.g);
    return out.f != nullptr;
}

static bool read_raw(const gfile & gf, int64_t id, std::vector<uint8_t> & buf) {
    const size_t n = gguf_get_tensor_size(gf.g, id);
    buf.resize(n);
    if (fseeko(gf.f, (off_t) (gf.doff + gguf_get_tensor_offset(gf.g, id)), SEEK_SET) != 0) return false;
    return fread(buf.data(), 1, n, gf.f) == n;
}

// every AI+ type: AI4 (1024 slab layout) and the unit-128 layout at rate 2..6 (AI2_U ... AI6_U + the private controls)
static bool is_ai(ggml_type t) { return ggml_ai4_rate(t) != 0; }
static int ai_k(ggml_type t) { const int k = ggml_ai4_rate(t); return k ? k : 4; }

// decode a whole tensor (ne0 x ne1 x ne2) to f32; false if the type has no CPU decoder here
static bool decode(ggml_type t, const int64_t * ne, const uint8_t * src, std::vector<float> & out, int nth) {
    const int64_t K = ne[0], M = ne[1], E = ne[2];
    out.resize((size_t) K * M * E);
    if (t == GGML_TYPE_F32) { memcpy(out.data(), src, out.size() * 4); return true; }
    if (is_ai(t)) {
        if (M % 16 || K % 128) return false;
        const int unit = ggml_ai4_is_unit(t), k = ai_k(t);
        const uint32_t C = ggml_ai4_trellis_mul(t);
        const size_t sb = (size_t) ai4x_slab_bytes_k(K, k);
        const int64_t ns = E * M / 16;
        std::atomic<int64_t> next(0);
        std::vector<std::thread> th;
        for (int i = 0; i < nth; ++i) th.emplace_back([&]() {
            for (int64_t s; (s = next++) < ns;) ai4x_dequantize_slab_kc(unit, k, C, src + s * sb, out.data() + (size_t) s * 16 * K, K);
        });
        for (auto & x : th) x.join();
        return true;
    }
    const ggml_type_traits * tr = ggml_get_type_traits(t);
    if (!tr || !tr->to_float || K % tr->blck_size) return false;
    const size_t rs = ggml_row_size(t, K);
    const int64_t nr = M * E;
    std::atomic<int64_t> next(0);
    std::vector<std::thread> th;
    for (int i = 0; i < nth; ++i) th.emplace_back([&]() {
        for (int64_t r; (r = next.fetch_add(64)) < nr;) {
            const int64_t n = std::min<int64_t>(64, nr - r);
            tr->to_float(src + r * rs, out.data() + (size_t) r * K, n * K);
        }
    });
    for (auto & x : th) x.join();
    return true;
}

// full input Hessians from llama-ai4-hessian: DIR/blk.L.<group>.hess (packed lower triangle)
static std::string hess_group_of(const std::string & name) {
    int L; char w[128];
    if (sscanf(name.c_str(), "blk.%d.%127[^.].weight", &L, w) != 2) return "";
    static const std::map<std::string, std::string> m = {
        {"attn_qkv", "ssm_in"}, {"attn_gate", "ssm_in"}, {"attn_q", "attn_in"}, {"attn_k", "attn_in"}, {"attn_v", "attn_in"},
        {"ffn_gate", "ffn_in"}, {"ffn_up", "ffn_in"}, {"ffn_down", "ffn_down"}, {"attn_output", "attn_out"}, {"ssm_out", "ssm_out"},
    };
    auto it = m.find(w);
    return it == m.end() ? "" : "blk." + std::to_string(L) + "." + it->second;
}

static bool load_hess(const std::string & fn, int64_t K, std::vector<float> & H) {
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

// per-expert lambda (mean x^2 per input column), same rule as llama-ai4-quantize
static bool get_lam(const common_imatrix & im, const std::string & name, int64_t K, int64_t e, int64_t n_e, std::vector<float> & lam) {
    auto it = im.entries.find(name);
    if (it == im.entries.end() || (int64_t) it->second.sums.size() != K * n_e) return false;
    const auto & en = it->second;
    const double cnt = en.counts.empty() ? 0.0 : (double) en.counts[(int64_t) en.counts.size() == n_e ? e : 0];
    if (!im.is_legacy && !(cnt > 0) && n_e > 1) return false;
    lam.resize(K);
    const float * sm = en.sums.data() + e * K;
    for (int64_t k = 0; k < K; ++k) lam[k] = im.is_legacy ? (cnt > 0 ? (float) (sm[k] / cnt) : sm[k]) : (cnt > 0 ? (float) (sm[k] / cnt) : 1.0f);
    return true;
}

struct errs { double sse = 0, ref = 0, wsse = 0, wref = 0, hsse = 0; };

static errs measure(const float * W, const float * R, int64_t K, int64_t M, int64_t E, const std::vector<std::vector<float>> & lam,
                    const std::vector<float> * H, int nth) {
    const int64_t nr = M * E;
    std::vector<errs> part(nth);
    std::atomic<int64_t> next(0);
    std::vector<std::thread> th;
    for (int t = 0; t < nth; ++t) th.emplace_back([&, t]() {
        errs & p = part[t];
        std::vector<double> d(K);
        for (int64_t r; (r = next++) < nr;) {
            const float * w = W + (size_t) r * K, * q = R + (size_t) r * K;
            const std::vector<float> & l = lam[r / M];
            double s = 0, rf = 0, ws = 0, wr = 0;
            for (int64_t k = 0; k < K; ++k) {
                const double e = (double) w[k] - q[k];
                d[k] = e;
                s += e * e; rf += (double) w[k] * w[k];
                if (!l.empty()) { ws += l[k] * e * e; wr += l[k] * (double) w[k] * w[k]; }
            }
            p.sse += s; p.ref += rf; p.wsse += ws; p.wref += wr;
            if (H) {
                double h = 0;
                for (int64_t i = 0; i < K; ++i) {
                    const float * hi = H->data() + (size_t) i * K;
                    double a = 0;
                    for (int64_t j = 0; j < K; ++j) a += hi[j] * d[j];
                    h += d[i] * a;
                }
                p.hsse += h;
            }
        }
    });
    for (auto & x : th) x.join();
    errs o;
    for (auto & p : part) { o.sse += p.sse; o.ref += p.ref; o.wsse += p.wsse; o.wref += p.wref; o.hsse += p.hsse; }
    return o;
}

static bool load_f32(const gfile & gf, int64_t id, const ggml_tensor * t, std::vector<float> & out, int nth) {
    std::vector<uint8_t> raw;
    if (!read_raw(gf, id, raw)) return false;
    const int64_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_BF16) { out.resize(n); ggml_bf16_to_fp32_row((const ggml_bf16_t *) raw.data(), out.data(), n); return true; }
    if (t->type == GGML_TYPE_F16) { out.resize(n); ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw.data(), out.data(), n); return true; }
    return decode(t->type, t->ne, raw.data(), out, nth);
}

static void usage(const char * a0) {
    fprintf(stderr,
        "usage: %s --bf16 S.gguf [--bf16 ...] [--imatrix I.gguf] --cand TAG=F.gguf [--cand ...] [--re REGEX]\n"
        "          [--hessian-dir D] [--threads N] --out T.tsv\n"
        "       %s --cpu-encode TYPE --bf16 S.gguf [--imatrix I.gguf] [--re REGEX] [--overlay-only] [--threads N] --out O.gguf\n"
        "Per-tensor x per-tier measured error table (CPU) for aiq_alloc.py; or a CPU reference tier encode (small models).\n"
        "  --cand TAG=F   a tier: every tensor of F that is also in the BF16 source gets a row with tier TAG\n"
        "  --re REGEX     only tensors whose name matches (std::regex_search)\n"
        "  --hessian-dir  full input Hessians (llama-ai4-hessian) -> hsse column (dense tensors only, K^2 per row)\n"
        "  --cpu-encode   ai2u | ai3u | ai4u | ai5u | ai6u | any ggml type name (q8_0, q6_K, q5_K, q4_K, iq4_xs, ...)\n"
        "  --overlay-only with --cpu-encode: write only the encoded tensors (a tier overlay)\n"
        "  --smul X       with --cpu-encode ai2u/ai3u/ai5u/ai6u: scale-seed multiplier override (default: the type's;\n"
        "                 encoder only, no format change)\n", a0, a0);
}

static int cpu_encode(const std::string & tname, const std::vector<gfile> & srcs, const common_imatrix * im, const std::string & re_s,
                      const std::string & out_p, int nth, bool overlay_only, float smul) {
    ggml_type T = GGML_TYPE_COUNT;
    if (tname == "ai4u" || tname == "ai4_u") T = GGML_TYPE_AI4_U;
    else if (tname == "ai5u" || tname == "ai5_u") T = GGML_TYPE_AI5_U;
    else if (tname == "ai2u" || tname == "ai2_u") T = GGML_TYPE_AI2_U;
    else if (tname == "ai3u" || tname == "ai3_u") T = GGML_TYPE_AI3_U;
    else if (tname == "ai6u" || tname == "ai6_u") T = GGML_TYPE_AI6_U;
    else for (int t = 0; t < GGML_TYPE_COUNT; ++t) { const char * n = ggml_type_name((ggml_type) t); if (n && tname == n) T = (ggml_type) t; }
    if (T == GGML_TYPE_COUNT || T == GGML_TYPE_AI4) { fprintf(stderr, "--cpu-encode: unsupported type %s\n", tname.c_str()); return 2; }
    if (srcs.size() != 1) { fprintf(stderr, "--cpu-encode takes exactly one --bf16\n"); return 2; }
    const gfile & s = srcs[0];
    const std::regex re(re_s.empty() ? std::string(".*\\.weight$") : re_s);
    gguf_context * out = gguf_init_empty();
    gguf_set_kv(out, s.g);
    const int64_t nt = gguf_get_n_tensors(s.g);
    std::vector<ggml_type> want(nt, GGML_TYPE_COUNT);
    for (int64_t i = 0; i < nt; ++i) {
        const char * name = gguf_get_tensor_name(s.g, i);
        ggml_tensor * t = ggml_get_tensor(s.ctx, name);
        const bool src_ok = t->type == GGML_TYPE_BF16 || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_F32;
        const bool tiles = is_ai(T) ? (t->ne[0] % 128 == 0 && t->ne[1] % 16 == 0) : (t->ne[0] % ggml_blck_size(T) == 0);
        if (src_ok && ggml_n_dims(t) >= 2 && t->ne[3] == 1 && tiles && std::regex_search(std::string(name), re)) {
            want[i] = T;
        }
        if (overlay_only && want[i] == GGML_TYPE_COUNT) continue;
        gguf_add_tensor(out, t);
        if (want[i] != GGML_TYPE_COUNT) gguf_set_tensor_type(out, name, T);
    }
    const std::string tmp = out_p + ".tmp";
    FILE * fo = fopen(tmp.c_str(), "wb");
    if (!fo) { fprintf(stderr, "cannot write %s\n", tmp.c_str()); return 1; }
    const size_t meta = gguf_get_meta_size(out), align = gguf_get_alignment(out);
    std::vector<uint8_t> zeros(std::max(meta, align), 0), raw, q;
    fwrite(zeros.data(), 1, meta, fo);
    int n_enc = 0;
    for (int64_t i = 0, oi = 0; i < nt; ++i) {
        const char * name = gguf_get_tensor_name(s.g, i);
        ggml_tensor * t = ggml_get_tensor(s.ctx, name);
        if (overlay_only && want[i] == GGML_TYPE_COUNT) continue;
        const int64_t o = oi++;
        if (want[i] == GGML_TYPE_COUNT) {
            if (!read_raw(s, i, raw)) return 1;
            q.swap(raw);
        } else {
            std::vector<float> W;
            if (!load_f32(s, i, t, W, nth)) { fprintf(stderr, "read %s failed\n", name); return 1; }
            const int64_t K = t->ne[0], M = t->ne[1], E = t->ne[2];
            q.assign(gguf_get_tensor_size(out, o), 0);
            for (int64_t e = 0; e < E; ++e) {
                std::vector<float> lam;
                const bool has = im && get_lam(*im, name, K, e, E, lam);
                const float * Wm = W.data() + (size_t) e * M * K;
                if (is_ai(T)) {
                    const int k = ai_k(T);
                    const uint32_t C = ggml_ai4_trellis_mul(T);
                    const float sm = smul > 0.0f ? smul : ggml_ai4_scale_mul(T);
                    const size_t sb = (size_t) ai4x_slab_bytes_k(K, k);
                    uint8_t * dst = q.data() + (size_t) e * (M / 16) * sb;
                    std::atomic<int64_t> next(0);
                    std::vector<std::thread> th;
                    for (int x = 0; x < nth; ++x) th.emplace_back([&]() {
                        ai4_work * ws = ai4_work_new();
                        for (int64_t sl; (sl = next++) < M / 16;) {
                            const float * l = has ? lam.data() : nullptr;
                            if (k == 4 && C == AI4_MUL) ai4u_quantize_slab(ws, Wm + (size_t) sl * 16 * K, K, l, nullptr, nullptr, dst + sl * sb);
                            else aik_quantize_slab_rb_c(ws, k, C, Wm + (size_t) sl * 16 * K, K, l, nullptr, nullptr, dst + sl * sb,
                                                        ai4u_rot_size(K), sm);
                        }
                        free(ws);
                    });
                    for (auto & x : th) x.join();
                } else {
                    const size_t rs = ggml_row_size(T, K);
                    ggml_quantize_chunk(T, Wm, q.data() + (size_t) e * M * rs, 0, M, K, has ? lam.data() : nullptr);
                }
            }
            n_enc++;
            fprintf(stderr, "enc %-32s %s -> %s %lld x %lld x %lld%s\n", name, ggml_type_name(t->type), ggml_type_name(T),
                    (long long) K, (long long) M, (long long) E, im ? "" : " (no imatrix)");
        }
        if (q.size() != gguf_get_tensor_size(out, o)) { fprintf(stderr, "size mismatch %s\n", name); return 1; }
        fwrite(q.data(), 1, q.size(), fo);
        fwrite(zeros.data(), 1, GGML_PAD(q.size(), align) - q.size(), fo);
    }
    std::vector<uint8_t> mb(meta);
    gguf_get_meta_data(out, mb.data());
    fseeko(fo, 0, SEEK_SET);
    fwrite(mb.data(), 1, meta, fo);
    if (fclose(fo) != 0 || rename(tmp.c_str(), out_p.c_str()) != 0) { fprintf(stderr, "write %s failed\n", out_p.c_str()); return 1; }
    fprintf(stderr, "wrote %s: %d tensors -> %s\n", out_p.c_str(), n_enc, ggml_type_name(T));
    return 0;
}

int main(int argc, char ** argv) {
    std::vector<std::string> f_bf16;
    std::vector<std::pair<std::string, std::string>> cands;
    std::string f_imat, f_out, re_s, hess_dir, enc_type;
    bool overlay_only = false;
    int nth = std::max(1u, std::min(4u, std::thread::hardware_concurrency()));
    float smul = -1.0f;   // encoder-only scale-seed multiplier override; the default is the type's ggml_ai4_scale_mul
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { usage(argv[0]); exit(2); } return argv[++i]; };
        if (a == "--bf16") f_bf16.push_back(next());
        else if (a == "--imatrix") f_imat = next();
        else if (a == "--cand") {
            const std::string v = next(); const size_t eq = v.find('=');
            if (eq == std::string::npos || eq == 0) { fprintf(stderr, "--cand wants TAG=FILE\n"); return 2; }
            cands.emplace_back(v.substr(0, eq), v.substr(eq + 1));
        }
        else if (a == "--re") re_s = next();
        else if (a == "--hessian-dir") hess_dir = next();
        else if (a == "--threads") nth = std::max(1, atoi(next().c_str()));
        else if (a == "--out") f_out = next();
        else if (a == "--cpu-encode") enc_type = next();
        else if (a == "--overlay-only") overlay_only = true;
        else if (a == "--smul") smul = (float) atof(next().c_str());
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); usage(argv[0]); return 2; }
    }
    if (f_bf16.empty() || f_out.empty() || (enc_type.empty() && cands.empty())) { usage(argv[0]); return 2; }
    std::vector<gfile> srcs(f_bf16.size());
    for (size_t i = 0; i < f_bf16.size(); ++i) if (!open_gguf(f_bf16[i], srcs[i])) { fprintf(stderr, "cannot open %s\n", f_bf16[i].c_str()); return 1; }
    common_imatrix im;
    const bool has_im = !f_imat.empty();
    if (has_im && !common_imatrix_load(f_imat, im)) { fprintf(stderr, "cannot load imatrix %s\n", f_imat.c_str()); return 1; }
    if (!enc_type.empty()) return cpu_encode(enc_type, srcs, has_im ? &im : nullptr, re_s, f_out, nth, overlay_only, smul);

    std::map<std::string, std::pair<int, int64_t>> src_of;   // name -> (file, id)
    for (size_t fi = 0; fi < srcs.size(); ++fi)
        for (int64_t i = 0; i < gguf_get_n_tensors(srcs[fi].g); ++i) src_of[gguf_get_tensor_name(srcs[fi].g, i)] = { (int) fi, i };
    std::vector<gfile> cf(cands.size());
    for (size_t c = 0; c < cands.size(); ++c) if (!open_gguf(cands[c].second, cf[c])) { fprintf(stderr, "cannot open %s\n", cands[c].second.c_str()); return 1; }
    const bool use_re = !re_s.empty();
    const std::regex re(use_re ? re_s : std::string(".*"));

    FILE * fo = fopen((f_out + ".tmp").c_str(), "w");
    if (!fo) { fprintf(stderr, "cannot write %s\n", f_out.c_str()); return 1; }
    fprintf(fo, "# llama-ai4-errtab: errors vs BF16 source; wsse/wref imatrix-weighted (-1 = no imatrix), hsse full-Hessian (-1 = none)\n");
    fprintf(fo, "tensor\ttag\ttype\tbytes\tn_weights\tsse\tref\twsse\twref\thsse\n");
    // tensor-major: decode the BF16 source once per tensor, then every candidate
    std::vector<std::string> names;
    for (auto & kv : src_of) if (!use_re || std::regex_search(kv.first, re)) names.push_back(kv.first);
    int n_rows = 0, n_skip = 0;
    for (const std::string & name : names) {
        std::vector<std::pair<size_t, int64_t>> hits;
        for (size_t c = 0; c < cf.size(); ++c) { const int64_t id = gguf_find_tensor(cf[c].g, name.c_str()); if (id >= 0) hits.emplace_back(c, id); }
        if (hits.empty()) continue;
        const gfile & sf = srcs[src_of[name].first];
        const ggml_tensor * st = ggml_get_tensor(sf.ctx, name.c_str());
        if (ggml_n_dims(st) < 2) continue;
        std::vector<float> W;
        if (!load_f32(sf, src_of[name].second, st, W, nth)) { fprintf(stderr, "skip %s: source type %s not readable\n", name.c_str(), ggml_type_name(st->type)); continue; }
        const int64_t K = st->ne[0], M = st->ne[1], E = st->ne[2];
        std::vector<std::vector<float>> lam(E);
        bool any_lam = false;
        if (has_im) for (int64_t e = 0; e < E; ++e) any_lam |= get_lam(im, name, K, e, E, lam[e]);
        std::vector<float> H;
        const std::string grp = hess_group_of(name);
        const bool has_h = !hess_dir.empty() && E == 1 && !grp.empty() && load_hess(hess_dir + "/" + grp + ".hess", K, H);
        for (auto & h : hits) {
            const gfile & c = cf[h.first];
            const ggml_tensor * ct = ggml_get_tensor(c.ctx, name.c_str());
            if (ct->ne[0] != K || ct->ne[1] != M || ct->ne[2] != E) { fprintf(stderr, "skip %s in %s: shape differs\n", name.c_str(), cands[h.first].first.c_str()); n_skip++; continue; }
            std::vector<float> R;
            if (!load_f32(c, h.second, ct, R, nth)) {
                fprintf(stderr, "skip %s in %s: no CPU decoder for %s\n", name.c_str(), cands[h.first].first.c_str(), ggml_type_name(ct->type));
                n_skip++; continue;
            }
            const errs e = measure(W.data(), R.data(), K, M, E, lam, has_h ? &H : nullptr, nth);
            fprintf(fo, "%s\t%s\t%s\t%zu\t%lld\t%.9e\t%.9e\t%.9e\t%.9e\t%.9e\n", name.c_str(), cands[h.first].first.c_str(),
                    ggml_type_name(ct->type), gguf_get_tensor_size(c.g, h.second), (long long) (K * M * E), e.sse, e.ref,
                    any_lam ? e.wsse : -1.0, any_lam ? e.wref : -1.0, has_h ? e.hsse : -1.0);
            fflush(fo);
            n_rows++;
            fprintf(stderr, "%-32s %-8s %-6s rel %.4e wrel %.4e\n", name.c_str(), cands[h.first].first.c_str(), ggml_type_name(ct->type),
                    e.sse / e.ref, any_lam ? e.wsse / e.wref : -1.0);
        }
    }
    fclose(fo);
    if (rename((f_out + ".tmp").c_str(), f_out.c_str()) != 0) { fprintf(stderr, "rename failed\n"); return 1; }
    fprintf(stderr, "wrote %s: %d rows, %d skipped\n", f_out.c_str(), n_rows, n_skip);
    return n_rows > 0 ? 0 : 1;
}
