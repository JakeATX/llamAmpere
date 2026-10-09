// SJ-KVaRN codec gate (PRD M0): the shared scalar codec and the ggml ops (TURBO_WHT 256, SJKVARN_SEAL)
// against the independent numpy oracle's record file.
//
//   test-sjkvarn-codec <tiles.bin> <records_oracle.bin> <n> [G D bits_k bits_v iters]
//
// tiles.bin: n groups of {K float32 [G][D], V float32 [G][D]} in the unrotated domain.
// records:   n records of ggml_sj_kvarn_rec_bytes(D,G,bits_k,bits_v) bytes.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-sjkvarn.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<uint8_t> read_file(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(n);
    if (fread(buf.data(), 1, n, f) != (size_t) n) { fprintf(stderr, "short read %s\n", path); exit(2); }
    fclose(f);
    return buf;
}

// rotate every token row with H_D (float32 butterfly) and round to fp16, as the cache does
static void rotate_rows(const float * src, int G, int D, std::vector<ggml_fp16_t> & out) {
    std::vector<float> row(D);
    out.resize((size_t) G*D);
    for (int t = 0; t < G; ++t) {
        memcpy(row.data(), src + (size_t) t*D, D*sizeof(float));
        ggml_sj_kvarn::hadamard(row.data(), D);
        for (int d = 0; d < D; ++d) {
            out[(size_t) t*D + d] = ggml_fp32_to_fp16(row[d]);
        }
    }
}

struct diff_stats {
    size_t rec_mismatch = 0;
    size_t meta_bytes_diff = 0;
    size_t payload_vals = 0, payload_diff = 0, payload_diff_gt1 = 0;
    size_t noise_rows = 0, noise_diff = 0; // rows whose fp16 step is subnormal: codes carry no information
};

// A K channel / V token row whose quantization step (the stored fp16 scale) is below the fp16 normal range is a
// (near-)constant row: its 16 codes span less than 1e-3 of the row value, so float32 and float64 balancing land on
// arbitrary codes. Such rows are counted separately and excluded from the "differs by more than 1" gate.
static bool noise_row(const uint8_t * rec, size_t scale_off) {
    ggml_fp16_t h;
    memcpy(&h, rec + scale_off, 2);
    return fabsf(GGML_FP16_TO_FP32(h)) < 6.103515625e-05f; // 2^-14
}

// The oracle writes the PRD section-3 record: token-major bit stream per payload and channel metadata in natural
// order. The sealer's record is fragment-ordered (ggml_sj_kvarn::code_bit / k_ch_idx / v_ch_idx); re-pack the oracle
// record into that layout so the two can be compared code by code.
static std::vector<uint8_t> oracle_to_layout(const uint8_t * o, const ggml_sj_kvarn::layout & l) {
    std::vector<uint8_t> r(o, o + l.bytes);
    const bool fk = ggml_sj_kvarn::frag_order(l.bits_k, l.D, l.G);
    const bool fv = ggml_sj_kvarn::frag_order(l.bits_v, l.D, l.G);
    if (!fk && !fv) {
        return r;
    }
    // oracle slot of value d in a row: 4-bit rows interleave each 8-value word (value j at nibble j/2 for even j,
    // 4 + j/2 for odd j); other widths are the identity
    auto slot = [](int d, int bits) -> uint32_t {
        if (bits != 4) return (uint32_t) d;
        const int j = d & 7;
        return (uint32_t) ((d & ~7) + ((j & 1) ? 4 + (j >> 1) : (j >> 1)));
    };
    memset(r.data(), 0, l.k_scale);
    for (int t = 0; t < l.G; ++t) {
        for (int d = 0; d < l.D; ++d) {
            const uint32_t k = ggml_sj_kvarn::unpack_code(o + l.k_payload, (uint32_t) t*(l.D*l.bits_k) + slot(d, l.bits_k)*l.bits_k, l.bits_k);
            const uint32_t v = ggml_sj_kvarn::unpack_code(o + l.v_payload, (uint32_t) t*(l.D*l.bits_v) + slot(d, l.bits_v)*l.bits_v, l.bits_v);
            ggml_sj_kvarn::pack_code(r.data() + l.k_payload, ggml_sj_kvarn::code_bit(t, d, false, l.D, l.G, l.bits_k), l.bits_k, k);
            ggml_sj_kvarn::pack_code(r.data() + l.v_payload, ggml_sj_kvarn::code_bit(t, d, true,  l.D, l.G, l.bits_v), l.bits_v, v);
        }
    }
    for (int d = 0; d < l.D; ++d) {
        memcpy(r.data() + l.k_scale + 2*ggml_sj_kvarn::k_ch_idx(d, l.D, l.G, l.bits_k), o + l.k_scale + 2*d, 2);
        memcpy(r.data() + l.k_zero  + 2*ggml_sj_kvarn::k_ch_idx(d, l.D, l.G, l.bits_k), o + l.k_zero  + 2*d, 2);
        memcpy(r.data() + l.v_ch    + 2*ggml_sj_kvarn::v_ch_idx(d, l.D, l.G, l.bits_v), o + l.v_ch    + 2*d, 2);
    }
    return r;
}

static void compare_records(const uint8_t * a, const uint8_t * b, const ggml_sj_kvarn::layout & l, diff_stats & st) {
    bool any = false;
    for (size_t i = l.k_scale; i < l.bytes; ++i) {
        if (a[i] != b[i]) { st.meta_bytes_diff++; any = true; }
    }
    std::vector<bool> k_noise(l.D), v_noise(l.G);
    for (int d = 0; d < l.D; ++d) { k_noise[d] = noise_row(a, l.k_scale + 2*ggml_sj_kvarn::k_ch_idx(d, l.D, l.G, l.bits_k)); st.noise_rows += k_noise[d]; }
    for (int t = 0; t < l.G; ++t) { v_noise[t] = noise_row(a, l.v_scale + 2*t); st.noise_rows += v_noise[t]; }
    for (int t = 0; t < l.G; ++t) {
        for (int d = 0; d < l.D; ++d) {
            const int qa = ggml_sj_kvarn::k_code(a, l, t, d);
            const int qb = ggml_sj_kvarn::k_code(b, l, t, d);
            st.payload_vals++;
            if (qa != qb) {
                any = true;
                if (k_noise[d]) { st.noise_diff++; } else { st.payload_diff++; if (abs(qa - qb) > 1) st.payload_diff_gt1++; }
            }
            const int va = ggml_sj_kvarn::v_code(a, l, t, d);
            const int vb = ggml_sj_kvarn::v_code(b, l, t, d);
            st.payload_vals++;
            if (va != vb) {
                any = true;
                if (v_noise[t]) { st.noise_diff++; } else { st.payload_diff++; if (abs(va - vb) > 1) st.payload_diff_gt1++; }
            }
        }
    }
    if (any) st.rec_mismatch++;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s tiles.bin records.bin n [G D bits_k bits_v iters]\n", argv[0]);
        return 2;
    }
    const int n      = atoi(argv[3]);
    const int G      = argc > 4 ? atoi(argv[4]) : 128;
    const int D      = argc > 5 ? atoi(argv[5]) : 256;
    const int bits_k = argc > 6 ? atoi(argv[6]) : 4;
    const int bits_v = argc > 7 ? atoi(argv[7]) : 4;
    const int iters  = argc > 8 ? atoi(argv[8]) : 16;

    const ggml_sj_kvarn::layout l = ggml_sj_kvarn::make_layout(D, G, bits_k, bits_v);
    if (l.bytes != ggml_sj_kvarn_rec_bytes(D, G, bits_k, bits_v)) {
        fprintf(stderr, "layout/rec_bytes disagree: %zu vs %zu\n", l.bytes, ggml_sj_kvarn_rec_bytes(D, G, bits_k, bits_v));
        return 1;
    }

    const std::vector<uint8_t> tiles = read_file(argv[1]);
    std::vector<uint8_t> recs        = read_file(argv[2]);
    const size_t tile_bytes = 2 * (size_t) G * D * sizeof(float);
    if (tiles.size() < (size_t) n * tile_bytes || recs.size() < (size_t) n * l.bytes) {
        fprintf(stderr, "files too small for n=%d (tiles %zu need %zu, records %zu need %zu)\n",
                n, tiles.size(), (size_t) n*tile_bytes, recs.size(), (size_t) n*l.bytes);
        return 2;
    }

    for (int g = 0; g < n; ++g) {
        const std::vector<uint8_t> r = oracle_to_layout(recs.data() + (size_t) g*l.bytes, l);
        memcpy(recs.data() + (size_t) g*l.bytes, r.data(), l.bytes);
    }

    int rc = 0;

    // ---- 1. scalar codec vs oracle, all n groups
    {
        diff_stats st;
        int n_skipped = 0;
        std::vector<ggml_fp16_t> kr, vr;
        std::vector<uint8_t> rec(l.bytes);
        for (int g = 0; g < n; ++g) {
            const float * K = (const float *) (tiles.data() + (size_t) g*tile_bytes);
            const float * V = K + (size_t) G*D;
            {
                // a sealer only ever sees finite fp16 rows; skip synthetic tiles carrying inf/nan
                bool finite = true;
                for (size_t i = 0; i < (size_t) 2*G*D && finite; ++i) finite = std::isfinite(K[i]);
                if (!finite) { printf("  group %d: non-finite input, skipped\n", g); ++n_skipped; continue; }
            }
            rotate_rows(K, G, D, kr);
            rotate_rows(V, G, D, vr);
            ggml_sj_kvarn::seal_group(kr.data(), vr.data(), D, l, iters, rec.data());
            diff_stats one;
            compare_records(rec.data(), recs.data() + (size_t) g*l.bytes, l, one);
            if (one.payload_diff_gt1 > 0 || one.meta_bytes_diff > 4) {
                printf("  group %d: %zu metadata bytes differ, %zu payload values differ, %zu by more than 1\n",
                       g, one.meta_bytes_diff, one.payload_diff, one.payload_diff_gt1);
            }
            st.rec_mismatch += one.rec_mismatch; st.meta_bytes_diff += one.meta_bytes_diff;
            st.payload_vals += one.payload_vals; st.payload_diff += one.payload_diff; st.payload_diff_gt1 += one.payload_diff_gt1;
            st.noise_rows += one.noise_rows; st.noise_diff += one.noise_diff;
        }
        const double frac = st.payload_vals ? (double) st.payload_diff / st.payload_vals : 0.0;
        printf("codec vs oracle: %d groups, %zu mismatching records, %zu metadata bytes differ, "
               "%zu/%zu payload values differ (%.2e), %zu differ by more than 1\n",
               n, st.rec_mismatch, st.meta_bytes_diff, st.payload_diff, st.payload_vals, frac, st.payload_diff_gt1);
        // Gate: the C codec (float32 libm) and the numpy oracle (float64 std chain) may land on different sides
        // of an fp16 rounding boundary for a handful of metadata halves (observed 17 of ~87K over 195 tiles) and
        // flip a code by one at an RTN tie; anything beyond that is a real codec divergence.
        const size_t meta_bytes = (size_t) (n - n_skipped) * (l.bytes - l.k_scale);
        const double meta_frac  = meta_bytes ? (double) st.meta_bytes_diff / meta_bytes : 0.0;
        printf("  metadata bytes differing: %.2e of %zu, groups skipped: %d, subnormal-step rows: %zu (%zu code diffs ignored)\n",
               meta_frac, meta_bytes, n_skipped, st.noise_rows, st.noise_diff);
        const bool ok = meta_frac < 1e-3 && st.payload_diff_gt1 == 0 && frac < 1e-4;
        printf("  %s\n", ok ? "OK" : "FAIL");
        if (!ok) rc = 1;
    }

    // ---- 2. ggml ops on the CPU backend: TURBO_WHT 256 (rotation) then SJKVARN_SEAL, first n_op groups
    {
        const int n_op = n < 8 ? n : 8;
        const int hkv  = 2;                      // fold pairs of groups into two heads
        const int n_groups = (n_op + hkv - 1) / hkv;
        const int n_tok = n_groups * G;

        ggml_init_params ip = { ggml_tensor_overhead()*16 + ggml_graph_overhead() + 4*(size_t) n_tok*hkv*D*sizeof(float) + (size_t) n_groups*hkv*l.bytes + 1024*1024, NULL, false };
        ggml_context * ctx = ggml_init(ip);

        ggml_tensor * kf = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) hkv*D, n_tok);
        ggml_tensor * vf = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) hkv*D, n_tok);
        float * kfp = (float *) kf->data;
        float * vfp = (float *) vf->data;
        memset(kfp, 0, ggml_nbytes(kf));
        memset(vfp, 0, ggml_nbytes(vf));
        for (int r = 0; r < n_op; ++r) {
            const int g = r / hkv, h = r % hkv;
            const float * K = (const float *) (tiles.data() + (size_t) r*tile_bytes);
            const float * V = K + (size_t) G*D;
            for (int t = 0; t < G; ++t) {
                memcpy(kfp + ((size_t) (g*G + t)*hkv + h)*D, K + (size_t) t*D, D*sizeof(float));
                memcpy(vfp + ((size_t) (g*G + t)*hkv + h)*D, V + (size_t) t*D, D*sizeof(float));
            }
        }

        ggml_tensor * kw = ggml_turbo_wht(ctx, kf, 0, 256, NULL);
        ggml_tensor * vw = ggml_turbo_wht(ctx, vf, 0, 256, NULL);
        ggml_tensor * kh = ggml_cast(ctx, kw, GGML_TYPE_F16);
        ggml_tensor * vh = ggml_cast(ctx, vw, GGML_TYPE_F16);
        ggml_tensor * body = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) l.bytes*hkv*n_groups);
        memset(body->data, 0xAB, ggml_nbytes(body));
        ggml_tensor * seal = ggml_sj_kvarn_seal(ctx, body, kh, vh, D, G, bits_k, bits_v, iters);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, seal);
        ggml_graph_compute_with_ctx(ctx, gf, 8);

        // rotation check: op output vs scalar hadamard, tolerance a few ulp
        double max_rot_err = 0.0;
        {
            std::vector<float> row(D);
            const float * kwp = (const float *) kw->data;
            for (int t = 0; t < n_tok; ++t) {
                for (int h = 0; h < hkv; ++h) {
                    memcpy(row.data(), kfp + ((size_t) t*hkv + h)*D, D*sizeof(float));
                    ggml_sj_kvarn::hadamard(row.data(), D);
                    for (int d = 0; d < D; ++d) {
                        const double e = fabs((double) row[d] - kwp[((size_t) t*hkv + h)*D + d]);
                        if (e > max_rot_err) max_rot_err = e;
                    }
                }
            }
        }
        diff_stats st;
        for (int r = 0; r < n_op; ++r) {
            compare_records((const uint8_t *) body->data + (size_t) r*l.bytes, recs.data() + (size_t) r*l.bytes, l, st);
        }
        const double frac = st.payload_vals ? (double) st.payload_diff / st.payload_vals : 0.0;
        printf("ggml ops (WHT256 + SJKVARN_SEAL, CPU): %d records, max rotation abs err %.3e, %zu mismatching records, "
               "%zu metadata bytes differ, %zu/%zu payload values differ (%.2e), %zu by more than 1\n",
               n_op, max_rot_err, st.rec_mismatch, st.meta_bytes_diff, st.payload_diff, st.payload_vals, frac, st.payload_diff_gt1);
        // the op rotates in float32 with the same butterfly, then rounds to fp16: a rare 1-ulp fp16 flip is possible
        const bool ok = max_rot_err < 1e-5 && st.payload_diff_gt1 == 0 && frac < 1e-3 && st.meta_bytes_diff < 16;
        printf("  %s\n", ok ? "OK" : "FAIL");
        if (!ok) rc = 1;
        ggml_free(ctx);
    }

    return rc;
}
