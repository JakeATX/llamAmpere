// KVarN attention oracle (PRD M3): the CPU kvarn flash-attention path (ops.cpp) against plain
// ggml_flash_attn_ext on the same data. Positions [0,S) and [B,N) are exact fp16 rows (sink + ring),
// [S,B) are records sealed with ggml_kvarn::seal_group; the plain reference sees those records decoded
// back to fp16, so the two graphs must agree up to fp16/fp32 rounding.
//
//   test-kvarn-attn            runs the built-in cases
//   test-kvarn-attn n_q N n_groups cap [nh hkv]

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-kvarn.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

struct cfg { int n_q, N, n_groups, cap, nh, hkv; };

static int run_case(const cfg & c, uint32_t seed) {
    const int D = 256, S = 128, G = 128;
    const int nh = c.nh, hkv = c.hkv, n_q = c.n_q, N = c.N, cap = c.cap;
    const int B = S + c.n_groups*G;
    const int qpos0 = N - n_q;
    GGML_ASSERT(N >= B && N - B <= cap && qpos0 >= 0 && nh % hkv == 0 && cap % 128 == 0);
    const int n_kv_pad = (N + 255)/256*256;
    const int n_q_pad  = (n_q + 63)/64*64;
    const ggml_kvarn::layout l = ggml_kvarn::make_layout(D, G, 4, 4);

    std::mt19937 rng(seed);
    auto uni = [&](float lo, float hi) { return lo + (hi - lo) * (float) (rng() / 4294967296.0); };

    // per-position, per-KV-head rows: index (p*hkv + h)*D + d
    std::vector<ggml_fp16_t> Kd((size_t) N*hkv*D), Vd((size_t) N*hkv*D);
    for (size_t i = 0; i < Kd.size(); ++i) { Kd[i] = ggml_fp32_to_fp16(uni(-1.0f, 1.0f)); Vd[i] = ggml_fp32_to_fp16(uni(-1.0f, 1.0f)); }

    // seal [S,B) and replace those rows by their decoded values so both graphs see the same K/V
    std::vector<uint8_t> body((size_t) l.bytes*hkv*c.n_groups);
    std::vector<float> tmp(D);
    for (int g = 0; g < c.n_groups; ++g) {
        for (int h = 0; h < hkv; ++h) {
            uint8_t * rec = body.data() + ((size_t) g*hkv + h)*l.bytes;
            const size_t base = ((size_t) (S + g*G)*hkv + h)*D;
            ggml_kvarn::seal_group(Kd.data() + base, Vd.data() + base, (size_t) hkv*D, l, 16, rec);
            for (int t = 0; t < G; ++t) {
                const size_t row = ((size_t) (S + g*G + t)*hkv + h)*D;
                ggml_kvarn::decode_k_row(rec, l, t, tmp.data());
                for (int d = 0; d < D; ++d) Kd[row + d] = ggml_fp32_to_fp16(tmp[d]);
                ggml_kvarn::decode_v_row(rec, l, t, tmp.data());
                for (int d = 0; d < D; ++d) Vd[row + d] = ggml_fp32_to_fp16(tmp[d]);
            }
        }
    }

    const size_t mem = ggml_tensor_overhead()*32 + ggml_graph_overhead()
        + (size_t) D*n_q*nh*4 + 2*(size_t) D*n_kv_pad*hkv*2 + 2*(size_t) D*(S+cap)*hkv*2
        + (size_t) n_kv_pad*n_q_pad*2 + body.size() + 2*(size_t) D*nh*n_q*4 + 64*1024*1024;
    ggml_init_params ip = { mem, NULL, false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, n_q, nh, 1);
    for (int64_t i = 0; i < ggml_nelements(q); ++i) ((float *) q->data)[i] = uni(-1.0f, 1.0f);

    // plain reference: k/v [D, n_kv_pad, hkv]; rows >= N are zero (masked anyway)
    ggml_tensor * kp = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, n_kv_pad, hkv, 1);
    ggml_tensor * vp = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, n_kv_pad, hkv, 1);
    memset(kp->data, 0, ggml_nbytes(kp)); memset(vp->data, 0, ggml_nbytes(vp));
    for (int p = 0; p < N; ++p) for (int h = 0; h < hkv; ++h) {
        memcpy((char *) kp->data + p*kp->nb[1] + h*kp->nb[2], Kd.data() + ((size_t) p*hkv + h)*D, D*2);
        memcpy((char *) vp->data + p*vp->nb[1] + h*vp->nb[2], Vd.data() + ((size_t) p*hkv + h)*D, D*2);
    }

    // kvarn: k/v ring [D, S+cap, hkv]; unused rows get a sentinel so any misread shows up
    ggml_tensor * kr = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, S + cap, hkv, 1);
    ggml_tensor * vr = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, S + cap, hkv, 1);
    {
        const ggml_fp16_t sent = ggml_fp32_to_fp16(7.0f);
        for (int64_t i = 0; i < ggml_nelements(kr); ++i) { ((ggml_fp16_t *) kr->data)[i] = sent; ((ggml_fp16_t *) vr->data)[i] = sent; }
    }
    for (int p = 0; p < N; ++p) {
        if (p >= S && p < B) continue;
        const int row = p < S ? p : S + (p - S) % cap;
        for (int h = 0; h < hkv; ++h) {
            memcpy((char *) kr->data + row*kr->nb[1] + h*kr->nb[2], Kd.data() + ((size_t) p*hkv + h)*D, D*2);
            memcpy((char *) vr->data + row*vr->nb[1] + h*vr->nb[2], Vd.data() + ((size_t) p*hkv + h)*D, D*2);
        }
    }

    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv_pad, n_q_pad, 1, 1);
    for (int j = 0; j < n_q_pad; ++j) for (int p = 0; p < n_kv_pad; ++p) {
        const bool vis = j < n_q ? (p <= qpos0 + j && p < N) : (p < N);
        ((ggml_fp16_t *) m->data)[(size_t) j*n_kv_pad + p] = ggml_fp32_to_fp16(vis ? 0.0f : -INFINITY);
    }

    ggml_tensor * bt = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) std::max<size_t>(body.size(), 1));
    if (!body.empty()) memcpy(bt->data, body.data(), body.size());
    ggml_tensor * desc = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, GGML_KVARN_DESC_N_ENTRIES);
    {
        int32_t * d = (int32_t *) desc->data;
        memset(d, 0, GGML_KVARN_DESC_N_ENTRIES*4);
        d[GGML_KVARN_DESC_S] = S; d[GGML_KVARN_DESC_CAP] = cap; d[GGML_KVARN_DESC_B] = B; d[GGML_KVARN_DESC_N] = N;
        d[GGML_KVARN_DESC_QPOS0] = qpos0; d[GGML_KVARN_DESC_G] = G; d[GGML_KVARN_DESC_D] = D;
        d[GGML_KVARN_DESC_RECBYTES] = (int32_t) l.bytes; d[GGML_KVARN_DESC_HKV] = hkv; d[GGML_KVARN_DESC_B_OLD] = B;
    }

    const float scale = 1.0f/sqrtf((float) D);
    ggml_tensor * ref = ggml_flash_attn_ext(ctx, q, kp, vp, m, scale, 0.0f, 0.0f);
    ggml_prec_set_acc(ref, GGML_PREC_F32);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, kr, vr, m, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_kvarn(out, bt, desc, 4, 4, n_kv_pad);
    ggml_prec_set_acc(out, GGML_PREC_F32);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ref);
    ggml_build_forward_expand(gf, out);
    const int nth = std::max(4u, std::min(16u, std::thread::hardware_concurrency()));
    ggml_graph_compute_with_ctx(ctx, gf, nth);

    // compare: out/ref are [D, nh, n_q]
    const float * a = (const float *) out->data;
    const float * b = (const float *) ref->data;
    double se = 0.0, sb = 0.0, maxabs = 0.0; int64_t worst = -1;
    for (int64_t i = 0; i < ggml_nelements(ref); ++i) {
        const double d = (double) a[i] - (double) b[i];
        se += d*d; sb += (double) b[i]*b[i];
        if (fabs(d) > maxabs) { maxabs = fabs(d); worst = i; }
    }
    const double nmse = sb > 0 ? se/sb : se;
    const int64_t w_row = worst / D, w_head = w_row % nh, w_q = w_row / nh;
    const bool ok = nmse < 1e-4 && maxabs < 5e-2;
    printf("  n_q=%d N=%d B=%d cap=%d nh=%d hkv=%d qpos0=%d: nmse=%.3e maxabs=%.3e (q %ld head %ld pos %d) %s\n",
           n_q, N, B, cap, nh, hkv, qpos0, nmse, maxabs, (long) w_q, (long) w_head, qpos0 + (int) w_q, ok ? "OK" : "FAIL");
    ggml_free(ctx);
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    std::vector<cfg> cases;
    if (argc >= 5) {
        cfg c = { atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), argc > 5 ? atoi(argv[5]) : 24, argc > 6 ? atoi(argv[6]) : 4 };
        cases.push_back(c);
    } else {
        cases = {
            {    1, 2000,  0, 2176, 24, 4 },  // decode, exact only (B = S)
            {    8, 1153,  8, 2176, 24, 4 },  // decode width 8, one ring row past the body
            {   64, 3500, 10, 2176, 24, 4 },  // body + ring wrap
            { 1024, 3000,  0, 3072, 24, 4 },  // prefill ubatch, exact only
            {  256, 3000,  8, 2048, 24, 4 },  // prefill ubatch, body + ring wrap
            {    1,  500,  2,  256, 16, 2 },  // small: N = B + 116, cap 256
        };
    }
    int fails = 0;
    for (size_t i = 0; i < cases.size(); ++i) fails += run_case(cases[i], 1234 + (uint32_t) i);
    printf("%s: %d/%zu cases passed\n", fails ? "FAIL" : "OK", (int) (cases.size() - fails), cases.size());
    return fails ? 1 : 0;
}
