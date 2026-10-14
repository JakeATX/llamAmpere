// AI4 (Ampere INT4, formerly D2) CPU reference checks: rotation parity with RQ4 (when its header is given), tile bitstream round trip incl.
// tail-biting wrap, slab quantize/dequantize error on a real-weight dump with and without imatrix, encoder speed.
// Build: cc -O3 -march=native -fopenmp test-ai4.c -I../../ggml/src -lm [-DAI4_RQ4_HEADER='"path/ggml-rq4.h"']
// Run:   test-ai4 [weights.f32 K rows [imatrix.f32]]   (rows % 16 == 0, K % 1024 == 0)
// Also checks the AI4_U (unit-128) layout: n = 256 rotation parity, bit-identical decode vs AI4 at K % 256 == 0, H128 at K = 640.
#include <stdio.h>
#include <time.h>
#include "ggml-ai4.h"
#ifdef AI4_RQ4_HEADER
#include AI4_RQ4_HEADER
#endif

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9*t.tv_nsec; }

static float gauss(void) {
    const float u1 = (rand() + 1.0f)/(RAND_MAX + 2.0f), u2 = (rand() + 1.0f)/(RAND_MAX + 2.0f);
    return sqrtf(-2.0f*logf(u1))*cosf(6.2831853f*u2);
}

int main(int argc, char ** argv) {
    int fail = 0;
    srand(1);
    {
        float x[256], y[256];
        for (int i = 0; i < 256; ++i) x[i] = y[i] = gauss();
        ai4_rotate256(y);
#ifdef AI4_RQ4_HEADER
        float z[256]; memcpy(z, x, sizeof z); rq4_rotate256(z);
        int same = memcmp(z, y, sizeof z) == 0;
        printf("rotation bit-identical to RQ4: %s\n", same ? "yes" : "NO");
        if (!same) fail = 1;
#endif
        ai4_unrotate256(y);
        double e = 0; for (int i = 0; i < 256; ++i) e += (x[i]-y[i])*(x[i]-y[i]);
        printf("rotation round trip err %.3g\n", e); if (e > 1e-9) fail = 1;
    }
    ai4_work * ws = ai4_work_new();
    {
        int bad = 0, n_tiles = 32;
        double t0 = now();
        for (int tile = 0; tile < n_tiles; ++tile) {
            float y[AI4_TILE]; uint16_t st[AI4_TILE]; uint32_t wd[16] = {0};
            for (int i = 0; i < AI4_TILE; ++i) y[i] = 1.1f*ai4_level_rms()*gauss();
            ai4_encode_tile(ws, y, st);
            if ((st[0] >> 4) != (st[AI4_TILE-1] & 0xFFF)) bad++;
            for (int i = 0; i < AI4_TILE; ++i) wd[i >> 3] |= (uint32_t)(st[i] & 0xF) << (4*(7 - (i & 7)));
            for (int n = 0; n < AI4_TILE; ++n) if (ai4_get_state(wd, n) != st[n]) { bad++; break; }
        }
        printf("tile round trip: %d/%d bad; %.2f ms per tile (2 Viterbi passes)\n", bad, n_tiles, 1e3*(now() - t0)/n_tiles);
        if (bad) fail = 1;
    }
    {   // synthetic slab: quantize/dequantize must reproduce the encoder's own scales*levels (format consistency)
        const int64_t K = 1024;
        float * w = malloc(sizeof(float)*AI4_TR*K), * r = malloc(sizeof(float)*AI4_TR*K);
        uint8_t * q = malloc(K/AI4_SB*AI4_SLAB_SB_BYTES);
        for (int64_t i = 0; i < AI4_TR*K; ++i) w[i] = 0.01f*gauss();
        ai4_quantize_slab(ws, w, K, NULL, NULL, NULL, q);
        ai4_dequantize_slab(q, r, K);
        double se = 0, sw = 0; for (int64_t i = 0; i < AI4_TR*K; ++i) { se += (w[i]-r[i])*(w[i]-r[i]); sw += w[i]*w[i]; }
        printf("synthetic gaussian slab rel MSE %.4e (expect ~4.5e-3)\n", se/sw);
        if (!(se/sw < 6e-3)) fail = 1;
        free(w); free(r); free(q);
    }
    {   // AI+ rate k: k = 4 helpers equal AI4_U's (state extraction, slab bytes); k = 5 tile round trip, tail-biting and
        // a lower slab error than k = 4 (AI5_U)
        int bad = 0;
        uint32_t wd[20];
        for (int i = 0; i < 16; ++i) wd[i] = (uint32_t) rand() * 2654435761u;
        for (int n = 0; n < AI4_TILE; ++n) if (aik_get_state(wd, 4, n) != ai4_get_state(wd, n)) bad++;
        printf("aik k=4 state == ai4 state: %s\n", bad ? "NO" : "yes");
        if (bad) fail = 1;
        bad = 0;
        for (int tile = 0; tile < 8; ++tile) {
            float y[AI4_TILE]; uint16_t st[AI4_TILE]; uint32_t w5[20] = {0};
            for (int i = 0; i < AI4_TILE; ++i) y[i] = 1.1f*ai4_level_rms()*gauss();
            aik_encode_tile(ws, 5, y, st);
            if ((st[0] >> 5) != (st[AI4_TILE-1] & 0x7FF)) bad++;
            for (int i = 0; i < AI4_TILE; ++i) aik_put_sym(w5, 5, i, st[i] & 31);
            for (int n = 0; n < AI4_TILE; ++n) if (aik_get_state(w5, 5, n) != st[n]) { bad++; break; }
        }
        printf("aik k=5 tile round trip + tail-biting: %d/8 bad\n", bad);
        if (bad) fail = 1;
        const int64_t K = 512;
        float * w = malloc(sizeof(float)*AI4_TR*K), * r = malloc(sizeof(float)*AI4_TR*K), * lam = malloc(sizeof(float)*K);
        uint8_t * q4 = malloc(K/128*AI4U_UNIT_BYTES(5)), * q4b = malloc(K/128*AI4U_UNIT_BYTES(5)), * q5 = malloc(K/128*AI4U_UNIT_BYTES(5));
        for (int64_t i = 0; i < AI4_TR*K; ++i) w[i] = 0.01f*gauss();
        for (int64_t i = 0; i < K; ++i) lam[i] = 0.5f + (float) rand()/RAND_MAX;
        ai4u_quantize_slab(ws, w, K, lam, NULL, NULL, q4);
        aik_quantize_slab_rb(ws, 4, w, K, lam, NULL, NULL, q4b, ai4u_rot_size(K), AI4_SCALE_MUL);
        const int same4 = memcmp(q4, q4b, K/128*AI4U_UNIT_BYTES4) == 0;
        aik_quantize_slab_rb(ws, 5, w, K, lam, NULL, NULL, q5, ai4u_rot_size(K), AI4_SCALE_MUL);
        double e4 = 0, e5 = 0, sw = 0;
        aik_dequantize_slab(4, q4, r, K);
        for (int64_t i = 0; i < AI4_TR*K; ++i) { e4 += (w[i]-r[i])*(w[i]-r[i]); sw += w[i]*w[i]; }
        aik_dequantize_slab(5, q5, r, K);
        for (int64_t i = 0; i < AI4_TR*K; ++i) e5 += (w[i]-r[i])*(w[i]-r[i]);
        printf("aik k=4 slab bytes == ai4u: %s; rel MSE k=4 %.4e k=5 %.4e (ratio %.3f)\n", same4 ? "yes" : "NO", e4/sw, e5/sw, e5/e4);
        if (!same4 || !(e5 < 0.5*e4)) fail = 1;
        free(w); free(r); free(lam); free(q4); free(q4b); free(q5);
    }
    {   // AI4_U (unit-128 layout): n = 256 helpers equal the 256 ones; at K % 256 == 0 the decoded values equal AI4's
        // bit for bit (same rotation, same encode, reordered bytes); K = 640 (H128) round trip error and imatrix LDLQ.
        float x[256], y[256], z[256];
        for (int i = 0; i < 256; ++i) x[i] = y[i] = z[i] = gauss();
        ai4_rotate256(y); ai4_rotate_n(z, 256);
        int same = memcmp(y, z, sizeof y) == 0;
        ai4_unrotate256(y); ai4_unrotate_n(z, 256);
        same &= memcmp(y, z, sizeof y) == 0;
        float h[128]; memcpy(h, x, sizeof h); ai4_rotate_n(h, 128); ai4_unrotate_n(h, 128);
        double eh = 0; for (int i = 0; i < 128; ++i) eh += (x[i]-h[i])*(x[i]-h[i]);
        printf("ai4u: rotate_n(256) == rotate256: %s; H128 round trip err %.3g\n", same ? "yes" : "NO", eh);
        if (!same || eh > 1e-9) fail = 1;

        const int64_t K = 1024;
        float * w = malloc(sizeof(float)*AI4_TR*K), * r1 = malloc(sizeof(float)*AI4_TR*K), * r2 = malloc(sizeof(float)*AI4_TR*K);
        float * lam = malloc(sizeof(float)*K);
        uint8_t * q1 = malloc(ai4x_slab_bytes(K)), * q2 = malloc(ai4x_slab_bytes(K));
        for (int64_t i = 0; i < AI4_TR*K; ++i) w[i] = 0.01f*gauss();
        for (int64_t k = 0; k < K; ++k) lam[k] = 0.2f + gauss()*gauss();
        ai4_quantize_slab(ws, w, K, lam, NULL, NULL, q1);
        ai4u_quantize_slab(ws, w, K, lam, NULL, NULL, q2);
        ai4_dequantize_slab(q1, r1, K);
        ai4u_dequantize_slab(q2, r2, K);
        const int eq = memcmp(r1, r2, sizeof(float)*AI4_TR*K) == 0;
        printf("ai4u K=1024 (H256) decode bit-identical to ai4: %s\n", eq ? "yes" : "NO");
        if (!eq) fail = 1;
        free(w); free(r1); free(r2); free(lam); free(q1); free(q2);

        const int64_t K2 = 640;
        w = malloc(sizeof(float)*AI4_TR*K2); r1 = malloc(sizeof(float)*AI4_TR*K2); r2 = malloc(sizeof(float)*AI4_TR*K2);
        q1 = malloc(ai4x_slab_bytes(K2));
        for (int64_t i = 0; i < AI4_TR*K2; ++i) w[i] = 0.01f*gauss();
        ai4u_quantize_slab(ws, w, K2, NULL, NULL, NULL, q1);
        ai4u_dequantize_slab(q1, r1, K2);
        double se = 0, sw = 0; for (int64_t i = 0; i < AI4_TR*K2; ++i) { se += (w[i]-r1[i])*(w[i]-r1[i]); sw += w[i]*w[i]; }
        printf("ai4u K=640 (H128) synthetic slab rel MSE %.4e (expect ~4.5e-3), %lld bytes\n", se/sw, (long long) ai4x_slab_bytes(K2));
        if (!(se/sw < 6e-3) || ai4x_slab_bytes(K2) != 5*1056) fail = 1;
        free(w); free(r1); free(r2); free(q1);
    }
    {   // AIBEST: the AI+ rates k = 2, 3, 6 and the per-type trellis constant (private controls AI6_UO / AI5_UN).
        // Tile bitstream round trip + tail-biting at every rate, then slab error ordering k2 > k3 > k4 > k5 > k6.
        static const int    ks[5] = { 2, 3, 4, 5, 6 };
        static const unsigned cs[5] = { 0x83DCD12Du, 0x83DCD12Du, 0x83DCD12Du, 0x83DCD12Du, 0xA1EEE321u };
        static const float  sms[5] = { 1.00f, 1.07f, 1.10f, 1.10f, 1.15f };
        int bad = 0;
        for (int ki = 0; ki < 5; ++ki) {
            const int k = ks[ki];
            for (int tile = 0; tile < 4; ++tile) {
                float y[AI4_TILE]; uint16_t st[AI4_TILE]; uint32_t wk[AIK_TILE_WORDS(6)];
                memset(wk, 0, sizeof wk);
                aik_work_set_mul(ws, cs[ki]);
                for (int i = 0; i < AI4_TILE; ++i) y[i] = 1.1f*aik_level_rms_c(cs[ki])*gauss();
                aik_encode_tile(ws, k, y, st);
                if ((st[0] >> k) != (st[AI4_TILE-1] & ((1u << (16 - k)) - 1u))) bad++;
                for (int i = 0; i < AI4_TILE; ++i) aik_put_sym(wk, k, i, st[i] & ((1u << k) - 1u));
                for (int n = 0; n < AI4_TILE; ++n) if (aik_get_state(wk, k, n) != st[n]) { bad++; break; }
            }
        }
        printf("aik k=2,3,4,5,6 tile round trip + tail-biting: %d/20 bad\n", bad);
        if (bad) fail = 1;

        const int64_t K = 256;
        float * w = malloc(sizeof(float)*AI4_TR*K), * r = malloc(sizeof(float)*AI4_TR*K);
        uint8_t * q = malloc((size_t) ai4x_slab_bytes_k(K, 6));
        double sw = 0, e[5];
        for (int64_t i = 0; i < AI4_TR*K; ++i) { w[i] = 0.01f*gauss(); sw += w[i]*w[i]; }
        for (int ki = 0; ki < 5; ++ki) {
            aik_quantize_slab_rb_c(ws, ks[ki], cs[ki], w, K, NULL, NULL, NULL, q, ai4u_rot_size(K), sms[ki]);
            aik_dequantize_slab_c(ks[ki], cs[ki], q, r, K);
            e[ki] = 0;
            for (int64_t i = 0; i < AI4_TR*K; ++i) e[ki] += (w[i]-r[i])*(w[i]-r[i]);
            e[ki] /= sw;
            printf("aik k=%d C=%08X smul %.2f rel MSE %.4e (%.3f bpw)\n", ks[ki], cs[ki], sms[ki], e[ki],
                   (32.0 + 256.0*ks[ki])*8.0/2048.0);
        }
        int ord = 1;
        for (int ki = 0; ki + 1 < 5; ++ki) ord = ord && e[ki] > e[ki+1];
        printf("aik rel MSE ordering k2 > k3 > k4 > k5 > k6: %s\n", ord ? "yes" : "NO");
        if (!ord) fail = 1;
        // the private control constants are a different decoder: the level tables must differ
        aik_work_set_mul(ws, 0x9C971CDDu);
        int ndiff = 0;
        for (unsigned st = 0; st < 4096; ++st) ndiff += aik_level_c(st, 0x9C971CDDu) != aik_level_c(st, AI4_MUL);
        printf("AI5_UN level table differs from AI4_MUL on %d/4096 states\n", ndiff);
        if (ndiff < 1024) fail = 1;
        aik_work_set_mul(ws, AI4_MUL);
        free(w); free(r); free(q);
    }
    if (argc >= 4) {
        const int64_t K = atoll(argv[2]); const int R = atoi(argv[3]);
        float * W = malloc(sizeof(float)*K*R), * lam = NULL; FILE * f = fopen(argv[1], "rb");
        if (!f || fread(W, sizeof(float), K*R, f) != (size_t)(K*R)) { printf("cannot read %s\n", argv[1]); return 1; }
        fclose(f);
        if (argc >= 5) {
            lam = malloc(sizeof(float)*K); f = fopen(argv[4], "rb");
            if (!f || fread(lam, sizeof(float), K, f) != (size_t)K) { printf("cannot read %s\n", argv[4]); return 1; }
            fclose(f);
        }
        const int NB = K/AI4_HB;
        float * Uall = lam ? malloc(sizeof(float)*AI4_HB*AI4_HB*(size_t)NB) : NULL; uint8_t * ok = lam ? malloc(NB) : NULL;
        for (int b = 0; lam && b < NB; ++b) ok[b] = (uint8_t) ai4_ldlq_prep(lam + b*AI4_HB, Uall + (size_t)b*AI4_HB*AI4_HB);
        for (int use_im = 0; use_im <= (lam ? 1 : 0); ++use_im) {
            double su = 0, sw = 0, nu = 0, nw = 0, t0 = now();
            #pragma omp parallel reduction(+:su,sw,nu,nw)
            {
                ai4_work * w2 = ai4_work_new();
                uint8_t * q = malloc(K/AI4_SB*AI4_SLAB_SB_BYTES); float * rr = malloc(sizeof(float)*AI4_TR*K);
                #pragma omp for schedule(dynamic,1)
                for (int s = 0; s < R/AI4_TR; ++s) {
                    const float * ws0 = W + (int64_t)s*AI4_TR*K;
                    ai4_quantize_slab(w2, ws0, K, use_im ? lam : NULL, use_im ? Uall : NULL, ok, q);
                    ai4_dequantize_slab(q, rr, K);
                    for (int64_t i = 0; i < AI4_TR*K; ++i) {
                        const double e = ws0[i] - rr[i], l = lam ? lam[i % K] : 1.0;
                        su += e*e; nu += (double) ws0[i]*ws0[i]; sw += l*e*e; nw += l*(double) ws0[i]*ws0[i];
                    }
                }
                free(w2); free(q); free(rr);
            }
            printf("real rows (%s): rel MSE %.4e, imatrix-weighted %.4e, %.1f s wall\n", use_im ? "imatrix LDLQ" : "no imatrix",
                   su/nu, sw/nw, now() - t0);
        }
    }
    printf(fail ? "FAIL\n" : "OK\n");
    return fail;
}
