// Check physical weighted error independently from the trellis cost, including ring closure and packed decode.
#include "ggml-ai4.h"
#include <stdio.h>

static double score(const ai4_work * ws, const float * y, const float * w, const uint16_t * states) {
    double sum = 0.0;
    for (int i = 0; i < AI4_TILE; ++i) {
        double e = ws->lev[states[i]] - y[i];
        sum += e*e*w[i];
    }
    return sum;
}

int main(void) {
    ai4_work * ws = ai4_work_new();
    float y[128], weight[128], unit[128];
    uint16_t legacy[128], same[128], fitted[128];
    int fail = 0;
    for (int i = 0; i < 128; ++i) {
        y[i] = 30.0f*sinf(i*1.713f) + 5.0f*cosf(i*0.17f);
        const float d = (i/8 == 0) ? 1.0f : 0.05f + (i/8)*0.02f;
        weight[i] = d*d;
        unit[i] = 1.0f;
    }
    for (int k = 2; k <= 6; ++k) {
        aik_encode_tile(ws, k, y, legacy);
        aik_encode_tile_metric(ws, k, y, unit, same);
        if (memcmp(legacy, same, sizeof legacy)) ++fail;
        // Hold the tail constraint fixed: the changed objective must improve physical error in that same search space.
        int tail = legacy[127] & ((1 << (16 - k)) - 1);
        aik_viterbi_metric(ws, k, y, weight, tail, fitted);
        double before = score(ws, y, weight, legacy), after = score(ws, y, weight, fitted);
        if (after > before*(1.0 + 1e-6)) ++fail;
        if ((fitted[0] >> k) != (fitted[127] & ((1 << (16 - k)) - 1))) ++fail;
        uint32_t packed[24] = {0};
        for (int i = 0; i < 128; ++i) aik_put_sym(packed, k, i, fitted[i] & ((1 << k) - 1));
        for (int i = 0; i < 128; ++i) if (aik_get_state(packed, k, i) != fitted[i]) ++fail;
        printf("k%d constant-weight byte parity; fixed-tail physical error %.9g -> %.9g (%.6fx)\n",
               k, before, after, after/before);
    }
    free(ws);
    printf("%s (%d failures)\n", fail ? "FAIL" : "PASS", fail);
    return fail ? 1 : 0;
}
