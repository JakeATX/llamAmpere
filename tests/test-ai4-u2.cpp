// AITk_U2 reference decode vs AITk_U (same tiles, same scale in every unit of a block, same level rule): must match exactly.
#include "../ggml/src/ggml-ai4.h"
#include "ggml-ai4-decode.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

int main() {
    int bad = 0;
    for (int k = 2; k <= 4; ++k) {
        const ggml_type type = k == 2 ? GGML_TYPE_AIT2_U2 : k == 3 ? GGML_TYPE_AIT3_U2 : GGML_TYPE_AIT4_U2;
        const uint32_t C = k == 2 ? 0xC3CCD12Du : k == 3 ? 0x87D8D12Du : 0xAA73876Bu;
        const bool registered = ggml_ai4_rate(type) == k && ggml_ai4v2_rate(type) == k && ggml_ai4_trellis_mul(type) == C &&
            ggml_ai4_is_unit(type) && ggml_ai4_layout(type) == 1 && ggml_ai4_code(type) == 0 && ggml_ai4_scale_mul(type) > 0.0f;
        printf("%s rate/decoder registration %s\n", ggml_type_name(type), registered ? "OK" : "FAIL");
        bad += !registered;
        for (int64_t K : { 640, 2560 }) {
            const int64_t nb = K/AI4V2_BK;
            std::vector<uint8_t> uu(ai4x_slab_bytes_k(K, k)), v2(nb*AI4V2_BLOCK_BYTES(k));
            srand(k*1000 + (int) K);
            for (auto & b : uu) b = rand();
            uint16_t sc[AI4_TR];
            for (int r = 0; r < AI4_TR; ++r) sc[r] = 0x2800 + (rand() & 0x3ff);   // positive fp16 around 0.03..0.06
            const int ub = AI4U_UNIT_BYTES(k), tb = AI4V2_UNITS*(AI4U_UK/AI4_TC)*AI4U_TILE_BYTES(k);
            for (int64_t u = 0; u < K/AI4U_UK; ++u) memcpy(uu.data() + u*ub, sc, sizeof sc);
            for (int64_t b = 0; b < nb; ++b) {
                memcpy(v2.data() + b*AI4V2_BLOCK_BYTES(k), sc, sizeof sc);
                for (int u = 0; u < AI4V2_UNITS; ++u)   // tiles of unit (5b + u) of the old slab
                    memcpy(v2.data() + b*AI4V2_BLOCK_BYTES(k) + AI4U_SCALE_BYTES + (size_t) u*(tb/AI4V2_UNITS),
                           uu.data() + (5*b + u)*ub + AI4U_SCALE_BYTES, tb/AI4V2_UNITS);
            }
            std::vector<float> a(16*K), b(16*K);
            aik_dequantize_slab_c(k, C, uu.data(), a.data(), K);
            aik_dequantize_slab_v2_c(k, C, v2.data(), b.data(), K);
            std::vector<float> decoded(16*K);
            const bool public_decode = ggml_ai4_slab_nbytes(type, K) == v2.size() &&
                ggml_ai4_dequantize_slab(type, v2.data(), K, 0, decoded.data()) &&
                memcmp(decoded.data(), b.data(), b.size()*sizeof(float)) == 0;
            printf("%s K=%lld public slab decode %s\n", ggml_type_name(type), (long long) K, public_decode ? "OK" : "FAIL");
            bad += !public_decode;
            double ee = 0, ss = 0;
            for (size_t i = 0; i < a.size(); ++i) { ee += (a[i]-b[i])*(double)(a[i]-b[i]); ss += (double) a[i]*a[i]; }
            const double rel = sqrt(ee/ss);
            const int ok = rel == 0.0 && ss > 0;   // same tiles, same level rule: bit-identical decode
            printf("k=%d K=%lld bytes/row-block %d  rel rms diff old-vs-v2 = %.5f %s\n", k, (long long) K, (int) (AI4V2_BLOCK_BYTES(k)/16), rel, ok ? "OK" : "FAIL");
            bad += !ok;
        }
    }
    return bad;
}
