#include "../ggml/src/ggml-cuda/fattn-mma-config.h"

#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <utility>

static void check(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "test-fattn-mma-config: %s\n", what);
        std::exit(1);
    }
}

static bool equal(const fattn_mma_config & a, const fattn_mma_config & b) {
    return a.nthreads == b.nthreads && a.occupancy == b.occupancy && a.nbatch_fa == b.nbatch_fa &&
           a.nbatch_K2 == b.nbatch_K2 && a.nbatch_V2 == b.nbatch_V2 && a.nbatch_combine == b.nbatch_combine &&
           a.nstages_target == b.nstages_target && a.Q_in_reg == b.Q_in_reg;
}

int main() {
    const ggml_type upstream[] = { GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_F32, GGML_TYPE_Q4_0,
                                  GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL };
    const ggml_type fork[] = { GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0, GGML_TYPE_TQ5_0, GGML_TYPE_TQ6_0 };
    for (ggml_type k : upstream) {
        for (ggml_type v : upstream) {
            check(ggml_cuda_fattn_mma_cand_pair(k, v) == (k == GGML_TYPE_Q8_0 && v == GGML_TYPE_Q8_0), "upstream pair policy");
        }
        for (ggml_type v : fork) {
            check(ggml_cuda_fattn_mma_cand_pair(k, v) && ggml_cuda_fattn_mma_cand_pair(v, k), "mixed pair policy");
        }
    }
    for (ggml_type k : fork) {
        for (ggml_type v : fork) {
            check(ggml_cuda_fattn_mma_cand_pair(k, v), "fork pair policy");
        }
    }

    struct config_case {
        int d;
        int ncols;
        fattn_mma_config cand;
        fattn_mma_config upstream;
    };
    const config_case changed[] = {
        {256, 16, { 64, 4, 32, 128, 128, 128, 2, true }, {256, 1, 64, 128, 128, 128, 2, true }},
        {512,  8, { 64, 4, 32, 256, 256, 128, 1, false}, {128, 2, 64, 128, 128, 128, 1, false}},
        {512, 16, { 64, 4, 32, 256, 256, 128, 1, false}, {256, 1, 64, 128, 128, 128, 1, false}},
        {512, 32, {128, 2, 32, 128, 128, 128, 1, false}, {256, 1, 32, 128, 128, 128, 1, false}},
    };
    for (const auto & c : changed) {
        check(ggml_cuda_fattn_mma_cand_shape(c.d, c.d, c.ncols), "changed shape");
        check(equal(ggml_cuda_fattn_mma_get_config_ampere(c.d, c.d, c.ncols, true), c.cand), "CAND config");
        check(equal(ggml_cuda_fattn_mma_get_config_ampere(c.d, c.d, c.ncols), c.upstream), "upstream config");
    }

    // Cover every query width and GQA tile, including padded single-query decode and the width-4 override.
    for (int width = 1; width <= 8; ++width) {
        for (int ncols2 : {1, 2, 4, 8}) {
            for (int min_width : {1, 2, 4}) {
                int ncols1 = 8/ncols2;
                while (ncols1 < width || (ncols2 == 8 && ncols1 < min_width)) {
                    ncols1 *= 2;
                }
                const int ncols = ncols1*ncols2;
                for (int d : {64, 80, 96, 112, 128, 256, 512}) {
                    const auto cand = ggml_cuda_fattn_mma_get_config_ampere(d, d, ncols, true);
                    const auto base = ggml_cuda_fattn_mma_get_config_ampere(d, d, ncols);
                    check(equal(cand, base) != ggml_cuda_fattn_mma_cand_shape(d, d, ncols), "shape isolation");
                    check(ggml_cuda_fattn_mma_tile_key(d, true, true) == 0, "CAND tile key");
                    check(ggml_cuda_fattn_mma_tile_key(d, false, true) == d, "upstream tile key");
                    check(ggml_cuda_fattn_mma_tile_key(d, true, false) == d, "non-Ampere tile key");
                }
            }
        }
    }
    for (const auto & dims : {std::pair<int, int>{192, 128}, {320, 256}, {576, 512}, {640, 512}}) {
        for (int ncols : {8, 16, 32, 64}) {
            check(!ggml_cuda_fattn_mma_cand_shape(dims.first, dims.second, ncols), "mixed head dimensions");
            check(equal(ggml_cuda_fattn_mma_get_config_ampere(dims.first, dims.second, ncols, true),
                        ggml_cuda_fattn_mma_get_config_ampere(dims.first, dims.second, ncols)), "unchanged mixed-dimension config");
        }
    }
    printf("test-fattn-mma-config: pair policy, CAND/upstream tiles and widths 1-8 passed\n");
    return 0;
}
