// [HQ4] CPU check of the prefill tile probe policy (ggml/src/ggml-cuda/fattn-prefill-policy.h) against an
// independent reference, over arms, compute capabilities, head sizes, query counts, GQA ratios and flags.

#include "../ggml/src/ggml-cuda/fattn-prefill-policy.h"

#include <cstdio>
#include <initializer_list>

// Today's ncols2 rule for SM86 non-Volta, non-RDNA (fattn.cu ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2).
static int today(int gqa, bool opt) {
    if (opt && gqa > 4) return 8;
    if (opt && gqa > 2) return 4;
    if (opt && gqa > 1) return 2;
    return 1;
}

int main() {
    int checks = 0, failures = 0;
    for (int arm : {0, 1, 2, 3, 4, 8, 16}) {
        for (int cc : {750, 800, 860, 870, 890}) {
            for (int d : {128, 256, 512}) {
                for (int dv : {128, 256}) {
                    for (int q : {1, 5, 8, 64, 127, 128, 129, 512, 4096}) {
                        for (int gqa : {1, 2, 3, 4, 5, 6, 7, 8, 12, 16}) {
                            for (bool one : {false, true}) {
                                for (bool opt : {false, true}) {
                                    const int got = ggml_cuda_fa_prefill_ncols2_pick(arm, cc, d, dv, q, gqa, one, opt);
                                    int ref = 0;
                                    if ((arm == 1 || arm == 2 || arm == 8) && cc == 860 && d == 256 && dv == 256 && q >= 128 && one) {
                                        if (arm == 1) {
                                            ref = 1;
                                        } else if (opt && gqa % arm == 0) {
                                            ref = arm;
                                        } else if (arm == 8 && opt && gqa > 4) {
                                            ref = 8;
                                        }
                                    }
                                    // arm 8 must be exactly today's launch wherever it applies
                                    if (got == 8 && today(gqa, opt) != 8) {
                                        fprintf(stderr, "FAIL arm 8 differs from today's rule gqa=%d opt=%d\n", gqa, opt);
                                        ++failures;
                                    }
                                    if (got != ref) {
                                        if (failures < 10) {
                                            fprintf(stderr, "FAIL arm=%d cc=%d d=%d dv=%d q=%d gqa=%d one=%d opt=%d got=%d ref=%d\n",
                                                arm, cc, d, dv, q, gqa, one, opt, got, ref);
                                        }
                                        ++failures;
                                    }
                                    ++checks;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // Qwen3.8 geometry (24/4 heads, GQA 6): arm 1 -> 1, arm 2 -> 2, arm 8 -> 8 (= today's), unset -> today's
    if (ggml_cuda_fa_prefill_ncols2_pick(1, 860, 256, 256, 512, 6, true, true) != 1 ||
        ggml_cuda_fa_prefill_ncols2_pick(2, 860, 256, 256, 512, 6, true, true) != 2 ||
        ggml_cuda_fa_prefill_ncols2_pick(8, 860, 256, 256, 512, 6, true, true) != 8 ||
        ggml_cuda_fa_prefill_ncols2_pick(0, 860, 256, 256, 512, 6, true, true) != 0 ||
        today(6, true) != 8) {
        fprintf(stderr, "FAIL Qwen3.8 geometry\n");
        ++failures;
    }
    if (failures) {
        fprintf(stderr, "test-fattn-prefill-policy: %d failure(s) in %d checks\n", failures, checks);
        return 1;
    }
    printf("test-fattn-prefill-policy: OK (%d checks)\n", checks);
    return 0;
}
