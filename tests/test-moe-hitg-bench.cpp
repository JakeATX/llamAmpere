// fork STRATAGAP: kernel-only timing of the fused hit gate/up/SwiGLU pass (GGML_MOE_HIT_FUSED) for every shape of
// GGML_MOE_HIT_FUSED_SHAPE, on Flash-Next expert sizes (2560 x 640, 10 experts per token), so shapes can be
// compared in seconds instead of 100K-token runs. Run only through the card queue.
//
//   test-moe-hitg-bench [reps]
//
// Prints, per gate/up type and token count: us per launch, GB/s of weights read, and whether the shape's output has
// the bits of the default shape.
#include <cstdio>
#include <cstdlib>

extern "C" bool ggml_cuda_moe_hitg_bench(int type_i, int n_tokens, int n_pool, int n_sets, int reps, int shape,
        unsigned seed, double * us, double * mb, int * n_union, int * same);

int main(int argc, char ** argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 300;
    // ggml_type values: IQ2_XXS 16, IQ2_XS 17, IQ3_XXS 18, IQ3_S 21, IQ2_S 22
    const int types[] = { 16, 17, 22, 18, 21 };
    const char * tnames[] = { "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s" };
    const char * shapes[] = { "4x2", "8x2", "4x1", "8x1", "4x2s", "8x2s", "8x1s",
                              "sw16x2", "sw16x2g4", "sw8x2", "sw8x2g4", "sw16x4", "sw16x1g4", "sw16x2w8",
                              "sw16x1g4w16", "sw16x2g4w8", "sw16x1g4s", "sw16x2s" };
    const int toks[] = { 3, 4, 5 };
    bool ok = true;
    for (int ti = 0; ti < 5; ++ti) {
        for (int n_tok : toks) {
            for (int sh = 0; sh < 18; ++sh) {
                double us = 0, mb = 0; int n_union = 0, same = 0;
                if (!ggml_cuda_moe_hitg_bench(types[ti], n_tok, 36, 3, reps, sh, 7u, &us, &mb, &n_union, &same)) {
                    printf("%-8s w%d %-9s unsupported\n", tnames[ti], n_tok, shapes[sh]);
                    continue;
                }
                ok = ok && same;
                printf("%-8s w%d %-9s %7.2f us  %6.1f GB/s  experts %2d  %5.1f MB  bits %s\n", tnames[ti], n_tok,
                       shapes[sh], us, mb / us * 1000.0, n_union, mb, same ? "same" : "DIFFERENT");
                fflush(stdout);
            }
        }
    }
    printf(ok ? "ALL SAME\n" : "SOME DIFFERENT\n");
    return ok ? 0 : 1;
}
