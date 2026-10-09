#pragma once

#include "ggml.h"

#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__MUSACC__)
#define GGML_FATTN_HD __host__ __device__
#else
#define GGML_FATTN_HD
#endif

// Keep the pre-#29152 tile for fork KV and the q8/q8 draft cache.
static constexpr GGML_FATTN_HD bool ggml_cuda_fattn_mma_cand_type(const ggml_type type) {
    return type == GGML_TYPE_TURBO2_0 || type == GGML_TYPE_TURBO3_0 || type == GGML_TYPE_TURBO4_0 ||
           type == GGML_TYPE_TQ5_0 || type == GGML_TYPE_TQ6_0;
}

static constexpr GGML_FATTN_HD bool ggml_cuda_fattn_mma_cand_pair(const ggml_type type_K, const ggml_type type_V) {
    return ggml_cuda_fattn_mma_cand_type(type_K) || ggml_cuda_fattn_mma_cand_type(type_V) ||
           (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q8_0);
}

static constexpr GGML_FATTN_HD bool ggml_cuda_fattn_mma_cand_shape(const int DKQ, const int DV, const int ncols) {
    return (DKQ == 256 && DV == 256 && ncols == 16) ||
           (DKQ == 512 && DV == 512 && (ncols == 8 || ncols == 16 || ncols == 32));
}

// A zero key selects the original generic tiles, including the thin ncols-8 tile.
static constexpr GGML_FATTN_HD int ggml_cuda_fattn_mma_tile_key(const int DKQ, const bool preserve_cand, const bool ampere) {
    return preserve_cand && ampere ? 0 : DKQ;
}

// Config options for the MMA kernel.
// Tile geometry changes the floating-point reduction order.
struct fattn_mma_config {
    int  nthreads;       // Number of threads per CUDA block.
    int  occupancy;      // Targeted occupancy for the MMA kernel.
    int  nbatch_fa;      // Number of KV rows per softmax rescaling of KQ rowsums and VKQ accumulators.
    int  nbatch_K2;      // Number of K half2 values in direction of DKQ to load in parallel.
    int  nbatch_V2;      // Number of V half2 values in direction of DV to load in parallel.
    int  nbatch_combine; // Number of VKQ half2 values in direction of DV to combine in parallel.
    int  nstages_target; // Number of pipeline stages to use ideally, 1 == always load data synchronously, 2 == preload data if there is hardware support.
    bool Q_in_reg;       // Whether the Q values should be kept permanently in registers.

    constexpr GGML_FATTN_HD fattn_mma_config(
            int nthreads, int occupancy, int nbatch_fa, int nbatch_K2, int nbatch_V2, int nbatch_combine, int nstages_target, bool Q_in_reg) :
        nthreads(nthreads), occupancy(occupancy), nbatch_fa(nbatch_fa), nbatch_K2(nbatch_K2), nbatch_V2(nbatch_V2), nbatch_combine(nbatch_combine),
        nstages_target(nstages_target), Q_in_reg(Q_in_reg) {}
};

#define GGML_CUDA_FATTN_MMA_CONFIG_CASE(DKQ_, DV_, ncols_, nthreads_, occupancy_, nbatch_fa_, nbatch_K2_, nbatch_V2_, nbatch_combine_, nstages_target_, Q_in_reg_) \
    if (DKQ == (DKQ_) && DV == (DV_) && ncols == (ncols_)) {                                                                                                       \
        static_assert((nthreads_)       % 32 == 0 && (nthreads_)       <= 512, "bad nthreads");                                                                    \
        static_assert(                               (occupancy_)      <=   8, "bad occupancy");                                                                   \
        static_assert((nbatch_fa_)      % 32 == 0 && (nbatch_fa_)      <= 256, "bad nbatch_fa");                                                                   \
        static_assert((nbatch_K2_)      %  4 == 0 && (nbatch_K2_)      <= 512, "bad nbatch_K2");                                                                   \
        static_assert((nbatch_V2_)      %  4 == 0 && (nbatch_V2_)      <= 256, "bad nbatch_V2");                                                                   \
        static_assert((nbatch_combine_) %  4 == 0 && (nbatch_combine_) <= 128, "bad nbatch_combine");                                                              \
        static_assert((nstages_target_)      >= 1 && (nstages_target_) <=   2, "bad nstages_target");                                                              \
        return fattn_mma_config{(nthreads_), (occupancy_), (nbatch_fa_), (nbatch_K2_), (nbatch_V2_), (nbatch_combine_), (nstages_target_), (Q_in_reg_)};           \
    }                                                                                                                                                              \

static constexpr GGML_FATTN_HD fattn_mma_config ggml_cuda_fattn_mma_get_config_ampere(const int DKQ, const int DV, const int ncols, const bool preserve_cand = false) {
    if (preserve_cand) {
        GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16,  64, 4, 32, 128, 128, 128, 2, true);
        GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8,  64, 4, 32, 256, 256, 128, 1, false);
        GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16,  64, 4, 32, 256, 256, 128, 1, false);
        GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 128, 2, 32, 128, 128, 128, 1, false);
    }

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64,  8, 128, 2, 128,  32,  32,  32, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 16, 128, 2,  64,  32,  32,  32, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 32, 128, 2,  64,  32,  32,  32, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 64, 128, 2,  64,  32,  32,  32, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80,  8, 128, 2, 128,  40,  40,  40, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 16, 128, 2,  64,  40,  40,  40, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 32, 128, 2,  64,  40,  40,  40, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 64, 128, 2,  64,  40,  40,  40, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96,  8, 128, 2, 128,  48,  48,  48, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 16, 128, 2,  64,  48,  48,  48, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 32, 128, 2,  64,  48,  48,  48, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 64, 128, 2,  64,  48,  48,  48, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112,  8, 128, 2, 128,  56,  56,  56, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 16, 128, 2,  64,  56,  56,  56, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 32, 128, 2,  64,  56,  56,  56, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 64, 128, 2,  64,  56,  56,  56, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128,  8, 128, 2, 128,  64,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 16, 128, 2,  64,  64,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 32, 128, 2,  64,  64,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 64, 128, 2,  64,  64,  64,  64, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128,  8,  64, 4,  64,  96,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 16,  64, 4,  32,  96,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 32, 128, 2,  32,  96,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 64, 128, 2,  32,  96,  64,  64, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256,  8, 128, 2,  64, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16, 256, 1,  64, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 128, 2,  32, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2,  32, 128, 128, 128, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 32, 128, 2,  32, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 64, 256, 1,  32, 128, 128, 128, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8, 128, 2,  64, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16, 256, 1,  64, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 256, 1,  32, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 256, 1,  32, 128, 128, 128, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8,  64, 4,  32, 288, 256, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16,  64, 4,  32, 288, 256, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2,  32, 160, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  32, 160, 128, 128, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(640, 512,  8,  64, 4,  32, 288, 256, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(640, 512, 16,  64, 4,  32, 288, 256, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(640, 512, 32, 128, 2,  32, 160, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(640, 512, 64, 256, 1,  32, 160, 128, 128, 1, false);

    return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
}


#undef GGML_FATTN_HD
