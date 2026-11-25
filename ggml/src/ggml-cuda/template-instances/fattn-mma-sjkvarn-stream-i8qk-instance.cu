// This file has been hand-created ([#237] int8-QK body of the SJ-KVaRN 4/4 stream kernel). Do NOT run
// generate_cu_files.py over it.
//
// GGML_SJKVARN_STREAM_I8QK=1 (default off) only: the rows-tile instances of the stream kernel (warps 4, 5, 6, 8; ring
// depth 4 and 6; plain and fused-rotation builds; with and without the live-record walk) with the int8-QK body. The
// default instances stay in fattn-mma-sjkvarn-direct-instance.cu and are unchanged. (Named fattn-mma-* for the CMake glob.)

#include "../fattn-sjkvarn-stream.cuh"

template <int max_warps, int NSTAGE, bool rot, bool lsa>
static fattn_kernel_t sj_i8qk_kernel() {
    return ggml_cuda_fattn_sj_kvarn_stream_kernel<max_warps, NSTAGE, false, rot, lsa, sj_kvarn_tile::rows, true>();
}

template <int max_warps, int NSTAGE>
static fattn_kernel_t sj_i8qk_pick(const bool rot, const bool lsa) {
    if (rot) {
        return lsa ? sj_i8qk_kernel<max_warps, NSTAGE, true, true>() : sj_i8qk_kernel<max_warps, NSTAGE, true, false>();
    }
    return lsa ? sj_i8qk_kernel<max_warps, NSTAGE, false, true>() : sj_i8qk_kernel<max_warps, NSTAGE, false, false>();
}

template <int max_warps>
static fattn_kernel_t sj_i8qk_stage(const int nstage, const bool rot, const bool lsa) {
    switch (nstage) {
        case 6:  return sj_i8qk_pick<max_warps, 6>(rot, lsa);
        case 4:  return sj_i8qk_pick<max_warps, 4>(rot, lsa);
        default: return nullptr;
    }
}

fattn_kernel_t ggml_cuda_fattn_sj_kvarn_stream_i8qk_kernel(int max_warps, int nstage, bool rot, bool lsa) {
    switch (max_warps) {
        case 4:  return sj_i8qk_stage<4>(nstage, rot, lsa);
        case 5:  return sj_i8qk_stage<5>(nstage, rot, lsa);
        case 6:  return sj_i8qk_stage<6>(nstage, rot, lsa);
        case 8:  return sj_i8qk_stage<8>(nstage, rot, lsa);
        default: return nullptr;
    }
}
