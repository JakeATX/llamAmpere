// [KQROWS] q4_K launcher, split from the other types so the three files compile in parallel.
#include "mmvq-kqrows-impl.cuh"

template void ggml_cuda_kqrows_launch<GGML_TYPE_Q4_K>(const void * vx, const void * vy_q8_1, float * dst,
        int ncols_x, int nrows_x, int ncols_dst, int stride_row_x, int stride_col_y, int stride_col_dst,
        int rows, int nwarps, cudaStream_t stream);
