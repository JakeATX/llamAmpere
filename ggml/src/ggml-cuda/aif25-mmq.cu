// [PP27] Launcher and kernel instances of magic2::mmq for the AIF25_U exact prompt path (GGML_CUDA_AIF25_MMQ), kept out of
// ai4.cu so that file does not grow by ten more heavy kernel instances (compile time).
#include "aif25-mmq.cuh"
#include "magic2-v2.cuh"
#include "../ggml-aif25.h"
#include "../ggml-aif175.h"

#include <atomic>

template <int XSH, int B = 5, int V = 2, uint32_t C = AIF25_MUL, bool ROW_SCALE = false, int SCALE_CHUNKS = 5>
static bool aif25_mmq_launch(int tile, int strips, const uint8_t * w, const int8_t * q, const float * scales, float * y,
                      int M, int K, int64_t N, cudaStream_t stream) {
    const int64_t gy = (N + tile - 1) / tile;
    if ((strips != 1 && strips != 2) || (tile != 32 && tile != 64 && tile != 128) || (tile == 128 && strips != 1)) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) {
            GGML_LOG_WARN("[AIF25_MMQ] tile=%d strips=%d is not a built variant (32/64/128 with 1 strip, 32/64 with 2); using mmv\n",
                    tile, strips);
        }
        return false;
    }
    if (gy > 65535 || N > INT32_MAX) {
        return false;
    }
    const dim3 grid((unsigned) ((M/16 + strips - 1)/strips), (unsigned) gy);
#define AIF25_MMQ_LAUNCH(NB, S) \
    magic2::mmq<B,V,C,XSH,NB,S,ROW_SCALE,SCALE_CHUNKS><<<grid,128*(S),0,stream>>>(w, q, scales, y, M, K, (int) N)
    if (strips == 2) {
        if (tile == 32) { AIF25_MMQ_LAUNCH(4, 2); } else { AIF25_MMQ_LAUNCH(8, 2); }
    } else {
        if (tile == 32) { AIF25_MMQ_LAUNCH(4, 1); } else if (tile == 64) { AIF25_MMQ_LAUNCH(8, 1); } else { AIF25_MMQ_LAUNCH(16, 1); }
    }
#undef AIF25_MMQ_LAUNCH
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// [KAIMULTI] verify widths 9..32 (2-4 users x MTP verify): one block per 16-row strip covers all N tokens (NB = ceil(N/8)
// 8-token blocks), so each weight unit is decoded once instead of once per 8 tokens, with the mmv register prefetch.
template <int XSH, int B = 5, int V = 2, uint32_t C = AIF25_MUL, bool ROW_SCALE = false, int SCALE_CHUNKS = 5>
static bool aif25_narrow_launch(const uint8_t * w, const int8_t * q, const float * scales, float * y, int M, int K, int64_t N,
                                cudaStream_t stream) {
    if (N < 9 || N > 32) {
        return false;
    }
    const dim3 grid((unsigned) (M/16), 1);
    const int nb = (int) ((N + 7)/8);
#define AIF25_NARROW_LAUNCH(NB) \
    magic2::mmq<B,V,C,XSH,NB,1,ROW_SCALE,SCALE_CHUNKS,true><<<grid,128,0,stream>>>(w, q, scales, y, M, K, (int) N)
    if (nb == 2) { AIF25_NARROW_LAUNCH(2); } else if (nb == 3) { AIF25_NARROW_LAUNCH(3); } else { AIF25_NARROW_LAUNCH(4); }
#undef AIF25_NARROW_LAUNCH
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_aif25_mmq_narrow(ggml_type type, int xsh, const uint8_t * w, const int8_t * q, const float * scales, float * y,
                                int M, int K, int64_t N, cudaStream_t stream) {
    // same template arguments as ggml_cuda_aif25_mmq (and so as the mmv call of each type)
    switch (type) {
        case GGML_TYPE_AIF25_U:
            return xsh ? aif25_narrow_launch<1>(w, q, scales, y, M, K, N, stream)
                       : aif25_narrow_launch<0>(w, q, scales, y, M, K, N, stream);
        case GGML_TYPE_AIF25_U2:
            return xsh ? aif25_narrow_launch<1, 5, 2, AIF25_MUL, true, 5>(w, q, scales, y, M, K, N, stream)
                       : aif25_narrow_launch<0, 5, 2, AIF25_MUL, true, 5>(w, q, scales, y, M, K, N, stream);
        case GGML_TYPE_AIF25_U2_512:
            return xsh && aif25_narrow_launch<1, 5, 2, AIF25_MUL, true, 4>(w, q, scales, y, M, K, N, stream);
        case GGML_TYPE_AIF175_U2:
            return xsh && aif25_narrow_launch<1, 7, 4, AIF175_MUL, true, 4>(w, q, scales, y, M, K, N, stream);
        default:
            return false;
    }
}

bool ggml_cuda_aif25_mmq(ggml_type type, int xsh, int tile, int strips, const uint8_t * w, const int8_t * q, const float * scales,
                         float * y, int M, int K, int64_t N, cudaStream_t stream) {
    // [K126] the row-scale types take the same template arguments as their mmv call in ggml_cuda_mul_mat_aif25
    switch (type) {
        case GGML_TYPE_AIF25_U:
            return xsh ? aif25_mmq_launch<1>(tile, strips, w, q, scales, y, M, K, N, stream)
                       : aif25_mmq_launch<0>(tile, strips, w, q, scales, y, M, K, N, stream);
        case GGML_TYPE_AIF25_U2:
            return xsh ? aif25_mmq_launch<1, 5, 2, AIF25_MUL, true, 5>(tile, strips, w, q, scales, y, M, K, N, stream)
                       : aif25_mmq_launch<0, 5, 2, AIF25_MUL, true, 5>(tile, strips, w, q, scales, y, M, K, N, stream);
        // K % 512 == 0: always one activation scale per 256 columns
        case GGML_TYPE_AIF25_U2_512:
            return xsh && aif25_mmq_launch<1, 5, 2, AIF25_MUL, true, 4>(tile, strips, w, q, scales, y, M, K, N, stream);
        case GGML_TYPE_AIF175_U2:
            return xsh && aif25_mmq_launch<1, 7, 4, AIF175_MUL, true, 4>(tile, strips, w, q, scales, y, M, K, N, stream);
        default:
            return false;
    }
}
