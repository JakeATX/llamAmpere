// [PP27] Launcher and kernel instances of magic2::mmq for the AIF25_U exact prompt path (GGML_CUDA_AIF25_MMQ), kept out of
// ai4.cu so that file does not grow by ten more heavy kernel instances (compile time).
#include "aif25-mmq.cuh"
#include "magic2-v2.cuh"
#include "../ggml-aif25.h"

#include <atomic>

template <int XSH>
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
    magic2::mmq<5,2,AIF25_MUL,XSH,NB,S><<<grid,128*(S),0,stream>>>(w, q, scales, y, M, K, (int) N)
    if (strips == 2) {
        if (tile == 32) { AIF25_MMQ_LAUNCH(4, 2); } else { AIF25_MMQ_LAUNCH(8, 2); }
    } else {
        if (tile == 32) { AIF25_MMQ_LAUNCH(4, 1); } else if (tile == 64) { AIF25_MMQ_LAUNCH(8, 1); } else { AIF25_MMQ_LAUNCH(16, 1); }
    }
#undef AIF25_MMQ_LAUNCH
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_aif25_mmq(int xsh, int tile, int strips, const uint8_t * w, const int8_t * q, const float * scales, float * y,
                         int M, int K, int64_t N, cudaStream_t stream) {
    return xsh ? aif25_mmq_launch<1>(tile, strips, w, q, scales, y, M, K, N, stream)
               : aif25_mmq_launch<0>(tile, strips, w, q, scales, y, M, K, N, stream);
}
