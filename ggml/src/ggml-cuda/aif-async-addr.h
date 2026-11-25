#pragma once
// [OPT10] Weight-unit addressing of the cp.async fractional AI+ matvec (aif-async-mmv.cu), kept free of CUDA so that
// tests/test-aif-async-addr.cpp can check it on the CPU against the addressing of magic2::mmv / magic2::mmq.
//
// A 16-row strip ms of a fractional AI+ matrix is nu = K / 128 units of UB bytes: a 32-byte header (16 fp16 row scales)
// and a PB-byte pair-stream payload. AIF25_U stores the units back to back (ms * nu + u). The row-scale types store
// blocks of one header + SC payloads (AIF25_U2: SC = 5; AIF25_U2_512 and AIF175_U2: SC = 4): unit u of strip ms is
// header + payload (u % SC) of block ms * (nu / SC) + u / SC. Warp w of a strip takes units u = w, w + 4, ...; the
// cursor walks that sequence with no divisions: !ROW_SCALE adds 4 units; SC == 4 keeps the payload slot (u % 4 == w)
// and adds one block; SC == 5 (4 == -1 mod 5) either moves the slot from 0 to 4 in the same block or lowers it by one
// in the next block. chunk(i) is the byte offset (from the matrix base) of the i-th 16 bytes of the current unit as the
// kernels place it in shared memory: chunks 0-1 = header, chunks 2.. = payload.

#include <cstdint>

#if defined(__CUDACC__)
#define AIF_ASYNC_HD __host__ __device__ __forceinline__
#else
#define AIF_ASYNC_HD inline
#endif

namespace aif_async {

template <int B, int V> constexpr int unit_bytes = 32 + 64 * (4 * B / V);   // == magic2::unit_bytes<B, V>

template <int B, int V, bool ROW_SCALE, int SC>
struct cursor {
    static constexpr int UB      = unit_bytes<B, V>;
    static constexpr int PB      = UB - 32;
    static constexpr int BLOCK_B = ROW_SCALE ? 32 + SC * PB : UB;
    static constexpr int CHUNKS  = UB / 16;
    static_assert(UB % 16 == 0 && PB % 16 == 0, "units are whole 16-byte chunks");
    static_assert(!ROW_SCALE || SC == 4 || SC == 5, "row-scale blocks hold 4 or 5 payloads");

    int64_t base;   // byte offset of the current unit (!ROW_SCALE) or of its block (ROW_SCALE)
    int     slot;   // u % SC (ROW_SCALE only)

    AIF_ASYNC_HD void init(int64_t ms, int nu, int warp) {
        if (ROW_SCALE) {
            base = ms * (nu / SC) * int64_t(BLOCK_B);   // warp < 4 <= SC: block 0 of the strip, slot = warp
            slot = warp;
        } else {
            base = (ms * nu + warp) * int64_t(UB);
            slot = 0;
        }
    }

    AIF_ASYNC_HD void advance() {
        if (!ROW_SCALE) {
            base += 4 * int64_t(UB);
        } else if (SC == 4) {
            base += BLOCK_B;
        } else if (slot == 0) {
            slot = 4;
        } else {
            slot -= 1;
            base += BLOCK_B;
        }
    }

    AIF_ASYNC_HD int64_t chunk(int i) const {
        if (!ROW_SCALE) {
            return base + 16 * int64_t(i);
        }
        return i < 2 ? base + 16 * i : base + 32 + int64_t(slot) * PB + 16 * int64_t(i - 2);
    }
};

} // namespace aif_async
