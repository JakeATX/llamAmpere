#pragma once
// Host-side clamps for the small-width CUDA launch sweep (backlog #244 step 1).
// The CUDA launchers call these. tests/test-rpb-knobs.cpp locks the same rules.
// No CUDA headers: a CPU test can include this file.

#include <cstdlib>
#include <cerrno>

struct rpb_ai4_ampere_range {
    int min_n;
    int max_n;
    bool rejected;

    bool contains(int n) const {
        return n >= min_n && n <= max_n;
    }
};

inline int rpb_ai4_width(const char * text, int fallback, bool & rejected) {
    if (text == nullptr || text[0] == '\0') {
        return fallback;
    }
    char * end = nullptr;
    errno = 0;
    const long n = std::strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || n < 1 || n > 64) {
        rejected = true;
        return fallback;
    }
    return (int) n;
}

inline rpb_ai4_ampere_range rpb_ai4_ampere_resolve(const char * min_text, const char * max_text) {
    rpb_ai4_ampere_range r{9, 32, false};
    r.min_n = rpb_ai4_width(min_text, 9, r.rejected);
    r.max_n = rpb_ai4_width(max_text, 32, r.rejected);
    if (r.rejected || r.min_n > r.max_n) {
        r = {9, 32, true};
    }
    return r;
}

// Today's compiles. An unset variable must pick these so the launch stays on
// the historical template instance.
inline constexpr int rpb_exl3_nwarps_default = 4;
inline constexpr int rpb_exl3_nwarps_allowed[] = {2, 4, 8};
inline constexpr int rpb_exl3_nwarps_n = 3;

inline constexpr int rpb_ai4_warps_default = 4;
inline constexpr int rpb_ai4_warps_allowed[] = {2, 4, 8};
inline constexpr int rpb_ai4_warps_n = 3;

inline constexpr int rpb_ai4_minb_default = 6;
inline constexpr int rpb_ai4_minb_allowed[] = {2, 4, 6, 8};
inline constexpr int rpb_ai4_minb_n = 4;

// text null or empty (variable unset): fallback, *rejected left alone.
// text set but not in the allowed list, or not an integer: fallback and *rejected = true.
// atoi-compatible: leading whitespace is skipped, trailing junk is ignored, base 10.
inline int rpb_pick_env(const char * text, const int * allowed, int n, int fallback, bool * rejected) {
    if (text == nullptr || text[0] == '\0') {
        return fallback;
    }
    char * end = nullptr;
    const long v = std::strtol(text, &end, 10);
    if (end == text || v > 2147483647L || v < -2147483647L - 1) {
        if (rejected) {
            *rejected = true;
        }
        return fallback;
    }
    const int iv = (int) v;
    for (int i = 0; i < n; ++i) {
        if (allowed[i] == iv) {
            if (rejected) {
                *rejected = false;
            }
            return iv;
        }
    }
    if (rejected) {
        *rejected = true;
    }
    return fallback;
}

struct rpb_ai4_mmv {
    int  warps;
    int  minb;
    bool rejected_warps;
    bool rejected_minb;
};

inline rpb_ai4_mmv rpb_ai4_resolve(const char * warps_text, const char * minb_text) {
    rpb_ai4_mmv r;
    r.rejected_warps = false;
    r.rejected_minb  = false;
    r.warps = rpb_pick_env(warps_text, rpb_ai4_warps_allowed, rpb_ai4_warps_n, rpb_ai4_warps_default, &r.rejected_warps);
    r.minb  = rpb_pick_env(minb_text,  rpb_ai4_minb_allowed,  rpb_ai4_minb_n,  rpb_ai4_minb_default,  &r.rejected_minb);
    return r;
}

// k-split used by ggml_cuda_exl3_gemv_ksplit. nwarps is the block's warp count.
// With nwarps == 4 this is the arithmetic the base commit compiled in.
inline constexpr int rpb_exl3_ksplit(int kt, int nt, int ktiles_max, int nwarps) {
    int ksplit = (1536 + nt - 1) / nt;
    const int ksplit_max = kt / (2 * nwarps);
    if (ksplit > ksplit_max) {
        ksplit = ksplit_max;
    }
    if (ksplit < 1) {
        ksplit = 1;
    }
    while ((kt + ksplit - 1) / ksplit > ktiles_max) {
        ksplit++;
    }
    return ksplit;
}

// exl3_gemv_cfg<T>, with EXL3_GEMV_MMA_MIN_T == 2. The .cu static_asserts this
// against the real cfg so the two cannot drift.
inline constexpr int rpb_exl3_xrows(int T) {
    const bool mma = T >= 2;
    return mma ? (T <= 8 ? 8 : 16) : T;
}

inline constexpr int rpb_exl3_ktiles_max(int T) {
    const int xrows = rpb_exl3_xrows(T);
    return xrows <= 2 ? 128 : (xrows <= 8 ? 64 : 32);
}

// Static shared bytes. Both kernels take 0 dynamic shared memory.
// Old kernel: part[nwarps][T][pitch] floats + xs[xrows][ktiles*8] half2.
// Dec kernel: part is double-buffered, and weight-major stages T rows of x.
inline constexpr int rpb_exl3_gemv_smem(int nwarps, int T, bool weight_major) {
    const int pitch = weight_major ? 20 : 16;
    const int part  = nwarps * T * pitch * 4;
    const int xs    = rpb_exl3_xrows(T) * rpb_exl3_ktiles_max(T) * 8 * 4;
    return part + xs;
}

inline constexpr int rpb_exl3_gemv_dec_smem(int nwarps, int T, bool weight_major) {
    const int pitch   = weight_major ? 20 : 16;
    const int part    = 2 * nwarps * T * pitch * 4;
    const int xs_rows = weight_major ? T : rpb_exl3_xrows(T);
    const int xs      = xs_rows * rpb_exl3_ktiles_max(T) * 8 * 4;
    return part + xs;
}

// 48 KiB is the static shared-memory ceiling that does not need an opt-in.
// Block is 32 * nwarps threads (one dimension), under the 1024-thread cap.
inline constexpr bool rpb_exl3_instance_ok(int nwarps, int T, bool dec, bool weight_major) {
    if (nwarps != 2 && nwarps != 4 && nwarps != 8) {
        return false;
    }
    if (T < 1 || T > 16) {
        return false;
    }
    if (nwarps * 32 > 1024) {
        return false;
    }
    const int smem = dec ? rpb_exl3_gemv_dec_smem(nwarps, T, weight_major)
                         : rpb_exl3_gemv_smem(nwarps, T, weight_major);
    return smem > 0 && smem <= 48 * 1024;
}

// Which pre-instantiated EXL3 kernel a non-default nwarps request may use.
// requested is already clamped to {2, 4, 8}. nwarps 4 always means "historical
// launch" (use_dec is meaningless then; the existing dispatcher picks the kernel).
// Alt kernels that exist:
//   T == 1, dec off for that T: k_exl3_gemv<BITS, 1, FUSED=false, WM=false, NWARPS>
//   T in 2..8, default dec variant: k_exl3_gemv_dec<BITS, T, WM, FR, PF2=false, NWARPS>
//     WM+FR is the default only when weight-major and fragment staging are on,
//     prefetch-2 is off, and (bits 3 or 4, or compute capability 8.6).
// Fused, T outside 1..8, and every other variant stay at 4 warps.
struct rpb_exl3_alt {
    int  nwarps;
    bool use_dec;
    bool ignored; // requested 2 or 8, but this shape has no alt instance
};

inline constexpr rpb_exl3_alt rpb_exl3_resolve(int requested, int T, bool fused, int bits,
                                    bool dec_enabled, int dec_tmin, bool weight_major, bool vw, bool pf2,
                                    int cc, bool build_weight_major) {
    rpb_exl3_alt r{};
    r.nwarps  = rpb_exl3_nwarps_default;
    r.use_dec = false;
    r.ignored = false;
    if (requested != 2 && requested != 4 && requested != 8) {
        r.ignored = true;
        return r;
    }
    if (requested == rpb_exl3_nwarps_default) {
        return r;
    }
    const bool t_ok    = T >= 1 && T <= 8;
    const bool bits_ok = bits >= 2 && bits <= 8;
    const bool dec_on  = dec_enabled && T >= dec_tmin;
    if (!fused && t_ok && bits_ok && T == 1 && !dec_on) {
        r.nwarps = requested;
        return r;
    }
    // 3- and 4-bit weight-major is on for every arch; other widths only on sm_86.
    const bool wm_ok = build_weight_major && weight_major && vw && !pf2 &&
                       (bits == 3 || bits == 4 || cc == 860);
    if (!fused && t_ok && bits_ok && T >= 2 && dec_on && wm_ok) {
        r.nwarps  = requested;
        r.use_dec = true;
        return r;
    }
    r.ignored = true;
    return r;
}

// AI+ mmv ring, per stage. Matches ai4k::mmv_stage_bytes: chunk bytes (1024, or
// 1280 at rate 5) + 16 fp16 scale words (64) + 8 activation scales (32) + 1024
// when activations are staged in shared memory.
inline constexpr int rpb_ai4_stage_bytes(bool xreg, bool k5) {
    const int ch = k5 ? 1280 : 1024;
    return ch + 64 + 32 + (xreg ? 0 : 1024);
}

inline constexpr int rpb_ai4_dynamic_smem(int warps, int stages, bool xreg, bool k5) {
    return warps * stages * rpb_ai4_stage_bytes(xreg, k5);
}

// red[warps - 1][4][32] floats. warps == 1 would be a zero-length array; not in the set.
inline constexpr int rpb_ai4_static_smem(int warps) {
    return (warps - 1) * 4 * 32 * 4;
}

// sm_86 (this build): 1024 threads/block, 48 warps/SM, 1536 threads/SM, 99 KiB shared/block.
// __launch_bounds__(warps*32, minb) is a legal register hint when those products fit.
// 8 warps and min blocks 8 is 64 warps / 2048 threads, which ptxas rejects, so that pair is out.
// Shared memory may keep fewer than minb blocks resident; the launch still succeeds.
inline constexpr bool rpb_ai4_mmv_ok(int warps, int minb, int stages, bool xreg, bool k5) {
    if (warps < 2 || minb < 1 || stages < 1) {
        return false;
    }
    const int threads = warps * 32;
    if (threads > 1024) {
        return false;
    }
    if ((long long) warps * minb > 48) {
        return false;
    }
    if ((long long) threads * minb > 1536) {
        return false;
    }
    const int smem = rpb_ai4_dynamic_smem(warps, stages, xreg, k5) + rpb_ai4_static_smem(warps);
    return smem > 0 && smem <= 99 * 1024;
}
