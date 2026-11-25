// INKNVFP4K Q8TC: host side (see q8tc-kern.cuh)
#include "q8tc.cuh"
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "q8tc-kern.cuh"
#endif

#include <cstdlib>
#include <cstring>
#include <vector>

#define Q8TC_MAX_ROWS 16384

int ggml_cuda_q8tc_mode() {
    // off unless set: the per-arch defaults table sets GGML_CUDA_Q8TC (0.7.5 Inkling: 2 = hi/lo; E21 ub5 KLD -4.4e-5)
    static const int m = [] { const char * e = getenv("GGML_CUDA_Q8TC"); const int v = e ? atoi(e) : 0; return v == 1 || v == 2 ? v : 0; }();
    return m;
}

static int q8tc_min_t() {
    static const int m = [] { const char * e = getenv("GGML_CUDA_Q8TC_MINT"); const int v = e ? atoi(e) : 4; return v < 2 ? 2 : v; }();
    return m;
}

// Q8TC-PT: GGML_CUDA_Q8TC_M1 = comma list of "<kind><first>-<last>" (kind g = grouped launch, o = plain launch, e.g.
// attn_output; layer from the weight name blk.N.) that run mode 1 while GGML_CUDA_Q8TC=2; "g" / "o" alone = all layers.
struct q8tc_sel { char kind; int lo, hi; };
static const std::vector<q8tc_sel> & q8tc_m1_sel() {
    static const std::vector<q8tc_sel> v = [] {
        std::vector<q8tc_sel> r;
        const char * e = getenv("GGML_CUDA_Q8TC_M1");
        for (const char * p = e; p && *p; ) {
            if (*p == ',') { ++p; continue; }
            q8tc_sel s{*p++, 0, 1 << 30};
            if (*p >= '0' && *p <= '9') {
                char * q = nullptr;
                s.lo = s.hi = (int) strtol(p, &q, 10);
                p = q;
                if (*p == '-') { const char * h = p + 1; s.hi = (int) strtol(h, &q, 10); s.hi = q == h ? 1 << 30 : s.hi; p = q; }   // "g5-" = 5..last
            }
            if (s.kind == 'g' || s.kind == 'o') { r.push_back(s); }
            while (*p && *p != ',') { ++p; }
            if (*p == ',') { ++p; }
        }
        return r;
    }();
    return v;
}

static int q8tc_tensor_mode(int mode, const ggml_tensor * w, bool group) {
    if (mode != 2 || w == nullptr || q8tc_m1_sel().empty()) {
        return mode;
    }
    int il = -1;
    if (strncmp(w->name, "blk.", 4) == 0) { il = atoi(w->name + 4); }
    for (const q8tc_sel & s : q8tc_m1_sel()) {
        if (s.kind == (group ? 'g' : 'o') && il >= s.lo && il <= s.hi) { return 1; }
    }
    return mode;
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// tuned on a 3090 (300 W) at the NQ shapes: K = 2304 (9 chunks) -> 3 warps, 4 CTAs/SM; K = 3072 (12 chunks) -> 6 warps, 2 CTAs/SM
// GGML_CUDA_Q8TC_IA=1 (Q8TC-HLF): mode 2 sums its hi / lo and k-step mmas in independent accumulators (no dependent
// mma chain; closer to the f32 path, not bit-exact vs IA=0). Default 0 until gated.
static bool q8tc_ia() {
    static const bool p = [] { const char * e = getenv("GGML_CUDA_Q8TC_IA"); return e && atoi(e) == 1; }();
    return p;
}

template <int NW, bool HILO, int MINB>
static void q8tc_launch(const q8a16_args & a, int T, int nb, cudaStream_t st) {
    if constexpr (HILO) {
        if (q8tc_ia()) {
            k_q8tc3<NW, HILO, MINB, true><<<a.blk_end[a.n - 1], 32*NW, NW*16*68*sizeof(uint32_t), st>>>(a, nb, T);
            return;
        }
    }
    k_q8tc3<NW, HILO, MINB><<<a.blk_end[a.n - 1], 32*NW, NW*16*68*sizeof(uint32_t), st>>>(a, nb, T);
}

bool ggml_cuda_q8tc_try(ggml_backend_cuda_context & ctx, q8a16_args & a, int T, int nb, const ggml_tensor * w, bool group) {
    const int mode = q8tc_tensor_mode(ggml_cuda_q8tc_mode(), w, group);
    if (mode == 0 || T < q8tc_min_t() || T > 8 || nb < 8 || nb % 8 != 0 || a.ids != nullptr || a.n < 1 ||
            !ampere_mma_available(ggml_cuda_info().devices[ctx.device].cc)) { // compiled sm_80+ code, not only the card
        return false;
    }
    for (int k = 0; k < a.n; ++k) {
        // 16-byte aligned rows (cp.async chunks), float4 activation reads, no head-sized members
        if (a.srow[k] % 16 != 0 || (uintptr_t) a.vx[k] % 16 != 0 || (uintptr_t) a.x[k] % 16 != 0 || a.sx[k] % 4 != 0 ||
                a.nrows[k] < 1 || a.nrows[k] > Q8TC_MAX_ROWS) {
            return false;
        }
    }
    uint32_t blk = 0;
    for (int k = 0; k < a.n; ++k) {
        blk += (uint32_t) ((a.nrows[k] + 15) / 16);
        a.blk_end[k] = blk;
    }
    cudaStream_t st = ctx.stream();
    const bool few = nb/8 <= 9;
    if (mode == 1) {
        few ? q8tc_launch<3, false, 4>(a, T, nb, st) : q8tc_launch<6, false, 2>(a, T, nb, st);
    } else {
        few ? q8tc_launch<3, true, 4>(a, T, nb, st) : q8tc_launch<6, true, 2>(a, T, nb, st);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#else

bool ggml_cuda_q8tc_try(ggml_backend_cuda_context &, q8a16_args &, int, int, const ggml_tensor *, bool) { return false; }

#endif
