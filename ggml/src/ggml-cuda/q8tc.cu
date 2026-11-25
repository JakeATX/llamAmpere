// INKNVFP4K Q8TC: host side (see q8tc-kern.cuh)
#include "q8tc.cuh"
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "q8tc-kern.cuh"
#endif

#include <climits>
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
bool ggml_cuda_q8tc_ia() {
    static const bool p = [] { const char * e = getenv("GGML_CUDA_Q8TC_IA"); return e && atoi(e) == 1; }();
    return p;
}

// GGML_CUDA_Q8TC_PIPE=1 (Q8TC-HLF PIPE): double-buffered chunk copies, same arithmetic (bit-exact); measured slower
// (+5..+22%), default off. Not combined with IA (IA wins when both are set).
static bool q8tc_pipe() {
    static const bool p = [] { const char * e = getenv("GGML_CUDA_Q8TC_PIPE"); return e && atoi(e) == 1; }();
    return p;
}

template <int NW, bool HILO, int MINB>
static void q8tc_launch(const q8a16_args & a, int T, int nb, cudaStream_t st) {
    if (q8tc_pipe() && !(HILO && ggml_cuda_q8tc_ia())) {
        const size_t sm = 2*NW*16*68*sizeof(uint32_t);   // 52 KB at NW = 6: over the 48 KB default
        static bool raised[GGML_CUDA_MAX_DEVICES] = {false};   // per device, like mmq
        const int id = ggml_cuda_get_device();
        if (!raised[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(k_q8tc3<NW, HILO, MINB, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm));
            raised[id] = true;
        }
        k_q8tc3<NW, HILO, MINB, false, true><<<a.blk_end[a.n - 1], 32*NW, sm, st>>>(a, nb, T);
        return;
    }
    if constexpr (HILO) {
        if (ggml_cuda_q8tc_ia()) {
            k_q8tc3<NW, HILO, MINB, true><<<a.blk_end[a.n - 1], 32*NW, NW*16*68*sizeof(uint32_t), st>>>(a, nb, T);
            return;
        }
    }
    k_q8tc3<NW, HILO, MINB><<<a.blk_end[a.n - 1], 32*NW, NW*16*68*sizeof(uint32_t), st>>>(a, nb, T);
}

int ggml_cuda_q8tc_prep(ggml_backend_cuda_context & ctx, q8a16_args & a, int T, int nb, const ggml_tensor * w, bool group) {
    const int mode = q8tc_tensor_mode(ggml_cuda_q8tc_mode(), w, group);
    if (mode == 0 || T < q8tc_min_t() || T > 8 || nb < 8 || nb % 8 != 0 || a.ids != nullptr || a.n < 1 ||
            !ampere_mma_available(ggml_cuda_info().devices[ctx.device].cc)) { // compiled sm_80+ code, not only the card
        return 0;
    }
    for (int k = 0; k < a.n; ++k) {
        // 16-byte aligned rows (cp.async chunks), float4 activation reads, no head-sized members
        if (a.srow[k] % 16 != 0 || (uintptr_t) a.vx[k] % 16 != 0 || (uintptr_t) a.x[k] % 16 != 0 || a.sx[k] % 4 != 0 ||
                a.nrows[k] < 1 || a.nrows[k] > Q8TC_MAX_ROWS) {
            return 0;
        }
    }
    uint32_t blk = 0;
    for (int k = 0; k < a.n; ++k) {
        blk += (uint32_t) ((a.nrows[k] + 15) / 16);
        a.blk_end[k] = blk;
    }
    return mode;
}

bool ggml_cuda_q8tc_try(ggml_backend_cuda_context & ctx, q8a16_args & a, int T, int nb, const ggml_tensor * w, bool group) {
    const int mode = ggml_cuda_q8tc_prep(ctx, a, T, nb, w, group);
    if (mode == 0) {
        return false;
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

// Q8TC-MU: GGML_CUDA_Q8TC_MU=1 (default off) runs a plain Q8_0 x f32 mul_mat of 9..32 columns (multi-slot draft verify)
// as ceil(T/8) Q8TC launches of near-equal column slices (4..8 columns each), instead of an MMQ tile over q8_1 activations.
static bool q8tc_mu() {
    static const bool p = [] { const char * e = getenv("GGML_CUDA_Q8TC_MU"); return e && atoi(e) == 1; }();
    return p;
}

bool ggml_cuda_q8tc_wide(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    if (!q8tc_mu() || ggml_cuda_q8tc_mode() == 0 || src0->type != GGML_TYPE_Q8_0) {
        return false;
    }
    const int64_t T = src1->ne[1];
    if (T < 9 || T > 32 || !src0->buffer || ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE ||
            src0->nb[0] != ggml_type_size(GGML_TYPE_Q8_0) || src0->ne[0] % QK8_0 != 0 || src0->ne[2] != 1 || src0->ne[3] != 1 ||
            src0->nb[1] != (size_t) (src0->ne[0]/QK8_0)*sizeof(block_q8_0) || src0->ne[1] > INT_MAX ||
            src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src1->ne[0] != src0->ne[0] ||
            src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[0] != src0->ne[1] || dst->ne[1] != T || dst->ne[2] != 1 || dst->ne[3] != 1 ||
            src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) || src1->nb[1] % 16 != 0 || dst->nb[1] % sizeof(float) != 0 ||
            (uintptr_t) src1->data % 16 != 0) {
        return false;
    }
    const int ns = (int) ((T + 7)/8);
    const int64_t sx = (int64_t) (src1->nb[1]/sizeof(float)), sd = (int64_t) (dst->nb[1]/sizeof(float));
    const int nb = (int) (src0->ne[0]/QK8_0);
    // slices differ only in the column offset (sx % 4 == 0 keeps x 16-byte aligned), so the narrowest one is the only
    // extra condition: when it is below MINT nothing is launched and the MMQ route runs the whole product
    if ((int) (T/ns) < q8tc_min_t()) {
        return false;
    }
    for (int s = 0, c0 = 0; s < ns; ++s) {
        const int w = (int) (T/ns) + (s < T % ns ? 1 : 0);
        q8a16_args a{};
        a.n = 1;
        a.vx[0] = (const char *) src0->data;
        a.x[0] = (const float *) src1->data + c0*sx;
        a.dst[0] = (float *) dst->data + c0*sd;
        a.srow[0] = (int64_t) src0->nb[1];
        a.nrows[0] = (int) src0->ne[1];
        a.sx[0] = sx;
        a.sd[0] = sd;
        if (!ggml_cuda_q8tc_try(ctx, a, w, nb, src0, false)) {
            GGML_ASSERT(s == 0 && "q8tc wide: a later slice was refused after the first was taken");
            return false;
        }
        c0 += w;
    }
    return true;
}

#else

bool ggml_cuda_q8tc_try(ggml_backend_cuda_context &, q8a16_args &, int, int, const ggml_tensor *, bool) { return false; }
bool ggml_cuda_q8tc_wide(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *) { return false; }
bool ggml_cuda_q8tc_ia() { return false; }
int ggml_cuda_q8tc_prep(ggml_backend_cuda_context &, q8a16_args &, int, int, const ggml_tensor *, bool) { return 0; }

#endif
