// Opt-in SIMD path for the AI+ unit MUL_MAT / MUL_MAT_ID ops of the CPU backend (GGML_CPU_AI4_AVX2=1, default off).
// The reference path dequantizes each 16-row slab to f32 in the original basis (with the inverse rotation) and runs f32
// dot products. Here each activation row is rotated once into the weights' basis instead (moe_ai::rotate), and the slab
// rows are dotted against it directly: the 1MAD types (C != 0) through the host expert-split kernels (moe_ai::slab_rows,
// AVX2 row-window / AVX-512 decode), AIT2P_U (C == 0) through the AVX2 pair-table kernel (ggml-ai4-avx2.h), which the
// host kernels run scalar. Same math, different float summation order.
#include "ai4-avx2.h"
#include "ggml-cpu-impl.h"

#include "../ggml-cuda/moe-ai-rows.h"
#include "../ggml-ai4-avx2.h"

#include <cstdlib>
#include <cstring>

static bool ai4_avx2_flag() {
    // read per op (not cached) so a process can A/B both paths; the work size check below keeps plan and compute agreeing
    const char * e = getenv("GGML_CPU_AI4_AVX2");
    return e && strcmp(e, "1") == 0;
}

static bool ai4_avx2_covered(const ggml_tensor * dst) {
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    if (!w || !x || dst->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32) {
        return false;
    }
    const int k = ggml_ai4_rate(w->type);
    const int64_t K = w->ne[0];
    if (k < 2 || k > 5 || K <= 0 || K % AI4U_UK != 0 || w->ne[1] % AI4_TR != 0 || x->ne[0] != K ||
            w->nb[0] != ggml_type_size(w->type) || w->nb[1] != ggml_row_size(w->type, K) ||
            x->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) || dst->ne[0] != w->ne[1]) {
        return false;
    }
    // the host kernels tell U from U2 (one row scale per 640 K) by the row size; the 512-K row-scale U2 types are not covered
    if (ggml_ai4v2_chunks(w->type) == 4 ||
            (ggml_ai4v2_rate(w->type) != 0) != moe_ai::slab_is_v2(w->nb[1], k, K)) {
        return false;
    }
    if (dst->op == GGML_OP_MUL_MAT) {
        return w->ne[2] == 1 && w->ne[3] == 1;
    }
    const ggml_tensor * ids = dst->src[2];
    return dst->op == GGML_OP_MUL_MAT_ID && ids && ids->type == GGML_TYPE_I32 && w->ne[3] == 1 && x->ne[3] == 1 &&
           x->ne[2] == ids->ne[1] && (x->ne[1] == 1 || x->ne[1] == ids->ne[0]);
}

bool ggml_cpu_ai4_avx2_use(const ggml_tensor * dst) {
    return ai4_avx2_flag() && ai4_avx2_covered(dst);
}

// the reference path's per-thread [16][K] scratch (kept, so a short plan still falls back cleanly), then the rotated
// activations, one K row per src1 row
size_t ggml_cpu_ai4_avx2_work_size(const ggml_tensor * dst, int nth) {
    const int64_t K = dst->src[0]->ne[0];
    return ((size_t) nth*16 + (size_t) ggml_nrows(dst->src[1]))*(size_t) K*sizeof(float);
}

static float * ai4_avx2_rotate_all(const ggml_compute_params * params, const ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[1];
    const int64_t K = x->ne[0];
    float * rot = (float *) params->wdata + (size_t) params->nth*16*K;
    const int64_t n = ggml_nrows(x);
    for (int64_t i = params->ith; i < n; i += params->nth) {
        const int64_t i1 = i % x->ne[1], i2 = (i / x->ne[1]) % x->ne[2], i3 = i / (x->ne[1]*x->ne[2]);
        moe_ai::rotate((const float *) ((const char *) x->data + i1*x->nb[1] + i2*x->nb[2] + i3*x->nb[3]), rot + i*K, K);
    }
    ggml_barrier(params->threadpool);
    return rot;
}

// rows [16 g, 16 g + 16) of the expert at w against nt <= 8 rotated rows: o[t][0..15]
static void ai4_avx2_slab(const ggml_tensor * w0, const char * w, int64_t g, int nt, const float * const * xs, float * const * os) {
    const int64_t K = w0->ne[0];
    const uint32_t C = ggml_ai4_trellis_mul(w0->type);
#if GGML_AI4_PAIR_AVX2
    if (C == 0) {
        ai4_pair_avx2::slab_dot((const uint8_t *) w + (size_t) g*AI4_TR*w0->nb[1], K, nt, xs, os);
        return;
    }
#endif
    moe_ai::slab_rows(w, w0->nb[1], ggml_ai4_rate(w0->type), C, g*AI4_TR, (g + 1)*AI4_TR, K, nt, xs, os);
}

static bool ai4_avx2_ready(const ggml_compute_params * params, const ggml_tensor * dst) {
    // all threads see the same flag, tensors and wsize, so they all take the same path
    return !params->use_ref && ggml_cpu_ai4_avx2_use(dst) && params->wsize >= ggml_cpu_ai4_avx2_work_size(dst, params->nth);
}

bool ggml_cpu_ai4_avx2_mul_mat(const ggml_compute_params * params, ggml_tensor * dst) {
    if (!ai4_avx2_ready(params, dst)) {
        return false;
    }
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    const int64_t K = w->ne[0], ng = w->ne[1]/AI4_TR, n = ggml_nrows(x);
    const float * rot = ai4_avx2_rotate_all(params, dst);
    for (int64_t g = params->ith; g < ng; g += params->nth) {
        for (int64_t i0 = 0; i0 < n; i0 += 8) {
            const int nt = (int) (n - i0 < 8 ? n - i0 : 8);
            const float * xs[8];
            float * os[8];
            for (int t = 0; t < nt; ++t) {
                const int64_t i = i0 + t;
                const int64_t i1 = i % x->ne[1], i2 = (i / x->ne[1]) % x->ne[2], i3 = i / (x->ne[1]*x->ne[2]);
                xs[t] = rot + i*K;
                os[t] = (float *) ((char *) dst->data + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3]) + g*AI4_TR;
            }
            ai4_avx2_slab(w, (const char *) w->data, g, nt, xs, os);
        }
    }
    return true;
}

// src0 [K, M, n_expert], src1 [K, ne11 (1 or n_used), n_tokens], ids [n_used, n_tokens], dst [M, n_used, n_tokens], as the
// reference: pairs with an id outside [0, n_expert) get zero rows; each used expert's slabs serve every pair routed to it
bool ggml_cpu_ai4_avx2_mul_mat_id(const ggml_compute_params * params, ggml_tensor * dst) {
    if (!ai4_avx2_ready(params, dst)) {
        return false;
    }
    const ggml_tensor * w   = dst->src[0];
    const ggml_tensor * x   = dst->src[1];
    const ggml_tensor * ids = dst->src[2];
    const int64_t K = w->ne[0], ng = w->ne[1]/AI4_TR, n_exp = w->ne[2];
    const int64_t n_ids = ids->ne[0], n_tok = ids->ne[1];
    auto id_of = [&](int64_t s, int64_t t) {
        return *(const int32_t *) ((const char *) ids->data + s*ids->nb[0] + t*ids->nb[1]);
    };
    for (int64_t t = 0; t < n_tok; ++t) {
        for (int64_t s = 0; s < n_ids; ++s) {
            const int32_t e = id_of(s, t);
            if (e >= 0 && e < n_exp) {
                continue;
            }
            float * d = (float *) ((char *) dst->data + s*dst->nb[1] + t*dst->nb[2]);
            for (int64_t g = params->ith; g < ng; g += params->nth) {
                memset(d + g*AI4_TR, 0, AI4_TR*sizeof(float));
            }
        }
    }
    const float * rot = ai4_avx2_rotate_all(params, dst);
    for (int64_t e = 0; e < n_exp; ++e) {
        bool used = false;
        for (int64_t t = 0; t < n_tok && !used; ++t) {
            for (int64_t s = 0; s < n_ids && !used; ++s) {
                used = id_of(s, t) == e;
            }
        }
        if (!used) {
            continue;
        }
        const char * we = (const char *) w->data + e*w->nb[2];
        for (int64_t g = params->ith; g < ng; g += params->nth) {
            const float * xs[8];
            float * os[8];
            int nt = 0;
            for (int64_t t = 0; t < n_tok; ++t) {
                for (int64_t s = 0; s < n_ids; ++s) {
                    if (id_of(s, t) != e) {
                        continue;
                    }
                    xs[nt] = rot + ((s % x->ne[1]) + t*x->ne[1])*K;
                    os[nt] = (float *) ((char *) dst->data + s*dst->nb[1] + t*dst->nb[2]) + g*AI4_TR;
                    if (++nt == 8) {
                        ai4_avx2_slab(w, we, g, nt, xs, os);
                        nt = 0;
                    }
                }
            }
            if (nt > 0) {
                ai4_avx2_slab(w, we, g, nt, xs, os);
            }
        }
    }
    return true;
}
