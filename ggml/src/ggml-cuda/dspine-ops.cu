// Fused DSpine draft and lattice CUDA kernels (GGML_CUDA_DSPINE_FUSE={0,1,2}, default 0).
// Modes:
//   0 (default): off
//   1: all 3 kernels (includes lattice_q6k_pack; not bit-identical to cuBLAS SGEMM K-tiling)
//   2: bit-exact kernels only (write_epilogue + refine_norm; matches unfused CUDA bit-for-bit)
// Covers three capture-safe, zero-sync, deterministic sm_86 subgraphs:
//   1. k_dspine_write_epilogue:   SQR -> SUM_ROWS -> SCALE -> SQRT -> MUL -> RESHAPE -> PAD -> RESHAPE -> ADD
//   2. k_dspine_refine_norm:      MUL(delta, rms) -> ADD(., raw) -> RMS_NORM -> MUL(., gamma)
//   3. k_dspine_lattice_q6k_pack: FILL -> CAST -> GET_ROWS(Q6_K) -> MUL_MAT(s) -> CONCAT -> PAD -> CONCAT

#include "dspine-ops.cuh"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

int64_t ggml_cuda_dspine_fuse_count = 0;

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

namespace dspine_ops {

static int dspine_fuse_mode() {
    static const int mode = []() {
        const char * e = getenv("GGML_CUDA_DSPINE_FUSE");
        if (e == nullptr || *e == '\0' || strcmp(e, "0") == 0) {
            return 0;
        }
        if (strcmp(e, "1") == 0) {
            return 1;
        }
        if (strcmp(e, "2") == 0) {
            return 2;
        }
        GGML_LOG_WARN("DSPINE FUSE: invalid GGML_CUDA_DSPINE_FUSE=\"%s\" (expected 0, 1 or 2); using 0\n", e);
        return 0;
    }();
    return mode;
}

static ggml_backend_buffer_t dspine_tensor_buf(const ggml_tensor * t) {
    while (t && t->view_src) {
        t = t->view_src;
    }
    return t ? t->buffer : nullptr;
}

static bool dspine_on_device(const ggml_tensor * t, const int dev) {
    ggml_backend_buffer_t buf = dspine_tensor_buf(t);
    return t != nullptr && t->data != nullptr && buf != nullptr &&
           buf->buft == ggml_backend_cuda_buffer_type(dev);
}

static bool dspine_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    if (!a || !b || !a->data || !b->data) {
        return false;
    }
    const uintptr_t a0 = (uintptr_t) a->data;
    const uintptr_t a1 = a0 + ggml_nbytes(a);
    const uintptr_t b0 = (uintptr_t) b->data;
    const uintptr_t b1 = b0 + ggml_nbytes(b);
    return a0 < b1 && b0 < a1;
}

static bool dspine_aligned16(const ggml_tensor * t) {
    return t != nullptr && t->data != nullptr && (((uintptr_t) t->data) & 15u) == 0u;
}

static bool dspine_shape_eq(const ggml_tensor * t, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    return t != nullptr && t->ne[0] == ne0 && t->ne[1] == ne1 && t->ne[2] == ne2 && t->ne[3] == ne3;
}

static bool dspine_has_external_use(const ggml_cgraph * cgraph, int first, int last, const ggml_tensor * t) {
    if (t->flags & GGML_TENSOR_FLAG_OUTPUT) {
        return true;
    }
    for (int j = 0; j < cgraph->n_nodes; ++j) {
        if (j >= first && j <= last) {
            continue;
        }
        const ggml_tensor * u = cgraph->nodes[j];
        if (u->op == GGML_OP_NONE) {
            continue;
        }
        if (u->view_src == t) {
            return true;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (u->src[s] == t) {
                return true;
            }
        }
    }
    return false;
}

static bool dspine_check_internal_uses(const ggml_cgraph * cgraph, int first, int last,
        const ggml_tensor * const * outputs, int n_outputs) {
    for (int k = first; k <= last; ++k) {
        const ggml_tensor * t = cgraph->nodes[k];
        bool is_out = false;
        for (int o = 0; o < n_outputs; ++o) {
            if (t == outputs[o]) {
                is_out = true;
                break;
            }
        }
        if (is_out) {
            continue;
        }
        if (dspine_has_external_use(cgraph, first, last, t)) {
            return false;
        }
    }
    return true;
}

// ------------------------------------------------------------------------------------------------
// Kernel 1: DSpine bridge write epilogue (9 nodes -> 1 launch)
// Matches unfused SQR + reduce_row_f32 (sum_rows_f32_cuda block size & num_unroll=8) + SCALE + SQRT
// + binbcast MUL + PAD + binbcast ADD bit-for-bit.
// ------------------------------------------------------------------------------------------------
template <int BLOCK_SIZE>
static __global__ __launch_bounds__(BLOCK_SIZE, 1) void k_dspine_write_epilogue(
        const float * __restrict__ cur,
        const float * __restrict__ cur_pos,
        const float * __restrict__ delta,
        float * __restrict__ out,
        float * __restrict__ rms_out,
        const int E,
        const int block,
        const float scale,
        const bool same_cur_pos) {
    const int r     = (int) blockIdx.x;
    const int b     = (int) blockIdx.y;
    const int tid   = (int) threadIdx.x;
    const int n_pos = block - 1;

    const int64_t row_full = (int64_t) b * block + r;
    const float * cur_row  = cur + row_full * E;
    float       * out_row  = out + row_full * E;

    if (r == 0) {
        for (int i = tid; i < E; i += BLOCK_SIZE) {
            out_row[i] = __fadd_rn(cur_row[i], 0.0f);
        }
        return;
    }

    const int64_t row_pos     = (int64_t) b * n_pos + (r - 1);
    const float * cur_pos_row = cur_pos + row_pos * E;
    const float * delta_row   = delta   + row_pos * E;

    constexpr int MAX_REG    = 16; // 16 * 512 = 8192 floats per row cached in registers
    constexpr int num_unroll = 8;
    float reg[MAX_REG];
    float temp[num_unroll];
    float sum_temp[num_unroll] = { 0.0f };

    int k = 0;
    for (int i = tid; i < E;) {
        for (int j = 0; j < num_unroll; ++j) {
            if (i < E) {
                const float x = cur_pos_row[i];
                if (same_cur_pos && k < MAX_REG) {
                    reg[k] = x;
                }
                temp[j] = __fmul_rn(x, x);
            } else {
                temp[j] = 0.0f;
            }
            i += BLOCK_SIZE;
            ++k;
        }
        for (int j = 0; j < num_unroll; ++j) {
            sum_temp[j] += temp[j];
        }
    }
    float sum = 0.0f;
    for (int j = 0; j < num_unroll; ++j) {
        sum += sum_temp[j];
    }

    __shared__ float shared_vals[32];
    sum = block_reduce<block_reduce_method::SUM, BLOCK_SIZE>(sum, shared_vals);

    const float rms = sqrtf(__fmul_rn(sum, scale));
    if (tid == 0 && rms_out != nullptr) {
        rms_out[row_pos] = rms;
    }

    k = 0;
    for (int i = tid; i < E; i += BLOCK_SIZE, ++k) {
        const float c = (same_cur_pos && k < MAX_REG) ? reg[k] : cur_row[i];
        const float d = delta_row[i];
        out_row[i] = __fadd_rn(c, __fmul_rn(d, rms));
    }
}

static int try_fuse_write_epilogue(ggml_backend_cuda_context * ctx, ggml_cgraph * cgraph, int i) {
    if (i + 8 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * n0 = cgraph->nodes[i + 0]; // SQR(cur_pos)
    const ggml_tensor * n1 = cgraph->nodes[i + 1]; // SUM_ROWS(n0)
    const ggml_tensor * n2 = cgraph->nodes[i + 2]; // SCALE(n1, inv_e)
    ggml_tensor       * n3 = cgraph->nodes[i + 3]; // SQRT(n2) = rms_pos
    const ggml_tensor * n4 = cgraph->nodes[i + 4]; // MUL(delta, n3)
    const ggml_tensor * n5 = cgraph->nodes[i + 5]; // RESHAPE(n4, [E, n_pos, B, 1])
    const ggml_tensor * n6 = cgraph->nodes[i + 6]; // PAD(n5, lp1=1)
    const ggml_tensor * n7 = cgraph->nodes[i + 7]; // RESHAPE(n6, [E, block*B, 1, 1])
    ggml_tensor       * n8 = cgraph->nodes[i + 8]; // ADD(cur, n7)

    if (n0->op != GGML_OP_SQR || n1->op != GGML_OP_SUM_ROWS || n2->op != GGML_OP_SCALE ||
        n3->op != GGML_OP_SQRT || n4->op != GGML_OP_MUL || n5->op != GGML_OP_RESHAPE ||
        n6->op != GGML_OP_PAD || n7->op != GGML_OP_RESHAPE || n8->op != GGML_OP_ADD) {
        return 0;
    }
    if (n1->src[0] != n0 || n2->src[0] != n1 || n3->src[0] != n2 ||
        n5->src[0] != n4 || n6->src[0] != n5 || n7->src[0] != n6) {
        return 0;
    }

    const ggml_tensor * cur_pos = n0->src[0];
    const ggml_tensor * delta   = (n4->src[0] == n3) ? n4->src[1] : (n4->src[1] == n3 ? n4->src[0] : nullptr);
    const ggml_tensor * cur     = (n8->src[0] == n7) ? n8->src[1] : (n8->src[1] == n7 ? n8->src[0] : nullptr);
    if (!cur_pos || !delta || !cur) {
        return 0;
    }

    const int64_t E     = n5->ne[0];
    const int64_t n_pos = n5->ne[1];
    const int64_t B     = n5->ne[2];
    const int64_t block = n_pos + 1;
    if (E <= 0 || n_pos <= 0 || B <= 0 || n5->ne[3] != 1 ||
        block > 65535 || B > 65535) {
        return 0;
    }

    if (cur_pos->type != GGML_TYPE_F32 || delta->type != GGML_TYPE_F32 || cur->type != GGML_TYPE_F32 ||
        n3->type != GGML_TYPE_F32 || n8->type != GGML_TYPE_F32) {
        return 0;
    }
    if (!ggml_is_contiguous(cur_pos) || !ggml_is_contiguous(delta) || !ggml_is_contiguous(cur) ||
        !ggml_is_contiguous(n3) || !ggml_is_contiguous(n8)) {
        return 0;
    }
    if (!dspine_shape_eq(cur_pos, E, n_pos * B, 1, 1) ||
        !dspine_shape_eq(delta,   E, n_pos * B, 1, 1) ||
        !dspine_shape_eq(n3,      1, n_pos * B, 1, 1) ||
        !dspine_shape_eq(cur,     E, block * B, 1, 1) ||
        !dspine_shape_eq(n8,      E, block * B, 1, 1) ||
        !dspine_shape_eq(n6,      E, block,     B, 1) ||
        !dspine_shape_eq(n7,      E, block * B, 1, 1)) {
        return 0;
    }

    const float scale = ggml_get_op_params_f32(n2, 0);
    const float bias  = ggml_get_op_params_f32(n2, 1);
    if (bias != 0.0f || scale < 0.0f) {
        return 0;
    }
    if (ggml_get_op_params_i32(n6, 0) != 0 || ggml_get_op_params_i32(n6, 1) != 0 ||
        ggml_get_op_params_i32(n6, 2) != 1 || ggml_get_op_params_i32(n6, 3) != 0 ||
        ggml_get_op_params_i32(n6, 4) != 0 || ggml_get_op_params_i32(n6, 5) != 0 ||
        ggml_get_op_params_i32(n6, 6) != 0 || ggml_get_op_params_i32(n6, 7) != 0 ||
        ggml_get_op_params_i32(n6, 8) != 0) {
        return 0;
    }

    const bool write_rms = dspine_has_external_use(cgraph, i, i + 8, n3);
    const ggml_tensor * outs[2] = { n8, n3 };
    if (!dspine_check_internal_uses(cgraph, i, i + 8, outs, write_rms ? 2 : 1)) {
        return 0;
    }

    if (ctx == nullptr) {
        return 8;
    }

    const int dev = ctx->device;
    if (!dspine_on_device(cur, dev) || !dspine_on_device(cur_pos, dev) ||
        !dspine_on_device(delta, dev) || !dspine_on_device(n8, dev) ||
        (write_rms && !dspine_on_device(n3, dev))) {
        return 0;
    }
    if (dspine_overlap(n8, cur) || dspine_overlap(n8, cur_pos) || dspine_overlap(n8, delta)) {
        return 0;
    }
    if (write_rms && (dspine_overlap(n3, cur) || dspine_overlap(n3, cur_pos) ||
                      dspine_overlap(n3, delta) || dspine_overlap(n3, n8))) {
        return 0;
    }

    const bool same_cur_pos = (B == 1) &&
            ((const char *) cur_pos->data == (const char *) cur->data + (size_t) E * sizeof(float));

    const int     nsm   = ggml_cuda_info().devices[dev].nsm;
    const int     nrows = (int) (n_pos * B);
    const int     bs    = ((nrows / nsm) < 2) ? 512 : (E < 1024 ? 32 : 128);

    static std::atomic_flag logged = ATOMIC_FLAG_INIT;
    if (!logged.test_and_set()) {
        GGML_LOG_INFO("DSPINE FUSE: write_epilogue active (E=%lld, block=%lld, B=%lld, bs=%d, write_rms=%d, same_cur_pos=%d)\n",
                (long long) E, (long long) block, (long long) B, bs, (int) write_rms, (int) same_cur_pos);
    }

    const dim3 grid((unsigned int) block, (unsigned int) B, 1);
    const float * cur_p     = (const float *) cur->data;
    const float * cur_pos_p = (const float *) cur_pos->data;
    const float * delta_p   = (const float *) delta->data;
    float       * out_p     = (float *) n8->data;
    float       * rms_p     = write_rms ? (float *) n3->data : nullptr;
    if (bs == 512) {
        k_dspine_write_epilogue<512><<<grid, 512, 0, ctx->stream()>>>(
                cur_p, cur_pos_p, delta_p, out_p, rms_p, (int) E, (int) block, scale, same_cur_pos);
    } else if (bs == 128) {
        k_dspine_write_epilogue<128><<<grid, 128, 0, ctx->stream()>>>(
                cur_p, cur_pos_p, delta_p, out_p, rms_p, (int) E, (int) block, scale, same_cur_pos);
    } else {
        k_dspine_write_epilogue<32><<<grid, 32, 0, ctx->stream()>>>(
                cur_p, cur_pos_p, delta_p, out_p, rms_p, (int) E, (int) block, scale, same_cur_pos);
    }

    ggml_cuda_dspine_fuse_count++;
    return 8;
}

// ------------------------------------------------------------------------------------------------
// Kernel 2: DSpine lattice refine + RMS norm (4 nodes -> 1 launch)
// Matches unfused binbcast MUL + ADD + default scalar rms_norm_mul_f32 (256/1024 threads) bit-for-bit.
// ------------------------------------------------------------------------------------------------
template <int BLOCK_SIZE>
static __global__ __launch_bounds__(BLOCK_SIZE, 1) void k_dspine_refine_norm(
        const float * __restrict__ delta,
        const float * __restrict__ rms,
        const float * __restrict__ raw,
        const float * __restrict__ gamma,
        float * __restrict__ out,
        const int E,
        const int C,
        const int P,
        const float eps) {
    const int c   = (int) blockIdx.x;
    const int p   = (int) blockIdx.y;
    const int b   = (int) blockIdx.z;
    const int tid = (int) threadIdx.x;

    const int64_t pb  = (int64_t) b * P + p;
    const int64_t row = pb * C + c;
    const float   r   = rms[pb];

    const float * delta_row = delta + row * E;
    const float * raw_row   = raw   + pb  * E;
    float       * out_row   = out   + row * E;

    constexpr int MAX_REG = 8; // 8 * 1024 = 8192 (or 8 * 256 = 2048 >= 1024) floats per row cached in registers
    float reg[MAX_REG];
    float tmp = 0.0f;

    int k = 0;
    for (int col = tid; col < E; col += BLOCK_SIZE, ++k) {
        const float xi = __fadd_rn(__fmul_rn(delta_row[col], r), raw_row[col]);
        if (k < MAX_REG) {
            reg[k] = xi;
        }
        tmp += xi * xi;
    }

    __shared__ float s_sum[32];
    tmp = block_reduce<block_reduce_method::SUM, BLOCK_SIZE>(tmp, s_sum);

    const float mean  = tmp / E;
    const float scale = rsqrtf(mean + eps);

    k = 0;
    for (int col = tid; col < E; col += BLOCK_SIZE, ++k) {
        const float xi = (k < MAX_REG) ? reg[k] : __fadd_rn(__fmul_rn(delta_row[col], r), raw_row[col]);
        out_row[col] = scale * xi * gamma[col];
    }
}

static int try_fuse_refine_norm(ggml_backend_cuda_context * ctx, ggml_cgraph * cgraph, int i) {
    if (i + 3 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * n0 = cgraph->nodes[i + 0]; // MUL(delta, rms)
    const ggml_tensor * n1 = cgraph->nodes[i + 1]; // ADD(n0, raw)
    const ggml_tensor * n2 = cgraph->nodes[i + 2]; // RMS_NORM(n1)
    ggml_tensor       * n3 = cgraph->nodes[i + 3]; // MUL(n2, gamma)

    if (n0->op != GGML_OP_MUL || n1->op != GGML_OP_ADD ||
        n2->op != GGML_OP_RMS_NORM || n3->op != GGML_OP_MUL) {
        return 0;
    }
    if (n2->src[0] != n1) {
        return 0;
    }

    const int64_t E = n3->ne[0];
    const int64_t C = n3->ne[1];
    const int64_t P = n3->ne[2];
    const int64_t B = n3->ne[3];
    if (E <= 0 || C <= 0 || P <= 0 || B <= 0 || P > 65535 || B > 65535) {
        return 0;
    }

    const ggml_tensor * s0 = n0->src[0];
    const ggml_tensor * s1 = n0->src[1];
    if (!s0 || !s1) {
        return 0;
    }
    const ggml_tensor * delta = (s0->ne[0] == E) ? s0 : (s1->ne[0] == E ? s1 : nullptr);
    const ggml_tensor * rms   = (delta == s0) ? s1 : s0;
    const ggml_tensor * raw   = (n1->src[0] == n0) ? n1->src[1] : (n1->src[1] == n0 ? n1->src[0] : nullptr);
    const ggml_tensor * gamma = (n3->src[0] == n2) ? n3->src[1] : (n3->src[1] == n2 ? n3->src[0] : nullptr);
    if (!delta || !rms || !raw || !gamma) {
        return 0;
    }

    if (delta->type != GGML_TYPE_F32 || rms->type != GGML_TYPE_F32 ||
        raw->type != GGML_TYPE_F32 || gamma->type != GGML_TYPE_F32 ||
        n0->type != GGML_TYPE_F32 || n1->type != GGML_TYPE_F32 ||
        n2->type != GGML_TYPE_F32 || n3->type != GGML_TYPE_F32) {
        return 0;
    }
    if (!ggml_is_contiguous(delta) || !ggml_is_contiguous(rms) ||
        !ggml_is_contiguous(raw) || !ggml_is_contiguous(gamma) || !ggml_is_contiguous(n3)) {
        return 0;
    }
    if (!dspine_shape_eq(delta, E, C, P, B) ||
        !dspine_shape_eq(rms,   1, 1, P, B) ||
        !dspine_shape_eq(raw,   E, 1, P, B) ||
        !dspine_shape_eq(gamma, E, 1, 1, 1) ||
        !dspine_shape_eq(n0,    E, C, P, B) ||
        !dspine_shape_eq(n1,    E, C, P, B) ||
        !dspine_shape_eq(n2,    E, C, P, B)) {
        return 0;
    }

    const ggml_tensor * outs[1] = { n3 };
    if (!dspine_check_internal_uses(cgraph, i, i + 3, outs, 1)) {
        return 0;
    }

    if (ctx == nullptr) {
        return 3;
    }

    const int dev = ctx->device;
    if (!dspine_on_device(delta, dev) || !dspine_on_device(rms, dev) ||
        !dspine_on_device(raw, dev) || !dspine_on_device(gamma, dev) || !dspine_on_device(n3, dev)) {
        return 0;
    }
    if (dspine_overlap(n3, delta) || dspine_overlap(n3, rms) ||
        dspine_overlap(n3, raw) || dspine_overlap(n3, gamma)) {
        return 0;
    }

    const float eps = ggml_get_op_params_f32(n2, 0);
    const int   bs  = (E < 1024) ? 256 : 1024;

    static std::atomic_flag logged = ATOMIC_FLAG_INIT;
    if (!logged.test_and_set()) {
        GGML_LOG_INFO("DSPINE FUSE: refine_norm active (E=%lld, C=%lld, P=%lld, B=%lld, bs=%d)\n",
                (long long) E, (long long) C, (long long) P, (long long) B, bs);
    }

    const dim3 grid((unsigned int) C, (unsigned int) P, (unsigned int) B);
    if (bs == 256) {
        k_dspine_refine_norm<256><<<grid, 256, 0, ctx->stream()>>>(
                (const float *) delta->data,
                (const float *) rms->data,
                (const float *) raw->data,
                (const float *) gamma->data,
                (float *) n3->data,
                (int) E, (int) C, (int) P, eps);
    } else {
        k_dspine_refine_norm<1024><<<grid, 1024, 0, ctx->stream()>>>(
                (const float *) delta->data,
                (const float *) rms->data,
                (const float *) raw->data,
                (const float *) gamma->data,
                (float *) n3->data,
                (int) E, (int) C, (int) P, eps);
    }

    ggml_cuda_dspine_fuse_count++;
    return 3;
}

// ------------------------------------------------------------------------------------------------
// Kernel 3: DSpine Q6_K lattice scoring + packing (10-22 nodes -> 1 launch, opt-in mode 1 only)
// Streams each candidate's Q6_K output row once directly into registers (no F32 dequant buffer),
// dots against the 1 (dedup pos-0) or K refined vectors, and writes [cand_id, scores, zero-pad]
// plus row 0 zeros directly into the packed [E, block, B] lattice tensor.
// Not bit-identical to unfused GET_ROWS(Q6_K) + cublasGemmStridedBatchedEx (proprietary K-tiling
// order), so GGML_CUDA_DSPINE_FUSE=2 skips this kernel for strict draft-exact runs.
// ------------------------------------------------------------------------------------------------
template <int MAX_K>
static __global__ void k_dspine_lattice_q6k_pack(
        const block_q6_K * __restrict__ output,
        const int32_t    * __restrict__ cand_ids,
        const float      * __restrict__ ref0,
        const float      * __restrict__ refr,
        float            * __restrict__ dst,
        const int E,
        const int64_t V,
        const int K,
        const int n_pos,
        const bool dedup) {
    const int k     = (int) blockIdx.x;
    const int p     = (int) blockIdx.y;
    const int b     = (int) blockIdx.z;
    const int tid   = (int) threadIdx.x; // 256 threads = 4 groups x 64 threads
    const int block = n_pos + 1;
    const int nb    = E / QK_K;

    // CTA (0, 0, b) zeroes the anchor row 0 of block b
    if (k == 0 && p == 0) {
        float4 * row0_4 = (float4 *) (dst + ((int64_t) b * block) * E);
        const int E4 = E >> 2;
        const float4 z = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int i = tid; i < E4; i += 256) {
            row0_4[i] = z;
        }
    }

    float * row_out = dst + ((int64_t) b * block + 1 + p) * E;

    // CTA (0, p, b) zeroes the trailing pad [K + K*K, E) of row (1 + p)
    if (k == 0) {
        const int pad_beg = K + K * K;
        for (int c = pad_beg + tid; c < E; c += 256) {
            row_out[c] = 0.0f;
        }
    }

    const int32_t tok = cand_ids[((int64_t) b * n_pos + p) * K + k];
    if (tid == 0) {
        row_out[k] = (float) tok;
    }

    const bool single_ref = dedup && (p == 0);
    const int  n_acc      = single_ref ? 1 : K;

    const float * ref_base = nullptr;
    if (single_ref) {
        ref_base = ref0 + (int64_t) b * E;
    } else if (dedup) {
        ref_base = refr + (((int64_t) b * (n_pos - 1) + (p - 1)) * K) * E;
    } else {
        ref_base = refr + (((int64_t) b * n_pos + p) * K) * E;
    }

    float acc[MAX_K];
#pragma unroll
    for (int j = 0; j < MAX_K; ++j) {
        acc[j] = 0.0f;
    }

    if (tok >= 0 && (int64_t) tok < V) {
        const block_q6_K * w_row = output + (int64_t) tok * nb;

        const int group = tid >> 6;          // 0..3
        const int tid64 = tid & 63;          // 0..63
        const int ip    = tid64 >> 5;        // 0 or 1
        const int il    = tid64 & 31;        // 0..31
        const int is    = 8 * ip + (il >> 4);
        const int base  = 128 * ip + il;

        for (int ib = group; ib < nb; ib += 4) {
            const block_q6_K & blk = w_row[ib];
            const float d = blk.d;
            const uint8_t * ql = blk.ql + 64 * ip + il;
            const uint8_t   qh = blk.qh[32 * ip + il];
            const int8_t  * sc = blk.scales + is;

            const float w0 = d * (float) sc[0] * (float) ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
            const float w1 = d * (float) sc[2] * (float) ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
            const float w2 = d * (float) sc[4] * (float) ((int8_t)((ql[ 0] >> 4)  | (((qh >> 4) & 3) << 4)) - 32);
            const float w3 = d * (float) sc[6] * (float) ((int8_t)((ql[32] >> 4)  | (((qh >> 6) & 3) << 4)) - 32);

            const int col = ib * QK_K + base;
            if (single_ref) {
                const float * x = ref_base + col;
                acc[0] += w0 * x[0] + w1 * x[32] + w2 * x[64] + w3 * x[96];
            } else {
                for (int j = 0; j < K; ++j) {
                    const float * x = ref_base + (int64_t) j * E + col;
                    acc[j] += w0 * x[0] + w1 * x[32] + w2 * x[64] + w3 * x[96];
                }
            }
        }
    }

    constexpr int NWARPS = 256 / WARP_SIZE;
    __shared__ float s_red[NWARPS][MAX_K];
    const int lane = tid & (WARP_SIZE - 1);
    const int warp = tid / WARP_SIZE;

    for (int j = 0; j < n_acc; ++j) {
        const float v = warp_reduce_sum(acc[j]);
        if (lane == 0) {
            s_red[warp][j] = v;
        }
    }
    __syncthreads();

    if (single_ref) {
        if (tid == 0) {
            float total = 0.0f;
            for (int w = 0; w < NWARPS; ++w) {
                total += s_red[w][0];
            }
            s_red[0][0] = total;
        }
        __syncthreads();
        if (tid < K) {
            row_out[K + tid * K + k] = s_red[0][0];
        }
    } else if (tid < K) {
        float total = 0.0f;
        for (int w = 0; w < NWARPS; ++w) {
            total += s_red[w][tid];
        }
        row_out[K + tid * K + k] = total;
    }
}

static bool match_head_slice_b1(const ggml_cgraph * cgraph, int & pos,
        const ggml_tensor * head_base, const ggml_tensor * head3,
        size_t expected_offs, int64_t expected_len, int64_t E, int64_t K,
        const ggml_tensor ** head_out) {
    if (pos + 3 >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * r2 = cgraph->nodes[pos + 0];
    const ggml_tensor * v2 = cgraph->nodes[pos + 1];
    const ggml_tensor * r3 = cgraph->nodes[pos + 2];
    const ggml_tensor * r4 = cgraph->nodes[pos + 3];
    if (r2->op != GGML_OP_RESHAPE || r2->src[0] != head3 ||
        v2->op != GGML_OP_VIEW    || v2->src[0] != r2 ||
        r3->op != GGML_OP_RESHAPE || r3->src[0] != v2 ||
        r4->op != GGML_OP_RESHAPE || r4->src[0] != r3) {
        return false;
    }
    if (r4->view_src != head_base || r4->view_offs != expected_offs ||
        !ggml_is_contiguous(r4) || !dspine_shape_eq(r4, E, K, expected_len, 1)) {
        return false;
    }
    *head_out = r4;
    pos += 4;
    return true;
}

static bool match_head_slice_bn(const ggml_cgraph * cgraph, int & pos,
        const ggml_tensor * head_base, const ggml_tensor * head3,
        size_t expected_offs, int64_t expected_len, int64_t E, int64_t K, int64_t n_pos, int64_t B,
        const ggml_tensor ** head_out) {
    if (pos + 2 >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * v3 = cgraph->nodes[pos + 0];
    const ggml_tensor * c3 = cgraph->nodes[pos + 1];
    const ggml_tensor * r4 = cgraph->nodes[pos + 2];
    if (v3->op != GGML_OP_VIEW || v3->src[0] != head3 ||
        c3->op != GGML_OP_CONT || c3->src[0] != v3 ||
        r4->op != GGML_OP_RESHAPE || r4->src[0] != c3) {
        return false;
    }
    const size_t stride1 = (size_t) E * (size_t) K * sizeof(float);
    const size_t stride2 = stride1 * (size_t) n_pos;
    if (v3->view_src != head_base || v3->view_offs != expected_offs ||
        !dspine_shape_eq(v3, E * K, expected_len, B, 1) ||
        v3->nb[0] != sizeof(float) || v3->nb[1] != stride1 || v3->nb[2] != stride2 ||
        !ggml_is_contiguous(r4) || !dspine_shape_eq(r4, E, K, expected_len, B)) {
        return false;
    }
    *head_out = r4;
    pos += 3;
    return true;
}

static int try_fuse_lattice_q6k_pack(ggml_backend_cuda_context * ctx, ggml_cgraph * cgraph, int i) {
    if (i + 9 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * n0 = cgraph->nodes[i + 0]; // FILL(0.0f) [E, 1, B, 1]
    const ggml_tensor * n1 = cgraph->nodes[i + 1]; // CPY (ggml_cast cand_run -> F32)
    const ggml_tensor * n2 = cgraph->nodes[i + 2]; // RESHAPE(cand_run -> 1D)
    const ggml_tensor * n3 = cgraph->nodes[i + 3]; // GET_ROWS(output, n2)
    const ggml_tensor * n4 = cgraph->nodes[i + 4]; // RESHAPE(n3 -> [E, K, n_pos, B])

    if (n0->op != GGML_OP_FILL || n1->op != GGML_OP_CPY || n2->op != GGML_OP_RESHAPE ||
        n3->op != GGML_OP_GET_ROWS || n4->op != GGML_OP_RESHAPE) {
        return 0;
    }
    if (n0->type != GGML_TYPE_F32 || !ggml_is_contiguous(n0) || n0->ne[1] != 1 || n0->ne[3] != 1) {
        return 0;
    }
    if (ggml_get_op_params_f32(n0, 0) != 0.0f) {
        return 0;
    }

    const ggml_tensor * cand_run = n1->src[0];
    const ggml_tensor * output   = n3->src[0];
    if (!cand_run || !output || n1->src[1] != n1 || n1->type != GGML_TYPE_F32 ||
        n2->src[0] != cand_run || n3->src[1] != n2 || n4->src[0] != n3) {
        return 0;
    }
    if (cand_run->type != GGML_TYPE_I32 || !ggml_is_contiguous(cand_run) || cand_run->ne[3] != 1) {
        return 0;
    }

    const int64_t E     = n0->ne[0];
    const int64_t K     = cand_run->ne[0];
    const int64_t n_pos = cand_run->ne[1];
    const int64_t B     = cand_run->ne[2];
    if (E <= 0 || E % QK_K != 0 || K < 1 || K > 32 || K + K * K > E ||
        n_pos < 1 || n_pos > 65535 || B < 1 || B > 65535 || n0->ne[2] != B) {
        return 0;
    }
    if (output->type != GGML_TYPE_Q6_K || !ggml_is_contiguous(output) ||
        output->ne[0] != E || output->ne[1] <= 0 || output->ne[2] != 1 || output->ne[3] != 1) {
        return 0;
    }
    if (!dspine_shape_eq(n2, K * n_pos * B, 1, 1, 1) ||
        !dspine_shape_eq(n4, E, K, n_pos, B)) {
        return 0;
    }

    const ggml_tensor * ref0  = nullptr;
    const ggml_tensor * refr  = nullptr;
    const ggml_tensor * table = nullptr;
    bool dedup = false;

    int pos = i + 5;
    if (pos >= cgraph->n_nodes) {
        return 0;
    }

    if (cgraph->nodes[pos]->op == GGML_OP_MUL_MAT && cgraph->nodes[pos]->src[0] == n4) {
        const ggml_tensor * mm = cgraph->nodes[pos];
        if (ggml_get_op_params_i32(mm, 1) != GGML_HINT_NONE || mm->src[1] == nullptr) {
            return 0;
        }
        if (n_pos == 1 && pos + 1 < cgraph->n_nodes &&
            cgraph->nodes[pos + 1]->op == GGML_OP_REPEAT &&
            cgraph->nodes[pos + 1]->src[0] == mm) {
            const ggml_tensor * rep = cgraph->nodes[pos + 1];
            ref0 = mm->src[1];
            if (!dspine_shape_eq(ref0, E, 1, 1, B) || !dspine_shape_eq(rep, K, K, 1, B)) {
                return 0;
            }
            table = rep;
            dedup = true;
            pos += 2;
        } else {
            refr = mm->src[1];
            if (!dspine_shape_eq(refr, E, K, n_pos, B) || !dspine_shape_eq(mm, K, K, n_pos, B)) {
                return 0;
            }
            table = mm;
            dedup = false;
            pos += 1;
        }
    } else if (n_pos > 1 && cgraph->nodes[pos]->op == GGML_OP_RESHAPE && cgraph->nodes[pos]->src[0] == n4) {
        const ggml_tensor * head3 = cgraph->nodes[pos++];
        if (head3->view_src != n3 || head3->view_offs != 0 || !dspine_shape_eq(head3, E * K, n_pos, B, 1)) {
            return 0;
        }

        const ggml_tensor * head0 = nullptr;
        const bool ok0 = (B == 1)
                ? match_head_slice_b1(cgraph, pos, n3, head3, 0, 1, E, K, &head0)
                : match_head_slice_bn(cgraph, pos, n3, head3, 0, 1, E, K, n_pos, B, &head0);
        if (!ok0 || pos + 1 >= cgraph->n_nodes) {
            return 0;
        }

        const ggml_tensor * mm0 = cgraph->nodes[pos++];
        const ggml_tensor * rep = cgraph->nodes[pos++];
        if (mm0->op != GGML_OP_MUL_MAT || mm0->src[0] != head0 ||
            ggml_get_op_params_i32(mm0, 1) != GGML_HINT_NONE ||
            rep->op != GGML_OP_REPEAT || rep->src[0] != mm0) {
            return 0;
        }
        ref0 = mm0->src[1];
        if (!dspine_shape_eq(ref0, E, 1, 1, B) || !dspine_shape_eq(rep, K, K, 1, B)) {
            return 0;
        }

        const size_t rest_offs = (size_t) E * (size_t) K * sizeof(float);
        const ggml_tensor * headr = nullptr;
        const bool okr = (B == 1)
                ? match_head_slice_b1(cgraph, pos, n3, head3, rest_offs, n_pos - 1, E, K, &headr)
                : match_head_slice_bn(cgraph, pos, n3, head3, rest_offs, n_pos - 1, E, K, n_pos, B, &headr);
        if (!okr || pos + 1 >= cgraph->n_nodes) {
            return 0;
        }

        const ggml_tensor * mmr   = cgraph->nodes[pos++];
        const ggml_tensor * cat_t = cgraph->nodes[pos++];
        if (mmr->op != GGML_OP_MUL_MAT || mmr->src[0] != headr ||
            ggml_get_op_params_i32(mmr, 1) != GGML_HINT_NONE ||
            cat_t->op != GGML_OP_CONCAT || ggml_get_op_params_i32(cat_t, 0) != 2 ||
            cat_t->src[0] != rep || cat_t->src[1] != mmr) {
            return 0;
        }
        refr = mmr->src[1];
        if (!dspine_shape_eq(refr, E, K, n_pos - 1, B) || !dspine_shape_eq(cat_t, K, K, n_pos, B)) {
            return 0;
        }
        table = cat_t;
        dedup = true;
    } else {
        return 0;
    }

    if (pos + 3 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * res_t = cgraph->nodes[pos + 0];
    const ggml_tensor * cat_r = cgraph->nodes[pos + 1];
    const ggml_tensor * pad_r = cgraph->nodes[pos + 2];
    ggml_tensor       * out   = cgraph->nodes[pos + 3];
    const int last = pos + 3;

    if (res_t->op != GGML_OP_RESHAPE || res_t->src[0] != table ||
        cat_r->op != GGML_OP_CONCAT  || ggml_get_op_params_i32(cat_r, 0) != 0 ||
        cat_r->src[0] != n1          || cat_r->src[1] != res_t ||
        pad_r->op != GGML_OP_PAD     || pad_r->src[0] != cat_r ||
        out->op   != GGML_OP_CONCAT  || ggml_get_op_params_i32(out, 0) != 1 ||
        out->src[0] != n0            || out->src[1] != pad_r) {
        return 0;
    }
    if (!dspine_shape_eq(res_t, K * K,     n_pos,     B, 1) ||
        !dspine_shape_eq(cat_r, K + K * K, n_pos,     B, 1) ||
        !dspine_shape_eq(pad_r, E,         n_pos,     B, 1) ||
        !dspine_shape_eq(out,   E,         n_pos + 1, B, 1)) {
        return 0;
    }
    if (ggml_get_op_params_i32(pad_r, 0) != 0 ||
        ggml_get_op_params_i32(pad_r, 1) != (int32_t) (E - K - K * K) ||
        ggml_get_op_params_i32(pad_r, 2) != 0 || ggml_get_op_params_i32(pad_r, 3) != 0 ||
        ggml_get_op_params_i32(pad_r, 4) != 0 || ggml_get_op_params_i32(pad_r, 5) != 0 ||
        ggml_get_op_params_i32(pad_r, 6) != 0 || ggml_get_op_params_i32(pad_r, 7) != 0 ||
        ggml_get_op_params_i32(pad_r, 8) != 0) {
        return 0;
    }
    if (out->type != GGML_TYPE_F32 || !ggml_is_contiguous(out)) {
        return 0;
    }
    if (ref0 && (ref0->type != GGML_TYPE_F32 || !ggml_is_contiguous(ref0))) {
        return 0;
    }
    if (refr && (refr->type != GGML_TYPE_F32 || !ggml_is_contiguous(refr))) {
        return 0;
    }

    const ggml_tensor * outs[1] = { out };
    if (!dspine_check_internal_uses(cgraph, i, last, outs, 1)) {
        return 0;
    }

    if (ctx == nullptr) {
        return last - i;
    }

    const int dev = ctx->device;
    if (!dspine_on_device(output, dev) || !dspine_on_device(cand_run, dev) ||
        !dspine_on_device(out, dev) ||
        (ref0 && !dspine_on_device(ref0, dev)) ||
        (refr && !dspine_on_device(refr, dev))) {
        return 0;
    }
    if (!dspine_aligned16(out)) {
        return 0;
    }
    if (dspine_overlap(out, output) || dspine_overlap(out, cand_run) ||
        (ref0 && dspine_overlap(out, ref0)) ||
        (refr && dspine_overlap(out, refr))) {
        return 0;
    }

    static std::atomic_flag logged = ATOMIC_FLAG_INIT;
    if (!logged.test_and_set()) {
        GGML_LOG_INFO("DSPINE FUSE: lattice_q6k_pack active (E=%lld, V=%lld, K=%lld, n_pos=%lld, B=%lld, dedup=%d)\n",
                (long long) E, (long long) output->ne[1], (long long) K, (long long) n_pos, (long long) B, (int) dedup);
    }

    const dim3 grid((unsigned int) K, (unsigned int) n_pos, (unsigned int) B);
    if (K <= 16) {
        k_dspine_lattice_q6k_pack<16><<<grid, 256, 0, ctx->stream()>>>(
                (const block_q6_K *) output->data,
                (const int32_t *) cand_run->data,
                ref0 ? (const float *) ref0->data : nullptr,
                refr ? (const float *) refr->data : nullptr,
                (float *) out->data,
                (int) E, output->ne[1], (int) K, (int) n_pos, dedup);
    } else {
        k_dspine_lattice_q6k_pack<32><<<grid, 256, 0, ctx->stream()>>>(
                (const block_q6_K *) output->data,
                (const int32_t *) cand_run->data,
                ref0 ? (const float *) ref0->data : nullptr,
                refr ? (const float *) refr->data : nullptr,
                (float *) out->data,
                (int) E, output->ne[1], (int) K, (int) n_pos, dedup);
    }

    ggml_cuda_dspine_fuse_count++;
    return last - i;
}

} // namespace dspine_ops

int ggml_cuda_try_fuse_dspine(ggml_backend_cuda_context * ctx, ggml_cgraph * cgraph, int i) {
    using namespace dspine_ops;
    const int mode = dspine_fuse_mode();
    if (mode == 0 || i < 0 || i >= cgraph->n_nodes) {
        return 0;
    }
    if (ctx != nullptr && !ctx->stream_context().concurrent_events.empty()) {
        return 0;
    }
    const ggml_op op = cgraph->nodes[i]->op;
    if (op == GGML_OP_SQR) {
        return try_fuse_write_epilogue(ctx, cgraph, i);
    }
    if (op == GGML_OP_MUL) {
        return try_fuse_refine_norm(ctx, cgraph, i);
    }
    if (op == GGML_OP_FILL && mode == 1) {
        return try_fuse_lattice_q6k_pack(ctx, cgraph, i);
    }
    return 0;
}

#else

int ggml_cuda_try_fuse_dspine(ggml_backend_cuda_context * ctx, ggml_cgraph * cgraph, int i) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(cgraph);
    GGML_UNUSED(i);
    return 0;
}

#endif
