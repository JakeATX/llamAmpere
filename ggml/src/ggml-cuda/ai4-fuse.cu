// AmperInteger (AI+) drop-in activation & dense-FFN fusion kernels — dedicated CUDA TU.
// Keeps kernel instantiations isolated from ggml-cuda.cu and ai4.cu.
#include "ai4-fuse.cuh"

namespace ai4fuse {

// -------------------------------------------------------------------------------------------------
// 1. launch_rms_norm_mul_quantize_act_ai4
// -------------------------------------------------------------------------------------------------

template <int XSH, int VARIANT, bool WRITE_F32>
static void dispatch_rms_norm_q(
        const float * x, int64_t sx, const float * gamma, float * dst_f32,
        int8_t * q, float * s, int K, int N, float eps, cudaStream_t stream) {
    const unsigned grid_x = (unsigned) (N < 65535 ? N : 65535);
    if (K <= 4096) {
        // 16 warps x 256 = 4096 -> REG_CACHE=1 (8 floats/thread, single HBM read of x!)
        k_rms_norm_mul_quantize_act_ai4<XSH, VARIANT, WRITE_F32, 16, 1>
            <<<grid_x, 16 * 32, 0, stream>>>(x, sx, gamma, dst_f32, q, s, K, N, eps);
    } else if (K <= 8192) {
        // 16 warps x 512 = 8192 -> REG_CACHE=2 (16 floats/thread, single HBM read of x! Covers K=5120!)
        k_rms_norm_mul_quantize_act_ai4<XSH, VARIANT, WRITE_F32, 16, 2>
            <<<grid_x, 16 * 32, 0, stream>>>(x, sx, gamma, dst_f32, q, s, K, N, eps);
    } else {
        // K > 8192 (e.g. 17408): streaming pass 2 from L1/L2
        k_rms_norm_mul_quantize_act_ai4<XSH, VARIANT, WRITE_F32, 16, 0>
            <<<grid_x, 16 * 32, 0, stream>>>(x, sx, gamma, dst_f32, q, s, K, N, eps);
    }
}

template <int XSH, bool WRITE_F32>
static void dispatch_rms_norm_q_var(
        const float * x, int64_t sx, const float * gamma, float * dst_f32,
        int8_t * q, float * s, int K, int N, float eps, int variant, cudaStream_t stream) {
    switch (variant) {
        case PREP_SWAP8:  dispatch_rms_norm_q<XSH, PREP_SWAP8,  WRITE_F32>(x, sx, gamma, dst_f32, q, s, K, N, eps, stream); break;
        case PREP_XPERM:  dispatch_rms_norm_q<XSH, PREP_XPERM,  WRITE_F32>(x, sx, gamma, dst_f32, q, s, K, N, eps, stream); break;
        case PREP_LINEAR: dispatch_rms_norm_q<XSH, PREP_LINEAR, WRITE_F32>(x, sx, gamma, dst_f32, q, s, K, N, eps, stream); break;
        default:          dispatch_rms_norm_q<XSH, PREP_PLAIN,  WRITE_F32>(x, sx, gamma, dst_f32, q, s, K, N, eps, stream); break;
    }
}

void launch_rms_norm_mul_quantize_act_ai4(
        const float * x, int64_t sx, const float * gamma, float * dst_f32,
        int8_t * q, float * s, int K, int N, float eps,
        int xsh, int variant, bool write_f32, cudaStream_t stream) {
    if (N <= 0 || K <= 0) {
        return;
    }
    if (xsh) {
        if (write_f32) { dispatch_rms_norm_q_var<1, true >(x, sx, gamma, dst_f32, q, s, K, N, eps, variant, stream); }
        else           { dispatch_rms_norm_q_var<1, false>(x, sx, gamma, dst_f32, q, s, K, N, eps, variant, stream); }
    } else {
        if (write_f32) { dispatch_rms_norm_q_var<0, true >(x, sx, gamma, dst_f32, q, s, K, N, eps, variant, stream); }
        else           { dispatch_rms_norm_q_var<0, false>(x, sx, gamma, dst_f32, q, s, K, N, eps, variant, stream); }
    }
}

// -------------------------------------------------------------------------------------------------
// 2. launch_swiglu_quantize_act_ai4
// -------------------------------------------------------------------------------------------------

template <int XSH, int VARIANT, bool WRITE_F32, int GLU_OP>
static void dispatch_swiglu_q_launch(
        const float * gate, int64_t s_gate, const float * up, int64_t s_up, float * dst_f32,
        int8_t * q, float * s, int K, int N, float alpha, float limit, cudaStream_t stream) {
    const unsigned gx = XSH ? (unsigned) ((K / RQ4_HB + 3) / 4) : (unsigned) ((K / 128 + 7) / 8);
    const unsigned gy = (unsigned) (N < 65535 ? N : 65535);
    k_swiglu_quantize_act_ai4<XSH, VARIANT, WRITE_F32, GLU_OP>
        <<<dim3(gx, gy), 128, 0, stream>>>(gate, s_gate, up, s_up, dst_f32, q, s, K, N, alpha, limit);
}

template <int XSH, bool WRITE_F32, int GLU_OP>
static void dispatch_swiglu_q_var(
        const float * gate, int64_t s_gate, const float * up, int64_t s_up, float * dst_f32,
        int8_t * q, float * s, int K, int N, int variant, float alpha, float limit, cudaStream_t stream) {
    switch (variant) {
        case PREP_SWAP8:  dispatch_swiglu_q_launch<XSH, PREP_SWAP8,  WRITE_F32, GLU_OP>(gate, s_gate, up, s_up, dst_f32, q, s, K, N, alpha, limit, stream); break;
        case PREP_XPERM:  dispatch_swiglu_q_launch<XSH, PREP_XPERM,  WRITE_F32, GLU_OP>(gate, s_gate, up, s_up, dst_f32, q, s, K, N, alpha, limit, stream); break;
        case PREP_LINEAR: dispatch_swiglu_q_launch<XSH, PREP_LINEAR, WRITE_F32, GLU_OP>(gate, s_gate, up, s_up, dst_f32, q, s, K, N, alpha, limit, stream); break;
        default:          dispatch_swiglu_q_launch<XSH, PREP_PLAIN,  WRITE_F32, GLU_OP>(gate, s_gate, up, s_up, dst_f32, q, s, K, N, alpha, limit, stream); break;
    }
}

template <int XSH, bool WRITE_F32>
static void dispatch_swiglu_q_op(
        const float * gate, int64_t s_gate, const float * up, int64_t s_up, float * dst_f32,
        int8_t * q, float * s, int K, int N, int variant, int glu_op, float alpha, float limit, cudaStream_t stream) {
    switch (glu_op) {
        case GLU_GEGLU:
            dispatch_swiglu_q_var<XSH, WRITE_F32, GLU_GEGLU>(gate, s_gate, up, s_up, dst_f32, q, s, K, N, variant, alpha, limit, stream);
            break;
        case GLU_SWIGLU_OAI:
            dispatch_swiglu_q_var<XSH, WRITE_F32, GLU_SWIGLU_OAI>(gate, s_gate, up, s_up, dst_f32, q, s, K, N, variant, alpha, limit, stream);
            break;
        case GLU_SWIGLU_CLAMP:
            dispatch_swiglu_q_var<XSH, WRITE_F32, GLU_SWIGLU_CLAMP>(gate, s_gate, up, s_up, dst_f32, q, s, K, N, variant, alpha, limit, stream);
            break;
        default:
            dispatch_swiglu_q_var<XSH, WRITE_F32, GLU_SWIGLU>(gate, s_gate, up, s_up, dst_f32, q, s, K, N, variant, alpha, limit, stream);
            break;
    }
}

void launch_swiglu_quantize_act_ai4(
        const float * gate, int64_t s_gate, const float * up, int64_t s_up, float * dst_f32,
        int8_t * q, float * s, int K, int N,
        int xsh, int variant, bool write_f32, int glu_op, float alpha, float limit, cudaStream_t stream) {
    if (N <= 0 || K <= 0) {
        return;
    }
    if (xsh) {
        if (write_f32) { dispatch_swiglu_q_op<1, true >(gate, s_gate, up, s_up, dst_f32, q, s, K, N, variant, glu_op, alpha, limit, stream); }
        else           { dispatch_swiglu_q_op<1, false>(gate, s_gate, up, s_up, dst_f32, q, s, K, N, variant, glu_op, alpha, limit, stream); }
    } else {
        if (write_f32) { dispatch_swiglu_q_op<0, true >(gate, s_gate, up, s_up, dst_f32, q, s, K, N, variant, glu_op, alpha, limit, stream); }
        else           { dispatch_swiglu_q_op<0, false>(gate, s_gate, up, s_up, dst_f32, q, s, K, N, variant, glu_op, alpha, limit, stream); }
    }
}

// -------------------------------------------------------------------------------------------------
// 3. launch_mmv_grp_spec_ai4 (Grouped Q/K/V for N = 9..32 across ALL 11 AI+ unit types)
// -------------------------------------------------------------------------------------------------

template <int BN, int LAY, int XSH>
static void dispatch_grp_spec_inst(
        const ai4k::mmv_group & grp, const int8_t * Xq, const float * Xs, int N, int K, cudaStream_t stream) {
    constexpr int smem = grp_spec_smem_bytes<BN, LAY, 4>();
    if constexpr (smem > 48 * 1024) {
        static bool attr_set[64] = {};
        int dev = 0;
        CUDA_CHECK_AI4K(cudaGetDevice(&dev));
        if (!attr_set[dev & 63]) {
            CUDA_CHECK_AI4K(cudaFuncSetAttribute(k_mmv_grp_spec<BN, LAY, XSH, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
            attr_set[dev & 63] = true;
        }
    }
    const dim3 grid((unsigned) grp.s0[grp.n], (unsigned) ((N + BN - 1) / BN));
    k_mmv_grp_spec<BN, LAY, XSH, 4><<<grid, 4 * 32, smem, stream>>>(grp, Xq, Xs, N, K);
}

template <int BN>
static void dispatch_grp_spec_bn(
        int lay, int xsh, const ai4k::mmv_group & grp, const int8_t * Xq, const float * Xs, int N, int K, cudaStream_t stream) {
    switch (lay) {
#define AI4F_GRP_SPEC_CASE(L) \
        case L: \
            if (xsh) { dispatch_grp_spec_inst<BN, L, 1>(grp, Xq, Xs, N, K, stream); } \
            else     { dispatch_grp_spec_inst<BN, L, 0>(grp, Xq, Xs, N, K, stream); } \
            break;
#define AI4F_GRP_SPEC_512(L) \
        case L: dispatch_grp_spec_inst<BN, L, 1>(grp, Xq, Xs, N, K, stream); break;
        AI4F_GRP_SPEC_CASE(ai4k::LAY_AIT2_U)
        AI4F_GRP_SPEC_CASE(ai4k::LAY_AIT2P_U)
        AI4F_GRP_SPEC_CASE(ai4k::LAY_AIT3_U)
        AI4F_GRP_SPEC_CASE(ai4k::LAY_AIT4_U)
        AI4F_GRP_SPEC_CASE(ai4k::LAY_AIT5_U)
        AI4F_GRP_SPEC_CASE(ai4k::LAY_AIT2_U2)
        AI4F_GRP_SPEC_CASE(ai4k::LAY_AIT3_U2)
        AI4F_GRP_SPEC_CASE(ai4k::LAY_AIT4_U2)
        AI4F_GRP_SPEC_512(ai4k::LAY_AIT2_U2_512)
        AI4F_GRP_SPEC_512(ai4k::LAY_AIT3_U2_512)
        AI4F_GRP_SPEC_512(ai4k::LAY_AIT4_U2_512)
#undef AI4F_GRP_SPEC_CASE
#undef AI4F_GRP_SPEC_512
        default: abort();
    }
}

void launch_mmv_grp_spec_ai4(
        int lay, int xsh, const ai4k::mmv_group & grp,
        const int8_t * Xq, const float * Xs, int N, int K, cudaStream_t stream) {
    if (N <= 16) {
        dispatch_grp_spec_bn<16>(lay, xsh, grp, Xq, Xs, N, K, stream);
    } else {
        dispatch_grp_spec_bn<32>(lay, xsh, grp, Xq, Xs, N, K, stream);
    }
}

// -------------------------------------------------------------------------------------------------
// 4. launch_dense_gate_up_glu_ai4 (Fused Dense Gate + Up + GLU, N = 1..32, ALL 11 AI+ types)
// -------------------------------------------------------------------------------------------------

template <int BN, int LAY, int XSH>
static void dispatch_dense_glu_inst(
        const uint8_t * W_gate, const uint8_t * W_up,
        const int8_t * Xq, const float * Xs, float * Y_glu,
        int M, int N, int K, int glu_op, float alpha, float limit, cudaStream_t stream) {
    constexpr int smem = dense_glu_smem_bytes<BN, LAY, 4>();
    if constexpr (smem > 48 * 1024) {
        static bool attr_set[64] = {};
        int dev = 0;
        CUDA_CHECK_AI4K(cudaGetDevice(&dev));
        if (!attr_set[dev & 63]) {
            CUDA_CHECK_AI4K(cudaFuncSetAttribute(k_dense_gate_up_glu<BN, LAY, XSH, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
            attr_set[dev & 63] = true;
        }
    }
    const dim3 grid((unsigned) (M / 16), (unsigned) ((N + BN - 1) / BN));
    k_dense_gate_up_glu<BN, LAY, XSH, 4><<<grid, 4 * 32, smem, stream>>>(W_gate, W_up, Xq, Xs, Y_glu, M, N, K, glu_op, alpha, limit);
}

template <int BN>
static void dispatch_dense_glu_bn(
        int lay, int xsh, int glu_op,
        const uint8_t * W_gate, const uint8_t * W_up,
        const int8_t * Xq, const float * Xs, float * Y_glu,
        int M, int N, int K, float alpha, float limit, cudaStream_t stream) {
    switch (lay) {
#define AI4F_DGLU_CASE(L) \
        case L: \
            if (xsh) { dispatch_dense_glu_inst<BN, L, 1>(W_gate, W_up, Xq, Xs, Y_glu, M, N, K, glu_op, alpha, limit, stream); } \
            else     { dispatch_dense_glu_inst<BN, L, 0>(W_gate, W_up, Xq, Xs, Y_glu, M, N, K, glu_op, alpha, limit, stream); } \
            break;
#define AI4F_DGLU_512(L) \
        case L: dispatch_dense_glu_inst<BN, L, 1>(W_gate, W_up, Xq, Xs, Y_glu, M, N, K, glu_op, alpha, limit, stream); break;
        AI4F_DGLU_CASE(ai4k::LAY_AIT2_U)
        AI4F_DGLU_CASE(ai4k::LAY_AIT2P_U)
        AI4F_DGLU_CASE(ai4k::LAY_AIT3_U)
        AI4F_DGLU_CASE(ai4k::LAY_AIT4_U)
        AI4F_DGLU_CASE(ai4k::LAY_AIT5_U)
        AI4F_DGLU_CASE(ai4k::LAY_AIT2_U2)
        AI4F_DGLU_CASE(ai4k::LAY_AIT3_U2)
        AI4F_DGLU_CASE(ai4k::LAY_AIT4_U2)
        AI4F_DGLU_512(ai4k::LAY_AIT2_U2_512)
        AI4F_DGLU_512(ai4k::LAY_AIT3_U2_512)
        AI4F_DGLU_512(ai4k::LAY_AIT4_U2_512)
#undef AI4F_DGLU_CASE
#undef AI4F_DGLU_512
        default: abort();
    }
}

void launch_dense_gate_up_glu_ai4(
        int lay, int xsh, int glu_op,
        const uint8_t * W_gate, const uint8_t * W_up,
        const int8_t * Xq, const float * Xs, float * Y_glu,
        int M, int N, int K, float alpha, float limit, cudaStream_t stream) {
    if (N <= 8) {
        dispatch_dense_glu_bn<8>(lay, xsh, glu_op, W_gate, W_up, Xq, Xs, Y_glu, M, N, K, alpha, limit, stream);
    } else if (N <= 16) {
        dispatch_dense_glu_bn<16>(lay, xsh, glu_op, W_gate, W_up, Xq, Xs, Y_glu, M, N, K, alpha, limit, stream);
    } else {
        dispatch_dense_glu_bn<32>(lay, xsh, glu_op, W_gate, W_up, Xq, Xs, Y_glu, M, N, K, alpha, limit, stream);
    }
}

// -------------------------------------------------------------------------------------------------
// 5. launch_dense_gate_up_glu_q8_ai4 (Full FFN Bridge: Gate + Up + GLU + H256/H128 + Q8)
// -------------------------------------------------------------------------------------------------

template <int BN, int LAY, int XSH_IN, int XSH_OUT>
static void dispatch_dense_glu_q8_inst(
        const uint8_t * W_gate, const uint8_t * W_up,
        const int8_t * Xq_in, const float * Xs_in,
        float * Y_glu_f32, int8_t * Xq_out, float * Xs_out,
        int M, int N, int K, int glu_op, float alpha, float limit, cudaStream_t stream) {
    constexpr int NWARPS = XSH_OUT ? 16 : 8;
    constexpr int RB     = XSH_OUT ? 256 : 128;
    constexpr int smem   = dense_glu_q8_smem_bytes<BN, LAY, XSH_OUT>();
    if constexpr (smem > 48 * 1024) {
        static bool attr_set[64] = {};
        int dev = 0;
        CUDA_CHECK_AI4K(cudaGetDevice(&dev));
        if (!attr_set[dev & 63]) {
            CUDA_CHECK_AI4K(cudaFuncSetAttribute(k_dense_gate_up_glu_q8<BN, LAY, XSH_IN, XSH_OUT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
            attr_set[dev & 63] = true;
        }
    }
    const dim3 grid((unsigned) (M / RB), (unsigned) ((N + BN - 1) / BN));
    k_dense_gate_up_glu_q8<BN, LAY, XSH_IN, XSH_OUT><<<grid, NWARPS * 32, smem, stream>>>(
        W_gate, W_up, Xq_in, Xs_in, Y_glu_f32, Xq_out, Xs_out, M, N, K, glu_op, alpha, limit);
}

template <int BN, int XSH_OUT>
static void dispatch_dense_glu_q8_bn(
        int lay, int xsh_in, int glu_op,
        const uint8_t * W_gate, const uint8_t * W_up,
        const int8_t * Xq_in, const float * Xs_in,
        float * Y_glu_f32, int8_t * Xq_out, float * Xs_out,
        int M, int N, int K, float alpha, float limit, cudaStream_t stream) {
    switch (lay) {
#define AI4F_DGLUQ8_CASE(L) \
        case L: \
            if (xsh_in) { dispatch_dense_glu_q8_inst<BN, L, 1, XSH_OUT>(W_gate, W_up, Xq_in, Xs_in, Y_glu_f32, Xq_out, Xs_out, M, N, K, glu_op, alpha, limit, stream); } \
            else        { dispatch_dense_glu_q8_inst<BN, L, 0, XSH_OUT>(W_gate, W_up, Xq_in, Xs_in, Y_glu_f32, Xq_out, Xs_out, M, N, K, glu_op, alpha, limit, stream); } \
            break;
#define AI4F_DGLUQ8_512(L) \
        case L: dispatch_dense_glu_q8_inst<BN, L, 1, XSH_OUT>(W_gate, W_up, Xq_in, Xs_in, Y_glu_f32, Xq_out, Xs_out, M, N, K, glu_op, alpha, limit, stream); break;
        AI4F_DGLUQ8_CASE(ai4k::LAY_AIT2_U)
        case ai4k::LAY_AIT2P_U:
            // For 16-warp H256 bridge at N>16, BN=16 (grid.y=2, 95 regs) avoids register spills from pair_tab addressing
            if constexpr (BN == 32 && XSH_OUT == 1) {
                if (xsh_in) { dispatch_dense_glu_q8_inst<16, ai4k::LAY_AIT2P_U, 1, 1>(W_gate, W_up, Xq_in, Xs_in, Y_glu_f32, Xq_out, Xs_out, M, N, K, glu_op, alpha, limit, stream); }
                else        { dispatch_dense_glu_q8_inst<16, ai4k::LAY_AIT2P_U, 0, 1>(W_gate, W_up, Xq_in, Xs_in, Y_glu_f32, Xq_out, Xs_out, M, N, K, glu_op, alpha, limit, stream); }
            } else {
                if (xsh_in) { dispatch_dense_glu_q8_inst<BN, ai4k::LAY_AIT2P_U, 1, XSH_OUT>(W_gate, W_up, Xq_in, Xs_in, Y_glu_f32, Xq_out, Xs_out, M, N, K, glu_op, alpha, limit, stream); }
                else        { dispatch_dense_glu_q8_inst<BN, ai4k::LAY_AIT2P_U, 0, XSH_OUT>(W_gate, W_up, Xq_in, Xs_in, Y_glu_f32, Xq_out, Xs_out, M, N, K, glu_op, alpha, limit, stream); }
            }
            break;
        AI4F_DGLUQ8_CASE(ai4k::LAY_AIT3_U)
        AI4F_DGLUQ8_CASE(ai4k::LAY_AIT4_U)
        AI4F_DGLUQ8_CASE(ai4k::LAY_AIT5_U)
        AI4F_DGLUQ8_CASE(ai4k::LAY_AIT2_U2)
        AI4F_DGLUQ8_CASE(ai4k::LAY_AIT3_U2)
        AI4F_DGLUQ8_CASE(ai4k::LAY_AIT4_U2)
        AI4F_DGLUQ8_512(ai4k::LAY_AIT2_U2_512)
        AI4F_DGLUQ8_512(ai4k::LAY_AIT3_U2_512)
        AI4F_DGLUQ8_512(ai4k::LAY_AIT4_U2_512)
#undef AI4F_DGLUQ8_CASE
#undef AI4F_DGLUQ8_512
        default: abort();
    }
}

void launch_dense_gate_up_glu_q8_ai4(
        int lay, int xsh_in, int xsh_out, int glu_op, bool write_f32,
        const uint8_t * W_gate, const uint8_t * W_up,
        const int8_t * Xq_in, const float * Xs_in,
        float * Y_glu_f32, int8_t * Xq_out, float * Xs_out,
        int M, int N, int K, float alpha, float limit, cudaStream_t stream) {
    float * y_f32 = write_f32 ? Y_glu_f32 : nullptr;
    if (xsh_out) {
        if (N <= 8)       { dispatch_dense_glu_q8_bn<8,  1>(lay, xsh_in, glu_op, W_gate, W_up, Xq_in, Xs_in, y_f32, Xq_out, Xs_out, M, N, K, alpha, limit, stream); }
        else if (N <= 16) { dispatch_dense_glu_q8_bn<16, 1>(lay, xsh_in, glu_op, W_gate, W_up, Xq_in, Xs_in, y_f32, Xq_out, Xs_out, M, N, K, alpha, limit, stream); }
        else              { dispatch_dense_glu_q8_bn<32, 1>(lay, xsh_in, glu_op, W_gate, W_up, Xq_in, Xs_in, y_f32, Xq_out, Xs_out, M, N, K, alpha, limit, stream); }
    } else {
        if (N <= 8)       { dispatch_dense_glu_q8_bn<8,  0>(lay, xsh_in, glu_op, W_gate, W_up, Xq_in, Xs_in, y_f32, Xq_out, Xs_out, M, N, K, alpha, limit, stream); }
        else if (N <= 16) { dispatch_dense_glu_q8_bn<16, 0>(lay, xsh_in, glu_op, W_gate, W_up, Xq_in, Xs_in, y_f32, Xq_out, Xs_out, M, N, K, alpha, limit, stream); }
        else              { dispatch_dense_glu_q8_bn<32, 0>(lay, xsh_in, glu_op, W_gate, W_up, Xq_in, Xs_in, y_f32, Xq_out, Xs_out, M, N, K, alpha, limit, stream); }
    }
}

} // namespace ai4fuse
