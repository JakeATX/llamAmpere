#include "fuse-relmerge.cuh"

// one block per (q head h, token t), one thread per dv
static __global__ void k_fuse_relmerge(const ggml_cuda_relmerge_args a) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int d = threadIdx.x;

    // soft_max_f32 over the row (far, near, 0): val = x*1 + 0, max, expf(val - max), the warp-reduced sum, which for
    // three live lanes adds (lane 0 + lane 2) + lane 1, then 1/sum and val*inv_sum
    const float la = __fadd_rn(a.lf[h + (int64_t) t*a.lf_t], a.cf[(int64_t) h*a.cf_h + (int64_t) t*a.cf_t]);
    const float lb = a.ln[h + (int64_t) t*a.ln_t];
    const float mx = fmaxf(fmaxf(fmaxf(-INFINITY, la), lb), 0.0f);
    const float ea = expf(la - mx);
    const float eb = expf(lb - mx);
    const float e0 = expf(0.0f - mx);
    const float inv_sum = 1.0f / __fadd_rn(__fadd_rn(ea, e0), eb);
    const float wa = ea*inv_sum;
    const float wb = eb*inv_sum;

    float oa;
    if (a.a1 == nullptr) {
        oa = ((const float *) a.a0)[(int64_t) t*a.a0_nb3/sizeof(float) + (int64_t) h*a.dv + d];
    } else {
        const int i1 = h % (2*a.g);
        const int i2 = h / (2*a.g);
        const char * p = i1 < a.g ? a.a0 + (int64_t) i1*a.a0_nb1 + (int64_t) i2*a.a0_nb2 + (int64_t) t*a.a0_nb3
                                  : a.a1 + (int64_t) (i1 - a.g)*a.a1_nb1 + (int64_t) i2*a.a1_nb2 + (int64_t) t*a.a1_nb3;
        oa = ((const float *) p)[d];
    }
    const float ob = a.ob[(int64_t) t*a.ob_t + (int64_t) h*a.dv + d];

    float r = __fadd_rn(__fmul_rn(oa, wa), __fmul_rn(ob, wb));
    if (a.gate) {
        const float gv  = a.gate[(int64_t) t*a.gate_t + (int64_t) h*a.dv + d];
        const float sig = 1.0f / (1.0f + expf(-gv));
        r = __fmul_rn(sig, r);
    }
    a.dst[(int64_t) t*a.dst_t + (int64_t) h*a.dv + d] = r;
}

void ggml_cuda_relmerge_launch(ggml_backend_cuda_context & ctx, const ggml_cuda_relmerge_args & args) {
    GGML_ASSERT(args.dv > 0 && args.dv <= 1024);
    k_fuse_relmerge<<<dim3(args.H, args.T, 1), args.dv, 0, ctx.stream()>>>(args);
    CUDA_CHECK(cudaGetLastError());
}
