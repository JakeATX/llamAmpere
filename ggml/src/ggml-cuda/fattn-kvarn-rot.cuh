// KVarN fused rotation (fork-only, GGML_KVARN_FUSED_ROT): the plain 256-point Sylvester Hadamard H/16 on Q and on the
// attention output, folded into the KVarN flash-attention kernels instead of two GGML_OP_TURBO_WHT graph nodes.
//
// Every helper below runs the radix-2 butterfly stages in ascending order h = 1, 2, ..., 128 with the lower index
// forming a + b and the upper a - b, then scales by 1/16 (exact). That is the same float sequence per element as
// k_turbo_wht_f32_plain256 (turbo-wht.cu), so the fused results are bit-identical to the unfused graph.
//
// Contract: a KVarN FA node whose op_params[GGML_KVARN_ROT_PARAM] is 256 expects Q in the unrotated basis and returns
// the output in the unrotated basis. A CUDA path handed such a node must apply both rotations itself (the in-kernel
// paths), or run under ggml_cuda_flash_attn_ext_kvarn_rot_unfused (fattn-kvarn-rot.cu), which strips the flag and
// runs the rotations as separate passes around the path.

#pragma once

#include "common.cuh"

#define GGML_KVARN_ROT_PARAM 8

static inline bool ggml_cuda_fattn_kvarn_rot(const ggml_tensor * dst) {
    return dst->src[6] != nullptr && ggml_get_op_params_i32(dst, GGML_KVARN_ROT_PARAM) == 256;
}

// one butterfly stage across lanes: the partner value sits in lane ^ m; the lane with bit m set holds the upper index
static __device__ __forceinline__ float kvarn_rot_xlane(const float x, const int lane, const int m) {
    const float p = __shfl_xor_sync(0xFFFFFFFF, x, m, WARP_SIZE);
    return (lane & m) ? p - x : x + p;
}

static __device__ __forceinline__ void kvarn_rot_pair(float & a, float & b) {
    const float x = a, y = b;
    a = x + y;
    b = x - y;
}

// Layout "row8": lane t holds elements 8t .. 8t+7 of one 256-row (the k_turbo_wht_f32_plain256 layout). Whole warp.
static __device__ __forceinline__ void kvarn_rot256_row8(float x[8], const int lane) {
#pragma unroll
    for (int h = 1; h < 8; h <<= 1) {
#pragma unroll
        for (int i = 0; i < 8; i += 2*h) {
#pragma unroll
            for (int j = i; j < i + h; ++j) {
                kvarn_rot_pair(x[j], x[j + h]);
            }
        }
    }
#pragma unroll
    for (int m = 1; m <= 16; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            x[j] = kvarn_rot_xlane(x[j], lane, m);
        }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        x[j] *= 0.0625f;
    }
}

// Layout "f2x4": lane t holds float2 v[i] = elements (2(t + 32i), 2(t + 32i) + 1), i = 0..3: the k = lane + 32i
// stride of the MMA kernels' Q load and output store. Whole warp. Element bits: 0 in-lane, 1..5 lane, 6..7 i.
static __device__ __forceinline__ void kvarn_rot256_f2x4(float2 v[4], const int lane) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        kvarn_rot_pair(v[i].x, v[i].y);                     // h = 1
    }
#pragma unroll
    for (int m = 1; m <= 16; m <<= 1) {                     // h = 2 .. 32
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[i].x = kvarn_rot_xlane(v[i].x, lane, m);
            v[i].y = kvarn_rot_xlane(v[i].y, lane, m);
        }
    }
    kvarn_rot_pair(v[0].x, v[1].x); kvarn_rot_pair(v[0].y, v[1].y); // h = 64
    kvarn_rot_pair(v[2].x, v[3].x); kvarn_rot_pair(v[2].y, v[3].y);
    kvarn_rot_pair(v[0].x, v[2].x); kvarn_rot_pair(v[0].y, v[2].y); // h = 128
    kvarn_rot_pair(v[1].x, v[3].x); kvarn_rot_pair(v[1].y, v[3].y);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i].x *= 0.0625f;
        v[i].y *= 0.0625f;
    }
}

// Layout "frag": the fp32 C fragments acc[c].x[l] of the KVarN stream decode kernels, channel d = 16c + (lane>>2) +
// 8*(l>>1), head column 2*(lane&3) + (l&1). Each head column's 256 channels are spread over the 8 lanes that share
// lane&3. Element bits: 0..2 lane bits 2..4, 3 = l>>1, 4..7 = c. Whole warp; columns never mix.
template <typename T_C, int NTC>
static __device__ __forceinline__ void kvarn_rot256_frag(T_C * acc, const int lane) {
    static_assert(NTC == 16, "D = 256");
#pragma unroll
    for (int m = 4; m <= 16; m <<= 1) {                     // h = 1, 2, 4
#pragma unroll
        for (int c = 0; c < NTC; ++c) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                acc[c].x[l] = kvarn_rot_xlane(acc[c].x[l], lane, m);
            }
        }
    }
#pragma unroll
    for (int c = 0; c < NTC; ++c) {                         // h = 8
        kvarn_rot_pair(acc[c].x[0], acc[c].x[2]);
        kvarn_rot_pair(acc[c].x[1], acc[c].x[3]);
    }
#pragma unroll
    for (int hc = 1; hc < NTC; hc <<= 1) {                  // h = 16 .. 128
#pragma unroll
        for (int c = 0; c < NTC; ++c) {
            if ((c & hc) == 0) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    kvarn_rot_pair(acc[c].x[l], acc[c + hc].x[l]);
                }
            }
        }
    }
#pragma unroll
    for (int c = 0; c < NTC; ++c) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            acc[c].x[l] *= 0.0625f;
        }
    }
}

// Block layout: 256 threads, thread t holds element t of one row (the combine / stream-k fixup kernels). sh is a
// 256-float shared buffer; every thread of the block must call this.
static __device__ __forceinline__ float kvarn_rot256_block(float x, const int tid, float * sh) {
    const int lane = tid % WARP_SIZE;
#pragma unroll
    for (int m = 1; m <= 16; m <<= 1) {                     // h = 1 .. 16
        x = kvarn_rot_xlane(x, lane, m);
    }
#pragma unroll
    for (int h = 32; h <= 128; h <<= 1) {                   // h = 32 .. 128
        sh[tid] = x;
        __syncthreads();
        const float p = sh[tid ^ h];
        x = (tid & h) ? p - x : x + p;
        __syncthreads();
    }
    return x*0.0625f;
}

// Q prologue of the stream decode kernels: rotate rows (row r = (query row nt, head column hc)) into the 256-float
// output slots this block owns and writes only at its end, so the kernel body reads the rotated Q from there.
// Q rows need 16-byte alignment (checked on the host). Ends with __syncthreads.
static __device__ __forceinline__ void kvarn_rot256_q_to_slots(
        const char * __restrict__ Q, float * __restrict__ dst, const int n_q, const int gqa, const int head0, const int ne02,
        const int nb01, const int nb02, const int lane, const int warp, const int nwarps) {
    for (int r = warp; r < n_q*gqa; r += nwarps) {
        const int nt = r / gqa, hc = r % gqa;
        const float4 * src = (const float4 *) (Q + (size_t) nt*nb01 + (size_t) (head0 + hc)*nb02) + 2*lane;
        float x[8];
        {
            const float4 a = src[0], b = src[1];
            x[0] = a.x; x[1] = a.y; x[2] = a.z; x[3] = a.w;
            x[4] = b.x; x[5] = b.y; x[6] = b.z; x[7] = b.w;
        }
        kvarn_rot256_row8(x, lane);
        float4 * out = (float4 *) (dst + ((size_t) (nt*ne02 + head0 + hc)*gridDim.y + blockIdx.y)*256) + 2*lane;
        out[0] = make_float4(x[0], x[1], x[2], x[3]);
        out[1] = make_float4(x[4], x[5], x[6], x[7]);
    }
    __syncthreads();
}

// host: Q rows readable as float4 (the stream kernels' Q prologue)
static inline bool ggml_cuda_fattn_kvarn_rot_q_aligned(const ggml_tensor * Q) {
    return ((uintptr_t) Q->data % 16) == 0 && Q->nb[1] % 16 == 0 && Q->nb[2] % 16 == 0 && Q->nb[0] == sizeof(float);
}

// Separate-pass rotation around any KVarN path (fattn-kvarn-rot.cu): Q is rotated into a pool buffer, run(ctx, dst')
// executes on a copy of dst without the flag and with Q replaced, then the output is rotated in place.
void ggml_cuda_flash_attn_ext_kvarn_rot_unfused(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        void (*run)(ggml_backend_cuda_context & ctx, ggml_tensor * dst));
