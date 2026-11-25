// INKATTN live tiles (fork): which 16-key tiles of an F16 mask hold any visible (> -inf) entry for the g*T query rows
// of each (sequence, KV head), compacted in key order. The decode kernel then walks only those tiles: a tile whose
// rows are all -inf adds nothing to the online softmax (exp(-inf - m) = 0, the running max and sum keep their values),
// so skipping it is exact; only the run boundaries move (rounding-level split of the partial sums).
// Used for the Inkling SWA / near-window nodes, whose cache is n_swa + n_ubatch cells but only ~n_swa + T are visible.

#include "inkattn.cuh"

namespace {

constexpr int LIVE_THREADS = 256;

static __device__ __forceinline__ bool inkattn_live_any16(const uint4 a) {
    // 8 f16 values; -inf is 0xFC00
    const uint32_t w[4] = {a.x, a.y, a.z, a.w};
    bool any = false;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        any |= (w[i] & 0xFFFFu) != 0xFC00u;
        any |= (w[i] >> 16) != 0xFC00u;
    }
    return any;
}

static __global__ void __launch_bounds__(LIVE_THREADS)
inkattn_live_kernel(const char * __restrict__ M, const int64_t nbm1, const int64_t nbm2, const int64_t nbm3, const int ne32,
        const int ne33, const int n_kv, const int T, const int g, const int Hkv, const int al16, const int NTmax,
        int * __restrict__ live, int * __restrict__ nlive) {
    const int kvh = blockIdx.x;
    const int seq = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int NT = (n_kv + 15)/16;
    const int R = g*T;
    // a broadcast mask (ne32 == 1) is the same for the g heads of the group: one row per token
    const int gstep = ne32 == 1 ? g : 1;

    __shared__ int wcount[LIVE_THREADS/WARP_SIZE];
    int * out = live + (int64_t) (seq*Hkv + kvh)*NTmax;
    int base = 0;

    for (int c0 = 0; c0 < NT; c0 += LIVE_THREADS) {
        const int t = c0 + tid;
        bool f = false;
        if (t < NT) {
            const int k0 = 16*t;
            const bool full = al16 && k0 + 16 <= n_kv;
            for (int r = 0; r < R && !f; r += gstep) {
                const int h = kvh*g + r % g;
                const half * row = (const half *) (M + (int64_t) (seq % ne33)*nbm3 + (int64_t) (h % ne32)*nbm2 + (int64_t) (r / g)*nbm1);
                if (full) {
                    const uint4 * p = reinterpret_cast<const uint4 *>(row + k0);
                    f = inkattn_live_any16(__ldg(p)) || inkattn_live_any16(__ldg(p + 1));
                } else {
                    for (int k = k0; k < min(k0 + 16, n_kv) && !f; ++k) {
                        f = __half_as_ushort(row[k]) != 0xFC00u;
                    }
                }
            }
        }
        const uint32_t b = __ballot_sync(0xFFFFFFFF, f);
        if (lane == 0) {
            wcount[warp] = __popc(b);
        }
        __syncthreads();
        int off = base;
        int tot = 0;
#pragma unroll
        for (int w = 0; w < LIVE_THREADS/WARP_SIZE; ++w) {
            off += w < warp ? wcount[w] : 0;
            tot += wcount[w];
        }
        if (f) {
            out[off + __popc(b & ((1u << lane) - 1u))] = t;
        }
        base += tot;
        __syncthreads();
    }
    if (tid == 0) {
        nlive[seq*Hkv + kvh] = base;
    }
}

} // namespace

void ggml_cuda_inkattn_live(const char * M, const int64_t nbm1, const int64_t nbm2, const int64_t nbm3, const int ne32, const int ne33,
        const int n_kv, const int T, const int g, const int Hkv, const int S, const bool al16, const int NTmax, int * live, int * nlive,
        cudaStream_t stream) {
    inkattn_live_kernel<<<dim3(Hkv, S), LIVE_THREADS, 0, stream>>>(M, nbm1, nbm2, nbm3, ne32, ne33, n_kv, T, g, Hkv, al16 ? 1 : 0,
        NTmax, live, nlive);
    CUDA_CHECK(cudaGetLastError());
}
