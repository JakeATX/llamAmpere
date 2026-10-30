#pragma once

// Copies the experts the plan listed for one projection into its scratch slots.
template<bool expert_rows>
static __global__ void moe_hf_copy_kernel(const void * const * src, const int * nstage, char * scratch,
        size_t stride, size_t copy_bytes) {
    const int n_stage = *nstage;
    if (n_stage == 0) {
        return;
    }
    const size_t chunks = copy_bytes / 16;
    const size_t total = chunks * (size_t) n_stage;
    const size_t step = (size_t) gridDim.x * blockDim.x;
    if (expert_rows) {
        // fork: visit one expert at a time to remove per-chunk division by the runtime row size.
        // Keep four loads in flight and copy the same bytes into the same scratch slots.
        for (int r = 0; r < n_stage; ++r) {
            const int4 * source = (const int4 *) src[r];
            int4 * dest = (int4 *) (scratch + (size_t) r * stride);
            size_t o = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
            for (; o + 3 * step < chunks; o += 4 * step) {
                int4 v[4];
#pragma unroll
                for (int k = 0; k < 4; ++k) { v[k] = __ldcs(source + o + k * step); }
#pragma unroll
                for (int k = 0; k < 4; ++k) { dest[o + k * step] = v[k]; }
            }
            for (; o < chunks; o += step) { dest[o] = __ldcs(source + o); }
        }
        return;
    }
    size_t c = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    // Four loads in flight per thread before the stores keep the link busy.
    for (; c + 3 * step < total; c += 4 * step) {
        int4 v[4];
        size_t r[4], o[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const size_t ck = c + k * step;
            r[k] = ck / chunks;
            o[k] = ck - r[k] * chunks;
            v[k] = __ldcs((const int4 *) src[r[k]] + o[k]);
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            ((int4 *) (scratch + r[k] * stride))[o[k]] = v[k];
        }
    }
    for (; c < total; c += step) {
        const size_t r = c / chunks;
        const size_t o = c - r * chunks;
        ((int4 *) (scratch + r * stride))[o] = __ldcs((const int4 *) src[r] + o);
    }
}

