#pragma once

// [TURBO-IDS] routing math of the direct-ids turbo expert matvec (ai4-turbo-ids.cu), shared with the CPU test
// tests/test-ai4-turbo-ids-route.cpp. One CTA per (16-row slab, slot u, token tok) reads its expert id itself:
// the same (Xq row, Y column) pair that ai4k::ids_helper would list for that id, with no routing launch.

#include <cstdint>

#if defined(__CUDACC__)
#define AI4_TIDS_HD __host__ __device__ __forceinline__
#else
#define AI4_TIDS_HD inline
#endif

struct ai4_tids_pair {
    int     expert;  // routed expert id as stored in ids
    bool    live;    // 0 <= expert < n_expert; otherwise the pair is skipped (its Y column is zeroed or left alone)
    int64_t xrow;    // quantized activation row: tok*ne11 + u % ne11 (ne11 = 1 broadcast or ne11 = n_used)
    int64_t ycol;    // output column: tok*n_used + u
};

// ids: int32 [n_used, n_tok] with byte strides nb0 / nb1
AI4_TIDS_HD ai4_tids_pair ai4_tids_route(const char * ids, int64_t nb0, int64_t nb1, int u, int tok,
                                         int n_used, int ne11, int n_expert) {
    ai4_tids_pair p;
    p.expert = *(const int32_t *) (ids + (int64_t) u*nb0 + (int64_t) tok*nb1);
    p.live   = p.expert >= 0 && p.expert < n_expert;
    p.xrow   = (int64_t) tok*ne11 + u % ne11;
    p.ycol   = (int64_t) tok*n_used + u;
    return p;
}
