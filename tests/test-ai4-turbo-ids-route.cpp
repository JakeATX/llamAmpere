// [TURBO-IDS] CPU check of the direct-ids routing math (ggml/src/ggml-cuda/ai4-turbo-ids-route.h) used by the opt-in
// GGML_CUDA_AI4_TURBO_IDS expert matvec. For random ids (skipped -1 / >= n_expert ids, repeats, padded token stride,
// ne11 = 1 and ne11 = n_used) it walks the CUDA grid (slab, u, tok) and checks against the ids_helper contract
// (pair (u, t) with a valid id reads Xq row t*ne11 + u % ne11 and writes Y column t*n_used + u):
//   - the multiset of (expert, xrow, ycol) for live CTAs equals the ids_helper pair lists,
//   - every (slab, ycol) output slice is written exactly once when skipped pairs are zeroed, and exactly the live ones
//     when they are not (y_into), and every write stays inside [M, n_used, n_tok],
//   - activation rows stay inside the ne11*n_tok quantized rows and weight units inside the expert stack.

#include "../ggml/src/ggml-cuda/ai4-turbo-ids-route.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <tuple>
#include <vector>

static int run_case(std::mt19937 & rng, int n_expert, int n_used, int n_tok, int ne11, int M, int K, int pad, bool zero) {
    const int64_t nb0 = sizeof(int32_t);
    const int64_t nb1 = (int64_t) (n_used + pad) * nb0;
    std::vector<char> buf((size_t) (nb1 * n_tok));
    std::uniform_int_distribution<int> pick(0, n_expert - 1);
    std::uniform_int_distribution<int> roll(0, 9);
    for (int t = 0; t < n_tok; ++t) {
        for (int u = 0; u < n_used + pad; ++u) {
            int e = pick(rng);
            const int r = roll(rng);
            if (u >= n_used) {
                e = 0x7fffffff;  // padding, must never be read
            } else if (r == 0) {
                e = -1;
            } else if (r == 1) {
                e = n_expert + (int) (rng() % 2);  // n_expert itself and past it
            } else if (r == 2 && u > 0) {
                e = *(int32_t *) (buf.data() + (u - 1) * nb0 + t * nb1);  // repeat within a token
            }
            *(int32_t *) (buf.data() + u * nb0 + t * nb1) = e;
        }
    }

    // ids_helper contract
    std::vector<std::tuple<int, int64_t, int64_t>> ref;
    for (int64_t p = 0; p < (int64_t) n_used * n_tok; ++p) {
        const int u = (int) (p % n_used), t = (int) (p / n_used);
        const int e = *(const int32_t *) (buf.data() + u * nb0 + t * nb1);
        if (e >= 0 && e < n_expert) {
            ref.emplace_back(e, (int64_t) t * ne11 + u % ne11, (int64_t) t * n_used + u);
        }
    }

    // direct grid walk
    const int slabs = M / 16;
    const int nk = K / 128;
    const int64_t unit = 1056;  // AIT4_U unit bytes; only the bounds relation matters
    const int64_t nb02 = (int64_t) slabs * nk * unit;
    const int64_t n_rows = (int64_t) ne11 * n_tok;
    const int64_t y_elems = (int64_t) M * n_used * n_tok;
    std::vector<int> writes((size_t) y_elems, 0);
    std::vector<std::tuple<int, int64_t, int64_t>> got;
    int bad = 0;
    for (int slab = 0; slab < slabs; ++slab) {
        for (int tok = 0; tok < n_tok; ++tok) {
            for (int u = 0; u < n_used; ++u) {
                const ai4_tids_pair pr = ai4_tids_route(buf.data(), nb0, nb1, u, tok, n_used, ne11, n_expert);
                if (pr.ycol < 0 || pr.ycol >= (int64_t) n_used * n_tok) { ++bad; continue; }
                if (!pr.live) {
                    if (zero) {
                        for (int i = 0; i < 16; ++i) writes[(size_t) (pr.ycol * M + slab * 16 + i)]++;
                    }
                    continue;
                }
                if (pr.xrow < 0 || pr.xrow >= n_rows) { ++bad; }
                const int64_t wlast = (int64_t) pr.expert * nb02 + ((int64_t) slab * nk + nk) * unit;
                if (wlast > (int64_t) n_expert * nb02) { ++bad; }
                for (int i = 0; i < 16; ++i) writes[(size_t) (pr.ycol * M + slab * 16 + i)]++;
                if (slab == 0) got.emplace_back(pr.expert, pr.xrow, pr.ycol);
            }
        }
    }
    std::sort(ref.begin(), ref.end());
    std::sort(got.begin(), got.end());
    if (ref != got) { ++bad; }
    for (int64_t c = 0; c < (int64_t) n_used * n_tok; ++c) {
        const int e = *(const int32_t *) (buf.data() + (c % n_used) * nb0 + (c / n_used) * nb1);
        const bool live = e >= 0 && e < n_expert;
        const int want = live || zero ? 1 : 0;
        for (int r = 0; r < M; ++r) {
            if (writes[(size_t) (c * M + r)] != want) { ++bad; break; }
        }
    }
    if (bad) {
        printf("FAIL n_expert %d n_used %d n_tok %d ne11 %d M %d K %d pad %d zero %d: %d errors\n",
               n_expert, n_used, n_tok, ne11, M, K, pad, (int) zero, bad);
    }
    return bad;
}

int main() {
    std::mt19937 rng(1234);
    int fails = 0, cases = 0;
    struct shape { int n_expert, n_used, M, K; };
    const shape shapes[] = {
        { 128, 8, 2304, 2304 }, { 128, 8, 2304, 1152 },  // Inkling-2-Mini gate_up / down
        { 4, 3, 48, 640 }, { 512, 10, 640, 2560 }, { 1, 1, 16, 128 },
    };
    for (const shape & s : shapes) {
        for (int n_tok = 1; n_tok <= 8; ++n_tok) {
            for (int ne11 : { 1, s.n_used }) {
                for (int pad : { 0, 3 }) {
                    for (bool zero : { true, false }) {
                        for (int rep = 0; rep < 4; ++rep) {
                            fails += run_case(rng, s.n_expert, s.n_used, n_tok, ne11, s.M, s.K, pad, zero) != 0;
                            ++cases;
                        }
                    }
                }
            }
        }
    }
    printf("ai4 turbo-ids route: %d / %d cases passed\n", cases - fails, cases);
    return fails == 0 ? 0 : 1;
}
