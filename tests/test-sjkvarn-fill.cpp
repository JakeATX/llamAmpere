#include "../ggml/src/ggml-cuda/fattn-sjkvarn-fill.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

static void require(bool ok) {
    if (!ok) {
        std::fprintf(stderr, "SJ fill split check failed\n");
        std::exit(1);
    }
}

int main() {
    int cases = 0;
    for (int nsm : {20, 28, 46, 68, 80, 82, 84, 108}) {
        for (int occupancy = 1; occupancy <= 5; ++occupancy) {
            for (int n_q = 1; n_q <= 8; ++n_q) {
                for (int n_kv : {256, 4096, 32768, 102400, 204800}) {
                    const int kw = n_q <= 4 ? 4/n_q : 1;
                    const int limit = n_kv/128;
                    const int first = std::min(limit, (nsm + 3)/4);
                    const int p = ggml_cuda_sj_fill_split(nsm, occupancy, 4, limit, n_kv, kw, 24*n_q);
                    require(p >= first && p <= limit);
                    const double best = ggml_cuda_sj_fill_cost(p, nsm, occupancy, 4, n_kv, kw, 24*n_q);
                    for (int candidate = first; candidate <= limit; ++candidate) {
                        const double cost = ggml_cuda_sj_fill_cost(candidate, nsm, occupancy, 4, n_kv, kw, 24*n_q);
                        require(best <= cost);
                        require(candidate >= p || best < cost);
                    }
                    ++cases;
                }
            }
        }
    }
    // The short, wide case favors less combine traffic; the long case can pay for more splits.
    require(ggml_cuda_sj_fill_split(84, 1, 4, 32, 4096, 1, 192) == 21);
    require(ggml_cuda_sj_fill_split(82, 2, 4, 1600, 204800, 1, 192) > 32);
    require(ggml_cuda_sj_fill_split(108, 5, 4, 1, 128, 4, 24) == 1);
    std::printf("SJ fill: %d scheduling cases passed\n", cases);
}
