#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// Keep assertions active without linking ggml in the standalone build.
#undef GGML_ASSERT
#define GGML_ASSERT(x) do { if (!(x)) { std::fprintf(stderr, "GGML_ASSERT(%s) failed at %s:%d\n", #x, __FILE__, __LINE__); std::abort(); } } while (0)

#include "../src/llama-kq-mask-pack.h"

static constexpr ggml_fp16_t keep = 0x0000;
static constexpr ggml_fp16_t drop = 0xFC00;

static void pack_reference(const std::vector<ggml_fp16_t> & scratch, uint32_t * words,
        int64_t n_rows, int64_t n_kv, int64_t n_w) {
    for (int64_t r = 0; r < n_rows; ++r) {
        const ggml_fp16_t * row = scratch.data() + r*n_kv;
        uint32_t          * w   = words + r*n_w;
        for (int64_t iw = 0; iw < n_w; ++iw) {
            const int64_t j0 = iw*32;
            const int64_t nj = std::min<int64_t>(32, n_kv - j0);
            uint32_t bits = 0;
            for (int64_t b = 0; b < nj; ++b) {
                const ggml_fp16_t v = row[j0 + b];
                // the f16 mask holds 0 or -INF here (ALiBi masks never take this path)
                GGML_ASSERT(v == keep || v == drop);
                bits |= (uint32_t) (v == keep) << b;
            }
            w[iw] = bits;
        }
    }
}

static void check(const std::vector<ggml_fp16_t> & src, int64_t n_rows, int64_t n_kv, const char * pattern) {
    const int64_t n_w = (n_kv + 31)/32;
    std::vector<uint32_t> reference(n_rows*n_w, 0xA5A5A5A5);
    std::vector<uint32_t> packed(n_rows*n_w, 0x5A5A5A5A);
    pack_reference(src, reference.data(), n_rows, n_kv, n_w);
    llama_kq_mask_pack_rows(src.data(), packed.data(), n_rows, n_kv, n_w, keep, drop);
    for (size_t i = 0; i < packed.size(); ++i) {
        if (packed[i] != reference[i]) {
            std::fprintf(stderr, "mismatch: n_kv=%lld n_rows=%lld pattern=%s word=%zu old=%08x new=%08x\n",
                    (long long) n_kv, (long long) n_rows, pattern, i, reference[i], packed[i]);
            std::abort();
        }
    }
}

static void benchmark(int64_t n_rows, std::mt19937 & rng) {
    const int64_t n_kv = 352256;
    const int64_t n_w = (n_kv + 31)/32;
    std::vector<ggml_fp16_t> src(n_rows*n_kv);
    for (auto & v : src) {
        v = (rng() & 1) ? keep : drop;
    }
    std::vector<uint32_t> reference(n_rows*n_w);
    std::vector<uint32_t> packed(n_rows*n_w);
    std::vector<double> old_ms, new_ms;
    pack_reference(src, reference.data(), n_rows, n_kv, n_w);
    llama_kq_mask_pack_rows(src.data(), packed.data(), n_rows, n_kv, n_w, keep, drop);
    for (int rep = 0; rep < 30; ++rep) {
        for (int run = 0; run < 2; ++run) {
            const bool old = ((rep + run) % 2) == 0;
            const auto start = std::chrono::steady_clock::now();
            if (old) {
                pack_reference(src, reference.data(), n_rows, n_kv, n_w);
            } else {
                llama_kq_mask_pack_rows(src.data(), packed.data(), n_rows, n_kv, n_w, keep, drop);
            }
            const auto end = std::chrono::steady_clock::now();
            (old ? old_ms : new_ms).push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }
        GGML_ASSERT(reference == packed);
    }
    std::sort(old_ms.begin(), old_ms.end());
    std::sort(new_ms.begin(), new_ms.end());
    const double old_median = (old_ms[14] + old_ms[15])/2;
    const double new_median = (new_ms[14] + new_ms[15])/2;
    std::printf("n_kv=%lld n_rows=%lld, 30 reps median: old %.3f ms, new %.3f ms, speedup %.2fx\n",
            (long long) n_kv, (long long) n_rows, old_median, new_median, old_median/new_median);
}

int main() {
    std::mt19937 rng(175);
    int cases = 0;
    for (int64_t n_kv : { 1, 31, 32, 33, 4095, 262144, 262147, 352256 }) {
        for (int64_t n_rows : { 1, 6, 18, 33 }) {
            std::vector<ggml_fp16_t> src(n_rows*n_kv);
            for (auto & v : src) {
                v = (rng() & 1) ? keep : drop;
            }
            check(src, n_rows, n_kv, "random");
            ++cases;
            for (int64_t prefix : { int64_t(0), int64_t(1), int64_t(15), int64_t(31), int64_t(32), int64_t(33), n_kv/2, n_kv - 1, n_kv }) {
                for (int64_t r = 0; r < n_rows; ++r) {
                    const int64_t end = std::min(n_kv, prefix + r);
                    std::fill(src.begin() + r*n_kv, src.begin() + r*n_kv + end, keep);
                    std::fill(src.begin() + r*n_kv + end, src.begin() + (r + 1)*n_kv, drop);
                }
                check(src, n_rows, n_kv, "causal-prefix");
                ++cases;
            }
            std::fill(src.begin(), src.end(), keep);
            check(src, n_rows, n_kv, "all-keep");
            ++cases;
            std::fill(src.begin(), src.end(), drop);
            check(src, n_rows, n_kv, "all-drop");
            ++cases;
        }
    }
    std::printf("Compared %d cases across 32 shapes\n", cases);
    benchmark(18, rng);
    benchmark(33, rng);
    std::puts("KQPACK TEST OK");
}
