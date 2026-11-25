// [OPT10] CPU reference for the cp.async fractional AI+ matvec (ggml/src/ggml-cuda/aif-async-mmv.cu): the only parts it
// does not share with magic2::mmv are the weight-unit addressing (aif-async-addr.h cursor) and the int32 -> float magic
// conversion. This checks, with no GPU:
//  1. for AIF25_U, AIF25_U2, AIF25_U2_512 and AIF175_U2, several K and strips, every warp's unit walk (u = w, w + 4, ...)
//     with the cursor places each 32-bit word of each unit at the shared-memory word magic2::mmv loads it to, read
//     from the same global word (the mmv / mmq ROW_SCALE and plain addressing), and every 16-byte copy is aligned;
//  2. the block / unit byte sizes are the format's (672 / 3232 / 2592 / 1824);
//  3. magic_i2f(x) == (float) x bit for bit for every |x| <= 2^21 (the IMMA bound of one 128-column unit).
// Exit 0 = pass, 1 = fail.
#include "../ggml/src/ggml-cuda/aif-async-addr.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>

static int g_fail = 0;

static float magic_i2f_host(int x) {
    const uint32_t bits = (uint32_t) (x + 0x4B400000);
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f - 12582912.0f;
}

template <int B, int V, bool ROW_SCALE, int SC>
static void check_type(const char * name, int expect_block_bytes, int k_mult) {
    using cur_t = aif_async::cursor<B, V, ROW_SCALE, SC>;
    constexpr int UW = cur_t::UB / 4;
    if (cur_t::BLOCK_B != expect_block_bytes) {
        std::fprintf(stderr, "%s: block bytes %d, want %d\n", name, cur_t::BLOCK_B, expect_block_bytes);
        ++g_fail;
    }
    long long words = 0;
    for (int kk : { 1, 2, 3, 8, 27, 34 }) {
        const int K = k_mult * kk, nu = K / 128;
        for (int64_t ms : { (int64_t) 0, (int64_t) 1, (int64_t) 7, (int64_t) 1087 }) {
            for (int warp = 0; warp < 4; ++warp) {
                cur_t cur;
                cur.init(ms, nu, warp);
                bool first = true;
                for (int u = warp; u < nu; u += 4) {
                    if (!first) {
                        cur.advance();
                    }
                    first = false;
                    for (int c = 0; c < cur_t::CHUNKS; ++c) {
                        const int64_t off = cur.chunk(c);
                        if (off % 16 != 0) {
                            std::fprintf(stderr, "%s K=%d ms=%lld u=%d chunk %d: offset %lld not 16-byte aligned\n", name, K,
                                    (long long) ms, u, c, (long long) off);
                            ++g_fail;
                            return;
                        }
                        for (int e = 0; e < 4; ++e) {
                            const int i = 4 * c + e;   // shared-memory word of the unit tile
                            int64_t want;               // global word magic2::mmv loads into tile word i
                            if (ROW_SCALE) {
                                const int64_t BW = 8 + SC * (UW - 8);
                                const int64_t block = (ms * (nu / SC) + u / SC) * BW;
                                want = i < 8 ? block + i : block + 8 + (u % SC) * (UW - 8) + i - 8;
                            } else {
                                want = (ms * nu + u) * UW + i;
                            }
                            const int64_t got = off / 4 + e;
                            ++words;
                            if (got != want) {
                                std::fprintf(stderr, "%s K=%d ms=%lld warp=%d u=%d word %d: global word %lld, mmv reads %lld\n",
                                        name, K, (long long) ms, warp, u, i, (long long) got, (long long) want);
                                ++g_fail;
                                return;
                            }
                        }
                    }
                }
            }
        }
    }
    std::fprintf(stderr, "%s: %lld tile words addressed as mmv: OK\n", name, words);
}

int main() {
    check_type<5, 2, false, 1>("AIF25_U", 672, 128);
    check_type<5, 2, true, 5>("AIF25_U2", 3232, 640);
    check_type<5, 2, true, 4>("AIF25_U2_512", 2592, 512);
    check_type<7, 4, true, 4>("AIF175_U2", 1824, 512);
    int bad = 0;
    for (int x = -(1 << 21); x <= (1 << 21); ++x) {
        const float a = magic_i2f_host(x), b = (float) x;
        uint32_t ua, ub;
        std::memcpy(&ua, &a, 4);
        std::memcpy(&ub, &b, 4);
        bad += ua != ub;
    }
    std::fprintf(stderr, "magic_i2f: %d of %d values differ from (float) x\n", bad, (1 << 22) + 1);
    g_fail += bad != 0;
    std::fprintf(stderr, "test-aif-async-addr: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
