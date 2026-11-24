// [PP27] AIF25_U exact prompt kernel (magic2::mmq) against the 8-token kernel (magic2::mmv), on the GPU. [K126] Also the
// row-scale types AIF175_U2, AIF25_U2 and AIF25_U2_512 (GGML_CUDA_AIF_RS_MMQ, default on), with their CPU slab decoders.
// Runs dense AIF25_U MUL_MATs at prompt widths with GGML_CUDA_AIF25_MMQ_CHECK=2, so the backend runs both kernels on
// every qualifying call and aborts if any output differs in any bit; the test then checks that every expected call was
// checked, and compares three 16-row slabs of every output with a CPU reference (aif25_dequantize_slab, double sums):
// NMSE <= 1e-3. Tile and strips come from the environment (GGML_CUDA_AIF25_MMQ, default 64; GGML_CUDA_AIF25_MMQ_STRIPS,
// default 1); the minimum width is set to 9 unless given.
// [KAIMULTI] With GGML_CUDA_AIF25_MMQ_MIN above 32 (ctest variant "narrow"), widths 9..32 take the decode-once narrow
// kernel (GGML_CUDA_AIF_NARROW, default on, range GGML_CUDA_AIF_NARROW_MIN/_MAX); it is checked against mmv the same way.
// [OPT10] With GGML_CUDA_AIF_ASYNC=1 (ctest variants "async", "async-nb1"), widths 9..32 / 33..63 take the cp.async
// kernel (aif-async-mmv.cu), checked against mmv the same way.
// [K126-4] With GGML_CUDA_AIF_STAGED_GEMM=1 (ctest variant "staged") the prompt widths take the cp.async-staged GEMM
// (aif-staged-gemm.cu) where it applies (K % 512 == 0), checked against mmv the same way; the real 27B shapes at N = 256,
// 512 and 2048 are added for every type (CPU reference on the first 128 tokens of those).
// Exit 0 = pass, 1 = fail, 77 = no CUDA device.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-aif25.h"
#include "ggml-aif175.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int g_checked = 0;
static int g_bad = 0;
static int g_staged = 0;

static void log_cb(ggml_log_level level, const char * text, void * user) {
    (void) level; (void) user;
    if (std::strstr(text, "[AIF_STAGED]") != nullptr) {
        ++g_staged;
        std::fputs(text, stderr);
    }
    if (std::strstr(text, "[AIF25_MMQ] check") != nullptr) {
        ++g_checked;
        if (std::strstr(text, ": 0 of ") == nullptr) {
            ++g_bad;
        }
        std::fputs(text, stderr);
    }
}

int main() {
    setenv("GGML_CUDA_AIF25_MMQ", "64", 0);
    setenv("GGML_CUDA_AIF25_MMQ_MIN", "9", 0);
    setenv("GGML_CUDA_AIF25_MMQ_CHECK", "2", 1);
    const int min_n = std::max(9, std::atoi(std::getenv("GGML_CUDA_AIF25_MMQ_MIN")));
    const auto env_or = [](const char * name, int def) { const char * e = std::getenv(name); return e ? std::atoi(e) : def; };
    const bool narrow = env_or("GGML_CUDA_AIF_NARROW", 1) != 0;
    // [OPT10] GGML_CUDA_AIF_ASYNC=1: every width above 8 not taken by mmq / narrow takes the cp.async kernel, also checked
    const bool async = env_or("GGML_CUDA_AIF_ASYNC", 0) != 0;
    const int narrow_lo = std::min(32, std::max(9, env_or("GGML_CUDA_AIF_NARROW_MIN", 9)));
    const int narrow_hi = std::min(32, std::max(9, env_or("GGML_CUDA_AIF_NARROW_MAX", 32)));
    ggml_log_set(log_cb, nullptr);
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("CUDA0");
    if (!dev) {
        std::fprintf(stderr, "CUDA0 unavailable; this test was not run\n");
        return 77;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) {
        std::fprintf(stderr, "CUDA0 init failed\n");
        return 1;
    }
    struct shape { int64_t m, k, n; ggml_type type = GGML_TYPE_AIF25_U; int64_t ref_n = 0; };
    // K = 5120 / 17408: activation scale per 256 columns; K = 640 / 384: per 128 (the XSH = 0 kernels).
    // M = 48 leaves a partial two-strip block; widths cover partial, exact and multi-tile token tiles.
    const std::vector<shape> shapes = {
        {17408, 5120, 64}, {17408, 5120, 65}, {17408, 5120, 100}, {17408, 5120, 1024},
        {5120, 17408, 64}, {5120, 17408, 512},
        {2560, 640, 8}, {2560, 640, 9}, {2560, 640, 16}, {2560, 640, 33}, {2560, 640, 63}, {2560, 640, 64},
        {2560, 640, 65}, {2560, 640, 127}, {2560, 640, 129}, {2560, 640, 256},
        {640, 2560, 33}, {640, 2560, 130},
        {48, 384, 77}, {48, 384, 200},
        // [K126] the row-scale types on the same mmq path: AIF175_U2 / AIF25_U2_512 (K % 512, 4 chunks per row scale),
        // AIF25_U2 (K % 640, 5 chunks; K = 1920 is the XSH = 0 kernel); M = 48 = a partial two-strip block
        {17408, 5120, 64, GGML_TYPE_AIF175_U2}, {17408, 5120, 256, GGML_TYPE_AIF175_U2}, {5120, 17408, 100, GGML_TYPE_AIF175_U2},
        {10240, 5120, 512, GGML_TYPE_AIF175_U2}, {5120, 6144, 2048, GGML_TYPE_AIF175_U2}, {48, 512, 77, GGML_TYPE_AIF175_U2},
        {6144, 5120, 9, GGML_TYPE_AIF175_U2}, {6144, 5120, 8, GGML_TYPE_AIF175_U2},
        {17408, 5120, 65, GGML_TYPE_AIF25_U2}, {12288, 5120, 256, GGML_TYPE_AIF25_U2}, {1024, 5120, 1024, GGML_TYPE_AIF25_U2},
        {48, 1920, 130, GGML_TYPE_AIF25_U2}, {2560, 1920, 33, GGML_TYPE_AIF25_U2},
        {17408, 5120, 64, GGML_TYPE_AIF25_U2_512}, {5120, 6144, 300, GGML_TYPE_AIF25_U2_512}, {48, 512, 129, GGML_TYPE_AIF25_U2_512},
        // [KAIMULTI] verify widths 9..32 (NB 2, 3 and 4, partial and full 8-token blocks) at the 27B shapes, every type
        {17408, 5120, 12, GGML_TYPE_AIF175_U2}, {5120, 17408, 20, GGML_TYPE_AIF175_U2}, {10240, 5120, 32, GGML_TYPE_AIF175_U2},
        {6144, 5120, 17, GGML_TYPE_AIF175_U2}, {5120, 6144, 25, GGML_TYPE_AIF175_U2}, {48, 512, 31, GGML_TYPE_AIF175_U2},
        {17408, 5120, 9, GGML_TYPE_AIF25_U2}, {1024, 5120, 16, GGML_TYPE_AIF25_U2}, {12288, 5120, 24, GGML_TYPE_AIF25_U2},
        {2560, 1920, 32, GGML_TYPE_AIF25_U2}, {48, 1920, 30, GGML_TYPE_AIF25_U2},
        {5120, 6144, 23, GGML_TYPE_AIF25_U2_512}, {48, 512, 10, GGML_TYPE_AIF25_U2_512},
        {2560, 640, 10}, {640, 2560, 31}, {48, 384, 23}, {17408, 5120, 28}, {2560, 640, 33},
    };
    const bool staged = env_or("GGML_CUDA_AIF_STAGED_GEMM", 0) != 0;
    std::vector<shape> all = shapes;
    if (staged) {
        for (auto mk : std::vector<std::pair<int64_t, int64_t>>{ { 17408, 5120 }, { 5120, 17408 }, { 10240, 5120 }, { 6144, 5120 }, { 5120, 6144 } }) {
            for (int64_t n : { 256, 512, 2048 }) {
                for (ggml_type t : { GGML_TYPE_AIF175_U2, GGML_TYPE_AIF25_U2, GGML_TYPE_AIF25_U2_512, GGML_TYPE_AIF25_U }) {
                    if (t == GGML_TYPE_AIF25_U2 && mk.second % 640 != 0) { continue; }
                    all.push_back({ mk.first, mk.second, n, t, 128 });
                }
            }
        }
        // partial CTA tiles: M not a multiple of 64 rows, N not a multiple of 64 tokens
        all.push_back({ 48, 512, 77, GGML_TYPE_AIF175_U2 });
        all.push_back({ 1040, 5120, 130, GGML_TYPE_AIF25_U2 });
        all.push_back({ 1040, 1024, 1000, GGML_TYPE_AIF25_U2_512 });
        all.push_back({ 2064, 1024, 65, GGML_TYPE_AIF25_U });
    }
    std::mt19937 rng(20261111);
    std::uniform_real_distribution<float> scale(0.002f, 0.02f), act(-1.0f, 1.0f);
    int expected = 0;
    bool ok = true;
    for (const shape & s : all) {
        ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * w = ggml_new_tensor_2d(ctx, s.type, s.k, s.m);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.k, s.n);
        ggml_tensor * y = ggml_mul_mat(ctx, w, x);
        ggml_set_name(y, "aif25_test");
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) {
            std::fprintf(stderr, "alloc failed for M=%lld K=%lld N=%lld\n", (long long) s.m, (long long) s.k, (long long) s.n);
            ok = false;
            ggml_free(ctx);
            break;
        }
        // units / blocks: 16 fp16 row scales (32 bytes), then random tile words. AIF25_U 672-byte 16 x 128 units;
        // AIF25_U2 3232-byte 16 x 640, AIF25_U2_512 2592-byte 16 x 512, AIF175_U2 1824-byte 16 x 512 blocks
        const int64_t unit = s.type == GGML_TYPE_AIF25_U2 ? 3232 : s.type == GGML_TYPE_AIF25_U2_512 ? 2592 :
                             s.type == GGML_TYPE_AIF175_U2 ? AIF175_BLOCK_BYTES : AIF25_UNIT_BYTES;
        const int64_t unit_k = s.type == GGML_TYPE_AIF25_U2 ? 640 : s.type == GGML_TYPE_AIF25_U ? 128 : 512;
        std::vector<uint8_t> wb(ggml_nbytes(w));
        for (size_t b = 0; b + unit <= wb.size(); b += unit) {
            ggml_fp16_t * d = (ggml_fp16_t *) (wb.data() + b);
            for (int i = 0; i < 16; ++i) {
                d[i] = ggml_fp32_to_fp16(scale(rng));
            }
            uint32_t * t = (uint32_t *) (wb.data() + b + 32);
            for (int64_t i = 0; i < (unit - 32)/4; ++i) {
                t[i] = (uint32_t) rng();
            }
        }
        std::vector<float> xb(ggml_nelements(x));
        for (float & v : xb) {
            v = act(rng);
        }
        ggml_backend_tensor_set(w, wb.data(), 0, wb.size());
        ggml_backend_tensor_set(x, xb.data(), 0, ggml_nbytes(x));
        if (!ggml_backend_supports_op(backend, y)) {
            std::fprintf(stderr, "CUDA0 does not run %s MUL_MAT (needs Ampere or newer); this test was not run\n", ggml_type_name(s.type));
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            ggml_backend_free(backend);
            return 77;
        }
        const int before = g_checked;
        ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS && ok;
        ggml_backend_synchronize(backend);
        const int want = s.n >= min_n || (narrow && s.n >= narrow_lo && s.n <= narrow_hi) || (async && s.n > 8 && s.n < min_n) ? 1 : 0;
        expected += want;
        std::vector<float> yb(ggml_nelements(y));
        ggml_backend_tensor_get(y, yb.data(), 0, ggml_nbytes(y));
        bool finite = true, nonzero = false;
        for (float v : yb) {
            finite = finite && std::isfinite(v);
            nonzero = nonzero || v != 0.0f;
        }
        // CPU reference on three slabs (first, middle, last): W rows dequantized and unrotated, double sums
        double err = 0.0, ref2 = 0.0;
        const int64_t slab_bytes = s.k / unit_k * unit, n_slabs = s.m / 16;
        std::vector<float> wr(16 * s.k);
        for (int64_t sl : { (int64_t) 0, n_slabs / 2, n_slabs - 1 }) {
            const uint8_t * slab = wb.data() + sl * slab_bytes;
            if (s.type == GGML_TYPE_AIF175_U2) {
                aif175_dequantize_slab(slab, wr.data(), s.k);
            } else if (s.type == GGML_TYPE_AIF25_U2) {
                aif25_dequantize_slab_v2(slab, wr.data(), s.k);
            } else if (s.type == GGML_TYPE_AIF25_U2_512) {
                aif25_dequantize_slab_v2_512(slab, wr.data(), s.k);
            } else {
                aif25_dequantize_slab(slab, wr.data(), s.k);
            }
            for (int64_t t = 0; t < (s.ref_n > 0 ? std::min(s.n, s.ref_n) : s.n); ++t) {
                for (int r = 0; r < 16; ++r) {
                    double acc = 0.0;
                    for (int64_t k = 0; k < s.k; ++k) {
                        acc += (double) wr[r * s.k + k] * xb[t * s.k + k];
                    }
                    const double e = yb[t * s.m + sl * 16 + r] - acc;
                    err += e * e;
                    ref2 += acc * acc;
                }
            }
        }
        const double nmse = ref2 > 0.0 ? err / ref2 : 1.0;
        const bool pass = finite && nonzero && g_checked - before == want && nmse <= 1e-3;
        std::fprintf(stderr, "%s M=%lld K=%lld N=%lld checks=%d (want %d) finite=%d nonzero=%d nmse=%.3g %s\n", ggml_type_name(s.type), (long long) s.m,
                (long long) s.k, (long long) s.n, g_checked - before, want, finite, nonzero, nmse, pass ? "OK" : "FAIL");
        ok = ok && pass;
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    ggml_backend_free(backend);
    ok = ok && g_bad == 0 && g_checked == expected && (!staged || g_staged == 1);
    std::fprintf(stderr, "test-aif25-mmq: tile=%s strips=%s min=%d staged=%d: %d of %d expected calls checked bit-identical, %d differ: %s\n",
            std::getenv("GGML_CUDA_AIF25_MMQ"), std::getenv("GGML_CUDA_AIF25_MMQ_STRIPS") ? std::getenv("GGML_CUDA_AIF25_MMQ_STRIPS") : "1",
            min_n, staged ? 1 : 0, g_checked, expected, g_bad, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
