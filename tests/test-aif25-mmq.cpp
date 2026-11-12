// [PP27] AIF25_U exact prompt kernel (magic2::mmq) against the 8-token kernel (magic2::mmv), on the GPU.
// Runs dense AIF25_U MUL_MATs at prompt widths with GGML_CUDA_AIF25_MMQ_CHECK=2, so the backend runs both kernels on
// every qualifying call and aborts if any output differs in any bit; the test then checks that every expected call was
// checked, and compares three 16-row slabs of every output with a CPU reference (aif25_dequantize_slab, double sums):
// NMSE <= 1e-3. Tile and strips come from the environment (GGML_CUDA_AIF25_MMQ, default 64; GGML_CUDA_AIF25_MMQ_STRIPS,
// default 1); the minimum width is set to 9 unless given.
// Exit 0 = pass, 1 = fail, 77 = no CUDA device.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-aif25.h"

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

static void log_cb(ggml_log_level level, const char * text, void * user) {
    (void) level; (void) user;
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
    struct shape { int64_t m, k, n; };
    // K = 5120 / 17408: activation scale per 256 columns; K = 640 / 384: per 128 (the XSH = 0 kernels).
    // M = 48 leaves a partial two-strip block; widths cover partial, exact and multi-tile token tiles.
    const std::vector<shape> shapes = {
        {17408, 5120, 64}, {17408, 5120, 65}, {17408, 5120, 100}, {17408, 5120, 1024},
        {5120, 17408, 64}, {5120, 17408, 512},
        {2560, 640, 8}, {2560, 640, 9}, {2560, 640, 16}, {2560, 640, 33}, {2560, 640, 63}, {2560, 640, 64},
        {2560, 640, 65}, {2560, 640, 127}, {2560, 640, 129}, {2560, 640, 256},
        {640, 2560, 33}, {640, 2560, 130},
        {48, 384, 77}, {48, 384, 200},
    };
    std::mt19937 rng(20261111);
    std::uniform_real_distribution<float> scale(0.002f, 0.02f), act(-1.0f, 1.0f);
    int expected = 0;
    bool ok = true;
    for (const shape & s : shapes) {
        ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_AIF25_U, s.k, s.m);
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
        // 672-byte units: 16 fp16 row scales (32 bytes), then 640 bytes of random tile words
        std::vector<uint8_t> wb(ggml_nbytes(w));
        for (size_t b = 0; b + 672 <= wb.size(); b += 672) {
            ggml_fp16_t * d = (ggml_fp16_t *) (wb.data() + b);
            for (int i = 0; i < 16; ++i) {
                d[i] = ggml_fp32_to_fp16(scale(rng));
            }
            uint32_t * t = (uint32_t *) (wb.data() + b + 32);
            for (int i = 0; i < 160; ++i) {
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
            std::fprintf(stderr, "CUDA0 does not run AIF25_U MUL_MAT (needs Ampere or newer); this test was not run\n");
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            ggml_backend_free(backend);
            return 77;
        }
        const int before = g_checked;
        ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS && ok;
        ggml_backend_synchronize(backend);
        const int want = s.n >= min_n ? 1 : 0;
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
        const int64_t slab_bytes = s.k / 128 * AIF25_UNIT_BYTES, n_slabs = s.m / 16;
        std::vector<float> wr(16 * s.k);
        for (int64_t sl : { (int64_t) 0, n_slabs / 2, n_slabs - 1 }) {
            aif25_dequantize_slab(wb.data() + sl * slab_bytes, wr.data(), s.k);
            for (int64_t t = 0; t < s.n; ++t) {
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
        std::fprintf(stderr, "M=%lld K=%lld N=%lld checks=%d (want %d) finite=%d nonzero=%d nmse=%.3g %s\n", (long long) s.m,
                (long long) s.k, (long long) s.n, g_checked - before, want, finite, nonzero, nmse, pass ? "OK" : "FAIL");
        ok = ok && pass;
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    ggml_backend_free(backend);
    ok = ok && g_bad == 0 && g_checked == expected;
    std::fprintf(stderr, "test-aif25-mmq: tile=%s strips=%s min=%d: %d of %d expected calls checked bit-identical, %d differ: %s\n",
            std::getenv("GGML_CUDA_AIF25_MMQ"), std::getenv("GGML_CUDA_AIF25_MMQ_STRIPS") ? std::getenv("GGML_CUDA_AIF25_MMQ_STRIPS") : "1",
            min_n, g_checked, expected, g_bad, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
