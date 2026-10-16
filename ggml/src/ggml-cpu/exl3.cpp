#include "exl3.h"
#include "ggml-cpu-impl.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "vec.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cstdlib>
#include <cstring>

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#include <immintrin.h>
#define GGML_EXL3_AVX2 1
#else
#define GGML_EXL3_AVX2 0
#endif

static bool exl3_avx2_usable() {
#if GGML_EXL3_AVX2 && (defined(__GNUC__) || defined(__clang__))
    static const bool usable = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
    return usable;
#else
    return false;
#endif
}

// fork: native EXL3 CPU experiment. The ordinary dense reference remains the default.
static bool exl3_opt_in() {
    const char * value = getenv("GGML_CPU_EXL3");
    return value && strcmp(value, "1") == 0;
}

bool ggml_cpu_exl3_enabled(void) {
    return exl3_opt_in();
}

static int exl3_piece_size() {
    const char * value = getenv("GGML_CPU_EXL3_PIECE");
    if (value && strcmp(value, "128") == 0) { return 128; }
    if (value && strcmp(value, "512") == 0) { return 512; }
    return 256;
}

static int exl3_batch_size() {
    const char * value = getenv("GGML_CPU_EXL3_BATCH");
    if (value && strcmp(value, "1") == 0) { return 1; }
    if (value && strcmp(value, "2") == 0) { return 2; }
    return 3;
}

static bool exl3_span_supported(const ggml_tensor * t) {
    size_t span = ggml_type_size(t->type);
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        const int64_t count = d == 0 ? t->ne[d] / ggml_blck_size(t->type) : t->ne[d];
        if (count <= 0 || (t->nb[d] && (uint64_t) (count - 1) > (SIZE_MAX - span) / t->nb[d])) { return false; }
        span += (count - 1) * t->nb[d];
    }
    return t->view_offs <= SIZE_MAX - span;
}

static bool exl3_side_supported(const ggml_tensor * side, int64_t length, int64_t experts) {
    return side && (side->type == GGML_TYPE_F32 || side->type == GGML_TYPE_F16) &&
        side->ne[0] == length && side->ne[1] == experts && side->ne[2] == 1 && side->ne[3] == 1 &&
        side->nb[0] == ggml_type_size(side->type) && side->nb[1] % side->nb[0] == 0 && side->view_offs % side->nb[0] == 0 &&
        exl3_span_supported(side);
}

bool ggml_cpu_exl3_supported(const ggml_tensor * dst) {
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    const bool indexed = dst->op == GGML_OP_MUL_MAT_ID;
    if (!w || !x || !ggml_exl3_bits(w->type) || x->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        w->ne[0] <= 0 || w->ne[0] > INT_MAX || w->ne[1] <= 0 || w->ne[2] <= 0 ||
        w->ne[0] % 256 || w->ne[1] % 16 || w->ne[3] != 1 ||
        w->nb[0] != ggml_type_size(w->type) || w->nb[1] != ggml_row_size(w->type, w->ne[0]) ||
        (size_t) w->ne[1] > SIZE_MAX / w->nb[1] ||
        w->nb[2] < w->nb[1] * w->ne[1] || w->nb[2] % sizeof(uint32_t) || w->view_offs % sizeof(uint32_t) ||
        x->ne[0] != w->ne[0] || x->ne[1] <= 0 || x->ne[2] <= 0 || x->ne[3] <= 0 ||
        x->ne[1] > INT64_MAX / x->ne[2] || x->ne[1] * x->ne[2] > INT64_MAX / x->ne[3] ||
        dst->ne[0] != w->ne[1] ||
        x->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return false;
    }
    if (!exl3_span_supported(w) || !exl3_span_supported(x) || !exl3_span_supported(dst)) { return false; }
    for (int d = 1; d < GGML_MAX_DIMS; ++d) {
        if (x->nb[d] % sizeof(float) || dst->nb[d] % sizeof(float)) { return false; }
    }
    if (x->view_offs % sizeof(float) || dst->view_offs % sizeof(float)) { return false; }
    if (indexed) {
        const ggml_tensor * ids = dst->src[2];
        if (!ids || ids->type != GGML_TYPE_I32 || ids->ne[0] <= 0 || ids->ne[1] <= 0 ||
            ids->ne[2] != 1 || ids->ne[3] != 1 ||
            ids->ne[0] > INT64_MAX / ids->ne[1] ||
            x->ne[3] != 1 || ids->ne[0] % x->ne[1] || ids->ne[1] != x->ne[2] ||
            dst->ne[1] != ids->ne[0] || dst->ne[2] != ids->ne[1] || dst->ne[3] != 1) {
            return false;
        }
        if (!exl3_span_supported(ids) || ids->ne[0] * ids->ne[1] > INT64_MAX / (w->ne[1] / 16)) { return false; }
    } else if (w->ne[2] != 1 || dst->ne[1] != x->ne[1] || dst->ne[2] != x->ne[2] || dst->ne[3] != x->ne[3]) {
        return false;
    } else if (x->ne[1] * x->ne[2] * x->ne[3] > INT64_MAX / (w->ne[1] / 16)) {
        return false;
    }
    const ggml_tensor * suh = dst->src[indexed ? 3 : 2];
    const ggml_tensor * svh = dst->src[indexed ? 4 : 3];
    return (!suh && !svh) || (w->ne[1] % 128 == 0 &&
        exl3_side_supported(suh, w->ne[0], w->ne[2]) && exl3_side_supported(svh, w->ne[1], w->ne[2]));
}

static int64_t exl3_rows(const ggml_tensor * dst) {
    return dst->op == GGML_OP_MUL_MAT_ID ? ggml_nelements(dst->src[2]) : ggml_nrows(dst->src[1]);
}

struct alignas(64) exl3_job {
    int64_t row[3];
    int32_t expert;
    int32_t count;
};

// Bound partial vectors to 64 MiB; larger operations use the existing dense or indexed reference.
static size_t exl3_fast_work_size(const ggml_tensor * dst) {
    const size_t cap = 64 * 1024 * 1024;
    const size_t rows = exl3_rows(dst);
    const size_t pieces = (dst->src[0]->ne[0] + exl3_piece_size() - 1) / exl3_piece_size();
    const size_t n = dst->src[0]->ne[1];
    if (!rows || rows > cap / sizeof(exl3_job) || pieces > (INT_MAX - 1024) / rows ||
        n > cap / sizeof(float) / rows / pieces) {
        return 0;
    }
    const size_t size = 127 + rows * sizeof(exl3_job) + rows * pieces * n * sizeof(float);
    return size <= cap ? size : 0;
}

size_t ggml_cpu_exl3_work_size(const ggml_tensor * dst, int nth) {
    GGML_ASSERT(ggml_cpu_exl3_supported(dst));
    const size_t k = dst->src[0]->ne[0];
    GGML_ASSERT(k <= SIZE_MAX / sizeof(float) / 17 / nth);
    const size_t ref = k * 17 * nth * sizeof(float);
    return std::max(ref, exl3_fast_work_size(dst));
}

static void exl3_wht128(float * v) {
    for (int h = 1; h < 128; h <<= 1) {
        for (int i = 0; i < 128; i += 2 * h) {
            for (int j = i; j < i + h; ++j) {
                const float a = v[j], b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
        }
    }
    for (int i = 0; i < 128; ++i) {
        v[i] *= 0.08838834764831845f;
    }
}

static int32_t exl3_expert(const ggml_tensor * dst, int64_t row) {
    if (dst->op != GGML_OP_MUL_MAT_ID) { return 0; }
    const ggml_tensor * ids = dst->src[2];
    int32_t expert;
    memcpy(&expert, (const char *) ids->data + (row % ids->ne[0]) * ids->nb[0] + (row / ids->ne[0]) * ids->nb[1], sizeof(expert));
    GGML_ASSERT(expert >= 0 && expert < dst->src[0]->ne[2]);
    return expert;
}

static const float * exl3_input(const ggml_tensor * dst, int64_t row) {
    const ggml_tensor * x = dst->src[1];
    if (dst->op == GGML_OP_MUL_MAT_ID) {
        const int64_t r = dst->src[2]->ne[0];
        return (const float *) ((const char *) x->data + ((row % r) % x->ne[1]) * x->nb[1] + (row / r) * x->nb[2]);
    }
    return (const float *) ((const char *) x->data + (row % x->ne[1]) * x->nb[1] +
        ((row / x->ne[1]) % x->ne[2]) * x->nb[2] + (row / (x->ne[1] * x->ne[2])) * x->nb[3]);
}

static float * exl3_output(ggml_tensor * dst, int64_t row) {
    return (float *) ((char *) dst->data + (row % dst->ne[1]) * dst->nb[1] +
        ((row / dst->ne[1]) % dst->ne[2]) * dst->nb[2] + (row / (dst->ne[1] * dst->ne[2])) * dst->nb[3]);
}

static float exl3_scale(const ggml_tensor * side, int32_t expert, int64_t i) {
    const char * data = (const char *) side->data + expert * side->nb[1] + i * side->nb[0];
    if (side->type == GGML_TYPE_F32) { return *(const float *) data; }
    return GGML_FP16_TO_FP32(*(const ggml_fp16_t *) data);
}

static void exl3_prepare(const float * x, const ggml_tensor * suh, int32_t expert, int64_t k, int length, float * out) {
    for (int i = 0; i < length; ++i) {
        out[i] = suh ? x[k + i] * exl3_scale(suh, expert, k + i) : x[k + i];
    }
    if (suh) {
        for (int i = 0; i < length; i += 128) { exl3_wht128(out + i); }
    }
}

#if GGML_EXL3_AVX2
struct exl3_window {
    int plane;
    int shift;
};

template<int bits> static constexpr std::array<exl3_window, 32> exl3_windows() {
    std::array<exl3_window, 32> result = {};
    for (int row = 0; row < 16; ++row) {
        for (int half = 0; half < 2; ++half) {
            const int t = 8 * ((row & 7) / 2) + (row & 1) + 2 * (row / 8) + 4 * half;
            const int start = (t + 1) * bits - 16;
            result[row * 2 + half] = {(start + 32) / 32, (start + 32) % 32};
        }
    }
    return result;
}

// Eight columns are 32*bits stream bits apart. Load each word plane once per tile.
template<int bits> struct exl3_tile {
    __m256i word[bits + 1];

    explicit exl3_tile(const void * data) {
        const __m256i offsets = _mm256_setr_epi32(0, bits, 2*bits, 3*bits, 4*bits, 5*bits, 6*bits, 7*bits);
        for (int p = 0; p < bits; ++p) {
            word[p + 1] = _mm256_i32gather_epi32((const int *) data + p, offsets, 4);
        }
        word[0] = _mm256_permutevar8x32_epi32(word[bits], _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6));
    }

    __m256i states(int row, int half) const {
        static constexpr auto windows = exl3_windows<bits>();
        const int p = windows[row * 2 + half].plane;
        const int shift = windows[row * 2 + half].shift;
        __m256i code;
        if (shift <= 16) {
            code = _mm256_srl_epi32(word[p], _mm_cvtsi32_si128(16 - shift));
        } else {
            code = _mm256_or_si256(_mm256_sll_epi32(word[p], _mm_cvtsi32_si128(shift - 16)),
                _mm256_srl_epi32(word[p + 1], _mm_cvtsi32_si128(48 - shift)));
        }
        return _mm256_and_si256(code, _mm256_set1_epi32(0xffff));
    }
};

static inline __m256 exl3_decode(__m256i code) {
    const __m256i x = _mm256_mullo_epi32(code, _mm256_set1_epi32(0x83DCD12Du));
    const __m256i pairs = _mm256_maddubs_epi16(x, _mm256_set1_epi8(1));
    const __m256i sum = _mm256_add_epi32(_mm256_madd_epi16(pairs, _mm256_set1_epi16(1)), _mm256_set1_epi32(0x6400));
    const __m128i h = _mm_packus_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
    const __m256 v = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtph_ps(h), _mm256_set1_ps(GGML_FP16_TO_FP32(0x1eee))),
        _mm256_set1_ps(GGML_FP16_TO_FP32(0xc931)));
    return _mm256_cvtph_ps(_mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

template<int bits> static void exl3_piece(const void * data, int64_t n, int64_t begin, int length,
        const float x[3][512], float * const out[3], int count) {
    for (int a = 0; a < count; ++a) { std::fill(out[a], out[a] + n, 0.0f); }
    const int64_t nt = n / 16;
    const uint32_t * stream = (const uint32_t *) data + (begin / 16) * nt * 8 * bits;
    for (int k = 0; k < length; k += 16) {
        for (int64_t g = 0; g < nt; ++g, stream += 8 * bits) {
            const exl3_tile<bits> tile(stream);
            __m256 acc[3][2];
            for (int a = 0; a < count; ++a) {
                acc[a][0] = _mm256_loadu_ps(out[a] + g * 16);
                acc[a][1] = _mm256_loadu_ps(out[a] + g * 16 + 8);
            }
            for (int r = 0; r < 16; ++r) {
                const __m256 w0 = exl3_decode(tile.states(r, 0));
                const __m256 w1 = exl3_decode(tile.states(r, 1));
                for (int a = 0; a < count; ++a) {
                    const __m256 v = _mm256_set1_ps(x[a][k + r]);
                    acc[a][0] = _mm256_fmadd_ps(v, w0, acc[a][0]);
                    acc[a][1] = _mm256_fmadd_ps(v, w1, acc[a][1]);
                }
            }
            for (int a = 0; a < count; ++a) {
                _mm256_storeu_ps(out[a] + g * 16, acc[a][0]);
                _mm256_storeu_ps(out[a] + g * 16 + 8, acc[a][1]);
            }
        }
    }
}

template<int bits> static void exl3_decode_tile(const void * data, uint16_t * states, float * weights) {
    const exl3_tile<bits> tile(data);
    for (int r = 0; r < 16; ++r) {
        for (int half = 0; half < 2; ++half) {
            const __m256i code = tile.states(r, half);
            uint32_t lanes[8];
            _mm256_storeu_si256((__m256i *) lanes, code);
            for (int c = 0; c < 8; ++c) { states[r * 16 + half * 8 + c] = lanes[c]; }
            _mm256_storeu_ps(weights + r * 16 + half * 8, exl3_decode(code));
        }
    }
}
#endif

bool ggml_cpu_exl3_decode_tile(const void * data, int bits, uint16_t * states, float * weights) {
#if GGML_EXL3_AVX2
    if (!exl3_avx2_usable()) { return false; }
#define EXL3_DECODE_CASE(b) case b: exl3_decode_tile<b>(data, states, weights); return true;
    switch (bits) {
        EXL3_DECODE_CASE(2) EXL3_DECODE_CASE(3) EXL3_DECODE_CASE(4) EXL3_DECODE_CASE(5)
        EXL3_DECODE_CASE(6) EXL3_DECODE_CASE(7) EXL3_DECODE_CASE(8)
    }
#undef EXL3_DECODE_CASE
#else
    GGML_UNUSED(data); GGML_UNUSED(bits); GGML_UNUSED(states); GGML_UNUSED(weights);
#endif
    return false;
}

bool ggml_cpu_exl3_mul_mat(const ggml_compute_params * params, ggml_tensor * dst) {
#if GGML_EXL3_AVX2
    if (!exl3_opt_in() || params->use_ref || !ggml_cpu_exl3_supported(dst) || !exl3_avx2_usable()) { return false; }
    const size_t size = exl3_fast_work_size(dst);
    if (!size) { return false; }
    GGML_ASSERT(params->wsize >= size);
    const ggml_tensor * w = dst->src[0];
    const bool indexed = dst->op == GGML_OP_MUL_MAT_ID;
    const ggml_tensor * suh = dst->src[indexed ? 3 : 2];
    const ggml_tensor * svh = dst->src[indexed ? 4 : 3];
    const int64_t k = w->ne[0], n = w->ne[1], rows = exl3_rows(dst);
    const int length = exl3_piece_size(), batch = exl3_batch_size();
    const int64_t pieces = (k + length - 1) / length;
    char * aligned = (char *) (((uintptr_t) params->wdata + 63) & ~(uintptr_t) 63);
    int64_t * job_count = (int64_t *) aligned;
    exl3_job * jobs = (exl3_job *) (aligned + 64);
    float * partial = (float *) (jobs + rows);
    if (params->ith == 0) {
        for (int64_t row = 0; row < rows; ++row) {
            jobs[row] = {};
            jobs[row].row[0] = row;
            jobs[row].expert = exl3_expert(dst, row);
        }
        std::sort(jobs, jobs + rows, [](const exl3_job & a, const exl3_job & b) {
            return a.expert != b.expert ? a.expert < b.expert : a.row[0] < b.row[0];
        });
        int64_t count = 0;
        for (int64_t row = 0; row < rows;) {
            exl3_job job = {};
            job.expert = jobs[row].expert;
            do { job.row[job.count++] = jobs[row++].row[0]; }
            while (job.count < batch && row < rows && jobs[row].expert == job.expert);
            jobs[count++] = job;
        }
        *job_count = count;
        ggml_threadpool_chunk_set(params->threadpool, 0);
    }
    ggml_barrier(params->threadpool);
    for (;;) {
        const int64_t task = ggml_threadpool_chunk_add(params->threadpool, 1);
        if (task >= *job_count * pieces) { break; }
        const exl3_job & job = jobs[task / pieces];
        const int64_t piece = task % pieces, begin = piece * length;
        const int nk = (int) std::min<int64_t>(length, k - begin);
        float x[3][512];
        float * out[3] = {};
        for (int a = 0; a < job.count; ++a) {
            exl3_prepare(exl3_input(dst, job.row[a]), suh, job.expert, begin, nk, x[a]);
            out[a] = partial + (job.row[a] * pieces + piece) * n;
        }
        const void * data = (const char *) w->data + job.expert * w->nb[2];
#define EXL3_PIECE_CASE(b) case b: exl3_piece<b>(data, n, begin, nk, x, out, job.count); break;
        switch (ggml_exl3_bits(w->type)) {
            EXL3_PIECE_CASE(2) EXL3_PIECE_CASE(3) EXL3_PIECE_CASE(4) EXL3_PIECE_CASE(5)
            EXL3_PIECE_CASE(6) EXL3_PIECE_CASE(7) EXL3_PIECE_CASE(8)
        }
#undef EXL3_PIECE_CASE
    }
    ggml_barrier(params->threadpool);
    // The reduction tree depends only on piece size, never on the worker assignment.
    const int64_t band = svh ? 128 : 16;
    for (int64_t task = params->ith; task < rows * (n / band); task += params->nth) {
        const int64_t row = task / (n / band), begin = (task % (n / band)) * band;
        float * out = exl3_output(dst, row) + begin;
        memcpy(out, partial + row * pieces * n + begin, band * sizeof(float));
        for (int64_t p = 1; p < pieces; ++p) {
            const float * in = partial + (row * pieces + p) * n + begin;
            for (int j = 0; j < band; ++j) { out[j] += in[j]; }
        }
        if (svh) {
            exl3_wht128(out);
            const int32_t expert = exl3_expert(dst, row);
            for (int j = 0; j < 128; ++j) { out[j] *= exl3_scale(svh, expert, begin + j); }
        }
    }
    return true;
#else
    GGML_UNUSED(params); GGML_UNUSED(dst);
    return false;
#endif
}

void ggml_cpu_exl3_mul_mat_ref(const ggml_compute_params * params, ggml_tensor * dst) {
    GGML_ASSERT(exl3_opt_in() && ggml_cpu_exl3_supported(dst));
    const ggml_tensor * w = dst->src[0];
    const bool indexed = dst->op == GGML_OP_MUL_MAT_ID;
    const ggml_tensor * suh = dst->src[indexed ? 3 : 2];
    const ggml_tensor * svh = dst->src[indexed ? 4 : 3];
    const int64_t k = w->ne[0], n = w->ne[1], rows = exl3_rows(dst);
    GGML_ASSERT(params->wsize >= (size_t) params->nth * 17 * k * sizeof(float));
    float * weights = (float *) params->wdata + params->ith * 17 * k;
    float * xbuf = weights + 16 * k;
    for (int64_t task = params->ith; task < rows * (n / 16); task += params->nth) {
        const int64_t row = task / (n / 16), g = task % (n / 16);
        const int32_t expert = exl3_expert(dst, row);
        const float * x = exl3_input(dst, row);
        if (suh) {
            exl3_prepare(x, suh, expert, 0, k, xbuf);
            x = xbuf;
        }
        ggml_exl3_dequantize_row_group((const char *) w->data + expert * w->nb[2], k, n, ggml_exl3_bits(w->type), g, weights);
        float * out = exl3_output(dst, row) + g * 16;
        for (int j = 0; j < 16; ++j) { ggml_vec_dot_f32(k, out + j, 0, weights + j * k, 0, x, 0, 1); }
    }
    if (svh) {
        ggml_barrier(params->threadpool);
        for (int64_t task = params->ith; task < rows * (n / 128); task += params->nth) {
            const int64_t row = task / (n / 128), begin = (task % (n / 128)) * 128;
            float * out = exl3_output(dst, row) + begin;
            exl3_wht128(out);
            const int32_t expert = exl3_expert(dst, row);
            for (int j = 0; j < 128; ++j) { out[j] *= exl3_scale(svh, expert, begin + j); }
        }
    }
}
