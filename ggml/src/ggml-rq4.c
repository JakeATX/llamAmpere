// RQ4 row quantize / dequantize entry points for ggml (see ggml-rq4.h for the format).

#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"
#include "ggml-rq4.h"

#include <assert.h>

// Dedicated lock for the GPTQ factor cache below: building one entry takes ~0.1 s, too long to hold ggml's
// global critical section.
#if defined(_WIN32)
#include <windows.h>
static SRWLOCK rq4_ulock = SRWLOCK_INIT;
static void rq4_lock(void)   { AcquireSRWLockExclusive(&rq4_ulock); }
static void rq4_unlock(void) { ReleaseSRWLockExclusive(&rq4_ulock); }
#else
#include <pthread.h>
static pthread_mutex_t rq4_ulock = PTHREAD_MUTEX_INITIALIZER;
static void rq4_lock(void)   { pthread_mutex_lock(&rq4_ulock); }
static void rq4_unlock(void) { pthread_mutex_unlock(&rq4_ulock); }
#endif

void quantize_row_rq4_g128_ref(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    assert(k % RQ4_SB == 0);
    rq4_quantize_row(x, (uint8_t *) y, k, 128, NULL);
}

void quantize_row_rq4_g64_ref(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    assert(k % RQ4_SB == 0);
    rq4_quantize_row(x, (uint8_t *) y, k, 64, NULL);
}

void dequantize_row_rq4_g128(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % RQ4_SB == 0);
    rq4_dequantize_row((const uint8_t *) x, y, k, 128);
}

void dequantize_row_rq4_g64(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % RQ4_SB == 0);
    rq4_dequantize_row((const uint8_t *) x, y, k, 64);
}

void quantize_row_rq4nl_g128_ref(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    assert(k % RQ4_SB == 0);
    rq4nl_quantize_row(x, (uint8_t *) y, k, 128, NULL);
}

void quantize_row_rq4nl_g64_ref(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    assert(k % RQ4_SB == 0);
    rq4nl_quantize_row(x, (uint8_t *) y, k, 64, NULL);
}

void dequantize_row_rq4nl_g128(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % RQ4_SB == 0);
    rq4_dequantize_row_g((const uint8_t *) x, y, k, 128, 1);
}

void dequantize_row_rq4nl_g64(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % RQ4_SB == 0);
    rq4_dequantize_row_g((const uint8_t *) x, y, k, 64, 1);
}

// Shared GPTQ factors per imatrix (rq4_gptq_prep_row). llama-quantize gives each worker a few rows per call, so
// building the factors per call would repeat a K/256 x 256^3 setup for every row. Entries are keyed by the
// imatrix contents, reference counted, and only unreferenced entries beyond the newest RQ4_UCACHE_KEEP are freed.
#define RQ4_UCACHE_KEEP 2
typedef struct rq4_uent {
    float           * lam;
    int64_t           K;
    float           * U;
    uint8_t         * ok;
    int               refs;
    struct rq4_uent * next;
} rq4_uent;

static rq4_uent * rq4_ucache = NULL;

static void rq4_ufree(rq4_uent * e) {
    free(e->lam); free(e->U); free(e->ok); free(e);
}

static rq4_uent * rq4_uacquire(const float * imatrix, int64_t K) {
    rq4_lock();
    rq4_uent * e = rq4_ucache;
    for (; e != NULL; e = e->next) {
        if (e->K == K && memcmp(e->lam, imatrix, sizeof(float)*K) == 0) {
            break;
        }
    }
    if (e == NULL) {
        e = (rq4_uent *) calloc(1, sizeof(rq4_uent));
        if (e != NULL) {
            e->K   = K;
            e->lam = (float *) malloc(sizeof(float)*K);
            e->U   = (float *) malloc(sizeof(float)*RQ4_HB*RQ4_HB*(K/RQ4_HB));
            e->ok  = (uint8_t *) malloc(K/RQ4_HB);
            if (e->lam == NULL || e->U == NULL || e->ok == NULL) {
                rq4_ufree(e);
                e = NULL;
            } else {
                memcpy(e->lam, imatrix, sizeof(float)*K);
                rq4_gptq_prep_row(imatrix, K, e->U, e->ok);
                e->next = rq4_ucache;
                rq4_ucache = e;
            }
        }
    }
    if (e != NULL) {
        e->refs++;
    }
    // drop unreferenced entries past the newest RQ4_UCACHE_KEEP
    int n = 0;
    for (rq4_uent ** pp = &rq4_ucache; *pp != NULL; ) {
        rq4_uent * c = *pp;
        if (++n > RQ4_UCACHE_KEEP && c->refs == 0) {
            *pp = c->next;
            rq4_ufree(c);
        } else {
            pp = &c->next;
        }
    }
    rq4_unlock();
    return e;
}

static void rq4_urelease(rq4_uent * e) {
    if (e == NULL) {
        return;
    }
    rq4_lock();
    e->refs--;
    rq4_unlock();
}

static size_t quantize_rq4_impl(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows,
                                int64_t n_per_row, const float * imatrix, int gs, int nl) {
    GGML_ASSERT(n_per_row % RQ4_SB == 0);
    const size_t row_size = (size_t) (n_per_row/RQ4_SB)*rq4_sb_bytes(gs);
    rq4_uent * e = imatrix ? rq4_uacquire(imatrix, n_per_row) : NULL;
    for (int64_t r = 0; r < nrows; ++r) {
        rq4_quantize_row_c(src + r*n_per_row, (uint8_t *) dst + r*row_size, n_per_row, gs, nl, imatrix,
                           e ? e->U : NULL, e ? e->ok : NULL);
    }
    rq4_urelease(e);
    return nrows*row_size;
}

size_t quantize_rq4_g128(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    return quantize_rq4_impl(src, dst, nrows, n_per_row, imatrix, 128, 0);
}

size_t quantize_rq4_g64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    return quantize_rq4_impl(src, dst, nrows, n_per_row, imatrix, 64, 0);
}

size_t quantize_rq4nl_g128(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    return quantize_rq4_impl(src, dst, nrows, n_per_row, imatrix, 128, 1);
}

size_t quantize_rq4nl_g64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    return quantize_rq4_impl(src, dst, nrows, n_per_row, imatrix, 64, 1);
}
