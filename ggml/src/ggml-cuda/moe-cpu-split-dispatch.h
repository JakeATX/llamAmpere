#pragma once
#include "moe-cpu-split.h"
#define GGML_COMMON_DECL_CPP
#include "../ggml-common.h"

using moe_cs_rows_fn = void (*)(const char *, size_t, int64_t, int64_t, int, const block_q8_K * const *, float * const *, int);
using moe_cs_q2_rows_fn = void (*)(int, int, const char *, size_t, const block_q8_0 *, size_t, const float *, const float *, size_t, float *, size_t, int);

// fork: only math has ISA copies; the worker pool and its state have one baseline build.
struct moe_cs_kernels {
    const char * (*ai_decode)();
    bool (*ai_r8_ok)(int, uint32_t, int64_t, int64_t);
    void (*ai_rotate)(const float *, void *, int64_t);
    void (*ai_rotate_q8)(const float *, void *, int64_t);
    void (*ai_f32)(const char *, size_t, int, uint32_t, int64_t, int64_t, int64_t, int, const float * const *, float * const *);
    void (*ai_q8)(const char *, size_t, int, uint32_t, int64_t, int64_t, int64_t, int, const void * const *, float * const *);
    void (*ai_keep)(const char *, size_t, int, uint32_t, int64_t, int64_t, int64_t, int, const void * const *, float * const *, uint8_t *);
    void (*ai_r8)(const uint8_t *, int64_t, int64_t, int64_t, int, const void * const *, float * const *);
    void (*q2_dot)(int, float *, const void *, const void *);
    void (*q2_reorder)(int, block_q8_0 *, float *, float *, bool);
    moe_cs_q2_rows_fn q2_rows[5];
    moe_cs_rows_fn (*iq_get)(int, int, int, bool);
    bool (*iq_unpack)(int, const void *, int, bool, unsigned char *, signed char *, int *);
};

extern const moe_cs_kernels moe_cs_scalar_kernels;
extern const moe_cs_kernels moe_cs_avx2_kernels;
const moe_cs_kernels & moe_cs_get_kernels();
