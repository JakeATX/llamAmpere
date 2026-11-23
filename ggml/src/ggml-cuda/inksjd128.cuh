#pragma once

#include "common.cuh"
#include "ggml-inksjd128.h"

// INKPREFILL: 4/4 SJ-KVaRN prefill of a GGML_INK_SJ_D128-marked node (packed D128 heads) as D 128 attention. False = not
// taken (the caller runs its own route). GGML_CUDA_INK_SJ_D128=0 = never taken.
bool ggml_cuda_ink_sj_d128_try(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
