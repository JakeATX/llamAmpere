// Targeted Qwen3.8 MTP4 verification instance: five query rows rounded to the
// existing eight-row MMA tile, packed across the model's GQA heads.

#include "../fattn-mma-f16.cuh"
#include "../fattn-mma-turbo.cuh"

DECL_FATTN_MMA_TURBO_CASE(256, 256, 8, 8, GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0);
DECL_FATTN_MMA_TURBO_CASE(256, 256, 8, 8, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);
