// This file has been hand-created (SJ-KVaRN MMA decode instance). Do NOT run generate_cu_files.py
// over it — that script deletes all *.cu including the turbo VEC instances.

#include "../fattn-mma-f16.cuh"
#include "../fattn-mma-sjkvarn.cuh"

DECL_FATTN_MMA_SJKVARN_CASE(256, 256, 2, 8);
