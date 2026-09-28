// Mixed KV: tq5 K + q8_0 V
// tq5 blocks hold 128 values, so only head dims that are multiples of 128 are instantiated.

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(128, GGML_TYPE_TQ5_0, GGML_TYPE_Q8_0);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_TQ5_0, GGML_TYPE_Q8_0);
