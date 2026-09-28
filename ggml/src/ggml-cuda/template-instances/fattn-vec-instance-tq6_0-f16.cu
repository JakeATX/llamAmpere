// Mixed KV: tq6 K + f16 V
// tq6 blocks hold 128 values, so only head dims that are multiples of 128 are instantiated.

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(128, GGML_TYPE_TQ6_0, GGML_TYPE_F16);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_TQ6_0, GGML_TYPE_F16);
