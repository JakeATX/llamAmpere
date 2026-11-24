#pragma once

#include "ggml-backend.h"

#include <cstring>

// fork: CUDA buffer types whose bodies are split: an authoritative pinned host master plus card slots.
// QSA sealed bodies come in two names: "CUDA_QSA_Resident_" (GGML_CUDA_QSA_RESIDENT_CLOCK=0) and "CUDA_QSA_Clock_"
// (the CLOCK pool, the default). A snapshot of their bytes belongs in RAM, not in a second card window.
static inline bool llama_buft_is_qsa_split(ggml_backend_buffer_type_t buft) {
    const char * name = ggml_backend_buft_name(buft);
    return strncmp(name, "CUDA_QSA_Resident_", 18) == 0 || strncmp(name, "CUDA_QSA_Clock_", 15) == 0;
}

// The QSA bodies above plus the LLAMA_LSA_RESIDENT strip/page pools: their memory breakdown is the card share
// (slots + meta) plus the host master.
static inline bool llama_buft_is_split_body(ggml_backend_buffer_type_t buft) {
    const char * name = ggml_backend_buft_name(buft);
    return llama_buft_is_qsa_split(buft) || strncmp(name, "CUDA_LSA_Strip_", 15) == 0 || strncmp(name, "CUDA_LSA_Page_", 14) == 0;
}
