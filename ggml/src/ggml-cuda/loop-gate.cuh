#include "common.cuh"

// GGML_OP_LOOP_GATE on CUDA (fork, Inkling 2 loop gate; see ggml.h): 2 kernels, the rank-R down products with the two
// sums of squares, then the gain/forget products and the gate.
void ggml_cuda_loop_gate(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_loop_gate_supported(const ggml_tensor * op);

// fork: any width; rows past LOOP_GATE_MAX_T take the prompt-width kernels (inklg_pp.cu)
bool ggml_cuda_loop_gate_op_supported(const ggml_tensor * op);

// fork LGSPLIT: G gates (1..4, one row width, the same T) of a verify batch past the decode kernels' 8 tokens (several users:
// 2 x 5 tokens) as balanced chunks of <= 8 tokens (10 -> 5 + 5), run(ctx, chunk gates, G) on each. Shallow tensor copies
// with the chunk's rows of x, m and dst; dst must be contiguous. false (nothing launched) when off, out of range or when
// run refuses the first chunk. GGML_CUDA_LG_SPLIT_MAXT=N: widest batch split (port: opt-in, default 0 = off: prompt-width
// kernel; 32 = the tested value). The op only reaches T > 8 with LLAMA_INKLING2_GATE_PP=1.
typedef bool (*ggml_cuda_loop_gate_run_t)(ggml_backend_cuda_context & ctx, ggml_tensor * const * gates, int G);
bool ggml_cuda_loop_gate_split(ggml_backend_cuda_context & ctx, ggml_tensor * const * gates, int G, ggml_cuda_loop_gate_run_t run);
