#pragma once

#include "common.cuh"

struct mmq_args;

bool ggml_cuda_should_use_mmq_prefill(const mmq_args & args);
void ggml_cuda_mul_mat_q_prefill(const mmq_args & args, cudaStream_t stream);
