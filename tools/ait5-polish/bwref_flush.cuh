#pragma once

struct bwref_ctx;

bwref_ctx * bwref_init(int device);
void        bwref_free(bwref_ctx * ctx);
void        bwref_flush_l2(bwref_ctx * ctx);
double      bwref_measure_gbps(bwref_ctx * ctx, int iters);
void        bwref_print_device(int device);

// Exact GPU cudaEvent timing on the specified cudaStream_t (passed as void* so callers stay pure C++).
void        bwref_timer_start(bwref_ctx * ctx, void * cuda_stream);
double      bwref_timer_stop_us(bwref_ctx * ctx, void * cuda_stream);
