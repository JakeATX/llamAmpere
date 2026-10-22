#pragma once
// fork STRATAGAP (GGML_MOE_CPU_SPLIT=1, default off): the host CPU computes part of each layer's missed experts
// while the card copies the rest over PCIe. One job per layer covers the whole expert (up, gate, SwiGLU, down), so
// the card pays one handshake per layer instead of one per projection. The card plans the split (moe-cache.cu),
// writes the job and the input rows into mapped host memory and raises `seq`; a pinned pool of spinning host threads
// computes the down-projection rows of those routes into mapped memory and raises `done`; a join kernel on the card
// waits for `done` and writes the rows into the down projection's output. Plain C++: the vector math lives in
// moe-cpu-split.cpp, which the host compiler builds with the CPU flags (AVX2 on the Zen 3 build).
#include <cstddef>
#include <cstdint>

#define MOE_CS_MAX_EXPERTS 16   // CPU experts per layer
#define MOE_CS_MAX_ROUTES  32   // CPU routes (token, expert slot) per layer

// Mapped host memory, written by the card (job fields, seq) and by the host (done, err).
struct moe_cs_job {
    alignas(128) volatile unsigned long long seq;
    alignas(128) volatile unsigned long long done;
    alignas(128) int32_t layer;
    int32_t n_exp;
    int32_t n_route;
    int32_t pad;
    int32_t expert[MOE_CS_MAX_EXPERTS];
    int32_t first[MOE_CS_MAX_EXPERTS + 1];   // routes of expert i are [first[i], first[i+1]) in x / out row order
    alignas(128) volatile unsigned long long gpu_timeouts;
};

// One projection bank of one layer, as the host reads it.
struct moe_cs_role {
    int type = -1;                           // ggml_type of the weights
    int64_t n_in = 0, n_out = 0, n_expert = 0;
    size_t row_bytes = 0;                    // bytes of one output row (n_in weights)
    const void * const * src = nullptr;      // [n_expert] host address of each expert's rows (may be null)
};

// Starts the pool once. x rows are [MOE_CS_MAX_ROUTES][x_stride] floats, out rows [MOE_CS_MAX_ROUTES][out_stride].
bool moe_cs_start(moe_cs_job * job, const float * x, size_t x_stride, float * out, size_t out_stride,
        int64_t max_ff, int64_t max_embd);
bool moe_cs_running();
// Registers (or refreshes) a layer. False when the host cannot compute it (types, shapes).
bool moe_cs_register_layer(int layer, const moe_cs_role roles[3]);
bool moe_cs_layer_ok(int layer);
// Times host jobs on a registered layer (the pool must be idle: no card job in flight). Fills the cost of one
// (expert, route) pair and the fixed cost of a job, in nanoseconds. False when no layer is registered.
bool moe_cs_probe(double * route_ns, double * fixed_ns);
// Statistics: [0] jobs, [1] experts, [2] routes, [3] busy ns, [4] null-source errors.
void moe_cs_stats(unsigned long long out[5]);
int moe_cs_threads();
