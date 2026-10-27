#pragma once
// fork MOEGAP (GGML_MOE_CPU_SPLIT=1, default off): the host CPU computes part of each layer's missed experts
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
    int32_t input_row[MOE_CS_MAX_ROUTES];    // equal IDs mean identical input rows within this job
    alignas(128) volatile unsigned long long gpu_timeouts;
    alignas(128) volatile unsigned long long busy_ns;   // host time of the last card job, written before `done`
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
// An optional waiting host thread may drain math tiles without joining the pool's barriers.
bool moe_cs_help_enabled();
bool moe_cs_help();
// Registers (or refreshes) a layer. False when the host cannot compute it (types, shapes).
bool moe_cs_register_layer(int layer, const moe_cs_role roles[3]);
bool moe_cs_layer_ok(int layer);
// Times host jobs on a registered layer (the pool must be idle: no card job in flight). Fills the cost of one
// (expert, route) pair and the fixed cost of a job, in nanoseconds. False when no layer is registered.
bool moe_cs_probe(double * route_ns, double * fixed_ns);
// Bench (tests only, pool idle): runs `reps` host jobs of `n_exp` experts (one route each) on `layer`, taking the
// experts in turn from `experts[0..n_list)`. Fills the mean ns per job and the mean ns per phase.
bool moe_cs_bench(int layer, const int * experts, int n_list, int n_exp, int reps, double * ns_per_job,
        double phase_ns[4]);
// Statistics: [0] jobs, [1] experts, [2] routes, [3] busy ns, [4] null-source errors, [5..8] ns per phase.
void moe_cs_stats(unsigned long long out[9]);
// Tests: the pool's IQ row kernel (moe-iq-rows.h) for `type` and nt = 1..8 tokens on rows [r0, r1): out[t][r - r0].
// y[t] are Q8_K rows; variant as in moe_iq::get. False when the build or the type has none.
bool moe_cs_iq_rows(int type, int nt, const void * w, size_t row_bytes, int64_t r0, int64_t r1, int64_t n_in,
        const void * const * y, float * const * out, int prefetch, int variant, bool wide = false);
// Tests: unpack slice `c` (0..7, 32 weights) of one weight block of `type` the way the IQ row kernels do:
// magnitudes (u8), signs (+1 / -1) and the two odd multipliers (weights 0..15, 16..31). `joined` selects the IQ3 grid
// arm. False when the build or the type has no row kernel.
bool moe_cs_iq_unpack(int type, const void * block, int c, bool joined, unsigned char * mag, signed char * sgn, int * mul);
int moe_cs_threads();
