#include "common.cuh"

// GGML_OP_SJKVARN_SEAL on CUDA: seals rotated F16, Q8_0 or stored-domain TQ6_0 K/V tiles into SJ-KVaRN records.
// Produces records identical to the CPU sealer (ggml-sjkvarn.h): same reduction contract, IEEE float
// division via __fdiv_rn (the build uses -use_fast_math), log domain in double.
void ggml_cuda_sj_kvarn_seal(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Strip-master prep only; no runtime caller yet. Buffers/IDs stay live until ctx.stream() finishes.
// IDs name individual head records, are in range, and are unique for pack. Canonical scratch is device memory.
// Both arenas are disjoint and 16-byte aligned, sized by n_records*canonical_bytes and master_records*master_bytes.
bool ggml_cuda_sj_kvarn_strip_master_copy(ggml_backend_cuda_context & ctx, void * master, size_t master_records,
        void * canonical, size_t n_records, const int32_t * record_ids, int bits_k, int bits_v, bool pack);

// Fixed master_bytes output stride; each mask compacts selected strips in ascending order after shared metadata.
// Device IDs/masks and disjoint aligned arenas stay live until ctx.stream() finishes; publish stage validity after it.
// IDs must be in range (repeats are allowed); an invalid device ID traps, it is not a recoverable host rejection.
bool ggml_cuda_sj_kvarn_strip_master_stage(ggml_backend_cuda_context & ctx, const void * master, size_t master_records,
        void * stage, size_t stage_capacity, size_t n_records, const int32_t * record_ids, const uint8_t * masks,
        int bits_k, int bits_v);

struct ggml_cuda_qsa_clock_map;
// Prefill prep only, no caller yet. Caller joins prefetch and clears stage entries before launch.
// Map and descriptor stay unchanged on ctx.stream(); all arenas are disjoint, aligned and live until it finishes.
// Scratch needs (n_out/64)*heads*canonical_bytes; capacity is per output. The caller must chunk and charge scratch.
// Writes sealed body rows only. Sink and tail rows still use the existing ring expansion.
bool ggml_cuda_sj_kvarn_strip_master_expand(ggml_backend_cuda_context & ctx, const ggml_cuda_qsa_clock_map & map,
        const ggml_tensor * desc, void * scratch, size_t scratch_capacity, half2 * K_out, half2 * V_out, size_t output_capacity,
        int n_out, int p_begin, int heads, int head_first, size_t k_head_stride, size_t v_head_stride);

// Opt-in: seal only new groups into bounded device scratch, then pack to mapped host.
// The descriptor's new physical groups must be unique; descriptor, sources and aligned master stay live on ctx.stream().
// Caller joins prefetch before this call and queues CLOCK refresh/readiness after it. False launches no work.
// Invalid device descriptor data traps after launch; the call does not promise atomic rejection of the batch.
bool ggml_cuda_sj_kvarn_seal_strip_master(ggml_backend_cuda_context & ctx, ggml_tensor * op,
        void * master, size_t master_records, size_t master_capacity);

bool ggml_cuda_sj_kvarn_seal_supported(const ggml_tensor * op);

// trellis body codebooks: every translation unit holding device copies registers their symbols at load time;
// init (once per device, on the first trellis seal or attention) overwrites them with the GGML_SJKVARN_TRELLIS_CB
// file when that override is set (the header defaults otherwise stay in place)
void ggml_cuda_sj_kvarn_trellis_cb_register(const void * sym_k, const void * sym_v);
void ggml_cuda_sj_kvarn_trellis_cb_init();
