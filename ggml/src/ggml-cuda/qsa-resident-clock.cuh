#pragma once

#include "common.cuh"

// Four-token pages keep the scalar codec's record metadata resident without changing its bytes.
constexpr int GGML_QSA_PAGE_CELLS = 4;
constexpr int GGML_QSA_PAGE_HEAD_BYTES = 1048;
constexpr int GGML_QSA_META_HEAD_BYTES = 1536;
// LLAMA_LSA_RESIDENT (opt-in): 16-cell strip slots in the scalar 4/4 record's own byte order (K strip 2,048, V strip
// 2,048, then the strip's k_tok, v_scale, v_zero halves, 32 each), i.e. the exact image the SJ-KVaRN stream kernel
// stages per strip, so a resident strip is read from its slot with the same bytes and the same math.
constexpr int GGML_LSA_STRIP_CELLS = 16;
constexpr int GGML_LSA_STRIP_HEAD_BYTES = 4192;
// LLAMA_LSA_RESIDENT on low-bit bodies (3/3t, 3/2t, any non-4/4 SJ-KVaRN record): a slot holds one whole 128-cell record
// of every KV head in the body's own byte order. CLOCK slots stay whole records; opt-in compact overflow strips keep
// the token-axis history bits separately. The low-bit kernels read a resident record from its slot, else the stage/master.
constexpr int GGML_LSA_PAGE_CELLS = 128;

struct ggml_cuda_qsa_clock_map {
    const char * host;
    char * card;
    char * meta;
    int32_t * table;
    int32_t * tags;
    uint32_t * stamp;
    int32_t * ref;
    int32_t * miss_page;
    int32_t * miss_slot;
    uint32_t * ctl; // epoch, CLOCK hand, current misses, active, overflow; u64 totals at 8,10,12
    uint32_t slots;
    uint32_t records;
    uint32_t group;
    uint32_t heads;
    size_t stride;
    uint32_t cells = GGML_QSA_PAGE_CELLS; // cells per page: 4 (QSA CLOCK pages), 16 (LSA strip slots), 128 (LSA record pages)
    int32_t layer = -1;                   // LSA strip mode: layer id for the counter log
    uint32_t log_every = 0;               // LSA strip mode: device log every N resolves (GGML_CUDA_LSA_RESIDENT_LOG)
    uint32_t * seen = nullptr;            // LSA record pages, GGML_CUDA_LSA_RESIDENT_ADMIT=1: per record, last own resolve that picked it
    // LSA record pages, GGML_CUDA_LSA_RESIDENT_STAGE=<records> (opt-in): one on-card staging area per buffer, shared by
    // its layers (they run one after another on one stream). A picked record that gets no CLOCK slot is copied into stage
    // index k for this resolve only (table entry -3 - k) instead of being read by the attention kernel over the bus;
    // the next resolve of the layer first puts those entries back to -1. stage_ctl: 0 count this resolve, u64 at 2 staged
    // total, u64 at 4 records that found the stage full (read from the master as before).
    char * stage = nullptr;
    int32_t * stage_page = nullptr;
    uint32_t * stage_ctl = nullptr;
    uint32_t stage_n = 0;
    uint32_t stage_all = 0; // decode must not read the host master
    size_t stage_stride = 0; // compact token-axis strips; zero keeps whole records
    uint32_t * stage_mask = nullptr;  // needed strips by master record, union over rows/heads
    uint32_t * stage_valid = nullptr; // copied strips by stage index and head
    unsigned long long * stage_stats = nullptr; // exact bytes, strips (all heads), records
    unsigned long long * overflow_hist = nullptr; // diagnostic: staged records with 0..8 live strips, union over rows/heads
    // No publisher yet. Bind only after all host/state/prefill paths support the opt-in master; keep its map immutable in flight.
    const char * strip_master = nullptr; // opt-in source only; CLOCK slots stay canonical
    size_t strip_master_capacity = 0;
    uint32_t strip_master_bits = 0; // K width << 8 | V width
};

bool ggml_cuda_qsa_clock_is_buft(ggml_backend_buffer_type_t buft);
ggml_backend_buffer_type_t ggml_backend_cuda_qsa_clock_buffer_type(ggml_backend_dev_t dev, size_t stride, uint32_t slots);
size_t ggml_backend_cuda_qsa_clock_device_bytes(ggml_backend_buffer_type_t buft, const ggml_tensor * body);
size_t ggml_backend_cuda_qsa_clock_shared_bytes(ggml_backend_buffer_type_t buft, const ggml_context * ctx);
bool ggml_cuda_lsa_page_full_stage(const ggml_tensor * body);
const char * ggml_cuda_qsa_clock_body(const ggml_tensor * body);
void ggml_cuda_qsa_clock_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);
void ggml_cuda_qsa_clock_refresh(ggml_backend_cuda_context & ctx, const ggml_tensor * body);
// False only for a canonical body; an active master must seal and refresh without a canonical fallback.
bool ggml_cuda_qsa_clock_seal_strip_master(ggml_backend_cuda_context & ctx, ggml_tensor * body);
// Host map only. The caller joins prefetch and clears stage entries with ggml_cuda_lsa_page_prepare before use.
const ggml_cuda_qsa_clock_map * ggml_cuda_qsa_clock_strip_master_map(ggml_backend_cuda_context & ctx, const ggml_tensor * body);
// LLAMA_LSA_RESIDENT (opt-in): same buffer (pinned host master, card meta, CLOCK slots) with 16-cell strip slots.
// Its body pointer carries tag 2: the SJ-KVaRN stream kernel reads resident strips from their slots and every other
// strip (and every other kernel) straight from the pinned master, so a strip that is not resident is never wrong.
ggml_backend_buffer_type_t ggml_backend_cuda_qsa_strip_buffer_type(ggml_backend_dev_t dev, size_t stride, uint32_t slots);
bool ggml_cuda_qsa_strip_body(const ggml_tensor * body);
// LLAMA_LSA_RESIDENT on low-bit bodies: same buffer with whole-record (128-cell) slots and no card meta. The body pointer
// (ggml_cuda_lsa_page_body) carries tag 2 like the strip map; ggml_cuda_lsa_page_prepare marks the records the decode
// mask keeps, gives CLOCK victims to the missing ones and copies them in before the low-bit attention kernel.
ggml_backend_buffer_type_t ggml_backend_cuda_qsa_page_buffer_type(ggml_backend_dev_t dev, size_t stride, uint32_t slots, uint32_t heads);
const char * ggml_cuda_lsa_page_body(const ggml_tensor * body); // tagged map for a page body, else body->data
void ggml_cuda_lsa_page_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);
// LLAMA_LSA_RESIDENT_STICKY: card record -> slot table of a page body (slot >= 0 = on the card), nullptr if the body
// has no record-page map; *records = table length
const int32_t * ggml_cuda_lsa_page_table(const ggml_tensor * body, uint32_t * records);
// resolve the strips of a live-record list (sj_live_record_list, entries g | strip mask << 24) and copy the misses
void ggml_cuda_qsa_strip_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * body, const int32_t * desc,
        const int32_t * list);
