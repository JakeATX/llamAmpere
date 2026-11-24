#pragma once

// fork (ONEGRAPH, backlog #101): GPU-anchored MoE expert-cache lookup for small cards (experts in host RAM plus a VRAM
// expert cache). Off unless LLAMA_MOE_ONEGRAPH=1.
//
// Handshake, per MoE layer visit of a decode graph (ggml_moe_og_slots, CUDA only). The host worker makes no CUDA call:
// a CUDA call can block on a driver lock held by another host thread (lazy kernel loading, allocations) while the GPU
// waits for the worker, a deadlock. The worker only reads and writes pinned host memory; the GPU does the copies.
//   1. the GPU takes the next sequence number from a device counter (ctrl[0]), writes the visit's expert ids to record
//      seq % GGML_MOE_OG_RING of a ring in pinned, mapped host memory, then publishes rec.seq = seq (doorbell)
//   2. it looks the ids up in the device slot map; if every expert is resident it writes the slots and goes on at once
//      (the host worker still reads the record, for the LRU, without holding the GPU)
//   3. on a miss it sets rec.need and spins (volatile loads, %globaltimer) until rec.done == seq. The worker plans the
//      lookup, writes the copies (registered host expert bytes -> device bank slots) and the slot map changes into the
//      record and then stores rec.done = seq. A second kernel of the op does the copies over PCIe, applies the map
//      changes and writes the slots. Past the timeout the GPU stops waiting, stores the sequence number in
//      ring->error and writes slot 0 for a still-missing expert (never out of range); the host aborts on a set error
//      before the next step.
// No host synchronization of the compute stream is needed, so the decode graph keeps no split at the lookup.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_MOE_OG_RING    64   // records in the ring
#define GGML_MOE_OG_MAX_IDS 64   // ids per record: 8 rows x 8 experts (decode and verify widths)
#define GGML_MOE_OG_MAX_COPY 192 // copies per record: 64 experts x 3 expert tensors
#define GGML_MOE_OG_MAX_UPD 256  // slot map changes per record: a fill and an eviction per expert

struct ggml_moe_og_copy {
    uint64_t src;       // device address of registered host memory
    uint64_t dst;       // device address in a bank
    uint64_t bytes;
};

// one doorbell record (written by the GPU, read by the host worker)
struct ggml_moe_og_rec {
    uint64_t seq;       // published last; 0 = never written
    int32_t  layer;     // layer visit the record belongs to (the op's layer parameter)
    int32_t  n_ids;
    int32_t  n_used;    // experts per token
    int32_t  need;      // 1 = the GPU found a miss and waits for done == seq
    int32_t  pad[2];
    int32_t  ids[GGML_MOE_OG_MAX_IDS];
    // response (host), valid once done == seq
    uint64_t done;
    int32_t  n_copy;
    int32_t  n_upd;
    struct ggml_moe_og_copy copy[GGML_MOE_OG_MAX_COPY];
    int32_t  upd[GGML_MOE_OG_MAX_UPD][2]; // {map index row*n_expert + expert, new slot}
};

struct ggml_moe_og_ring {
    uint64_t error;     // first sequence number that timed out (GPU) or broke the protocol (host); 0 = none
    uint64_t last;      // last sequence number the GPU published
    uint64_t pad[6];
    struct ggml_moe_og_rec rec[GGML_MOE_OG_RING];
};

// ctrl tensor (device memory, at least GGML_MOE_OG_CTRL_BYTES), u64 words written by the GPU: [0] sequence counter,
// [1] sequence number of the current lookup, [2] its state (0 all hits, 1 served miss, 2 timed out)
#define GGML_MOE_OG_CTRL_BYTES 64

// Host side: answer record `rec` (the response fields must be written first)
static inline void ggml_moe_og_answer(volatile struct ggml_moe_og_rec * rec, uint64_t seq) {
    __atomic_store_n((uint64_t *) &rec->done, seq, __ATOMIC_RELEASE);
}

// Host side: copy the record with sequence number `next` (or a newer one if the reader fell behind) into *out.
// Returns its sequence number, or 0 if it is not published yet. Seqlock style: the copy is retried if the GPU
// rewrote the record meanwhile.
static inline uint64_t ggml_moe_og_poll(const volatile struct ggml_moe_og_ring * ring, uint64_t next, struct ggml_moe_og_rec * out) {
    const volatile struct ggml_moe_og_rec * rec = &ring->rec[next % GGML_MOE_OG_RING];
    for (int tries = 0; tries < 4; ++tries) {
        const uint64_t s = __atomic_load_n((const uint64_t *) &rec->seq, __ATOMIC_ACQUIRE);
        if (s < next) {
            // the reader may have fallen a whole ring behind: the newest record is ring->last
            const uint64_t last = __atomic_load_n((const uint64_t *) &ring->last, __ATOMIC_ACQUIRE);
            if (last >= next + GGML_MOE_OG_RING) {
                next = last;
                rec  = &ring->rec[next % GGML_MOE_OG_RING];
                continue;
            }
            return 0;
        }
        out->layer  = rec->layer;
        out->n_ids  = rec->n_ids;
        out->n_used = rec->n_used;
        out->need   = rec->need;
        const int n = out->n_ids < 0 ? 0 : out->n_ids > GGML_MOE_OG_MAX_IDS ? GGML_MOE_OG_MAX_IDS : out->n_ids;
        for (int i = 0; i < n; ++i) {
            out->ids[i] = rec->ids[i];
        }
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        if (__atomic_load_n((const uint64_t *) &rec->seq, __ATOMIC_ACQUIRE) == s) {
            out->seq = s;
            return s;
        }
    }
    return 0;
}

#ifdef __cplusplus
}
#endif
