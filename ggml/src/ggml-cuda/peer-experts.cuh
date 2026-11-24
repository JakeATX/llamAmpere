#pragma once

// Peer expert tier transport (GGML_PEER_EXPERTS, default OFF: nothing in the default paths calls it).
// Two GeForce cards without P2P exchange expert activation / output rows through pinned, portable, mapped host
// memory. Each mailbox slot has two directions (0 = rows to the peer, 1 = rows back to the main card); every
// direction has a data area, a doorbell word (sequence number, written after the data with a system fence) and a
// status word. A shared abort word stops every waiter. Waits are bounded on the device clock: on a timeout or an
// abort the waiter writes zeros, stores its sequence number in the status word and the caller takes the normal
// miss path for that step. Reached through ggml_backend_reg_get_proc_address (names below), so the exported
// ggml API does not change.

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>

struct ggml_cuda_peer_mbox;

// "ggml_backend_cuda_peer_mbox_new": n_slots slots, each with bytes_in (dir 0) and bytes_out (dir 1) of row data
typedef ggml_cuda_peer_mbox * (*ggml_cuda_peer_mbox_new_t)(int n_slots, size_t bytes_in, size_t bytes_out);
// "ggml_backend_cuda_peer_mbox_free"
typedef void (*ggml_cuda_peer_mbox_free_t)(ggml_cuda_peer_mbox * mb);
// "ggml_backend_cuda_peer_post": on the stream of backend, copy the contiguous F32 or I32 device tensor src (32-bit words, bit-exact) into the
// data area of (slot, dir), then ring the doorbell with seq. false: bad arguments or src larger than the area.
typedef bool (*ggml_cuda_peer_post_t)(ggml_backend_t backend, ggml_cuda_peer_mbox * mb, int slot, int dir,
        const ggml_tensor * src, uint32_t seq);
// "ggml_backend_cuda_peer_wait": on the stream of backend, wait until the doorbell of (slot, dir) reaches seq
// (bounded by timeout_ns, or the abort word), then copy the data area into the contiguous F32 or I32 device tensor dst.
// On a timeout or abort dst is zero and the status word of (slot, dir) holds seq.
typedef bool (*ggml_cuda_peer_wait_t)(ggml_backend_t backend, ggml_cuda_peer_mbox * mb, int slot, int dir,
        ggml_tensor * dst, uint32_t seq, uint64_t timeout_ns);
// "ggml_backend_cuda_peer_status": host read of the status word of (slot, dir); equal to seq = that wait failed
typedef uint32_t (*ggml_cuda_peer_status_t)(const ggml_cuda_peer_mbox * mb, int slot, int dir);
// "ggml_backend_cuda_peer_abort": host store of the abort word (non-zero releases every waiter with a failure)
typedef void (*ggml_cuda_peer_abort_t)(ggml_cuda_peer_mbox * mb, uint32_t value);

ggml_cuda_peer_mbox * ggml_backend_cuda_peer_mbox_new(int n_slots, size_t bytes_in, size_t bytes_out);
void     ggml_backend_cuda_peer_mbox_free(ggml_cuda_peer_mbox * mb);
bool     ggml_backend_cuda_peer_post(ggml_backend_t backend, ggml_cuda_peer_mbox * mb, int slot, int dir,
        const ggml_tensor * src, uint32_t seq);
bool     ggml_backend_cuda_peer_wait(ggml_backend_t backend, ggml_cuda_peer_mbox * mb, int slot, int dir,
        ggml_tensor * dst, uint32_t seq, uint64_t timeout_ns);
uint32_t ggml_backend_cuda_peer_status(const ggml_cuda_peer_mbox * mb, int slot, int dir);
void     ggml_backend_cuda_peer_abort(ggml_cuda_peer_mbox * mb, uint32_t value);
