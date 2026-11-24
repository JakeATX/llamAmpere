#pragma once

#include "ggml-backend.h"

#include <cstdint>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

struct llama_model;
struct ggml_context;

// [PEERX] second-card expert tier (GGML_PEER_EXPERTS=1, default off; see llama-peer-experts.cpp).
// A second GPU of the context keeps full copies of host-resident expert tensors, layer by layer up to a VRAM cap.
// The graph then computes on that card the routed experts the main card lacks: the share of the MoE cache misses
// that would run on the CPU (--moe-cache-cpu-share), and every routed expert of a host layer that is not read through
// the cache. Activations, ids and expert rows cross the cards through the no-P2P mailbox transport (peer-experts.cu)
// via the CUDA backend's copy hook. Refused under -sm tensor (nothing changes there).
class llama_peer_experts {
public:
    // nullptr unless GGML_PEER_EXPERTS is 1 (or 2 = one-card test mode: the peer may be the main card) and a peer
    // device with room for at least one layer is among the backends
    static std::unique_ptr<llama_peer_experts> create(const llama_model & model, const std::vector<ggml_backend_t> & backends);

    ~llama_peer_experts();

    ggml_backend_t backend() const { return peer; }

    // the backend that computes the rest of layer il (where the peer rows are merged)
    ggml_backend_t main_backend(int il) const;

    // the peer copy of expert tensor w, nullptr if w is not on the peer
    ggml_tensor * get(const ggml_tensor * w) const;

    // every expert tensor of layer il is on the peer
    bool has_layer(int il) const;

    // ubatches with more tokens keep the default path (LLAMA_PEER_EXPERTS_MAX_TOKENS, 0 = no limit)
    int64_t max_tokens() const { return n_tokens_max; }

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    // after a synchronize: logs and turns the mailbox copy hook off (driver copies from then on) if a mailbox wait
    // failed since the last call; the decode that hit it may have read zero rows. false = a failure was seen.
    bool check();

private:
    llama_peer_experts() = default;

    ggml_backend_t peer = nullptr;
    std::vector<ggml_backend_t> main_of_layer;      // per layer, nullptr = not on the peer
    std::unordered_map<const ggml_tensor *, ggml_tensor *> copies;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    int64_t n_tokens_max = INT64_MAX;

    void     (*copy_enable)(bool)  = nullptr;
    uint64_t (*copy_failures)()    = nullptr;
    uint64_t n_fail_seen = 0;
};
