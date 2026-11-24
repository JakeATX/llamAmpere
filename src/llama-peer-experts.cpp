// [PEERX] second-card expert tier: llama-side wiring of the GGML_PEER_EXPERTS transport (default off).
//
// The transport (ggml/src/ggml-cuda/peer-experts.cu) moves F32/I32 rows between two cards without P2P through
// pinned, mapped host memory. This file puts expert weights on the second card and makes the graph use them:
//   - create(): picks the host-resident MoE layers (experts in a host weight buffer, the rest of the layer on a GPU),
//     in layer order, until the VRAM cap of the peer card is reached, and copies their expert tensors there (each
//     looped/shared tensor once). The copies sit in a weight buffer of the peer device, so the scheduler runs the
//     MUL_MAT_ID that reads them on the peer; the graph pins the rest of that expert chain there too.
//   - llm_graph_context::build_moe_ffn: a layer on the peer computes there
//       (a) the share of the MoE cache misses that would run on the CPU (the cache's CPU map), built before the
//           cached experts so the main card computes its hits while the peer computes the misses, or
//       (b) all routed experts when the layer is not read through the cache (no cache, or ubatches past its cap);
//     and the main card adds the peer rows (each part reads the other's experts as zero, as with the CPU share).
//   - the CUDA copy hook: the cross-card copies of the expert input, ids and rows go through the mailbox (no host
//     wait on either card). check() turns the hook off after a failed (timed out) wait.
// Flags: GGML_PEER_EXPERTS=1 (on, 2+ GPUs) / 2 (test mode: one GPU is enough, for one-card identity checks),
// LLAMA_PEER_EXPERTS_DEV=<device name or GPU index> (default: the last GPU), LLAMA_PEER_EXPERTS_MIB=<cap> (default:
// free VRAM of the peer minus LLAMA_PEER_EXPERTS_RESERVE_MIB, default 1536), LLAMA_PEER_EXPERTS_MAX_TOKENS=<n>
// (0 = no limit), GGML_PEER_EXPERTS_COPY=0 (driver copies instead of the mailbox hook).
// Not under -sm tensor, not for draft/MTP contexts, not in the fit's dry run (no_alloc).

#include "llama-peer-experts.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <string>

static int peer_env_int(const char * name, int def) {
    const char * e = getenv(name);
    return e && e[0] ? atoi(e) : def;
}

static bool peer_is_host_weight(const ggml_tensor * t) {
    return t != nullptr && t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer) && t->data != nullptr;
}

static bool peer_is_gpu(ggml_backend_t b) {
    ggml_backend_dev_t dev = b ? ggml_backend_get_device(b) : nullptr;
    return dev && (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU ||
                   ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_IGPU);
}

std::unique_ptr<llama_peer_experts> llama_peer_experts::create(const llama_model & model, const std::vector<ggml_backend_t> & backends) {
    const int mode = peer_env_int("GGML_PEER_EXPERTS", 0);
    if (mode != 1 && mode != 2) {
        return nullptr;
    }
    if (model.hparams.n_expert == 0) {
        return nullptr;
    }
    if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        LLAMA_LOG_WARN("%s: GGML_PEER_EXPERTS ignored: not supported with -sm tensor\n", __func__);
        return nullptr;
    }
    if (model.no_alloc()) {
        return nullptr; // the fit's dry run: the tier sizes itself from the free VRAM of the real context
    }

    std::vector<ggml_backend_t> gpus;
    for (ggml_backend_t b : backends) {
        if (peer_is_gpu(b)) {
            gpus.push_back(b);
        }
    }
    if (gpus.size() < (mode == 2 ? 1u : 2u)) {
        LLAMA_LOG_WARN("%s: GGML_PEER_EXPERTS=%d ignored: %zu GPU backend(s) in this context (2 needed; =2 is the one-card test mode)\n",
                __func__, mode, gpus.size());
        return nullptr;
    }
    ggml_backend_t peer = gpus.back();
    if (const char * want = getenv("LLAMA_PEER_EXPERTS_DEV"); want && want[0]) {
        peer = nullptr;
        for (size_t i = 0; i < gpus.size(); ++i) {
            const char * name = ggml_backend_dev_name(ggml_backend_get_device(gpus[i]));
            if (strcmp(name, want) == 0 || (std::isdigit((unsigned char) want[0]) && (size_t) atoi(want) == i)) {
                peer = gpus[i];
            }
        }
        if (!peer) {
            LLAMA_LOG_WARN("%s: GGML_PEER_EXPERTS ignored: LLAMA_PEER_EXPERTS_DEV=%s is not a GPU of this context\n", __func__, want);
            return nullptr;
        }
    }
    ggml_backend_dev_t peer_dev = ggml_backend_get_device(peer);

    // the mailbox copy hook of the CUDA backend
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(peer_dev);
    auto * copy_enable   = (void (*)(bool)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_peer_copy_enable");
    auto * copy_failures = (uint64_t (*)()) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_peer_copy_failures");
    if (!copy_enable || !copy_failures) {
        LLAMA_LOG_WARN("%s: GGML_PEER_EXPERTS ignored: the peer backend has no mailbox transport (CUDA only)\n", __func__);
        return nullptr;
    }

    // the layers whose experts live in host memory while the layer itself runs on a GPU (a layer of the peer card
    // itself just keeps its experts there)
    const size_t n_layer = model.layers.size();
    std::vector<ggml_backend_t> main_of(n_layer, nullptr);
    std::vector<std::vector<ggml_tensor *>> exps(n_layer);
    for (size_t il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
            if (t) {
                exps[il].push_back(t);
            }
        }
        bool ok = !exps[il].empty();
        for (ggml_tensor * t : exps[il]) {
            // whole rows per expert plane, one plane per expert (the copy is made contiguous on the peer)
            ok = ok && peer_is_host_weight(t) && t->ne[3] == 1 && t->nb[1] == ggml_row_size(t->type, t->ne[0]) &&
                t->nb[2] >= t->nb[1] * t->ne[1];
        }
        ggml_backend_dev_t dev = ok ? model.dev_layer((int) il) : nullptr;
        for (ggml_backend_t b : gpus) {
            if (dev && ggml_backend_get_device(b) == dev) {
                main_of[il] = b;
            }
        }
    }

    // VRAM cap of the peer
    size_t cap = 0;
    if (const int mib = peer_env_int("LLAMA_PEER_EXPERTS_MIB", 0); mib > 0) {
        cap = (size_t) mib * 1024 * 1024;
    } else {
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(peer_dev, &free, &total);
        const size_t reserve = (size_t) std::max(0, peer_env_int("LLAMA_PEER_EXPERTS_RESERVE_MIB", 1536)) * 1024 * 1024;
        cap = free > reserve ? free - reserve : 0;
    }

    auto res = std::unique_ptr<llama_peer_experts>(new llama_peer_experts());
    res->peer = peer;
    res->main_of_layer.assign(n_layer, nullptr);
    if (const int n = peer_env_int("LLAMA_PEER_EXPERTS_MAX_TOKENS", 0); n > 0) {
        res->n_tokens_max = n;
    }

    // pick layers in order until the cap (a tensor shared by looped layers counts once)
    std::vector<ggml_tensor *> picked;
    size_t total = 0;
    int n_picked = 0;
    for (size_t il = 0; il < n_layer; ++il) {
        if (!main_of[il]) {
            continue;
        }
        size_t add = 0;
        for (ggml_tensor * t : exps[il]) {
            if (std::find(picked.begin(), picked.end(), t) == picked.end()) {
                add += GGML_PAD(ggml_row_size(t->type, t->ne[0]) * t->ne[1] * t->ne[2], 256);
            }
        }
        if (total + add > cap) {
            break;
        }
        total += add;
        for (ggml_tensor * t : exps[il]) {
            if (std::find(picked.begin(), picked.end(), t) == picked.end()) {
                picked.push_back(t);
            }
        }
        res->main_of_layer[il] = main_of[il];
        if (res->il_first < 0) {
            res->il_first = (int) il;
        }
        n_picked++;
    }
    if (picked.empty()) {
        LLAMA_LOG_WARN("%s: GGML_PEER_EXPERTS: no host expert layer fits on %s (cap %.0f MiB)\n", __func__,
                ggml_backend_dev_name(peer_dev), cap / 1048576.0);
        return nullptr;
    }

    ggml_init_params ip = { (picked.size() + 1) * ggml_tensor_overhead(), nullptr, true };
    res->ctx = ggml_init(ip);
    for (size_t i = 0; i < picked.size(); ++i) {
        const ggml_tensor * w = picked[i];
        ggml_tensor * t = ggml_new_tensor_3d(res->ctx, w->type, w->ne[0], w->ne[1], w->ne[2]);
        ggml_format_name(t, "peerx.%zu", i); // not "*_exps.weight": the scheduler's expert-weight copy paths stay away
        res->copies[w] = t;
    }
    res->buf = ggml_backend_alloc_ctx_tensors_from_buft(res->ctx, ggml_backend_dev_buffer_type(peer_dev));
    if (!res->buf) {
        LLAMA_LOG_WARN("%s: GGML_PEER_EXPERTS: allocating %.0f MiB on %s failed\n", __func__, total / 1048576.0,
                ggml_backend_dev_name(peer_dev));
        return nullptr;
    }
    ggml_backend_buffer_set_usage(res->buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (const auto & [w, t] : res->copies) {
        const size_t plane = t->nb[2];
        if (w->nb[2] == plane) {
            ggml_backend_tensor_set(t, w->data, 0, plane * w->ne[2]);
        } else {
            for (int64_t e = 0; e < w->ne[2]; ++e) {
                ggml_backend_tensor_set(t, (const char *) w->data + e * w->nb[2], e * plane, plane);
            }
        }
    }

    res->copy_enable   = copy_enable;
    res->copy_failures = copy_failures;
    const bool hook = peer_env_int("GGML_PEER_EXPERTS_COPY", 1) != 0;
    if (hook) {
        copy_enable(true);
        res->n_fail_seen = copy_failures();
    }
    // WARN, not INFO: the fit wraps the context creation in a quiet logger, and this opt-in tier must be visible
    LLAMA_LOG_WARN("%s: [PEERX] %d host expert layers (%zu tensors, %.0f MiB) on %s%s; cross-card copies: %s\n", __func__,
            n_picked, picked.size(), total / 1048576.0, ggml_backend_dev_name(peer_dev),
            mode == 2 ? " (test mode)" : "", hook ? "mailbox" : "driver");
    return res;
}

llama_peer_experts::~llama_peer_experts() {
    if (buf) {
        ggml_backend_buffer_free(buf);
    }
    if (ctx) {
        ggml_free(ctx);
    }
}

ggml_backend_t llama_peer_experts::main_backend(int il) const {
    return il >= 0 && (size_t) il < main_of_layer.size() ? main_of_layer[il] : nullptr;
}

ggml_tensor * llama_peer_experts::get(const ggml_tensor * w) const {
    const auto it = w ? copies.find(w) : copies.end();
    return it == copies.end() ? nullptr : it->second;
}

bool llama_peer_experts::has_layer(int il) const {
    return main_backend(il) != nullptr;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_peer_experts::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    if (buf) {
        res[ggml_backend_buffer_get_type(buf)] += ggml_backend_buffer_get_size(buf);
    }
    return res;
}

bool llama_peer_experts::check() {
    if (!copy_failures) {
        return true;
    }
    const uint64_t n = copy_failures();
    if (n == n_fail_seen) {
        return true;
    }
    LLAMA_LOG_ERROR("%s: [PEERX] %llu mailbox wait(s) timed out: those rows were read as zero; cross-card copies use "
            "the driver from now on\n", __func__, (unsigned long long) (n - n_fail_seen));
    n_fail_seen = n;
    copy_enable(false);
    return false;
}
