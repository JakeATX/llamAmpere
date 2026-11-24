// [SPLITOVL] overlapped verify windows under layer split (LLAMA_SPLIT_OVERLAP=1, default off).
//
// With -sm layer on two cards, card B is idle while card A runs its layers of a verify window and the reverse.
// A caller that knows the next window's tokens before the current one is resolved (it assumes the current window
// is fully accepted and guesses its bonus token) can queue that next window right away: card A starts it while
// card B is still on the current one. The scheduler's pipeline-parallel input copies already keep the two graphs
// apart on the device, so what is missing is on the host side, and this file adds it:
//   - two output sets (logits / embeddings / output ids), so the queued window does not overwrite the rows of the
//     window still being read (decode into slot 0, then slot 1, read slot 0, decode the next window into slot 0);
//   - one event per slot and GPU backend, so reading a slot waits for that window only, not the queued one.
// Recurrent state (GDN / conv) needs no extra copy: the caller widens the rollback ring (n_rs_seq >= 2 windows), and
// a miss is a plain llama_memory_seq_rm back to the last accepted position after llama_synchronize().
// The context refuses (init returns false) unless: the env flag is set, the model is split by layer over 2+ devices,
// the scheduler runs pipeline parallel, and no backend sampler or [D1] head is attached. -sm tensor is left alone.

#include "llama-context.h"
#include "llama-ext.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

// 0/unset = off, 1 = on, 2 = test mode: skip the split/device/scheduler checks so the double-buffered outputs and a
// caller's rollback logic can be checked on one card (no overlap gain there, the windows simply queue on one stream)
static int split_overlap_env() {
    const char * e = getenv("LLAMA_SPLIT_OVERLAP");
    return e != nullptr ? atoi(e) : 0;
}

bool llama_context::split_overlap_init() {
    auto & so = split_ovl;
    if (so.enabled) {
        return true;
    }
    const int flag = split_overlap_env();
    if (flag != 1 && flag != 2) {
        return false;
    }
    const bool test_mode = flag == 2;
    const char * why = nullptr;
    if (test_mode) {
        // checks below the split ones still apply
    } else if (model.split_mode() != LLAMA_SPLIT_MODE_LAYER) {
        why = "needs -sm layer";
    } else if (model.n_devices() < 2) {
        why = "needs 2+ devices";
    } else if (!cparams.pipeline_parallel) {
        why = "needs the pipeline-parallel scheduler (full offload, no tensor overrides, no MoE cache)";
    }
    if (why) {
        // reported below
    } else if (!sampling.samplers.empty()) {
        why = "backend samplers are not supported";
    } else if (d1) {
        why = "the [D1] head is not supported";
    } else if (mtp_pipe.enabled) {
        why = "LLAMA_MTP_PIPELINE is not supported";
    }
    if (why) {
        LLAMA_LOG_WARN("%s: LLAMA_SPLIT_OVERLAP=1 ignored: %s\n", __func__, why);
        return false;
    }
    for (auto & b : backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(b.get());
        if (!dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        so.ev_backend.push_back(b.get());
        for (auto & evs : so.ev) {
            evs.emplace_back(ggml_backend_event_new(dev));
            if (!evs.back()) {
                LLAMA_LOG_WARN("%s: LLAMA_SPLIT_OVERLAP=1 ignored: event creation failed\n", __func__);
                so.ev[0].clear();
                so.ev[1].clear();
                so.ev_backend.clear();
                return false;
            }
        }
    }
    if (so.ev_backend.size() < (test_mode ? 1u : 2u)) {
        LLAMA_LOG_WARN("%s: LLAMA_SPLIT_OVERLAP=%d ignored: too few GPU backends\n", __func__, flag);
        so.ev[0].clear();
        so.ev[1].clear();
        so.ev_backend.clear();
        return false;
    }
    // finish anything queued before the double buffering starts
    synchronize();
    so.enabled = true;
    so.active  = 0;
    LLAMA_LOG_INFO("%s: overlapped verify windows on (LLAMA_SPLIT_OVERLAP=%d%s, %zu GPU backends)\n", __func__,
            flag, test_mode ? " test mode, no split checks" : "", so.ev_backend.size());
    return true;
}

void llama_context::split_overlap_swap_outputs() {
    auto & st = split_ovl.stash;
    std::swap(buf_output,     st.buf_output);
    std::swap(output_file,    st.output_file);
    std::swap(logits,         st.logits);
    std::swap(embd,           st.embd);
    std::swap(embd_nextn,     st.embd_nextn);
    std::swap(embd_layer_inp, st.embd_layer_inp);
    std::swap(output_ids,     st.output_ids);
    std::swap(n_outputs,      st.n_outputs);
}

int32_t llama_context::split_overlap_decode(const llama_batch & batch, int32_t slot) {
    auto & so = split_ovl;
    if (!so.enabled || slot < 0 || slot > 1) {
        return -4;
    }
    if (slot != so.active) {
        // the slot about to be written must not still be in flight (the caller read it, or waits now)
        if (so.recorded[slot]) {
            split_overlap_wait(slot);
        }
        split_overlap_swap_outputs();
        so.active = slot;
    }
    const int32_t ret = llama_decode(this, batch);
    if (ret != 0) {
        return ret;
    }
    if (!output_swaps.empty()) {
        // multi-sequence reorder is host work on the rows: finish this window now (never the case for one sequence)
        synchronize();
        output_reorder();
    }
    for (size_t i = 0; i < so.ev_backend.size(); ++i) {
        ggml_backend_event_record(so.ev[slot][i].get(), so.ev_backend[i]);
    }
    so.recorded[slot] = true;
    so.n_decodes++;
    return 0;
}

bool llama_context::split_overlap_wait(int32_t slot) {
    auto & so = split_ovl;
    if (!so.enabled || slot < 0 || slot > 1) {
        return false;
    }
    if (so.recorded[slot]) {
        for (auto & ev : so.ev[slot]) {
            ggml_backend_event_synchronize(ev.get());
        }
        so.recorded[slot] = false;
    }
    return true;
}

float * llama_context::split_overlap_logits_ith(int32_t slot, int32_t i) {
    auto & so = split_ovl;
    if (!so.enabled || slot < 0 || slot > 1) {
        return nullptr;
    }
    split_overlap_wait(slot);
    const bool live = slot == so.active;
    const buffer_view<float> & lg  = live ? logits     : so.stash.logits;
    const std::vector<int32_t> & ids = live ? output_ids : so.stash.output_ids;
    const uint32_t n_out             = live ? n_outputs  : so.stash.n_outputs;
    if (lg.data == nullptr || i < 0 || (size_t) i >= ids.size()) {
        return nullptr;
    }
    const int32_t j = ids[i];
    if (j < 0 || (uint32_t) j >= n_out) {
        return nullptr;
    }
    return lg.data + (size_t) j * model.vocab.n_tokens();
}

//
// C API
//

bool llama_split_overlap_init(llama_context * ctx) {
    return ctx && ctx->split_overlap_init();
}

int32_t llama_split_overlap_decode(llama_context * ctx, llama_batch batch, int32_t slot) {
    return ctx ? ctx->split_overlap_decode(batch, slot) : -4;
}

bool llama_split_overlap_wait(llama_context * ctx, int32_t slot) {
    return ctx && ctx->split_overlap_wait(slot);
}

float * llama_split_overlap_logits_ith(llama_context * ctx, int32_t slot, int32_t i) {
    return ctx ? ctx->split_overlap_logits_ith(slot, i) : nullptr;
}
