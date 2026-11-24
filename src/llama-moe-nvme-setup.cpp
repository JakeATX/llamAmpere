#include "llama-moe-nvme.h"

// fork (NVMETIER, backlog #161): registration of the mapped host experts with the NVMe tier (see llama-moe-nvme.h).

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-model-loader.h"
#include "llama-mmap.h"

#include <cstdlib>
#include <cstring>

#if defined(__linux__)
#include <fcntl.h>
#endif

bool llama_moe_nvme_is_expert_name(const char * name) {
    static const char * suffixes[] = {
        ".ffn_gate_exps.weight", ".ffn_up_exps.weight", ".ffn_down_exps.weight", ".ffn_gate_up_exps.weight" };
    const size_t length = strlen(name);
    for (const char * suffix : suffixes) {
        const size_t n = strlen(suffix);
        if (length >= n && strcmp(name + length - n, suffix) == 0) {
            return true;
        }
    }
    return false;
}

void llama_moe_nvme_setup(const llama_model & model, llama_model_loader & ml) {
    if (!llama_moe_nvme_enabled()) {
        return;
    }
    if (!ml.use_mmap || ml.mappings.empty()) {
        LLAMA_LOG_WARN("%s: [NVMETIER] the model is not mapped, the NVMe tier is off\n", __func__);
        return;
    }
    llama_moe_nvme_tier * tier = llama_moe_nvme_global_create();
    if (tier == nullptr) {
        return;
    }
    const char * drop_env = getenv("LLAMA_MOE_NVME_DROP");
    const bool drop = drop_env == nullptr || strcmp(drop_env, "0") != 0;
    size_t n = 0, bytes = 0, skipped = 0;
    for (const auto & layer : model.layers) {
        for (const ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
            if (t == nullptr || t->buffer == nullptr || t->data == nullptr || t->view_src != nullptr || t->extra != nullptr ||
                    !ggml_backend_buffer_is_host(t->buffer) || !ggml_is_contiguous(t) || t->ne[3] != 1) {
                continue;
            }
            const auto it = ml.weights_map.find(ggml_get_name(t));
            if (it == ml.weights_map.end() || it->second.idx >= ml.mappings.size() || !ml.mappings[it->second.idx]) {
                continue;
            }
            const auto & w = it->second;
            // only tensors that read the mapped file bytes (not a copy: arena, repack, private remap)
            if ((const uint8_t *) t->data != (const uint8_t *) ml.mappings[w.idx]->addr() + w.offs ||
                    t->nb[2]*(size_t) t->ne[2] != ggml_nbytes(t)) {
                skipped++;
                continue;
            }
            const int fd = ml.files.at(w.idx)->file_id();
            if (tier->add(t->data, fd, w.offs, t->nb[2], t->ne[2], &model)) {
                n++;
                bytes += ggml_nbytes(t);
#if defined(__linux__)
                if (drop) {
                    // a cold start: the experts are read from the file by the tier, not from the page cache
                    posix_fadvise(fd, (off_t) w.offs, (off_t) ggml_nbytes(t), POSIX_FADV_DONTNEED);
                }
#endif
            }
        }
    }
    if (n == 0) {
        LLAMA_LOG_WARN("%s: [NVMETIER] no mapped host expert tensor (%zu skipped), the NVMe tier is off\n", __func__, skipped);
        llama_moe_nvme_global_release(&model);
        return;
    }
    const llama_moe_nvme_stats st = tier->stats();
    LLAMA_LOG_INFO("%s: [NVMETIER] %zu expert tensors, %.2f MiB on NVMe behind a %.0f MiB RAM tier (%s page cache)\n",
            __func__, n, bytes/1048576.0, st.budget/1048576.0, drop ? "dropped" : "kept");
}
