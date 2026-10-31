#pragma once
#include "gguf.h"
#include "../ggml/src/ggml-ait.h"
#include <cstring>

// Inspect the stored metadata directly: command-line overrides cannot relabel old bytes.
// Every shard is checked before any weight payload is read or uploaded.
static inline const char * llama_ait_format_error(const gguf_context * meta) {
    bool used[8] = {};
    bool pair_u = false;   // 121 = AIT2P_U: unit bytes, AIT2P pair table
    for (int64_t i = 0; i < gguf_get_n_tensors(meta); ++i) {
        const int type = (int) gguf_get_tensor_type(meta, i);
        if (type >= 110 && type <= 117) { used[type - 110] = true; }
        if (type == 121) { pair_u = true; }
    }
    if (!(used[0] || used[1] || used[2] || used[3] || used[4] || used[5] || used[6] || used[7] || pair_u)) { return nullptr; }
    const int version = gguf_find_key(meta, AIT_FORMAT_VERSION_KEY);
    if (version < 0 || gguf_get_kv_type(meta, version) != GGUF_TYPE_UINT32 ||
        gguf_get_val_u32(meta, version) != AIT_FORMAT_VERSION) {
        return "AIT weights require ait.format_version=2 (uint32); old or unversioned AIT files must be re-encoded with the per-tier constants";
    }
    const char * keys[8] = { "ait.multiplier.110", "ait.multiplier.111", "ait.multiplier.112", nullptr,
                            "ait.multiplier.114", "ait.multiplier.115", "ait.multiplier.116", "ait.multiplier.117" };
    for (int i = 0; i < 8; ++i) {
        if (!used[i] || !keys[i]) { continue; }
        const int key = gguf_find_key(meta, keys[i]);
        if (key < 0 || gguf_get_kv_type(meta, key) != GGUF_TYPE_UINT32 ||
            gguf_get_val_u32(meta, key) != AIT_MUL_FOR_TYPE(110 + i)) {
            return "AIT multiplier metadata is missing or incompatible; re-encode using the format-2 tier constants";
        }
    }
    if (used[3] || pair_u) {
        const int key = gguf_find_key(meta, AIT2P_HASH_KEY);
        if (key < 0 || gguf_get_kv_type(meta, key) != GGUF_TYPE_STRING ||
            std::strcmp(gguf_get_val_str(meta, key), AIT2P_HASH_VERSION) != 0) {
            return "AIT2P / AIT2P_U require the format-2 quadratic20-table1024-v1 hash/table contract";
        }
    }
    return nullptr;
}
