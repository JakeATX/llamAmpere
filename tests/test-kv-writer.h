#pragma once

// Test-side replacement for the removed llama_model_saver(arch, gguf_ctx) constructor (upstream b12274 keeps only
// llama_model_saver(model)). Writes arch-resolved GGUF keys into a fixture context, like the kv_writer of
// test-llama-archs.cpp. The context is not owned.

#include "gguf.h"
#include "ggml.h"
#include "../src/llama-arch.h"
#include "../src/llama-ext.h" // llama_arch_kv_name

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

struct test_kv_writer {
    struct gguf_context * gguf_ctx;
    const llm_arch        arch;

    test_kv_writer(enum llm_arch arch, struct gguf_context * gguf_ctx) : gguf_ctx(gguf_ctx), arch(arch) {}

    std::string name(enum llm_kv key) const { return llama_arch_kv_name(arch, key); }

    void add_kv(enum llm_kv key, uint32_t     value) { gguf_set_val_u32 (gguf_ctx, name(key).c_str(), value); }
    void add_kv(enum llm_kv key, int32_t      value) { gguf_set_val_i32 (gguf_ctx, name(key).c_str(), value); }
    void add_kv(enum llm_kv key, uint64_t     value) { gguf_set_val_u64 (gguf_ctx, name(key).c_str(), value); }
    void add_kv(enum llm_kv key, float        value) { gguf_set_val_f32 (gguf_ctx, name(key).c_str(), value); }
    void add_kv(enum llm_kv key, bool         value) { gguf_set_val_bool(gguf_ctx, name(key).c_str(), value); }
    void add_kv(enum llm_kv key, const char * value) { gguf_set_val_str (gguf_ctx, name(key).c_str(), value); }

    void add_kv(enum llm_kv key, const std::vector<std::string> & value) {
        std::vector<const char *> tmp(value.size());
        for (size_t i = 0; i < value.size(); ++i) {
            tmp[i] = value[i].c_str();
        }
        gguf_set_arr_str(gguf_ctx, name(key).c_str(), tmp.data(), tmp.size());
    }

    // arrays (std::vector / std::array of a numeric type), written whole like the old saver without per_layer
    template <typename Container>
    void add_kv(enum llm_kv key, const Container & value) {
        using T = typename Container::value_type;
        if (value.size() == 0) {
            return;
        }
        enum gguf_type type;
        if      (std::is_same<T, uint8_t >::value) { type = GGUF_TYPE_UINT8;   }
        else if (std::is_same<T, int8_t  >::value) { type = GGUF_TYPE_INT8;    }
        else if (std::is_same<T, uint32_t>::value) { type = GGUF_TYPE_UINT32;  }
        else if (std::is_same<T, int32_t >::value) { type = GGUF_TYPE_INT32;   }
        else if (std::is_same<T, uint64_t>::value) { type = GGUF_TYPE_UINT64;  }
        else if (std::is_same<T, float   >::value) { type = GGUF_TYPE_FLOAT32; }
        else if (std::is_same<T, bool    >::value) { type = GGUF_TYPE_BOOL;    }
        else { GGML_ABORT("test_kv_writer: unsupported array type"); }
        gguf_set_arr_data(gguf_ctx, name(key).c_str(), type, value.data(), value.size());
    }
};
