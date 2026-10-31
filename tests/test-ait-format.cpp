// Format-version and independent exhaustive CPU level checks. No GPU needed.
#include "../src/llama-ait-format.h"
#include "ggml.h"
#include "../ggml/src/ggml-ai4.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <initializer_list>

static int checks = 0;
static void require(bool ok, const char * message) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL %s\n", message); std::exit(1); }
}
static gguf_context * fixture(int type) {
    auto * meta = gguf_init_empty();
    if (type >= 0) {
        ggml_init_params params = {ggml_tensor_overhead(), nullptr, true};
        auto * ctx = ggml_init(params);
        auto * tensor = ggml_new_tensor_2d(ctx, (ggml_type) type, 1024, 1024);
        ggml_set_name(tensor, "test.weight");
        gguf_add_tensor(meta, tensor);
        ggml_free(ctx);
    }
    return meta;
}
static void stamp(gguf_context * meta) {
    gguf_set_val_u32(meta, AIT_FORMAT_VERSION_KEY, 2);
    gguf_set_val_u32(meta, "ait.multiplier.110", 0xC3CCD12Du);
    gguf_set_val_u32(meta, "ait.multiplier.111", 0x87D8D12Du);
    gguf_set_val_u32(meta, "ait.multiplier.112", 0xAA73876Bu);
    gguf_set_val_u32(meta, "ait.multiplier.114", 0xAA73876Bu);
    gguf_set_val_u32(meta, "ait.multiplier.115", 0xC3CCD12Du);
    gguf_set_val_u32(meta, "ait.multiplier.116", 0x87D8D12Du);
    gguf_set_val_u32(meta, "ait.multiplier.117", 0xAA73876Bu);
    gguf_set_val_str(meta, AIT2P_HASH_KEY, "quadratic20-table1024-v1");
}
int main() {
    const uint32_t multipliers[3] = {0xC3CCD12Du, 0x87D8D12Du, 0xAA73876Bu};
    const char * keys[3] = {"ait.multiplier.110", "ait.multiplier.111", "ait.multiplier.112"};
    for (int k = 2; k <= 4; ++k) {
        require(AIT_MUL_FOR_TYPE(108+k) == multipliers[k-2], "fixed type multiplier");
        for (uint32_t state = 0; state < 65536; ++state) {
            uint32_t value = (uint32_t) ((uint64_t) state * multipliers[k-2]);
            int sum = 0;
            for (int b = 0; b < 4; ++b) { sum += (value >> (8*b)) & 255; }
            require(ait_level(state, k) == ((sum+2)/4)-128, "independent byte-sum level");
            require(aik_level_c(state, ggml_ai4_trellis_mul((ggml_type) (113+k))) == ((sum+2)/4)-128, "independent unit byte-sum level");
        }
    }
    for (int type = 110; type <= 117; ++type) {
        auto * meta = fixture(type);
        require(llama_ait_format_error(meta) != nullptr, "unversioned rejected");
        gguf_set_val_u32(meta, AIT_FORMAT_VERSION_KEY, 1);
        require(llama_ait_format_error(meta) != nullptr, "old version rejected");
        gguf_set_val_str(meta, AIT_FORMAT_VERSION_KEY, "2");
        require(llama_ait_format_error(meta) != nullptr, "string version rejected");
        gguf_set_val_u32(meta, AIT_FORMAT_VERSION_KEY, 2);
        require(llama_ait_format_error(meta) != nullptr, "missing decoder metadata rejected");
        stamp(meta);
        require(llama_ait_format_error(meta) == nullptr, "valid format accepted");
        if (type < 113) {
            gguf_set_val_u32(meta, keys[type-110], multipliers[(type-109)%3]);
            require(llama_ait_format_error(meta) != nullptr, "cross-tier multiplier rejected");
            gguf_set_val_str(meta, keys[type-110], "wrong type");
            require(llama_ait_format_error(meta) != nullptr, "string multiplier rejected");
        } else if (type >= 115) {
            const int k = type - 113;
            const auto t = (ggml_type) type;
            require(ggml_ai4_rate(t) == k && ggml_ai4_layout(t) == 1, "unit rate and layout");
            require(ggml_ait_rate(t) == 0, "unit is not tiled");
            require(ggml_ai4_trellis_mul(t) == multipliers[k-2], "unit own multiplier");
            require(AIT_MUL_FOR_TYPE(type) == multipliers[k-2], "unit metadata multiplier");
            require(ggml_blck_size(t) == 128 && ggml_row_size(t, 640) == 5*(2+16*k), "unit H128 row bytes");
            require(std::fabs(ggml_ai4_scale_mul(t) - (k == 2 ? 1.00f : k == 3 ? 1.05f : 1.09f)) < 1e-6f, "unit scale seed");
            char key[32];
            std::snprintf(key, sizeof(key), "ait.multiplier.%d", type);
            gguf_set_val_u32(meta, key, multipliers[(k-1)%3]);
            require(llama_ait_format_error(meta) != nullptr, "cross-tier unit multiplier rejected");
            gguf_set_val_str(meta, key, "wrong type");
            require(llama_ait_format_error(meta) != nullptr, "unit string multiplier rejected");
        } else if (type == 114) {
            require(ggml_ai4_rate((ggml_type) type) == 5, "AIT5_U rate");
            require(ggml_ai4_trellis_mul((ggml_type) type) == 0xAA73876Bu, "AIT5_U own multiplier");
            require(ggml_blck_size((ggml_type) type) == 128 && ggml_row_size((ggml_type) type, 1024) == 8*82, "AIT5_U row bytes");
            gguf_set_val_u32(meta, "ait.multiplier.114", 0);
            require(llama_ait_format_error(meta) != nullptr, "wrong AIT5_U multiplier rejected");
        } else {
            gguf_set_val_str(meta, AIT2P_HASH_KEY, "wrong table");
            require(llama_ait_format_error(meta) != nullptr, "wrong pair table rejected");
        }
        gguf_free(meta);
    }
    {
        // AIT2P_U: unit bytes, pair-table metadata, no multiplier
        const auto t = GGML_TYPE_AIT2P_U;
        require(ggml_ai4_rate(t) == 2 && ggml_ai4_layout(t) == 1 && ggml_ai4_code(t) == 1, "AIT2P_U rate, layout, code");
        require(ggml_ai4_code(GGML_TYPE_AIT2_U) == 0 && ggml_ai4_code(GGML_TYPE_AIT2P) == -1, "unit codes");
        require(ggml_ai4_trellis_mul(t) == 0 && ggml_ait_rate(t) == 0, "AIT2P_U has no multiplier, is not tiled");
        require(ggml_blck_size(t) == 128 && ggml_row_size(t, 640) == 5*34, "AIT2P_U row bytes");
        auto * meta = fixture((int) t);
        require(llama_ait_format_error(meta) != nullptr, "unversioned AIT2P_U rejected");
        stamp(meta);
        require(llama_ait_format_error(meta) == nullptr, "valid AIT2P_U accepted");
        gguf_set_val_str(meta, AIT2P_HASH_KEY, "wrong table");
        require(llama_ait_format_error(meta) != nullptr, "AIT2P_U wrong pair table rejected");
        gguf_free(meta);
        // independent decode of one random tile: bit-serial MSB-first stream, pair windows of 20 bits ending at 4p + 4
        uint32_t wd[8];
        uint32_t x = 0x12345679u;
        for (int rep = 0; rep < 64; ++rep) {
            for (uint32_t & w : wd) { x = x * 1664525u + 1013904223u; w = x ^ (x >> 15); }
            for (int n = 0; n < 128; ++n) {
                const int p = n / 2;
                uint32_t win = 0;
                for (int b = 4*p + 4 - 20; b < 4*p + 4; ++b) {
                    const int bb = (b + 256) % 256;
                    win = (win << 1) | ((wd[bb / 32] >> (31 - bb % 32)) & 1u);
                }
                const uint32_t e = ((win * (win + 1u)) >> 10) & 1023u;
                require(aik_level_n(wd, 2, 0, n) == ait2p_tab[2*e + n % 2], "independent AIT2P_U pair level");
            }
        }
    }
    for (int type : {-1, (int) GGML_TYPE_F16}) {
        auto * meta = fixture(type);
        require(llama_ait_format_error(meta) == nullptr, "non-AIT model unaffected");
        gguf_free(meta);
    }
    std::printf("PASS %d checks\n", checks);
    return 0;
}
