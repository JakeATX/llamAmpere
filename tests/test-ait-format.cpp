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
    for (int type : {-1, (int) GGML_TYPE_F16}) {
        auto * meta = fixture(type);
        require(llama_ait_format_error(meta) == nullptr, "non-AIT model unaffected");
        gguf_free(meta);
    }
    auto * fractional = fixture(GGML_TYPE_AIF25_U);
    require(ggml_row_size(GGML_TYPE_AIF25_U,128) == 42, "fractional exact 2.625-bpw bytes");
    require(ggml_ai4_rate(GGML_TYPE_AIF25_U) == 0, "fractional never enters integer-rate dispatch");
    require(llama_ait_format_error(fractional) != nullptr, "unversioned fractional bytes rejected");
    gguf_set_val_str(fractional, "ait.fractional.contract", "wrong");
    require(llama_ait_format_error(fractional) != nullptr, "wrong fractional contract rejected");
    gguf_set_val_str(fractional, "ait.fractional.contract", AIF25_CONTRACT);
    require(llama_ait_format_error(fractional) == nullptr, "fractional own-constant contract accepted");
    gguf_free(fractional);
    std::printf("PASS %d checks\n", checks);
    return 0;
}
