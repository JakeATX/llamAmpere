#include "../src/llama-model-loader.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

int main() {
    FILE * file = tmpfile();
    if (!file) { return 1; }
    try {
        ggml_backend_cpu_reg();
        ggml_context_ptr ctx(ggml_init({ 1u << 20, nullptr, false }));
        gguf_context_ptr meta(gguf_init_empty());
        gguf_set_val_str(meta.get(), "general.architecture", "llama");
        for (const char * name : { "token_embd.weight", "output.weight" }) {
            auto * t = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 64, 16);
            ggml_set_name(t, name);
            memset(t->data, 0x5a, ggml_nbytes(t));
            gguf_add_tensor(meta.get(), t);
        }
        auto * ai3 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_AIT3_U2, 640, 16);
        if (GGML_TYPE_AIT3_U2 != 123 || strcmp(ggml_type_name(ai3->type), "ait3_u2") || ggml_type_size(ai3->type) != 242) {
            throw std::runtime_error("type 123 traits");
        }
        ggml_set_name(ai3, "blk.0.ffn_up_exps.weight");
        memset(ai3->data, 0x39, ggml_nbytes(ai3));
        gguf_add_tensor(meta.get(), ai3);
        if (!gguf_write_to_file_ptr(meta.get(), file, false)) { throw std::runtime_error("write fixture"); }
        fflush(file);
        fseek(file, 0, SEEK_END);
        std::vector<unsigned char> before(ftell(file)), after(before.size());
        rewind(file);
        if (fread(before.data(), 1, before.size(), file) != before.size()) { throw std::runtime_error("read fixture"); }
        for (bool private_dense : { false, true }) {
            rewind(file);
            std::vector<std::string> splits;
            llama_model_loader loader(nullptr, nullptr, nullptr, "", splits, file, LLAMA_LOAD_MODE_MMAP, false, false, false, nullptr, nullptr);
            loader.init_mappings(false, nullptr, private_dense);
            for (const auto & entry : loader.weights_map) {
                const auto & w = entry.second;
                auto * expected = ggml_get_tensor(ctx.get(), entry.first.c_str());
                auto * mapped = (unsigned char *) loader.mappings.at(w.idx)->addr() + w.offs;
                if (w.tensor->type != expected->type || memcmp(mapped, expected->data, ggml_nbytes(expected))) {
                    throw std::runtime_error("mapped weight bytes or type differ");
                }
#ifdef __linux__
                if (private_dense && entry.first == "token_embd.weight") { mapped[0] ^= 0xff; }
#endif
            }
        }
        rewind(file);
        if (fread(after.data(), 1, after.size(), file) != after.size() || before != after) { throw std::runtime_error("fixture changed"); }
        ggml_moe_bank_onepath_set_live(true);
        if (!ggml_moe_bank_onepath_live()) { throw std::runtime_error("live flag set"); }
        ggml_moe_bank_onepath_set_live(false);
        if (ggml_moe_bank_onepath_live()) { throw std::runtime_error("live flag clear"); }
        puts("GGUF model mappings: plain/private request, F32 and type 123 bytes, unchanged file, atomic flag PASS");
    } catch (const std::exception & error) {
        fprintf(stderr, "%s\n", error.what());
        fclose(file);
        return 1;
    }
    fclose(file);
    return 0;
}
