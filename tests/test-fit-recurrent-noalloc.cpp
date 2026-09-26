// The -fit memory probe creates contexts from a no_alloc model and reads the device free memory while the probe context
// is alive (common/fit.cpp common_get_device_memory_data_impl). Every context buffer must therefore only be sized, not
// allocated, in that mode, or the probe's free memory drops by the buffer while memory_breakdown() also reports it.
// This test checks the recurrent state (llama_memory_recurrent, with rollback snapshots n_rs_seq > 0), or for deepseek4
// the DSV4 compressor states (llama_dsv4_comp_state, also with n_rs_seq snapshot planes), on the CPU:
//   1. a no_alloc context reports the same context bytes as a real context with the same parameters;
//   2. creating the no_alloc context does not make those bytes resident (RSS grows by far less than their size).
// usage: test-fit-recurrent-noalloc -m <recurrent, hybrid or deepseek4 model, e.g. the generated qwen35-dense.gguf or
//        deepseek4-moe.gguf>

#include "arg.h"
#include "common.h"
#include "llama.h"

#include "../src/llama-ext.h"

#include <clocale>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <unistd.h>

static int64_t resident_bytes() {
#if defined(__linux__)
    FILE * f = std::fopen("/proc/self/statm", "r");
    if (f == nullptr) {
        return -1;
    }
    long long size = 0;
    long long resident = 0;
    const int n = std::fscanf(f, "%lld %lld", &size, &resident);
    std::fclose(f);
    if (n != 2) {
        return -1;
    }
    return (int64_t) resident * (int64_t) sysconf(_SC_PAGESIZE);
#else
    return -1;
#endif
}

static size_t context_bytes(const llama_context * ctx) {
    size_t ret = 0;
    for (const auto & [buft, mb] : llama_get_memory_breakdown(ctx)) {
        ret += mb.context;
    }
    return ret;
}

static llama_context_params make_cparams(const common_params & params) {
    llama_context_params cparams = common_context_params_to_llama(params);
    // many sequences with rollback snapshots so that the recurrent state is large (~150 MiB for qwen35-dense) and
    // clearly visible in the resident set when it is allocated
    cparams.n_seq_max   = 128;
    cparams.kv_unified  = true;
    cparams.n_ctx       = 4096;
    cparams.n_rs_seq    = 8;
    cparams.n_batch     = 512;
    cparams.n_ubatch    = 512;
    cparams.offload_kqv = false;
    return cparams;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    ggml_backend_load_all();

    if (resident_bytes() < 0) {
        std::fprintf(stderr, "%s: no /proc/self/statm, skipping\n", __func__);
        return 0;
    }

    llama_model_params mparams = common_model_params_to_llama(params);
    mparams.n_gpu_layers = 0; // keep the state in host memory, where the resident set shows a real allocation

    // the probe: no_alloc model, no loading, as in common_get_device_memory_data_impl
    size_t  bytes_probe = 0;
    int64_t rss_growth  = 0;
    {
        llama_model_params mparams_probe = mparams;
        mparams_probe.no_alloc  = true;
        mparams_probe.load_mode = LLAMA_LOAD_MODE_NONE;
        llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams_probe);
        if (model == nullptr) {
            std::fprintf(stderr, "%s: failed to load the model (no_alloc)\n", __func__);
            return 1;
        }
        char arch[64] = {0};
        llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
        const bool is_dsv4 = std::strcmp(arch, "deepseek4") == 0;
        if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model) && !is_dsv4) {
            std::fprintf(stderr, "%s: skipping for non-recurrent model\n", __func__);
            llama_model_free(model);
            return 0;
        }

        const int64_t rss_before = resident_bytes();
        llama_context * ctx = llama_init_from_model(model, make_cparams(params));
        const int64_t rss_after = resident_bytes();
        if (ctx == nullptr) {
            std::fprintf(stderr, "%s: failed to create the no_alloc context\n", __func__);
            llama_model_free(model);
            return 1;
        }
        bytes_probe = context_bytes(ctx);
        rss_growth  = rss_after - rss_before;
        llama_free(ctx);
        llama_model_free(model);
    }

    // the real context with the same parameters
    size_t bytes_real = 0;
    {
        llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
        if (model == nullptr) {
            std::fprintf(stderr, "%s: failed to load the model\n", __func__);
            return 1;
        }
        llama_context * ctx = llama_init_from_model(model, make_cparams(params));
        if (ctx == nullptr) {
            std::fprintf(stderr, "%s: failed to create the context\n", __func__);
            llama_model_free(model);
            return 1;
        }
        bytes_real = context_bytes(ctx);
        llama_free(ctx);
        llama_model_free(model);
    }

    constexpr double MiB = 1024.0*1024.0;
    std::fprintf(stderr, "%s: context bytes no_alloc %.2f MiB, real %.2f MiB, RSS growth while creating the no_alloc context %.2f MiB\n",
        __func__, bytes_probe/MiB, bytes_real/MiB, rss_growth/MiB);

    int failures = 0;
    if (bytes_real < (size_t) (32*MiB)) {
        std::fprintf(stderr, "FAIL: the context is too small (%.2f MiB) for the resident-set check to mean anything\n", bytes_real/MiB);
        failures++;
    }
    if (bytes_probe != bytes_real) {
        std::fprintf(stderr, "FAIL: a no_alloc context should report the context bytes a real context allocates\n");
        failures++;
    }
    if (rss_growth > (int64_t) (bytes_real / 2)) {
        std::fprintf(stderr, "FAIL: creating a no_alloc context should not allocate its state (RSS grew by %.2f MiB)\n", rss_growth/MiB);
        failures++;
    }

    if (failures != 0) {
        return 1;
    }
    std::printf("fit recurrent no_alloc tests passed\n");
    return 0;
}
