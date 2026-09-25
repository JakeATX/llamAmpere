#include "fit.h"

#include <cstdint>
#include <cstdio>

static int failures = 0;

static void expect(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        failures++;
    }
}

int main() {
    constexpr size_t MiB = 1024*1024;

    // Qwen3.5 27B MTP draft with f16 KV, 4 sequences, unified KV: one nextn layer of 4 KV heads x 256 dims,
    // 4 KiB per token; measured at the minimum context of 4096 tokens with a 126 MiB compute buffer
    const common_fit_extra_memory mtp = {0, 16*MiB, 126*MiB};

    {
        // the fit's probe context n_ctx_train * n_seq_max, next to a main compute buffer of 1187 MiB
        const common_fit_extra_memory at = common_fit_extra_memory_at(mtp, 4096, 1048576, 1187*MiB, true);
        expect(at.model == 0, "shared weights should stay uncounted");
        expect(at.context == 4096*MiB, "the draft KV cache should scale linearly to 1M tokens (4 KiB per token)");
        expect(at.compute == 0, "a compute buffer that fits in the main context's buffer should add nothing");
    }

    {
        const common_fit_extra_memory at = common_fit_extra_memory_at(mtp, 4096, 53248, 300*MiB, true);
        expect(at.context == 208*MiB, "the draft KV cache at 53248 tokens should be 208 MiB");
        expect(at.compute == 0, "the draft compute should ride on the main context's buffer");
    }

    {
        // a draft graph larger than the main one gets a private buffer of its own size (ggml_gallocr donor rule)
        const common_fit_extra_memory at = common_fit_extra_memory_at(mtp, 4096, 4096, 100*MiB, true);
        expect(at.context == 16*MiB, "the measured context should be kept unscaled");
        expect(at.compute == 126*MiB, "a compute buffer larger than the donor's should be counted in full");
    }

    {
        // equal sizes still share: the donor's buffer is adopted when it is large enough
        const common_fit_extra_memory at = common_fit_extra_memory_at(mtp, 4096, 8192, 126*MiB, true);
        expect(at.compute == 0, "a compute buffer equal to the donor's should be shared");
        expect(at.context == 32*MiB, "doubling the context should double the draft KV cache");
    }

    {
        // separate draft models (or LLAMA_SHARED_COMPUTE=0) keep their own buffers and are measured at each context
        const common_fit_extra_memory draft = {512*MiB, 64*MiB, 200*MiB};
        const common_fit_extra_memory at = common_fit_extra_memory_at(draft, 16384, 16384, 1000*MiB, false);
        expect(at.model == 512*MiB, "a separate draft model should keep its weights");
        expect(at.context == 64*MiB, "a measurement at the same context should be used as is");
        expect(at.compute == 200*MiB, "an unshared compute buffer should always be counted");
    }

    {
        // scaling rounds up so the fit never under-reserves
        const common_fit_extra_memory odd = {0, 1001, 0};
        const common_fit_extra_memory at = common_fit_extra_memory_at(odd, 3, 4, 0, true);
        expect(at.context == 1335, "a scaled context should be rounded up");
    }

    if (failures != 0) {
        std::fprintf(stderr, "%d fit extra model tests failed\n", failures);
        return 1;
    }
    std::printf("fit extra model tests passed\n");
    return 0;
}
