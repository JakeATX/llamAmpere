// fork (R4, NEXTPF): CPU checks of llama_moe_prefetch (src/llama-moe-prefetch.h): env budget, routing record and
// accounting, and that the worker writes exactly the submitted bytes and wait() returns only after they landed.
#include "../src/llama-moe-prefetch.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

int main() {
    // env budget
    unsetenv("LLAMA_MOE_CACHE_NEXT_PREFETCH");
    CHECK(llama_moe_prefetch::env_budget() == 0);
    CHECK(llama_moe_prefetch::from_env(4) == nullptr);
    setenv("LLAMA_MOE_CACHE_NEXT_PREFETCH", "100", 1);
    CHECK(llama_moe_prefetch::env_budget() == 64);
    setenv("LLAMA_MOE_CACHE_NEXT_PREFETCH", "-3", 1);
    CHECK(llama_moe_prefetch::env_budget() == 0);
    setenv("LLAMA_MOE_CACHE_NEXT_PREFETCH", "6", 1);
    auto pf = llama_moe_prefetch::from_env(4);
    CHECK(pf && pf->budget() == 6);
    unsetenv("LLAMA_MOE_CACHE_NEXT_PREFETCH");

    // record: distinct, sorted, replaces
    const int32_t ids_a[] = { 5, 1, 5, 2, 9, 1 };
    pf->record(3, ids_a, 6);
    CHECK((pf->predicted(3) == std::vector<int32_t>{ 1, 2, 5, 9 }));
    const int32_t ids_b[] = { 7, 7 };
    pf->record(3, ids_b, 2);
    CHECK((pf->predicted(3) == std::vector<int32_t>{ 7 }));
    CHECK(pf->predicted(2).empty());

    // account: counts marked experts the next lookup selects, once
    pf->mark(1, { 4, 8, 11 });
    const int32_t ids_c[] = { 8, 3, 11, 8 };
    pf->account(1, ids_c, 4);
    CHECK(pf->n_used == 2);
    pf->account(1, ids_c, 4);
    CHECK(pf->n_used == 2);

    // worker: random jobs into a CPU buffer, compared with the same writes done in place
    ggml_init_params ip = { 2*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    const size_t n = 1 << 20;
    ggml_tensor * bank = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) n);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    CHECK(buf != nullptr);
    std::vector<uint8_t> src(n), ref(n, 0), got(n);
    std::mt19937 rng(42);
    for (auto & b : src) {
        b = (uint8_t) rng();
    }
    ggml_backend_buffer_clear(buf, 0);
    size_t bytes = 0;
    for (int round = 0; round < 2000; ++round) {
        std::vector<llama_moe_prefetch::job> jobs;
        const int nj = 1 + (int) (rng() % 6);
        for (int j = 0; j < nj; ++j) {
            const size_t size = 1 + rng() % 4096;
            const size_t off  = rng() % (n - size);
            const size_t from = rng() % (n - size);
            jobs.push_back({ bank, src.data() + from, off, size });
            memcpy(ref.data() + off, src.data() + from, size);
            bytes += size;
        }
        pf->submit(std::move(jobs));
        if (round % 7 == 0) {
            pf->wait();
            ggml_backend_tensor_get(bank, got.data(), 0, n);
            CHECK(memcmp(got.data(), ref.data(), n) == 0);
        }
    }
    pf->wait();
    ggml_backend_tensor_get(bank, got.data(), 0, n);
    CHECK(memcmp(got.data(), ref.data(), n) == 0);
    CHECK(pf->n_bytes == bytes);
    CHECK(pf->n_rounds == 2000);
    pf->submit({});
    CHECK(pf->n_rounds == 2000);

    // destructor with work still queued: the jobs finish first
    {
        std::vector<llama_moe_prefetch::job> jobs = { { bank, src.data(), 0, n } };
        pf->submit(std::move(jobs));
        pf.reset();
        ggml_backend_tensor_get(bank, got.data(), 0, n);
        CHECK(memcmp(got.data(), src.data(), n) == 0);
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    if (n_fail) {
        fprintf(stderr, "test-moe-prefetch: %d checks failed\n", n_fail);
        return 1;
    }
    printf("OK: test-moe-prefetch (env budget, record/account, 2000 worker rounds byte-exact, drain on destroy)\n");
    return 0;
}
