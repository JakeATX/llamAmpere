// fork (INKCACHEUX): pure checks of the expert-cache policy pieces (no model, no GPU)
//   GS1 layer / MiB list parsing, GS4 heat save + reload + blend + mismatch, #249 fit and minimax choice, GS9 schedule
#include "../src/llama-moe-cache-policy.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <thread>
#include <vector>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

static void set_env(const char * k, const char * v) {
#ifdef _WIN32
    _putenv_s(k, v ? v : "");
#else
    if (v) { setenv(k, v, 1); } else { unsetenv(k); }
#endif
}

static void test_parse() {
    std::vector<uint8_t> m;
    std::string err;
    CHECK(llama_moe_cache_parse_layers("0,4-6,9", 10, m, err));
    CHECK(m.size() == 10 && m[0] && !m[1] && m[4] && m[5] && m[6] && !m[7] && m[9]);
    CHECK(!llama_moe_cache_parse_layers("0,4-20", 10, m, err));
    CHECK(!llama_moe_cache_parse_layers("3-1", 10, m, err));
    CHECK(!llama_moe_cache_parse_layers("1,,2", 10, m, err));
    CHECK(!llama_moe_cache_parse_layers("a", 10, m, err));
    CHECK(!llama_moe_cache_parse_layers("", 10, m, err));

    std::vector<double> mib;
    CHECK(llama_moe_cache_parse_mib_list("4096,2048.5", mib) && mib.size() == 2 && mib[1] == 2048.5);
    CHECK(!llama_moe_cache_parse_mib_list("4096,-1", mib));
    CHECK(!llama_moe_cache_parse_mib_list("x", mib));

    llama_moe_cache_controls c;
    set_env("LLAMA_MOE_CACHE_DEV_MIB", "1000");
    set_env("LLAMA_MOE_CACHE_LAYERS", "2-3");
    c.from_env(8);
    CHECK(c.dev_budget_mib(0) == 1000 && c.dev_budget_mib(3) == 1000);
    CHECK(!c.layer_selected(1) && c.layer_selected(2) && c.layer_selected(3) && !c.layer_selected(4));
    llama_moe_cache_controls c2;
    set_env("LLAMA_MOE_CACHE_DEV_MIB", "1000,500");
    set_env("LLAMA_MOE_CACHE_SLOTS", "64");
    c2.from_env(8);
    CHECK(c2.slots == 64 && c2.dev_mib.empty()); // the slot count wins
    set_env("LLAMA_MOE_CACHE_DEV_MIB", nullptr);
    set_env("LLAMA_MOE_CACHE_SLOTS", nullptr);
    set_env("LLAMA_MOE_CACHE_LAYERS", nullptr);
    llama_moe_cache_controls c3;
    c3.from_env(8);
    CHECK(!c3.any() && c3.dev_budget_mib(0) < 0 && c3.layer_selected(7) && !c3.req_stats && !c3.boundary);
}

static void test_heat() {
    char path[256];
    snprintf(path, sizeof(path), "test-moe-heat-%d.bin", (int) std::rand());
    std::remove(path);
    set_env("LLAMA_MOE_CACHE_HEAT", path);
    set_env("LLAMA_MOE_CACHE_HEAT_WEIGHT", "0.25");
    set_env("LLAMA_MOE_CACHE_HEAT_SAVE_S", "0");

    llama_moe_cache_heat h;
    h.init(2, 4, 1234);
    CHECK(h.on());
    CHECK(!h.load()); // no file yet
    CHECK(!h.save()); // nothing learned
    const int32_t ids0[] = { 1, 2, 1, 3 }; // 2 rows x 2 used in layer 0
    h.observe(0, ids0, 4);
    h.add_rows(0, 2);
    CHECK(h.save());
    auto r = h.rates();
    CHECK(r[0] == 0.0f && r[1] == 1.0f && r[2] == 0.5f && r[3] == 0.5f && r[4] == 0.0f);

    llama_moe_cache_heat h2;
    h2.init(2, 4, 1234);
    CHECK(h2.load());
    CHECK(h2.rate_old.size() == 8 && h2.rate_old[1] == 1.0f);
    const int32_t ids1[] = { 0, 2 }; // 1 row
    h2.observe(0, ids1, 2);
    h2.add_rows(0, 1);
    r = h2.rates(); // old*(0.75) + new*(0.25)
    CHECK(std::fabs(r[0] - 0.25f) < 1e-6f && std::fabs(r[1] - 0.75f) < 1e-6f && std::fabs(r[2] - 0.625f) < 1e-6f);
    CHECK(r[5] == 0.0f); // layer 1 saw no rows: old rate (0) kept
    h2.maybe_save();     // save_s = 0 and new rows: saves
    llama_moe_cache_heat h3;
    h3.init(2, 4, 1234);
    CHECK(h3.load() && std::fabs(h3.rate_old[2] - 0.625f) < 1e-6f);

    llama_moe_cache_heat h4;
    h4.init(2, 4, 999); // other layout
    CHECK(!h4.load());
    llama_moe_cache_heat h5;
    h5.init(3, 4, 1234);
    CHECK(!h5.load());

    std::remove(path);
    set_env("LLAMA_MOE_CACHE_HEAT", nullptr);
    llama_moe_cache_heat off;
    off.init(2, 4, 1);
    CHECK(!off.on());
}

// synthetic layer: CPU a + c*k + d*m, GPU g0 + g*m, the layer takes the max, plus noise
static double layer_ms(size_t n, size_t m, std::mt19937 & rng) {
    const double a = 0.20, c = 0.08, d = 0.02, g0 = 0.15, g = 0.12;
    const size_t k = n - m;
    const double cpu = k > 0 ? a + c*k + d*m : 0.0;
    const double gpu = g0 + g*m;
    std::normal_distribution<double> noise(0.0, 0.005);
    return std::max(cpu, gpu) + noise(rng);
}

static void test_fit_minimax() {
    // feed the fitter directly (no clocks): samples at mixed splits
    llama_moe_split_policy p;
    set_env("LLAMA_MOE_CACHE_SPLIT", "minimax");
    p.init(0.5f);
    CHECK(p.mode == llama_moe_split_policy::MINIMAX);
    std::mt19937 rng(42);
    for (int it = 0; it < 400; ++it) {
        const size_t n = 4 + it % 9;
        const size_t m = (it*7) % (n + 1);
        const size_t k = n - m;
        const double y = layer_ms(n, m, rng);
        const double xc[3] = { 1.0, (double) k, (double) m };
        const double xg[2] = { 1.0, (double) m };
        // label by the true bounding branch (the online path labels by the fitted one)
        const double cpu = k > 0 ? 0.20 + 0.08*k + 0.02*m : 0.0, gpu = 0.15 + 0.12*m;
        if (k > 0 && cpu >= gpu) { p.cpu.add(xc, y); p.cpu.solve(true); }
        else { p.gpu.add(xg, y); p.gpu.solve(false); p.n_gpu_pcie += m > 0; }
    }
    CHECK(p.ready());
    fprintf(stderr, "fit: CPU %.4f %.4f %.4f GPU %.4f %.4f\n", p.cpu.theta[0], p.cpu.theta[1], p.cpu.theta[2], p.gpu.theta[0], p.gpu.theta[1]);
    CHECK(std::fabs(p.cpu.theta[1] - 0.08) < 0.01 && std::fabs(p.gpu.theta[1] - 0.12) < 0.01);
    CHECK(p.cpu.theta[2] <= p.cpu.theta[1]);
    // n = 10, true max(CPU, GPU) by PCIe count m: m=4 0.76, m=5 0.75 (best), m=6 0.87; extremes m=0 1.00, m=10 1.35
    uint32_t carry = 0;
    p.n_lookups = 1;
    const size_t n_cpu = p.n_cpu(10, carry);
    fprintf(stderr, "minimax: n 10 -> %zu on the CPU\n", n_cpu);
    CHECK(n_cpu == 5 || n_cpu == 6);
    set_env("LLAMA_MOE_CACHE_SPLIT", nullptr);
}

static void test_online_and_sched() {
    // online path through on_ids/record with real (sleep-free) timestamps: just check bookkeeping and windows
    set_env("LLAMA_MOE_CACHE_SPLIT", "sched");
    llama_moe_split_policy p;
    p.init(0.83f);
    CHECK(p.mode == llama_moe_split_policy::SCHED && std::fabs(p.frac - 0.17) < 1e-6);
    uint32_t carry = 0;
    size_t tot_cpu = 0, tot = 0;
    for (int i = 0; i < 1000; ++i) { tot_cpu += p.n_cpu(7, carry); tot += 7; }
    CHECK(std::fabs((double) tot_cpu/tot - 0.83) < 0.01); // the carry keeps the share right on average
    // fake a ready fit, then run windows: updates at 48, 144, 240 only
    p.cpu.n = 100; p.gpu.n = 100; p.n_gpu_pcie = 50;
    p.cpu.theta[1] = 0.08; p.gpu.theta[1] = 0.12;
    for (int w = 0; w < 300; ++w) {
        for (int il = 0; il < 4; ++il) { p.on_ids(il, 1); } // no record(): no samples, the fake fit stays
        p.window_end();
    }
    CHECK(p.n_windows == 300);
    CHECK(p.n_updates == 3);
    // 0.08/(0.08 + 1.85*0.12) = 0.265 -> 0.25 on the 0.05 grid
    CHECK(std::fabs(p.frac - 0.25) < 1e-9);
    CHECK(p.frac_hist.size() == 4);
    // prompt windows do not count
    const int64_t nw = p.n_windows;
    for (int il = 0; il < 4; ++il) { p.on_ids(il, 512); }
    p.window_end();
    CHECK(p.n_windows == nw);

    set_env("LLAMA_MOE_CACHE_SPLIT", nullptr);
    set_env("LLAMA_MOE_CACHE_SPLIT_Q256", "64");
    llama_moe_split_policy f;
    f.init(0.5f);
    CHECK(f.mode == llama_moe_split_policy::FIXED && f.frac == 0.25);
    set_env("LLAMA_MOE_CACHE_SPLIT_Q256", nullptr);

    set_env("LLAMA_MOE_CACHE_SPLIT", "minimax");
    llama_moe_split_policy z;
    z.init(0.0f); // no CPU share: no CPU part, policy off
    CHECK(!z.active());
    set_env("LLAMA_MOE_CACHE_SPLIT", nullptr);
    llama_moe_split_policy o;
    o.init(0.5f);
    CHECK(!o.active());
}

int main() {
    test_parse();
    test_heat();
    test_fit_minimax();
    test_online_and_sched();
    if (n_fail) {
        fprintf(stderr, "test-moe-cache-policy: %d FAILED\n", n_fail);
        return 1;
    }
    fprintf(stderr, "test-moe-cache-policy: OK\n");
    return 0;
}
