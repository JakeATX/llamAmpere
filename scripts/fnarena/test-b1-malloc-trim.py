"""Exercise the production trim helper with live and freed CPU allocations."""
import json
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/llama-context.cpp"


def main():
    source = SOURCE.read_text()
    start = source.index("static void b1_memory_trim(const char * phase) {")
    end = source.index("\n}\n", start) + 3
    helper = source[start:end]
    preamble = r'''
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <malloc.h>
#include <thread>
#include <vector>
#define LLAMA_LOG_INFO(...) fprintf(stderr, __VA_ARGS__)
static int64_t ggml_time_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static size_t rss() {
    FILE * f = fopen("/proc/self/smaps_rollup", "r");
    if (!f) { abort(); }
    char line[256];
    size_t value = 0;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "Rss: %zu kB", &value) == 1) { break; }
    }
    fclose(f);
    return value * 1024;
}
'''
    fixture = r'''
int main() {
    if (!mallopt(M_MMAP_THRESHOLD, 1024 * 1024 * 1024) ||
            !mallopt(M_TRIM_THRESHOLD, 1024 * 1024 * 1024)) { return 1; }
    constexpr size_t bytes = 65536;
    std::vector<unsigned char *> blocks;
    for (size_t i = 0; i < 4096; ++i) {
        auto * p = static_cast<unsigned char *>(malloc(bytes));
        if (!p) { return 2; }
        memset(p, static_cast<unsigned char>(i), bytes);
        blocks.push_back(p);
    }
    for (size_t i = 0; i < blocks.size(); ++i) {
        if (i % 16 != 0) { free(blocks[i]); blocks[i] = nullptr; }
    }
    unsetenv("LLAMA_B1_MALLOC_TRIM");
    errno = EDOM;
    b1_memory_trim("disabled-unset");
    if (errno != EDOM) { return 3; }
    for (const char * value : { "0", "true", "01" }) {
        setenv("LLAMA_B1_MALLOC_TRIM", value, 1);
        errno = EDOM;
        b1_memory_trim("disabled-value");
        if (errno != EDOM) { return 4; }
    }
    const size_t before = rss();
    setenv("LLAMA_B1_MALLOC_TRIM", "1", 1);
    errno = EDOM;
    b1_memory_trim("fixture-freed");
    if (errno != EDOM) { return 5; }
    const size_t after = rss();
    for (size_t i = 0; i < blocks.size(); ++i) {
        if (!blocks[i]) { continue; }
        for (size_t j = 0; j < bytes; ++j) {
            if (blocks[i][j] != static_cast<unsigned char>(i)) { return 6; }
        }
    }
    std::atomic<int> ready{0};
    std::atomic<bool> checked{false};
    std::atomic<bool> valid{true};
    std::vector<std::thread> workers;
    for (int i = 0; i < 4; ++i) {
        workers.emplace_back([&, i] {
            std::vector<unsigned char> live(1024 * 1024, static_cast<unsigned char>(i + 1));
            ready.fetch_add(1);
            while (!checked.load()) { std::this_thread::yield(); }
            for (unsigned char c : live) { if (c != i + 1) { valid.store(false); } }
        });
    }
    while (ready.load() != 4) { std::this_thread::yield(); }
    b1_memory_trim("fixture-live-workers");
    checked.store(true);
    for (auto & worker : workers) { worker.join(); }
    if (!valid.load()) { return 7; }
    for (auto * p : blocks) { free(p); }
    printf("{\"before_rss_bytes\":%zu,\"after_rss_bytes\":%zu,\"released_rss_bytes\":%zu}\n",
        before, after, before > after ? before - after : 0);
    return before > after && before - after >= 128 * 1024 * 1024 ? 0 : 8;
}
'''
    with tempfile.TemporaryDirectory(prefix="llama.cpp-trim-") as directory:
        path = Path(directory)
        (path / "fixture.cpp").write_text(preamble + helper + fixture)
        subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                        "-pthread", str(path / "fixture.cpp"), "-o", str(path / "fixture")], check=True)
        result = subprocess.run([str(path / "fixture")], check=True, text=True, capture_output=True)
        assert "disabled" not in result.stderr
        assert result.stderr.count("b1_memory_trim:") == 2
        print(result.stderr, end="")
        report = json.loads(result.stdout)
        report["scope"] = "CPU helper allocation fixture only; no model RAM or CUDA claim"
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
