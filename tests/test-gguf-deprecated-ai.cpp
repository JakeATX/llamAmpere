// A GGUF that names a retired pre-format-2 AI+ tensor type must be refused at load with a clear one-line error, while
// current types and plain F32 still load. Crafts tiny one-tensor files by hand. No GPU needed.
#include "ggml.h"
#include "gguf.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::string g_log;
static void capture(enum ggml_log_level, const char * text, void *) { g_log += text; }

static int checks = 0;
static void require(bool ok, const char * what) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL %s\nlog: %s\n", what, g_log.c_str()); std::exit(1); }
}

template <typename T> static void put(std::vector<uint8_t> & b, T v) {
    const uint8_t * p = (const uint8_t *) &v;
    b.insert(b.end(), p, p + sizeof(T));
}

// one tensor "t.weight" [ne0, ne1] of raw type id `type`; payload of `nbytes` zero bytes (0 = header only)
static std::string write_fixture(const char * dir, int type, int64_t ne0, int64_t ne1, size_t nbytes) {
    std::vector<uint8_t> b = {'G', 'G', 'U', 'F'};
    put<uint32_t>(b, 3);
    put<uint64_t>(b, 1);   // tensors
    put<uint64_t>(b, 0);   // kv pairs
    const char name[] = "t.weight";
    put<uint64_t>(b, sizeof(name) - 1);
    b.insert(b.end(), name, name + sizeof(name) - 1);
    put<uint32_t>(b, 2);
    put<uint64_t>(b, (uint64_t) ne0);
    put<uint64_t>(b, (uint64_t) ne1);
    put<uint32_t>(b, (uint32_t) type);
    put<uint64_t>(b, 0);
    while (b.size() % 32) { b.push_back(0); }
    b.resize(b.size() + nbytes, 0);
    const std::string path = std::string(dir) + "/type" + std::to_string(type) + ".gguf";
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f || std::fwrite(b.data(), 1, b.size(), f) != b.size()) { std::fprintf(stderr, "cannot write %s\n", path.c_str()); std::exit(1); }
    std::fclose(f);
    return path;
}

static gguf_context * load(const std::string & path) {
    g_log.clear();
    gguf_init_params params = {/*no_alloc =*/ true, /*ctx =*/ nullptr};
    return gguf_init_from_file(path.c_str(), params);
}

int main() {
    ggml_log_set(capture, nullptr);
    const char * tmp = std::getenv("TMPDIR");
    const std::string dir = tmp && *tmp ? tmp : "/tmp";

    struct { int id; const char * name; } deprecated[] = {
        {70, "AI4"}, {71, "AI4_U"}, {72, "AI3_U"}, {73, "AI2_U"}, {74, "AI5_U"}, {75, "AI6_U"},
        {86, "AI6_UO"}, {87, "AI5_UN"}, {96, "AI2_S"}, {97, "AI3_S"}, {98, "AI4_S"}, {99, "AI5_S"}, {100, "AI6_S"},
    };
    for (const auto & d : deprecated) {
        const std::string path = write_fixture(dir.c_str(), d.id, 1024, 16, 0);
        gguf_context * ctx = load(path);
        require(ctx == nullptr, "deprecated type must not load");
        char want[160];
        std::snprintf(want, sizeof(want), "deprecated AI+ format %s (type %d) is no longer supported; re-encode with a current type", d.name, d.id);
        require(g_log.find(want) != std::string::npos, "deprecated type gives the clear error text");
        std::remove(path.c_str());
    }

    // controls: F32 and every current AI+ type id still parse (header level)
    const int keep[] = {GGML_TYPE_F32, GGML_TYPE_AIT2, GGML_TYPE_AIT3, GGML_TYPE_AIT4, GGML_TYPE_AIT2P, GGML_TYPE_AIT5_U,
                        GGML_TYPE_AIT2_U, GGML_TYPE_AIT3_U, GGML_TYPE_AIT4_U, GGML_TYPE_AIT2P_U};
    for (int id : keep) {
        const int64_t ne0 = 1024, ne1 = 32;
        const size_t nbytes = ggml_row_size((ggml_type) id, ne0) * ne1;
        const std::string path = write_fixture(dir.c_str(), id, ne0, ne1, nbytes);
        gguf_context * ctx = load(path);
        require(ctx != nullptr, "current type still loads");
        require(g_log.find("deprecated") == std::string::npos, "current type is not called deprecated");
        gguf_free(ctx);
        std::remove(path.c_str());
    }
    std::printf("test-gguf-deprecated-ai: %d checks passed\n", checks);
    return 0;
}
