#include "../src/llama-lsa.h"
#include "../src/llama-io.h"
#include "ggml-cpu.h"
#include "gguf.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

struct state_writer : llama_io_write_i {
    std::vector<uint8_t> data;
    void write(const void * src, size_t size) override {
        const auto * p = (const uint8_t *) src;
        data.insert(data.end(), p, p + size);
    }
    void write_tensor(ggml_tensor * t, size_t offset, size_t size) override {
        const size_t start = data.size();
        data.resize(start + size);
        ggml_backend_tensor_get(t, data.data() + start, offset, size);
    }
    size_t n_bytes() override { return data.size(); }
};

struct state_reader : llama_io_read_i {
    const std::vector<uint8_t> & data;
    size_t pos = 0;
    explicit state_reader(const std::vector<uint8_t> & data) : data(data) {}
    void read(void * dst, size_t size) override {
        if (size > data.size() - pos) { throw std::runtime_error("short test state"); }
        memcpy(dst, data.data() + pos, size);
        pos += size;
    }
    void read_tensor(ggml_tensor * t, size_t offset, size_t size) override {
        if (size > data.size() - pos) { throw std::runtime_error("short test tensor"); }
        ggml_backend_tensor_set(t, data.data() + pos, offset, size);
        pos += size;
    }
    size_t n_bytes() override { return pos; }
};

static void make_map(const std::string & path, float query) {
    auto * ctx = ggml_init({ 2*ggml_tensor_overhead() + 2048, nullptr, false });
    assert(ctx);
    auto * meta = gguf_init_empty();
    gguf_set_val_u32(meta, "lsa.index_dim", 2);
    for (const char * name : { "lsa.0.wq", "lsa.0.wk" }) {
        auto * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
        ggml_set_name(t, name);
        for (int i = 0; i < 4; ++i) { ((float *) t->data)[i] = name[7] == 'q' ? query : 1.0f; }
        gguf_add_tensor(meta, t);
    }
    assert(gguf_write_to_file(meta, path.c_str(), false));
    gguf_free(meta);
    ggml_free(ctx);
}

static int exercise(const std::string & dir, const std::string & mode) {
    const bool off = mode == "off";
    const std::string map = dir + (mode == "changed" ? "/b.gguf" : mode == "write" ? "/a.gguf" : "/copy.gguf");
    for (const char * name : { "LLAMA_LSA_IDX_Q8", "LLAMA_LSA_BLOCK_SUM", "LLAMA_LSA_COMP",
                              "LLAMA_LSA_IDX_TEST_NO_STATE", "LLAMA_LSA_IDX_TEST_SKIP_INVALIDATE" }) {
        setenv(name, "0", 1);
    }
    setenv("LLAMA_LSA_MODE", off ? "off" : "key", 1);
    setenv("LLAMA_LSA_MAP", map.c_str(), 1);
    setenv("LLAMA_LSA_IDX_CACHE", "1", 1);
    auto * ctx = ggml_init({ ggml_tensor_overhead(), nullptr, true });
    assert(ctx);
    auto * k = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2, 4);
    ggml_set_name(k, "cache_k_l0");
    llama_lsa_idx_slot slot;
    assert(llama_lsa_idx_get(k, 2, ggml_backend_cpu_buffer_type(), &slot));
    const std::vector<uint16_t> expected = { 0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700, 0x4800 };
    if (mode == "write" || off) {
        ggml_backend_tensor_set(slot.t, expected.data(), 0, 16);
        llama_lsa_idx_commit(k, slot.epoch);
        state_writer out;
        llama_lsa_idx_state_write(out, 4);
        if (off) {
            assert(out.data.empty());
        } else {
            assert(out.data.size() > 44);
            std::ofstream file(dir + "/state", std::ios::binary);
            file.write((const char *) out.data.data(), out.data.size());
            assert(file.good());
        }
    } else {
        std::ifstream file(dir + "/state", std::ios::binary);
        assert(file.good());
        std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)), {});
        if (mode == "digest") { data.at(12) ^= 1; }
        if (mode == "flags") { data.at(8) = 2; }
        if (mode == "legacy") {
            data.erase(data.begin() + 8, data.begin() + 44);
            const uint32_t version = 2;
            memcpy(data.data() + 4, &version, sizeof(version));
        }
        bool rejected = false;
        try {
            state_reader in(data);
            llama_lsa_idx_state_read(in);
            assert(in.n_bytes() == data.size());
        } catch (const std::runtime_error & e) {
            rejected = true;
            const std::string what = e.what();
            assert(what.find(mode == "flags" ? "flags" : mode == "legacy" ? "no map identity" : "SHA256 mismatch") != std::string::npos);
        }
        assert(rejected == (mode != "same"));
        assert(llama_lsa_idx_get(k, 2, ggml_backend_cpu_buffer_type(), &slot));
        assert(slot.valid == !rejected);
        std::vector<uint16_t> actual(8);
        ggml_backend_tensor_get(slot.t, actual.data(), 0, 16);
        assert(actual == (rejected ? std::vector<uint16_t>(8, 0) : expected));
    }
    const ggml_tensor * caches[] = { k };
    llama_lsa_idx_forget(caches, 1);
    ggml_free(ctx);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc == 3) { return exercise(argv[1], argv[2]); }
    char dir[] = "/tmp/test-lsa-state-XXXXXX";
    assert(mkdtemp(dir));
    make_map(std::string(dir) + "/a.gguf", 1.0f);
    make_map(std::string(dir) + "/copy.gguf", 1.0f);
    make_map(std::string(dir) + "/b.gguf", 2.0f);
    for (const char * mode : { "write", "same", "changed", "digest", "flags", "legacy", "off" }) {
        const pid_t pid = fork();
        assert(pid >= 0);
        if (pid == 0) { execl(argv[0], argv[0], dir, mode, (char *) nullptr); _exit(127); }
        int status = 0;
        assert(waitpid(pid, &status, 0) == pid);
        assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    }
    for (const char * name : { "/a.gguf", "/copy.gguf", "/b.gguf", "/state" }) {
        assert(remove((std::string(dir) + name).c_str()) == 0);
    }
    assert(rmdir(dir) == 0);
    puts("LSA state: same map, changed query map, digest, flags, legacy, and OFF path PASS");
    return 0;
}
