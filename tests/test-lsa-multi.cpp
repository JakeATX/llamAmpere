// LLAMA_LSA_MULTI=1 (MULTI4LSA): per-stream LSA side caches. Each check runs in its own process (the LSA config is
// read once per process): "multi" = stream isolation of the index cache, commit, invalidate, owner guard, state
// suffix and forget; "off" = without the flag every stream is stream 0 (the old single set); "refuse" = the startup
// check refuses the combinations multi does not serve.
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

static void make_map(const std::string & path) {
    auto * ctx = ggml_init({ 2*ggml_tensor_overhead() + 2048, nullptr, false });
    assert(ctx);
    auto * meta = gguf_init_empty();
    gguf_set_val_u32(meta, "lsa.index_dim", 2);
    for (const char * name : { "lsa.0.wq", "lsa.0.wk" }) {
        auto * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
        ggml_set_name(t, name);
        for (int i = 0; i < 4; ++i) { ((float *) t->data)[i] = 1.0f; }
        gguf_add_tensor(meta, t);
    }
    assert(gguf_write_to_file(meta, path.c_str(), false));
    gguf_free(meta);
    ggml_free(ctx);
}

static llama_lsa_idx_slot get_in(int32_t strm, const ggml_tensor * k) {
    llama_lsa_stream_scope scope(strm);
    llama_lsa_idx_slot slot;
    assert(llama_lsa_idx_get(k, 2, ggml_backend_cpu_buffer_type(), &slot));
    return slot;
}

static int exercise(const std::string & dir, const std::string & mode) {
    for (const char * name : { "LLAMA_LSA_IDX_Q8", "LLAMA_LSA_BLOCK_SUM", "LLAMA_LSA_COMP", "LLAMA_LSA_PACKED",
                              "LLAMA_LSA_IDX_TEST_NO_STATE", "LLAMA_LSA_IDX_TEST_SKIP_INVALIDATE" }) {
        setenv(name, "0", 1);
    }
    unsetenv("LLAMA_LSA_RESIDENT_FIRST");
    setenv("LLAMA_LSA_MODE", "key", 1);
    setenv("LLAMA_LSA_MAP", (dir + "/a.gguf").c_str(), 1);
    setenv("LLAMA_LSA_IDX_CACHE", "1", 1);
    setenv("LLAMA_LSA_MULTI", mode == "off" ? "0" : "1", 1);
    if (mode == "refuse") {
        setenv("LLAMA_LSA_PACKED", "1", 1);
        assert(llama_lsa_cfg().multi);
        // packed mask + multi: refused on SJ-KVaRN 4/4; a plain cache is refused for multi alone
        assert(!llama_lsa_check_cache(4, 4, GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16).empty());
        assert(!llama_lsa_check_cache(0, 0, GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16).empty());
        return 0;
    }
    if (mode == "accept") {
        setenv("LLAMA_LSA_RESIDENT_FIRST", "6", 1);
        assert(llama_lsa_cfg().multi);
        assert(llama_lsa_cfg().resident_first == 6);
        assert(llama_lsa_describe().find(" multi=1 resident_first=6") != std::string::npos);
        assert(llama_lsa_check_cache(4, 4, GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16).empty());
        return 0;
    }
    if (mode == "lead" || mode == "lead_off") {
        // LLAMA_LSA_J3_LEAD: look-ahead and first = RESIDENT_FIRST; with it J3 runs per stream and on 4/4 strips.
        // Without RESIDENT_FIRST the knob is dropped and J3 + multi (and J3 on 4/4) stay refused.
        setenv("LLAMA_LSA_RESIDENT", "50", 1);
        setenv("LLAMA_LSA_J3", "1", 1);
        setenv("LLAMA_LSA_J3_LEAD", "1", 1);
        if (mode == "lead") {
            setenv("LLAMA_LSA_RESIDENT_FIRST", "6", 1);
        }
        const auto & c = llama_lsa_cfg();
        assert(c.j3 && c.multi);
        if (mode == "lead") {
            assert(c.j3_lead && c.j3_first == 6 && c.j3_ahead == 6);
            assert(llama_lsa_describe().find(" j3_first=6 j3_lead=1 j3_ahead=6") != std::string::npos);
            assert(llama_lsa_check_cache(4, 4, GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16).empty());
        } else {
            assert(!c.j3_lead && c.j3_first == 3 && c.j3_ahead == 3);
            assert(llama_lsa_describe().find("j3_lead") == std::string::npos);
            assert(!llama_lsa_check_cache(4, 4, GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16).empty());
        }
        return 0;
    }

    auto * ctx = ggml_init({ ggml_tensor_overhead(), nullptr, true });
    assert(ctx);
    auto * k = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2, 4);
    ggml_set_name(k, "cache_k_l0");
    const ggml_tensor * caches[] = { k };
    const std::vector<uint16_t> expected = { 0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700, 0x4800 };

    if (mode == "off") {
        assert(!llama_lsa_cfg().multi && llama_lsa_cfg().resident_first == 0);
        assert(llama_lsa_describe().find("multi=") == std::string::npos);
        llama_lsa_set_stream(3);
        assert(llama_lsa_stream() == 0); // ignored: one set of caches
        llama_lsa_idx_slot s0 = get_in(0, k);
        llama_lsa_idx_slot s3 = get_in(3, k);
        assert(s0.t == s3.t);
        llama_lsa_idx_commit(k, s0.epoch);
        assert(get_in(5, k).valid);
        llama_lsa_idx_forget(caches, 1);
        ggml_free(ctx);
        return 0;
    }

    // multi: stream 1 and stream 2 own separate index caches for the same K tensor
    assert(llama_lsa_cfg().multi);
    llama_lsa_idx_slot s1 = get_in(1, k);
    llama_lsa_idx_slot s2 = get_in(2, k);
    assert(s1.t && s2.t && s1.t != s2.t);
    assert(!s1.valid && !s2.valid);
    ggml_backend_tensor_set(s1.t, expected.data(), 0, 16);
    { llama_lsa_stream_scope sc(1); llama_lsa_idx_commit(k, s1.epoch); }
    assert(get_in(1, k).valid);
    assert(!get_in(2, k).valid); // a commit of stream 1 never validates stream 2
    assert(llama_lsa_stream() == 0); // scopes restore the previous stream

    // state suffix: stream 1 has a valid layer, stream 2 none
    state_writer w1, w2;
    { llama_lsa_stream_scope sc(1); llama_lsa_idx_state_write(w1, 4); }
    { llama_lsa_stream_scope sc(2); llama_lsa_idx_state_write(w2, 4); }
    assert(w1.data.size() > 44);
    assert(w2.data.empty());

    // invalidating stream 2 leaves stream 1 valid; invalidating every stream does not
    llama_lsa_idx_invalidate(caches, 1, "test", 2);
    assert(get_in(1, k).valid);
    llama_lsa_idx_invalidate(caches, 1, "test", -1);
    assert(!get_in(1, k).valid);

    // restore stream 1's suffix into stream 2 (a sequence restored into another slot): stream 2 gets the rows
    {
        llama_lsa_stream_scope sc(2);
        state_reader in(w1.data);
        llama_lsa_idx_state_read(in);
        assert(in.n_bytes() == w1.data.size());
    }
    llama_lsa_idx_slot r2 = get_in(2, k);
    assert(r2.valid && r2.t == s2.t);
    std::vector<uint16_t> actual(8);
    ggml_backend_tensor_get(r2.t, actual.data(), 0, 16);
    assert(actual == expected);
    assert(!get_in(1, k).valid); // stream 1 untouched by the restore

    // owner guard per stream: sequence 3 in stream 3 never makes stream 2 stale; a foreign sequence in stream 2 does
    { llama_lsa_stream_scope sc(2); llama_lsa_seq_guard(2, true, false, false); }
    { llama_lsa_stream_scope sc(3); llama_lsa_seq_guard(3, true, false, false); }
    assert(get_in(2, k).valid);
    { llama_lsa_stream_scope sc(2); llama_lsa_seq_guard(2, false, false, false); }
    assert(get_in(2, k).valid);
    { llama_lsa_stream_scope sc(2); llama_lsa_seq_guard(7, false, false, false); }
    assert(!get_in(2, k).valid);

    // forget drops every stream's entry: a later get starts unbuilt and zeroed
    { llama_lsa_stream_scope sc(2); llama_lsa_idx_commit(k, get_in(2, k).epoch); }
    llama_lsa_idx_forget(caches, 1);
    llama_lsa_idx_slot f2 = get_in(2, k);
    assert(!f2.valid);
    ggml_backend_tensor_get(f2.t, actual.data(), 0, 16);
    assert(actual == std::vector<uint16_t>(8, 0));
    llama_lsa_idx_forget(caches, 1);
    ggml_free(ctx);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc == 3) { return exercise(argv[1], argv[2]); }
    char dir[] = "/tmp/test-lsa-multi-XXXXXX";
    assert(mkdtemp(dir));
    make_map(std::string(dir) + "/a.gguf");
    for (const char * mode : { "multi", "off", "refuse", "accept", "lead", "lead_off" }) {
        const pid_t pid = fork();
        assert(pid >= 0);
        if (pid == 0) { execl(argv[0], argv[0], dir, mode, (char *) nullptr); _exit(127); }
        int status = 0;
        assert(waitpid(pid, &status, 0) == pid);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            fprintf(stderr, "LSA multi: mode %s FAILED\n", mode);
            return 1;
        }
    }
    assert(remove((std::string(dir) + "/a.gguf").c_str()) == 0);
    assert(rmdir(dir) == 0);
    puts("LSA multi: per-stream index caches, commit, invalidate, owner guard, state suffix, forget, off path, refusals, J3 lead PASS");
    return 0;
}
