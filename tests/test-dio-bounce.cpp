// Safety check for --load-mode dio: O_DIRECT reads go through a plain bounce buffer (llama_file::read_raw_bounced)
// and must give the same tensor bytes as mmap. Runs the loader's chunked, aligned read for every tensor of a GGUF.
// usage: test-dio-bounce MODEL.gguf   (exit 77 = skipped: no model, or the filesystem refuses O_DIRECT)

#include "gguf.h"
#include "../src/llama-impl.h"
#include "../src/llama-mmap.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

struct free_deleter { void operator()(void * p) const { llama_aligned_free(p); } };

int main(int argc, char ** argv) {
    llama_aligned_free(nullptr);
    for (size_t alignment : { sizeof(void *), (size_t) 64, (size_t) 4096, (size_t) 2 << 20 }) {
        for (size_t size : { (size_t) 1, (size_t) 4097, (size_t) LLAMA_DIRECT_IO_BUFFER_SIZE }) {
            std::unique_ptr<void, free_deleter> p(llama_aligned_alloc(alignment, size));
            if (!p || (uintptr_t) p.get() % alignment) { fprintf(stderr, "bad aligned allocation\n"); return 1; }
            memset(p.get(), 0x5a, size);
        }
    }
#ifdef _WIN32
    FILE * tmp = tmpfile();
    if (!tmp) { return 1; }
    const bool direct = llama_file(tmp).has_direct_io();
    fclose(tmp);
    if (direct) { fprintf(stderr, "buffered Windows file reported direct I/O\n"); return 1; }
#endif
    if (argc > 1 && strcmp(argv[1], "--alloc-only") == 0) { printf("aligned allocation: PASS\n"); return 0; }
    const char * path = argc > 1 ? argv[1] : getenv("LLAMA_TEST_DIO_MODEL");
    if (!path || !path[0]) {
        printf("test-dio-bounce: no model given, skipped\n");
        return 77;
    }
    gguf_init_params params = { /*.no_alloc = */ true, /*.ctx = */ nullptr };
    gguf_context * gctx = gguf_init_from_file(path, params);
    if (!gctx) { fprintf(stderr, "cannot read %s\n", path); return 1; }

    llama_file direct(path, "rb", true);
    if (!direct.has_direct_io()) {
        printf("test-dio-bounce: O_DIRECT not available for %s, skipped\n", path);
        gguf_free(gctx);
        return 77;
    }
    llama_file plain(path, "rb", false);
    llama_mmap map(&plain, 0);
    const uint8_t * base = (const uint8_t *) map.addr();

    const size_t alignment = direct.read_alignment();
    const size_t data_offs = gguf_get_data_offset(gctx);
    const int64_t n_tensors = gguf_get_n_tensors(gctx);

    int failures = 0;
    // 64 MiB = the loader's bounce; 1 MiB forces several bounce chunks per read
    for (size_t bounce_size : { (size_t) LLAMA_DIRECT_IO_BUFFER_SIZE, (size_t) 1 << 20 }) {
        std::unique_ptr<void, free_deleter> bounce(llama_aligned_alloc(std::max<size_t>(alignment, 4096), bounce_size));
        // stands in for the pinned upload slot: same size and alignment rules as the loader
        const size_t slot_size = LLAMA_DIRECT_IO_BUFFER_SIZE + 2 * alignment;
        std::unique_ptr<void, free_deleter> slot(llama_aligned_alloc(64, slot_size));
        if (!bounce || !slot) { fprintf(stderr, "allocation failed\n"); return 1; }
        std::vector<uint8_t> out;
        size_t bytes = 0;
        int64_t checked = 0;
        for (int64_t i = 0; i < n_tensors; ++i) {
            const size_t offset = data_offs + gguf_get_tensor_offset(gctx, i);
            const size_t n_size = gguf_get_tensor_size(gctx, i);
            if (n_size == 0) { continue; }
            out.assign(n_size, 0);
            // the loader's upload loop (llama-model-loader.cpp), with the slot in place of pinned memory
            const size_t aligned_offset = offset & ~(alignment - 1);
            const size_t offset_from_alignment = offset - aligned_offset;
            const size_t read_end = (offset + n_size + alignment - 1) & ~(alignment - 1);
            direct.seek(aligned_offset, SEEK_SET);
            size_t bytes_read = 0, data_read = 0;
            while (bytes_read < read_end - aligned_offset) {
                const size_t read_size = std::min<size_t>(LLAMA_DIRECT_IO_BUFFER_SIZE, read_end - aligned_offset - bytes_read);
                const uintptr_t dest = ((uintptr_t) slot.get() + alignment - 1) & ~(uintptr_t) (alignment - 1);
                direct.read_raw_bounced((void *) dest, read_size, bounce.get(), bounce_size);
                uintptr_t data = dest;
                size_t n = read_size;
                if (bytes_read == 0) { data += offset_from_alignment; n -= offset_from_alignment; }
                if (aligned_offset + bytes_read + read_size > offset + n_size) { n -= read_end - (offset + n_size); }
                memcpy(out.data() + data_read, (const void *) data, n);
                data_read += n;
                bytes_read += read_size;
            }
            if (data_read != n_size || memcmp(out.data(), base + offset, n_size) != 0) {
                fprintf(stderr, "MISMATCH tensor %s (bounce %zu)\n", gguf_get_tensor_name(gctx, i), bounce_size);
                failures++;
            }
            bytes += n_size;
            checked++;
        }
        printf("test-dio-bounce: bounce %zu MiB: %lld tensors, %.1f MiB byte-identical to mmap, alignment %zu, %d mismatches\n",
                bounce_size >> 20, (long long) checked, bytes / 1048576.0, alignment, failures);
    }
    gguf_free(gctx);
    printf(failures ? "FAIL\n" : "PASS\n");
    return failures ? 1 : 0;
}
