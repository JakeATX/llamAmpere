// Only the pure memcpy boundary AFTER context synchronization/reordering/flushing.
// Resolve and verify the source mapping before either thread I/O counter sample.
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <cctype>
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

using copy_fn = void (*)(void *, const void *, size_t);
static pthread_once_t initialized = PTHREAD_ONCE_INIT;
static copy_fn real_copy;
static copy_fn real_write;
using compare_fn = int (*)(const void *, const void *, size_t);
static compare_fn real_compare;
static void resolve() {
    real_copy = (copy_fn) dlsym(RTLD_NEXT, "llama_copy_moe_output");
    real_write = (copy_fn) dlsym(RTLD_NEXT, "llama_write_moe_output");
    real_compare = (compare_fn) dlsym(RTLD_NEXT, "llama_compare_moe_output");
}
static uint64_t io_bytes() {
    char data[1024];
    int fd = (int) syscall(SYS_openat, AT_FDCWD, "/proc/thread-self/io", O_RDONLY | O_CLOEXEC, 0);
    if (fd < 0) { return UINT64_MAX; }
    ssize_t n = syscall(SYS_read, fd, data, sizeof(data) - 1);
    syscall(SYS_close, fd);
    if (n <= 0) { return UINT64_MAX; }
    data[n] = 0;
    const char * field = strstr(data, "\nread_bytes:");
    return field ? strtoull(field + strlen("\nread_bytes:"), nullptr, 10) : UINT64_MAX;
}
static bool source_mapping(const void * src, size_t bytes, char * path, uint64_t & offset) {
    const char * directory = getenv("LLAMA_MOE_ARENA_OUTPUT_DIR");
    char canonical[PATH_MAX];
    if (!directory || !realpath(directory, canonical)) { return false; }
    const size_t prefix = strlen(canonical);
    FILE * maps = fopen("/proc/self/maps", "r");
    if (!maps) { return false; }
    char line[PATH_MAX + 256];
    bool found = false;
    while (fgets(line, sizeof(line), maps)) {
        uintptr_t begin, end;
        uint64_t file_offset, inode;
        char permissions[5];
        int n = 0;
        if (sscanf(line, "%" SCNxPTR "-%" SCNxPTR " %4s %" SCNx64 " %*s %" SCNu64 " %n",
                   &begin, &end, permissions, &file_offset, &inode, &n) != 5 || !n) { continue; }
        const uintptr_t address = (uintptr_t) src;
        if (address < begin || address >= end || bytes > end - address || permissions[0] != 'r' || !inode) { continue; }
        char * name = line + n;
        name[strcspn(name, "\n")] = 0;
        const char * stem = "/llama-moe-output-";
        const size_t stem_size = strlen(stem);
        if (strlen(name) < prefix + stem_size + 6 || strncmp(name, canonical, prefix) || strncmp(name + prefix, stem, stem_size)) { break; }
        const char * suffix = name + prefix + stem_size;
        if (strlen(suffix) != 6 + strlen(" (deleted)")) { break; }
        bool letters = true;
        for (int i = 0; i < 6; ++i) { letters = letters && isalnum((unsigned char) suffix[i]); }
        if (!letters || strcmp(suffix + 6, " (deleted)") || strlen(name) >= PATH_MAX ||
                file_offset > UINT64_MAX - (address - begin)) { break; }
        strcpy(path, name);
        offset = file_offset + address - begin;
        found = true;
        break;
    }
    fclose(maps);
    return found;
}
extern "C" void llama_copy_moe_output(void * dst, const void * src, size_t bytes) {
    const int saved_errno = errno;
    pthread_once(&initialized, resolve);
    if (!real_copy) { _exit(126); }
    char path[PATH_MAX] = {};
    uint64_t offset = 0;
    const bool owned = bytes && source_mapping(src, bytes, path, offset);
    timespec start{}, end{};
    clock_gettime(CLOCK_REALTIME, &start);
    const uint64_t before = io_bytes();
    errno = saved_errno;
    real_copy(dst, src, bytes);
    const int result_errno = errno;
    const uint64_t after = io_bytes();
    clock_gettime(CLOCK_REALTIME, &end);
    if (!owned || before == UINT64_MAX || after == UINT64_MAX || after < before) {
        fprintf(stderr, "FNARENA_ACTIVATION_IO_ERROR tid=%ld reason=%s\n", syscall(SYS_gettid), owned ? "counter" : "source");
    } else if (after > before) {
        fprintf(stderr, "FNARENA_ACTIVATION_IO start=%lld.%09ld end=%lld.%09ld tid=%ld read_bytes=%" PRIu64 " offset=%" PRIu64 " bytes=%zu scope=owned_output_memcpy path=%s\n",
                (long long) start.tv_sec, start.tv_nsec, (long long) end.tv_sec, end.tv_nsec,
                syscall(SYS_gettid), after - before, offset, bytes, path);
    }
    errno = result_errno;
}

// Layout operands contain serialized metadata. Reject any file-backed second operand.
static bool anonymous_operand(const void * ptr, size_t bytes) {
    FILE * maps = fopen("/proc/self/maps", "r");
    if (!maps) { return false; }
    char line[PATH_MAX + 256];
    bool found = false;
    while (fgets(line, sizeof(line), maps)) {
        uintptr_t begin, end;
        uint64_t offset, inode;
        char permissions[5];
        int n = 0;
        if (sscanf(line, "%" SCNxPTR "-%" SCNxPTR " %4s %" SCNx64 " %*s %" SCNu64 " %n",
                   &begin, &end, permissions, &offset, &inode, &n) != 5 || !n) { continue; }
        const uintptr_t address = (uintptr_t) ptr;
        if (address >= begin && address < end && bytes <= end - address) {
            found = permissions[0] == 'r' && inode == 0;
            break;
        }
    }
    fclose(maps);
    return found;
}
static int layout_operation(void * owned, const void * metadata, size_t bytes, bool compare) {
    const int saved_errno = errno;
    pthread_once(&initialized, resolve);
    if ((compare && !real_compare) || (!compare && !real_write)) { _exit(126); }
    if (!bytes) {
        errno = saved_errno;
        if (compare) { return real_compare(owned, metadata, bytes); }
        real_write(owned, metadata, bytes);
        return 0;
    }
    char path[PATH_MAX] = {};
    uint64_t offset = 0;
    const bool valid = source_mapping(owned, bytes, path, offset) && anonymous_operand(metadata, bytes);
    timespec start{}, end{};
    clock_gettime(CLOCK_REALTIME, &start);
    const uint64_t before = io_bytes();
    errno = saved_errno;
    int result = 0;
    if (compare) { result = real_compare(owned, metadata, bytes); }
    else { real_write(owned, metadata, bytes); }
    const int result_errno = errno;
    const uint64_t after = io_bytes();
    clock_gettime(CLOCK_REALTIME, &end);
    if (before == UINT64_MAX || after == UINT64_MAX || after < before) {
        fprintf(stderr, "FNARENA_LAYOUT_IO_ERROR tid=%ld reason=counter\n", syscall(SYS_gettid));
    } else if (after > before && !valid) {
        fprintf(stderr, "FNARENA_LAYOUT_IO_ERROR tid=%ld reason=operand\n", syscall(SYS_gettid));
    } else if (after > before) {
        fprintf(stderr, "FNARENA_LAYOUT_IO op=%s start=%lld.%09ld end=%lld.%09ld tid=%ld read_bytes=%" PRIu64 " offset=%" PRIu64 " bytes=%zu path=%s\n",
                compare ? "compare" : "write", (long long) start.tv_sec, start.tv_nsec, (long long) end.tv_sec, end.tv_nsec,
                syscall(SYS_gettid), after - before, offset, bytes, path);
    }
    errno = result_errno;
    return result;
}
extern "C" void llama_write_moe_output(void * dst, const void * src, size_t bytes) {
    layout_operation(dst, src, bytes, false);
}
extern "C" int llama_compare_moe_output(const void * lhs, const void * rhs, size_t bytes) {
    return layout_operation(const_cast<void *>(lhs), rhs, bytes, true);
}
