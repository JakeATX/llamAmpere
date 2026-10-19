// Trace only the staging API that reads target activation output, never model weights.
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

struct llama_context;
using copy_fn = bool (*)(llama_context *, float *, int32_t, int32_t);
static pthread_once_t initialized = PTHREAD_ONCE_INIT;
static copy_fn real_copy;
static void resolve() { real_copy = (copy_fn) dlsym(RTLD_NEXT, "llama_copy_embeddings_nextn"); }
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
extern "C" bool llama_copy_embeddings_nextn(llama_context * ctx, float * dst, int32_t first, int32_t count) {
    const int saved_errno = errno;
    pthread_once(&initialized, resolve);
    if (!real_copy) { _exit(126); }
    timespec start{}, end{};
    clock_gettime(CLOCK_REALTIME, &start);
    const uint64_t before = io_bytes();
    errno = saved_errno;
    const bool ok = real_copy(ctx, dst, first, count);
    const int result_errno = errno;
    const uint64_t after = io_bytes();
    clock_gettime(CLOCK_REALTIME, &end);
    if (before == UINT64_MAX || after == UINT64_MAX || after < before) {
        fprintf(stderr, "FNARENA_ACTIVATION_IO_ERROR tid=%ld\n", syscall(SYS_gettid));
    } else if (after > before) {
        fprintf(stderr, "FNARENA_ACTIVATION_IO start=%lld.%09ld end=%lld.%09ld tid=%ld result=%d read_bytes=%" PRIu64 " first=%d count=%d scope=nextn_output_copy\n",
                (long long) start.tv_sec, start.tv_nsec, (long long) end.tv_sec, end.tv_nsec,
                syscall(SYS_gettid), (int) ok, after - before, first, count);
    }
    errno = result_errno;
    return ok;
}
