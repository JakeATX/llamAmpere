#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
static uint64_t io_bytes(void) {
    char data[1024];
    int fd = open("/proc/thread-self/io", O_RDONLY | O_CLOEXEC);
    if (fd < 0) return UINT64_MAX;
    ssize_t n = read(fd, data, sizeof(data)-1); close(fd);
    if (n <= 0) return UINT64_MAX;
    data[n] = 0;
    const char * field = strstr(data, "\nread_bytes:");
    return field ? strtoull(field + strlen("\nread_bytes:"), 0, 10) : UINT64_MAX;
}
static void emit(const char * op, const char * path, uint64_t before, struct timespec start, int result) {
    int saved = errno;
    uint64_t after = io_bytes();
    struct timespec end; clock_gettime(CLOCK_REALTIME, &end);
    if (before != UINT64_MAX && after != UINT64_MAX && after > before)
        fprintf(stderr,"FNARENA_META_IO op=%s start=%lld.%09ld end=%lld.%09ld tid=%ld result=%d read_bytes=%" PRIu64 " path=%s\n",op,(long long)start.tv_sec,start.tv_nsec,(long long)end.tv_sec,end.tv_nsec,syscall(SYS_gettid),result,after-before,path);
    errno = saved;
}
int posix_fallocate(int fd, off_t offset, off_t length) {
    static int (*real_call)(int,off_t,off_t);
    if (!real_call) real_call = dlsym(RTLD_NEXT,"posix_fallocate");
    char link[64], path[4096]; snprintf(link,sizeof(link),"/proc/self/fd/%d",fd);
    ssize_t n=readlink(link,path,sizeof(path)-1);
    if(n<0)n=0;
    path[n]=0;
    struct timespec start;clock_gettime(CLOCK_REALTIME,&start);
    uint64_t before=io_bytes();int result=real_call(fd,offset,length);
    emit("posix_fallocate",path,before,start,result);return result;
}
int mkstemp(char * path) {
    static int (*real_call)(char *);
    if (!real_call) real_call = dlsym(RTLD_NEXT,"mkstemp");
    struct timespec start;clock_gettime(CLOCK_REALTIME,&start);
    uint64_t before=io_bytes();int result=real_call(path);
    emit("mkstemp",path,before,start,result);return result;
}
int unlink(const char * path) {
    static int (*real_call)(const char *);
    if (!real_call) real_call = dlsym(RTLD_NEXT,"unlink");
    struct timespec start;clock_gettime(CLOCK_REALTIME,&start);
    uint64_t before=io_bytes();int result=real_call(path);
    emit("unlink",path,before,start,result);return result;
}
