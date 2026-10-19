"""Run under the shared build slot: compile a cold-file API fixture and test attribution."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from nonweight_io import activation_copies

root = Path(__file__).resolve().parents[2]
tracer = Path(sys.argv[1]).resolve()
line = 'FNARENA_ACTIVATION_IO start=2.000 end=3.000 tid=2 result=1 read_bytes=4096 first=0 count=16 scope=nextn_output_copy'
assert activation_copies(line, 1, 4, 8192)['activation_copy_bytes'] == 4096
for text, start, end, total in [
    (line, 2.1, 4, 8192), (line, 1, 4, 2048), (line + '\n' + line, 1, 4, 8192),
    (line.replace('result=1', 'result=0'), 1, 4, 8192),
    (line.replace('count=16', 'count=0'), 1, 4, 8192),
    ('FNARENA_ACTIVATION_IO_ERROR tid=2', 1, 4, 8192),
]:
    try:
        activation_copies(text, start, end, total)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid attribution accepted')
with tempfile.TemporaryDirectory(prefix='copy-trace-test-', dir=root / 'ram-cut') as directory:
    work = Path(directory)
    (work / 'mock.cpp').write_text('''
#include <cerrno>
#include <cstring>
#include <cstdint>
struct llama_context { float * data; };
extern "C" bool llama_copy_embeddings_nextn(llama_context * ctx, float * dst, int32_t first, int32_t count) {
    if (first < 0 || count < 0) { errno = ERANGE; return false; }
    memcpy(dst, ctx->data + first, (size_t)count * sizeof(float));
    errno = EAGAIN;
    return true;
}
''')
    (work / 'driver.cpp').write_text('''
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
struct llama_context { float * data; };
extern "C" bool llama_copy_embeddings_nextn(llama_context *, float *, int32_t, int32_t);
int main() {
    const size_t bytes = 4u << 20;
    char name[] = "activation-XXXXXX";
    int fd = mkstemp(name);
    if (fd < 0) { return 2; }
    unlink(name);
    std::vector<uint32_t> original(bytes / 4), output(bytes / 4);
    for (size_t i = 0; i < original.size(); ++i) { original[i] = uint32_t(i * 2654435761u); }
    if (write(fd, original.data(), bytes) != (ssize_t)bytes || fsync(fd) || posix_fadvise(fd, 0, bytes, POSIX_FADV_DONTNEED)) { return 3; }
    void * data = mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
    if (data == MAP_FAILED) { return 4; }
    llama_context ctx{(float *)data};
    if (!llama_copy_embeddings_nextn(&ctx, (float *)output.data(), 0, bytes / 4) || errno != EAGAIN || output != original) { return 5; }
    if (llama_copy_embeddings_nextn(&ctx, (float *)output.data(), -1, 1) || errno != ERANGE) { return 6; }
    munmap(data, bytes);
    close(fd);
    puts("exact bytes and errno: OK");
}
''')
    subprocess.run(['g++', '-shared', '-fPIC', '-O2', str(work / 'mock.cpp'), '-o', str(work / 'libmock.so')], check=True)
    subprocess.run(['g++', '-O2', str(work / 'driver.cpp'), '-L' + str(work), '-lmock', '-Wl,-rpath,' + str(work), '-o', str(work / 'llama.cpp-copy-fixture')], check=True)
    start = time.time()
    proc = subprocess.run([str(work / 'llama.cpp-copy-fixture')], cwd=work, env=dict(os.environ, LD_PRELOAD=str(tracer)), text=True, capture_output=True, check=True)
    end = time.time()
    assert 'exact bytes and errno: OK' in proc.stdout
    account = activation_copies(proc.stderr, start, end, 64 << 20)
    assert account['activation_copy_bytes'] >= 4 << 20, proc.stderr
    print(json.dumps(dict(stdout=proc.stdout, trace=proc.stderr, accounting=account, parser_checks='PASS'), indent=2))
