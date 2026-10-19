"""Compile a cold-file pure-copy fixture; prove earlier synchronization reads stay unknown."""
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
line = 'FNARENA_ACTIVATION_IO start=2.000 end=3.000 tid=2 read_bytes=4096 offset=0 bytes=4096 scope=owned_output_memcpy path=/test/llama-moe-output-Ab1234 (deleted)'
assert activation_copies(line, 1, 4, 8192, '/test')['activation_copy_bytes'] == 4096
for text, start, end, total in [
    (line, 2.1, 4, 8192), (line, 1, 4, 2048), (line + '\n' + line, 1, 4, 8192),
    (line.replace('owned_output_memcpy', 'nextn_output_copy'), 1, 4, 8192),
    (line.replace('bytes=4096 scope', 'bytes=0 scope'), 1, 4, 8192),
    (line.replace('/test/', '/weights/'), 1, 4, 8192),
    (line.replace('llama-moe-output-Ab1234', 'token_embd.weight'), 1, 4, 8192),
    ('FNARENA_ACTIVATION_IO_ERROR tid=2', 1, 4, 8192),
]:
    try:
        activation_copies(text, start, end, total, '/test')
    except ValueError:
        pass
    else:
        raise AssertionError('invalid attribution accepted')
with tempfile.TemporaryDirectory(prefix='copy-trace-test-', dir=root / 'ram-cut') as directory:
    work = Path(directory)
    (work / 'mock.cpp').write_text('''
#include <cerrno>
#include <cstring>
#include <cstddef>
extern "C" void llama_copy_moe_output(void * dst, const void * src, size_t bytes) {
    memcpy(dst, src, bytes);
    errno = EAGAIN;
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
extern "C" void llama_copy_moe_output(void *, const void *, size_t);
int main() {
    const size_t bytes = 4u << 20;
    char name[] = "llama-moe-output-XXXXXX";
    char other[] = "synchronization-input-XXXXXX";
    int fd = mkstemp(name), weight = mkstemp(other);
    if (fd < 0 || weight < 0) { return 2; }
    unlink(name); unlink(other);
    std::vector<uint32_t> original(bytes / 4), output(bytes / 4);
    for (size_t i = 0; i < original.size(); ++i) { original[i] = uint32_t(i * 2654435761u); }
    for (int file : {fd, weight}) {
        if (write(file, original.data(), bytes) != (ssize_t)bytes || fsync(file) || posix_fadvise(file, 0, bytes, POSIX_FADV_DONTNEED)) { return 3; }
    }
    void * data = mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
    void * sync_input = mmap(nullptr, bytes, PROT_READ, MAP_SHARED, weight, 0);
    if (data == MAP_FAILED || sync_input == MAP_FAILED) { return 4; }
    // Simulate CPU model work during synchronization, BEFORE entering the pure copy boundary.
    memcpy(output.data(), sync_input, bytes);
    if (output != original) { return 5; }
    errno = 73;
    llama_copy_moe_output(output.data(), data, bytes);
    if (errno != EAGAIN || output != original) { return 6; }
    // A non-owned source must produce an attribution error, preserving copy behavior/errno.
    if (getenv("BAD_SOURCE")) {
        llama_copy_moe_output(output.data(), sync_input, bytes);
        if (errno != EAGAIN || output != original) { return 7; }
    }
    munmap(data, bytes); munmap(sync_input, bytes); close(fd); close(weight);
    puts("exact bytes, errno and outside-scope synchronization: OK");
}
''')
    subprocess.run(['g++', '-shared', '-fPIC', '-O2', str(work / 'mock.cpp'), '-o', str(work / 'libmock.so')], check=True)
    subprocess.run(['g++', '-O2', str(work / 'driver.cpp'), '-L' + str(work), '-lmock', '-Wl,-rpath,' + str(work), '-o', str(work / 'llama.cpp-copy-fixture')], check=True)
    start = time.time()
    proc = subprocess.run([str(work / 'llama.cpp-copy-fixture')], cwd=work, env=dict(os.environ, LD_PRELOAD=str(tracer), LLAMA_MOE_ARENA_OUTPUT_DIR=str(work)), text=True, capture_output=True, check=True)
    end = time.time()
    account = activation_copies(proc.stderr, start, end, 8 << 20, work)
    assert account['activation_copy_bytes'] == 4 << 20, proc.stderr
    bad = subprocess.run([str(work / 'llama.cpp-copy-fixture')], cwd=work, env=dict(os.environ, LD_PRELOAD=str(tracer), LLAMA_MOE_ARENA_OUTPUT_DIR=str(work), BAD_SOURCE='1'), text=True, capture_output=True, check=True)
    assert 'FNARENA_ACTIVATION_IO_ERROR' in bad.stderr
    print(json.dumps(dict(stdout=proc.stdout, trace=proc.stderr, accounting=account, parser_checks='PASS', non_owned_source_rejected=True), indent=2))
