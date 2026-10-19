"""Compile a cold-file pure-copy fixture; prove earlier synchronization reads stay unknown."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from nonweight_io import activation_copies, activation_layout

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


# Compile the exact production layout class with only its tensor descriptors stubbed.
context = (root / 'src/llama-context.cpp').read_text()
layout = context[context.index('class llama_io_write_layout :'):context.index('const llama_memory_recurrent * llama_context::capture_mem()')]
with tempfile.TemporaryDirectory(prefix='layout-trace-test-', dir=root / 'ram-cut') as directory:
    work = Path(directory)
    arena = (root / 'src/llama-moe-arena.cpp').read_text()
    wrappers = arena[arena.index('void llama_copy_moe_output('):arena.index('llama_moe_output::llama_moe_output(')]
    (work / 'operations.cpp').write_text('#include <cstring>\n#include <cstddef>\nextern "C" {\n' + wrappers + '}\n')
    header = r'''
#include <cerrno>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
extern "C" void llama_write_moe_output(void *, const void *, size_t);
extern "C" int llama_compare_moe_output(const void *, const void *, size_t);
struct ggml_tensor {};
struct llama_context { struct ckpt_write { ggml_tensor * t; size_t offset, size, blob_off; int il; }; };
struct llama_io_write_i {
    virtual ~llama_io_write_i() = default;
    virtual void write(const void *, size_t) = 0;
    virtual void write_tensor(ggml_tensor *, size_t, size_t) = 0;
    virtual size_t n_bytes() = 0;
};
'''
    driver = r'''
static uint64_t read_bytes() {
    FILE * f = fopen("/proc/self/io", "r");
    char line[256]; unsigned long long value = 0;
    if (!f) { exit(21); }
    while (fgets(line, sizeof(line), f)) { if (sscanf(line, "read_bytes: %llu", &value) == 1) { break; } }
    fclose(f); return value;
}
int main() {
    const size_t bytes = 4u << 20;
    char name[] = "llama-moe-output-XXXXXX", other[] = "model-operand-XXXXXX";
    int fd = mkstemp(name), other_fd = mkstemp(other);
    if (fd < 0 || other_fd < 0) { return 2; }
    unlink(name); unlink(other);
    std::vector<uint8_t> original(bytes);
    for (size_t i = 0; i < bytes; ++i) { original[i] = uint8_t(i * 29 + 7); }
    for (int file : {fd, other_fd}) {
        if (write(file, original.data(), bytes) != (ssize_t)bytes || fsync(file) || posix_fadvise(file, 0, bytes, POSIX_FADV_DONTNEED)) { return 3; }
    }
    auto * data = (uint8_t *)mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    auto * other_data = (uint8_t *)mmap(nullptr, bytes, PROT_READ, MAP_SHARED, other_fd, 0);
    if (data == MAP_FAILED || other_data == MAP_FAILED) { return 4; }
    std::vector<llama_context::ckpt_write> writes;
    const bool traced = getenv("RAW_LAYOUT") == nullptr;
    auto evict = [&]() {
        if (msync(data, bytes, MS_SYNC) || madvise(data, bytes, MADV_DONTNEED) || posix_fadvise(fd, 0, bytes, POSIX_FADV_DONTNEED)) { exit(5); }
    };
    evict();
    uint64_t before = read_bytes();
    llama_io_write_layout check(data, bytes, writes, true, traced);
    check.write(original.data(), bytes);
    if (!check.matched() || check.n_bytes() != bytes) { return 6; }
    uint64_t compare_reads = read_bytes() - before;
    evict();
    before = read_bytes();
    llama_io_write_layout fill(data, bytes, writes, false, traced);
    fill.write(original.data(), 1);
    uint64_t write_reads = read_bytes() - before;
    if (memcmp(data, original.data(), bytes)) { return 7; }
    llama_io_write_layout mismatch(data, bytes, writes, true, traced);
    original[0] ^= 1;
    mismatch.write(original.data(), 1);
    if (mismatch.matched()) { return 8; }
    // A failed comparison must keep the later comparison short-circuited.
    mismatch.write(nullptr, 1);
    bool rejected = false;
    try { mismatch.write(original.data(), bytes); } catch (const std::runtime_error &) { rejected = true; }
    if (!rejected) { return 9; }
    errno = 91;
    llama_write_moe_output(data, original.data(), 0);
    if (llama_compare_moe_output(data, original.data(), 0) || errno != 91) { return 10; }
    // Resident read-only metadata must remain usable without granting it any read exemption.
    static constexpr uint32_t magic = 0xaf143cd8;
    uint32_t magic_copy;
    memcpy(&magic_copy, &magic, sizeof(magic));
    llama_write_moe_output(data, &magic, sizeof(magic));
    if (llama_compare_moe_output(data, &magic, sizeof(magic)) || memcmp(data, &magic_copy, sizeof(magic))) { return 11; }
    memcpy(data, original.data(), bytes);
    if (getenv("BAD_OPERAND")) { llama_compare_moe_output(data, other_data, bytes); }
    if (getenv("BAD_OWNED")) { llama_compare_moe_output(other_data, original.data(), bytes); }
    if (getenv("COLD_MAGIC")) {
        evict();
        llama_write_moe_output(data, &magic, sizeof(magic));
        if (memcmp(data, &magic_copy, sizeof(magic))) { return 12; }
    }
    printf("{\"compare_reads\":%llu,\"write_reads\":%llu,\"exact\":true}\n", (unsigned long long)compare_reads, (unsigned long long)write_reads);
    munmap(data, bytes); munmap(other_data, bytes); close(fd); close(other_fd);
}
'''
    (work / 'driver.cpp').write_text(header + layout + driver)
    subprocess.run(['g++', '-shared', '-fPIC', '-O2', str(work / 'operations.cpp'), '-o', str(work / 'liboperations.so')], check=True)
    subprocess.run(['g++', '-Wall', '-Wextra', '-Werror', '-O2', str(work / 'driver.cpp'), '-L' + str(work), '-loperations', '-Wl,-rpath,' + str(work), '-o', str(work / 'llama.cpp-layout-fixture')], check=True)
    evidence = {}
    for mode in ['raw', 'traced', 'bad', 'bad_owned', 'cold_magic']:
        env = dict(os.environ, LD_PRELOAD=str(tracer), LLAMA_MOE_ARENA_OUTPUT_DIR=str(work))
        if mode == 'raw': env['RAW_LAYOUT'] = '1'
        if mode == 'bad': env['BAD_OPERAND'] = '1'
        if mode == 'bad_owned': env['BAD_OWNED'] = '1'
        if mode == 'cold_magic': env['COLD_MAGIC'] = '1'
        start = time.time()
        proc = subprocess.run([str(work / 'llama.cpp-layout-fixture')], cwd=work, env=env, text=True, capture_output=True, check=True)
        end = time.time()
        values = json.loads(proc.stdout)
        if mode in ['bad', 'bad_owned', 'cold_magic']:
            assert 'FNARENA_LAYOUT_IO_ERROR' in proc.stderr
            errors = [line for line in proc.stderr.splitlines() if 'FNARENA_LAYOUT_IO_ERROR' in line]
            expected = {
                'bad': ('op=compare', 'owned_valid=1 metadata_anonymous=0', 'metadata_path=' + str(work) + '/model-operand-'),
                'bad_owned': ('op=compare', 'owned_valid=0 metadata_anonymous=1', 'metadata_path='),
                'cold_magic': ('op=write', 'bytes=4 owned_valid=1 metadata_anonymous=0', 'metadata_path=' + str(work / 'llama.cpp-layout-fixture')),
            }[mode]
            assert any(all(field in line for field in expected) for line in errors), proc.stderr
            try: activation_layout(proc.stderr, start, end, 8 << 20, work)
            except ValueError: pass
            else: raise AssertionError('file-backed metadata accepted')
        else:
            account = activation_layout(proc.stderr, start, end, 8 << 20, work)
            expected = values['compare_reads'] + values['write_reads'] if mode == 'traced' else 0
            assert account['activation_layout_bytes'] == expected, (values, proc.stderr, account)
            assert values['compare_reads'] >= 4 << 20, values
        evidence[mode] = dict(values=values, trace=proc.stderr)
    sample = 'FNARENA_LAYOUT_IO op=compare start=2.0 end=3.0 tid=2 read_bytes=4096 offset=0 bytes=4096 path=' + str(work) + '/llama-moe-output-Ab1234 (deleted)\n'
    for invalid in [sample + sample, sample.replace('op=compare', 'op=callback'), sample[:-1], sample.replace('llama-moe-output-Ab1234', 'token_embd.weight')]:
        try: activation_layout(invalid, 1, 4, 8192, work)
        except ValueError: pass
        else: raise AssertionError('invalid layout attribution accepted')
    print(json.dumps(dict(layout_fixture=evidence, production_class=True, exact_bytes=True, mismatch_bounds_short_circuit='PASS', parser_rejections='PASS'), indent=2))
