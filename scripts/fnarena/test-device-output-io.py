"""Reject unproved operations, overlap, cross-request reads and incomplete records."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
from nonweight_io import activation_device_copies
root='/test'
line='FNSHIP_NONWEIGHT_PROBE op=cuda_d2h_async_owned start=2.0 end=3.0 tid=7 result=0 read_bytes=4096 path=/test/llama-moe-output-Ab1234 (deleted)\n'
def run(text=line,start=1,end=4,total=8192):return activation_device_copies(text,start,end,total,root)
assert run()['activation_device_copy_bytes']==4096
assert run()['remaining_read_bytes']==4096
for op in ['cuda_d2h_owned','cuda_d2h_2d_owned','cuda_d2h_2d_async_owned']:assert run(line.replace('cuda_d2h_async_owned',op))['activation_device_copy_bytes']==4096
for op in ['cuda_h2d_owned','cuda_h2d_async_owned','msync_owned','close_owned','unlink_empty','pread']:assert run(line.replace('cuda_d2h_async_owned',op))['activation_device_copy_bytes']==0
assert run(start=3.1,end=4)['activation_device_copy_bytes']==0
assert run(line+line.replace('tid=7','tid=8'))['activation_device_copy_bytes']==8192
for bad in [line+line,line+line.replace('2.0','2.5'),line.replace('/test/','/weights/'),line.replace(' (deleted)',''),line.replace('Ab1234','../bad'),line.replace('result=0','result=1'),line.replace('read_bytes=4096','read_bytes=0'),line.replace('tid=7','tid=0'),line.replace('start=2.0','start=nan'),'FNSHIP_NONWEIGHT_PROBE_ERROR op=cuda_d2h_owned\n']:
 try:run(bad)
 except ValueError:pass
 else:raise AssertionError(bad)
for n in range(1,len(line)):
 try:run(line[:n])
 except ValueError:pass
 else:
  if n<len('FNSHIP_NONWEIGHT_PROBE ') or line[:n].startswith('FNSHIP_NONWEIGHT_PROBE '):raise AssertionError(('partial',n))
for kw in [dict(total=4095),dict(total=None),dict(total=True),dict(start=2.1),dict(end=2.9),dict(start=float('nan')),dict(end=0)]:
 try:run(**kw)
 except ValueError:pass
 else:raise AssertionError(kw)
print('PASS device output scope, four operations, byte conservation, thread overlap, boundaries, malformed and all partial prefixes')

parser = argparse.ArgumentParser()
parser.add_argument('--probe-source', type=Path, default=Path(__file__).with_name('trace-ship-nonweight.c'))
args = parser.parse_args()
source = args.probe_source.read_text()
# Extract the production ownership helpers, without loading or mocking CUDA.
begin = source.find('static const char *prefix=')
if begin < 0:
    begin = source.index('static int own(')
helpers = source[begin:source.index('static uint64_t reads(')]
helpers += source[source.index('static int mapped('):source.index('static void logio(')]
fixture = r"""
#define _GNU_SOURCE
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
""" + helpers + r"""
int main(int argc, char ** argv) {
    if (argc != 4) { return 2; }
    char name[4096], path[4096], reported[4096], foreign[4096];
    if (snprintf(name, sizeof(name), "%s/llama-moe-output-Ab1234", argv[1]) >= (int) sizeof(name) ||
            snprintf(path, sizeof(path), "%s/llama-moe-output-XXXXXX", argv[1]) >= (int) sizeof(path) ||
            snprintf(foreign, sizeof(foreign), "%s/llama-moe-output-Ab1234", argv[3]) >= (int) sizeof(foreign)) { return 3; }
    unsetenv("LLAMA_MOE_ARENA_OUTPUT_DIR");
    if (own(name, 0)) { return 4; }
    setenv("LLAMA_MOE_ARENA_OUTPUT_DIR", argv[1], 1);
    if (!own(name, 0) || own(name, 1) || own(foreign, 0)) { return 10; }
    size_t length = strlen(name);
    strcat(name, " (deleted)");
    if (!own(name, 1) || own(name, 0)) { return 11; }
    name[length - 1] = '!';
    if (own(name, 1)) { return 12; }
    name[length - 1] = '4'; name[length] = 0;
    strcat(name, "extra");
    if (own(name, 0)) { return 13; }
    int fd = mkstemp(path);
    if (fd < 0 || ftruncate(fd, 8192)) { return 14; }
    void * data = mmap(NULL, 8192, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (data == MAP_FAILED || mapped(data, 8192, reported) || unlink(path)) { return 15; }
    if (!mapped(data, 8192, reported) || !own(reported, 1) || mapped(data, 8193, reported) ||
            mapped((void *) (UINTPTR_MAX - 1), 4, reported) || mapped(data, 0, reported)) { return 16; }
    setenv("LLAMA_MOE_ARENA_OUTPUT_DIR", argv[2], 1);
    if (!mapped(data, 8192, reported)) { return 17; }
    setenv("LLAMA_MOE_ARENA_OUTPUT_DIR", argv[3], 1);
    if (mapped(data, 8192, reported)) { return 18; }
    setenv("LLAMA_MOE_ARENA_OUTPUT_DIR", argv[1], 1);
    void * ro = mmap(NULL, 8192, PROT_READ, MAP_SHARED, fd, 0);
    void * priv = mmap(NULL, 8192, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    void * anon = mmap(NULL, 8192, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ro == MAP_FAILED || priv == MAP_FAILED || anon == MAP_FAILED || mapped(ro, 8192, reported) ||
            mapped(priv, 8192, reported) || mapped(anon, 8192, reported)) { return 19; }
    setenv("LLAMA_MOE_ARENA_OUTPUT_DIR", "/", 1);
    if (!own("/llama-moe-output-Ab1234", 0) || own("/other/llama-moe-output-Ab1234", 0)) { return 20; }
    setenv("LLAMA_MOE_ARENA_OUTPUT_DIR", "/nonexistent-fnarena-directory", 1);
    if (own(path, 0)) { return 21; }
    munmap(anon, 8192); munmap(priv, 8192); munmap(ro, 8192); munmap(data, 8192); close(fd);
    puts("PASS production own/mapped helpers: configured path, alias, root, deletion, bounds and foreign/private/read-only refusal");
    return 0;
}
"""
with tempfile.TemporaryDirectory(prefix='llama.cpp-device-output-owner-') as directory:
    work = Path(directory)
    own_dir, foreign_dir, alias = work / 'output', work / 'output-other', work / 'alias'
    own_dir.mkdir()
    foreign_dir.mkdir()
    alias.symlink_to(own_dir, target_is_directory=True)
    (work / 'fixture.c').write_text(fixture)
    subprocess.run(['cc', '-O2', '-Wall', '-Wextra', '-Werror', str(work / 'fixture.c'), '-o', str(work / 'llama.cpp-owner-fixture')], check=True)
    subprocess.run([str(work / 'llama.cpp-owner-fixture'), str(own_dir), str(alias), str(foreign_dir)], check=True,
                   env={k: v for k, v in os.environ.items() if k != 'LD_PRELOAD'})
