"""Check the preloaded allocation tracer without reading any model weights."""
import concurrent.futures
import ctypes
import errno
import os
from pathlib import Path
import sys
import tempfile

assert any(str(Path(p).resolve()) in Path('/proc/self/maps').read_text() for p in os.environ.get('LD_PRELOAD', '').split(':') if p), 'preload the tracer library'
lib = ctypes.CDLL(None, use_errno=True)
allocate = lib.posix_fallocate
allocate.argtypes = [ctypes.c_int, ctypes.c_long, ctypes.c_long]
allocate.restype = ctypes.c_int
create = lib.mkstemp
create.argtypes = [ctypes.c_char_p]
create.restype = ctypes.c_int
remove = lib.unlink
remove.argtypes = [ctypes.c_char_p]
remove.restype = ctypes.c_int
ctypes.set_errno(73)
assert allocate(-1, 0, 4096) == errno.EBADF and ctypes.get_errno() == 73
with tempfile.TemporaryDirectory(dir=sys.argv[1]) as directory:
    def check(i):
        name = ctypes.create_string_buffer(os.fsencode(directory + '/llama-moe-output-XXXXXX'))
        fd = create(name)
        assert fd >= 0
        try:
            os.write(fd, b'known activation bytes')
            ctypes.set_errno(73)
            assert allocate(fd, 0, 1024 * 1024) == 0 and ctypes.get_errno() == 73
            assert os.fstat(fd).st_size == 1024 * 1024
            assert os.pread(fd, 22, 0) == b'known activation bytes'
            assert os.pread(fd, 16, 512 * 1024) == bytes(16)
        finally:
            os.close(fd)
            assert remove(name) == 0
        assert remove(name) == -1 and ctypes.get_errno() == errno.ENOENT
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(check, range(32)))
print('Allocation tracer return codes, errno, original/extended bytes and concurrent calls PASS')
