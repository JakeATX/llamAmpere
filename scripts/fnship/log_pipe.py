"""Forward owned server logs without filesystem writes in the server."""
import os
from pathlib import Path
import select
import threading


class ServerLogPipe:
    """Keep filesystem log writes in the parent, outside server I/O counters."""
    def __init__(self, pipe, output):
        self.pipe = pipe
        self.output = output
        self.lock = threading.Lock()
        self.eof = False
        self.error = None
        self.bytes = 0
        self.read_bytes = 0
        os.set_blocking(pipe.fileno(), False)
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    @staticmethod
    def thread_reads():
        lines = Path('/proc/thread-self/io').read_text().splitlines()
        return int(next(line.split()[1] for line in lines if line.startswith('read_bytes:')))

    def sync(self):
        with self.lock:
            if self.error is not None:
                raise RuntimeError('server log forwarding failed: ' + self.error)
            for _ in range(64):
                if self.eof:
                    break
                try:
                    data = os.read(self.pipe.fileno(), 65536)
                except BlockingIOError:
                    break
                if not data:
                    self.eof = True
                    break
                before = self.thread_reads()
                offset = 0
                while offset < len(data):
                    count = os.write(self.output.fileno(), memoryview(data)[offset:])
                    if count <= 0:
                        raise RuntimeError('short server log write')
                    offset += count
                after = self.thread_reads()
                if after < before:
                    raise RuntimeError('server log I/O counter regressed')
                self.bytes += len(data)
                self.read_bytes += after - before

    def run(self):
        try:
            while not self.eof:
                select.select([self.pipe], [], [], 0.2)
                self.sync()
        except Exception as error:
            with self.lock:
                self.error = repr(error)

    def finish(self):
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError('server log pipe did not reach EOF after owned-server cleanup')
        self.pipe.close()
        if self.error is not None:
            raise RuntimeError('server log forwarding failed: ' + self.error)
        if not self.eof:
            raise RuntimeError('server log pipe was not fully drained')
        return dict(transport='parent-drained pipe', forwarded_bytes=self.bytes,
                    parent_log_write_read_bytes=self.read_bytes,
                    scope='Whole server lifetime, including load and cleanup. Parent writes only bytes received from the owned pipe to server.log; no server request bytes exempted.')
