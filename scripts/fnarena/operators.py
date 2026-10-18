"""Run the CUDA cache suite with host and whole-card resource limits."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

bundle = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
output.mkdir(parents=True, exist_ok=False)
assert os.environ.get('PARENT_HOLDS_GPU_LOCK') == '1'
assert subprocess.check_output(['systemctl', '--user', 'is-active', 'llama.cpp-sysguard'], text=True).strip() == 'active'
environment = dict(os.environ, LD_LIBRARY_PATH=str(bundle / 'bin'), GGML_MOE_CACHE_TEST_TMPDIR=str(output))
manifest = json.loads((bundle / 'manifest.json').read_text())
for name, record in manifest['libraries'].items():
    with (bundle / 'bin' / name).open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == record['sha256'], name
command = [str(bundle / 'bin/llama.cpp-test-moe-cache')]
started = time.monotonic()
reason = None
samples = []
with (output / 'test.log').open('w') as log:
    process = subprocess.Popen(command, env=environment, stdout=log, stderr=log, start_new_session=True)
    try:
        while process.poll() is None:
            available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))) * 1024
            used, limit = map(float, subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,power.limit', '--format=csv,noheader,nounits'], text=True).strip().split(','))
            samples.append({'elapsed_s': time.monotonic() - started, 'available_bytes': available, 'vram_mib': used})
            if available < 10 * 2**30 or used > 23552 or limit != 350:
                reason = 'host/card/power limit'
            if any(shutil.disk_usage(path).free < floor * 10**9 for path, floor in (('/', 40), ('/mnt/models', 15), ('/mnt/overflow', 50))):
                reason = 'disk floor'
            if time.monotonic() - started > 540:
                reason = '540-second operator limit'
            if reason:
                break
            time.sleep(.5)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
text = (output / 'test.log').read_text(errors='replace')
passed = process.returncode == 0 and reason is None and 'cache-arena-stride: OK' in text
result = {'status': 'passed' if passed else 'failed', 'source_commit': manifest['source_commit'], 'returncode': process.returncode, 'guard': reason, 'samples': samples, 'command': command}
(output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({key: value for key, value in result.items() if key != 'samples'}))
sys.exit(0 if passed else 1)
