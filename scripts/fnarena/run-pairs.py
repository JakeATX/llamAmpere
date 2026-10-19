"""Wait outside the card lock for arena gates, then run matched quality and speed arms."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from compare import compare

root = Path('/mnt/models/fnarena')
candidate = Path(sys.argv[1]).resolve()
previous = Path(sys.argv[2]).resolve()
prerequisite = Path(sys.argv[3]).resolve()
series = sys.argv[4]
assert series.replace('-', '').isalnum()
output = root / 'artifacts' / series
output.mkdir(parents=True, exist_ok=False)
harness = candidate / 'harness'
environment = dict(os.environ, XDG_RUNTIME_DIR='/run/user/1000', DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus')
steps = []
pairs = []


def persist(status, **extra):
    (output / 'result.json').write_text(json.dumps({'status': status, 'steps': steps, 'pairs': pairs, 'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), **extra}, indent=2) + '\n')


def arm(label, bundle, options, full=False):
    # Each arm acquires and releases its own lease; prerequisites never own the card.
    seconds = 1490 if full else 590
    settings = {'SHIP_SERVER': str(bundle / 'bin/llama.cpp-server'), 'SHIP_LIB': str(bundle / 'bin'),
                'SHIP_CELL': series + '-' + label, 'SHIP_PROFILE': str(root / 'data/moe/flash-next.profile'),
                'SHIP_PROMPT': '100000' if full else '2048', 'SHIP_PREDICT': '5120' if full else '32',
                'SHIP_RELOAD': '0', 'SHIP_LIMIT': str(seconds - 10), 'SHIP_RESERVE_MB': '1024' if full else '2560',
                'PARENT_HOLDS_GPU_LOCK': '1', 'SHIP_CONFIG': str(harness / 'output-file.json'), **options}
    command = ['flock', '/home/jake-k/qwen38-bench/.locks/gpu.lock', 'systemd-run', '--user', '--wait',
               '--unit=llama.cpp-fnarena-' + series + '-' + label, '-p', 'CPUWeight=100' if full else 'CPUWeight=10',
               '-p', 'MemoryMax=50G', '-p', 'TimeoutStopSec=10', '-p', 'LimitCORE=0', 'env',
               'XDG_RUNTIME_DIR=/run/user/1000', 'DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus',
               *[key + '=' + value for key, value in settings.items()],
               'timeout', '--signal=TERM', '--kill-after=10', str(seconds), sys.executable, str(harness / 'gate.py'), 'flashnext']
    with (output / (label + '.unit.log')).open('w') as log:
        completed = subprocess.run(command, env=environment, stdout=log, stderr=log)
    result_path = root / 'artifacts' / (series + '-' + label) / 'result.json'
    steps.append({'label': label, 'command': command, 'unit_returncode': completed.returncode, 'result': str(result_path)})
    persist('running')
    assert completed.returncode == 0
    result = json.loads(result_path.read_text())
    assert result['status'] == 'passed' and not result['violations'], str(result_path)
    return result_path


try:
    persist('waiting-for-arena-gate', prerequisite=str(prerequisite))
    while not prerequisite.exists():
        time.sleep(5)
    gate = json.loads(prerequisite.read_text())
    assert gate['status'] == 'passed', 'arena operator/load/restart prerequisite failed'
    while not (previous / 'manifest.json').exists():
        persist('waiting-for-fixed-baseline', previous=str(previous))
        time.sleep(5)
    manifest = json.loads((candidate / 'manifest.json').read_text())
    for bundle in (candidate, previous):
        revision = json.loads((bundle / 'manifest.json').read_text())['source_commit']
        assert subprocess.run(['git', '-C', str(root), 'merge-base', '--is-ancestor',
                               '5c7567cea674855f5aff801bc4bc06e37e779837', revision]).returncode == 0, 'known-broken scheduler baseline refused'
    assert gate['source_commit'] == manifest['source_commit']
    cache_result = root / 'artifacts' / (prerequisite.parent.name + '-model') / 'result.json'
    cache = json.loads(cache_result.read_text())
    assert cache['status'] == 'passed' and cache['reload_checked'] and not cache['violations']
    for name in ('prompt.bin', 'prompt.bin.ckpt', 'prompt.bin.dft'):
        assert cache['saved_files'][name]['bytes'] > 0, 'missing main/MTP/checkpoint cache evidence'
        assert cache['serialized_bytes_equal'][name], 'MTP cache bytes differ'
    (output / 'cache-proof.json').write_text(json.dumps({'result': str(cache_result), 'sha256': hashlib.sha256(cache_result.read_bytes()).hexdigest()}, indent=2) + '\n')
    for name, digest in json.loads((harness / 'manifest.json').read_text()).items():
        assert hashlib.sha256((harness / name).read_bytes()).hexdigest() == digest, name
    baseline_quality = arm('quality-previous', previous, {'ARENA_BASELINE': '1', 'ARENA_MODE': 'off', 'ARENA_TEACHER': '1'})
    candidate_quality = arm('quality-candidate', candidate, {'ARENA_MODE': 'on', 'ARENA_TEACHER': '1', 'SHIP_PREVIOUS_KL': str(baseline_quality.parent / 'logprobs.npy')})
    quality = json.loads(candidate_quality.read_text())['kl_previous_to_current']
    for index in range(16):
        pair = {}
        order = ('previous', 'candidate') if index % 2 == 0 else ('candidate', 'previous')
        for side in order:
            pair[side] = str(arm(f'pair{index+1}-{side}', candidate if side == 'candidate' else previous,
                                 {'ARENA_MODE': 'on'} if side == 'candidate' else {'ARENA_BASELINE': '1', 'ARENA_MODE': 'off'}, full=True))
        pairs.append(pair)
        persist('running', quality=quality)
        if len(pairs) in (8, 16):
            comparison = compare(pairs)
            persist('running', quality=quality, comparison=comparison)
            if comparison['relative_speed_passed']:
                break
    persist('complete', quality=quality, comparison=compare(pairs))
except Exception as error:
    persist('failed', error=repr(error))
    raise
