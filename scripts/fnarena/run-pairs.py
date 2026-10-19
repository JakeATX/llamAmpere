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
runner_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def persist(status, **extra):
    (output / 'result.json').write_text(json.dumps({'status': status, 'steps': steps, 'pairs': pairs, 'runner_sha256': runner_sha256, **extra}, indent=2) + '\n')


def arm(label, bundle, options, full=False):
    # Each arm acquires and releases its own lease; prerequisites never own the card.
    seconds = 1490 if full else 590
    settings = {'SHIP_SERVER': str(bundle / 'bin/llama.cpp-server'), 'SHIP_LIB': str(bundle / 'bin'),
                'SHIP_CELL': series + '-' + label, 'SHIP_PROFILE': str(root / 'data/moe/flash-next.profile'),
                'SHIP_PROMPT': '100000' if full else '2048', 'SHIP_PREDICT': '5120' if full else '32',
                'SHIP_RELOAD': '0', 'SHIP_LIMIT': str(seconds - 10), 'SHIP_RESERVE_MB': '1024' if full else '2560',
                'PARENT_HOLDS_GPU_LOCK': '1', 'SHIP_CONFIG': str(harness / 'output-file.json'), **options}
    # Frozen e8 still refuses to pin unless 10 GiB will remain. A 2K arena arm
    # that starts below about 52 GiB repeats the gate7 miss. Recheck after the lock.
    arena_on = options.get('ARENA_MODE', 'on') != 'off'
    need_kib = str(52 * 1024 * 1024) if arena_on else '0'
    command = ['bash', str(root / 'scripts/fnarena/ram-locked-run.sh'), need_kib,
               'systemd-run', '--user', '--wait', '--collect',
               '--unit=llama.cpp-fnarena-' + series + '-' + label, '-p', 'CPUWeight=100' if full else 'CPUWeight=10',
               '-p', 'MemoryMax=50G', '-p', 'TimeoutStopSec=10', '-p', 'LimitCORE=0', 'env',
               'XDG_RUNTIME_DIR=/run/user/1000', 'DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus',
               *[key + '=' + value for key, value in settings.items()],
               'timeout', '--signal=TERM', '--kill-after=10', str(seconds), sys.executable, str(harness / 'gate.py'), 'flashnext']
    while True:
        with (output / (label + '.unit.log')).open('w') as log:
            completed = subprocess.run(command, env=environment, stdout=log, stderr=log)
        if completed.returncode == 75 and arena_on:
            persist('waiting-for-ram', label=label)
            time.sleep(5)
            continue
        break
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
    baseline_revision = json.loads((previous / 'manifest.json').read_text())['source_commit']
    while True:
        try:
            baseline_gates = json.loads((previous / 'gates.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            baseline_gates = {}
        assert baseline_gates.get('source_commit', baseline_revision) == baseline_revision, 'wrong baseline proof'
        if baseline_gates.get('probability_requests_100k') == 16:
            assert baseline_gates['saved_cache_bytes_exact'] and baseline_gates['flash_8192_128_reload'] == 'passed'
            assert baseline_gates['targeted_cuda_cases'] >= 163
            for evidence in baseline_gates['evidence']:
                assert hashlib.sha256(Path(evidence['path']).read_bytes()).hexdigest() == evidence['sha256']
            break
        persist('waiting-for-baseline-regression', previous=str(previous))
        time.sleep(5)
    (output / 'baseline-proof.json').write_text(json.dumps(baseline_gates, indent=2) + '\n')
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
    # Full 100K+5120 arms are timed speed runs. The owner rule is that those
    # use BEST_CONFIG.md, which this frozen arena bundle does not. Zero skips them.
    full_pairs = int(os.environ.get('ARENA_FULL_PAIRS', '0'))
    assert full_pairs in (0, 8, 16), 'planned looks are 8 and 16, or 0 for quality only'
    for index in range(full_pairs):
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
    if full_pairs:
        persist('complete', quality=quality, comparison=compare(pairs))
    else:
        persist('complete', quality=quality)
except Exception as error:
    persist('failed', error=repr(error))
    raise
