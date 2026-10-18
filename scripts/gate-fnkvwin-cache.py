#!/usr/bin/env python3
"""Gate a 100K Flash-Next split-cache cold reload and exact saved-file bytes."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


spec = importlib.util.spec_from_file_location('bench_fnkvwin', Path(__file__).with_name('bench-fnkvwin.py'))
assert spec is not None and spec.loader is not None
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def check_live(arm):
    bench.require(time.monotonic() < arm.deadline, 'cache gate deadline exceeded')
    bench.require(not arm.failure, arm.failure)
    bench.require(arm.process.poll() is None, f'owned server exited: {arm.process.returncode}')


def manifest(arm, directory, prefix):
    result = {}
    for path in sorted(directory.glob(prefix + '*')):
        bench.require(path.is_file() and not path.is_symlink(), f'unexpected cache entry: {path}')
        bench.disk_check()
        digest = hashlib.sha256()
        size = 0
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(2**20), b''):
                check_live(arm)
                digest.update(chunk)
                size += len(chunk)
        result[path.name[len(prefix):]] = {'bytes': size, 'sha256': digest.hexdigest()}
    bench.require('' in result and '.ckpt' in result, 'saved cache bundle lacks main state or checkpoint sidecar')
    bench.require(all(record['bytes'] > 0 for record in result.values()), 'saved cache file is empty')
    return result


def completion(arm, request, name):
    check_live(arm)
    arm.phase = 'inference'
    bench.write_json(arm.output / (name + '-request.json'), request)
    response = arm.http('/completion', request)
    arm.phase = name
    bench.write_json(arm.output / (name + '-response.json'), response)
    check_live(arm)
    tokens = response.get('tokens', [])
    bench.require(len(tokens) == 8 and all(type(token) is int for token in tokens), f'{name}: expected 8 token IDs')
    bench.require(response.get('tokens_predicted') == 8, f'{name}: wrong generation count')
    bench.require(not response.get('truncated'), f'{name}: context was truncated')
    return response


def action(arm, operation, filename):
    check_live(arm)
    arm.phase = operation
    response = arm.http('/slots/0?action=' + operation, {'filename': filename})
    bench.write_json(arm.output / (operation + '-' + filename + '.json'), response)
    check_live(arm)
    return response


def run(args):
    bench.require(args.out.resolve().is_relative_to(bench.GATES.resolve()), 'artifacts must be under /mnt/models/fnkvwin/gates')
    bench.require(os.environ.get('PARENT_HOLDS_GPU_LOCK') == '1', 'parent must hold gpu.lock and set PARENT_HOLDS_GPU_LOCK=1')
    bench.require(subprocess.check_output(['systemctl', '--user', 'is-active', 'llama.cpp-sysguard'], text=True, timeout=5).strip() == 'active', 'llama.cpp-sysguard must be active')
    bench.disk_check()
    args.out.mkdir(parents=True, exist_ok=False)
    cache = args.out.resolve() / 'cache'
    cache.mkdir()
    args.mode = 'cache'
    args.resident = 32768
    args.extra_server_args = ['--slot-save-path', str(cache), '--cache-ram', '0', '--ctx-checkpoints', '4']
    started = time.monotonic()
    deadline = started + args.timeout
    result = {'mode': 'cache', 'status': 'failed', 'resident_cells': args.resident, 'prompt_tokens': bench.PROMPT_N,
              'diagnostic_tokens': 8, 'mtp': False, 'cold_process_restart': True, 'reserve_mib': args.reserve_mib,
              'timeout_s': args.timeout, 'arms': []}
    arm = None

    def interrupted(signum, _frame):
        raise RuntimeError(f'interrupted by signal {signum}')

    prior_sigterm = signal.signal(signal.SIGTERM, interrupted)
    try:
        ids = json.loads(bench.PROMPT_IDS.read_text())
        bench.require(len(ids) == bench.PROMPT_N and all(type(token) is int for token in ids), 'prompt must contain exactly 100000 token IDs')
        result['prompt_sha256'] = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
        request = bench.payload(ids, 'kl')
        request.pop('n_probs')
        request.pop('post_sampling_probs')
        request.update(id_slot=0, cache_prompt=True)
        for label in ('cold', 'reload'):
            bench.require(time.monotonic() < deadline, 'cache gate deadline exceeded before launch')
            output = args.out / label
            output.mkdir()
            arm = bench.ServerArm(args, output)
            arm.started, arm.deadline = started, deadline
            arm_result = {'label': label}
            result['arms'].append(arm_result)
            try:
                loading = time.monotonic()
                arm.start()
                arm_result['load_s'] = time.monotonic() - loading
                log = (output / 'server.log').read_text(errors='replace')
                layers = set(re.findall(r'QSA layer (\d+):', log))
                bench.require(len(layers) == 12, f'expected 12 split QSA layers, found {len(layers)}')
                if label == 'cold':
                    initial = completion(arm, request, 'initial')
                    timings = initial.get('timings', {})
                    bench.require(timings.get('prompt_n') == bench.PROMPT_N, 'server did not evaluate the full 100000-token prompt')
                    bench.require(timings.get('cache_n', 0) == 0, 'initial prompt unexpectedly reused cache')
                    result['initial_timings'] = timings
                    saved = action(arm, 'save', 'prompt.bin')
                    bench.require(bench.PROMPT_N <= saved.get('n_saved', 0) <= bench.PROMPT_N + 8, 'saved state does not contain the 100K prompt')
                    result['save'] = saved
                    result['saved_files'] = manifest(arm, cache, 'prompt.bin')
                    continuation = dict(request, prompt=ids + initial['tokens'])
                    live = completion(arm, continuation, 'live-continuation')
                    result['live_continuation_timings'] = live.get('timings', {})
                else:
                    restored = action(arm, 'restore', 'prompt.bin')
                    result['restore'] = restored
                    bench.require(restored.get('n_restored') == saved['n_saved'], 'restored token count differs')
                    result['resave'] = action(arm, 'save', 'roundtrip.bin')
                    result['roundtrip_files'] = manifest(arm, cache, 'roundtrip.bin')
                    result['serialized_bytes_equal'] = result['saved_files'] == result['roundtrip_files']
                    bench.require(result['serialized_bytes_equal'], 'saved cache file sizes or byte hashes differ after cold reload')
                    bench.require(manifest(arm, cache, 'prompt.bin') == result['saved_files'], 'original saved cache files changed')
                    reloaded = completion(arm, continuation, 'reload-continuation')
                    result['reload_continuation_timings'] = reloaded.get('timings', {})
                    result['continuation_tokens_exact'] = live['tokens'] == reloaded['tokens']
                    result['continuation_comparison'] = 'Greedy token equality is diagnostic; exact serialized bytes are required.'
                    maximum_eval = len(continuation['prompt']) - saved['n_saved'] + 1
                    for name, response in (('live', live), ('reload', reloaded)):
                        evaluated = response.get('timings', {}).get('prompt_n')
                        bench.require(type(evaluated) is int and 0 < evaluated <= maximum_eval, f'{name} continuation reprocessed the saved prefix')
                check_live(arm)
            finally:
                arm_result.update(arm.finish())
                if arm.failure:
                    raise RuntimeError(arm.failure)
                arm = None
        result['status'] = 'complete'
    except Exception as error:
        result['error'] = str(error)
    finally:
        if arm is not None:
            arm.finish()
        signal.signal(signal.SIGTERM, prior_sigterm)
        result['total_wall_s'] = time.monotonic() - started
        bench.write_json(args.out / 'result.json', result)
        print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if result['status'] == 'complete' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8095)
    parser.add_argument('--cache-budget-mib', type=int, default=0, help='0 selects --moe-cache auto')
    parser.add_argument('--reserve-mib', type=int, default=3072)
    parser.add_argument('--timeout', type=float, default=600, help='shared deadline for both server processes, at most 600 seconds')
    args = parser.parse_args()
    bench.require(0 < args.timeout <= 600, 'timeout must be positive and at most 600 seconds')
    bench.require(args.cache_budget_mib >= 0 and args.reserve_mib >= 0, 'cache budget and reserve must be nonnegative')
    bench.require(1 <= args.port <= 65535, 'invalid port')
    return run(args)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as error:
        print(f'FNKVWIN cache gate: {error}', file=sys.stderr)
        sys.exit(1)
