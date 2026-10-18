#!/usr/bin/env python3
"""Run predeclared 100K+5120 KV-window pairs after explicit correctness gates."""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import time


ROOT = Path('/mnt/models/fnkvwin')
GPU_LOCK = Path('/home/jake-k/qwen38-bench/.locks/gpu.lock')
HARNESS = Path(__file__).with_name('bench-fnkvwin.py').resolve()
spec = importlib.util.spec_from_file_location('bench_fnkvwin', HARNESS)
assert spec is not None and spec.loader is not None
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with path.open() as handle:
        return json.load(handle)


def positive(value, name):
    bench.require(type(value) in (int, float) and math.isfinite(value) and value > 0, f'invalid {name}')
    return float(value)


def validate_gate(kind, data):
    status = data.get('status')
    if status is not None:
        bench.require(status in ('complete', 'passed', 'pass'), f'{kind} gate did not pass')
    bench.require(not data.get('error') and not data.get('guard_failure') and not data.get('stop_reason'), f'{kind} gate reports a failure')
    if kind == 'operator':
        commands = data.get('commands', [])
        summary = data.get('test_summary', {})
        if commands:
            bench.require(status is not None and all(command.get('returncode') == 0 for command in commands), 'operator run-unit did not complete successfully')
            for executable in ('test-sjkvarn-attn', 'test-sjkvarn-state'):
                bench.require(any(executable in ' '.join(command.get('command', [])) and '--resident' in command.get('command', []) for command in commands), f'operator evidence lacks the resident {executable} gate')
        else:
            bench.require(status is not None and summary.get('executed', 0) > 0 and summary.get('passed') == summary['executed'], 'operator evidence has zero or missing executed cases')
    elif kind == 'kl':
        bench.require(data.get('prompt_tokens') == 100000, 'KL evidence is not a recorded 100K comparison')
        bench.require(data.get('matched_tokens') is True and data.get('infinite_kl') is False, 'KL evidence lacks matched finite distributions')
        bench.require(type(data.get('n_vocab')) is int and data['n_vocab'] > 0 and len(data.get('steps', [])) == 8, 'KL evidence lacks eight full-vocabulary steps')
        for field in ('mean_kl_nats', 'max_kl_nats'):
            value = data.get(field)
            bench.require(type(value) in (int, float) and math.isfinite(value) and value >= -1e-6, f'invalid KL statistic: {field}')
        for step in data['steps']:
            value = step.get('kl_nats')
            bench.require(step.get('infinite_kl') is False and type(value) in (int, float) and math.isfinite(value) and value >= -1e-6, 'invalid KL step')
    elif kind == 'niah':
        bench.require(status is not None and data.get('mode') == 'niah' and data.get('resident_cells') == 32768, 'NIAH evidence is not a resident gate')
        bench.require(data.get('timings', {}).get('prompt_n') == 100000, 'NIAH evidence is not 100K')
        matches = data.get('needle_matches', {})
        bench.require(len(matches) >= 3 and all(value is True for value in matches.values()), 'NIAH retrieval did not pass')
    elif kind == 'cache':
        bench.require(status is not None and data.get('mode') == 'cache' and data.get('resident_cells') == 32768, 'cache evidence is not a resident gate')
        bench.require(data.get('prompt_tokens') == 100000 and data.get('serialized_bytes_equal') is True, 'cache gate lacks exact 100K saved bytes')
        bench.require(data.get('saved_files') and data['saved_files'] == data.get('roundtrip_files'), 'cache file manifests differ or are missing')
        bench.require(data.get('save', {}).get('n_saved', 0) >= 100000 and data.get('restore', {}).get('n_restored') == data['save']['n_saved'], 'cache gate restored the wrong token count')
    return 'complete'


def runtime_snapshot(binary):
    paths = {binary}
    for prefix in ('libllama', 'libggml', 'libmtmd'):
        paths.update(path.resolve(strict=True) for path in binary.parent.glob(prefix + '*.so*') if path.is_file())
    bench.require(any(path.name.startswith('libllama') for path in paths), 'server build lacks libllama')
    return [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path)} for path in sorted(paths)]


def validate_runtime(actual, expected, binary):
    expected_hashes = {item['path']: item['sha256'] for item in expected}
    bench.require(actual and len(actual) == len({item['path'] for item in actual}), 'missing or duplicate runtime identity')
    hashes = {item['path']: item['sha256'] for item in actual}
    bench.require(str(binary) in hashes and any(Path(path).name.startswith('libllama') for path in hashes), 'incomplete loaded runtime identity')
    bench.require(all(expected_hashes.get(path) == digest for path, digest in hashes.items()), 'runtime differs from predeclared build')
    return hashes


def bind_evidence(kind, path, data, expected, binary):
    identities = []
    runtime = data.get('runtime_identity')
    candidate = data.get('candidate_runtime_identity') or (runtime.get('candidate') if isinstance(runtime, dict) else None)
    if candidate:
        identities.append(('candidate_runtime_identity', candidate))
    bench.require(kind != 'kl' or candidate, 'KL comparison lacks candidate runtime identity')
    if kind != 'kl' and isinstance(runtime, list):
        identities.append(('runtime_identity', runtime))
    if kind == 'operator':
        for index, command in enumerate(data.get('commands', [])):
            if command.get('runtime_identity'):
                identities.append((f'commands[{index}].runtime_identity', command['runtime_identity']))
    runtime_files = []
    relatives = ['cold/runtime.json', 'reload/runtime.json'] if kind == 'cache' else ['runtime.json'] if kind in ('operator', 'niah') else []
    for relative in relatives:
        sibling = path.parent / relative
        if sibling.is_file():
            identities.append((str(sibling), read_json(sibling)))
            runtime_files.append({'path': str(sibling), 'sha256': sha256(sibling)})
    current_libraries = {Path(item['path']).name: item['sha256'] for item in expected
                         if Path(item['path']).name.startswith(('libllama', 'libggml', 'libmtmd'))}
    executable_hash = next(item['sha256'] for item in expected if item['path'] == str(binary))
    bindings = []
    for source, identity in identities:
        bench.require(isinstance(identity, list) and identity, f'invalid gate runtime identity: {source}')
        libraries = [item for item in identity if Path(item['path']).name.startswith(('libllama', 'libggml', 'libmtmd'))]
        bench.require(libraries, f'gate has no project-library identity: {source}')
        for item in libraries:
            bench.require(current_libraries.get(Path(item['path']).name) == item['sha256'], f'{kind} gate used different candidate library: {item["path"]}')
        executable_match = None
        if kind != 'operator':
            executables = [item for item in identity if item not in libraries]
            bench.require(len(executables) == 1 and executables[0]['sha256'] == executable_hash, f'{kind} gate candidate executable differs from --candidate')
            executable_match = True
        bindings.append({'source': source, 'project_libraries_matched': len(libraries), 'candidate_executable_matches': executable_match})
    return {'runtime_bindings': bindings, 'related_runtime_files': runtime_files,
            'runtime_binding_note': 'Candidate hashes checked wherever emitted; operator test executable may differ; baseline identity is not compared.'}


def validate_arm(data, response, request, expected_request, resident, parent_unit):
    bench.require(data.get('status') == 'complete' and data.get('mode') == 'protocol' and data.get('resident_cells') == resident, 'incomplete or wrong protocol arm')
    bench.require(not data.get('guard_failure') and not data.get('error'), 'arm guard or runtime failure')
    bench.require(request == expected_request, 'request differs from the predeclared full protocol')
    final = response.get('final', {})
    tokens = response.get('tokens', [])
    bench.require(final.get('stop') is True and final.get('tokens_predicted') == 5120, 'missing final response or incomplete generation')
    bench.require(len(tokens) == 5120 and all(type(token) is int for token in tokens), 'stream did not deliver exactly 5120 token IDs')
    bench.require(final.get('tokens_evaluated') == 100000 and not final.get('truncated'), 'prompt was incomplete or truncated')
    timings = data.get('timings', {})
    bench.require(timings == final.get('timings') and timings.get('prompt_n') == 100000 and timings.get('cache_n', 0) == 0, 'full cold-prompt timings unavailable')
    bench.require(timings.get('predicted_n', 5120) == 5120, 'timings have the wrong generated count')
    for phase in ('before', 'after'):
        cpu = data.get('cpu_inference_' + phase, {})
        bench.require(cpu.get('cpu.weight') == '100' and cpu.get('unit') == parent_unit, 'arm CPUWeight or parent unit changed')
    bench.require(0 < data.get('peak_vram_mib', 0) <= bench.MAX_VRAM_MIB, 'missing or excessive VRAM peak')
    bench.require(data.get('min_mem_available_bytes', 0) >= bench.MIN_AVAILABLE, 'arm memory floor failed')
    prefill = positive(data.get('prefill_tok_s'), 'prefill speed')
    decode = positive(data.get('decode_tok_s'), 'decode speed')
    bench.require(prefill == positive(timings.get('prompt_per_second'), 'prompt timing') and decode == positive(timings.get('predicted_per_second'), 'decode timing'), 'reported speeds disagree with native timings')
    hits = data.get('expert_cache_last_log_stats', [])
    for entry in hits:
        bench.require(0 <= entry['hits'] <= entry['requests'] and entry['requests'] > 0 and 0 <= entry['hit_percent'] <= 100, 'invalid expert-cache counter')
    return {'prefill_tok_s': prefill, 'decode_tok_s': decode, 'ttc_s': positive(data.get('ttc_s'), 'TTC'),
            'mean_inference_power_w': positive(data.get('mean_inference_power_w'), 'GPU power'),
            'peak_vram_mib': data['peak_vram_mib'], 'expert_cache_last_log_stats': hits,
            'expert_cache_counter_scope': 'Last logged CUDA cache snapshot; may include helper/draft activity.',
            'min_mem_available_bytes': data['min_mem_available_bytes']}


def analyze(pairs):
    evidence = maximum = 1.0
    rows = []
    for index, pair in enumerate(pairs):
        baseline, candidate = pair['baseline'], pair['candidate']
        delta = candidate['decode_tok_s'] - baseline['decode_tok_s']
        evidence *= 1.9 if delta > 0 else 0.1 if delta < 0 else 1.0
        maximum = max(maximum, evidence)
        rows.append({'pair': index, **pair, 'decode_win': delta > 0, 'decode_tie': delta == 0,
                     'decode_gain_tok_s': delta, 'decode_ratio': candidate['decode_tok_s'] / baseline['decode_tok_s'],
                     'prefill_ratio': candidate['prefill_tok_s'] / baseline['prefill_tok_s'],
                     'ttc_ratio': candidate['ttc_s'] / baseline['ttc_s'], 'e_value': evidence,
                     'max_e_value': maximum, 'one_sided_anytime_p_bound': min(1.0, 1.0 / maximum)})
    mean_gain = statistics.mean(row['decode_gain_tok_s'] for row in rows) if rows else None
    bound = min(1.0, 1.0 / maximum)
    aggregates = {}
    for label in ('baseline', 'candidate'):
        if rows:
            aggregates[label] = {field: statistics.mean(row[label][field] for row in rows)
                                 for field in ('prefill_tok_s', 'decode_tok_s', 'ttc_s', 'mean_inference_power_w')}
            aggregates[label]['peak_vram_mib'] = max(row[label]['peak_vram_mib'] for row in rows)
    return {'pairs': rows, 'completed_pairs': len(rows), 'one_sided_anytime_p_bound': bound,
            'e_value': evidence, 'max_e_value': maximum, 'mean_paired_decode_gain_tok_s': mean_gain,
            'accepted_faster': bool(rows and bound <= 0.01 and mean_gain > 0), 'aggregates': aggregates,
            'absolute_ship_targets_met_by_candidate_means': bool(rows and aggregates['candidate']['decode_tok_s'] >= 143 and aggregates['candidate']['prefill_tok_s'] >= 3170),
            'assumption': 'Under the null, each next untied decode sign has conditional win probability at most 0.5; ties multiply evidence by 1.',
            'method': 'Fixed alternative 0.95 versus null 0.5, with Ville bound 1/max(E), including initial E=1.'}


def process_info(pid):
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
        fields = text[text.rfind(')') + 2:].split()
        return {'start': fields[19], 'group': int(fields[2]), 'state': fields[0]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def remember_children(pid, owned):
    info = process_info(pid)
    if info is None:
        return
    owned[pid] = info
    try:
        children = Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
    except (FileNotFoundError, ProcessLookupError):
        return
    for child in children:
        remember_children(int(child), owned)


def stop_owned(child, owned):
    if child is not None and child.poll() is None:
        remember_children(child.pid, owned)
    for sig, seconds in ((signal.SIGTERM, 3), (signal.SIGKILL, 2)):
        for pid, original in list(owned.items()):
            current = process_info(pid)
            if current is not None and current['start'] == original['start'] and current['state'] != 'Z':
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if child is not None:
                child.poll()
            live = [pid for pid, original in owned.items()
                    if (current := process_info(pid)) is not None and current['start'] == original['start'] and current['state'] != 'Z']
            if not live:
                break
            for pid in live:
                remember_children(pid, owned)
            time.sleep(0.05)
    if child is not None and child.poll() is None:
        child.wait(timeout=1)


def launch_arm(args, label, output, binary, expected_runtime):
    owned = {}
    child = None
    with GPU_LOCK.open('a') as lock:
        while True:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                bench.disk_check()
                bench.require(bench.available_bytes() >= bench.MIN_AVAILABLE, 'memory below 10 GiB while waiting for GPU lock')
                time.sleep(0.5)
        started = time.monotonic()
        try:
            bench.require(runtime_snapshot(binary) == expected_runtime, 'selected build changed since plan was recorded')
            command = ['timeout', '--kill-after=15s', '1450s', sys.executable, str(HARNESS), 'protocol',
                       '--server', str(binary), '--resident', '0' if label == 'baseline' else '32768',
                       '--out', str(output), '--timeout', '1400', '--port', str(args.port),
                       '--reserve-mib', str(args.reserve_mib), '--cache-budget-mib', str(args.cache_budget_mib)]
            env = {key: value for key, value in os.environ.items() if not key.startswith(('GGML_', 'LLAMA_', 'LD_LIBRARY_PATH'))}
            env['PARENT_HOLDS_GPU_LOCK'] = '1'
            with output.with_suffix('.launcher.log').open('x') as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while child.poll() is None:
                    remember_children(child.pid, owned)
                    bench.require(time.monotonic() - started < 1450, 'arm exceeded 1450-second lease')
                    bench.disk_check()
                    bench.require(bench.available_bytes() >= bench.MIN_AVAILABLE, 'MemAvailable below 10 GiB')
                    time.sleep(0.25)
            bench.require(child.returncode == 0, f'{label} arm failed or timed out: {child.returncode}; see {output}')
        finally:
            stop_owned(child, owned)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return {'gpu_lock_elapsed_s': time.monotonic() - started, 'launcher_returncode': child.returncode}


def validate_command(command, resident, args):
    argv = command['argv']
    env = command['environment']
    required = {'-m': str(bench.MODEL), '-md': str(bench.MTP), '--load-mode': 'mmap', '-c': '131072',
                '-b': '131072', '-ub': '12288', '-ctk': 'sjkvarn4', '-ctv': 'sjkvarn4',
                '--spec-type': 'draft-mtp-adaptive', '--moe-cache': str(args.cache_budget_mib) if args.cache_budget_mib else 'auto'}
    for flag, value in required.items():
        bench.require(argv.count(flag) == 1 and argv[argv.index(flag) + 1] == value, f'wrong protocol option: {flag}')
    normalized = argv[1:]
    if resident:
        bench.require(normalized.count('--kv-resident') == 1 and normalized[normalized.index('--kv-resident') + 1] == '32768', 'resident flag missing')
        offset = normalized.index('--kv-resident')
        normalized = normalized[:offset] + normalized[offset + 2:]
    else:
        bench.require('--kv-resident' not in normalized, 'baseline unexpectedly uses a resident window')
    bench.require(env.get('GGML_CUDA_MOE_CACHE_RESERVE_MB') == str(args.reserve_mib), 'cache reserve changed')
    for flag in ('LLAMA_MOE_CACHE_RELEASE_PROMPT', 'LLAMA_B1_CACHE_WARM', 'LLAMA_B1_LAYER_RESERVE'):
        bench.require(env.get(flag) == '1', f'protocol phase setting missing: {flag}')
    environment = {key: value for key, value in env.items() if key != 'LD_LIBRARY_PATH'}
    return {'argv': normalized, 'environment': environment}


def run(args):
    bench.require(os.environ.get('PARENT_HOLDS_GPU_LOCK') != '1', 'do not hold the GPU lock around the pairs runner; each arm acquires it independently')
    cpu = bench.cpu_accounting()
    bench.require(cpu['unit'].startswith('llama.cpp-fnkvwin-') and cpu['cpu.weight'] == '100', 'pairs require a llama.cpp-fnkvwin-* parent unit with CPUWeight=100')
    bench.require(subprocess.check_output(['systemctl', '--user', 'is-active', 'llama.cpp-sysguard'], text=True, timeout=5).strip() == 'active', 'llama.cpp-sysguard must be active')
    out = args.out.resolve()
    bench.require(out.is_relative_to(bench.GATES.resolve()), 'output must be under /mnt/models/fnkvwin/gates')
    binaries = {'candidate': args.candidate.resolve(strict=True), 'baseline': (args.baseline or args.candidate).resolve(strict=True)}
    bench.require(all(binary.is_relative_to('/mnt/models') for binary in binaries.values()), 'binaries must be under /mnt/models')
    runtimes = {label: runtime_snapshot(binary) for label, binary in binaries.items()}
    evidence = {}
    for kind in ('operator', 'kl', 'niah', 'cache'):
        path = getattr(args, kind + '_evidence').resolve(strict=True)
        data = read_json(path)
        evidence[kind] = {'path': str(path), 'sha256': sha256(path), 'validated_status': validate_gate(kind, data),
                          **bind_evidence(kind, path, data, runtimes['candidate'], binaries['candidate'])}
    ids = read_json(bench.PROMPT_IDS)
    bench.require(len(ids) == 100000 and all(type(token) is int for token in ids), 'protocol requires exactly 100000 prompt IDs')
    expected_request = bench.payload(ids, 'protocol')
    expected_prompt_hash = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
    rng = random.Random(args.order_seed)
    pairs = [{'index': index, 'order': ['baseline', 'candidate'] if rng.getrandbits(1) else ['candidate', 'baseline'],
              **{label: str(out / f'pair-{index:02d}-{label}') for label in binaries}} for index in range(args.max_pairs)]
    plan = {'order_seed': args.order_seed, 'pairs': pairs, 'max_pairs': args.max_pairs, 'primary_metric': 'decode_tok_s',
            'null_win_probability': 0.5, 'alternative_win_probability': 0.95, 'p_bound_threshold': 0.01,
            'stopping_rule': 'p_bound <= 0.01 and positive mean paired decode gain, using only complete pairs',
            'prompt_tokens': 100000, 'generated_tokens': 5120, 'temperature': 1.0, 'sampling_seed': 11,
            'prompt_sha256': expected_prompt_hash, 'correctness_evidence': evidence, 'runtime': runtimes,
            'binaries': {label: str(binary) for label, binary in binaries.items()}, 'resident_cells': {'baseline': 0, 'candidate': 32768},
            'benchmark_timeout_s': 1400, 'arm_timeout_s': 1450, 'cleanup_limit_s': 15,
            'cpu_parent': cpu, 'reserve_mib': args.reserve_mib, 'cache_budget_mib': args.cache_budget_mib,
            'harness_sha256': sha256(HARNESS), 'runner_sha256': sha256(Path(__file__)),
            'comparison_scope': 'same binary resident0 versus32768' if binaries['baseline'] == binaries['candidate'] else 'different explicit binaries; runtime changes may also affect performance'}
    bench.disk_check()
    out.mkdir(parents=True, exist_ok=False)
    bench.write_json(out / 'plan.json', plan)
    for kind, record in evidence.items():
        bench.write_json(out / (kind + '-evidence.json'), read_json(Path(record['path'])))
    completed = []
    loaded = {}
    configuration = None
    result = {'status': 'incomplete', 'accepted_faster': False, 'completed_pairs': 0}

    def interrupted(signum, _frame):
        raise RuntimeError(f'interrupted by signal {signum}')

    handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for pair in pairs:
            metrics = {'order': pair['order'], 'paths': {label: pair[label] for label in binaries}}
            for label in pair['order']:
                bench.require(sha256(HARNESS) == plan['harness_sha256'], 'benchmark harness changed since plan')
                for record in evidence.values():
                    for source in [record, *record['related_runtime_files']]:
                        bench.require(sha256(Path(source['path'])) == source['sha256'], 'correctness evidence changed since plan')
                output = Path(pair[label])
                lease = launch_arm(args, label, output, binaries[label], runtimes[label])
                data, response, request = (read_json(output / name) for name in ('result.json', 'response.json', 'request.json'))
                bench.require(data.get('prompt_sha256') == expected_prompt_hash, 'prompt hash changed')
                metrics[label] = {**validate_arm(data, response, request, expected_request, 0 if label == 'baseline' else 32768, cpu['unit']), **lease}
                actual_runtime = validate_runtime(read_json(output / 'runtime.json'), runtimes[label], binaries[label])
                bench.require(actual_runtime == loaded.setdefault(label, actual_runtime), f'{label} loaded runtime changed across pairs')
                normalized = validate_command(read_json(output / 'command.json'), 0 if label == 'baseline' else 32768, args)
                if configuration is None:
                    configuration = normalized
                bench.require(normalized == configuration, 'settings differ between paired arms')
                bench.write_json(out / 'progress.json', {'pair': pair['index'], 'completed_arm': label, 'metrics': metrics})
            completed.append(metrics)
            result = {**analyze(completed), 'status': 'complete' if len(completed) == args.max_pairs else 'running'}
            bench.write_json(out / 'comparison.json', result)
            print(json.dumps({'completed_pairs': len(completed), 'p_bound': result['one_sided_anytime_p_bound'], 'mean_decode_gain_tok_s': result['mean_paired_decode_gain_tok_s']}), flush=True)
            if result['accepted_faster']:
                result['status'] = 'complete'
                result['stop_reason'] = 'predeclared anytime bound reached with positive measured decode gain'
                break
        result.setdefault('stop_reason', 'maximum planned pairs reached')
    except Exception as error:
        result = {**analyze(completed), 'status': 'failed', 'accepted_faster': False, 'error': str(error)}
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        bench.write_json(out / 'result.json', result)
        print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if result['status'] == 'complete' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, default=ROOT / 'build/bin/llama-server')
    parser.add_argument('--baseline', type=Path, help='default: candidate binary, with resident window disabled')
    parser.add_argument('--out', type=Path, required=True)
    for kind in ('operator', 'kl', 'niah', 'cache'):
        parser.add_argument('--' + kind + '-evidence', type=Path, required=True)
    parser.add_argument('--order-seed', type=int, default=20261019)
    parser.add_argument('--max-pairs', type=int, default=12)
    parser.add_argument('--port', type=int, default=8095)
    parser.add_argument('--reserve-mib', type=int, default=3072)
    parser.add_argument('--cache-budget-mib', type=int, default=0)
    args = parser.parse_args()
    bench.require(1 <= args.max_pairs <= 12, 'max-pairs must be from 1 to 12')
    bench.require(args.reserve_mib >= 0 and args.cache_budget_mib >= 0, 'cache reserve and budget must be nonnegative')
    bench.require(1 <= args.port <= 65535, 'invalid port')
    return run(args)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as error:
        print(f'FNKVWIN paired runner: {error}', file=sys.stderr)
        sys.exit(1)
