#!/usr/bin/env python3
"""Run one FNKVWIN server arm, or compare two full-vocabulary KL artifacts."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


MODEL_DIR = Path('/mnt/models/flash-next')
MODEL = MODEL_DIR / 'Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf'
MTP = MODEL_DIR / 'mtp-Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS.gguf'
PROMPT_IDS = Path('/mnt/models/stratamatch-home/prompt-ids.json')
GATES = Path('/mnt/models/fnkvwin/gates')
PROMPT_N = 100000
MIN_AVAILABLE = 10 * 2**30
MAX_VRAM_MIB = 23552
FAST_ENV = ('LLAMA_B1 LLAMA_QSA_SEL_DIRECT GGML_CUDA_MMQ_ID_CAP '
            'GGML_CUDA_GET_ROWS_NARROW GGML_CUDA_LID_EXACT GGML_CUDA_TOPK_SELECT '
            'GGML_CUDA_MMID_PAR LLAMA_HC_BF16_ONCE GGML_CUDA_GDN_R16 '
            'GGML_CUDA_CONCAT_TILE GGML_CUDA_NORM_CAST').split()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def disk_check():
    for path, gib in (('/', 40), ('/mnt/models', 15), ('/mnt/overflow', 50)):
        require(shutil.disk_usage(path).free >= gib * 2**30, f'{path} free space below {gib} GiB')


def write_json(path, data):
    disk_check()
    with path.open('w') as handle:
        json.dump(data, handle, allow_nan=False, separators=(',', ':'))
        handle.write('\n')


def available_bytes():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable unavailable')


def probability_rows(response, n_vocab):
    rows = response.get('completion_probabilities', [])
    require(len(rows) == 8, f'expected 8 distributions, received {len(rows)}')
    generated = response.get('tokens', [])
    require(generated == [row['id'] for row in rows], 'generated token IDs disagree with probability rows')
    result = []
    for step, row in enumerate(rows):
        items = row.get('top_logprobs', [])
        require(len(items) == n_vocab, f'step {step}: truncated vocabulary {len(items)}/{n_vocab}')
        ids = [item['id'] for item in items]
        require(set(ids) == set(range(n_vocab)), f'step {step}: missing or duplicate vocabulary IDs')
        values = [0.0] * n_vocab
        for item in items:
            value = item['logprob']
            value = -math.inf if value is None else float(value)
            require(value <= 1e-6 and not math.isnan(value), f'step {step}: invalid log probability')
            values[item['id']] = value
        mass = math.fsum(math.exp(value) for value in values)
        require(abs(mass - 1.0) <= 1e-3, f'step {step}: probability mass {mass} is not normalized')
        result.append((values, mass))
    return result


def compare(args):
    baseline = json.loads((args.baseline / 'result.json').read_text())
    candidate = json.loads((args.candidate / 'result.json').read_text())
    require(baseline['status'] == candidate['status'] == 'complete', 'both KL arms must complete')
    require(baseline['mode'] == candidate['mode'] == 'kl', 'comparison requires KL arms')
    require(baseline['prompt_sha256'] == candidate['prompt_sha256'], 'prompt token IDs differ')
    require(baseline['n_vocab'] == candidate['n_vocab'], 'vocabulary sizes differ')
    response_p = json.loads((args.baseline / 'response.json').read_text())
    response_q = json.loads((args.candidate / 'response.json').read_text())
    require(response_p['tokens'] == response_q['tokens'], 'generated token IDs differ: later steps are not matched-token distributions')
    rows_p = probability_rows(response_p, baseline['n_vocab'])
    rows_q = probability_rows(response_q, candidate['n_vocab'])
    steps = []
    for step, ((lp, mass_p), (lq, mass_q)) in enumerate(zip(rows_p, rows_q)):
        norm_p, norm_q = math.log(mass_p), math.log(mass_q)
        terms = []
        zero_support = False
        for p, q in zip(lp, lq):
            if p == -math.inf:
                continue
            if q == -math.inf:
                zero_support = True
                break
            terms.append(math.exp(p - norm_p) * ((p - norm_p) - (q - norm_q)))
        divergence = math.inf if zero_support else math.fsum(terms)
        steps.append({'step': step, 'token_id': response_p['tokens'][step],
                      'kl_nats': None if zero_support else divergence,
                      'infinite_kl': zero_support, 'baseline_mass': mass_p, 'candidate_mass': mass_q})
    finite = [step['kl_nats'] for step in steps if not step['infinite_kl']]
    result = {'direction': 'KL(baseline || candidate)', 'normalization': 'full-vocabulary log probabilities normalized by measured mass',
              'matched_tokens': True, 'n_vocab': baseline['n_vocab'], 'steps': steps,
              'mean_kl_nats': statistics.mean(finite) if len(finite) == len(steps) else None,
              'max_kl_nats': max(finite) if len(finite) == len(steps) else None,
              'infinite_kl': len(finite) != len(steps)}
    if args.output:
        target = args.output.resolve()
        require(target.is_relative_to(GATES.resolve()), 'comparison output must be under /mnt/models/fnkvwin/gates')
        write_json(target, result)
    print(json.dumps(result, allow_nan=False))


class ServerArm:
    def __init__(self, args, output):
        self.args, self.output = args, output
        self.process = None
        self.log = None
        self.thread = None
        self.stop = threading.Event()
        self.failure = None
        self.samples = []
        self.phase = 'load'
        self.started = time.monotonic()
        self.deadline = self.started + args.timeout
        self.base_url = f'http://127.0.0.1:{args.port}'

    def kill_owned_server(self):
        if self.process is None or self.process.poll() is not None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=5)
        except ProcessLookupError:
            pass

    def monitor(self):
        try:
            with (self.output / 'monitor.jsonl').open('w') as handle:
                while not self.stop.is_set():
                    require(time.monotonic() < self.deadline, 'arm deadline exceeded')
                    require(self.process.poll() is None, f'owned server exited: {self.process.returncode}')
                    available = available_bytes()
                    require(available >= MIN_AVAILABLE, 'MemAvailable below 10 GiB')
                    disk_check()
                    gpu = subprocess.check_output(
                        ['nvidia-smi', '--query-gpu=memory.used,power.draw,power.limit', '--format=csv,noheader,nounits'],
                        text=True, timeout=5).strip().splitlines()
                    require(len(gpu) == 1, 'expected the single home GPU')
                    vram, power, limit = [float(value.strip()) for value in gpu[0].split(',')]
                    require(vram <= MAX_VRAM_MIB, 'whole-GPU VRAM exceeds 23552 MiB')
                    require(limit == 350.0, 'home GPU power limit must be 350 W')
                    status = Path(f'/proc/{self.process.pid}/status').read_text()
                    memory = {line.split(':', 1)[0]: int(line.split()[1]) for line in status.splitlines()
                              if line.startswith(('VmRSS:', 'VmHWM:'))}
                    sample = {'elapsed_s': time.monotonic() - self.started, 'phase': self.phase,
                              'vram_mib': vram, 'power_w': power, 'power_limit_w': limit,
                              'rss_kib': memory.get('VmRSS', 0), 'rss_peak_kib': memory.get('VmHWM', 0),
                              'mem_available_bytes': available}
                    self.samples.append(sample)
                    handle.write(json.dumps(sample) + '\n')
                    handle.flush()
                    self.stop.wait(0.5)
        except Exception as error:
            if not self.stop.is_set():
                self.failure = str(error)
                self.kill_owned_server()

    def start(self):
        require(os.environ.get('PARENT_HOLDS_GPU_LOCK') == '1', 'parent must hold gpu.lock and set PARENT_HOLDS_GPU_LOCK=1')
        unit = Path(Path('/proc/self/cgroup').read_text().strip().split('::', 1)[1]).name
        require(unit.startswith('llama.cpp-') and unit.endswith(('.service', '.scope')),
                'parent must run this harness in a llama.cpp-* systemd unit for sysguard coverage')
        require(available_bytes() >= MIN_AVAILABLE, 'MemAvailable below 10 GiB before launch')
        disk_check()
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', self.args.port))
        executable = self.output / 'llama.cpp-fnkvwin-server'
        executable.symlink_to(self.args.server.resolve(strict=True))
        memory_only = self.args.mode == 'memory'
        context = str(self.args.context) if memory_only else '131072'
        batch, ubatch = ('512', '512') if memory_only else ('131072', '12288')
        cache = 'off' if memory_only else (str(self.args.cache_budget_mib) if self.args.cache_budget_mib else 'auto')
        command = [str(executable), '-m', str(MODEL), '--load-mode', 'mmap', '-c', context,
                   '--parallel', '1', '-b', batch, '-ub', ubatch, '-t', '8', '-ngl', '999',
                   '-fa', 'on', '--cpu-moe', '-fitt', '1536', '--moe-cache', cache,
                   '-ctk', 'sjkvarn4', '-ctv', 'sjkvarn4',
                   '--host', '127.0.0.1', '--port', str(self.args.port), '--no-webui', '--no-cache-disk']
        if memory_only:
            command += ['--no-warmup']
        if self.args.resident:
            command += ['--kv-resident', str(self.args.resident)]
        if self.args.mode == 'protocol':
            command += ['-md', str(MTP), '--cpu-moe-draft', '--spec-type', 'draft-mtp-adaptive',
                        '--spec-draft-n-max', '4', '--spec-draft-ubatch', '2048']
        else:
            command += ['--spec-type', 'none']
        env = os.environ.copy()
        env.update({key: '1' for key in FAST_ENV})
        env['LLAMA_PREFETCH_THREADS'] = '8'
        env['LD_LIBRARY_PATH'] = str(self.args.server.resolve().parent)
        env['GGML_CUDA_MOE_CACHE_STATS'] = '4096'
        # Record only known performance settings, not the inherited environment.
        write_json(self.output / 'command.json', {'argv': command, 'environment':
                   {key: env[key] for key in [*FAST_ENV, 'LLAMA_PREFETCH_THREADS', 'GGML_CUDA_MOE_CACHE_STATS', 'LD_LIBRARY_PATH']}})
        self.log = (self.output / 'server.log').open('w')
        self.process = subprocess.Popen(command, env=env, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        self.thread = threading.Thread(target=self.monitor, daemon=True)
        self.thread.start()
        while True:
            require(not self.failure, self.failure)
            require(self.process.poll() is None, f'server exited: {self.process.returncode}')
            try:
                if self.http('/health', timeout=2):
                    break
            except (urllib.error.URLError, TimeoutError):
                pass
            require(time.monotonic() < self.deadline, 'server startup deadline exceeded')
            self.stop.wait(0.5)
        return time.monotonic() - self.started

    def http(self, route, data=None, timeout=None):
        remaining = self.deadline - time.monotonic()
        require(remaining > 0, 'arm deadline exceeded')
        request = urllib.request.Request(self.base_url + route, data=None if data is None else json.dumps(data).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=min(timeout or remaining, remaining)) as response:
            return json.load(response)

    def finish(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=6)
        self.kill_owned_server()
        if self.log:
            self.log.close()
        stats = {'peak_vram_mib': max((s['vram_mib'] for s in self.samples), default=None),
                 'peak_rss_kib': max((s['rss_peak_kib'] for s in self.samples), default=None),
                 'min_mem_available_bytes': min((s['mem_available_bytes'] for s in self.samples), default=None)}
        inference = [s['power_w'] for s in self.samples if s['phase'] == 'inference']
        stats['mean_inference_power_w'] = statistics.mean(inference) if inference else None
        stats['guard_failure'] = self.failure
        if (self.output / 'server.log').exists():
            text = (self.output / 'server.log').read_text(errors='replace')
            matches = re.findall(r'\[moe-cache\] CUDA(\d+) hits=(\d+)/(\d+) \(([\d.]+)%\)', text)
            stats['expert_cache_last_log_stats'] = [dict(device=int(d), hits=int(h), requests=int(n), hit_percent=float(p))
                                                     for d, h, n, p in matches[-1:]]
        return stats


def tokenize(arm, text):
    return arm.http('/tokenize', {'content': text, 'add_special': False, 'parse_special': True})['tokens']


def niah_prompt(arm):
    needles = [('OPAL', 'V7K2R9QM'), ('CEDAR', 'H4P8Z6TW'), ('MARBLE', 'B9N3X5LC')]
    depths = [0.01, 0.50, 0.95]
    filler = ' The archive entry contains routine weather and inventory observations.'
    unit_tokens = len(tokenize(arm, filler * 16)) / 16
    units, padding = int(PROMPT_N / unit_tokens) - 100, 0
    placements = [int(PROMPT_N * depth / unit_tokens) for depth in depths]
    for _ in range(16):
        pieces, previous = [], 0
        for (name, code), position in zip(needles, placements):
            position = max(previous, min(position, units))
            pieces.append(filler * (position - previous))
            pieces.append(f'\nThe unique retrieval code for {name} is {code}.\n')
            previous = position
        pieces += [filler * (units - previous), ' a' * padding,
                   '\nReturn the retrieval codes for OPAL, CEDAR, and MARBLE, in that order. Output only the three codes.']
        messages = [{'role': 'system', 'content': 'Read the archive and retrieve the exact requested codes.'},
                    {'role': 'user', 'content': ''.join(pieces)}]
        template_request = {'messages': messages, 'chat_template_kwargs': {'enable_thinking': False}}
        rendered = arm.http('/apply-template', template_request)['prompt']
        ids = tokenize(arm, rendered)
        difference = PROMPT_N - len(ids)
        if difference == 0:
            positions = [len(tokenize(arm, rendered[:rendered.index(f'The unique retrieval code for {name}')]))
                         for name, _ in needles]
            if any(abs(position - PROMPT_N * depth) > unit_tokens + 2 for position, depth in zip(positions, depths)):
                placements = [placement - round((position - PROMPT_N * depth) / unit_tokens)
                              for placement, position, depth in zip(placements, positions, depths)]
                continue
            return ids, {'needles': dict(needles), 'needle_positions': positions,
                         'needle_depths': [position / PROMPT_N for position in positions],
                         'template_request': template_request, 'rendered_prompt': rendered}
        if difference > 0:
            padding += difference
        elif padding >= -difference:
            padding += difference
        else:
            units -= math.ceil((-difference - padding) / unit_tokens)
            padding = 0
    raise RuntimeError('could not create an exactly 100000-token templated retrieval prompt')


def payload(ids, mode):
    result = {'prompt': ids, 'n_predict': 5120, 'temperature': 1.0, 'seed': 11, 'top_k': 40,
              'top_p': 0.95, 'min_p': 0, 'repeat_penalty': 1.0, 'samplers': ['top_k', 'top_p', 'temperature'],
              'ignore_eos': True, 'logit_bias': [[i, False] for i in [248044, 248046, 248063, 248064, 248065]],
              'cache_prompt': False, 'stream': mode == 'protocol', 'return_tokens': True}
    if mode == 'kl':
        result.update(n_predict=8, temperature=0.0, n_probs=300000, post_sampling_probs=False)
    elif mode == 'niah':
        result.update(n_predict=64, temperature=0.0, ignore_eos=False)
    return result


def stream_protocol(arm, request):
    started = time.monotonic()
    first = None
    final = None
    contents, tokens = [], []
    req = urllib.request.Request(arm.base_url + '/completion', data=json.dumps(request).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=arm.deadline - time.monotonic()) as response, (arm.output / 'response.jsonl').open('w') as handle:
        for raw in response:
            require(not arm.failure, arm.failure)
            disk_check()
            line = raw.decode().strip()
            if not line:
                continue
            elapsed = time.monotonic() - started
            handle.write(json.dumps({'elapsed_s': elapsed, 'line': line}) + '\n')
            handle.flush()
            if not line.startswith('data:') or line[5:].strip() == '[DONE]':
                continue
            item = json.loads(line[5:].strip())
            if 'error' in item:
                raise RuntimeError(str(item['error']))
            if first is None and (item.get('content') or item.get('tokens')):
                first = elapsed
            contents.append(item.get('content', ''))
            tokens.extend(item.get('tokens', []))
            if item.get('stop'):
                final = item
    require(final is not None, 'no final protocol timing response')
    return {'final': final, 'content': ''.join(contents), 'tokens': tokens}, first


def memory_snapshot(arm, context):
    gpu = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used,memory.free,memory.total,power.draw,power.limit',
         '--format=csv,noheader,nounits'], text=True, timeout=5).strip().splitlines()
    require(len(gpu) == 1, 'expected the single home GPU')
    used, free, total, power, limit = [float(value.strip()) for value in gpu[0].split(',')]
    require(used <= MAX_VRAM_MIB, 'whole-GPU VRAM exceeds 23552 MiB at memory snapshot')
    require(available_bytes() >= MIN_AVAILABLE, 'MemAvailable below 10 GiB at memory snapshot')
    require(limit == 350.0, 'home GPU power limit must be 350 W')
    text = (arm.output / 'server.log').read_text(errors='replace')
    memory_lines = [line for line in text.splitlines() if re.search(r'memory|buffer|QSA layer|MiB|GiB', line, re.IGNORECASE)]
    layers = []
    pattern = (r'QSA layer (\d+): (\d+) resident body groups plus (\d+) staging rows, '
               r'([\d.]+) MiB pinned body, ([\d.]+) MiB VRAM freed for expert cache')
    for layer, groups, staging, pinned, freed in re.findall(pattern, text):
        layers.append({'layer': int(layer), 'resident_body_groups': int(groups), 'staging_rows': int(staging),
                       'pinned_body_mib_logged': float(pinned), 'body_vram_freed_mib_logged': float(freed)})
    require(not arm.args.resident or layers, 'resident allocation did not produce QSA allocation logs')
    configs = [{'device': int(device), 'group_stride_bytes': int(stride), 'resident_groups': int(groups)}
               for device, stride, groups in sorted(set(re.findall(r'CUDA_QSA_Resident_(\d+)_(\d+)_(\d+)', text)))]
    # MiB messages are rounded. Only derive exact card bytes when one buffer geometry covers every QSA layer.
    card_bytes = None
    if len(configs) == 1 and layers and all(layer['resident_body_groups'] <= configs[0]['resident_groups'] for layer in layers):
        card_bytes = sum(layer['resident_body_groups'] * configs[0]['group_stride_bytes'] for layer in layers)
    return {'context_cells': context, 'load_only': True, 'inference_run': False, 'expert_cache': 'off',
            'loaded_gpu': {'used_mib': used, 'free_mib': free, 'total_mib': total, 'power_w': power, 'power_limit_w': limit},
            'qsa_layers': layers, 'resident_buffer_configs': configs,
            'qsa_body_card_bytes_from_log_geometry': card_bytes, 'qsa_body_ram_bytes_exact': None,
            'body_accounting_note': 'Pinned body and VRAM savings are rounded MiB log values; exact RAM bytes are not logged.',
            'server_memory_log_lines': memory_lines}


def run(args):
    require(args.out.resolve().is_relative_to(GATES.resolve()), 'artifacts must be under /mnt/models/fnkvwin/gates')
    require(os.environ.get('PARENT_HOLDS_GPU_LOCK') == '1', 'parent must hold gpu.lock and set PARENT_HOLDS_GPU_LOCK=1')
    disk_check()
    args.out.mkdir(parents=True, exist_ok=False)
    arm = ServerArm(args, args.out)
    def interrupted(signum, _frame):
        raise RuntimeError(f'interrupted by signal {signum}')
    prior_sigterm = signal.signal(signal.SIGTERM, interrupted)
    result = {'mode': args.mode, 'resident_cells': args.resident, 'status': 'failed'}
    try:
        result['load_s'] = arm.start()
        if args.mode == 'memory':
            result.update(memory_snapshot(arm, args.context))
        else:
            if args.mode == 'niah':
                ids, construction = niah_prompt(arm)
                write_json(args.out / 'niah-prompt.json', construction)
            else:
                ids = json.loads(PROMPT_IDS.read_text())
            require(len(ids) == PROMPT_N and all(type(t) is int for t in ids), 'prompt must contain exactly 100000 token IDs')
            result['prompt_sha256'] = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
            request = payload(ids, args.mode)
            write_json(args.out / 'request.json', request)
            if args.mode == 'kl':
                models = arm.http('/v1/models')
                write_json(args.out / 'models.json', models)
                result['n_vocab'] = models['data'][0]['meta']['n_vocab']
                require(result['n_vocab'] <= request['n_probs'], 'n_probs does not cover the full vocabulary')
            arm.phase = 'inference'
            started = time.monotonic()
            if args.mode == 'protocol':
                response, result['ttft_s'] = stream_protocol(arm, request)
                final = response['final']
            else:
                response = arm.http('/completion', request)
                final = response
            result['ttc_s'] = time.monotonic() - started
            arm.phase = 'complete'
            write_json(args.out / 'response.json', response)
            result['timings'] = final.get('timings', {})
            result['prefill_tok_s'] = result['timings'].get('prompt_per_second')
            result['decode_tok_s'] = result['timings'].get('predicted_per_second')
            require(result['timings'].get('prompt_n') == PROMPT_N, 'server did not evaluate the full 100000-token prompt')
            require(result['timings'].get('cache_n', 0) == 0, 'prompt cache was reused')
            if args.mode == 'kl':
                probability_rows(response, result['n_vocab'])
                require(len(response['tokens']) == 8, 'KL arm did not generate exactly 8 tokens')
            elif args.mode == 'niah':
                result['needle_matches'] = {name: code in final.get('content', '') for name, code in construction['needles'].items()}
                require(all(result['needle_matches'].values()), 'one or more retrieval codes are missing')
            else:
                require(final.get('tokens_predicted') == 5120, 'protocol must generate exactly 5120 tokens')
        require(not arm.failure, arm.failure)
        result['status'] = 'complete'
    except Exception as error:
        result['error'] = str(error)
    finally:
        result.update(arm.finish())
        if arm.failure:
            result.update(status='failed', error=arm.failure)
        result['total_wall_s'] = time.monotonic() - arm.started
        signal.signal(signal.SIGTERM, prior_sigterm)
        write_json(args.out / 'result.json', result)
        print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if result['status'] == 'complete' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='mode', required=True)
    for mode in ('kl', 'niah', 'protocol', 'memory'):
        command = commands.add_parser(mode)
        command.add_argument('--server', type=Path, required=True)
        command.add_argument('--resident', type=int, choices=(0, 32768), required=True)
        command.add_argument('--out', type=Path, required=True)
        if mode == 'memory':
            command.add_argument('--context', type=int, choices=(100096, 262144), default=100096)
            command.set_defaults(cache_budget_mib=0)
        else:
            command.add_argument('--cache-budget-mib', type=int, default=0, help='0 selects --moe-cache auto')
        command.add_argument('--port', type=int, default=8095)
        command.add_argument('--timeout', type=float, default=1500 if mode == 'protocol' else 600)
    command = commands.add_parser('compare')
    command.add_argument('baseline', type=Path)
    command.add_argument('candidate', type=Path)
    command.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.mode == 'compare':
        compare(args)
        return 0
    require(0 < args.timeout <= (1500 if args.mode == 'protocol' else 600), 'timeout exceeds the mode limit')
    require(args.cache_budget_mib >= 0, 'cache budget must be nonnegative')
    require(1 <= args.port <= 65535, 'invalid port')
    return run(args)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as error:
        print(f'FNKVWIN harness: {error}', file=sys.stderr)
        sys.exit(1)
