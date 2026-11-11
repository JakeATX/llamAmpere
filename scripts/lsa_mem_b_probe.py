#!/usr/bin/env python3
"""Step B: rs3-scale ABAB and paired F16/20% numerical check; reuse the guarded J3 server."""
import importlib.util
import csv
import json
import math
import os
import re
import sys
import time
from array import array
from pathlib import Path

JOB = Path(os.environ['LSA_MEM_B_JOB_DIR']).resolve()
J = JOB.parent
spec = importlib.util.spec_from_file_location('j3', J / 'memA1-r309/probe.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
p.B = '/mnt/overflow/lsa/lsa-memB-r317-cpu33a'
p.O = JOB / 'run'
xver = p.xver
sys.path.insert(0, str(J.parent.parent / 'LOOPCMP'))
from looprule_v3 import rule_v3

base_environment = p.environment
resident = False
collect_only = os.environ.get('LSA_MEM_B_COLLECT_ONLY') == '1'
analyze_only = os.environ.get('LSA_MEM_B_ANALYZE_ONLY') == '1'
assert not (collect_only and analyze_only)


def environment(base):
    e = base_environment(base)
    e['GGML_CUDA_LSA_RESIDENT_STAGE_ALL'] = '1'
    return e


xver.arms.server_env = environment
base_config = p.config
base_init = xver.Server.__init__
base_request = xver.Server.request
base_stop = xver.Server.stop
base_check = xver.Server.check
FOOTER = 'LSA MEMB COMPLETE '


def contract(name, requests):
    greedy_arm = name in ('a0', 'b0', 'a1', 'b1')
    assert greedy_arm or name in ('f16_20_nores', 'f16_20_stage'), 'unknown arm'
    assert len(requests) == (2 if greedy_arm else 129), (name, 'request count')
    for i, row in enumerate(requests):
        assert set(row) == {'prompt_n', 'predicted_n', 'prompt_len', 'tokens'}, (name, 'request fields')
        assert all(type(v) is int for v in row.values()), (name, 'request value type')
        expect = (64 if i == 0 else 512) if greedy_arm else 1
        assert row['predicted_n'] == expect, (name, 'incomplete generation')
        assert row['prompt_len'] == (98000 if greedy_arm else 100000 + (0 if i == 0 else 1 + 4*(i-1))), (name, 'prompt length')
        if i == 0:
            assert row['prompt_n'] == row['prompt_len'], (name, 'initial prompt not evaluated')
        elif greedy_arm:
            # SJ group checkpoints restore 98,000 to 97,792, replaying 208 rows.
            assert row['prompt_n'] == 208, (name, 'wrong checkpoint replay')
        else:
            assert 1 <= row['prompt_n'] <= 8, (name, 'not cached decode')
        if greedy_arm:
            assert row['tokens'] == expect, (name, 'missing generated IDs')
    # Only <=8-row graphs resolve residency. The 208-row greedy replay does not.
    # The last sampled token of each response has not yet been evaluated.
    resolves = sum((i > 0 and row['prompt_n'] <= 8) + row['predicted_n'] - 1
                   for i, row in enumerate(requests))
    assert resolves == (574 if greedy_arm else 128), (name, 'resolve count')
    return resolves


def resources(name, started, ended):
    assert all(type(v) in (int, float) and math.isfinite(v) for v in (started, ended))
    assert 0 < started <= ended, (name, 'resource time bounds')
    text = (p.O / f'{name}.resources.jsonl').read_text(encoding='utf-8')
    assert text and text.endswith('\n'), (name, 'missing or truncated resource samples')
    rows = [json.loads(line) for line in text.splitlines()]
    if collect_only:
        return dict(started=started, ended=ended, samples=len(rows), first=rows[0]['epoch'], last=rows[-1]['epoch'])
    for row in rows:
        assert set(row) == {'epoch', 'ram_kib', 'vram_mib'}, (name, 'resource schema')
        assert type(row['epoch']) in (int, float) and math.isfinite(row['epoch'])
        assert type(row['ram_kib']) is int and row['ram_kib'] >= 5242880, (name, 'RAM floor')
        assert type(row['vram_mib']) is int and 0 <= row['vram_mib'] <= 23552, (name, 'VRAM cap')
    epochs = [started] + [r['epoch'] for r in rows] + [ended]
    # NVML has a 5s timeout, polling sleeps .5s, and shutdown joins for up to 6s.
    assert all(0 <= b-a <= 12 for a, b in zip(epochs, epochs[1:])), (name, 'resource coverage gap')
    assert all(a['epoch'] < b['epoch'] for a, b in zip(rows, rows[1:])), (name, 'resource order')
    return dict(started=started, ended=ended, samples=len(rows), first=rows[0]['epoch'], last=rows[-1]['epoch'])


def init(self, name, *args):
    self.mem_b_name = name
    self.mem_b_requests = []
    self.mem_b_completed = False
    self.mem_b_started = time.time()
    base_init(self, name, *args)


def check_server(self, require_listener=True):
    result = base_check(self, require_listener)
    assert self.monitor.is_alive(), 'resource monitor exited'
    return result


def request(self, path, body):
    result = base_request(self, path, body)
    if collect_only:
        with (p.O / f'{self.mem_b_name}.responses.jsonl').open('a') as output:
            output.write(json.dumps(result) + '\n')
    assert path == '/completion', 'unexpected endpoint'
    timing = result['timings']
    self.mem_b_requests.append(dict(prompt_n=timing['prompt_n'], predicted_n=timing['predicted_n'],
        prompt_len=len(body['prompt']), tokens=len(result.get('tokens', []))))
    return result


def stop(self):
    monitor_live = self.monitor.is_alive()
    base_stop(self)
    assert self.proc.returncode == 0, ('server shutdown rc', self.proc.returncode)
    assert monitor_live and not self.monitor.is_alive(), 'resource monitor failed or did not stop'
    assert not self.mem_b_completed, 'duplicate shutdown receipt'
    resolves = (sum((i > 0 and row['prompt_n'] <= 8) + row['predicted_n'] - 1
        for i, row in enumerate(self.mem_b_requests)) if collect_only else contract(self.mem_b_name, self.mem_b_requests))
    receipt = dict(arm=self.mem_b_name, pid=self.proc.pid, rc=self.proc.returncode,
        resolves=resolves, requests=self.mem_b_requests,
        resources=resources(self.mem_b_name, self.mem_b_started, time.time()))
    log_path = p.O / f'{self.mem_b_name}.server.log'
    text = log_path.read_text(encoding='utf-8')
    assert text.endswith('\n') and FOOTER not in text, 'truncated or repeated server log'
    with log_path.open('a') as output:
        output.write(FOOTER + json.dumps(receipt, sort_keys=True) + '\n')
    self.mem_b_completed = True


xver.Server.__init__ = init
xver.Server.request = request
xver.Server.stop = stop
xver.Server.check = check_server


def config(refine=0, keep=0, recall=False):
    a = base_config(refine, keep, recall)
    a.env = [v for v in a.env if not v.startswith('LLAMA_LSA_RESIDENT=')]
    if keep and resident:
        a.env += ['LLAMA_LSA_RESIDENT=55']
    return a


p.config = config


def expected_headers(name):
    a = config(128, 44840) if name in ('a0', 'b0', 'a1', 'b1') else config(16, 18336)
    cmd = xver.arms.server_cmd('sj44', p.B + '/llama-server', a.model, a.ctx, a.port, 'auto:65536')
    cmd += ['--slot-save-path', a.slot_dir, '--cache-ram', '0']
    return 'cmd: ' + ' '.join(cmd), 'explicit env: ' + str(a.env)


def check_cuda(path):
    with path.open() as source:
        rows = [r for r in csv.reader(source) if len(r) > 7 and r[3] == 'FLASH_ATTN_EXT_SJKVARN_TRELLIS']
    assert len(rows) == 2 and all(r[2] == 'CUDA0' and r[5] == 'test' and r[6:8] == ['1', '1'] for r in rows), rows
    assert sorted(re.search(r'(?:^|,)n_q=(1|8),', r[4]).group(1) for r in rows) == ['1', '8'], rows


def read_log(name):
    text = (p.O / f'{name}.server.log').read_text(encoding='utf-8')
    assert text.endswith('\n'), (name, 'truncated log')
    lines = text.splitlines()
    assert len(lines) >= 3 and tuple(lines[:2]) == expected_headers(name), (name, 'wrong or incomplete header')
    assert all(sum(line.startswith(prefix) for line in lines) == 1 for prefix in ('cmd:', 'explicit env:')), (name, 'duplicate header')
    assert text.count(FOOTER) == 1 and lines[-1].startswith(FOOTER), (name, 'missing or misplaced footer')
    receipt = json.loads(lines[-1][len(FOOTER):])
    assert set(receipt) == {'arm', 'pid', 'rc', 'resolves', 'requests', 'resources'}, (name, 'footer schema')
    assert receipt['arm'] == name and type(receipt['rc']) is int and receipt['rc'] == 0, (name, 'failed child')
    assert type(receipt['pid']) is int and receipt['pid'] > 0, (name, 'missing child')
    expected = contract(name, receipt['requests'])
    assert type(receipt['resolves']) is int and receipt['resolves'] == expected, (name, 'incomplete footer')
    observed = receipt['resources']
    assert observed == resources(name, observed['started'], observed['ended']), (name, 'resource receipt mismatch')
    assert not re.search(r'CUDA (?:error|failure|runtime error)|cudaError|illegal (?:memory access|instruction)|'
        r'device-side assert|(?:GGML_)?(?:ASSERT|ABORT)|terminate called|Segmentation fault|core dumped',
        text, re.IGNORECASE), (name, 'fatal server diagnostic')
    assert not any(s in text for s in ('has no device copy', 'LSA SPEC RECALL INVALID',
        'full stage allocation failed', 'shared full stage is too small', 'will run on the CPU')), name
    assert resident == (name in ('b0', 'b1', 'f16_20_stage')), (name, 'wrong resident arm')
    diagnostics = re.findall(r'LSA (?:ST|RES|resident)[^\n]*', text)
    if not resident:
        assert not diagnostics, (name, 'unexpected resident diagnostic')
        return dict(completion=receipt, diagnostic_rows=0)
    layers = set(range(3, 64, 4))
    stage_pattern = (r'LSA STAGE: layer (\d+) resolves=(\d+) requested=(\d+) capacity=(\d+) '
        r'clock_over_slots=(\d+) host_records=(\d+)')
    periodic_pattern = (r'LSA RESIDENT: layer (\d+) resolves=(\d+) picked=(\d+) fetched=(\d+) '
        r'over_slots=(\d+) slots=(\d+) record_bytes=(\d+)')
    slot_pattern = (r'LSA RESIDENT: layer (\d+) slots=(\d+) records, resolves=(\d+) picked=(\d+) '
        r'fetched=(\d+) over_slots=(\d+), per resolve: ([0-9]+\.[0-9]+) MiB fetched \+ ([0-9]+\.[0-9]+) MiB read from host past the slots')
    terminal_stage_pattern = (r'LSA RESIDENT: layer (\d+) stage=(\d+) records, staged=(\d+) stage_full=(\d+), '
        r'per resolve: ([0-9]+\.[0-9]+) MiB copied to the stage, ([0-9]+\.[0-9]+) MiB still read from host')
    allocation_pattern = r'LSA RESIDENT: stage of (\d+) records \(([0-9]+\.[0-9]+) MiB\) for records past the slots'
    init_pattern = r'LSA RESIDENT: 128-cell record slots, records=(\d+) heads=(\d+) \(([0-9]+\.[0-9]+) MiB per layer\)'
    body_pattern = (r'LSA resident layer (\d+) \(K3/V2 record pages\): (\d+) of (\d+) records on the card '
        r'\(55\.0%\), ([0-9]+\.[0-9]+) MiB slots vs ([0-9]+\.[0-9]+) MiB body, ([0-9]+\.[0-9]+) MiB pinned')
    patterns = (stage_pattern, periodic_pattern, slot_pattern, terminal_stage_pattern,
        allocation_pattern, init_pattern, body_pattern)
    groups = [[] for _ in patterns]
    for row in diagnostics:
        matches = [(i, re.fullmatch(pattern, row)) for i, pattern in enumerate(patterns)]
        matches = [(i, match) for i, match in matches if match]
        assert len(matches) == 1, (name, 'malformed, truncated or unexpected diagnostic', row)
        i, match = matches[0]
        groups[i].append(match.groups())
    stage, counters, terminal, terminal_stage, allocations, inits, bodies = groups
    stage = [tuple(map(int, row)) for row in stage]
    counters = [tuple(map(int, row)) for row in counters]
    expected_keys = {(layer, resolve) for layer in layers for resolve in range(16, expected+1, 16)}
    assert len(stage) == len(counters) == len(expected_keys), (name, 'periodic row count')
    assert {(r[0], r[1]) for r in stage} == {(r[0], r[1]) for r in counters} == expected_keys, (name, 'missing or duplicate resolve')
    # No admission or prefetch in these arms: every CLOCK overflow takes one stage record.
    assert all(r[5] == 0 and 0 <= r[2] == r[4] <= r[3] and r[3] > 0 for r in stage), (name, 'host read or stage overflow')
    assert all(r[4] == 0 and r[5] > 0 and r[6] > 0 for r in counters), (name, 'periodic overflow or invalid counter')
    for rows in (terminal, terminal_stage, bodies):
        assert len(rows) == len(layers) and {int(r[0]) for r in rows} == layers, (name, 'missing or duplicate terminal/body layer')
    body_map = {int(r[0]): r for r in bodies}
    slot_map = {int(r[0]): r for r in terminal}
    stage_map = {int(r[0]): r for r in terminal_stage}
    per = {}
    for layer in sorted(layers):
        sr = [r for r in stage if r[0] == layer]
        cr = [r for r in counters if r[0] == layer]
        assert [r[1] for r in sr] == [r[1] for r in cr] == list(range(16, expected+1, 16)), (name, layer, 'counter order')
        slots, records = map(int, body_map[layer][1:3])
        assert slots == math.ceil(.55*records) and 0 < slots < records and all(r[3] == records for r in sr), (name, layer, 'body capacity')
        assert all(r[5] == slots and r[3] <= r[2] <= r[1]*records and r[2] > 0 for r in cr), (name, layer, 'counter bounds')
        assert all(a[2] <= b[2] and a[3] <= b[3] for a, b in zip(cr, cr[1:])), (name, layer, 'counter decrease')
        t = slot_map[layer]
        assert list(map(int, t[1:3])) == [slots, expected] and int(t[5]) == 0 and float(t[7]) == 0, (name, layer, 'incomplete terminal or host reads')
        assert cr[-1][2] <= int(t[3]) <= expected*records and cr[-1][3] <= int(t[4]) <= int(t[3]), (name, layer, 'terminal totals')
        history = [(0, 0, 0)] + [(r[1], r[2], r[3]) for r in cr]
        history += [(expected, int(t[3]), int(t[4]))]
        assert all(0 <= b[2]-a[2] <= min(b[1]-a[1], (b[0]-a[0])*slots) and
            0 <= b[1]-a[1] <= (b[0]-a[0])*records for a, b in zip(history, history[1:])), (name, layer, 'impossible counter interval')
        st = stage_map[layer]
        # Staging totals lag one resolve. Do not count the final sample at TF resolve 128.
        sampled = sum(r[2] for r in sr if r[1] < expected)
        staged = int(st[2])
        last = sr[-1][2] if sr[-1][1] == expected else records
        # Each resolve can serve at most slots picks there. Only the final stage may be uncounted.
        lower = max(sampled, int(t[3])-expected*slots-last)
        upper = min((expected-1)*records, int(t[3])-int(t[4])-(last if sr[-1][1] == expected else 0))
        assert lower <= staged <= upper, (name, layer, 'contradictory stage total')
        assert abs(float(st[4])-staged*cr[0][6]/2**20/expected) <= .0051, (name, layer, 'stage rate mismatch')
        assert abs(float(t[6])-int(t[4])*cr[0][6]/2**20/expected) <= .0051, (name, layer, 'fetch rate mismatch')
        per[str(layer)] = dict(samples=len(sr), resolves=expected,
            requested_mean=sum(r[2] for r in sr)/len(sr), requested_max=max(r[2] for r in sr), capacity=[records])
    assert all(int(r[1]) == int(body_map[int(r[0])][2]) and int(r[3]) == 0 and float(r[5]) == 0 and
        0 <= int(r[2]) <= expected*int(r[1]) for r in terminal_stage), (name, 'terminal stage-full or bounds')
    strides = {r[6] for r in counters}
    capacities = {r[3] for r in stage}
    assert len(strides) == len(capacities) == 1, (name, 'inconsistent geometry')
    stride, capacity = next(iter(strides)), next(iter(capacities))
    # One CUDA device, one shared backend buffer/type, identical geometry in all layers.
    assert len(allocations) == 1 and all(int(n) == capacity and
        math.isfinite(float(mib)) and abs(float(mib)-capacity*stride/2**20) <= 0.1
        for n, mib in allocations), (name, 'allocation mismatch')
    assert len(inits) == 1 and all(int(heads) == 4 and int(slots) == int(bodies[0][1])
        and abs(float(mib)-int(slots)*stride/2**20) <= 0.1 for slots, heads, mib in inits), (name, 'missing or inconsistent init')
    assert all(all(math.isfinite(float(v)) for v in row[3:]) and
        all(abs(float(value)-int(row[1 if i == 0 else 2])*stride/2**20) <= .1
        for i, value in enumerate(row[3:])) for row in bodies), (name, 'body allocation mismatch')
    body_bytes = sum(int(r[2])*stride for r in bodies)
    slot_bytes = sum(int(r[1])*stride for r in bodies)
    stage_bytes = len(allocations)*capacity*stride
    return dict(completion=receipt, diagnostic_rows=len(diagnostics), host_records=0,
        terminal=terminal, terminal_stage=terminal_stage, per_layer=per,
        slot_bytes=slot_bytes, stage_bytes=stage_bytes, full_body_bytes=body_bytes,
        effective_card_fraction=(slot_bytes+stage_bytes)/body_bytes,
        stage_allocation_mib_logged=allocations,
        note='payload fraction only; maps/index/ring/model/compute extra; prefill can read host KV')


def greedy(name, tokens):
    server = xver.Server(name, p.B, config(128, 44840))
    try:
        warm = server.gen(tokens[:p.P], 64)
        out = server.gen(tokens[:p.P], 512)
    finally:
        server.stop()
    (p.O / f'{name}.warm.json').write_text(json.dumps(warm, indent=1))
    if not collect_only:
        assert len(warm['tokens']) == 64
        assert len(out['tokens']) == 512 and out['timings']['predicted_n'] == 512
        assert out['timings']['prompt_n'] < 1024, (name, 'warm prefix not reused')
    out['stage'] = {} if collect_only else read_log(name)
    out['loop_v3'] = rule_v3(out['tokens'])
    (p.O / f'{name}.json').write_text(json.dumps(out, indent=1))
    if not collect_only:
        assert not out['loop_v3']['looped_v3'], (name, 'loop rule v3', out['loop_v3'])
    return out


def main():
    global resident
    assert analyze_only or os.environ.get('PARENT_HOLDS_GPU_LOCK') == '1'
    assert Path(p.B, 'COMMIT').read_text().split()[0] == '9abe482b7'
    assert json.loads((J/'memB-r320/off/xver.json').read_text())['pass'] is True
    if analyze_only:
        check_cuda(JOB / 'cuda.log')
    tokens = None
    if not analyze_only:
        p.O.mkdir()
        (p.O/'slots').mkdir()
        values = array('i')
        with open(p.T, 'rb') as source:
            values.fromfile(source, 100513)
        assert values.itemsize == 4
        tokens = values.tolist()
    p.P = 98000
    results = []
    for name, resident in [('a0', False), ('b0', True), ('a1', False), ('b1', True)]:
        if analyze_only:
            out = json.loads((p.O / f'{name}.json').read_text())
            out['stage'] = read_log(name)
            out['loop_v3'] = rule_v3(out['tokens'])
            assert len(out['tokens']) == 512 and not out['loop_v3']['looped_v3'], name
        else:
            out = greedy(name, tokens)
        results.append(out)
        print(name, out['timings'], out['stage'], flush=True)
    if collect_only:
        p.P = 100000
        for name, resident in [('f16_20_nores', False), ('f16_20_stage', True)]:
            p.teacher(name, True, tokens)
        (p.O/'collection.json').write_text(json.dumps(dict(arms=6, collected=True, verdict='UNASSESSED')))
        print('LSA MEMB COLLECTION COMPLETE arms=6 verdict=UNASSESSED', flush=True)
        return
    ids_equal = all(r['tokens'] == results[0]['tokens'] for r in results)
    ms = [r['timings']['predicted_per_token_ms'] for r in results]
    assert all(math.isfinite(v) and v > 0 for v in ms), 'invalid timing'
    summary = dict(rs3_scale_ms_token=ms, ids_equal=ids_equal,
        loop_v3=[r['loop_v3'] for r in results],
        drift_ok=abs(ms[0]-ms[2]) <= 0.5, resident_17_5_ok=max(ms[1],ms[3]) <= 17.5,
        stage=[r['stage'] for r in results],
        remaining='scalar/trellis width5/8/cold overflow/prefill resident coverage and full memory_breakdown accounting remain open; screen_pass is not full B pass',
        scope='98K/512 greedy/no MTP/graphs OFF diagnostic; not product throughput or context-fit proof')
    (p.O/'summary.json').write_text(json.dumps(summary, indent=1))
    assert ids_equal, 'resident vs nonresident generated tokens differ'
    # r314 was never built. Use its delivered pre-B numerical path plus a same-binary nores control.
    p.P = 100000
    for name, resident in [('f16_20_nores', False), ('f16_20_stage', True)]:
        if not analyze_only:
            p.teacher(name, True, tokens)
        summary[name] = read_log(name)
    sys.path.insert(0, str(J))
    from kld_analyze import load, per_pos
    _, off = load(J/'memA1-r309/run/off32.jsonl')
    _, old = load(J/'memA1-r309/run/f16_20_tf.jsonl')
    _, nores = load(p.O/'f16_20_nores.jsonl')
    _, staged = load(p.O/'f16_20_stage.jsonl')
    assert set(off) == set(old) == set(nores) == set(staged) and len(off) == 128
    q = {name: float(per_pos(off, arm)[1].mean()) for name, arm in
         [('old',old), ('nores',nores), ('staged',staged)]}
    q['staged_vs_nores_kld'] = float(per_pos(nores, staged)[1].mean())
    assert all(math.isfinite(v) for v in q.values()), 'invalid KLD'
    q['first_top1_equal'] = old[min(old)]['ids'][0] == staged[min(staged)]['ids'][0]
    q['delta_vs_preB'] = q['staged']-q['old']
    summary['teacher_forced_top1024'] = q
    summary['screen_pass'] = ids_equal and summary['drift_ok'] and summary['resident_17_5_ok'] and q['first_top1_equal'] and abs(q['delta_vs_preB']) <= .0005 and q['staged_vs_nores_kld'] <= 1e-6
    (p.O/'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)
    assert summary['screen_pass'], 'B missed gate; diagnose original logs before more code'


if __name__ == '__main__':
    main()
