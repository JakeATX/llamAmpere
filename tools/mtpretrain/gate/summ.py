#!/usr/bin/env python3
"""MTPRETRAIN gate summary. usage: summ.py OUT   (OUT/<tag>/<label>.s<seeds>/<slot>/ from cell2.sh + depth_speed.py)
Speed tags g100, g8: per seed tok/s, acceptance, tok/round, ms/round (server 'round cost by draft width' lines), paired
B/A per seed, mean ratio with a t 95% CI (df = n-1), wins, peak MiB. Identity tags id8 (exact MTP), id8off (MTP off):
greedy token ids A vs B, first divergence."""
import glob, json, math, os, re, sys
from statistics import mean, stdev

W = re.compile(r'w(\d+): (\d+) rounds, ([\d.]+) tok/round, ([\d.]+) ms/round')
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201}


def rounds(log):
    """task id -> (rounds, tokens, ms) from the per-task round-cost lines."""
    out = {}
    for line in open(log, errors='replace'):
        if 'round cost by draft width' not in line:
            continue
        t = int(re.search(r'task (\d+)', line).group(1))
        ws = [tuple(float(x) for x in m.groups()) for m in W.finditer(line)]
        out[t] = (sum(w[1] for w in ws), sum(w[1] * w[2] for w in ws), sum(w[1] * w[3] for w in ws))
    return out


def cells(o, tag):
    rows = {}
    for slot in glob.glob(f'{o}/{tag}/*.s*/*/'):
        lab = os.path.basename(os.path.dirname(os.path.dirname(slot))).split('.')[0]
        rj = slot + 'rows.json'
        if not os.path.exists(rj):
            continue
        rb = rounds(slot + 'server.log')
        gen = {}
        for line in open(slot + 'server.log', errors='replace'):
            m = re.search(r'task (\d+) \|\s+eval time =\s+[\d.]+ ms /\s+(\d+) tokens', line)
            if m:
                gen[int(m.group(1))] = int(m.group(2))
        tasks = [t for t in sorted(gen) if t in rb and gen[t] > 16]
        peak = json.load(open(slot + 'COMPLETE.json'))['peak_mib'] if os.path.exists(slot + 'COMPLETE.json') else None
        data = json.load(open(rj))
        groups, k = [], 0  # a seed's reply may span turns: take tasks in order until they add up to its generated count
        for r in data:
            cur, acc = [], 0
            while k < len(tasks) and acc < r['generated']:
                cur.append(tasks[k]); acc += gen[tasks[k]]; k += 1
            if acc != r['generated']:
                print(f"MISMATCH {slot} seed {r['seed']}: grouped {acc} tokens vs generated {r['generated']}; tok/round and ms/round unreliable")
            groups.append(cur)
        log = open(slot + 'server.log', errors='replace').read()
        ovl = len(re.findall(r'overlay patch \S+: 15 tensors replaced', log))
        for i, r in enumerate(data):
            R = Tk = Ms = 0
            if i < len(groups):
                for t in groups[i]:
                    R += rb[t][0]; Tk += rb[t][1]; Ms += rb[t][2]
            rows.setdefault(lab, {})[r['seed']] = dict(tok_s=r['tok_s'], acc=r['acceptance'], valid=r['valid'], gen=r['generated'],
                prompt=r['prefix_tokens'], draft_n=r['draft_n'], tpr=Tk / R if R else float('nan'), mspr=Ms / R if R else float('nan'),
                peak=peak, slot=slot, ovl=ovl)
    return rows


def check_overlay(rows, tag):
    """B/BX servers must log the 15-tensor overlay, A/AX servers must not; otherwise the A/B is void."""
    bad = [(lab, s) for lab, d in rows.items() for s, v in d.items() if (v['ovl'] > 0) != lab.startswith('B')]
    print(f'{tag}_OVERLAY ' + ('OK (B loads 15 tensors, A none)' if not bad else f'VOID: wrong overlay state {bad}'))
    return not bad


def speed(o, tag):
    rows = cells(o, tag)  # overlay proof comes from the -lv 4 identity cells (same DS_ENV); loader INFO is hidden here
    A, B = rows.get('A', {}), rows.get('B', {})
    seeds = sorted(set(A) & set(B))
    print(f'\n## {tag}: prompt {A[seeds[0]]["prompt"] if seeds else "?"} tokens')
    print('| seed | A tok/s | B tok/s | B/A | A accept | B accept | A tok/round | B tok/round | A ms/round | B ms/round |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for s in seeds:
        a, b = A[s], B[s]
        print(f"| {s} | {a['tok_s']:.2f} | {b['tok_s']:.2f} | {b['tok_s'] / a['tok_s']:.4f} | {a['acc']:.3f} | {b['acc']:.3f} | {a['tpr']:.3f} | {b['tpr']:.3f} | {a['mspr']:.2f} | {b['mspr']:.2f} |")
    if len(seeds) < 2 or not all(A[s]['valid'] and B[s]['valid'] for s in seeds):
        print(f'{tag}_SUMMARY incomplete or invalid: A {sorted(A)} B {sorted(B)}'); return
    r = [B[s]['tok_s'] / A[s]['tok_s'] for s in seeds]
    n = len(r); m = mean(r); se = stdev(r) / math.sqrt(n); h = T975[n - 1] * se
    wins = sum(x > 1 for x in r)
    ma, mb = mean(A[s]['tok_s'] for s in seeds), mean(B[s]['tok_s'] for s in seeds)
    acca, accb = mean(A[s]['acc'] for s in seeds), mean(B[s]['acc'] for s in seeds)
    ta, tb = mean(A[s]['tpr'] for s in seeds), mean(B[s]['tpr'] for s in seeds)
    pa, pb = max(A[s]['peak'] or 0 for s in seeds), max(B[s]['peak'] or 0 for s in seeds)
    print(f'{tag}_SUMMARY n={n}: A {ma:.2f} tok/s, B {mb:.2f} tok/s, B/A mean {m:.4f} 95% CI [{m - h:.4f}, {m + h:.4f}] '
          f'(paired se {se:.4f}), wins {wins}/{n}; acceptance A {acca:.3f} B {accb:.3f}; tok/round A {ta:.3f} B {tb:.3f}; '
          f'peak MiB A {pa} B {pb}')


def ident(o, tag, la, lb):
    sa = glob.glob(f'{o}/{tag}/{la}.s*/*/tokens_seed*_turn0.json'); sb = glob.glob(f'{o}/{tag}/{lb}.s*/*/tokens_seed*_turn0.json')
    if not sa or not sb:
        print(f'{tag}_IDENTITY missing ({la} {len(sa)} files, {lb} {len(sb)} files)'); return
    ta, tb = json.load(open(sa[0])), json.load(open(sb[0]))
    rows = cells(o, tag)
    check_overlay(rows, tag)
    dn = {lab: [v['draft_n'] for v in rows.get(lab, {}).values()] for lab in (la, lb)}
    if tag == 'id8off' and any(x for v in dn.values() for x in v):
        print(f'{tag}_IDENTITY VOID: MTP was not off (draft_n {dn})'); return
    d = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
    if d is None and len(ta) == len(tb):
        print(f'{tag}_IDENTITY IDENTICAL {len(ta)} tokens ({la} vs {lb}; draft_n {dn})')
    else:
        print(f'{tag}_IDENTITY DIFFER first divergence at token {d} of {min(len(ta), len(tb))} ({la} vs {lb}; draft_n {dn}); check top-2 margin')


def main():
    o = sys.argv[1]
    print('MTPRETRAIN gate: HOME RTX 3090 Ti 350 W, 27B P76 AI4 (Swift 1.5), v0.7.2.1 rel86 bundle, bare --preset 24dense -c 262144, '
          'chat temp 1 top_k 20 top_p 0.95, 5,120-token reply, LLAMA_ARG_NO_CACHE_DISK=1; A = in-file MTP head, B = retrained head (overlay)')
    for tag in ('g100', 'g8'):
        speed(o, tag)
    ident(o, 'id8', 'AX', 'BX')
    ident(o, 'id8off', 'A', 'B')


if __name__ == '__main__':
    main()
