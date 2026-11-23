#!/usr/bin/env python3
"""DSPARKQAT gate summary: per row, S (shipped) vs N (retrained) decode tok/s, tok/round, acceptance; paired by seed.
usage: gate_sum.py OUTDIR   (reads OUTDIR/<row>.<arm>.s<seed>/result.json)"""
import json, math, re, sys
from pathlib import Path

O = Path(sys.argv[1]); rows = {}
for d in sorted(O.glob('*.*.s*')):
    m = re.fullmatch(r'(\w+)\.([SN])\.s(\d+)', d.name)
    if not m or not (d / 'result.json').exists():
        continue
    r = json.loads((d / 'result.json').read_text()); sp = r.get('spec') or {}; rr = (r.get('rows') or [{}])[0]
    pp = None   # per-position acceptance d1..d4 (LLAMA_SPEC_ACCPOS=1), weighted by drafted tokens over the run's requests
    if (d / 'server.log').exists():
        L = (d / 'server.log').read_text(errors='replace')
        pr = re.findall(r'draft acceptance = [0-9.]+ \( *\d+ accepted / *(\d+) generated\)[^\n]*\n[^\n]*acc per pos = \(([^)]*)\)', L)
        w = [(int(g), [float(v) for v in a.split(',')]) for g, a in pr if a.strip()]
        if w and sum(g for g, _ in w):
            pp = [sum(g * v[i] for g, v in w) / sum(g for g, _ in w) for i in range(min(len(v) for _, v in w))]
    ok = r.get('error') is None and rr.get('valid') and sp.get('tok_per_round') is not None and rr.get('decode_tok_s') is not None
    rows.setdefault(m[1], {}).setdefault(m[2], {})[int(m[3])] = dict(
        ok=ok, tps=rr.get('decode_tok_s'), tpr=sp.get('tok_per_round'), acc=sp.get('acceptance_rate'), pf=rr.get('prefill_tok_s'),
        ptok=r.get('prompt_tokens'), gen=rr.get('generated'), pp=pp)
out = {}
for row in ('100k', '32k', '3k'):
    if row not in rows:
        continue
    S, N = rows[row].get('S', {}), rows[row].get('N', {})
    seeds = sorted(k for k in S if k in N and S[k]['ok'] and N[k]['ok'])
    if not seeds:
        print(row, 'no valid pairs'); continue
    def mean(arm, k): return sum(arm[s][k] for s in seeds) / len(seeds)
    res = dict(seeds=seeds, prompt_tokens=S[seeds[0]]['ptok'])
    for k in ('tps', 'tpr', 'acc', 'pf'):
        a, b = mean(S, k), mean(N, k); diffs = [N[s][k] - S[s][k] for s in seeds]
        sd = (sum((x - sum(diffs) / len(diffs)) ** 2 for x in diffs) / max(len(diffs) - 1, 1)) ** .5
        res[k] = dict(S=round(a, 4), N=round(b, 4), pct=round(100 * (b / a - 1), 2), N_wins=sum(x > 0 for x in diffs),
                      paired_se_pct=round(100 * sd / math.sqrt(len(diffs)) / a, 2))
    for arm, A in (('S', S), ('N', N)):
        v = [A[s]['pp'] for s in seeds if A[s]['pp']]
        res['acc_pos_' + arm] = [round(sum(x[i] for x in v) / len(v), 4) for i in range(min(map(len, v)))] if v else None
    out[row] = res
    print(f"{row}: prompt {res['prompt_tokens']} tok, {len(seeds)} paired seeds | decode tok/s S {res['tps']['S']:.2f} N {res['tps']['N']:.2f} "
          f"({res['tps']['pct']:+.2f}%, N wins {res['tps']['N_wins']}/{len(seeds)}, se {res['tps']['paired_se_pct']}%) | tok/round S {res['tpr']['S']:.3f} "
          f"N {res['tpr']['N']:.3f} ({res['tpr']['pct']:+.2f}%, {res['tpr']['N_wins']}/{len(seeds)}) | accepted/drafted S {res['acc']['S']:.4f} N {res['acc']['N']:.4f} | per-pos acc d1..d4 S {res['acc_pos_S']} N {res['acc_pos_N']} | "
          f"prefill S {res['pf']['S']:.0f} N {res['pf']['N']:.0f}")
h = out.get('100k')
if h and len(h['seeds']) >= 6:
    go = h['tps']['pct'] > 0 and h['tpr']['pct'] > 0
    print('VERDICT', 'GO' if go else 'DROP', '(rule: retrained beats shipped on server tok/round AND tok/s at 100K, same card)')
    out['verdict'] = 'GO' if go else 'DROP'
if h and len(h['seeds']) < 6:
    print('NO_VERDICT: only', len(h['seeds']), 'valid 100K pairs (need 6)')
(O / 'summary.json').write_text(json.dumps(out, indent=1))
