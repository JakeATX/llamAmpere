#!/usr/bin/env python3
"""DEFPATH best-config gate: the default run of a model must not be slower than the arch table's best config.

The best config per arch and card class is the JSON table in common/arch-defaults.cpp (between the ARCHJSON markers).
Arms, same binary, same card, same protocol (chat template, temp 1, >= 5,120-token reply, cold prefill of --plen tokens):
  default = only the protocol flags (no preset, no tuning env)
  best    = the table's best row: --preset <name> + args + env
Order ABBA per seed (odd seeds default first, even seeds best first), one server boot per cell (inkbench.py speed mode).
FAIL when, over the paired seeds, the default's mean decode or prefill is more than --tol percent below best, when a
default cell fails or is short, or (--strict-engage) when the default's engaged-path report lists a differing lever.

usage: defpath_gate.py --bin BUILD/bin --model M.gguf [--card 24g] [--seeds 1,2,3,4,5,6] [--plen 100000] [--out DIR]
       [--bench /mnt/data/inkflags/h/inkbench.py] [--pin 0,5-8,17-19] [--tol 1.0] [--ctx 262144] [--dry-run]
Run it under the card queue (it boots llama-server on CUDA_VISIBLE_DEVICES / INK_GPU, default 0).
"""
import argparse, json, os, re, statistics as st, struct, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def table(src):
    txt = Path(src).read_text()
    m = re.search(r'R"ARCHJSON\((.*?)\)ARCHJSON"', txt, re.S)
    if not m:
        sys.exit(f'no ARCHJSON table in {src}')
    return json.loads(m.group(1))


def gguf_arch(path):
    """general.architecture from a GGUF header (no numpy)."""
    with open(path, 'rb') as f:
        if f.read(4) != b'GGUF':
            return ''
        ver, = struct.unpack('<I', f.read(4))
        n_t, n_kv = struct.unpack('<QQ', f.read(16))
        rs = lambda: f.read(struct.unpack('<Q', f.read(8))[0]).decode('utf-8', 'replace')
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        def skip(t):
            if t == 8: rs()
            elif t == 9:
                et, n = struct.unpack('<IQ', f.read(12))
                for _ in range(n): skip(et)
            else: f.read(sizes[t])
        for _ in range(n_kv):
            k = rs(); t, = struct.unpack('<I', f.read(4))
            if k == 'general.architecture' and t == 8:
                return rs()
            skip(t)
    return ''


def row_for(tab, arch):
    for r in tab['rows']:
        if arch in r['arch']:
            return r
    return None


def engaged(log):
    """the engaged-path block of a server log: (lines, differ list)"""
    lines, differ, on = [], [], False
    for l in Path(log).read_text(errors='replace').splitlines():
        if 'engaged path (DEFPATH' in l:
            on, lines = True, [l]
            continue
        if on:
            if re.search(r'\s\[(ON|OFF|OVERRIDE|N/A) *\]', l) or 'lever(s) differ' in l or 'all levers match' in l or 'no arch-defaults row' in l:
                lines.append(l)
                m = re.search(r'lever\(s\) differ from the \S+ best config: (.*)$', l)
                if m: differ = [x.strip() for x in m.group(1).split(',')]
                if 'lever(s) differ' in l or 'all levers match' in l or 'no arch-defaults row' in l:
                    on = False
            else:
                on = False
    return lines, differ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', required=True); ap.add_argument('--model', required=True)
    ap.add_argument('--src', default=str(HERE.parent.parent / 'common' / 'arch-defaults.cpp'))
    ap.add_argument('--card', default='24g', choices=['24g', 'small']); ap.add_argument('--seeds', default='1,2,3,4,5,6')
    ap.add_argument('--plen', type=int); ap.add_argument('--npred', type=int); ap.add_argument('--ctx', type=int, default=262144)
    ap.add_argument('--out', default='defpath_gate_out'); ap.add_argument('--bench', default='/mnt/data/inkflags/h/inkbench.py')
    ap.add_argument('--pin', default=''); ap.add_argument('--tol', type=float); ap.add_argument('--port', type=int, default=18791)
    ap.add_argument('--default-env', default='', help='extra env for the default arm (e.g. a kill switch A/B)')
    ap.add_argument('--best-extra', default='', help='extra args for the best arm (e.g. -md path when the preset cannot resolve it)')
    ap.add_argument('--strict-engage', action='store_true'); ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    tab = table(a.src); proto = tab['protocol']
    plen = a.plen or proto['fill']; npred = a.npred or proto['npred']; tol = proto['fail_if_default_slower_pct'] if a.tol is None else a.tol
    arch = gguf_arch(a.model); row = row_for(tab, arch)
    print(f'DEFPATH gate: table {tab["version"]}, model {a.model}, arch {arch or "?"}, card {a.card}, fill {plen}, npred {npred}, seeds {a.seeds}, tol {tol}%')
    if not row:
        print(f'FAIL no arch-defaults row for {arch!r}: add one to {a.src} (best config from its first speed gate)'); return 2
    best = (row.get('best') or {}).get(a.card)
    if not best:
        print(f'SKIP no frozen {a.card} best config for {row["name"]}: {row.get("best_note", "")}'); return 3
    base_env = 'LLAMA_ARG_NO_CACHE_DISK=1'
    arms = {
        'default': dict(extra='', env=(base_env + ' ' + a.default_env).strip()),
        'best': dict(extra=(f'--preset {best["preset"]} ' if best.get('preset') else '') + best.get('args', '') + ' ' + a.best_extra,
                     env=(base_env + ' ' + best.get('env', '')).strip()),
    }
    print(f'  best = {arms["best"]["extra"].strip()} env {arms["best"]["env"]} ({best.get("evidence", "")})')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(',')]
    res = {}
    for sd in seeds:
        order = ['default', 'best'] if sd % 2 == 1 else ['best', 'default']
        for arm in order:
            cell = out / f'{arm}.s{sd}'
            cmd = ([ 'taskset', '-c', a.pin ] if a.pin else []) + [sys.executable, '-I', a.bench, '--bin', a.bin, '--model', a.model,
                   '--out', str(cell), '--mode', 'speed', '--plen', str(plen), '--npred', str(npred), '--ctx', str(a.ctx),
                   '--seeds', str(sd), '--port', str(a.port), '--env', arms[arm]['env'], '--extra', arms[arm]['extra'].strip()]
            print('CELL', arm, sd, ' '.join(cmd), flush=True)
            if a.dry_run:
                continue
            p = subprocess.run(cmd, capture_output=True, text=True)
            row_m = re.search(r'^ROW (\d+) ([\d.]+) ([\d.]+) (\d+)', p.stdout, re.M)
            if row_m:
                res[(arm, sd)] = dict(pf=float(row_m[2]), dec=float(row_m[3]), gen=int(row_m[4]))
                print(f'  {arm} s{sd}: prefill {row_m[2]} decode {row_m[3]} gen {row_m[4]}', flush=True)
            else:
                print(f'  {arm} s{sd}: NO ROW\n{p.stdout[-2000:]}{p.stderr[-2000:]}', flush=True)
    if a.dry_run:
        return 0
    fails = []
    pairs = [(res[('default', s)], res[('best', s)]) for s in seeds if ('default', s) in res and ('best', s) in res]
    for s in seeds:
        d = res.get(('default', s))
        if not d: fails.append(f'default s{s} failed')
        elif d['gen'] < npred: fails.append(f'default s{s} short ({d["gen"]} < {npred})')
    print(f'\n{"arm":<8} {"prefill tok/s":>14} {"decode tok/s":>13}  per seed decode')
    for arm in ('default', 'best'):
        c = [res[(arm, s)] for s in seeds if (arm, s) in res]
        if c:
            print(f'{arm:<8} {st.mean(x["pf"] for x in c):14.1f} {st.mean(x["dec"] for x in c):13.2f}  ' + ' '.join(f'{x["dec"]:.2f}' for x in c))
    if pairs:
        dd = [d['dec'] / b['dec'] - 1 for d, b in pairs]; dp = [d['pf'] / b['pf'] - 1 for d, b in pairs]
        print(f'default vs best (n={len(pairs)} pairs): decode {100*st.mean(dd):+.2f}% (faster {sum(x > 0 for x in dd)}/{len(dd)}), '
              f'prefill {100*st.mean(dp):+.2f}% (faster {sum(x > 0 for x in dp)}/{len(dp)})')
        if 100 * st.mean(dd) < -tol: fails.append(f'default decode {100*st.mean(dd):+.2f}% < -{tol}%')
        if 100 * st.mean(dp) < -tol: fails.append(f'default prefill {100*st.mean(dp):+.2f}% < -{tol}%')
    else:
        fails.append('no paired seeds')
    logs = sorted(out.glob('default.s*/server.log'))
    if logs:
        lines, differ = engaged(logs[0])
        print('\nengaged path of the default arm (' + str(logs[0]) + '):')
        print('\n'.join(lines) if lines else '  (no DEFPATH report in the log: binary without DEFPATH or LLAMA_ENGAGED_REPORT=0)')
        if a.strict_engage and differ: fails.append('default differs from best on: ' + ', '.join(differ))
        if not lines: fails.append('no engaged-path report')
    print('\nRESULT', 'FAIL: ' + '; '.join(fails) if fails else 'PASS')
    (out / 'gate.json').write_text(json.dumps(dict(arch=arch, card=a.card, table=tab['version'], tol=tol,
        cells={f'{k[0]}.s{k[1]}': v for k, v in res.items()}, fails=fails), indent=1))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
