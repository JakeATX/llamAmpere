#!/usr/bin/env python3
"""DEFSYNC table check (CPU only, no model): every env entry of the arch-defaults table must agree with the code.

The table (common/arch-defaults.cpp, between the ARCHJSON markers) is the one source of default settings. Per row entry:
  env_on    the code default must be ON (unset = on, the "off" value turns it off)
  apply_env the code default must be OFF / a different value (the table sets it), so the table is what turns it on
  watch     the code default must be OFF (gated)
and no code other than the table may set a table name with setenv / _putenv_s / common_set_env or a {"NAME", "v"} pair
(the 11-19 trap and the 0.7.5 llama_model_load stopgap). Card presets (common/preset.cpp) are a visible second layer
and are listed as INFO, not failures.

The reader's default comes from its getenv() call and the two lines after it (or the helper it calls). Patterns that
the classifier does not know print as UNKNOWN (INFO) so they can be added; they never pass as agreement silently.

usage: check_table.py [--root SRC_TREE]      exit 0 = PASS, 1 = FAIL
"""
import argparse, json, re, sys
from pathlib import Path

# (pattern, default when unset), most specific first; X = the getenv value. First match wins.
PATTERNS = [
    (r'if\s*\(\s*!X\s*\|\|\s*!\*X\s*\)\s*\{?\s*return\s+0', 'OFF'),                                    # if (!e || !*e) return 0
    (r'!\(\s*X\s*(!=\s*nullptr\s*)?&&\s*\*X\s*&&\s*(std::)?atoi\(X\)\s*!=\s*0\s*\)', 'OFF'),             # !(e && *e && atoi(e) != 0)
    (r'!\(\s*X\s*(!=\s*nullptr\s*)?(&&\s*\*?X\s*)?&&\s*((std::)?atoi\(X\)\s*==\s*0|strcmp\(X,\s*"0"\)\s*==\s*0)\s*\)', 'ON'),
    (r'if\s*\(\s*X\s*&&\s*(std::)?atoi\(X\)\s*==\s*0\s*\)\s*\{?\s*return', 'ON'),                      # if (e && atoi(e) == 0) return
    (r'X\s*\?\s*(std::)?ato(i|ll)\(X\)\s*:\s*0\b', 'OFF'), (r'X\s*&&\s*\*X\s*\?\s*(std::)?atoi\(X\)\s*:\s*0\b', 'OFF'),
    (r'X\s*\?\s*(std::)?ato(i|ll)\(X\)\s*:\s*[1-9]', 'ON'),                                                # e ? atoi(e) : 4
    (r'(!\s*X|X\s*==\s*nullptr)\s*\|\|', 'ON'),                                                           # !e || atoi(e) != 0
    (r'(return\s+|=\s*|\(\s*)X\s*(!=\s*nullptr\s*)?&&\s*(\*X\s*&&\s*|X\[0\]\s*&&\s*)?((std::)?atoi\(X\)|strcmp\(X,\s*"1"\)\s*==\s*0)', 'OFF'),
]
SRC_DIRS = ['src', 'common', 'ggml/src', 'tools/server', 'tools/llama-bench', 'tools/batched-bench', 'tools/fit-params']
EXTS = {'.c', '.cc', '.cpp', '.h', '.hpp', '.cu', '.cuh'}


def table(root):
    txt = (root / 'common' / 'arch-defaults.cpp').read_text()
    return json.loads(re.search(r'R"ARCHJSON\((.*?)\)ARCHJSON"', txt, re.S).group(1))


def files(root):
    for d in SRC_DIRS:
        for p in (root / d).rglob('*'):
            if p.suffix in EXTS and p.is_file():
                yield p


def classify(window):
    w = re.sub(r'\s+', ' ', window)
    for pat, cls in PATTERNS:
        if re.search(pat, w): return cls
    return None


def reader_defaults(root, name, texts, helpers):
    """code default per reader site of `name`: list of (file:line, ON|OFF|UNKNOWN)"""
    out = []
    for p, lines in texts.items():
        if p.name == 'arch-defaults.cpp': continue
        for i, l in enumerate(lines):
            if f'"{name}"' not in l: continue
            m = re.search(r'(\w+)\s*=\s*(std::)?getenv\("' + re.escape(name) + r'"\)', l)
            win = '\n'.join(lines[i:i + 3])
            if m:
                cls = classify(re.sub(r'\b' + re.escape(m.group(1)) + r'\b', 'X', win))
            elif re.search(r'(std::)?getenv\("' + re.escape(name) + r'"\)', l):
                cls = classify(re.sub(r'(std::)?getenv\("' + re.escape(name) + r'"\)', 'X', win))
            else:
                h = re.search(r'\b(\w+)\("' + re.escape(name) + r'"\)', l)
                if not h or h.group(1) not in helpers: continue
                cls = helpers[h.group(1)]
            out.append((f'{p.relative_to(root)}:{i + 1}', cls or 'UNKNOWN'))
    return out


def helper_defaults(texts):
    """static bool/int helper(const char * name) { getenv(name) ... }: default of 'unset' per helper name"""
    h = {}
    for p, lines in texts.items():
        src = '\n'.join(lines)
        for m in re.finditer(r'static\s+(bool|int)\s+(\w+)\s*\(\s*const char\s*\*\s*(\w+)\s*\)\s*\{(.*?)\n\}', src, re.S):
            body, arg = m.group(4), m.group(3)
            g = re.search(r'(\w+)\s*=\s*(std::)?getenv\(' + arg + r'\)', body)
            if g:
                cls = classify(re.sub(r'\b' + g.group(1) + r'\b', 'X', body))
            else:
                inner = re.search(r'return\s+(\w+)\(' + arg + r'\)\s*!=\s*0', body)
                cls = ('ON' if h.get(inner.group(1)) == 'ON' else 'OFF') if inner and inner.group(1) in h else None
            if cls: h[m.group(2)] = cls
    # a second pass for helpers defined via another helper (gq_env_on -> gq_env_int)
    for p, lines in texts.items():
        src = '\n'.join(lines)
        for m in re.finditer(r'static\s+bool\s+(\w+)\s*\(\s*const char\s*\*\s*(\w+)\s*\)\s*\{\s*return\s+(\w+)\(\2\)\s*!=\s*0;', src):
            if m.group(3) in h: h[m.group(1)] = h[m.group(3)]
    return h


def setters(name, texts):
    """code that sets `name` (not the table): file:line list, preset ones separately"""
    pat = re.compile(r'(setenv|_putenv_s|common_set_env|set_unset|llama_lsa_setenv)\s*\(\s*"' + re.escape(name) + r'"|\{\s*"' + re.escape(name) + r'"\s*,\s*"[^"]*"\s*\}')
    hits, preset = [], []
    for p, lines in texts.items():
        if p.name == 'arch-defaults.cpp': continue
        for i, l in enumerate(lines):
            if pat.search(l) and not l.lstrip().startswith('//'):
                (preset if p.name == 'preset.cpp' else hits).append(f'{p.name}:{i + 1}')
    return hits, preset


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', default=str(Path(__file__).resolve().parents[2]))
    root = Path(ap.parse_args().root)
    tab = table(root)
    texts = {p: p.read_text(errors='replace').splitlines() for p in files(root)}
    helpers = helper_defaults(texts)
    want = {'env_on': 'ON', 'apply_env': 'OFF', 'watch': 'OFF'}
    fails = 0
    for row in tab['rows']:
        for lst, exp in want.items():
            for e in row.get(lst, []):
                name = e['env']
                rd = reader_defaults(root, name, texts, helpers)
                got = sorted({c for _, c in rd})
                bad = [f'{w} = {c}' for w, c in rd if c not in (exp, 'UNKNOWN')]
                unk = [w for w, c in rd if c == 'UNKNOWN']
                hits, preset = setters(name, texts)
                st = 'FAIL' if bad or hits or not rd else 'OK'
                if not rd: bad = ['no reader found']
                print(f'{st:4} {row["name"][:22]:22} {lst:9} {name:34} code default {"/".join(got) or "-":12}'
                      + (f' | table says {exp}: ' + '; '.join(bad) if bad else '')
                      + (f' | hidden setter: {", ".join(hits)}' if hits else '')
                      + (f' | INFO preset sets it: {", ".join(preset)}' if preset else '')
                      + (f' | INFO unclassified reader: {", ".join(unk)}' if unk else ''))
                fails += st == 'FAIL'
    print(f'CHECK_TABLE {"PASS" if not fails else "FAIL"}: {fails} entr{"y" if fails == 1 else "ies"} disagree with the code (table {tab["version"]})')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
