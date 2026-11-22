#!/usr/bin/env python3
"""DSPARKTRAIN server check: boots llama-server (target + drafter, the bench flags), sends the inkbench speed prompt
(chat template, TASK + wikitext-2 test, --plen tokens) and replies of --npred tokens per seed at temp 1 / top_k 20 /
top_p 0.95, then reads the server's own counters (tok/round = mean len, acc per pos, round cost by width, tok/s).
Writes OUT/result.json and OUT/calib.seq (prompt + reply tokens of every seed, for llama-dspark-dump) so the offline
metric can be computed on exactly the sequences the server produced.
usage: spec_check.py --bin DIR --model T.gguf --drafter D.gguf --out DIR [--exact] [--seeds 1] [--plen 3000]
                     [--npred 3072] [--wiki PATH] [--port 18631] [--gpu 0]"""
import argparse, json, os, re, struct, subprocess, time, urllib.request
from pathlib import Path

TASK = ("Below is a long excerpt of encyclopedia articles. Write a detailed study guide for it: for every article, list the key facts, "
        "dates and people, explain how they connect, and finish with fifty exam questions with worked answers.\n\n")   # = inkbench.TASK


def post(base, path, body, timeout=7200):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def tok(base, text):
    return post(base, '/tokenize', dict(content=text, add_special=False, parse_special=True))['tokens']


def template(base, msgs):
    return post(base, '/apply-template', dict(messages=msgs, add_generation_prompt=True))['prompt']


def build_prompt(base, plen, wiki):
    """= inkbench.build_prompt"""
    text = Path(wiki).read_text()
    lo, hi = 0, min(len(text), plen * 8)
    while hi - lo > 64:
        mid = (lo + hi) // 2
        n = len(tok(base, template(base, [dict(role='user', content=TASK + text[:mid])])))
        lo, hi = (mid, hi) if n <= plen else (lo, mid)
    return tok(base, template(base, [dict(role='user', content=TASK + text[:lo])]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', required=True); ap.add_argument('--model', required=True); ap.add_argument('--drafter', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--exact', action='store_true')
    ap.add_argument('--seeds', default='1'); ap.add_argument('--plen', type=int, default=3000); ap.add_argument('--npred', type=int, default=3072)
    ap.add_argument('--wiki', default='/mnt/data/inkflags/data/wiki.test.raw'); ap.add_argument('--port', type=int, default=18631)
    ap.add_argument('--ctx', type=int, default=16384); ap.add_argument('--depth', type=int, default=4); ap.add_argument('--extra', default='')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_', 'TURBO_'))}
    env['LD_LIBRARY_PATH'] = a.bin + ':' + env.get('LD_LIBRARY_PATH', ''); env['LLAMA_SPEC_ACCPOS'] = '1'; env['LLAMA_ARG_NO_CACHE_DISK'] = '1'
    cmd = [os.path.join(a.bin, 'llama-server'), '-m', a.model, '--host', '127.0.0.1', '--port', str(a.port), '-c', str(a.ctx), '-ngl', '99',
           '-fa', 'on', '--parallel', '1', '-t', '8', '-tb', '8', '--jinja', '--no-webui',
           '--model-draft', a.drafter, '--spec-type', 'draft-dspark', '-ngld', '99', '-ubd', '64', '--spec-draft-n-max', str(a.depth)] + (['--no-spec-mb-mtp'] if a.exact else []) + a.extra.split()
    log = open(out / 'server.log', 'w')
    t0 = time.time()
    p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    base = f'http://127.0.0.1:{a.port}'
    res = dict(cmd=cmd, exact=a.exact, rows=[])
    seqs = []
    try:
        for _ in range(900):
            if p.poll() is not None:
                raise SystemExit('server exited')
            try:
                with urllib.request.urlopen(base + '/health', timeout=1) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(1)
        res['load_s'] = round(time.time() - t0, 1)
        ids = build_prompt(base, a.plen, a.wiki)
        res['prompt_tokens'] = len(ids)
        post(base, '/completion', dict(prompt=ids, n_predict=1, temperature=0.0, cache_prompt=False))  # warm-up
        for seed in [int(s) for s in a.seeds.split(',')]:
            body = dict(prompt=ids, n_predict=a.npred, temperature=1.0, top_k=20, top_p=0.95, min_p=0.0, repeat_penalty=1.0, seed=seed,
                        return_tokens=True, cache_prompt=False, stream=False)
            r = post(base, '/completion', body); tm = r.get('timings', {})
            res['rows'].append(dict(seed=seed, n=len(r['tokens']), stop_type=r.get('stop_type'), tok_s=tm.get('predicted_per_second'),
                                    draft_n=tm.get('draft_n'), draft_n_accepted=tm.get('draft_n_accepted')))
            seqs.append((ids, r['tokens']))
            print('ROW', res['rows'][-1], flush=True)
    finally:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(30)
            except subprocess.TimeoutExpired:
                p.kill()
        log.close()
    L = (out / 'server.log').read_text(errors='replace')
    res['mean_len'] = [float(x) for x in re.findall(r'mean len = *([0-9.]+)', L)]
    res['acc_per_pos'] = [[float(v) for v in m.split(',')] for m in re.findall(r'acc per pos = \(([^)]*)\)', L) if m.strip()]
    res['round_cost'] = re.findall(r'round cost by draft width = \((.*)\)', L)
    res['spec_pq'] = re.findall(r'spec pq: (.*)', L)
    res['depth_log'] = re.findall(r'(DSpark draft depth.*)', L)[:2]
    res['draft_kv_log'] = re.findall(r'(draft KV cache.*)', L)[:2]
    with open(out / 'calib.seq', 'wb') as f:
        f.write(struct.pack('<i', len(seqs)))
        for pr, g in seqs:
            s = pr + g
            f.write(struct.pack('<ii', len(s), len(pr))); f.write(struct.pack(f'<{len(s)}i', *s))
    (out / 'result.json').write_text(json.dumps(res, indent=1))
    print('DONE', json.dumps({k: res[k] for k in ('prompt_tokens', 'mean_len', 'acc_per_pos', 'round_cost')}), flush=True)


if __name__ == '__main__':
    main()
