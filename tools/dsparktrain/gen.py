#!/usr/bin/env python3
"""DSPARKTRAIN: on-policy replies from the AI4 target through llama-server (drafting off; the target's own law).
Sampler = production chat sampler (temp 1.0, top_k 20, top_p 0.95, min_p 0, repeat_penalty 1). Messages are templated
by the server (/apply-template, generation prompt on) and tokenized there, so the sequence is exactly what the server
would decode. seed = per-prompt hash. Prompts over --max-prompt tokens are skipped (logged as SKIP rows). Resumes
(skips ids already in OUT). Stops taking prompts once the reply-token total reaches --budget.
usage: gen.py --prompts P.jsonl --out G.jsonl [--port 18480] [--workers 16] [--npred 4096] [--max-prompt 2048]
              [--budget 2000000] [--splits train,sel,rep]"""
import argparse, hashlib, json, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor


def post(base, path, body, timeout=7200):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompts', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--port', type=int, default=18480); ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--npred', type=int, default=4096); ap.add_argument('--max-prompt', type=int, default=2048)
    ap.add_argument('--budget', type=int, default=2_000_000); ap.add_argument('--splits', default='train,sel,rep')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    base = f'http://127.0.0.1:{a.port}'
    splits = set(a.splits.split(','))
    prompts = [p for p in (json.loads(l) for l in open(a.prompts)) if p['split'] in splits]
    if a.limit:
        prompts = prompts[:a.limit]
    done, total = set(), 0
    try:
        for l in open(a.out):
            d = json.loads(l); done.add(d['id']); total += len(d.get('gen_tokens') or [])
    except FileNotFoundError:
        pass
    lock = threading.Lock()
    st = dict(total=total, n=len(done), t0=time.time(), gen0=total, fail=0)
    fout = open(a.out, 'a')

    def work(p):
        with lock:
            if st['total'] >= a.budget:
                return
        tpl = post(base, '/apply-template', dict(messages=p['messages'], add_generation_prompt=True))['prompt']
        toks = post(base, '/tokenize', dict(content=tpl, add_special=False, parse_special=True))['tokens']
        if len(toks) > a.max_prompt:
            row = dict(id=p['id'], split=p['split'], skip='prompt_too_long', prompt_len=len(toks))
        else:
            seed = int(hashlib.sha256(p['id'].encode()).hexdigest()[:8], 16) % 2_000_000_000
            body = dict(prompt=toks, n_predict=a.npred, temperature=1.0, top_k=20, top_p=0.95, min_p=0.0, repeat_penalty=1.0,
                        seed=seed, return_tokens=True, cache_prompt=False, stream=False)
            r = post(base, '/completion', body)
            row = dict(id=p['id'], split=p['split'], chash=p.get('chash'), source=p['source'], topic=p['topic'],
                       prompt_tokens=toks, gen_tokens=r['tokens'], stop_type=r.get('stop_type'),
                       predicted_ms=r.get('timings', {}).get('predicted_ms'))
        with lock:
            fout.write(json.dumps(row) + '\n'); fout.flush()
            st['total'] += len(row.get('gen_tokens') or []); st['n'] += 1
            if st['n'] % 20 == 0:
                dt = time.time() - st['t0']
                print(f"GEN {st['n']} prompts, {st['total']} reply tokens, {(st['total'] - st['gen0']) / dt:.1f} tok/s aggregate", flush=True)

    def safe(p):
        try:
            work(p)
        except Exception as e:  # retried on resume
            print(f'FAIL {p["id"]} {e!r}', flush=True)
            with lock:
                st['fail'] += 1

    todo = [p for p in prompts if p['id'] not in done]
    with ThreadPoolExecutor(a.workers) as ex:
        for f in [ex.submit(safe, p) for p in todo]:
            f.result()
    dt = time.time() - st['t0']
    msg = f"{st['n']} prompts, {st['total']} reply tokens, {(st['total'] - st['gen0']) / max(dt, 1e-9):.1f} tok/s aggregate, {dt:.0f} s"
    if st['fail']:
        print(f"GEN_INCOMPLETE {st['fail']} failed, {msg}", flush=True)
        return 1
    print(f'GEN_DONE {msg}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
