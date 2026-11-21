#!/usr/bin/env python3
"""MTPRETRAIN: generate on-policy replies with the quantized model through llama-server.
Sampler = production chat sampler (temp 1.0, top_k 20, top_p 0.95, min_p 0); seed = per-prompt hash; reply cap N_PREDICT.
Plain-text prompts get the server's chat template (/apply-template, one user message), so every prompt ends in the
assistant turn opener the model sees in chat. Writes one JSON line per prompt: id, split, source, topic, prompt_tokens,
gen_tokens, stop_type, draft_n, draft_n_accepted. Resumes (skips ids already in OUT). Stops taking new prompts once
the generated-token total reaches --budget.
usage: gen.py --prompts P.jsonl --out G.jsonl [--port 18480] [--workers 8] [--npred 2048] [--budget 2200000]"""
import argparse, hashlib, json, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor


def post(base, path, body, timeout=3600):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompts', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--port', type=int, default=18480); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--npred', type=int, default=2048); ap.add_argument('--budget', type=int, default=2_200_000)
    a = ap.parse_args()
    base = f'http://127.0.0.1:{a.port}'
    prompts = [json.loads(l) for l in open(a.prompts)]
    done, total = set(), 0
    try:
        for l in open(a.out):
            d = json.loads(l); done.add(d['id']); total += len(d['gen_tokens'])
    except FileNotFoundError:
        pass
    lock = threading.Lock(); state = dict(total=total, n=len(done), t0=time.time(), gen0=total, fail=0)
    fout = open(a.out, 'a')

    def work(p):
        with lock:
            if state['total'] >= a.budget:
                return
        if 'prompt_tokens' in p:
            toks = p['prompt_tokens']
        else:
            tpl = post(base, '/apply-template', dict(messages=[dict(role='user', content=p['text'])]))['prompt']
            toks = post(base, '/tokenize', dict(content=tpl, add_special=False, parse_special=True))['tokens']
        seed = int(hashlib.sha256(p['id'].encode()).hexdigest()[:8], 16) % 2_000_000_000
        body = dict(prompt=toks, n_predict=a.npred, temperature=1.0, top_k=20, top_p=0.95, min_p=0.0, repeat_penalty=1.0,
                    seed=seed, return_tokens=True, cache_prompt=False, stream=False)
        r = post(base, '/completion', body)
        tm = r.get('timings', {})
        row = dict(id=p['id'], split=p['split'], chash=p.get('chash'), source=p['source'], topic=p['topic'], prompt_tokens=toks, gen_tokens=r['tokens'],
                   stop_type=r.get('stop_type'), draft_n=tm.get('draft_n'), draft_n_accepted=tm.get('draft_n_accepted'),
                   predicted_ms=tm.get('predicted_ms'))
        with lock:
            fout.write(json.dumps(row) + '\n'); fout.flush()
            state['total'] += len(r['tokens']); state['n'] += 1
            if state['n'] % 20 == 0:
                dt = time.time() - state['t0']
                print(f"GEN {state['n']} prompts, {state['total']} tokens, {(state['total'] - state['gen0']) / dt:.1f} tok/s aggregate", flush=True)

    def safe(p):
        try:
            work(p)
        except Exception as e:  # one failed request must not stop the run; the id is retried on resume
            print(f"FAIL {p['id']} {e!r}", flush=True)
            with lock:
                state['fail'] += 1

    todo = [p for p in prompts if p['id'] not in done]
    with ThreadPoolExecutor(a.workers) as ex:
        for f in [ex.submit(safe, p) for p in todo]:
            f.result()
    if state['fail']:  # no GEN_DONE: the runner reruns gen.py, which resumes and retries only the failed ids
        print(f"GEN_INCOMPLETE {state['fail']} failed, {state['n']} prompts, {state['total']} tokens", flush=True)
        return 1
    print(f"GEN_DONE {state['n']} prompts, {state['total']} tokens", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
