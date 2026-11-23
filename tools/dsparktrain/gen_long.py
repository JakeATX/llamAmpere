#!/usr/bin/env python3
"""DSPARKQAT long-context on-policy replies (~100K fill) from a running llama-server (the NQ target, 4/4 KV, -np 1).
Per doc (prep_long.py jsonl): user message = context cut to fit + "\\n\\n" + task, chat-templated and tokenized by the server;
the cut is found once per doc (by bisection on chars, with the longest task) so both tasks share the doc prefix and the
second reply reuses the cached prefill. Sampler = bench sampler (temp 1, top_k 20, top_p 0.95, min_p 0, rp 1), seed per
(doc, task). Writes gens.jsonl rows (id, split=train, topic, prompt_tokens, gen_tokens, stop_type) for mkseq.py. Resumes.
usage: gen_long.py --docs L.jsonl --out G.jsonl [--port 18480] [--plen-lo 92000 --plen-hi 99000] [--npred 3072]"""
import argparse, hashlib, json, random, time, urllib.request


def post(base, path, body, timeout=7200):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ids_of(base, user):
    tpl = post(base, '/apply-template', dict(messages=[dict(role='user', content=user)], add_generation_prompt=True))['prompt']
    return post(base, '/tokenize', dict(content=tpl, add_special=False, parse_special=True))['tokens']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--docs', required=True); ap.add_argument('--out', required=True); ap.add_argument('--port', type=int, default=18480)
    ap.add_argument('--plen-lo', type=int, default=92000); ap.add_argument('--plen-hi', type=int, default=99000)
    ap.add_argument('--npred', type=int, default=3072)
    a = ap.parse_args()
    base = f'http://127.0.0.1:{a.port}'
    done = set()
    try:
        done = {json.loads(l)['id'] for l in open(a.out)}
    except FileNotFoundError:
        pass
    fout = open(a.out, 'a')
    docs = [json.loads(l) for l in open(a.docs)]
    t0 = time.time(); ntok = 0
    for d in docs:
        keys = [f"long:{d['id']}:t{j}" for j in range(len(d['tasks']))]
        if all(k in done for k in keys):
            continue
        h = int(hashlib.sha256(d['id'].encode()).hexdigest()[:8], 16)
        plen = random.Random(h).randint(a.plen_lo, a.plen_hi)
        longest = max(d['tasks'], key=len); ctx = d['context']
        lo, hi = 0, len(ctx)
        if len(ids_of(base, ctx + '\n\n' + longest)) <= plen:
            lo = hi
        while hi - lo > 256:
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if len(ids_of(base, ctx[:mid] + '\n\n' + longest)) <= plen else (lo, mid)
        for j, (k, task) in enumerate(zip(keys, d['tasks'])):
            if k in done:
                continue
            ids = ids_of(base, ctx[:lo] + '\n\n' + task)
            t1 = time.time()
            r = post(base, '/completion', dict(prompt=ids, n_predict=a.npred, temperature=1.0, top_k=20, top_p=0.95, min_p=0.0,
                                               repeat_penalty=1.0, seed=(h + j) % 2**31, return_tokens=True, cache_prompt=True, stream=False))
            g = r['tokens']; tm = r.get('timings', {}); ntok += len(g)
            fout.write(json.dumps(dict(id=k, split='train', topic=d['domain'], prompt_tokens=ids, gen_tokens=g, stop_type=r.get('stop_type'))) + '\n')
            fout.flush()
            print(f"GEN {k} prompt {len(ids)} reply {len(g)} stop {r.get('stop_type')} prefill_n {tm.get('prompt_n')} "
                  f"{tm.get('prompt_per_second', 0):.0f} tok/s decode {tm.get('predicted_per_second', 0):.1f} tok/s wall {time.time() - t1:.0f} s", flush=True)
    print(f"GEN_LONG_DONE {len(docs)} docs, {ntok} new reply tokens, {time.time() - t0:.0f} s", flush=True)


if __name__ == '__main__':
    main()
