#!/usr/bin/env python3
"""INKLING2 bench: one llama-server boot, then greedy / speed / chat-tool cells. Model-agnostic (no Qwen token ids).
usage: inkbench.py --bin DIR --model GGUF --out DIR --mode greedy|speed|chat [--plen 3000] [--seeds 1,2] [--npred 5120]
  greedy: 256-token temp-0 (top_k 1) reply to a fixed short chat prompt -> tokens.json
  speed : chat-templated prompt of --plen tokens (wikitext-2 test text + task), per seed: one cold prefill (n_predict 1,
          cache_prompt false) then a temp-1 reply of >= --npred tokens (follow-up user turns on EOS, as depth_speed.py)
  chat  : one plain chat and one tool call through /v1/chat/completions
The server gets identical flags for every --bin (upstream or ours); VRAM is sampled at 4 Hz on the whole card."""
import argparse, json, os, signal, subprocess, threading, time, urllib.request
from pathlib import Path

WIKI = '/mnt/data/inklsa/prompts/wiki.test.raw'
FOLLOWUPS = ["Good. Now review everything you produced above for mistakes or gaps, then extend it with the next logical stage of the work, written out in full detail.",
 "Continue. Write out completely every remaining piece you referred to but did not fully produce, and add concrete tests or checks for each one.",
 "Now write a thorough handover document for a colleague who has to take this over tomorrow: what was done, why, how to verify it, and every open question.",
 "The colleague came back with three objections to your handover. Anticipate the three most likely objections, answer each in depth, and revise the affected parts."]
TASK = ("Below is a long excerpt of encyclopedia articles. Write a detailed study guide for it: for every article, list the key facts, "
        "dates and people, explain how they connect, and finish with fifty exam questions with worked answers.\n\n")
GREEDY_PROMPT = "Explain, step by step and in detail, how a compiler turns a C function into machine code. Cover lexing, parsing, IR, optimisation passes, register allocation and instruction selection."
GREEDY_PROMPTS = [GREEDY_PROMPT,
 "Write a complete Python module that implements an LRU cache with time-based expiry, with type hints, docstrings and a full pytest test suite.",
 "A train leaves city A at 9:00 at 80 km/h and another leaves city B, 500 km away, at 10:30 at 120 km/h towards A. Work out exactly when and where they meet, then generalise the method and give three more worked examples."]


def post(base, path, body, timeout=7200):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def vram_mon(stop, samples):
    while not stop.is_set():
        try:
            samples.append(float(subprocess.check_output(['nvidia-smi', '--id=1', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True, timeout=10).strip()))
        except Exception:
            pass
        stop.wait(0.25)


def boot(a, log):
    b = Path(a.bin)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_', 'TURBO_'))}
    env['CUDA_VISIBLE_DEVICES'] = '1'; env['LLAMA_ARG_NO_CACHE_DISK'] = '1'; env['LD_LIBRARY_PATH'] = str(b)
    for kv in a.env.split():
        k, v = kv.split('=', 1); env[k] = v
    cmd = [str(b / 'llama-server'), '-m', a.model, '--host', '127.0.0.1', '--port', str(a.port), '-c', str(a.ctx), '-ngl', '99', '-fa', 'on',
           '--parallel', '1', '-t', '8', '-tb', '8', '--jinja', '--no-webui'] + a.extra.split()
    p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(1800):
        if p.poll() is not None:
            return None, cmd
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{a.port}/health', timeout=1) as r:
                if r.status == 200:
                    return p, cmd
        except Exception:
            time.sleep(1)
    stop(p); return None, cmd


def stop(p):
    try: os.killpg(p.pid, signal.SIGTERM)
    except Exception: pass
    try: p.wait(60)
    except Exception:
        try: os.killpg(p.pid, signal.SIGKILL)
        except Exception: pass
    time.sleep(3)


def tok(base, text):
    return post(base, '/tokenize', dict(content=text, add_special=False, parse_special=True))['tokens']


def template(base, msgs, gen=True):
    return post(base, '/apply-template', dict(messages=msgs, add_generation_prompt=gen))['prompt']


def build_prompt(base, plen):
    """chat-templated prompt of ~plen tokens: TASK + the first N chars of wikitext-2 test (N by bisection, deterministic)."""
    text = Path(WIKI).read_text()
    lo, hi = 0, min(len(text), plen * 8)
    while hi - lo > 64:
        mid = (lo + hi) // 2
        n = len(tok(base, template(base, [dict(role='user', content=TASK + text[:mid])])))
        lo, hi = (mid, hi) if n <= plen else (lo, mid)
    p = template(base, [dict(role='user', content=TASK + text[:lo])])
    return p, tok(base, p)


def suffix_tpl(base):
    two = template(base, [dict(role='user', content='X'), dict(role='assistant', content='ZZQQ'), dict(role='user', content='FOLLOWUPMARKER')])
    return two[two.index('ZZQQ') + 4:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', required=True); ap.add_argument('--model', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--mode', required=True, choices=['greedy', 'speed', 'chat']); ap.add_argument('--plen', type=int, default=3000)
    ap.add_argument('--seeds', default='1,2'); ap.add_argument('--npred', type=int, default=5120); ap.add_argument('--ctx', type=int, default=40960)
    ap.add_argument('--port', type=int, default=18611); ap.add_argument('--extra', default='')
    ap.add_argument('--gprompt', type=int, default=0); ap.add_argument('--env', default=''); ap.add_argument('--gn', type=int, default=256); ap.add_argument('--warm', action='store_true'); ap.add_argument('--slot', default=''); ap.add_argument('--chain', action='store_true')  # seed k continues seed k-1's conversation (no cache rollback; fill grows per seed)  # slot file name under --slot-save-path: restore if present, else save after the warm prefill  # one cached prefill, no per-seed cold prefill
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); base = f'http://127.0.0.1:{a.port}'
    stop_ev = threading.Event(); samples = []
    th = threading.Thread(target=vram_mon, args=(stop_ev, samples), daemon=True); th.start()
    with (out / 'server.log').open('w') as log:
        p, cmd = boot(a, log)
    res = dict(env=a.env, cmd=cmd, mode=a.mode, plen=a.plen, npred=a.npred, gpu=subprocess.check_output(['nvidia-smi', '--id=1', '--query-gpu=name', '--format=csv,noheader'], text=True).strip())
    err = None
    try:
        assert p is not None, 'boot failed'
        idle = max(samples) if samples else None; res['vram_after_load_mib'] = idle
        if a.mode == 'greedy':
            pr = template(base, [dict(role='user', content=GREEDY_PROMPTS[a.gprompt])]); ids = tok(base, pr)
            r = post(base, '/completion', dict(prompt=ids, n_predict=a.gn, temperature=0.0, top_k=1, seed=1, cache_prompt=False, return_tokens=True, n_probs=1))
            res.update(prompt_tokens=len(ids), tokens=r['tokens'], content=r.get('content', ''), stop_type=r.get('stop_type'), timings=r.get('timings'),
                       top1_logprob=[(c.get('top_logprobs') or [{}])[0].get('logprob') for c in r.get('completion_probabilities', [])])
        elif a.mode == 'speed':
            pr, ids = build_prompt(base, a.plen); stpl = suffix_tpl(base); res['prompt_tokens'] = len(ids)
            sdir = next((a.extra.split()[i + 1] for i, x in enumerate(a.extra.split()) if x == '--slot-save-path'), None)
            if a.slot and sdir and os.path.exists(os.path.join(sdir, a.slot)):
                t0 = time.time(); res['slot_restore'] = post(base, '/slots/0?action=restore', dict(filename=a.slot)); res['slot_restore_s'] = time.time() - t0
            t0 = time.time(); w = {'timings': None} if 'slot_restore' in res else post(base, '/completion', dict(prompt=ids[:-1] if a.chain else ids, n_predict=1, temperature=0.0, cache_prompt=a.warm))  # untimed warm-up (chain: all but the last token, so seed 1 extends the cache without a rollback)
            if a.slot and sdir and 'slot_restore' not in res:
                res['slot_save'] = post(base, '/slots/0?action=save', dict(filename=a.slot))
            res['warm_prefill'] = dict(wall_s=time.time() - t0, timings=w.get('timings'))
            rows = []; chain_toks = list(ids)
            for si, seed in enumerate([int(s) for s in a.seeds.split(',')]):
                pf = (res['warm_prefill']['timings'] or {'prompt_n': 0, 'prompt_ms': 1}) if a.warm else post(base, '/completion', dict(prompt=ids, n_predict=1, temperature=0.0, cache_prompt=False))['timings']
                toks = list(ids) if not a.chain else (chain_toks if si == 0 else chain_toks + tok(base, stpl.replace('FOLLOWUPMARKER', FOLLOWUPS[(si + 1) % len(FOLLOWUPS)])))
                fill0 = len(toks); gen_total = 0; turn = 0; turns = []
                while gen_total < a.npred and turn < 20 and len(toks) + (a.npred - gen_total) <= a.ctx - 16:
                    body = dict(prompt=toks, n_predict=a.npred - gen_total, temperature=1.0, top_k=20, top_p=0.95, min_p=0.0, seed=seed + turn,
                                repeat_penalty=1.0, return_tokens=True, cache_prompt=True, stream=False)
                    t0 = time.time(); r = post(base, '/completion', body); wall = time.time() - t0; tm = r.get('timings', {}); g = r['tokens']
                    (out / f'completion_s{seed}_t{turn}.txt').write_text(r.get('content', ''))
                    turns.append(dict(turn=turn, prompt_n=tm.get('prompt_n'), predicted_n=tm.get('predicted_n'), predicted_ms=tm.get('predicted_ms'), draft_n=tm.get('draft_n'), draft_n_accepted=tm.get('draft_n_accepted'), stop_type=r.get('stop_type'), wall_s=wall))
                    toks += g; gen_total += len(g)
                    if r.get('stop_type') != 'eos' or gen_total >= a.npred:
                        break
                    turn += 1; toks += tok(base, stpl.replace('FOLLOWUPMARKER', FOLLOWUPS[turn % len(FOLLOWUPS)]))
                if a.chain:
                    chain_toks = toks
                pn = sum(t['predicted_n'] or 0 for t in turns); pm = sum(t['predicted_ms'] or 0 for t in turns)
                row = dict(seed=seed, fill_start=fill0, prefill_n=pf.get('prompt_n'), prefill_ms=pf.get('prompt_ms'), prefill_tok_s=pf['prompt_n'] * 1000 / pf['prompt_ms'],
                           generated=gen_total, decode_tok_s=pn * 1000 / pm if pm else None, turns=turns, valid=gen_total >= a.npred,
                           draft_n=sum(t['draft_n'] or 0 for t in turns), draft_n_accepted=sum(t['draft_n_accepted'] or 0 for t in turns))
                rows.append(row); print('ROW', seed, round(row['prefill_tok_s'], 1), round(row['decode_tok_s'] or 0, 2), gen_total, flush=True)
            res['rows'] = rows
        else:
            r1 = post(base, '/v1/chat/completions', dict(messages=[dict(role='user', content='What is 17 * 23? Answer briefly.')], max_tokens=2048, temperature=0.0))
            tools = [dict(type='function', function=dict(name='get_weather', description='Get the current weather for a city',
                     parameters=dict(type='object', properties=dict(city=dict(type='string'), unit=dict(type='string', enum=['c', 'f'])), required=['city'])))]
            r2 = post(base, '/v1/chat/completions', dict(messages=[dict(role='user', content='What is the weather in Paris right now? Use the tool.')], tools=tools, max_tokens=2048, temperature=0.0))
            res.update(chat=r1, tool=r2)
            m1 = r1['choices'][0]['message']; m2 = r2['choices'][0]['message']
            res['check'] = dict(chat_has_reasoning=bool(m1.get('reasoning_content')), chat_has_content=bool(m1.get('content')),
                                chat_no_think_tags_in_content='think' not in (m1.get('content') or '').lower(),
                                tool_has_tool_calls=bool(m2.get('tool_calls')), tool_name=(m2.get('tool_calls') or [{}])[0].get('function', {}).get('name'),
                                tool_args=(m2.get('tool_calls') or [{}])[0].get('function', {}).get('arguments'), tool_finish=r2['choices'][0].get('finish_reason'))
            print('CHECK', res['check'], flush=True)
    except Exception as e:
        err = repr(e); print('FAIL', err, flush=True)
    finally:
        if p is not None: stop(p)
        stop_ev.set(); th.join()
    import re
    try:
        log = (out / 'server.log').read_text(errors='replace')
        summ = [(int(m.group(1)), int(m.group(2)), float(m.group(3))) for m in re.finditer(r'draft acceptance = [0-9.]+ \( *(\d+) accepted / *(\d+) generated\), mean len = *([0-9.]+)', log)]
        rnds = [(int(m.group(1)), float(m.group(2))) for m in re.finditer(r'w\d+: (\d+) rounds, ([0-9.]+) tok/round', log)]
        if summ:
            acc = sum(x[0] for x in summ); dr = sum(x[1] for x in summ); nr = sum(r for r, _ in rnds)
            res['spec'] = dict(requests=len(summ), drafted=dr, accepted=acc, acceptance_rate=acc / dr if dr else None,
                               rounds=nr, tok_per_round=sum(r * t for r, t in rnds) / nr if nr else None,
                               server_mean_len=[x[2] for x in summ])
    except Exception as e:
        res['spec_err'] = repr(e)
    res.update(error=err, vram_peak_mib=max(samples) if samples else None)
    (out / 'result.json').write_text(json.dumps(res, indent=1))
    print('DONE', a.mode, 'err', err, 'peak', res['vram_peak_mib'], flush=True)
    return 0 if err is None else 1


if __name__ == '__main__':
    raise SystemExit(main())
