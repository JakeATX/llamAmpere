#!/usr/bin/env python3
"""DSPARKTRAIN: prompt set for on-policy DSpark drafter data on Inkling 2 Mini.
Sources (none is a speed/quality gate prompt):
  heal = heal-r9-prompts_train.jsonl, enc = encbeat-l1r2 corpus prompts (both Qwen token ids: decoded with the Qwen
  tokenizer, split into <|im_start|>role ... <|im_end|> messages, the trailing assistant opener dropped; gen.py
  re-templates them with the Inkling server's /apply-template), g2p = gr2plan-heal-r416 prompts (plain text).
Leak guard: a prompt is dropped if any 64-character window (whitespace-normalised) of it occurs in the gate texts:
  inkbench TASK / GREEDY_PROMPTS / FOLLOWUPS, wikitext-2 test (speed-bench filler), the inkqual/inkqual2 panels.
Splits by a hash of the prompt's document group (first 2,000 normalised characters), so prompts sharing a document
never straddle: train / sel (held-out, checkpoint selection) / rep (held-out, report only). 6% + 6%.
usage: prep_prompts.py OUT.jsonl [--max-chars 14000]"""
import hashlib, json, os, random, re, sys
from collections import Counter

HEAL = '/home/jakek/opus12-s34/in/heal-r9-prompts_train.jsonl'
ENC = '/home/jakek/encbeat-l1r2/corpus/prompts.jsonl'
G2P = '/home/jakek/gr2plan-heal-r416/data/prompts.jsonl'
QWEN_TOK = '/home/jakek/qwen38-bench/models/qwen38-27b-bf16/tokenizer.json'
WIKI = '/mnt/data/inkflags/data/wiki.test.raw'
INKBENCH = '/mnt/data/inkflags/h/inkbench.py'
PANELS = ['/mnt/data/inkqual/panel/panel.json', '/mnt/data/inkqual2/panel/cands.json', '/mnt/data/inkqual2/panel/cands-popqa.json']
HEAL_TAKE = {'mathematics': 700, 'agentic-workflows': 900, 'software-engineering': 900, 'natural-sciences': 400,
             'computing-and-engineering': 300, 'general-reasoning': 300, 'instruction-following': 100}
W = 64
SEL_PCT, REP_PCT = 6, 6


def norm(s):
    return ' '.join(s.split())


def gate_shingles():
    texts = []
    ns = {}
    exec(compile(re.search(r'FOLLOWUPS = .*?\n(?=def )', open(INKBENCH).read(), re.S).group(0), INKBENCH, 'exec'), ns)
    texts += ns['FOLLOWUPS'] + [ns['TASK']] + ns['GREEDY_PROMPTS']
    if os.path.exists(WIKI):
        texts.append(open(WIKI, errors='replace').read())
    for p in PANELS:
        if os.path.exists(p):
            texts += [json.dumps(x) if not isinstance(x, str) else x for x in _walk_strings(json.load(open(p)))]
    sh = set()
    for t in texts:
        t = norm(t)
        for i in range(0, max(1, len(t) - W + 1), 16):
            sh.add(hash(t[i:i + W]))
    return sh


def _walk_strings(o):
    if isinstance(o, str):
        if len(o) >= W:
            yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from _walk_strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk_strings(v)


def leaks(text, sh):
    t = norm(text)
    return any(hash(t[i:i + W]) in sh for i in range(0, max(1, len(t) - W + 1)))


def qwen_messages(tok, ids):
    s = tok.decode(ids, skip_special_tokens=False)
    msgs = []
    for seg in s.split('<|im_start|>')[1:]:
        seg = seg.split('<|im_end|>')[0]
        role, _, body = seg.partition('\n')
        body = re.sub(r'<think>.*?</think>\s*', '', body, flags=re.S)
        msgs.append(dict(role=role.strip(), content=body))
    while msgs and msgs[-1]['role'] == 'assistant':
        msgs.pop()   # generation opener ('<think>\n')
    return msgs


def main():
    from tokenizers import Tokenizer
    outp = sys.argv[1]
    max_chars = int(sys.argv[sys.argv.index('--max-chars') + 1]) if '--max-chars' in sys.argv else 14000
    tok = Tokenizer.from_file(QWEN_TOK)
    rng = random.Random(20261122)
    sh = gate_shingles()
    out = []
    by = {}
    for l in open(HEAL):
        d = json.loads(l)
        by.setdefault(d['topic'], []).append(d)
    for topic, n in HEAL_TAKE.items():
        rows = sorted(by.get(topic, []), key=lambda d: d['trace_id'])
        rng.shuffle(rows)
        for d in rows[:n]:
            out.append(dict(id='heal:' + d['trace_id'], source='heal/' + d['source'], topic=topic,
                            messages=qwen_messages(tok, d['prompt_tokens'])))
    for l in open(ENC):
        d = json.loads(l)
        out.append(dict(id='enc:' + d['cid'], source='encbeat/' + d['source'], topic=d['domain'],
                        messages=qwen_messages(tok, d['prompt'])))
    for l in open(G2P):
        d = json.loads(l)
        out.append(dict(id='g2p:' + d['id'], source='gr2plan/' + d['source'], topic=d['id'].rsplit('-', 2)[0],
                        messages=[dict(role='user', content=d['text'])]))
    assert len({d['id'] for d in out}) == len(out), 'duplicate ids'
    kept, seen, drop = [], set(), Counter()
    for d in out:
        full = '\n'.join(m['content'] for m in d['messages'])
        if not d['messages'] or d['messages'][-1]['role'] != 'user':
            drop['no_user_last'] += 1; continue
        if len(full) > max_chars:
            drop['too_long'] += 1; continue
        ch = hashlib.sha256(norm(full)[-600:].encode()).hexdigest()[:16]
        if ch in seen:
            drop['dup'] += 1; continue
        if leaks(full, sh):
            drop['gate_overlap'] += 1; continue
        seen.add(ch)
        g = int(hashlib.sha256(norm(full)[:2000].encode()).hexdigest()[:8], 16) % 100
        d['split'] = 'sel' if g < SEL_PCT else 'rep' if g < SEL_PCT + REP_PCT else 'train'
        d['chash'] = ch
        kept.append(d)
    rng.shuffle(kept)  # gen.py works in file order and may stop at a budget: the order is the mix
    with open(outp, 'w') as f:
        for d in kept:
            f.write(json.dumps(d) + '\n')
    print(len(kept), dict(drop), dict(Counter(d['split'] for d in kept)), Counter(d['topic'] for d in kept).most_common(12))


if __name__ == '__main__':
    main()
