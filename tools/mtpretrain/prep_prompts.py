#!/usr/bin/env python3
"""MTPRETRAIN: build the prompt set for on-policy MTP head data.
Sources (all already on Taichi; none is a speed-gate trajectory):
  heal  = heal-r9-prompts_train.jsonl (token ids, chat template + '<think>\\n'), stratified by topic (math capped)
  enc   = encbeat-l1r2 corpus/prompts.jsonl (token ids; rag / agentic / coding with documents)
  g2p   = gr2plan-heal-r416 data/prompts.jsonl (plain text; gen.py applies the server's chat template)
Content hash chash = sha256 of the last 256 prompt tokens (token prompts) or the last 600 characters (text prompts);
duplicates by chash are dropped. The held-out split (8%) is chosen by a hash of the prompt's first 1,000 tokens / 2,000
characters, so near-identical prompts and prompts sharing a long document cannot straddle train and held-out.
Prompts over MAX_PROMPT tokens are dropped (sequence = prompt + reply <= 5,120 tokens).
usage: prep_prompts.py OUT.jsonl"""
import hashlib, json, random, sys
from collections import Counter

HEAL = '/home/jakek/opus12-s34/in/heal-r9-prompts_train.jsonl'
ENC = '/home/jakek/encbeat-l1r2/corpus/prompts.jsonl'
G2P = '/home/jakek/gr2plan-heal-r416/data/prompts.jsonl'
HEAL_TAKE = {'mathematics': 260, 'agentic-workflows': 300, 'software-engineering': 220, 'natural-sciences': 120,
             'computing-and-engineering': 60, 'general-reasoning': 40, 'instruction-following': 4}
MAX_PROMPT = 3072
HELDOUT_PCT = 8


def heldout(ch):
    return int(hashlib.sha256(ch.encode()).hexdigest()[:8], 16) % 100 < HELDOUT_PCT


def chash(d):
    key = json.dumps(d['prompt_tokens'][-256:]) if 'prompt_tokens' in d else ' '.join(d['text'].split())[-600:]
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def main():
    out = []
    rng = random.Random(20261121)
    heal = [json.loads(l) for l in open(HEAL)]
    by = {}
    for d in heal:
        if len(d['prompt_tokens']) <= MAX_PROMPT:
            by.setdefault(d['topic'], []).append(d)
    for topic, n in HEAL_TAKE.items():
        rows = sorted(by.get(topic, []), key=lambda d: d['trace_id'])
        rng.shuffle(rows)
        for d in rows[:n]:
            out.append(dict(id='heal:' + d['trace_id'], source='heal/' + d['source'], topic=topic, prompt_tokens=d['prompt_tokens']))
    for d in (json.loads(l) for l in open(ENC)):
        if len(d['prompt']) <= MAX_PROMPT:
            out.append(dict(id='enc:' + d['cid'], source='encbeat/' + d['source'], topic=d['domain'], prompt_tokens=d['prompt']))
    for d in (json.loads(l) for l in open(G2P)):
        out.append(dict(id='g2p:' + d['id'], source='gr2plan/' + d['source'], topic=d['id'].rsplit('-', 2)[0], text=d['text']))
    ids = [d['id'] for d in out]
    assert len(ids) == len(set(ids)), 'duplicate prompt ids'
    seen, ded = set(), []
    for d in out:
        d['chash'] = chash(d)
        if d['chash'] not in seen:
            seen.add(d['chash']); ded.append(d)
    print(f'dropped {len(out) - len(ded)} duplicate prompts (by content hash)')
    out = ded
    for d in out:
        # split on the shared-document group (first 1,000 tokens / 2,000 characters), so prompts that reuse a long
        # document with a different question stay on one side
        g = json.dumps(d['prompt_tokens'][:1000]) if 'prompt_tokens' in d else ' '.join(d['text'].split())[:2000]
        d['split'] = 'heldout' if heldout(g) else 'train'
    rng.shuffle(out)  # gen.py works in file order and may stop at a token budget, so the order is the mix
    with open(sys.argv[1], 'w') as f:
        for d in out:
            f.write(json.dumps(d) + '\n')
    print(len(out), Counter(d['split'] for d in out), Counter(d['topic'] for d in out).most_common())


if __name__ == '__main__':
    main()
