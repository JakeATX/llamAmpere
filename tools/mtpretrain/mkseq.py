#!/usr/bin/env python3
"""MTPRETRAIN: gens.jsonl -> seq.bin (input of llama-mtp-dump-h) + seq.json (per sequence: id, split, prompt_len, len).
Sequence = prompt_tokens + gen_tokens, cut at MAX_LEN. Sequences whose reply is shorter than MIN_GEN are dropped.
usage: mkseq.py G.jsonl OUTDIR"""
import json, struct, sys
from pathlib import Path

MAX_LEN = 5120
MIN_GEN = 16


def main():
    out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    seqs, meta = [], []
    for l in open(sys.argv[1]):
        d = json.loads(l)
        if len(d['gen_tokens']) < MIN_GEN:
            continue
        s = (d['prompt_tokens'] + d['gen_tokens'])[:MAX_LEN]
        seqs.append(s)
        meta.append(dict(id=d['id'], split=d['split'], chash=d.get('chash'), topic=d['topic'], prompt_len=len(d['prompt_tokens']), len=len(s)))
    with open(out / 'seq.bin', 'wb') as f:
        f.write(struct.pack('<i', len(seqs)))
        for s in seqs:
            f.write(struct.pack('<i', len(s))); f.write(struct.pack(f'<{len(s)}i', *s))
    (out / 'seq.json').write_text(json.dumps(meta))
    (out / 'tokens.json').write_text(json.dumps(seqs))
    n_tr = sum(m['len'] - m['prompt_len'] for m in meta if m['split'] == 'train')
    n_ho = sum(m['len'] - m['prompt_len'] for m in meta if m['split'] == 'heldout')
    print(f"{len(seqs)} sequences, {sum(m['len'] for m in meta)} tokens, reply tokens train {n_tr} heldout {n_ho}")


if __name__ == '__main__':
    main()
