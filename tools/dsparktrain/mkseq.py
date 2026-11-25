#!/usr/bin/env python3
"""DSPARKTRAIN: gens.jsonl -> SEQ.bin for llama-dspark-dump (int32 n_seq; per seq int32 len, int32 n_prompt, tokens)
+ seq.json (per sequence: id, split, topic, n_prompt, len, stop_type). One SEQ file per split (train/sel/rep) so the
splits are dumped to separate files. Replies shorter than MIN_GEN are dropped; sequences are cut at MAX_LEN.
usage: mkseq.py G.jsonl OUTDIR [--max-len 6144]"""
import json, struct, sys
from pathlib import Path

MIN_GEN = 32


def main():
    out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    max_len = int(sys.argv[sys.argv.index('--max-len') + 1]) if '--max-len' in sys.argv else 6144
    by = {}
    for l in open(sys.argv[1]):
        d = json.loads(l)
        if 'gen_tokens' not in d or len(d['gen_tokens']) < MIN_GEN:
            continue
        s = (d['prompt_tokens'] + d['gen_tokens'])[:max_len]
        if len(s) - len(d['prompt_tokens']) < MIN_GEN:
            continue
        by.setdefault(d['split'], []).append((s, dict(id=d['id'], split=d['split'], topic=d.get('topic'), n_prompt=len(d['prompt_tokens']),
                                                       len=len(s), stop_type=d.get('stop_type'))))
    for split, rows in sorted(by.items()):
        with open(out / f'{split}.seq', 'wb') as f:
            f.write(struct.pack('<i', len(rows)))
            for s, m in rows:
                f.write(struct.pack('<ii', len(s), m['n_prompt'])); f.write(struct.pack(f'<{len(s)}i', *s))
        (out / f'{split}.seq.json').write_text(json.dumps([m for _, m in rows]))
        print(f"{split}: {len(rows)} seqs, {sum(m['len'] for _, m in rows)} tokens, reply {sum(m['len'] - m['n_prompt'] for _, m in rows)}")


if __name__ == '__main__':
    main()
