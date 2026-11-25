#!/usr/bin/env python3
"""DRAFT-WORDWT: whole-word token mask for the drafter QAT loss (train.py --word-wt).
WORD = a normal token (token_type 1) whose byte-decoded piece starts with a space followed by a letter (" the", " Paris");
the rule of the acceptance-by-class analysis that found WORD drafts accepted at 0.861 x the mean rate (Inkling DSpark,
100K, 39% of drafted positions). Output: bool .npy [n_vocab] (True = WORD) + a one-line summary.
usage: mkwordmask.py TARGET.gguf OUT.npy"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'gguf-py'))
from gguf import GGUFReader  # noqa: E402


def byte_decoder():
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def is_word(piece):
    if len(piece) < 2 or piece[0] != ' ':
        return False
    t = piece.strip()
    return bool(t) and t[0].isalpha()


def main():
    r = GGUFReader(sys.argv[1], 'r')
    f = r.fields['tokenizer.ggml.tokens']
    toks = [bytes(f.parts[i]).decode('utf-8', 'replace') for i in f.data]
    tt = r.fields.get('tokenizer.ggml.token_type')
    types = [int(tt.parts[i][0]) for i in tt.data] if tt else [1] * len(toks)
    U = byte_decoder()
    m = np.zeros(len(toks), bool)
    for i, t in enumerate(toks):
        if types[i] != 1:
            continue
        try:
            s = bytes(U[c] for c in t).decode('utf-8', 'replace')
        except KeyError:
            s = t
        m[i] = is_word(s)
    np.save(sys.argv[2], m)
    print(f'WORDMASK {sys.argv[2]}: n_vocab {len(m)}, WORD {int(m.sum())} ({100 * m.mean():.1f}% of the vocab)')


if __name__ == '__main__':
    main()
