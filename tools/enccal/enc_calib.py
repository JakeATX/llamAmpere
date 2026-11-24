#!/usr/bin/env python3
"""ENCCALIB: self-generated, token-weighted calibration for the encoder's activation statistics.

The calibration text is the model's own answers (sampled from the reference model through a running llama-server),
so the statistics see the activations the model actually produces when it talks, not someone else's prose. Each
token gets a weight: digits and entity / proper-noun tokens 1.0 (they carry the facts the fact-retention gate
checks), everything else --floor (default 0.1); prompt tokens get --prompt-weight (default: the floor).
The weights feed llama-imatrix through LLAMA_IMATRIX_TOKEN_WEIGHTS (with LLAMA_IMATRIX_INPUT_IDS = the ids file),
both opt-in; with them unset llama-imatrix is unchanged.

Subcommands
  gen      URL PROMPTS.jsonl OUT_PREFIX [--n-predict 1024] [--parallel 4] [--temp 1.0] [--seed 1] [--reasoning medium]
           PROMPTS.jsonl: one {"id":..., "prompt":...} per line. Writes OUT_PREFIX.i32 (all samples' prompt+answer
           ids, concatenated), OUT_PREFIX.u8 (1 = generated token), OUT_PREFIX.json (per-sample meta). Resumable:
           finished samples are kept in OUT_PREFIX.parts/.
  weights  MODEL.gguf OUT_PREFIX [--floor 0.1] [--prompt-weight F] [--key-weight 1.0]
           Reads OUT_PREFIX.i32/.u8 + the model's vocab, writes OUT_PREFIX.w.f32 (one float per token) and prints
           the key-token share.
  selftest
"""
import argparse, json, os, re, sys, threading, urllib.request

import numpy as np


# ------------------------------------------------------------------------------------------------------------ gen
def _post(base, path, body, t=3000):
    r = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=t).read().decode())


def _one(a, item, k):
    pdir = a.out + '.parts'
    fp = os.path.join(pdir, f'{k:05d}')
    if os.path.exists(fp + '.json'):
        return
    msgs = [{'role': 'user', 'content': item['prompt']}]
    kw = {'reasoning_effort': a.reasoning} if a.reasoning else {}
    rendered = _post(a.url, '/apply-template', {'messages': msgs, 'chat_template_kwargs': kw})['prompt']
    pids = _post(a.url, '/tokenize', {'content': rendered, 'add_special': False, 'parse_special': True})['tokens']
    res = _post(a.url, '/completion', {'prompt': pids, 'n_predict': a.n_predict, 'temperature': a.temp, 'top_k': 0,
                                      'top_p': 1.0, 'min_p': 0.0, 'seed': a.seed + k, 'cache_prompt': False,
                                      'return_tokens': True, 'stream': False})
    g = res['tokens']
    ids = np.array(pids + g, np.int32)
    m = np.zeros(len(ids), np.uint8); m[len(pids):] = 1
    ids.tofile(fp + '.i32'); m.tofile(fp + '.u8')
    json.dump(dict(id=item.get('id', k), prompt_tokens=len(pids), generated=len(g), stop_type=res.get('stop_type')),
              open(fp + '.json', 'w'))


def cmd_gen(a):
    items = [json.loads(l) for l in open(a.prompts) if l.strip()]
    os.makedirs(a.out + '.parts', exist_ok=True)
    todo = list(enumerate(items)); lock = threading.Lock(); err = []
    def worker():
        while True:
            with lock:
                if not todo or err:
                    return
                k, it = todo.pop(0)
            try:
                _one(a, it, k)
            except Exception as e:  # keep going on the other samples; report at the end
                with lock:
                    err.append((k, repr(e)))
    th = [threading.Thread(target=worker) for _ in range(a.parallel)]
    [t.start() for t in th]; [t.join() for t in th]
    pdir = a.out + '.parts'
    ks = sorted(int(f[:-5]) for f in os.listdir(pdir) if f.endswith('.json'))
    ids = np.concatenate([np.fromfile(os.path.join(pdir, f'{k:05d}.i32'), np.int32) for k in ks])
    msk = np.concatenate([np.fromfile(os.path.join(pdir, f'{k:05d}.u8'), np.uint8) for k in ks])
    meta = [json.load(open(os.path.join(pdir, f'{k:05d}.json'))) for k in ks]
    ids.tofile(a.out + '.i32'); msk.tofile(a.out + '.u8')
    json.dump(dict(samples=meta, tokens=int(ids.size), generated=int(msk.sum()), temp=a.temp, seed=a.seed,
                   n_predict=a.n_predict, reasoning=a.reasoning, errors=err), open(a.out + '.json', 'w'), indent=1)
    print(f'GEN samples={len(ks)}/{len(items)} tokens={ids.size} generated={int(msk.sum())} errors={len(err)}')
    if err:
        sys.exit(1)


# -------------------------------------------------------------------------------------------------------- weights
def _byte_decoder():
    # inverse of the byte-level BPE byte->unicode table
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('\xa1'), ord('\xac') + 1)) + list(range(ord('\xae'), ord('\xff') + 1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def load_pieces(model_path):
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.environ.get('ENCCAL_GGUF_PY'), os.path.join(here, '..', '..', 'gguf-py')):
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    import gguf
    r = gguf.GGUFReader(model_path)
    f = r.fields['tokenizer.ggml.tokens']
    toks = [bytes(f.parts[i]).decode('utf-8', 'surrogateescape') for i in f.data]
    tt = r.fields.get('tokenizer.ggml.token_type')
    types = [int(tt.parts[i][0]) for i in tt.data] if tt else [1] * len(toks)
    mf = r.fields.get('tokenizer.ggml.model')
    model = bytes(mf.parts[mf.data[0]]).decode() if mf else 'gpt2'
    return pieces_from_vocab(toks, types, model)


def pieces_from_vocab(toks, types, model):
    dec = _byte_decoder()
    out = []
    for s, ty in zip(toks, types):
        if ty != 1:                      # control / unused / user-defined special: no text
            out.append(b'')
        elif model == 'gpt2':
            out.append(bytes(dec.get(ch, 0x3f) for ch in s))
        else:                            # sentencepiece style
            out.append(s.replace('▁', ' ').encode('utf-8', 'surrogateescape'))
    return out


_DIGIT = re.compile(rb'[0-9]')
_WORD = re.compile(rb"[A-Z][A-Za-z0-9'\-]*[A-Za-z0-9]|[A-Z]")
_SENT = re.compile(rb'(^|[.!?]["\')\]]?\s+|\n\s*|\n\s*([-*#>]+|\d+[.)])\s*)$')   # after ':' or '(' a capital is not a sentence start
_COMMON = {b'I', b'A', b'The', b'This', b'That', b'It', b'In', b'If', b'For', b'And', b'But', b'So', b'We', b'You',
           b'Yes', b'No', b'Here', b'There', b'What', b'When', b'Why', b'How', b'Let', b'Note', b'Step', b'Also', b'Then',
           b'Now', b'First', b'Second', b'Finally', b'However', b'OK', b'Okay'}


def key_spans(text):
    """Byte mask over text: 1 = digit or entity / proper-noun character."""
    m = np.zeros(len(text), np.uint8)
    for x in _DIGIT.finditer(text):
        m[x.start()] = 1
    words = list(_WORD.finditer(text))
    for i, x in enumerate(words):
        w = x.group(0); s = x.start()
        if s > 0 and (text[s - 1:s].isalnum() or text[s - 1:s] in (b'_',)):
            continue                     # inside an identifier like fooBar
        sent_start = bool(_SENT.search(text[max(0, s - 8):s])) or s == 0
        acronym = len(w) >= 2 and w.isupper()
        nxt = words[i + 1] if i + 1 < len(words) else None
        multi = nxt is not None and nxt.start() - x.end() == 1 and text[x.end():nxt.start()] == b' '
        if acronym or (not sent_start and w not in _COMMON) or (sent_start and multi and w not in _COMMON):
            m[s:x.end()] = 1
    return m


def token_weights(ids, gen_mask, pieces, floor, key_w, prompt_w):
    w = np.full(ids.size, floor, np.float32)
    w[gen_mask == 0] = prompt_w
    # work per contiguous generated region so prompt text never leaks into answer spans
    edges = np.flatnonzero(np.diff(np.concatenate([[0], gen_mask.astype(np.int8), [0]])))
    n_key = 0
    for a, b in zip(edges[::2], edges[1::2]):
        bs = [pieces[t] if 0 <= t < len(pieces) else b'' for t in ids[a:b]]
        off = np.cumsum([0] + [len(x) for x in bs])
        km = key_spans(b''.join(bs))
        for j in range(b - a):
            if off[j + 1] > off[j] and km[off[j]:off[j + 1]].any():
                w[a + j] = key_w; n_key += 1
    return w, n_key


def cmd_weights(a):
    ids = np.fromfile(a.prefix + '.i32', np.int32); msk = np.fromfile(a.prefix + '.u8', np.uint8)
    assert ids.size == msk.size, 'ids/mask length mismatch'
    pieces = load_pieces(a.model)
    pw = a.floor if a.prompt_weight is None else a.prompt_weight
    w, nk = token_weights(ids, msk, pieces, a.floor, a.key_weight, pw)
    w.tofile(a.prefix + '.w.f32')
    ng = int(msk.sum())
    print(f'WEIGHTS tokens={ids.size} generated={ng} key={nk} ({100.0 * nk / max(ng, 1):.1f}% of generated) '
          f'floor={a.floor} prompt={pw} key={a.key_weight} mean={w.mean():.4f} -> {a.prefix}.w.f32')


def cmd_selftest(_):
    t = b'In 1969, Neil Armstrong walked on the Moon. The NASA budget was $25 billion. it was big.\nParis is nice.'
    m = key_spans(t)
    got = {t[x.start():x.end()] for x in re.finditer(rb'\S+', t) if m[x.start():x.end()].any()}
    for want in (b'1969,', b'Neil', b'Armstrong', b'Moon.', b'NASA', b'$25'):
        assert want in got, (want, got)
    t2 = b'Crew: Young, Duke (Jul 1972). Note: done.'
    m2 = key_spans(t2); got2 = {t2[x.start():x.end()] for x in re.finditer(rb'\S+', t2) if m2[x.start():x.end()].any()}
    for want in (b'Young,', b'Duke', b'(Jul', b'1972).'):
        assert want in got2, (want, got2)
    for bad in (b'In', b'The', b'walked', b'it', b'big.\nParis'):
        assert bad not in got, (bad, got)
    # Paris starts a sentence alone: not weighted (heuristic), documented limitation
    vocab = ['Ġthe', 'Ġ19', '69', 'ĠNeil', 'Ġwalked', '<|end|>']
    pc = pieces_from_vocab(vocab, [1, 1, 1, 1, 1, 3], 'gpt2')
    ids = np.array([0, 1, 2, 3, 4, 5, 0], np.int32); msk = np.array([0, 1, 1, 1, 1, 1, 1], np.uint8)
    w, nk = token_weights(ids, msk, pc, 0.1, 1.0, 0.05)
    assert w.tolist() == [np.float32(0.05), 1.0, 1.0, 1.0, np.float32(0.1), np.float32(0.1), np.float32(0.1)], w.tolist()
    print('SELFTEST OK')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('gen'); p.add_argument('url'); p.add_argument('prompts'); p.add_argument('out')
    p.add_argument('--n-predict', type=int, default=1024); p.add_argument('--parallel', type=int, default=4)
    p.add_argument('--temp', type=float, default=1.0); p.add_argument('--seed', type=int, default=1)
    p.add_argument('--reasoning', default='medium')
    p = sp.add_parser('weights'); p.add_argument('model'); p.add_argument('prefix')
    p.add_argument('--floor', type=float, default=0.1); p.add_argument('--prompt-weight', type=float)
    p.add_argument('--key-weight', type=float, default=1.0)
    sp.add_parser('selftest')
    a = ap.parse_args()
    dict(gen=cmd_gen, weights=cmd_weights, selftest=cmd_selftest)[a.cmd](a)


if __name__ == '__main__':
    main()
