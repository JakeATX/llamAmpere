#!/usr/bin/env python3
"""DSPARKTRAIN parity: torch drafter vs the llama.cpp drafter (llama-dspark-dump --ds-parity) on the same features.
A (argmax walk): max |logit diff| per slot over the whole vocab, argmax agreement, first-slot top-20 set overlap.
B (sampled walk, the server default): with the dumped Gumbel noise, drawn-id agreement and candidate-set agreement.
Weights: the drafter GGUF's own Q8_0 codes (dequantized, == the torch fake-quant of the masters) and its shipped fold;
--refold also reports the torch-recomputed fold (what training/export use).
usage: parity.py --par PREFIX --w0 W0DIR --drafter Q8_0-fold.gguf --bf16 BF16.gguf [--refold] [--out parity.json]"""
import argparse, json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dspark as D  # noqa: E402


def load_par(prefix):
    raw = np.fromfile(prefix + '.par.bin', dtype=np.int32)
    nA, walk, K, nv, ne, rank, block, L = (int(x) for x in raw[:8])
    o = 8
    rows = []
    f = raw.view(np.float32)
    for _ in range(nA):
        r = dict(pos=int(raw[o])); o += 1
        r['noise'] = f[o:o + walk * K].reshape(walk, K); o += walk * K
        r['A'] = f[o:o + walk * nv].reshape(walk, nv); o += walk * nv
        r['Aid'] = raw[o:o + walk].copy(); o += walk
        r['Bid'] = raw[o:o + walk].copy(); o += walk
        r['Bc'] = raw[o:o + walk * K].reshape(walk, K).copy(); o += walk * K
        r['Bl'] = f[o:o + walk * K].reshape(walk, K).copy(); o += walk * K
        r['ctx'] = f[o:o + ne].copy(); o += ne
        rows.append(r)
    assert o == raw.size, (o, raw.size)
    feat = np.fromfile(prefix + '.par.feat', dtype=np.float32).reshape(L, 4, ne)
    return dict(nA=nA, walk=walk, K=K, nv=nv, L=L), rows, feat


def run(W, fr, hdr, rows, feat, toks, dev):
    walk, K = hdr['walk'], hdr['K']
    featt = torch.from_numpy(feat).to(dev)
    amax = max(r['pos'] for r in rows)
    ctx = D.context_layers(W, featt[:amax], torch.arange(amax, device=dev))
    res = dict(maxdiff=[0.0] * walk, meandiff=[0.0] * walk, top20_maxdiff=[0.0] * walk, argmax_agree=[0] * walk, B_id_agree=[0] * walk, B_cand_agree=[0] * walk, B_logit_maxdiff=0.0,
               f_rnn_maxdiff=0.0, n=len(rows))
    for r in rows:
        a = torch.tensor([r['pos']], device=dev)
        res['f_rnn_maxdiff'] = max(res['f_rnn_maxdiff'], float((ctx[1][r['pos'] - 1] - torch.from_numpy(r['ctx']).to(dev)).abs().max()))
        cur = D.block_forward(W, fr, ctx[0], a, toks[a], 0, torch.float32)
        f_ctx = ctx[1][a - 1]
        # A: argmax walk (the graph's own first-maximum prev)
        state = torch.zeros(1, D.RANK, device=dev)
        prev = toks[a]
        for i in range(1, walk + 1):
            state, lat = D.markov_step(W, prev, state, cur[:, i], f_ctx)
            l = D.slot_logits(W, fr, cur[:, i], lat)[0]
            ref = torch.from_numpy(r['A'][i - 1][:D.V_UNPAD]).to(dev)
            res['maxdiff'][i - 1] = max(res['maxdiff'][i - 1], float((l - ref).abs().max()))
            res['meandiff'][i - 1] += float((l - ref).abs().mean()) / len(rows)
            top = torch.topk(ref, 20).indices
            res['top20_maxdiff'][i - 1] = max(res['top20_maxdiff'][i - 1], float((l[top] - ref[top]).abs().max()))
            am = int(torch.argmax(l))
            res['argmax_agree'][i - 1] += int(am == int(r['Aid'][i - 1]))
            prev = torch.tensor([int(r['Aid'][i - 1])], device=dev)   # follow the engine's chain so later slots compare like for like
        # B: sampled walk with the same Gumbel noise
        state = torch.zeros(1, D.RANK, device=dev)
        prev = toks[a]
        for i in range(1, walk + 1):
            state, lat = D.markov_step(W, prev, state, cur[:, i], f_ctx)
            l = D.slot_logits(W, fr, cur[:, i], lat)[0]
            vals, cand = torch.topk(l, K)
            g = torch.from_numpy(r['noise'][i - 1]).to(dev)
            # ggml_top_k returns the K ids unordered; noise k belongs to the engine's k-th candidate, so draw over the
            # engine's candidate order with our logits (same law; equal ids when the logits agree)
            bc = torch.from_numpy(r['Bc'][i - 1].astype(np.int64)).to(dev)
            pick = int(bc[int(torch.argmax(l[bc] / 1.0 + g))])
            res['B_id_agree'][i - 1] += int(pick == int(r['Bid'][i - 1]))
            res['B_cand_agree'][i - 1] += int(set(cand.tolist()) == set(r['Bc'][i - 1].tolist()))
            res['B_logit_maxdiff'] = max(res['B_logit_maxdiff'], float((l[bc] - torch.from_numpy(r['Bl'][i - 1]).to(dev)).abs().max()))
            prev = torch.tensor([int(r['Bid'][i - 1])], device=dev)
    for k in ('argmax_agree', 'B_id_agree', 'B_cand_agree'):
        res[k] = [x / len(rows) for x in res[k]]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--par', required=True); ap.add_argument('--w0', required=True)
    ap.add_argument('--drafter', required=True); ap.add_argument('--bf16', required=True)
    ap.add_argument('--seq', required=True, help='SEQ.bin the parity dump read (first sequence)')
    ap.add_argument('--refold', action='store_true'); ap.add_argument('--out')
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    hdr, rows, feat = load_par(a.par)
    raw = np.fromfile(a.seq, dtype=np.int32)
    toks = torch.from_numpy(raw[3:3 + int(raw[1])].astype(np.int64)).to(dev)
    fr = D.Frozen(a.w0, dev)
    m = D.load_drafter_masters(a.drafter, a.bf16)
    t = D.load_gguf_tensors(a.drafter)
    fold = (torch.from_numpy(D.gguf_f32(t['dspark_hidden_out.fold'])).to(dev), torch.from_numpy(D.gguf_f32(t['dspark_hidden_out.fold_bias'])).to(dev))
    m = {k: v.to(dev) for k, v in m.items()}
    out = dict(header=hdr)
    for aq in (True, False):   # with / without the engine activation-rounding emulation
        D.ACT_Q = aq
        out[f'shipped_fold_actq{int(aq)}'] = run(D.effective(m, fr, fold), fr, hdr, rows, feat, toks, dev)
        if a.refold:
            out[f'torch_fold_actq{int(aq)}'] = run(D.effective(m, fr), fr, hdr, rows, feat, toks, dev)
    D.ACT_Q = True
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
