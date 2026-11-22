#!/usr/bin/env python3
"""Extract the frozen target-side tensors the drafter reads and prove the torch Q8_0 quantizer is byte-exact.
usage: prep_weights.py TARGET_AI4.gguf TRUTH_BF16.gguf DRAFTER_Q8_0_FOLD.gguf DRAFTER_BF16.gguf OUT_DIR
writes OUT_DIR/target.pt (tok_embd + output.weight Q8_0 codes/scales, token_embd_norm), OUT_DIR/lm_truth.bf16 (raw
[V_PAD, N_EMBD] BF16 truth output.weight, also the mini truth GGUF lm_truth.gguf for ink_dspark_fold.py),
OUT_DIR/prep_check.json (quantizer byte check, fold reproduction error)."""
import json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dspark as D  # noqa: E402

tgt, truth, dq8, dbf, out = sys.argv[1:6]
os.makedirs(out, exist_ok=True)
rep = {}
tt = D.load_gguf_tensors(tgt)
o = {}
for name, key in (('token_embd.weight', 'tok_embd'), ('output.weight', 'output')):
    t = tt[name]
    assert t.tensor_type == D.Q.Q8_0, (name, t.tensor_type)
    q, d = D.q8_codes(t)
    o[key + '_q'], o[key + '_d'] = torch.from_numpy(q), torch.from_numpy(d)
    rep[name] = list(q.shape)
o['tok_norm'] = torch.from_numpy(D.gguf_f32(tt['token_embd_norm.weight']))
torch.save(o, os.path.join(out, 'target.pt'))
del tt

rt = D.gguf.GGUFReader(truth)
lm = [t for t in rt.tensors if t.name == 'output.weight'][0]
assert lm.tensor_type == D.Q.BF16
raw = np.asarray(lm.data).reshape(-1).view(np.uint16)
assert raw.size == D.V_PAD * D.N_EMBD, raw.size
raw.tofile(os.path.join(out, 'lm_truth.bf16'))
w = D.gguf.GGUFWriter(os.path.join(out, 'lm_truth.gguf'), 'inkling2')
w.add_tensor_info('output.weight', list(lm.data.shape), lm.data.dtype, lm.n_bytes, raw_dtype=D.Q.BF16)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
w.write_tensor_data(np.asarray(lm.data)); w.close()
del rt, lm, raw

# quantizer check: torch Q8_0 of the BF16 drafter == shipped Q8_0 bytes; and requant(deq(shipped)) == shipped
tq, tb = D.load_gguf_tensors(dq8), D.load_gguf_tensors(dbf)
bad_bf, bad_id, n = [], [], 0
for name in D.q8_names():
    q0, d0 = D.q8_codes(tq[name])
    x = torch.from_numpy(D.gguf_f32(tb[name]))
    q1, d1 = D.q8_0_quant(x)
    if not (np.array_equal(q1.numpy(), q0) and np.array_equal(d1.numpy().view(np.uint16), d0.view(np.uint16))):
        bad_bf.append([name, int((q1.numpy() != q0).sum())])
    q2, d2 = D.q8_0_quant(D.q8_0_deq(torch.from_numpy(q0), torch.from_numpy(d0)))
    if not (np.array_equal(q2.numpy(), q0) and np.array_equal(d2.numpy().view(np.uint16), d0.view(np.uint16))):
        bad_id.append([name, int((q2.numpy() != q0).sum())])
    n += 1
rep['q8_tensors'] = n
rep['q8_from_bf16_mismatch'] = bad_bf
rep['q8_requant_identity_mismatch'] = bad_id

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
fr = D.Frozen(out, dev)
H = torch.from_numpy(D.gguf_f32(tb[D.H_NAME])).to(dev)
b = torch.from_numpy(D.gguf_f32(tb['dspark_hidden_out.bias'])).to(dev)
P, c = fr.fold(H, b)
P0 = torch.from_numpy(D.gguf_f32(tq['dspark_hidden_out.fold'])).to(dev)
c0 = torch.from_numpy(D.gguf_f32(tq['dspark_hidden_out.fold_bias'])).to(dev)
rep['fold_shape'] = list(P0.shape)
rep['fold_max_abs_diff'] = float((P - P0).abs().max())
rep['fold_frac_f16_mismatch'] = float((P != P0).float().mean())
rep['fold_bias_max_abs_diff'] = float((c - c0).abs().max())
rep['fold_absmax'] = float(P0.abs().max())
json.dump(rep, open(os.path.join(out, 'prep_check.json'), 'w'), indent=1)
print(json.dumps(rep))
