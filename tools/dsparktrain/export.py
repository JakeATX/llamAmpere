#!/usr/bin/env python3
"""DSPARKTRAIN export: fp32 masters (train.py best.pt / last.pt) -> drafter GGUFs with the shipped file's tensor order,
types and every KV, then the fold via /mnt/data/inkflags/h/ink_dspark_fold.py (or --fold-script).
  OUT-Q8_0.gguf      : Q8_0 tensors = exact gguf Q8_0 of the masters (the training forward), F32 tensors as is
  OUT-BF16.gguf      : BF16 companion (bf16 of the masters; the fold reads H and b from it, as for the shipped file)
  OUT-Q8_0-fold.gguf : + dspark_hidden_out.fold / fold_bias from the mini truth lm head (W0/lm_truth.gguf)
Round-trip check: re-reads the fold file and compares every tensor with the torch forward weights (Q8_0 codes must be
identical; fold within one f16 ulp). Writes OUT-export.json.
usage: export.py CKPT.pt Q8_0.gguf BF16.gguf W0DIR OUT_PREFIX [--fold-script PATH] [--python PY]"""
import json, os, subprocess, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dspark as D  # noqa: E402

FOLD = '/mnt/data/inkflags/h/ink_dspark_fold.py'


def copy_kv(r, w):
    G = D.gguf
    w.kv_data[0].clear()
    for f in r.fields.values():
        if f.name.startswith('GGUF.'):
            continue
        vt = f.types[0]
        if vt == G.GGUFValueType.ARRAY:
            st = f.types[-1]
            vals = [bytes(f.parts[i]).decode('utf-8', 'surrogateescape') if st == G.GGUFValueType.STRING else f.parts[i].tolist()[0] for i in f.data]
            w.add_key_value(f.name, vals, vt, sub_type=st)
        elif vt == G.GGUFValueType.STRING:
            w.add_key_value(f.name, bytes(f.parts[f.data[0]]).decode('utf-8', 'surrogateescape'), vt)
        else:
            w.add_key_value(f.name, f.parts[f.data[0]].tolist()[0], vt)


def q8_bytes(x):
    q, d = D.q8_0_quant(x)
    r, c = q.shape
    out = np.empty((r, c // 32, 34), np.uint8)
    out[:, :, :2] = d.numpy().view(np.uint8).reshape(r, c // 32, 2)
    out[:, :, 2:] = q.numpy().view(np.uint8).reshape(r, c // 32, 32)
    return out.reshape(r, c // 32 * 34)


def bf16_bytes(x):
    return x.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


def write_like(src, dst, M):
    G = D.gguf
    r = G.GGUFReader(src)
    w = G.GGUFWriter(dst, 'x')
    copy_kv(r, w)
    datas = []
    for t in r.tensors:
        if t.name in M:
            x = M[t.name].float().cpu()
            if t.tensor_type == D.Q.Q8_0:
                a = q8_bytes(x.reshape(x.shape[0] if x.dim() > 1 else 1, -1))
            elif t.tensor_type == D.Q.BF16:
                a = bf16_bytes(x)
            elif t.tensor_type == D.Q.F32:
                a = x.numpy().astype(np.float32)
            else:
                raise SystemExit(f'unexpected type {t.tensor_type} for {t.name}')
            td = np.asarray(t.data)
            a = np.ascontiguousarray(a).reshape(-1).view(np.uint8).view(td.dtype).reshape(td.shape)
            assert a.nbytes == t.n_bytes, (t.name, a.nbytes, t.n_bytes)
        else:
            a = np.asarray(t.data)
            print('copy unchanged', t.name)
        w.add_tensor_info(t.name, list(a.shape), a.dtype, a.nbytes, raw_dtype=t.tensor_type)
        datas.append(a)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for a in datas:
        w.write_tensor_data(a)
    w.close()


def main():
    ck, q8, bf, w0, out = sys.argv[1:6]
    fold_script = sys.argv[sys.argv.index('--fold-script') + 1] if '--fold-script' in sys.argv else FOLD
    py = sys.argv[sys.argv.index('--python') + 1] if '--python' in sys.argv else sys.executable
    M = torch.load(ck, map_location='cpu', weights_only=True)
    write_like(q8, out + '-Q8_0.gguf', M)
    write_like(bf, out + '-BF16.gguf', M)
    env = dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(D.__file__), '..', '..', 'gguf-py'))
    subprocess.run([py, fold_script, out + '-Q8_0.gguf', out + '-BF16.gguf', os.path.join(w0, 'lm_truth.gguf'), out + '-Q8_0-fold.gguf'],
                   check=True, env=env)
    # round trip
    rep = dict(ckpt=ck, files=[out + s for s in ('-Q8_0.gguf', '-BF16.gguf', '-Q8_0-fold.gguf')])
    T = D.load_gguf_tensors(out + '-Q8_0-fold.gguf')
    ship = D.load_gguf_tensors(q8.replace('-Q8_0.gguf', '-Q8_0-fold.gguf')) if os.path.exists(q8.replace('-Q8_0.gguf', '-Q8_0-fold.gguf')) else None
    if ship is not None:
        rep['same_names_types_order'] = [(t.name, t.tensor_type) for t in T.values()] == [(t.name, t.tensor_type) for t in ship.values()]
        rep['same_shapes'] = all(list(T[n].shape) == list(ship[n].shape) for n in T)
    bad, changed = [], 0
    for n in D.q8_names() + [D.H_NAME]:
        q0, d0 = D.q8_codes(T[n])
        q1, d1 = D.q8_0_quant(M[n].float())
        if not (np.array_equal(q0, q1.numpy()) and np.array_equal(d0.view(np.uint16), d1.numpy().view(np.uint16))):
            bad.append(n)
        if ship is not None:
            changed += int((D.q8_codes(ship[n])[0] != q0).sum())
    for n in D.f32_names():
        if not np.array_equal(D.gguf_f32(T[n]), M[n].float().numpy()):
            bad.append(n)
    rep['tensor_mismatch'] = bad
    rep['q8_codes_changed_vs_shipped'] = changed
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    fr = D.Frozen(w0, dev)
    P, c = fr.fold(M[D.H_NAME].float().to(dev), M['dspark_hidden_out.bias'].float().to(dev))
    P0 = torch.from_numpy(D.gguf_f32(T['dspark_hidden_out.fold'])).to(dev)
    rep['fold_max_abs_diff_vs_torch'] = float((P - P0).abs().max())
    rep['fold_frac_diff'] = float((P != P0).float().mean())
    rep['fold_bias_max_abs_diff'] = float((c - torch.from_numpy(D.gguf_f32(T['dspark_hidden_out.fold_bias'])).to(dev)).abs().max())
    rep['bytes'] = {f: os.path.getsize(f) for f in rep['files']}
    json.dump(rep, open(out + '-export.json', 'w'), indent=1)
    print(json.dumps(rep))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
