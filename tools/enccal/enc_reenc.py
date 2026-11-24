#!/usr/bin/env python3
"""CI-REENC: re-encode a healed quantized model once from its BF16 source, with the healed model's activations as the
target. No training: encode cost only (plus one imatrix pass on the healed file).

Why: a heal (scale + norm tuning of the quantized file) moves the activations every later layer sees. The original
encode used statistics from the unhealed model; re-encoding each tensor from BF16 against the healed model's
activation statistics lets the codes match the inputs they will really get. Tool only: no runtime or format change.

Steps (each is a subcommand; `all` runs plan -> imatrix -> encode -> carry -> check):
  plan     HEALED.gguf BF16.gguf OUT_DIR
           Reads the healed file's tensor types. Writes OUT_DIR/tensor_types.txt (anchored `^name$=type`, one per
           quantized tensor, for llama-quantize --tensor-type-file) and OUT_DIR/plan.json (base type, embedding and
           output types, float tensors to carry). Refuses when a healed tensor is missing from BF16 or its shape differs.
  imatrix  HEALED.gguf CALIB.txt OUT_DIR [--bin llama-imatrix] [--chunks N] [--ngl 99] [-- extra imatrix args]
           Activation statistics from the healed model (LLAMA_IMATRIX_INPUT_IDS / LLAMA_IMATRIX_TOKEN_WEIGHTS from the
           environment pass through, so ENCCALIB's weighted calibration works here too). -> OUT_DIR/imatrix.healed.gguf
  encode   BF16.gguf OUT_DIR [--bin llama-quantize] [--threads N] [--imatrix FILE]
           llama-quantize from BF16 with the healed imatrix and the healed file's exact per-tensor types.
           -> OUT_DIR/reenc.gguf
  carry    HEALED.gguf OUT_DIR
           Copies the healed float tensors (norms, healed scales: any non-quantized tensor whose values differ from
           the re-encoded file) into OUT_DIR/reenc.gguf in place. Same shape required; f32/f16/bf16 are converted.
  check    HEALED.gguf OUT_DIR [--kld-base FILE --ppl-bin llama-perplexity --ngl 99]
           Every tensor type of reenc.gguf equals the healed file's; sizes; optional KLD of both files against a
           llama-perplexity --kl-divergence-base file (healed vs re-encoded, same base), printed side by side.
  all      HEALED.gguf BF16.gguf CALIB.txt OUT_DIR [options of the steps]

Needs numpy and the repo's gguf-py (found next to this file). Run it with a venv python, e.g.
  ~/.venvs/encbeat-l1r2/bin/python tools/enccal/enc_reenc.py all healed.gguf src-bf16.gguf calib.txt out/
"""
import argparse, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'gguf-py'))

import numpy as np  # noqa: E402
import gguf  # noqa: E402

T = gguf.GGMLQuantizationType
FLOAT_TYPES = {T.F32: np.float32, T.F16: np.float16}
EMBD = 'token_embd.weight'
OUTPUT = 'output.weight'


def _tensors(path):
    r = gguf.GGUFReader(path, 'r')
    return r, {t.name: t for t in r.tensors}


def _is_float(tt):
    return tt in (T.F32, T.F16, T.BF16)


def _shape(t):
    return tuple(int(x) for x in t.shape)


def _as_f32(t):
    a = np.asarray(t.data)
    if t.tensor_type == T.BF16:
        return (a.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(-1)
    return a.astype(np.float32).reshape(-1)


def _from_f32(x, tt):
    if tt == T.BF16:
        u = x.astype(np.float32).view(np.uint32)
        u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16  # round to nearest even
        return u.astype(np.uint16)
    return x.astype(FLOAT_TYPES[tt])


def _run(cmd, env=None):
    print('+', ' '.join(cmd), flush=True)
    rc = subprocess.call(cmd, env=env)
    if rc != 0:
        sys.exit(f'step failed (rc {rc}): {cmd[0]}')


# ------------------------------------------------------------------------------------------------------------- plan
def cmd_plan(a):
    os.makedirs(a.out, exist_ok=True)
    _, h = _tensors(a.healed)
    _, s = _tensors(a.bf16)
    bad = [n for n in h if n not in s or _shape(h[n]) != _shape(s[n])]
    if bad:
        sys.exit(f'refusing: {len(bad)} healed tensors missing from BF16 or with another shape, e.g. {bad[:5]}')
    quant = {n: t.tensor_type for n, t in h.items() if not _is_float(t.tensor_type)}
    if not quant:
        sys.exit('refusing: the healed file has no quantized tensors')
    counts = {}
    for n, tt in quant.items():
        if n not in (EMBD, OUTPUT):
            counts[tt.name] = counts.get(tt.name, 0) + 1
    base = max(counts, key=counts.get) if counts else next(iter(quant.values())).name
    with open(os.path.join(a.out, 'tensor_types.txt'), 'w') as f:
        for n, tt in sorted(quant.items()):
            if n in (EMBD, OUTPUT):
                continue
            f.write('^' + re.escape(n).lower() + '$=' + tt.name.lower() + '\n')
    plan = dict(healed=os.path.abspath(a.healed), bf16=os.path.abspath(a.bf16), base_type=base,
                embd_type=h[EMBD].tensor_type.name if EMBD in h else None,
                output_type=h[OUTPUT].tensor_type.name if OUTPUT in h else None,
                n_quant=len(quant), type_counts=counts,
                carry=sorted(n for n, t in h.items() if _is_float(t.tensor_type)))
    json.dump(plan, open(os.path.join(a.out, 'plan.json'), 'w'), indent=1)
    print(f'plan: {len(quant)} quantized tensors, base {base}, types {counts}, {len(plan["carry"])} float tensors')


def _plan(out):
    p = os.path.join(out, 'plan.json')
    if not os.path.exists(p):
        sys.exit('run `plan` first')
    return json.load(open(p))


# ---------------------------------------------------------------------------------------------------------- imatrix
def cmd_imatrix(a):
    os.makedirs(a.out, exist_ok=True)
    cmd = [a.bin, '-m', a.healed, '-f', a.calib, '-o', os.path.join(a.out, 'imatrix.healed.gguf'), '-ngl', str(a.ngl)]
    if a.chunks:
        cmd += ['--chunks', str(a.chunks)]
    _run(cmd + list(a.extra or []))


# ----------------------------------------------------------------------------------------------------------- encode
def cmd_encode(a):
    p = _plan(a.out)
    im = a.imatrix or os.path.join(a.out, 'imatrix.healed.gguf')
    if not os.path.exists(im):
        sys.exit(f'no imatrix at {im}: run `imatrix` first')
    cmd = [a.bin, '--imatrix', im, '--tensor-type-file', os.path.join(a.out, 'tensor_types.txt')]
    if p['embd_type'] and p['embd_type'] not in ('F32', 'F16', 'BF16'):
        cmd += ['--token-embedding-type', p['embd_type'].lower()]
    if p['output_type'] and p['output_type'] not in ('F32', 'F16', 'BF16'):
        cmd += ['--output-tensor-type', p['output_type'].lower()]
    cmd += [a.bf16, os.path.join(a.out, 'reenc.gguf'), p['base_type'].lower()]
    if a.threads:
        cmd.append(str(a.threads))
    _run(cmd)


# ------------------------------------------------------------------------------------------------------------ carry
def cmd_carry(a):
    out = os.path.join(a.out, 'reenc.gguf')
    _, h = _tensors(a.healed)
    r, o = _tensors(out)
    n_copied = n_same = 0
    with open(out, 'r+b') as f:
        for n, ht in h.items():
            if not _is_float(ht.tensor_type):
                continue
            ot = o.get(n)
            if ot is None or _shape(ot) != _shape(ht) or not _is_float(ot.tensor_type):
                sys.exit(f'refusing carry: {n} missing in reenc.gguf, or another shape / a quantized type there')
            hv = _as_f32(ht)
            if np.array_equal(hv, _as_f32(ot)):
                n_same += 1
                continue
            raw = np.ascontiguousarray(_from_f32(hv, ot.tensor_type)).tobytes()
            if len(raw) != ot.n_bytes:
                sys.exit(f'refusing carry: {n} byte size {len(raw)} != {ot.n_bytes}')
            f.seek(ot.data_offset)
            f.write(raw)
            n_copied += 1
    print(f'carry: {n_copied} healed float tensors written into reenc.gguf, {n_same} already equal')


# ------------------------------------------------------------------------------------------------------------ check
def _kld(ppl_bin, model, base, ngl):
    txt = subprocess.run([ppl_bin, '-m', model, '--kl-divergence-base', base, '--kl-divergence', '-ngl', str(ngl)],
                         capture_output=True, text=True).stdout + ''
    m = re.search(r'Mean\s+KLD:\s+([0-9.eE+-]+)\s*±\s*([0-9.eE+-]+)', txt)
    return (float(m.group(1)), float(m.group(2))) if m else None


def cmd_check(a):
    out = os.path.join(a.out, 'reenc.gguf')
    _, h = _tensors(a.healed)
    _, o = _tensors(out)
    diff = [(n, h[n].tensor_type.name, o[n].tensor_type.name if n in o else 'missing') for n in h
            if n not in o or o[n].tensor_type != h[n].tensor_type]
    print(f'check: {len(h)} tensors, {len(diff)} with another type' + (f', e.g. {diff[:5]}' if diff else ''))
    print(f'size: healed {os.path.getsize(a.healed)/2**30:.3f} GiB, reenc {os.path.getsize(out)/2**30:.3f} GiB')
    res = dict(type_mismatch=len(diff))
    if a.kld_base:
        kh = _kld(a.ppl_bin, a.healed, a.kld_base, a.ngl)
        ko = _kld(a.ppl_bin, out, a.kld_base, a.ngl)
        print(f'KLD vs base: healed {kh}, reenc {ko}')
        res.update(kld_healed=kh, kld_reenc=ko)
    json.dump(res, open(os.path.join(a.out, 'check.json'), 'w'), indent=1)
    if diff:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)

    def opts(p, *names):
        if 'imatrix' in names:
            p.add_argument('--imatrix-bin', dest='im_bin', default='llama-imatrix')
            p.add_argument('--chunks', type=int, default=0)
            p.add_argument('--extra', nargs=argparse.REMAINDER)
        if 'encode' in names:
            p.add_argument('--quantize-bin', dest='q_bin', default='llama-quantize')
            p.add_argument('--threads', type=int, default=0)
            p.add_argument('--imatrix', default=None)
        if 'check' in names:
            p.add_argument('--kld-base', default=None)
            p.add_argument('--ppl-bin', default='llama-perplexity')
        p.add_argument('--ngl', type=int, default=99)

    p = sp.add_parser('plan'); p.add_argument('healed'); p.add_argument('bf16'); p.add_argument('out')
    p = sp.add_parser('imatrix'); p.add_argument('healed'); p.add_argument('calib'); p.add_argument('out'); opts(p, 'imatrix')
    p = sp.add_parser('encode'); p.add_argument('bf16'); p.add_argument('out'); opts(p, 'encode')
    p = sp.add_parser('carry'); p.add_argument('healed'); p.add_argument('out')
    p = sp.add_parser('check'); p.add_argument('healed'); p.add_argument('out'); opts(p, 'check')
    p = sp.add_parser('all'); p.add_argument('healed'); p.add_argument('bf16'); p.add_argument('calib'); p.add_argument('out')
    opts(p, 'imatrix', 'encode', 'check')
    a = ap.parse_args()
    a.bin = getattr(a, 'im_bin', None) if a.cmd == 'imatrix' else getattr(a, 'q_bin', None)
    if a.cmd == 'all':
        # each step picks its own binary
        steps = [(cmd_plan, None), (cmd_imatrix, a.im_bin), (cmd_encode, a.q_bin), (cmd_carry, None), (cmd_check, None)]
        for fn, b in steps:
            a.bin = b
            fn(a)
        return
    dict(plan=cmd_plan, imatrix=cmd_imatrix, encode=cmd_encode, carry=cmd_carry, check=cmd_check)[a.cmd](a)


if __name__ == '__main__':
    main()
