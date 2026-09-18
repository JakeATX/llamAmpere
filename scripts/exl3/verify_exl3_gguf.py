#!/usr/bin/env python3
"""EX2 M1 converter check: for chosen tensors, reconstruct W_eff from the GGUF (EXL3 bytes + .suh + .svh) with the numpy
reference and compare bit-exactly (as float32) against W_eff reconstructed from the original safetensors with the stock
converter's V-head permutation applied to the reconstructed matrix. CPU only.
usage: verify_exl3_gguf.py GGUF MODEL_DIR  [gguf_name=hf_prefix[:kind] ...]   kind in {none,qkv_out,z_out,out_proj_in}
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
import exl3_ref, gguf
from gguf import GGUFReader

NK, NV, HK, HV = 16, 48, 128, 128      # linear_num_key_heads, linear_num_value_heads, key/value head dims (Qwen3.8-27B)
def vperm(n):                          # new[j] = old[perm[j]] for NV*HV features grouped [NK, NV/NK, HV] -> [NV/NK, NK, HV]
    return np.arange(n).reshape(NK, NV // NK, HV).transpose(1, 0, 2).reshape(-1)

gguf_path, model_dir = sys.argv[1], sys.argv[2]
specs = sys.argv[3:] or ["blk.0.attn_gate.weight=model.language_model.layers.0.linear_attn.in_proj_z:z_out",
                         "blk.0.ssm_out.weight=model.language_model.layers.0.linear_attn.out_proj:out_proj_in",
                         "blk.0.attn_qkv.weight=model.language_model.layers.0.linear_attn.in_proj_qkv:qkv_out",
                         "blk.3.attn_k.weight=model.language_model.layers.3.self_attn.k_proj:none",
                         "blk.64.nextn.eh_proj.weight=mtp.fc:none"]
r = GGUFReader(gguf_path)
tens = {t.name: t for t in r.tensors}
kv = {k: v for k, v in r.fields.items() if ".exl3." in k}
print("exl3 KV:", {k: bytes(v.parts[-1]).decode() if v.types and v.types[0] == gguf.GGUFValueType.STRING else int(v.parts[-1][0]) for k, v in kv.items()})
ok_all = True
for spec in specs:
    gname, rest = spec.split("="); hf, kind = rest.split(":")
    t = tens[gname]; bits = int(t.tensor_type.name.split("_")[1]); N, K = int(t.shape[1]), int(t.shape[0])
    raw = np.asarray(t.data).reshape(-1).view(np.int16).reshape(K // 16, N // 16, 16 * bits)
    suh_g = np.asarray(tens[gname.replace(".weight", ".suh")].data).astype(np.float16)
    svh_g = np.asarray(tens[gname.replace(".weight", ".svh")].data).astype(np.float16)
    C_g, b_g = exl3_ref.reconstruct_C(raw); W_g = exl3_ref.reconstruct_W(C_g, suh_g, svh_g).astype(np.float32)
    tr = exl3_ref.load_safetensors_tensor(model_dir, hf + ".trellis")
    suh = exl3_ref.load_safetensors_tensor(model_dir, hf + ".suh").astype(np.float16)
    svh = exl3_ref.load_safetensors_tensor(model_dir, hf + ".svh").astype(np.float16)
    C_o, b_o = exl3_ref.reconstruct_C(tr); W_o = exl3_ref.reconstruct_W(C_o, suh, svh).astype(np.float32)   # [K, N]
    if kind == "z_out":
        W_o = W_o[:, vperm(N)]
    elif kind == "qkv_out":
        qk = NK * HK * 2; p = np.concatenate([np.arange(qk), qk + vperm(N - qk)]); W_o = W_o[:, p]
    elif kind == "out_proj_in":
        W_o = W_o[vperm(K), :]
    same = np.array_equal(W_g, W_o); ok_all &= same and (b_g == b_o == bits)
    print(f"{gname}: {t.tensor_type.name} K={K} N={N} kind={kind} bits {b_g}/{b_o}/{bits} W_eff bit-exact={'PASS' if same else 'FAIL'} "
          f"max|Δ|={np.abs(W_g - W_o).max():.3e} |W|max={np.abs(W_o).max():.3f}")
print("VERIFY", "PASS" if ok_all else "FAIL")
