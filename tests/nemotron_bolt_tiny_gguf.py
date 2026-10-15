#!/usr/bin/env python3
"""Write a tiny, randomly initialized GGUF with the Nemotron 3.5 Bolt layout.

Bolt (nemotron_h_moe) is a Mamba-2 / attention / LatentMoE hybrid whose MoE
layers each have their own routed-expert count and expert width (Star Elastic
search). This script builds a ~100 MB model with the same structure so the
loader, graph, KV-cache types, offload and slot save/restore paths can be
tested on CPU without the 20 GB checkpoint.

usage: nemotron_bolt_tiny_gguf.py OUT.gguf [--vocab-from REAL_BOLT.gguf] [--seed N]
Without --vocab-from it writes a small byte-level BPE vocab (256 tokens).
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gguf-py"))
import gguf  # noqa: E402

# layer pattern, like Bolt: M = mamba2, E = latent MoE, A = attention (Bolt has 40 M, 40 E, 8 A)
PATTERN   = ["M", "E", "M", "A", "E", "M", "E", "M", "A", "E"]
N_EXPERT  = {1: 8, 4: 12, 6: 6, 9: 10}      # per-layer routed experts (heterogeneous on purpose)
N_FF_EXP  = {1: 128, 4: 192, 6: 128, 9: 256}  # per-layer expert width
N_USED    = 3
N_EMBD    = 256
N_LATENT  = 64
N_FF_SH   = 384
N_HEAD    = 4
N_HEAD_KV = 2
HEAD_DIM  = 64
SSM_STATE = 16
SSM_GROUP = 2
SSM_HEADS = 8                      # mamba heads (time_step_rank)
SSM_INNER = 2 * N_EMBD             # expand 2
SSM_CONV  = 4
N_CTX     = 4096


def read_vocab(path):
    r = gguf.GGUFReader(path)
    out = {}
    for k in ("tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.token_type"):
        f = r.fields[k]
        out[k] = [f.contents(i) for i in range(len(f.data))] if hasattr(f, "contents") else None
    out["pre"] = r.fields["tokenizer.ggml.pre"].contents()
    out["eos"] = r.fields["tokenizer.ggml.eos_token_id"].contents()
    out["bos"] = r.fields["tokenizer.ggml.bos_token_id"].contents()
    tmpl = r.fields.get("tokenizer.chat_template")
    out["chat_template"] = tmpl.contents() if tmpl is not None else None
    return out


def byte_vocab():
    from gguf.vocab import SpecialVocab  # noqa: F401  (import check only)
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    toks = [chr(c) for c in cs]
    toks += ["<unk>", "<s>", "</s>"]
    types = [1] * 256 + [3, 3, 3]
    return {"tokenizer.ggml.tokens": toks, "tokenizer.ggml.merges": [], "tokenizer.ggml.token_type": types,
            "pre": "default", "eos": 258, "bos": 257, "chat_template": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--vocab-from", default=None)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    voc = read_vocab(a.vocab_from) if a.vocab_from else byte_vocab()
    n_vocab = len(voc["tokenizer.ggml.tokens"])
    n_layer = len(PATTERN)

    w = gguf.GGUFWriter(a.out, "nemotron_h_moe")
    w.add_name("nemotron-bolt-tiny-random")
    w.add_context_length(N_CTX)
    w.add_embedding_length(N_EMBD)
    w.add_block_count(n_layer)
    w.add_vocab_size(n_vocab)
    ff = [N_FF_EXP.get(i, 0) for i in range(n_layer)]
    w.add_feed_forward_length(ff)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv([N_HEAD_KV if p == "A" else 0 for p in PATTERN])
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_rope_dimension_count(HEAD_DIM)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_layer_norm_eps(1e-5)
    w.add_expert_count(max(N_EXPERT.values()))
    w.add_expert_used_count([N_USED if p == "E" else 0 for p in PATTERN])
    w.add_expert_feed_forward_length(ff)
    w.add_expert_shared_feed_forward_length(N_FF_SH)
    w.add_expert_shared_count(1)
    w.add_expert_weights_norm(True)
    w.add_expert_weights_scale(2.5)
    w.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
    w.add_expert_group_count(1)
    w.add_expert_group_used_count(1)
    w.add_moe_latent_size(N_LATENT)
    w.add_ssm_conv_kernel(SSM_CONV)
    w.add_ssm_state_size(SSM_STATE)
    w.add_ssm_group_count(SSM_GROUP)
    w.add_ssm_inner_size(SSM_INNER)
    w.add_ssm_time_step_rank(SSM_HEADS)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16)

    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre(voc["pre"])
    w.add_token_list(voc["tokenizer.ggml.tokens"])
    w.add_token_types(voc["tokenizer.ggml.token_type"])
    if voc["tokenizer.ggml.merges"]:
        w.add_token_merges(voc["tokenizer.ggml.merges"])
    w.add_bos_token_id(voc["bos"])
    w.add_eos_token_id(voc["eos"])
    w.add_add_bos_token(False)
    if voc["chat_template"]:
        w.add_chat_template(voc["chat_template"])

    def t(name, shape, scale=None, dtype=np.float16, kind="normal"):
        # shape is numpy order (outer..inner) = reversed ggml ne
        if kind == "ones":
            x = np.ones(shape, dtype=np.float32)
        elif kind == "zeros":
            x = np.zeros(shape, dtype=np.float32)
        else:
            fan_in = shape[-1]
            x = rng.standard_normal(shape, dtype=np.float32) * (scale if scale is not None else 1.0 / np.sqrt(fan_in))
        w.add_tensor(name, x.astype(dtype))

    t("token_embd.weight", (n_vocab, N_EMBD), scale=0.5)
    t("output.weight", (n_vocab, N_EMBD))
    t("output_norm.weight", (N_EMBD,), dtype=np.float32, kind="ones")
    d_in_proj = 2 * SSM_INNER + 2 * SSM_GROUP * SSM_STATE + SSM_HEADS
    d_conv = SSM_INNER + 2 * SSM_GROUP * SSM_STATE
    for il, p in enumerate(PATTERN):
        b = f"blk.{il}."
        t(b + "attn_norm.weight", (N_EMBD,), dtype=np.float32, kind="ones")
        if p == "M":
            t(b + "ssm_in.weight", (d_in_proj, N_EMBD))
            t(b + "ssm_conv1d.weight", (d_conv, SSM_CONV), scale=0.5, dtype=np.float32)
            t(b + "ssm_conv1d.bias", (d_conv,), scale=0.1, dtype=np.float32)
            t(b + "ssm_dt.bias", (SSM_HEADS,), scale=0.5, dtype=np.float32)
            a_log = -np.exp(rng.uniform(0.0, 2.0, size=(SSM_HEADS, 1)).astype(np.float32))
            w.add_tensor(b + "ssm_a", a_log)
            t(b + "ssm_d", (SSM_HEADS, 1), dtype=np.float32, kind="ones")
            t(b + "ssm_norm.weight", (SSM_GROUP, SSM_INNER // SSM_GROUP), dtype=np.float32, kind="ones")
            t(b + "ssm_out.weight", (N_EMBD, SSM_INNER))
        elif p == "A":
            t(b + "attn_q.weight", (N_HEAD * HEAD_DIM, N_EMBD))
            t(b + "attn_k.weight", (N_HEAD_KV * HEAD_DIM, N_EMBD))
            t(b + "attn_v.weight", (N_HEAD_KV * HEAD_DIM, N_EMBD))
            t(b + "attn_output.weight", (N_EMBD, N_HEAD * HEAD_DIM))
        else:
            ne, nf = N_EXPERT[il], N_FF_EXP[il]
            t(b + "ffn_gate_inp.weight", (ne, N_EMBD), dtype=np.float32)
            t(b + "exp_probs_b.bias", (ne,), scale=0.1, dtype=np.float32)
            t(b + "ffn_latent_down.weight", (N_LATENT, N_EMBD))
            t(b + "ffn_latent_up.weight", (N_EMBD, N_LATENT))
            t(b + "ffn_up_exps.weight", (ne, nf, N_LATENT))
            t(b + "ffn_down_exps.weight", (ne, N_LATENT, nf))
            t(b + "ffn_up_shexp.weight", (N_FF_SH, N_EMBD))
            t(b + "ffn_down_shexp.weight", (N_EMBD, N_FF_SH))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {a.out}: {n_layer} layers {''.join(PATTERN)}, experts/layer {N_EXPERT}, vocab {n_vocab}")


if __name__ == "__main__":
    main()
