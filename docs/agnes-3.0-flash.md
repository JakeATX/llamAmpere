# Agnes 3.0 Flash

This fork loads **Agnes 3.0 Flash** (`AgnesForConditionalGeneration`) for text
generation, including its MTP head. Upstream llama.cpp does not.

## What the model is

Agnes 3.0 Flash is a 72-layer dense hybrid. It is architecturally Qwen3.5 in
almost every respect, which is why it loads through the existing `QWEN35` path
rather than a new architecture:

| | |
|---|---|
| layers | 72 text + 1 MTP (`blk.72`) |
| layer mix | 54 delta-rule (`agnes_delta_attention`) + 18 global (`agnes_global_attention`) |
| global attention | every 4th layer, at indices 3, 7, 11, … 71 |
| hidden size | 5120 |
| FFN | 17408, **plus a second parallel SwiGLU at 2048** |
| attention | 24 heads / 4 KV heads, head_dim 256, output gate, partial rotary 0.25 |
| linear attention | 16 K heads / 48 V heads, head dim 128, conv kernel 4, swish output gate |
| vocabulary | 248,320 |
| max context | 262,144 |
| rope | mrope interleaved, sections [11, 11, 10], theta 1e7 |

## The one thing that is not Qwen3.5

Every layer runs a **second, narrower SwiGLU on the same input** and adds it into
the same residual stream. From `modeling_agnes.py`, `AgnesMLP.forward`:

```python
y = down_proj(act_fn(gate_proj(x)) * up_proj(x))
if parallel_ffn is not None:
    y = y + parallel_ffn(x)
```

Both branches consume the same `x`; neither is a gate on the other. This is a
full second feed-forward network on all 72 layers, and it is where the extra
parameters live.

Support for it is three additions:

- a KV key `{arch}.feed_forward_parallel_length` → `hparams.n_ff_par`
- tensors `blk.N.ffn_{gate,down,up}_par`, created only when `n_ff_par > 0`
- `build_layer_ffn` evaluates both branches from the same input and sums them

The MTP block is excluded: `mtp.layers.0.mlp` has no `parallel_ffn`.

A Qwen3.5 or Qwen3.8 checkpoint does not write the new KV key, so `n_ff_par`
stays 0, the whole block is skipped, and those models load exactly as before.

## Converting

```sh
python convert_hf_to_gguf.py --outtype bf16 \
    --outfile Agnes-3.0-Flash-BF16.gguf  /path/to/Agnes-3.0-Flash
```

A correct conversion has **1188 tensors**: 1521 in the checkpoint minus 333
vision-tower tensors, which are not exported. Of those, **216 are `ffn_*_par`**
(72 layers × 3) at shapes `{5120, 2048}` for gate and up and `{2048, 5120}` for
down. All 15 MTP tensors land at `blk.72`.

Header keys worth checking after a convert:

```
qwen35.block_count                    73
qwen35.feed_forward_parallel_length   2048
qwen35.nextn_predict_layers           1
qwen35.full_attention_interval        4
```

Note that `general.architecture` reads `qwen35`, not `agnes`. That is
deliberate — the model loads through that path.

## Running

```sh
./build-sm86/bin/llama-server -m Agnes-3.0-Flash-IQ4_XS.gguf \
    -ngl 99 -fa on -c 32768 --parallel 1 --jinja
```

Add the MTP head with `--spec-type draft-mtp --spec-draft-n-max 3`.

## Vision

Not supported. The 333 vision-tower tensors are dropped at conversion and there
is no Agnes encoder on the `mmproj` path. This is a text-only path.

## Interoperability

The KV key, the tensor names and the tensor mapping are byte-identical to
`quimmedes/cafe-llama.cpp`. GGUFs convert and load across both trees.

## If loading fails

The parallel-branch tensors are **required**, not optional. Once
`feed_forward_parallel_length` is present, a missing or misnamed
`ffn_*_par` tensor aborts the load with a named error rather than being
silently dropped. That is intentional: silently dropping 216 tensors produces a
model that loads, runs, and generates fluent text that is quietly wrong.
