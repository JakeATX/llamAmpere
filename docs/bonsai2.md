# Bonsai 2 on llamAmpere v0.3.1

llamAmpere v0.3.1 adds CPU and CUDA inference for Prism ML's Ternary Bonsai 2 27B, and an SM86 decode path for its two ternary containers.

## Model and runtime

Use the official [GGUF release](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf). The model uses Qwen3.8's `qwen35` architecture with signed, blockwise Hadamard transforms folded into the weights. Packed weights alone are insufficient: activations need the matching transform, embeddings need the inverse transform, and folded GDN output projections need grouped value-head order.

- `PQ2_0`: group 128, tensor type 142, 34 bytes per block.
- `PTQ1_0`: group 128, tensor type 143, 28 bytes per block.
- Existing upstream `Q2_0` remains group 64, tensor type 42. TurboQuant and ConvRot IDs 43 through 50 remain unchanged.
- CUDA supports packed decode, native quantized prefill, row lookup, and signed FWHT. CPU supports codecs, dot products, and FWHT. Other GPU backends are not part of this port's validation.
- No Python, PyTorch, Triton, or extra runtime library is required for inference. Use the existing CUDA toolkit and CMake build.
- Hadamard metadata is validated at load time and transform coverage is checked when a graph is built. Folded output heads use the full projection where local vocabulary-shortlist shortcuts would omit the transform.

Prism kernels and runtime changes are adapted from [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp), pinned to `5d80cff0b8cb9f2bf823cfc4e71e3abb97f290d6`. Principal changes: `67c1af28b`, `64d1ed1b6`, `c09e6defb`, `34af866af`, `df01263d1`, `b73833108`, `ed4119b4b`, `8bbb28b76`, `e0c828b90`, `633168fb6`, `acdccf9c1`, `01fd9521c`, `9294043b2`, `7f292be19`, `6f1c6908d`, and `1bc46a8a7`. This is a focused port, not a merge of Prism's speculative decoder or release machinery. Upstream authors retain their original copyright under the repository MIT license; model weights are Apache 2.0.

## Build and run

```sh
cmake -S . -B build-bonsai -G Ninja \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON
cmake --build build-bonsai -j 4 --target \
  llama-cli llama-completion llama-server llama-bench llama-quantize \
  test-turbo-quant test-quantize-fns test-backend-ops

hf download prism-ml/Ternary-Bonsai-2-27B-gguf \
  --revision 6ed5e12bf84b7a63069882c91dd9e9218647d17b \
  Ternary-Bonsai-2-27B-PTQ1_0.gguf --local-dir models/bonsai2

build-bonsai/bin/llama-server \
  -m models/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  -ngl 99 -fa on -c 32768 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
```

The PTQ1_0 checkpoint is 5,946,648,928 bytes; SHA256 `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3`. Allow additional memory for context, recurrent state, and compute buffers. Lower `-ngl` and context size when the GPU is shared. Vision requires the separate mmproj file and has not been validated by this port.

The PQ2_0 checkpoint at the same revision is 7,206,168,928 bytes; SHA256 `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1`.

Start validation with ordinary decoding. Compatibility of a particular external speculative drafter is a separate test; the existing Qwen vocabulary maps should not be assumed appropriate for this checkpoint.

## Review checks

```sh
build-bonsai/bin/test-turbo-quant
build-bonsai/bin/test-quantize-fns
build-bonsai/bin/test-backend-ops -o MUL_MAT_HADAMARD
build-bonsai/bin/test-backend-ops -p '(pq2_0|ptq1_0)'
```

Check executed counts, not only the final backend verdict. Preserve the raw logs and distinguish baseline failures from port regressions. The unchanged v0.3.1 baseline has strided `GET_ROWS` failures for `tq4_1s` where both CPU and CUDA contain NaNs.

## SM86 decode kernels (v0.3.1)

Both containers hold the same ternary values and block scales (verified tensor by tensor); they differ only in packing. `PQ2_0` stores 2-bit codes and unpacks with one byte permute per 4 weights. `PTQ1_0` stores 5 trits per byte and needs a multiply-by-3 digit extraction, so its GEMV is ALU-bound rather than bandwidth-bound on a 3090 Ti: at 16K context `PQ2_0` decodes at 69.7 tok/s single-token and 104.8 tok/s with the MTP drafter, `PTQ1_0` at 62.8 tok/s single-token and 59.1 with the drafter, for about 1.1 GB more peak VRAM (release build, one 16K rag prompt, two seeds, full runs of at least 5,000 generated tokens at temperature 1.0). On `PTQ1_0` no drafter depth pays at temperature 1.0: depths 1, 2 and 3 decode at 60.5, 59.0 and 59.1 tok/s, because a width-4 verify pass costs about 2.85 single-token steps at 0.55 acceptance. Pick `PQ2_0` unless the card cannot hold it, and run `PTQ1_0` without the drafter.

The v0.3.1 `PTQ1_0` kernels do the following, all bit-exact against the Prism reference (greedy output hash unchanged):

- Verify widths 2-8 (speculative decoding) run a cross-column reuse kernel: each 128-weight block is split over 4 lanes, each lane unpacks its quarter once and dots every column. The upstream path re-unpacked the block per column per row and spilled 540-1260 bytes per thread at widths 2-4, which made a width-4 verify cost 4.4 single-token steps. On by default on cc 8.6; `GGML_CUDA_SM86_PTQ1_REUSE=0` restores the generic path.
- Rows per block at widths 2-4 drop from 8 to 2 for `PTQ1_0` (register pressure).
- The dp4a products run on the unsigned digits 0..2 and the exact per-sub-block activation sum is subtracted once at the end, instead of a byte-wise `-1` correction on every 4-weight group. Same values, same summation order.

## Validation

Validated on an RTX 3090 Ti (SM86), Release CUDA build:

- `test-turbo-quant` and `test-quantize-fns`: exit 0, including the new packed-codec and CPU-dot checks.
- Hadamard backend filter: 32/32 executed CUDA cases passed against the CPU reference, including F16 and F32.
- Ternary backend filter: 478/478 executed CUDA cases passed against the CPU reference, including decode, prefill, and row lookup.
- Official PTQ1_0 checkpoint: CPU and full GPU offload produced byte-identical 16-token greedy completions to a separately built, unmodified Prism CPU reference at the source revision above.
- Official PQ2_0 checkpoint: full GPU offload produced the same completion. PTQ1_0 with `-ctk q8_0 -ctv turbo3 -fa on` also matched.
- A 17-token prefill prompt matched the reference's immediate EOS on both GPU formats; this covers native prefill selection but is not a decoding-quality test.
- A 19-token continuation prompt followed by 16 generated tokens also matched the pristine CPU reference on both GPU formats, exercising native prefill plus decode. Prompt: `This is a geography exercise. Complete each sentence with the correct city. The capital of France is`. Output SHA256: `3f5c9eec18bd6f2aafa433cb8e969d9f134c809d7a0bfc2ffbf606d1b6022917`.
- Real PTQ1_0 benchmark with `-p 512 -n 32 -r 1 -ngl 99 -ctk q8_0 -ctv turbo3`: exit 0. This is a cache-path smoke test, not a performance claim.

Reproduce the short reference comparison with each build and checkpoint:

```sh
build-bonsai/bin/llama-completion \
  -m models/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  -ngl 99 -c 512 -b 64 -ub 64 -t 6 --temp 0 --seed 42 -n 16 \
  --no-conversation --no-display-prompt -p 'The capital of France is'
```

Use `-ngl 0` for the CPU reference. The output is ` Paris.\nThe capital of Germany is Berlin.\nThe capital of Italy is\n\n`, SHA256 `8d38418fef11ed83fdbb7e45f4b61bbcc979905965868241cbe1039357125a2d`.

The pre-existing baseline backend sweep has two failing `GET_ROWS(type=tq4_1s,n=256,m=5,r=4,be1=7,be2=1,vs0=1)` cases (`v=0` and `v=1`) and one `MUL_MAT_ID(type_a=iq4_xs,...,n_mats=17,n_used=1,...,k=512)` case with error 0.0031 versus tolerance 0.0005; none involve the ternary types. Long-context quality, vision, and hardware other than SM86 remain unvalidated. Short greedy matches are smoke tests, not a model-quality evaluation.

Ready-to-run files with the Qwen3.8 MTP draft head fused in (`PQ2_0` and `PTQ1_0`) are published at [jakeatx/Qwen3.8-27B-GGUF](https://huggingface.co/jakeatx/Qwen3.8-27B-GGUF).

## Credits

Bonsai 2 27B is a Prism ML model (weights Apache 2.0, https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf).
The PQ2_0 and PTQ1_0 quantization formats, the folded signed-Hadamard runtime, and the CPU/CUDA kernels
in this port are adapted from Prism ML's llama.cpp fork (https://github.com/PrismML-Eng/llama.cpp, MIT,
same license text as this repository) at the commit listed above. Cite the model as Prism ML asks:

```bibtex
@techreport{bonsai2_27b,
    title   = {Bonsai 2 27B: A 27B Ternary Reasoning Model},
    author  = {Prism ML},
    year    = {2026},
    month   = {September},
    url     = {https://prismml.com}
}
```
