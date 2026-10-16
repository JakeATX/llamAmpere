# Experimental Nemotron-H decode fusions

Two CUDA graph fusions are available for single-token, single-sequence decode on sm_80 and sm_86. Both are disabled by default. Set each switch before starting the process:

```sh
GGML_CUDA_NEMOTRON_SSM=1 GGML_CUDA_NEMOTRON_MOE=1 ./build/bin/llama-cli -m model.gguf -ngl 99 -fa on -c 32768 -n 128 -p "Hello"
```

`GGML_CUDA_NEMOTRON_SSM=1` replaces the convolution window update, biased SiLU convolution, one-step SSM scan, cache writes, D skip and SwiGLU with one kernel. It supports fp32 state, convolution width 4, state width 96 or 128, and one rollback snapshot. The grouped norm and projections remain separate. Cache storage and serialization are unchanged. This switch takes precedence over the older optional convolution-ring fusion.

`GGML_CUDA_NEMOTRON_MOE=1` replaces a routed expert down projection and its weighted reduction with activation quantization followed by one grouped GEMV/combine kernel. Supported weights are Q2_0, PQ2_0, Q2_K, IQ2_XXS, IQ2_XS and IQ2_S. Top-k must be between 2 and 32. The existing device router and grouped up projection remain in use. The shared expert has different dimensions and remains separate.

Unsupported shapes, multiple tokens or sequences, split/host weight buffers, concurrent event schedules, observable intermediates and unsafe allocation aliases use the existing path. `GGML_CUDA_DISABLE_FUSION=1` disables both. No new graph operation or cache format is introduced.

The opt-in backend cases include the real 128-head, 64-channel, 96-state SSM dimensions and 512-expert MoE widths/top-k values:

```sh
GGML_TEST_NEMOTRON_DECODE=1 GGML_CUDA_NEMOTRON_SSM=1 GGML_CUDA_NEMOTRON_MOE=1 ./build/bin/test-backend-ops -b CUDA0 -o 'NEMOTRON_.*'
GGML_TEST_NEMOTRON_DECODE=1 GGML_CUDA_NEMOTRON_SSM=1 ./build/bin/test-backend-ops perf -b CUDA0 -o NEMOTRON_SSM_DECODE
GGML_TEST_NEMOTRON_DECODE=1 GGML_CUDA_NEMOTRON_MOE=1 ./build/bin/test-backend-ops perf -b CUDA0 -o NEMOTRON_MOE_DECODE
```

These composite tests check fusion counters as well as graph outputs. A CPU-only run validates the reference graphs; it does not validate the CUDA fusions. GPU numerical parity, greedy token identity, graph replay, disk cache restore and performance require hardware validation before enabling either switch in production.
