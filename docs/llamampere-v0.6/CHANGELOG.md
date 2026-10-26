# v0.6 changes in progress

## Packed attention masks for cards with 12 GB or less

The packed-mask port stores each binary attention-mask entry in one bit instead of two bytes. The global default remains OFF, including on 24 GB cards. Recommended 12 GB, 10 GB and 8 GB launch configurations must set `LLAMA_KQ_MASK_PACKED=1`; this setting does not choose the model or make an oversized model fit. The current 12 GB launch command in the repository README includes the setting. On Windows PowerShell, set `$env:LLAMA_KQ_MASK_PACKED="1"` before the launch command. Explicit `LLAMA_KQ_MASK_PACKED=0` disables packing.

Frozen R10 results on Swift 1.5 with SJ-KVaRN 3/2t, context 204,800 and microbatch 512:

- Whole-card peak fell by 188 MiB at 199,212 prompt tokens plus 5,120 generated tokens. All generated tokens and text matched exactly. Save/restore, prefix reuse and rollback also matched.
- Six same-binary Taichi OFF/ON/OFF brackets measured generation +0.1213% (95% interval +0.0174% to +0.2253%; one-sided p=0.0151) and prompt processing -1.5876% (interval -1.6378% to -1.5374%; p=2.66e-9). This is a memory-saving option with a measured prompt-processing cost. No new speed campaign is required for the port.
- Native CUDA MMA and int8-QK readers consume packed bits. Unsupported routes, including sparse gathering, resident-body fetch and direct decode readers, expand to F16 scratch and may lose the device-memory saving. Other backends, ALiBi, training, KV streaming and models with custom mask layouts retain their existing masks.

These are historical R10 measurements, not resulting-source qualification. Before integration, the port needs CPU and native CUDA operator suites, prompt-cache identity, and the 12 GB release recipe at 204,800 context with OFF/ON output identity and sampled whole-card peak. Exact checks on the current binary explicitly pass `--no-spec-mb-mtp`. The separate bounded host-timer attempt and raw-result collection remain tracked in MASK175.
