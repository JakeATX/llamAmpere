# Flash-Next MTP head variants

Run `scripts/quantize-flash-next-mtp.sh VARIANT` to create a separate draft-head GGUF from the BF16 source in `/mnt/models/flash-next`. The default output directory is `/mnt/models/fnspec/head-quants`; `MODEL_DIR`, `OUTPUT_DIR`, and `QUANTIZER` can override the paths. The recipe refuses to overwrite an existing GGUF or partial output. It checks available memory and disk space before starting and every five seconds while quantizing.

Build `llama-quantize` in the CPU build and copy it beside the original as `build-cpu/bin/llama.cpp-fnspec-quantize` before running the recipe. Use the required `llama.cpp-fnspec-*` systemd unit, build locks, and CPU and memory limits when building. The recipe uses eight CPU threads and a 512 MiB tensor-row buffer; it does not use the GPU.

The supported variants are:

- `Q2_K`: Q2_K for embedding, output, attention, and expert gate/up tensors. Sensitive hyper-connection and indexer tensors stay BF16; the hidden-state projection stays Q8_0. Expert down rows have 640 columns, so the existing quantizer falls back from Q2_K to Q4_0 for those tensors.
- `IQ3-Q4-output`: the existing `mtp48-types.txt` tensor map with only the output projection changed from Q5_K to Q4_K. This gives a smaller option while retaining the existing expert types.
- `IQ3-Q2-output`: the same mixed tensor map with only the output projection changed to Q2_K. It preserves the original CPU embedding precision while saving the same GPU output-buffer space as `IQ3-Q2-io`.
- `IQ3-Q2-io`: the existing mixed tensor map with both the embedding and output projection changed to Q2_K.
- `Q4_K_S`: a Q4_K control with the same sensitive tensors preserved as `Q2_K`. This whole-head variant is larger than the existing IQ3 head. The quantizer falls back to Q5_0 for 640-column expert down rows.

All variants start from BF16, without requantizing the IQ3 artifact. They use `/mnt/models/flash-next/imatrix-swiftfn-v1mix-mtp48proxy.gguf`, the existing proxy importance matrix. That matrix has no embedding or output entries, so those tensors use the quantizer's unweighted path. The IQ3 variants use explicit tensor types for the full existing mixed map; command-line embedding and output types take precedence over that map.

Use the resulting GGUF as the existing MTP draft-model argument. The quantization recipe changes neither runtime defaults nor the target-model weights. Compare acceptance and the full interleaved protocol against `/mnt/models/flash-next/mtp-Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS.gguf` before selecting a default. A smaller file alone does not establish a speed or acceptance improvement.

Each completed file receives a SHA-256 sidecar. Local inventories under `head-quants` record exact file and tensor sizes, all tensor types and shapes, and the metadata comparison against the IQ3 baseline. Artifacts and manifests stay outside the source commit.

`scripts/bench-fnspec-helper-memory.py` follows the owner's 12:55Z rule: n-gram (PLE) lookups may read NVMe and are reported per request. Other model tensors must remain resident; unidentified request reads cannot pass final acceptance. Up to 1 GiB can remain provisional during a diagnostic request; larger residuals stop it immediately. Only traced WILLNEED calls wholly inside PLE qualify. Mixed tensor ranges, expert reads, uncertain request-boundary calls, and missing counters stop an arm. The earlier 12:40Z zero-PLE-read hold is revoked.

`--ple-profile /mnt/models/.../profile.rows` remains optional. When selected, the runner snapshots and hashes the profile, requires cache activation, and refuses logged misses through teardown. It is not required for ordinary PLE lookup arms. `--activation-io-trace` can separately account successful owned activation-file allocation metadata through FNARENA's strict path/interval/conservation validator; it does not exempt model or activation payload reads. Unknown bytes still fail.

Full helper arms require 53 GiB available at admission (NEXT_24H), a 7 GiB running floor and a 23,552 MiB whole-card limit. FNSHIP's two reversed full reference rounds precede these arms. Completed runs below 300 W during decode or below 90% device hits are diagnostics. Missing measurements and processing failures cannot become diagnostic passes.

`LLAMA_MTP_PIN_BORROWED_HOST=1` pins only successfully borrowed CPU output/embedding tensors in their donor's existing storage. GPU tensors and the PLE lookup table are excluded. Locks belong to the donor, survive repeated helper loads and stay alive through borrowed-storage ownership. A failed lock refuses helper loading; it never silently proceeds with evictable borrowed weights. The option is off until enabled by a launch configuration. It establishes residency for those tensors, not attribution of historical unknown bytes.

CUDA donors prepare only `token_embd.weight` and `output.weight` as private writable views at their existing mapped addresses, then register successfully borrowed host tensors using the expert arena's CUDA host-registration API. The driver rejects shared file views on this machine. Model-file bytes are unchanged; PLE retains its mapping. Registration may make private pages resident through driver copy-on-write, while the original file cache is reclaimable. No second tensor buffer or explicit weight-copy allocation is retained. Donor-owned unregister precedes mapping teardown. This does not raise the service's 8 MiB `mlock` limit. CPU donors retain checked `mlock`. Any mapping/registration failure refuses startup.

`--gather-io-trace` snapshots and hashes the validated CPU gather interceptor. Its source tensor must match the complete inventory's name, path and starting file offset. Calling-thread intervals may not overlap mapped-read attribution; every remaining process byte must still be identified before an arm counts.
