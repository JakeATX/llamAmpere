# llamAmpere v0.6.3.3

Small cards get memory-bounded prompt passes by default. This release also fixes Windows loading and build paths and adds CPU expert-split diagnostics.

## Changes

- Flash-Next chooses prompt passes that fit temporary card memory. Below 23,552 MiB total card memory, the launcher also enables 512-row chunks for prompt tails of at least 1,024 tokens. Explicit environment settings take precedence. Set `LLAMA_B1_ADAPTIVE_PASSES=0` to restore fixed prompt passes. Normally free 24 GB cards keep their previous plan; a busy 24 GB card can use adaptive passes.
- On hybrid Intel CPUs, the Linux launcher counts physical performance cores and uses at most eight threads. Other CPUs retain their previous thread selection. No thread affinity is set.
- Windows uses normal model mappings when private remapping is unavailable, and MSVC uses supported atomic calls. Failed CUDA host-memory pinning can fall back off Linux. Fixes the private-remapping failure in [#94](https://github.com/JakeATX/llamAmpere/issues/94) and addresses Windows build/load reports [#93](https://github.com/JakeATX/llamAmpere/issues/93) and [#82](https://github.com/JakeATX/llamAmpere/issues/82). These changes do not prove the cause of a reported unknown-type error.
- Startup logs show whether CPU expert splitting is active, why it is inactive, and the chosen CPU share. `--moe-cpu-share 0.6` requests a 60% share when the existing split path is eligible; leaving it unset preserves automatic selection. This helps diagnose [#88](https://github.com/JakeATX/llamAmpere/issues/88); no speed improvement on the reporter's computer is established.
- Unsupported SJ-KVaRN settings now name the selected codec and its supported pairs. Scalar 4/3 remains supported; trellis 4/3 is unsupported.
- Both README build commands now select this release, including the Swift 27B command that still selected v0.6.2.

Thanks to @Pragmatism0220 for the small-card request in [#11](https://github.com/JakeATX/llamAmpere/issues/11), @francoispda for the 12 GB and Windows reports, @fchamieh for #82, and @justin025 for #88.

## Measurements and limits

The small-card study used an RTX 3090 limited to 12 GB, about 90 GiB of system RAM, and 350 W. Three product-mode ABBA blocks used 100K-token prompts, temperature 1, and 5,120 generated tokens per request. Prompt reading improved by 30.29% geometrically. Mean writing was 69.125 versus 69.859 tokens/s, with expert-cache hit snapshots of 60.62% versus 60.17%. Writing varied across seeds; these tests do not prove writing equivalence or a general speed gain.

Native 12 GB and 16 GB cards and hybrid Intel timing remain unmeasured. A 64 GB host reached the RAM floor or ran out of memory in testing; no 64 GB fit promise is made. These AI3 results do not predict third-party model performance.

The source candidate passed full-vocabulary checks on three long coding, agentic, and RAG prompts. Its measured drift was within the declared comparison against the prior release's own 12-versus-24 GB variation. This is not a claim of identical arithmetic. Prompt save/restore remains supported; fresh and restored conversations can choose different words when scores are close, and repeated-run differences also occur in the prior release.

Windows evidence includes MinGW compile/link and a CPU-only Wine mapping/model smoke from the Windows fix lane. Native MSVC and Windows CUDA model loading remain unverified. The tensor-split check emulates two devices on one card; it is not a physical two-card test. Flash-Next tensor split remains unsupported and refuses cleanly.

Linux installs build from source. This is a source-only release with no prebuilt binaries.
