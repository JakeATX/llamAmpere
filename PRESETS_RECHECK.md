# Presets: recheck before 0.7 ships

Jake enabled all four best-known profiles on 2026-11-16T01:50Z. This lifts the old small-card approval hold. No speed or final-build fit claim is made.

| Preset | Source | Recheck before ship | Owner |
| --- | --- | --- | --- |
| 24dense | V07/INTEG3/QC_READY_integ3.md; ASYMCTX/MERGE_NOTE_kqpack_f2.md; DECODE20/QC_decode20_dsjrm24h.md | Final combined defaults; 1u load/decode under 23,552 MiB; 3u asymmetric slots and populated prompt-cache save/restore. Packed mask and dynamic ub/mask wait for their QC. DSJ/RM24H passed design QC; inherit INTEG3's eligible-path default after integration, with its kill switch. | COORD-taichi / INTEG3 |
| 12dense | DENSE8/opus-12gb-path/r205s3/r205s3.sh, U then X, plus driver extras; L0 d28a3c00 | Launch and short decode under 12,288 MiB; LSA engaged. Requested L0 differs from R205S3's measured 4cde6151. Confirm owner-selected model/sidecar paths and final settings. | COORD-bc |
| 10dense | LSA/REAL10GB/RESULT_speed10.md line 5; speed10/s10.sh and s10.py D2HA | Launch and short decode under 10,240 MiB; LSA engaged. Confirm model and sidecar paths and final settings; historical proxy data are not a new-build result. | COORD-bc |
| 8dense | LSA/REAL8GB/card8/lsa-8gb.env | Launch and short decode under 8,192 MiB; LSA, both overlays and exact head engaged. Uses the current base, with no F2 reserve=7/graphs-OFF delta. Switch to F2 + strips ON + graphs OFF only after COORD-bc posts the f205 pair. Confirm final healed model and all three sidecars. | COORD-bc |

All small profiles enforce one user. TODO: multi-user for small cards (future). Explicit user flags and shell environment override preset defaults. The caller supplies the matching model; presets do not download models.

The minimum acceptance is launch plus a short decode for each preset under its class cap, and a 24dense three-user asymmetric check. Preserve prompt caching. No old-binary speed or quality result validates the final combined build. Deeper fit/fidelity work remains a pre-ship recheck, not a new blocker on enabling the flags.
