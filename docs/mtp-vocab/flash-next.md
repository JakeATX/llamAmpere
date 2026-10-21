# Flash-Next 64K draft vocabulary

The tokenizer is shared with Qwen3.8-27B; proposal rankings are model-specific. The preferred qwen4exp list is calibrated on Flash-Next output and draft candidates. Qwen3.8-27B bitmaps and the legacy qwen4exp 32K list are unchanged.

Selection uses three 4096-token coding/agentic/RAG generations, the known 100K/5120 failure case and its full-head proposal candidates, plus a 2% uniform prior over the old 64K list. The 100K timed regression therefore overlaps calibration: it is a known-case regression, not held-out evidence. A separate 2048-token LCB completion measures held-out final-token coverage only. No general acceptance claim follows from that coverage.

The list adds 234 IDs and removes 234, keeping exactly 65536 rows. Reproduce with scripts/mtp-vocab-shortlist.py; corpus manifest, exact command, coverage and matched-head evidence are recorded in SHIP_20261019/SHORTLIST/{list-calibration.json, head-audit-result.json,rebuild_when_idle.py}.

Calibration masses are coding/agentic/RAG/output 0.15 each, full-head draft candidates 0.38 and the old-list prior 0.02. The new list covers every observed token in those five Flash-Next corpora. The separate LCB completion has 2046/2048 tokens covered by both old and new lists.

On the adjacent 100K-prompt / 5120-output regression pair, map none and revised auto both accepted 3544/5370 draft tokens (0.659963), with identical output tokens. Revised auto measured 46.61 tok/s versus 45.80 for none (+1.77%). Native draft cost was 5.86 versus 6.80 ms/round at width 3 and 7.16 versus 8.42 at width 4. This validates the calibrated case; held-out acceptance remains unmeasured.
