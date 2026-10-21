# Confidence-driven MTP depth replay

`LLAMA_SPEC_CONFLOG=/path/run.jsonl` enables buffered, append-only server telemetry. Unset or empty leaves confidence collection and file access off. This also works through the CLI's shared server context. No depth controller changes.

Each process/load writes a header with `event`, `model`, `n_max`, `n_min_adaptive`, and `spec_type`. Each logical verify round has `slot`, `request`, `n_past`, `depth`, `drafted`, `accepted`, `draft_p1`, `draft_p2`, `target_p1`, and `t_round_ms`. `n_past` is the position before the last sampled token is added. `depth` is the effective cap, including equal-depth batching. The draft arrays have `drafted` entries. The target array has `drafted + 1` entries, including the bonus row. Only its first `accepted + 1` entries describe generated tokens with a correct prefix.

Sampling probabilities are reused for the draft. A logits-only draft computes top-2 probabilities from the existing host row. Chained argmax drafting adds the second probability to its existing softmax output. Target confidence is the raw verify-logits softmax top-1, before request sampling transforms. Logging uses host verification because backend verification can omit raw logits. Unknown confidence is JSON `null`; the simulator rejects it instead of treating it as zero. N-gram drafters have no probability distribution and are not suitable fit inputs.

`emitted` records the actual output width. After a checkpoint replay, `committed_p1` holds the target confidences for that committed output, including any extra replay bonus. These optional fields preserve logged-depth throughput and history when a restore emits two bonus tokens.

Checkpoint restoration retains the original verify data. The replay is charged to that logical round instead of duplicating the observation. The timer starts before drafting and ends at the acceptance accounting hook. It includes verification, checkpoint replays, confidence extraction, and shared work with other slots. Use `--parallel 1` for timing fits. A mutex protects the shared file. The header counts toward the 64-line flush interval; normal destruction flushes the remaining buffer.

## Fit and evaluate

Python 3 and numpy are the only dependencies. Collect fixed depth 6 for dense models or 5 for sparse models, with the probability cutoff disabled. For MTP, `--spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-p-min 0` disables the adaptive controller. Also unset `LLAMA_SPEC_DEPTH_MIX` and `LLAMA_SPEC_ADAPT_COST` for these outcome logs.

```sh
python3 tools/cdepth/cdepth_sim.py fit --logs seed1.jsonl seed2.jsonl --out model.json
python3 tools/cdepth/cdepth_sim.py eval --logs seed3.jsonl --model model.json --range 3 6 --out summary.json
```

Each position gets a conditional logistic curve on the prior rolling target-confidence mean, draft top-1, and draft margin, plus an intercept. Positions after the first failed position are censored. Damped Newton steps use standardized features and a small ridge penalty. Constant labels use a smoothed intercept. The last 20% of each stream selects among 8-, 16-, and 32-token windows by held-out log loss. The selected window is refit on all training logs. Histories reset across files, headers, slots, and requests. Position discontinuities also reset history in older logs without request ids; current logs retain history across a context shift in the same request. Empty history starts at 0.5. Current-round target probabilities never enter its features.

Fixed-depth timing alone cannot identify `a + b * depth`. The fit then uses constant cost and emits a warning. For a useful throughput comparison, collect separate mixed-depth timing logs on the training seeds with `LLAMA_SPEC_DEPTH_MIX=3:4:5:6`, and pass `--cost-logs cost1.jsonl cost2.jsonl` to `fit`. The cost fit constrains both coefficients to be nonnegative. Treat extrapolation beyond the observed cost depths with care.

The table goes to stdout. JSON goes to `--out`, or follows the table if that option is omitted. Streak uses the existing production bounds 3..4 by default. `--streak-range 3 6` compares the same state machine over the wider range. Sparse replay uses `--range 2 5`.

Cdepth chooses the largest predicted accepted-draft-token count per fitted millisecond. Reported throughput also counts the replacement/bonus token.

Replay uses `min(logged accepted, chosen depth)` and includes one replacement/bonus token in throughput. At exactly the logged depth it also preserves any recorded extra replay bonus; smaller depths use the single-bonus estimate because their restore behavior is unknown. It never invents outcomes beyond actual drafted depth, even in a nominal fixed-max run with a shortened final round. Fixed policies skip unavailable rounds and report coverage; compare policies on equal cohorts. The oracle optimizes aggregate hindsight throughput under the fitted cost line, which bounds every policy evaluated on that same cohort.

This is an offline estimate at logged round starts. A different depth would change later round boundaries. Cdepth sees every logged draft confidence, including positions an incremental implementation might not compute. Logging adds overhead. A simulated gain over 1% needs coordinator validation before any on-device policy work.

## CPU checks

```sh
python3 tools/cdepth/check_conflog.py
OPENBLAS_NUM_THREADS=1 python3 -m unittest tools.cdepth.test_cdepth_sim -v
```

The tests compile host arithmetic and extracted writer methods with stubbed context calls. They do not load models, link CUDA, start servers, or start systemd units.
