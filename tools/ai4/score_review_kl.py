#!/usr/bin/env python3
"""Score rental KL arms on an exact recorded generated-token mask, with paired block confidence intervals.

Manifest: {"trajectories": {domain: {"tokens_i32": path, "score_mask_u8": path}},
           "arms": {arm: {domain: csv_path}}, "comparisons": [[candidate, reference], ...]}.
Paths may be absolute or relative to the manifest. CSVs must contain only selected positions.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_scores(path, ids, mask, minimum):
    if len(mask) != len(ids) or np.any(mask > 1) or not len(mask) or mask[-1]:
        raise ValueError("score mask must match IDs, contain only 0/1, and exclude the final unpredicted position")
    expected = np.flatnonzero(mask)
    if len(expected) < minimum:
        raise ValueError(f"need >= {minimum} generated scores, got {len(expected)}")
    rows = {}
    with Path(path).open() as source:
        for row in csv.DictReader(source):
            pos = int(row["pos"])
            if pos in rows:
                raise ValueError(f"{path}: duplicate position {pos}")
            kld, nll = float(row["kld"]), float(row["nll_base"])
            if not np.isfinite(kld) or not np.isfinite(nll):
                raise ValueError(f"{path}: non-finite KL or reference NLL")
            rows[pos] = (int(row["tok"]), nll, kld)
    if sorted(rows) != expected.tolist():
        raise ValueError(f"{path}: positions disagree with generated-region score mask")
    if any(rows[int(pos)][0] != int(ids[pos + 1]) for pos in expected):
        raise ValueError(f"{path}: next-token IDs disagree with recorded trajectory")
    identity = tuple((int(pos), rows[int(pos)][0], rows[int(pos)][1]) for pos in expected)
    return identity, np.array([rows[int(pos)][2] for pos in expected], dtype=np.float64)


def ratio_ci(candidate, reference, width, resamples, seed):
    den = sum(v.sum() for v in reference.values())
    if not den > 0:
        raise ValueError("reference pooled KL must be positive")
    point = float(sum(v.sum() for v in candidate.values()) / den)
    rng = np.random.default_rng(seed)
    draws = np.zeros(resamples)
    denominator = np.zeros(resamples)
    for domain in reference:
        a, b = candidate[domain], reference[domain]
        if len(a) != len(b):
            raise ValueError("paired arms have different lengths")
        # Keep the final short block. Every generated token contributes to the point estimate and resampling.
        starts = np.arange(0, len(a), width)
        av, bv = np.add.reduceat(a, starts), np.add.reduceat(b, starts)
        indices = rng.integers(0, len(starts), size=(resamples, len(starts)))
        draws += av[indices].sum(axis=1)
        denominator += bv[indices].sum(axis=1)
    if np.any(denominator <= 0):
        raise ValueError("non-positive bootstrap reference KL")
    lo, hi = np.percentile(draws / denominator, [2.5, 97.5])
    return {"ratio": point, "ci95": [float(lo), float(hi)], "block_tokens": width}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--minimum-tokens", type=int, default=5000)
    ap.add_argument("--resamples", type=int, default=4000)
    args = ap.parse_args()
    if args.minimum_tokens < 1 or args.resamples < 1: ap.error("token count and resamples must be positive")
    spec = json.loads(args.manifest.read_text())
    base = args.manifest.resolve().parent
    def path(value):
        return base / value
    trajectories = {name: (np.fromfile(path(t["tokens_i32"]), dtype="<i4"),
                           np.fromfile(path(t["score_mask_u8"]), dtype=np.uint8))
                    for name, t in spec["trajectories"].items()}
    if not trajectories or not spec["arms"]: raise ValueError("manifest has no trajectories or arms")
    identity, scores = {}, {}
    for arm, files in spec["arms"].items():
        if files.keys() != trajectories.keys(): raise ValueError(f"{arm}: trajectory coverage mismatch")
        scores[arm] = {}
        for domain, (ids, mask) in trajectories.items():
            same, values = load_scores(path(files[domain]), ids, mask, args.minimum_tokens)
            if domain in identity and identity[domain] != same:
                raise ValueError(f"{arm}/{domain}: position/token/reference NLL mismatch")
            identity[domain] = same
            scores[arm][domain] = values
    result = {"generated_tokens": {d: len(v) for d, v in identity.items()},
              "resamples": args.resamples, "seed": 714, "comparisons": {}}
    result["mean_kl"] = {arm: float(sum(v.sum() for v in domains.values()) / sum(len(v) for v in domains.values()))
                         for arm, domains in scores.items()}
    for candidate, reference in spec["comparisons"]:
        result["comparisons"][f"{candidate}/{reference}"] = [
            ratio_ci(scores[candidate], scores[reference], width, args.resamples, 714) for width in (200, 500)]
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
