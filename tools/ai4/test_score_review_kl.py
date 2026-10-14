#!/usr/bin/env python3
"""CPU tests of token/mask alignment and paired ratio statistics for the rental."""
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
from score_review_kl import load_scores, ratio_ci


class RentalScoreTests(unittest.TestCase):
    def test_mask_includes_first_generated_prediction(self):
        ids = np.arange(6, dtype=np.int32) + 100
        mask = np.array([0, 0, 1, 1, 1, 0], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "score.csv"
            rows = [{"pos": p, "tok": int(ids[p + 1]), "nll_base": 3.0, "kld": 0.2} for p in (2, 3, 4)]
            def write(items):
                with path.open("w") as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader(); writer.writerows(items)
            write(rows)
            identity, values = load_scores(path, ids, mask, 3)
            self.assertEqual(identity[0][:2], (2, 103))
            self.assertEqual(len(values), 3)
            for bad in (rows[1:], rows + [rows[0]], [dict(rows[0], tok=999), *rows[1:]],
                        [dict(rows[0], kld="nan"), *rows[1:]]):
                write(bad)
                with self.assertRaises(ValueError): load_scores(path, ids, mask, 3)

    def test_rejects_final_unpredicted_mask(self):
        with self.assertRaises(ValueError):
            load_scores("unused", np.array([1, 2]), np.array([0, 1]), 1)

    def test_constant_ratio_and_short_tail_are_retained(self):
        ref = {"coding": np.arange(1, 503, dtype=np.float64), "rag": np.ones(601)}
        cand = {k: v*0.9 for k, v in ref.items()}
        for width in (200, 500):
            result = ratio_ci(cand, ref, width, 4000, 714)
            self.assertAlmostEqual(result["ratio"], 0.9)
            for endpoint in result["ci95"]: self.assertAlmostEqual(endpoint, 0.9)

    def test_wrong_lengths_fail(self):
        with self.assertRaises(ValueError):
            ratio_ci({"rag": np.ones(2)}, {"rag": np.ones(3)}, 200, 20, 714)


if __name__ == "__main__":
    unittest.main()
