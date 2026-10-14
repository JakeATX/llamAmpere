#!/usr/bin/env python3
"""CPU regressions for fit-half isolation, exact recipes, and paired KL inputs."""
import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import kl_plan


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="aireview-kl-plan-")
        self.directory = Path(self.tmp.name)
        self.cut = mock.patch.object(kl_plan, "CUT", {"gate": 0})
        self.block = mock.patch.object(kl_plan, "BLK", 2)
        self.cut.start(); self.block.start()

    def tearDown(self):
        self.block.stop(); self.cut.stop(); self.tmp.cleanup()

    def write_arm(self, arm, values, edits=None):
        path = self.directory / f"kl.{arm}.gate.csv"
        rows = [{"pos": i, "tok": 100 + i, "kld": v, "nll_base": 3.0} for i, v in enumerate(values)]
        if edits:
            edits(rows)
        with path.open("w") as out:
            writer = csv.DictWriter(out, fieldnames=["pos", "tok", "kld", "nll_base"])
            writer.writeheader(); writer.writerows(rows)
        return path

    def scale_choices(self, even_value, odd_value):
        self.write_arm("AIB_SC5R", [1.0] * 8)
        self.write_arm("AIB_SC5S6", [even_value] * 2 + [odd_value] * 2 + [even_value] * 2 + [odd_value] * 2)
        head = {"name": "output.weight", "K": 1024, "M": 128, "n": 131072, "k4": 5, "group": None, "chunk": None}
        results = {}
        for half in ("all", "even", "odd"):
            fit = {"rows": [head], "half": half, "head": {6: -0.2}, "A": {}, "f": {"up": 1.0}}
            with contextlib.redirect_stdout(io.StringIO()):
                results[half] = kl_plan.k6_decide(str(self.directory), fit, B=200)
        return results

    def test_even_scale_choice_excludes_odd_data(self):
        choices = self.scale_choices(0.6, 1.4)
        self.assertEqual(choices["all"]["mode"], "row")
        self.assertEqual(choices["even"]["mode"], "span")
        self.assertEqual(choices["odd"]["mode"], "row")
        self.assertAlmostEqual(choices["even"]["cand"]["span"]["ratio"], 0.6)
        self.assertAlmostEqual(choices["even"]["gain"]["output.weight"], 0.4)

    def test_odd_scale_choice_excludes_even_data(self):
        choices = self.scale_choices(1.4, 0.6)
        self.assertEqual(choices["all"]["mode"], "row")
        self.assertEqual(choices["even"]["mode"], "row")
        self.assertEqual(choices["odd"]["mode"], "span")
        self.assertAlmostEqual(choices["odd"]["gain"]["output.weight"], 0.4)

    def test_exact_list_copies_actual_tiers_and_resolves_head(self):
        reference = self.directory / "reference.list"
        output = self.directory / "exact.list"
        reference.write_text("blk.0.ffn_gate.weight ai5un\noutput.weight keep\n")
        shapes = {"blk.0.ffn_gate.weight": (1024, 128), "output.weight": (1024, 256)}
        with mock.patch.object(kl_plan, "plan_template", return_value=(shapes, 1)):
            with contextlib.redirect_stdout(io.StringIO()):
                kl_plan.cmd_exact_list(reference, output, 6)
        self.assertIn("blk.0.ffn_gate.weight ai5u\n", output.read_text())
        self.assertIn("output.weight ai6u\n", output.read_text())

    def test_exact_list_rejects_unresolved_or_contradictory_head(self):
        reference = self.directory / "reference.list"
        output = self.directory / "exact.list"
        reference.write_text("output.weight keep\n")
        with self.assertRaises(ValueError):
            kl_plan.cmd_exact_list(reference, output, None)
        reference.write_text("output.weight ai5u\n")
        with self.assertRaises(ValueError):
            kl_plan.cmd_exact_list(reference, output, 6)
        self.assertFalse(output.exists())

    def test_exact_list_rejects_missing_body_tensor(self):
        reference = self.directory / "reference.list"
        reference.write_text("output.weight keep\n")
        shapes = {"blk.0.ffn_gate.weight": (1024, 128), "output.weight": (1024, 256)}
        with mock.patch.object(kl_plan, "plan_template", return_value=(shapes, 1)):
            with self.assertRaisesRegex(ValueError, "missing blk.0.ffn_gate"):
                kl_plan.cmd_exact_list(reference, self.directory / "exact.list", 6)

    def test_loader_rejects_duplicate_gap_and_nonfinite(self):
        for edit in (lambda rows: rows.append(rows[0].copy()), lambda rows: rows.pop(3),
                     lambda rows: rows[3].update(kld=float("nan"))):
            with self.subTest(edit=edit):
                self.write_arm("bad", [1.0] * 8, edit)
                with self.assertRaises(ValueError):
                    kl_plan.load_arm(str(self.directory), "bad")

    def test_pairing_rejects_token_reference_and_tail_changes(self):
        self.write_arm("base", [1.0] * 8)
        base = kl_plan.load_arm(str(self.directory), "base")
        for edit in (lambda rows: rows[3].update(tok=999), lambda rows: rows[3].update(nll_base=3.1),
                     lambda rows: rows.append({"pos": 8, "tok": 108, "kld": 1.0, "nll_base": 3.0})):
            with self.subTest(edit=edit):
                self.write_arm("candidate", [0.9] * 8, edit)
                candidate = kl_plan.load_arm(str(self.directory), "candidate")
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    kl_plan.assert_paired(base, candidate)

    def test_pairing_accepts_changed_candidate_kl(self):
        self.write_arm("base", [1.0] * 8)
        self.write_arm("candidate", [0.9] * 8)
        base = kl_plan.load_arm(str(self.directory), "base")
        candidate = kl_plan.load_arm(str(self.directory), "candidate")
        kl_plan.assert_paired(base, candidate)
        self.assertAlmostEqual(kl_plan.dk_se(candidate, base, "even", B=20)[0], -0.1)


if __name__ == "__main__":
    unittest.main()
