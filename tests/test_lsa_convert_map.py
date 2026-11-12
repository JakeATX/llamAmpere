#!/usr/bin/env python3
"""Calibration dump conversion: visible keys, rotary padding and GGUF map layout."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "lsa_convert_map.py"
spec = importlib.util.spec_from_file_location("lsa_convert_map", SCRIPT)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class TestPcaMap(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.prefix = Path(self.tmp.name) / "calibration"
        self.keys = np.zeros((96, 8), dtype="<f4")
        self.keys[:64, 2:4] = np.tile([[3, 0], [0, 2], [-3, 0], [0, -2]], (16, 1))
        self.keys[:, [0, 1, 4, 5]] = 1e5  # Rotary rows must not enter the fit.
        self.keys[64:, 6:8] = 1e6  # Invisible padding must not enter the fit.
        self.mask = np.concatenate([np.zeros(64), np.full(32, -np.inf)]).astype("<f4")
        self.index = "3 k 8 96 1 1 ok\n3 mask 96 1 1 1 ok\n"
        self.write_dump()

    def write_dump(self):
        Path(f"{self.prefix}.index.txt").write_text(self.index)
        self.keys.tofile(f"{self.prefix}.L3.k.f32")
        self.mask.tofile(f"{self.prefix}.L3.mask.f32")

    def fit(self):
        return converter.read_pca_dump(self.prefix, 4, 2, 2, 2, 64)

    def test_subspace_and_visible_keys(self):
        tensors, sha = self.fit()
        basis = tensors["3.W_K_idx"]
        np.testing.assert_allclose(basis @ basis.T, np.diag([1, 1, 0, 0]), atol=1e-6)
        np.testing.assert_array_equal(basis, tensors["3.W_Q_idx"])
        self.assertEqual(len(sha), 64)
        self.keys[0, 2] += 1
        self.write_dump()
        self.assertNotEqual(sha, self.fit()[1])

    def test_reject_bad_dump(self):
        with self.subTest("truncated"):
            Path(f"{self.prefix}.L3.k.f32").write_bytes(b"short")
            with self.assertRaises(ValueError):
                self.fit()
        self.write_dump()
        with self.subTest("nonfinite"):
            self.keys[0, 2] = np.nan
            self.write_dump()
            with self.assertRaises(ValueError):
                self.fit()
        self.keys[0, 2] = 3
        self.write_dump()
        with self.subTest("duplicate"):
            Path(f"{self.prefix}.index.txt").write_text(self.index * 2)
            with self.assertRaises(ValueError):
                self.fit()

    def test_gguf_padding(self):
        out = Path(self.tmp.name) / "map.gguf"
        subprocess.run([sys.executable, str(SCRIPT), str(self.prefix), str(out), "--pca-dump", "--head-dim", "4",
                        "--rotary", "2", "--n-head-kv", "2", "--index-dim", "2", "--pca-samples", "64"], check=True)
        reader = converter.gguf.GGUFReader(str(out))
        self.assertEqual({t.name for t in reader.tensors}, {"lsa.3.wq", "lsa.3.wk"})
        for tensor in reader.tensors:
            np.testing.assert_array_equal(tensor.shape, [8, 2])
            np.testing.assert_array_equal(tensor.data[:, [0, 1, 4, 5]], 0)
            np.testing.assert_allclose(tensor.data @ tensor.data.T, np.eye(2), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
