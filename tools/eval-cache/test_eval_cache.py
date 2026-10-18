#!/usr/bin/env python3
"""CPU HTTP regression. Export the model with test-mtp-state --save-model first."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("eval_cache", Path(__file__).with_name("llama-eval-cache.py"))
assert spec is not None and spec.loader is not None
eval_cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_cache)


@unittest.skipUnless(os.environ.get("LLAMA_SERVER_BIN_PATH") and os.environ.get("LLAMA_TEST_MTP_MODEL_FILE"),
                     "set LLAMA_SERVER_BIN_PATH and LLAMA_TEST_MTP_MODEL_FILE to local CPU fixtures")
class EvalCacheHTTPTest(unittest.TestCase):
    def test_legacy_read_only_falls_back_and_build_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(os.environ.get("EVAL_CACHE_TEST_OUTPUT", tmp))
            root.mkdir(parents=True, exist_ok=True)
            folder = root / "cache"
            self.assertFalse(folder.exists(), "use a new EVAL_CACHE_TEST_OUTPUT directory for each run")
            args = ["-m", os.environ["LLAMA_TEST_MTP_MODEL_FILE"], "--device", "none", "-ngl", "0", "-ngld", "0",
                    "-c", "512", "-b", "64", "-ub", "64", "-t", "2", "-tb", "2", "--parallel", "1",
                    "--cache-ram", "0", "--no-cache-disk", "-ctk", "q8_0", "-ctv", "q8_0"]
            request = {"prompt": [5 + i % 24 for i in range(96)], "n_predict": 16, "temperature": 0,
                       "return_tokens": True, "ignore_eos": True, "cache_prompt": True}

            def start(name, extra):
                server = eval_cache.Server(os.environ["LLAMA_SERVER_BIN_PATH"], args, root / f"{name}.log", extra)
                self.addCleanup(server.stop)
                return server

            fresh = start("fresh", [])
            try:
                fresh.wait_ready(60)
                reference = fresh.post("/completion", request, timeout=60)
            finally:
                fresh.stop()

            builder = start("builder", ["--eval-cache", str(folder), "--eval-cache-build"])
            try:
                builder.wait_ready(60)
                builder.post("/completion", dict(request, n_predict=1, eval_cache_save_at=len(request["prompt"])), timeout=60)
            finally:
                builder.stop()

            index_path = folder / "eval-cache.json"
            index = json.loads(index_path.read_text())
            self.assertTrue(index["entries"])
            valid = start("valid", ["--eval-cache", str(folder)])
            try:
                valid.wait_ready(60)
                cached = valid.post("/completion", request, timeout=60)
            finally:
                valid.stop()
            self.assertEqual(cached["tokens"], reference["tokens"])
            self.assertEqual(cached["content"], reference["content"])
            self.assertIn("eval cache: restored entry", (root / "valid.log").read_text())
            self.assertLess(cached["timings"]["prompt_n"], reference["timings"]["prompt_n"])

            index["format_version"] = 1
            index_path.write_text(json.dumps(index) + "\n")
            for path in (folder / "entries").glob("*.evh"):
                data = path.read_bytes()
                path.write_bytes(data[:4] + struct.pack("<I", 1) + data[8:])

            def snapshot():
                return {str(path.relative_to(folder)): (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
                        for path in folder.rglob("*") if path.is_file()}

            before = snapshot()
            legacy = start("legacy", ["--eval-cache", str(folder)])
            try:
                legacy.wait_ready(60)
                result = legacy.post("/completion", request, timeout=60)
                legacy.post("/completion", dict(request, cache_prompt=False, eval_cache_save_at=len(request["prompt"])), timeout=60)
            finally:
                legacy.stop()
            self.assertEqual(result["tokens"], reference["tokens"])
            self.assertEqual(result["content"], reference["content"])
            self.assertEqual(result["timings"]["prompt_n"], len(request["prompt"]))
            self.assertEqual(result["timings"].get("cache_n", 0), 0)
            log = (root / "legacy.log").read_text()
            self.assertIn("has format version 1, this server reads version 2", log)
            self.assertIn("skipping read-only cache, using fresh prefill", log)
            self.assertNotIn("eval cache: restored entry", log)
            self.assertEqual(snapshot(), before)

            refused = start("refused", ["--eval-cache", str(folder), "--eval-cache-build"])
            try:
                self.assertEqual(refused.proc.wait(timeout=60), 1)
            finally:
                refused.stop()
            self.assertIn("eval cache: refusing to start", (root / "refused.log").read_text())
            self.assertIn("has format version 1", (root / "refused.log").read_text())
            self.assertEqual(snapshot(), before)
            (root / "result.json").write_text(json.dumps({"fresh": reference, "valid": cached, "legacy": result,
                                                        "files_before": before, "files_after": snapshot()}, indent=2) + "\n")
            print(f"valid prompt_n={cached['timings']['prompt_n']}; legacy prompt_n={result['timings']['prompt_n']}; "
                  f"fresh prompt_n={reference['timings']['prompt_n']}; outputs match; legacy files unchanged")


if __name__ == "__main__":
    unittest.main()
