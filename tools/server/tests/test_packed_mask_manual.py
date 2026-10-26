#!/usr/bin/env python3
"""Local packed-mask identity, allocation and bracket checks under the GPU lock."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request


def request(port, path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as response:
        return json.load(response)


def distinct5(items):
    if len(items) < 5:
        return None
    return len({tuple(items[i:i + 5]) for i in range(len(items) - 4)}) / (len(items) - 4)


class Server:
    def __init__(self, args, tag, packed, kv, ubatch):
        self.args, self.tag = args, tag
        self.log_path = args.out / f"{tag}.log"
        self.log = self.log_path.open("w")
        self.samples, self.exceeded = [], False
        self.done = threading.Event()
        cmd = [args.server if packed else args.base_server, "-m", args.model,
               "--host", "127.0.0.1", "--port", str(args.port), "-c", str(args.context),
               "-b", "2048", "-ub", str(ubatch), "-t", "6", "-tb", "6", "-ngl", str(args.ngl),
               "-fa", "on", "-fit", "off", "-ctk", kv[0], "-ctv", kv[1], "--parallel", "1",
               "--no-webui", "--no-warmup", "--cache-ram", "0", "--no-cache-disk",
               "--spec-type", "none", "--no-spec-mb-mtp", "--log-verbosity", "4",
               "--ctx-checkpoints", "0", "--slot-save-path", str(args.out)]
        if kv[0].startswith("sjkvarn"):
            cmd += ["--sjkvarn-sink", "128", "--sjkvarn-sink-type", "f16",
                    "--sjkvarn-staging-type", "tq6_0", "--sjkvarn-tail", "4096",
                    "--sjkvarn-tail-max", "8192"]
        if args.fit12:
            for flag in ("-t", "-tb", "--ctx-checkpoints"):
                index = cmd.index(flag)
                del cmd[index:index + 2]
            for flag in ("--no-warmup", "--no-cache-disk"):
                cmd.remove(flag)
            cmd[cmd.index("-b") + 1] = "4096"
            cmd[cmd.index("--spec-type") + 1] = "draft-mtp"
            cmd += ["--sjkvarn-body-type", "sjkvarn4t", "--cache-type-s", "q8_0",
                    "--spec-draft-n-max", "4", "--spec-draft-p-min", "0",
                    "--spec-draft-vocab-map", "auto:65536", "--spec-draft-type-k", "q8_0",
                    "--spec-draft-type-v", "q8_0", "--spec-draft-window", "8192",
                    "--temp", "1.0", "--top-k", "20", "--top-p", "0.95", "--min-p", "0",
                    "--no-context-shift", "--jinja"]
        env = os.environ.copy()
        env["LLAMA_KQ_MASK_PACKED"] = "1" if packed else "0"
        env["GGML_CUDA_FATTN_ALLOC_LOG"] = "1"
        if args.fit12:
            env["LLAMA_MTP_DRAFT_COMPUTE_LEAN"] = "1"
        self.cmd = cmd
        self.session = f"mask175-{os.getpid()}-{tag}" if args.nsys else None
        if self.session:
            cmd = ["nsys", "launch", "--session-new=" + self.session,
                   "--trace=cuda", "--cuda-graph-trace=graph"] + cmd
        self.proc = subprocess.Popen(cmd, env=env, stdout=self.log, stderr=subprocess.STDOUT)
        self.thread = threading.Thread(target=self.monitor, daemon=True)
        self.thread.start()
        try:
            for _ in range(480):
                if self.proc.poll() is not None:
                    raise RuntimeError(f"server exited: {self.log_path}")
                try:
                    request(args.port, "/health")
                    break
                except (OSError, urllib.error.HTTPError):
                    time.sleep(0.25)
            else:
                raise RuntimeError(f"server did not start: {self.log_path}")
            if self.session:
                subprocess.run(["nsys", "start", "--session=" + self.session, "--sample=none",
                                "--cpuctxsw=none", "--output=" + str(args.out / tag)], check=True, timeout=60)
        except BaseException:
            self.stop()
            raise

    def monitor(self):
        while not self.done.is_set():
            rss = card = proc = 0
            available = None
            try:
                status = Path(f"/proc/{self.proc.pid}/status").read_text()
                rss = int(re.search(r"VmRSS:\s+(\d+)", status)[1])/1024
                card = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                           "--format=csv,noheader,nounits"], text=True).strip())
                rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                       "--format=csv,noheader,nounits"], text=True).splitlines()
                proc = next((int(row.split(",")[1]) for row in rows
                             if row.split(",")[0].strip() == str(self.proc.pid)), 0)
                available = int(re.search(r"MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text())[1])
                if card > 23552 or available < 5*1024*1024:
                    self.exceeded = True
                    self.proc.terminate()
            except (OSError, TypeError, ValueError, subprocess.CalledProcessError):
                pass
            self.samples.append([time.time(), rss, card, proc,
                                 None if available is None else available/(1024*1024)])
            self.done.wait(0.5)

    def complete(self, tokens, count=128, temperature=0, cache=False):
        result = request(self.args.port, "/completion", {
            "prompt": tokens, "n_predict": count, "temperature": temperature,
            "seed": self.args.seed, "top_k": 1 if temperature == 0 else (20 if self.args.fit12 else 40),
            "top_p": 0.95, "min_p": 0, "ignore_eos": True, "cache_prompt": cache, "return_tokens": True, "id_slot": 0})
        assert result["tokens_predicted"] == count
        assert len(result["tokens"]) == count, "incomplete token IDs"
        response = {"hash": hashlib.sha256(json.dumps(result["tokens"]).encode()).hexdigest(),
                    "timings": result["timings"]}
        if count == 128:
            response["tokens"] = result["tokens"]
        if count >= 5000 and temperature == 1:
            quality = {"distinct5_token_ratio": distinct5(result["tokens"]),
                       "distinct5_word_ratio": distinct5(result["content"].split())}
            generation_path = self.args.out / f"{self.tag}.generation.json"
            generation_path.write_text(json.dumps({"tokens": result["tokens"],
                                       "content": result["content"], **quality}, indent=2))
            response.update(quality=quality, generation_file=generation_path.name)
        return response

    def stop(self):
        if self.session:
            subprocess.run(["nsys", "stop", "--session=" + self.session], check=True, timeout=60)
            # An exited target has no active session; keep the original failure and save resource samples.
            subprocess.run(["nsys", "shutdown", "--session=" + self.session], check=False, timeout=60)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.done.set()
        self.thread.join(timeout=5)
        self.log.close()
        (self.args.out / f"{self.tag}.samples.json").write_text(json.dumps(self.samples))
        assert not self.exceeded, "VRAM or available RAM limit exceeded"

    def summary(self):
        return {"command": self.cmd, "rss_peak_mib": max((x[1] for x in self.samples), default=0),
                "card_peak_mib": max((x[2] for x in self.samples), default=0),
                "process_peak_mib": max((x[3] for x in self.samples), default=0)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--base-server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fit12", action="store_true", help="use the production 12 GB MTP command")
    parser.add_argument("--nsys", action="store_true", help="profile each arm with graph tracing")
    parser.add_argument("--port", type=int, default=8275)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--depth", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=175)
    parser.add_argument("--ngl", type=int, default=99)
    parser.add_argument("--ubatch", type=int, default=512)
    parser.add_argument("--kv", default="sjkvarn3,sjkvarn3")
    parser.add_argument("--mode", choices=["identity", "vram", "deep", "speed"], default="identity")
    parser.add_argument("--repetitions", type=int, default=3, help="complete base/new/base speed brackets")
    parser.add_argument("--repeat-index", type=int, default=0, help="first bracket index in a split run")
    parser.add_argument("--generation-request", type=Path, help="full coding task in a chat template for speed output")
    args = parser.parse_args()
    assert args.repetitions > 0 and args.repeat_index >= 0, "invalid bracket range"
    assert args.mode == "speed" or (args.repetitions == 3 and args.repeat_index == 0), \
        "bracket options apply only to speed mode"
    assert os.environ.get("PARENT_HOLDS_GPU_LOCK") == "1", "run inside the GPU lock"
    args.out.mkdir(parents=True, exist_ok=True)
    kv = tuple(args.kv.split(","))
    assert len(kv) == 2
    report = {"mode": args.mode, "seed": args.seed, "cells": []}
    if args.mode == "speed":
        report.update(repetitions=args.repetitions, repeat_index=args.repeat_index)
    generation_request = None
    if args.generation_request:
        assert args.mode in ("deep", "speed"), "the alternate task requires long generation"
        generation_request = args.generation_request.read_text()
        (args.out / "generation-request.txt").write_text(generation_request)
        report["generation_request_sha256"] = hashlib.sha256(generation_request.encode()).hexdigest()
    tokens = None
    arms = [(f"{tag}-r{r}", packed) for r in range(args.repeat_index, args.repeat_index + args.repetitions)
            for tag, packed in [("base1", False), ("new", True), ("base2", False)]] if args.mode == "speed" else [("base", False), ("new", True)]
    for tag, packed in arms:
        server = Server(args, tag, packed, kv, args.ubatch)
        try:
            if tokens is None:
                seed = request(args.port, "/tokenize", {"content": "def lookup(cache, key):\n    return cache.get(key)\n", "add_special": True})["tokens"]
                tail = request(args.port, "/tokenize", {"content": "\nWrite tests for lookup."})["tokens"]
                tokens = (seed*((args.depth + len(seed) - 1)//len(seed)))[:args.depth] + tail
                if generation_request:
                    marker = "__PACKED_MASK_BODY__"
                    template = request(args.port, "/apply-template", {"messages": [{"role": "user", "content": marker}]})["prompt"]
                    header, found, footer = template.partition(marker)
                    assert found, "template omitted the user message"
                    head = request(args.port, "/tokenize", {"content": header, "add_special": True})["tokens"]
                    seed = request(args.port, "/tokenize", {"content": "def lookup(cache, key):\n    return cache.get(key)\n"})["tokens"]
                    length = args.depth - len(head)
                    tail = request(args.port, "/tokenize", {"content": generation_request + footer})["tokens"]
                    tokens = head + (seed*((length + len(seed) - 1)//len(seed)))[:length] + tail
                if args.mode in ("deep", "speed"):
                    assert len(tokens) + 5120 <= args.context, "speed prompt and output exceed context"
                    report["generation_prompt_hash"] = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
            response = server.complete(tokens, count=5120 if args.mode in ("deep", "speed") else 128,
                                       temperature=1 if args.mode in ("deep", "speed") else 0)
            cell = {"tag": tag, **response}
            if args.mode == "identity":
                saved = request(args.port, "/slots/0?action=save", {"filename": "mask-cache.bin"})
                request(args.port, "/slots/0?action=erase", {})
                restored = request(args.port, "/slots/0?action=restore", {"filename": "mask-cache.bin"})
                continuation = tokens + response["tokens"] + request(args.port, "/tokenize", {"content": "\nUse an empty dictionary."})["tokens"]
                warm = server.complete(continuation, cache=True)
                cold = server.complete(continuation, cache=False)
                cell.update(save=saved, restore=restored, continuation_hash=warm["hash"], warm=warm, cold=cold)
                (args.out / f"{tag}.cache.json").write_text(json.dumps(cell, indent=2))
                assert warm["timings"]["cache_n"] == saved["n_saved"] > 0, "restored state was not reused"
                assert cold["timings"]["cache_n"] == 0, "cold continuation reused cached state"
                assert warm["hash"] == cold["hash"], f"{tag}: prompt-cache identity failed"
                rollback_prompt = continuation[:-4]
                rollback = server.complete(rollback_prompt, cache=True)
                rollback_cold = server.complete(rollback_prompt, cache=False)
                cell.update(rollback=rollback, rollback_cold=rollback_cold)
                (args.out / f"{tag}.cache.json").write_text(json.dumps(cell, indent=2))
                assert rollback["hash"] == rollback_cold["hash"], f"{tag}: rollback identity failed"
                assert rollback["timings"]["cache_n"] > 0, "rollback discarded the complete prefix"
                assert restored["n_restored"] == saved["n_saved"]
                assert restored["n_read"] == saved["n_written"] > 0
            if packed:
                assert re.search(r"packed KQ mask\s*= on", server.log_path.read_text()), "packed graph input was not enabled"
        finally:
            server.stop()
        if args.mode in ("deep", "speed"):
            cell["mask_expand_logged"] = "mask_packed_expand" in server.log_path.read_text()
        cell.update(server.summary())
        report["cells"].append(cell)
        (args.out / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(cell), flush=True)
    if args.mode not in ("deep", "speed"):
        assert len({cell["hash"] for cell in report["cells"]}) == 1, "packed/base token identity failed"
        if args.mode == "identity":
            assert len({cell["continuation_hash"] for cell in report["cells"]}) == 1
            assert len({cell["rollback"]["hash"] for cell in report["cells"]}) == 1
    else:
        assert len({cell["hash"] for cell in report["cells"]}) == 1, "packed/base token identity failed"
        assert all(cell["quality"]["distinct5_word_ratio"] is not None and
                   cell["quality"]["distinct5_word_ratio"] >= 0.5 for cell in report["cells"]), \
            "collapsed generation cell (word distinct5 below 50%)"
    for path in args.out.glob("mask-cache.bin*"):
        path.unlink()
    print("PASS", flush=True)


if __name__ == "__main__":
    main()
