#!/usr/bin/env python3
"""Model file checksums and margin-filtered greedy self-check prompts.

  crc     FILE.gguf [-o FILE.crc.json]       per-tensor CRC-32 of a GGUF (plus a header CRC), written as JSON
  verify  FILE.gguf [--crc FILE.crc.json]    recompute and compare; exit 1 naming each tensor that differs
  filter  --url URL --prompts P.jsonl -o S.json [--margin 0.11] [--n-predict 64]
          run each prompt greedy on a running llama-server and keep only the prompts whose top-1 logprob leads the
          top-2 by >= margin at every generated token. Those outputs cannot flip under cross-card or cross-OS
          arithmetic differences, so a gate or a first-run check that compares ids has no false failures.
          Writes the kept prompts with their token ids.
  check   --url URL --selfcheck S.json       run the kept prompts greedy and compare the ids; exit 1 on a mismatch.
          The first-run GPU self-check of a package: a wrong kernel, a damaged file or a bad driver shows up as a
          different id, not as a vague quality drop.

Prompts file: one JSON object per line, {"prompt": "..."} or {"messages": [...]} (chat), optional "name".
The log-probabilities come from the server before sampling (n_probs, post_sampling_probs false), so the top-1 to
top-2 logprob gap is the logit gap.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
import zlib
from pathlib import Path

CHUNK = 1 << 24


def crc_file_range(f, offset: int, n: int) -> int:
    f.seek(offset)
    c = 0
    while n > 0:
        b = f.read(min(CHUNK, n))
        if not b:
            raise IOError(f"short read at {offset}")
        c = zlib.crc32(b, c)
        n -= len(b)
    return c & 0xFFFFFFFF


def tensor_crcs(path: str) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gguf-py"))
    from gguf import GGUFReader  # type: ignore

    r = GGUFReader(path, "r")
    tensors = sorted(r.tensors, key=lambda t: t.data_offset)
    out = {"file": os.path.basename(path), "size": os.path.getsize(path), "tensors": {}}
    with open(path, "rb") as f:
        first = tensors[0].data_offset if tensors else os.path.getsize(path)
        out["header_crc32"] = f"{crc_file_range(f, 0, first):08x}"
        for t in tensors:
            out["tensors"][t.name] = {
                "offset": int(t.data_offset),
                "n_bytes": int(t.n_bytes),
                "crc32": f"{crc_file_range(f, int(t.data_offset), int(t.n_bytes)):08x}",
            }
    return out


def cmd_crc(a) -> int:
    res = tensor_crcs(a.file)
    dst = a.output or a.file + ".crc.json"
    with open(dst, "w") as f:
        json.dump(res, f, indent=1)
    print(f"{len(res['tensors'])} tensors, header {res['header_crc32']} -> {dst}")
    return 0


def cmd_verify(a) -> int:
    ref_path = a.crc or a.file + ".crc.json"
    with open(ref_path) as f:
        ref = json.load(f)
    if os.path.getsize(a.file) != ref["size"]:
        print(f"FAIL size {os.path.getsize(a.file)} != {ref['size']} (truncated or wrong file)")
        return 1
    cur = tensor_crcs(a.file)
    bad = []
    if cur["header_crc32"] != ref["header_crc32"]:
        bad.append("(header)")
    for name, e in ref["tensors"].items():
        c = cur["tensors"].get(name)
        if c is None or c["crc32"] != e["crc32"]:
            bad.append(name)
    extra = sorted(set(cur["tensors"]) - set(ref["tensors"]))
    bad += [f"(unexpected) {n}" for n in extra]
    if bad:
        print(f"FAIL {len(bad)} of {len(ref['tensors'])} tensors differ:")
        for n in bad:
            print("  " + n)
        return 1
    print(f"OK {len(ref['tensors'])} tensors match")
    return 0


def post(url: str, path: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run_greedy(url: str, item: dict, n_predict: int, timeout: float) -> list[tuple[int, float]]:
    """ids with the top-1 minus top-2 logprob gap of each generated token"""
    body = {"n_predict": n_predict, "temperature": 0.0, "top_k": 1, "seed": 0, "n_probs": 2,
            "post_sampling_probs": False, "cache_prompt": False, "stream": False}
    if "messages" in item:
        body["messages"] = item["messages"]
        res = post(url, "/apply-template", {"messages": item["messages"]}, timeout)
        body.pop("messages")
        body["prompt"] = res["prompt"]
    else:
        body["prompt"] = item["prompt"]
    res = post(url, "/completion", body, timeout)
    out = []
    for p in res.get("completion_probabilities", []):
        top = p.get("top_logprobs", [])
        gap = float("inf") if len(top) < 2 else float(top[0]["logprob"]) - float(top[1]["logprob"])
        out.append((int(p["id"]), gap))
    return out


def cmd_filter(a) -> int:
    kept = []
    with open(a.prompts) as f:
        items = [json.loads(l) for l in f if l.strip()]
    for i, it in enumerate(items):
        toks = run_greedy(a.url, it, a.n_predict, a.timeout)
        min_gap = min((g for _, g in toks), default=0.0)
        ok = len(toks) > 0 and min_gap >= a.margin
        name = it.get("name", f"p{i}")
        print(f"{'keep' if ok else 'drop'} {name}: {len(toks)} tokens, min gap {min_gap:.3f}")
        if ok:
            kept.append({**it, "name": name, "ids": [t for t, _ in toks], "min_gap": round(min_gap, 4)})
    with open(a.output, "w") as f:
        json.dump({"margin": a.margin, "n_predict": a.n_predict, "prompts": kept}, f, indent=1)
    print(f"kept {len(kept)} of {len(items)} -> {a.output}")
    return 0 if kept else 1


def cmd_check(a) -> int:
    with open(a.selfcheck) as f:
        sc = json.load(f)
    bad = 0
    for it in sc["prompts"]:
        ids = [t for t, _ in run_greedy(a.url, it, sc["n_predict"], a.timeout)]
        n = next((k for k, (x, y) in enumerate(zip(ids, it["ids"])) if x != y), None)
        if n is None and len(ids) != len(it["ids"]):
            n = min(len(ids), len(it["ids"]))
        if n is None:
            print(f"OK   {it['name']}")
        else:
            bad += 1
            print(f"FAIL {it['name']}: first differing token {n}")
    print(f"{len(sc['prompts']) - bad} of {len(sc['prompts'])} self-check prompts match")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("crc"); p.add_argument("file"); p.add_argument("-o", "--output")
    p = sub.add_parser("verify"); p.add_argument("file"); p.add_argument("--crc")
    p = sub.add_parser("filter")
    p.add_argument("--url", default="http://127.0.0.1:8080"); p.add_argument("--prompts", required=True)
    p.add_argument("-o", "--output", required=True); p.add_argument("--margin", type=float, default=0.11)
    p.add_argument("--n-predict", type=int, default=64); p.add_argument("--timeout", type=float, default=600)
    p = sub.add_parser("check")
    p.add_argument("--url", default="http://127.0.0.1:8080"); p.add_argument("--selfcheck", required=True)
    p.add_argument("--timeout", type=float, default=600)
    a = ap.parse_args()
    return {"crc": cmd_crc, "verify": cmd_verify, "filter": cmd_filter, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
