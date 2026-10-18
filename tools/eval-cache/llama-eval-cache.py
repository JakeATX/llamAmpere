#!/usr/bin/env python3
"""Build, check and list eval-cache folders for llama-server --eval-cache DIR.

An eval re-sends the same long prompts. `build` runs every prompt of a JSONL file through llama-server once and
stores the prefilled state, so a server started with `--eval-cache DIR` (same model, same KV flags) loads it from
disk instead of prefilling it again. Only the part of a prompt that is not stored is prefilled.

  build   llama-eval-cache.py build  --dir DIR --prompts prompts.jsonl [--server PATH] -- <llama-server flags>
  verify  llama-eval-cache.py verify --dir DIR [--identity N --prompts prompts.jsonl [--server PATH] -- <flags>]
  ls      llama-eval-cache.py ls     --dir DIR

prompts.jsonl, one request per line, sent exactly as written (pass `reasoning_effort`, `chat_template_kwargs`,
`tools`, ... the way your eval sends them, because they change the prompt text):
  {"id": "q1", "messages": [{"role": "user", "content": "..."}]}        -> /v1/chat/completions
  {"id": "q2", "prompt": "..."}                                         -> /completion
  optional "group": "doc-7" puts prompts that share a long document into one shared entry (default: automatic)

The llama-server flags after `--` must be the flags you will run the eval with (model, -ctk/-ctv, SJ-KVaRN,
--cache-type-s, drafter / MTP, -fa, -b/-ub). Context size and --parallel may differ. The tool adds
--eval-cache DIR --eval-cache-build --parallel N (default 1) --cache-ram 0 --slot-save-path and a free --port.

Only the Python standard library is used.
"""

import argparse
import hashlib
import json
import os
import struct
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

MANIFEST = "manifest.json"
INDEX = "eval-cache.json"
MANIFEST_FORMAT = "llamampere-eval-cache-manifest"
MANIFEST_VERSION = 1
INTERNAL_FIELDS = ("id", "group")


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# server process

class Server:
    def __init__(self, binary, args, log_path, extra):
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        cmd = [binary] + list(args) + list(extra) + ["--host", "127.0.0.1", "--port", str(self.port)]
        self.cmd = cmd
        self.log_path = log_path
        self.log_f = open(log_path, "ab")
        self.log_f.write(("\n### " + " ".join(shlex.quote(c) for c in cmd) + "\n").encode())
        self.log_f.flush()
        self.proc = subprocess.Popen(cmd, stdout=self.log_f, stderr=subprocess.STDOUT)

    def wait_ready(self, timeout):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.proc.returncode}; see {self.log_path}")
            try:
                with urllib.request.urlopen(self.base + "/health", timeout=5) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            time.sleep(0.5)
        raise RuntimeError(f"llama-server did not become ready in {timeout} s; see {self.log_path}")

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=600) as r:
            return json.loads(r.read())

    def post(self, path, body, timeout=24 * 3600):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"POST {path} failed: HTTP {e.code}: {e.read().decode(errors='replace')[:2000]}")

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.log_f.close()


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def find_server(arg):
    if arg:
        return arg
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "..", "..", "build", "bin", "llama-server"),
                 os.path.join(here, "..", "..", "build-bl", "bin", "llama-server")):
        if os.path.exists(cand):
            return os.path.normpath(cand)
    return "llama-server"


# ---------------------------------------------------------------------------------------------------------------------
# prompts

def load_prompts(path):
    prompts = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "messages" not in rec and "prompt" not in rec:
                raise SystemExit(f"{path}:{ln}: a line needs 'messages' (chat) or 'prompt' (plain completion)")
            rec.setdefault("id", f"line{ln}")
            prompts.append(rec)
    ids = [p["id"] for p in prompts]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{path}: prompt ids must be unique")
    return prompts


def entry_stamps(d):
    """{entry id: (mtime_ns, size) of its header file} for every entry in the index."""
    out = {}
    try:
        ents = read_index(d)["entries"]
    except (OSError, ValueError, KeyError):
        return out
    for e in ents:
        try:
            st = os.stat(os.path.join(d, "entries", e["id"] + ".evh"))
            out[e["id"]] = (st.st_mtime_ns, st.st_size)
        except OSError:
            pass
    return out


def header_tokens(d, eid):
    """The prompt tokens stored in an entry header (magic, u32 version, u64 key hash, u32 key length, key,
    u64 token count, i32 tokens), or None if the header cannot be read."""
    try:
        with open(os.path.join(d, "entries", eid + ".evh"), "rb") as f:
            head = f.read(20)
            if len(head) < 20 or head[:4] != b"LEVC":
                return None
            klen = struct.unpack("<I", head[16:20])[0]
            f.seek(20 + klen)
            ntok = struct.unpack("<Q", f.read(8))[0]
            raw = f.read(4 * ntok)
            if len(raw) != 4 * ntok:
                return None
            return list(struct.unpack(f"<{ntok}i", raw))
    except (OSError, struct.error):
        return None


def request_body(rec):
    return {k: v for k, v in rec.items() if k not in INTERNAL_FIELDS}


def endpoint(rec):
    return "/v1/chat/completions" if "messages" in rec else "/completion"


def tokenize(srv, rec):
    """the prompt token ids exactly as the server builds them for this request"""
    if "messages" in rec:
        text = srv.post("/apply-template", request_body(rec))["prompt"]
    else:
        text = rec["prompt"]
        if not isinstance(text, str):
            raise SystemExit(f"prompt {rec['id']}: only string prompts are supported")
    return srv.post("/tokenize", {"content": text, "add_special": True, "parse_special": True})["tokens"]


def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def make_groups(prompts, toks, min_shared):
    """explicit 'group' fields first; the rest grouped by sorting the token lists and joining neighbors that share
    at least min_shared leading tokens"""
    groups = {}
    rest = []
    for p in prompts:
        if "group" in p:
            groups.setdefault("g:" + str(p["group"]), []).append(p["id"])
        else:
            rest.append(p["id"])
    rest.sort(key=lambda i: toks[i])
    run = []
    auto = 0
    for i in rest:
        if run and lcp(toks[run[-1]], toks[i]) >= min_shared:
            run.append(i)
            continue
        if run:
            groups[f"auto{auto}"] = run
            auto += 1
        run = [i]
    if run:
        groups[f"auto{auto}"] = run
    return list(groups.values())


def sha256_tokens(t):
    return hashlib.sha256(json.dumps(t, separators=(",", ":")).encode()).hexdigest()


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_index(d):
    with open(os.path.join(d, INDEX), encoding="utf-8") as f:
        return json.load(f)


def flag_value(args, names):
    for i, a in enumerate(args):
        for n in names:
            if a == n and i + 1 < len(args):
                return args[i + 1]
            if a.startswith(n + "="):
                return a.split("=", 1)[1]
    return None


# ---------------------------------------------------------------------------------------------------------------------
# build

def cmd_build(a):
    os.makedirs(a.dir, exist_ok=True)
    prompts = load_prompts(a.prompts)
    server_args = list(a.server_args)
    for bad in ("--eval-cache", "--eval-cache-build", "--port", "--host", "-np", "--parallel"):
        if bad in server_args:
            raise SystemExit(f"do not pass {bad} after --, the tool sets it")
    import tempfile
    slot_dir = tempfile.mkdtemp(prefix="evc-slots-")
    # every request starts from an empty cache (all slots erased), as a cold run in verify does; with --kv-unified the
    # slots share one cache, and on sliding-window models its size depends on --parallel, so build with the eval's
    srv = Server(find_server(a.server), server_args, os.path.join(a.dir, "build-server.log"),
                 ["--eval-cache", a.dir, "--eval-cache-build", "--parallel", str(a.parallel), "--cache-ram", "0",
                  "--slot-save-path", slot_dir])
    t_start = time.time()
    try:
        srv.wait_ready(a.timeout)
        props = srv.get("/props")
        evc = props.get("eval_cache", {})
        n_ubatch = int(evc.get("n_ubatch", 512))
        log(f"server ready (n_batch {evc.get('n_batch')}, n_ubatch {n_ubatch}), {len(prompts)} prompts")

        toks = {p["id"]: tokenize(srv, p) for p in prompts}
        by_id = {p["id"]: p for p in prompts}

        # margin below the shortest member: a prompt splits its last n_ubatch + 4 tokens into their own batches
        # (SJ-KVaRN also rounds those cuts down to its grid), so a shared entry must end before any of them
        margin = 4 + n_ubatch + (a.margin if a.margin is not None else n_ubatch + 256)

        groups = make_groups(prompts, toks, a.min_shared)
        plan = []  # (representative id, save_at, members)
        for g in groups:
            if len(g) == 1:
                plan.append((g[0], len(toks[g[0]]), g))
                continue
            shared = min(lcp(toks[g[0]], toks[i]) for i in g[1:])
            n_min = min(len(toks[i]) for i in g)
            at = min(shared, n_min - margin)
            if at < a.min_shared // 2 or at <= 0:
                # not worth a shared entry: store each member on its own
                for i in g:
                    plan.append((i, len(toks[i]), [i]))
                continue
            plan.append((g[0], at, g))
            if a.full:
                for i in g:
                    plan.append((i, len(toks[i]), [i]))

        prompt_rows = {}
        for n, (rep, at, members) in enumerate(plan, 1):
            body = request_body(by_id[rep])
            body.update({"eval_cache_save_at": at, "cache_prompt": False, "max_tokens": 1, "n_predict": 1,
                         "stream": False})
            for k in range(a.parallel):
                srv.post(f"/slots/{k}?action=erase", {})
            before = entry_stamps(a.dir)
            t0 = time.time()
            res = srv.post(endpoint(by_id[rep]), body)
            usage_n = (res.get("usage") or {}).get("prompt_tokens")
            if usage_n is None:
                usage_n = (res.get("timings") or {}).get("prompt_n")
            if usage_n is not None and usage_n != len(toks[rep]):
                log(f"warning: prompt {rep}: the server prefilled {usage_n} tokens, /tokenize gave {len(toks[rep])}; "
                    f"the tool's token view differs from the request's (entry is still valid, check the template)")
            # the entry this request wrote: new in the index, or its header file was rewritten (same id)
            entry = None
            after = entry_stamps(a.dir)
            for e in read_index(a.dir)["entries"]:
                if e["id"] in after and after[e["id"]] != before.get(e["id"]) and e["n_tokens"] <= at and \
                        (entry is None or e["n_tokens"] > entry["n_tokens"]):
                    entry = e
            if entry is None:
                log(f"[{n}/{len(plan)}] {rep}: no entry stored (the prompt fits in one batch, nothing to skip)")
            else:
                log(f"[{n}/{len(plan)}] {rep}: entry {entry['id']} {entry['n_tokens']} tokens "
                    f"({len(members)} prompt(s), {time.time() - t0:.1f} s)")
            for i in members:
                if entry is not None and len(toks[i]) > entry["n_tokens"] and \
                        toks[i][:entry["n_tokens"]] == toks[rep][:entry["n_tokens"]]:
                    row = prompt_rows.get(i)
                    if row is None or row["entry"] is None or entry["n_tokens"] > row["entry_tokens"]:
                        prompt_rows[i] = {"entry": entry["id"], "entry_tokens": entry["n_tokens"]}
                else:
                    prompt_rows.setdefault(i, {"entry": None, "entry_tokens": 0})
    finally:
        srv.stop()

    # which entry serves which prompt: the longest stored prompt that the prompt starts with, the same rule the
    # server uses, read from the entry headers (not from which file changed during which request)
    stored = []
    for e in read_index(a.dir)["entries"]:
        t = header_tokens(a.dir, e["id"])
        if t is not None:
            stored.append((len(t), e["id"], t))
    for p in prompts:
        pt = toks[p["id"]]
        best = None
        for n, eid, t in stored:
            if n < len(pt) and pt[:n] == t and (best is None or n > best[0]):
                best = (n, eid)
        prompt_rows[p["id"]] = {"entry": best[1], "entry_tokens": best[0]} if best else {"entry": None, "entry_tokens": 0}

    idx = read_index(a.dir)
    entries = []
    for e in idx["entries"]:
        files = {}
        for ext in ("evh", "tgt", "dft"):
            p = os.path.join(a.dir, "entries", f"{e['id']}.{ext}")
            if os.path.exists(p):
                files[ext] = {"bytes": os.path.getsize(p), "sha256": sha256_file(p)}
        entries.append({"id": e["id"], "n_tokens": e["n_tokens"], "files": files})

    model = flag_value(server_args, ("-m", "--model")) or props.get("model_path")   # -hf: the server knows the file
    model_info = {"path": model}
    if model and os.path.exists(model):
        model_info["bytes"] = os.path.getsize(model)
        if not a.no_model_hash:
            log(f"hashing {model} ...")
            model_info["sha256"] = sha256_file(model)

    manifest = {
        "format": MANIFEST_FORMAT,
        "format_version": MANIFEST_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "build_seconds": round(time.time() - t_start, 1),
        "model": model_info,
        "server_args": server_args,
        "n_batch": evc.get("n_batch"),
        "n_ubatch": n_ubatch,
        "build_parallel": a.parallel,
        "key": idx["key"],
        "key_hash": idx["key_hash"],
        "prompts_file": os.path.basename(a.prompts),
        "prompts": [{
            "id": p["id"],
            "n_tokens": len(toks[p["id"]]),
            "tokens_sha256": sha256_tokens(toks[p["id"]]),
            "entry": prompt_rows.get(p["id"], {}).get("entry"),
            "entry_tokens": prompt_rows.get(p["id"], {}).get("entry_tokens", 0),
            "suffix_tokens": len(toks[p["id"]]) - prompt_rows.get(p["id"], {}).get("entry_tokens", 0),
        } for p in prompts],
        "entries": entries,
    }
    with open(os.path.join(a.dir, MANIFEST + ".tmp"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    os.replace(os.path.join(a.dir, MANIFEST + ".tmp"), os.path.join(a.dir, MANIFEST))

    tot = sum(f["bytes"] for e in entries for f in e["files"].values())
    covered = sum(r["entry_tokens"] for r in manifest["prompts"])
    total = sum(r["n_tokens"] for r in manifest["prompts"])
    log(f"done: {len(entries)} entries, {tot / 2**30:.3f} GiB, {covered} of {total} prompt tokens stored "
        f"({100.0 * covered / max(total, 1):.1f}%), manifest {os.path.join(a.dir, MANIFEST)}")
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# verify

def cmd_verify(a):
    bad = 0
    try:
        idx = read_index(a.dir)
    except (OSError, ValueError) as e:
        log(f"FAIL: cannot read {INDEX}: {e}")
        return 1
    man = None
    mp = os.path.join(a.dir, MANIFEST)
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as f:
            man = json.load(f)
        if man.get("format") != MANIFEST_FORMAT:
            log("FAIL: manifest.json is not an eval-cache manifest")
            bad += 1
        if man.get("key") != idx.get("key"):
            log("FAIL: manifest key differs from the folder index key")
            bad += 1
    else:
        log("note: no manifest.json (folder built without the tool); checking files only")

    want = {e["id"]: e for e in (man or {}).get("entries", [])}
    for e in idx.get("entries", []):
        for ext in ("evh", "tgt") + (("dft",) if e.get("has_dft") else ()):
            p = os.path.join(a.dir, "entries", f"{e['id']}.{ext}")
            if not os.path.exists(p):
                log(f"FAIL: {p} missing")
                bad += 1
                continue
            m = want.get(e["id"], {}).get("files", {}).get(ext)
            if m:
                if os.path.getsize(p) != m["bytes"]:
                    log(f"FAIL: {p} size {os.path.getsize(p)}, manifest {m['bytes']}")
                    bad += 1
                elif not a.quick and sha256_file(p) != m["sha256"]:
                    log(f"FAIL: {p} sha256 differs from the manifest")
                    bad += 1
    log(f"files: {len(idx.get('entries', []))} entries checked, {bad} problem(s)")

    if man and man.get("model", {}).get("sha256") and a.model_hash:
        mpath = man["model"]["path"]
        if not os.path.exists(mpath):
            log(f"note: model {mpath} not found, model hash not checked")
        elif sha256_file(mpath) != man["model"]["sha256"]:
            log(f"FAIL: {mpath} sha256 differs from the one the folder was built with")
            bad += 1
        else:
            log("model sha256 matches")

    if a.identity:
        bad += identity_check(a, man)

    log("verify: " + ("OK" if bad == 0 else f"{bad} FAILURE(S)"))
    return 0 if bad == 0 else 1


def run_prompts(a, prompts, with_cache, tag):
    import tempfile
    slot_dir = tempfile.mkdtemp(prefix="evc-slots-")
    extra = ["--parallel", str(a.parallel), "--cache-ram", "0", "--slot-save-path", slot_dir]
    if with_cache:
        extra += ["--eval-cache", a.dir]
    srv = Server(find_server(a.server), a.server_args, os.path.join(a.dir, f"verify-{tag}.log"), extra)
    out = {}
    try:
        t_start = time.time()
        srv.wait_ready(a.timeout)
        out["_startup_s"] = time.time() - t_start
        for p in prompts:
            for k in range(a.parallel):
                srv.post(f"/slots/{k}?action=erase", {})
            body = request_body(p)
            body.update({"temperature": 0, "top_k": 1, "seed": 1, "max_tokens": a.n_predict,
                         "n_predict": a.n_predict, "cache_prompt": True, "stream": False})
            t0 = time.time()
            res = srv.post(endpoint(p), body)
            if "choices" in res:
                ch = res["choices"][0]
                text = (ch.get("message") or {}).get("content") or ch.get("text") or ""
                rc = (ch.get("message") or {}).get("reasoning_content") or ""
                text = rc + "\x00" + text
            else:
                text = res.get("content", "")
            tm = res.get("timings") or {}
            out[p["id"]] = {"text": text, "prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n"),
                            "prompt_ms": tm.get("prompt_ms"), "predicted_n": tm.get("predicted_n"),
                            "predicted_ms": tm.get("predicted_ms"), "seconds": time.time() - t0}
    finally:
        srv.stop()
    return out


def identity_check(a, man):
    if not a.prompts:
        log("FAIL: --identity needs --prompts")
        return 1
    prompts = load_prompts(a.prompts)[: a.identity]
    log(f"identity: {len(prompts)} prompts, temperature 0, {a.n_predict} tokens, with and without the eval cache")
    hit = run_prompts(a, prompts, True, "cache")
    cold = run_prompts(a, prompts, False, "nocache")
    bad = 0
    n_hit = 0
    rows = []
    for p in prompts:
        h = hit[p["id"]]
        c = cold[p["id"]]
        used = h["cache_n"] is not None and h["cache_n"] > 0
        n_hit += used
        same = h["text"] == c["text"]
        bad += not same
        log(f"  {p['id']}: {'identical' if same else 'DIFFERENT'}; with cache prefilled {h['prompt_n']} "
            f"(loaded {h['cache_n']}) in {h['seconds']:.2f} s, without {c['prompt_n']} in {c['seconds']:.2f} s"
            f" (prompt time {(h['prompt_ms'] or 0) / 1000:.2f} s vs {(c['prompt_ms'] or 0) / 1000:.2f} s)")
        rows.append({"id": p["id"], "identical": same, "hit": bool(used),
                     **{f"cache_{k}": h[k] for k in ("prompt_n", "cache_n", "prompt_ms", "predicted_n", "predicted_ms", "seconds")},
                     **{f"nocache_{k}": c[k] for k in ("prompt_n", "cache_n", "prompt_ms", "predicted_n", "predicted_ms", "seconds")}})
    log(f"identity: {len(prompts) - bad}/{len(prompts)} identical, {n_hit}/{len(prompts)} used the eval cache")
    wall_h = sum(r["cache_seconds"] for r in rows)
    wall_c = sum(r["nocache_seconds"] for r in rows)
    log(f"identity: request time for all {len(rows)} prompts {wall_h:.1f} s with the cache, {wall_c:.1f} s without "
        f"(server start {hit['_startup_s']:.1f} s / {cold['_startup_s']:.1f} s)")
    if a.report:
        with open(a.report, "w") as f:
            json.dump({"parallel": a.parallel, "n_predict": a.n_predict, "server_args": a.server_args,
                       "startup_s": {"cache": hit["_startup_s"], "nocache": cold["_startup_s"]},
                       "prompts": rows}, f, indent=1)
        log(f"report written to {a.report}")
    if n_hit == 0:
        log("FAIL: no prompt used the eval cache")
        bad += 1
    return bad


# ---------------------------------------------------------------------------------------------------------------------
# ls

def cmd_ls(a):
    idx = read_index(a.dir)
    print(f"{a.dir}: format {idx.get('format_version')}, key hash {idx.get('key_hash')}")
    for field in idx.get("key", "").split(";"):
        print(f"  {field}")
    tot = 0
    for e in idx.get("entries", []):
        print(f"  entry {e['id']}: {e['n_tokens']} tokens, {e['bytes'] / 2**20:.1f} MiB{', drafter' if e.get('has_dft') else ''}")
        tot += e["bytes"]
    print(f"{len(idx.get('entries', []))} entries, {tot / 2**30:.3f} GiB")
    return 0


def main():
    argv = sys.argv[1:]
    server_args = []
    if "--" in argv:
        k = argv.index("--")
        argv, server_args = argv[:k], argv[k + 1:]

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build (or extend) a folder")
    b.add_argument("--dir", required=True, help="eval-cache folder")
    b.add_argument("--prompts", required=True, help="JSONL file, one request per line")
    b.add_argument("--server", help="llama-server binary (default: build/bin/llama-server)")
    b.add_argument("--min-shared", type=int, default=1024,
                   help="prompts that share at least this many leading tokens get one shared entry (default 1024)")
    b.add_argument("--margin", type=int, default=None,
                   help="extra tokens a shared entry stays below the shortest member (default n_ubatch + 256)")
    b.add_argument("--full", action="store_true", help="also store every grouped prompt on its own (more disk)")
    b.add_argument("--no-model-hash", action="store_true", help="skip the model sha256 in the manifest")
    b.add_argument("--parallel", type=int, default=1,
                   help="slots of the build server (default 1); only matters with --kv-unified on sliding-window "
                        "models, where it should match the eval's --parallel for byte-identical output")
    b.add_argument("--timeout", type=int, default=1800, help="seconds to wait for the server to load")

    v = sub.add_parser("verify", help="check a folder's files, optionally the output identity")
    v.add_argument("--dir", required=True)
    v.add_argument("--quick", action="store_true", help="sizes only, no sha256 of the state files")
    v.add_argument("--model-hash", action="store_true", help="also re-hash the model file")
    v.add_argument("--identity", type=int, default=0,
                   help="run the first N prompts at temperature 0 with and without the cache and compare the text")
    v.add_argument("--prompts", help="JSONL prompts for --identity")
    v.add_argument("--n-predict", type=int, default=64)
    v.add_argument("--parallel", type=int, default=1, help="slots of the --identity servers (default 1)")
    v.add_argument("--server")
    v.add_argument("--timeout", type=int, default=1800)
    v.add_argument("--report", help="write per-prompt timings of the --identity runs to this JSON file")

    l = sub.add_parser("ls", help="list a folder")
    l.add_argument("--dir", required=True)

    a = ap.parse_args(argv)
    a.server_args = server_args
    if a.cmd == "build":
        if not server_args:
            ap.error("build needs the llama-server flags after --")
        return cmd_build(a)
    if a.cmd == "verify":
        if a.identity and not server_args:
            ap.error("--identity needs the llama-server flags after --")
        return cmd_verify(a)
    return cmd_ls(a)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError, ValueError) as e:
        log(f"error: {e}")
        sys.exit(1)
