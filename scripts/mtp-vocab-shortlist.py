#!/usr/bin/env python3
"""Build and score draft vocabulary shortlists (llama-mtp-vocab-v1) for any tokenizer, without model weights.

Backlog #77. Only the GGUF's tokenizer arrays and tensor headers are read, never the weights. The ranking is
the token frequency of the corpora (per-domain mass), with the W3 hard-keep sets (control / user-defined
tokens, byte pieces, ASCII singles, space+ASCII, whitespace pieces, short digit pieces) and a non-Latin
reserve. Every size is a prefix of one ordering, so the smaller lists are subsets of the larger ones.
Coverage is measured on held-out sources: expected acceptance loss of a restricted draft head is about the
uncovered fraction (W3 CRITIC_REVIEW F1). Acceptance-weighted ranking needs verify traces from the model
itself and is not done here.

    python scripts/mtp-vocab-shortlist.py --gguf Qwen3.8-27B-ATX-4-XS.gguf \\
        --tokenize-bin build/bin/llama-tokenize \\
        --corpus 'coding:auto:text:runs/completions/*coding*.txt' \\
        --corpus 'stem:auto:qid:json:kvq/results/**/*.jsonl@reasoning,content' \\
        --corpus 'mixed:split=select:jsonids:W3/responses/*.json@tokens' \\
        --compare atx64k=docs/mtp-vocab/atx_65536.txt --out out/ --prefix qwen3.8-27b

--corpus DOMAIN:SPLIT:KIND:GLOB[@FIELD,FIELD...]
    KIND   text     the whole file is one source
           json     JSON object, array of objects, or JSONL; each record's FIELDs (dotted paths) joined by a newline
           ids      raw int32 token ids (one source per file)
           jsonids  JSON/JSONL records whose FIELD is a list of token ids
    DOMAIN a name, or field=KEY: each JSON record's domain is record[KEY]
    SPLIT  train | held | auto (one source in three is held out, by a stable hash of the file and record index)
           | auto:KEY (the hash of record[KEY], so repeats of one prompt stay on one side) | KEY=VALUE (a record is
           train when record[KEY] == VALUE, else held)
Token-id corpora come from the tokenizer named by --ids-gguf (default --gguf). When its fingerprint differs from
the target's, the ids are decoded to text with the source vocabulary and re-tokenized with the target's.
Text is tokenized by --hf-tokenizer (tokenizer.json, needs the `tokenizers` package) or by llama-tokenize in
vocab-only mode (--tokenize-bin; special tokens parsed, no BOS).
"""
import argparse
import collections
import glob
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import unicodedata

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gguf-py"))
from gguf import GGUFReader  # noqa: E402

_spec = importlib.util.spec_from_file_location("gen_builtin", os.path.join(HERE, "gen-mtp-vocab-builtin.py"))
gen_builtin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen_builtin)

TT_NORMAL, TT_UNKNOWN, TT_CONTROL, TT_USER, TT_UNUSED, TT_BYTE = 1, 2, 3, 4, 5, 6
NONLATIN = {"cjk", "cyrillic", "arabic", "thai", "indic", "other_script"}
WS = set(b" \t\n\r\x0b\x0c")
PAD_RE = re.compile(r"^\[PAD\d+\]$|^<\|?pad[_\d|]*\|?>$", re.IGNORECASE)


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


B2U = bytes_to_unicode()
U2B = {v: k for k, v in B2U.items()}


class Vocab:
    """token strings, types and byte pieces of one GGUF tokenizer"""

    def __init__(self, path):
        self.path = path
        r = GGUFReader(path, "r")
        self.fp, self.n, self.arch = gen_builtin.tokenizer_info(path)
        self.toks = [t.decode("utf-8", "replace") for t in gen_builtin.str_array(r, "tokenizer.ggml.tokens")]
        tf = r.fields.get("tokenizer.ggml.token_type")
        self.types = [int(tf.parts[i][0]) for i in tf.data] if tf is not None else [TT_NORMAL] * self.n
        self.model = (gen_builtin.str_array(r, "tokenizer.ggml.model") or [b"?"])[0].decode()
        self.byte_level = self.model == "gpt2"
        self.pieces = [self._piece_bytes(i) for i in range(self.n)]
        self.tensors = {t.name: t for t in r.tensors}

    def _piece_bytes(self, i):
        p, t = self.toks[i], self.types[i]
        if t in (TT_CONTROL, TT_USER, TT_UNUSED, TT_UNKNOWN):
            return p.encode("utf-8")
        if self.byte_level:
            try:
                return bytes(U2B[c] for c in p)
            except KeyError:
                return p.encode("utf-8")
        m = re.fullmatch(r"<0x([0-9A-Fa-f]{2})>", p)
        if t == TT_BYTE and m:
            return bytes([int(m.group(1), 16)])
        return p.replace("▁", " ").encode("utf-8")

    def is_pad(self, i):
        return self.types[i] == TT_UNUSED or bool(PAD_RE.match(self.toks[i]))

    def byte_piece_ids(self):
        if self.byte_level:
            ix = {p: i for i, p in enumerate(self.toks)}
            return {ix[B2U[b]] for b in range(256) if B2U[b] in ix}
        return {i for i in range(self.n) if self.types[i] == TT_BYTE}

    def decode(self, ids):
        return b"".join(self.pieces[i] for i in ids if 0 <= i < self.n).decode("utf-8", "replace")

    def classify(self, i):
        t = self.types[i]
        if self.is_pad(i):
            return "pad"
        if t in (TT_CONTROL, TT_USER, TT_UNKNOWN):
            return "control"
        raw = self.pieces[i]
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return "byte_fragment"
        if len(raw) == 1 and (raw[0] >= 128 or raw[0] < 32) and raw[0] not in WS:
            return "byte_fragment"
        s = text.strip(" \t\n\r\x0b\x0c")
        if not s:
            return "whitespace"
        if s.isdigit() and s.isascii():
            return "digit"
        if all(not ch.isalnum() for ch in s):
            return "ascii_symbol" if s.isascii() else "unicode_symbol"
        scripts = set()
        for ch in s:
            if not ch.isalpha():
                continue
            try:
                head = unicodedata.name(ch).split()[0]
            except ValueError:
                scripts.add("other_script")
                continue
            if head in ("CJK", "HIRAGANA", "KATAKANA", "HANGUL", "BOPOMOFO", "IDEOGRAPHIC", "HALFWIDTH", "FULLWIDTH"):
                scripts.add("cjk")
            elif head == "CYRILLIC":
                scripts.add("cyrillic")
            elif head == "ARABIC":
                scripts.add("arabic")
            elif head == "THAI":
                scripts.add("thai")
            elif head in ("DEVANAGARI", "BENGALI", "TAMIL", "TELUGU", "GUJARATI", "KANNADA", "MALAYALAM", "GURMUKHI", "ORIYA", "SINHALA"):
                scripts.add("indic")
            elif head == "LATIN" or ch.isascii():
                scripts.add("latin")
            else:
                scripts.add("other_script")
        if not scripts:
            return "ascii_symbol" if s.isascii() else "unicode_symbol"
        if scripts == {"latin"}:
            return "latin_ascii" if s.isascii() else "latin_ext"
        if len(scripts) > 1:
            return "mixed"
        return next(iter(scripts))

    def hard_keep(self):
        hard = {i for i in range(self.n) if self.types[i] in (TT_CONTROL, TT_USER) and not self.is_pad(i)}
        hard |= self.byte_piece_ids()
        by_bytes = {}
        for i, p in enumerate(self.pieces):
            if self.types[i] in (TT_NORMAL, TT_BYTE):
                by_bytes.setdefault(p, i)
        for c in range(32, 127):
            for p in (bytes([c]), b" " + bytes([c])):
                if p in by_bytes:
                    hard.add(by_bytes[p])
        for i, p in enumerate(self.pieces):
            if self.types[i] != TT_NORMAL or not p:
                continue
            if all(b in WS for b in p) or (len(p) <= 3 and p.isdigit()):
                hard.add(i)
        return {i for i in hard if not self.is_pad(i)}


class Tokenizer:
    """text -> target token ids: HF tokenizers, or llama-tokenize in vocab-only mode"""

    def __init__(self, gguf, hf, binary):
        self.gguf, self.bin, self.hf = gguf, binary, None
        if hf:
            from tokenizers import Tokenizer as HFTok
            self.hf = HFTok.from_file(hf if hf.endswith(".json") else os.path.join(hf, "tokenizer.json"))

    def encode(self, text):
        if not text:
            return np.zeros(0, np.int64)
        if self.hf is not None:
            return np.asarray(self.hf.encode(text, add_special_tokens=False).ids, np.int64)
        if not self.bin:
            raise SystemExit("text corpora need --tokenize-bin (llama-tokenize) or --hf-tokenizer")
        # llama-tokenize parses special tokens by default (--no-parse-special turns it off)
        p = subprocess.run([self.bin, "-m", self.gguf, "--stdin", "--ids", "--no-bos", "--log-disable"],
                           input=text.encode("utf-8", "replace"), capture_output=True, check=False)
        out = p.stdout.decode("utf-8", "replace").strip().splitlines()
        line = next((ln for ln in reversed(out) if ln.startswith("[")), None)
        if p.returncode != 0 or line is None:
            raise SystemExit(f"llama-tokenize failed (rc {p.returncode}): {p.stderr.decode('utf-8', 'replace')[-400:]}")
        return np.asarray(json.loads(line), np.int64)


def get_path(rec, path):
    cur = rec
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def records(path):
    """JSON object / array / JSONL -> list of dicts"""
    txt = open(path, encoding="utf-8", errors="replace").read()
    try:
        d = json.loads(txt)
        return d if isinstance(d, list) else [d]
    except json.JSONDecodeError:
        return [json.loads(ln) for ln in txt.splitlines() if ln.strip()]


def stable_held(key):
    return int(hashlib.sha1(key.encode()).hexdigest(), 16) % 3 == 2


def load_corpus(spec, target, tok, src_vocab):
    """yields (split, source, ids) for one --corpus spec"""
    parts = spec.split(":", 3)
    if len(parts) == 4 and parts[1] == "auto" and parts[2] not in ("text", "json", "ids", "jsonids"):
        # auto:KEY
        parts = spec.split(":", 4)
        domain, split, kind, rest = parts[0], "auto:" + parts[2], parts[3], parts[4]
    elif len(parts) == 4:
        domain, split, kind, rest = parts
    else:
        raise SystemExit(f"bad --corpus {spec!r}")
    pattern, _, fields = rest.partition("@")
    fields = [f for f in fields.split(",") if f]
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise SystemExit(f"--corpus {spec!r}: no files match")
    same_tok = src_vocab.fp == target.fp

    def dom_of(rec):
        if domain.startswith("field="):
            v = get_path(rec, domain[6:]) if rec is not None else None
            return str(v) if v is not None else "unknown"
        return domain

    def side(rec, key):
        if split in ("train", "held"):
            return split
        if split == "auto":
            return "held" if stable_held(key) else "train"
        if split.startswith("auto:"):
            v = get_path(rec, split[5:]) if rec is not None else None
            return "held" if stable_held(str(v) if v is not None else key) else "train"
        k, _, v = split.partition("=")
        return "train" if rec is not None and str(get_path(rec, k)) == v else "held"

    texts = collections.defaultdict(list)  # (domain, split) -> [(source, text)]
    out = []
    for f in files:
        if kind == "text":
            texts[(dom_of(None), side(None, f))].append((f, open(f, encoding="utf-8", errors="replace").read()))
        elif kind == "ids":
            ids = np.fromfile(f, dtype=np.int32).astype(np.int64)
            if same_tok:
                out.append((dom_of(None), side(None, f), f, ids))
            else:
                texts[(dom_of(None), side(None, f))].append((f, src_vocab.decode(ids)))
        elif kind in ("json", "jsonids"):
            for k, rec in enumerate(records(f)):
                key = f"{f}#{k}"
                s = (dom_of(rec), side(rec, key))
                if kind == "json":
                    vals = [get_path(rec, fl) for fl in fields]
                    t = "\n".join(v for v in vals if isinstance(v, str) and v)
                    if t:
                        texts[s].append((key, t))
                else:
                    v = get_path(rec, fields[0]) if fields else None
                    if not isinstance(v, list) or not v:
                        continue
                    ids = np.asarray(v, np.int64)
                    if same_tok:
                        out.append((s[0], s[1], key, ids))
                    else:
                        texts[s].append((key, src_vocab.decode(ids)))
        else:
            raise SystemExit(f"--corpus {spec!r}: unknown kind {kind}")
    # one tokenizer call per (domain, split): sources joined by a blank line (a few boundary tokens per source)
    for (d, s), items in texts.items():
        ids = tok.encode("\n\n".join(t for _, t in items))
        out.append((d, s, f"{pattern} ({len(items)} {d} sources, {s})", ids))
    for _, _, src, ids in out:
        if ids.size and (ids.min() < 0 or ids.max() >= target.n):
            raise SystemExit(f"{src}: token id out of range for the target vocabulary ({target.n})")
    return out


def coverage(ids, mask):
    cov = mask[ids]
    r = {"n": int(ids.size), "coverage": float(cov.mean()) if ids.size else None}
    if ids.size >= 3:
        r["win3_full"] = float((cov[:-2] & cov[1:-1] & cov[2:]).mean())
    return r


def read_map(path, n_vocab):
    mv, ids = gen_builtin.read_map(path)
    if mv != n_vocab:
        raise SystemExit(f"{path}: map is for {mv} tokens, the target has {n_vocab}")
    return ids


def builtin_matches(fp):
    inc = os.path.join(HERE, "..", "src", "llama-mtp-vocab-builtin-data.inc")
    if not os.path.exists(inc):
        return []
    rx = re.compile(r'\{ "([^"]+)", "([^"]+)", "([^"]+)", 0x([0-9a-f]{16})ULL, (\d+), (\d+),')
    return [m.groups() for m in rx.finditer(open(inc).read()) if int(m.group(4), 16) == fp]


def head_cost(target, sizes):
    """bytes one draft step reads from the output head, full vs shortlisted [est: weight bytes only]"""
    rows = []
    for name, t in target.tensors.items():
        if name == "output.weight" or name.endswith("nextn.shared_head_head.weight") or name == "token_embd.weight":
            shape = [int(x) for x in t.shape]
            n_rows = shape[1] if len(shape) > 1 else 1
            row_b = int(t.n_bytes) // max(1, n_rows)
            rows.append({"tensor": name, "type": t.tensor_type.name, "n_embd": shape[0], "rows": n_rows, "row_bytes": row_b,
                         "full_MiB": round(row_b * n_rows / 2**20, 2),
                         "shortlist_MiB": {str(n): round(row_b * min(n, n_rows) / 2**20, 2) for n in sizes}})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True, help="GGUF of the target model (tokenizer + tensor headers only)")
    ap.add_argument("--hf-tokenizer", help="tokenizer.json (or its directory) for text corpora")
    ap.add_argument("--tokenize-bin", help="llama-tokenize binary for text corpora (vocab-only load of --gguf)")
    ap.add_argument("--ids-gguf", help="GGUF whose tokenizer produced the ids/jsonids corpora (default --gguf)")
    ap.add_argument("--corpus", action="append", default=[], required=True)
    ap.add_argument("--mass", action="append", default=[], help="DOMAIN=W (default: equal mass per train domain)")
    ap.add_argument("--sizes", default="16384,32768,65536,98304")
    ap.add_argument("--reserve", type=int, default=1024, help="non-Latin reserve size (W3: 1024)")
    ap.add_argument("--down", type=float, default=0.1, help="non-Latin down-weight outside the reserve (W3: 0.1)")
    ap.add_argument("--compare", action="append", default=[], help="NAME=MAP: existing list to score on the same held-out data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="shortlist")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)

    target = Vocab(a.gguf)
    src = target if not a.ids_gguf or os.path.samefile(a.ids_gguf, a.gguf) else Vocab(a.ids_gguf)
    tok = Tokenizer(a.gguf, a.hf_tokenizer, a.tokenize_bin)
    V = target.n
    print(f"target {a.gguf}: arch {target.arch}, tokenizer {target.model}, n_vocab {V}, fingerprint {target.fp:016x}", flush=True)
    for m in builtin_matches(target.fp):
        print(f"  built-in list with this tokenizer: {m[0]} (family {m[1]}, arch {m[2]}, {m[5]} rows)")
    if src is not target:
        print(f"ids corpora from {a.ids_gguf}: fingerprint {src.fp:016x} ({'same tokenizer' if src.fp == target.fp else 'different: decode + re-tokenize'})")

    train = collections.defaultdict(list)
    held = collections.defaultdict(list)
    manifest = []
    for spec in a.corpus:
        for dom, s, source, ids in load_corpus(spec, target, tok, src):
            (train if s == "train" else held)[dom].append(ids)
            manifest.append({"domain": dom, "split": s, "source": source, "n": int(ids.size)})
    print("corpora:", {d: (int(sum(x.size for x in train[d])), int(sum(x.size for x in held[d]))) for d in sorted(set(train) | set(held))},
          "(train, held) tokens", f"{time.time() - t0:.0f}s", flush=True)
    if not train:
        raise SystemExit("no train tokens")

    mass = {d: 1.0 for d in train}
    for m in a.mass:
        k, _, v = m.partition("=")
        if k not in train:
            raise SystemExit(f"--mass {m}: no train tokens for domain {k}")
        mass[k] = float(v)
    tot = sum(mass.values())
    w = np.zeros(V, np.float64)
    groups = {}
    for d, arrs in train.items():
        ids = np.concatenate(arrs)
        c = np.bincount(ids, minlength=V).astype(np.float64)
        w += mass[d] / tot * c / ids.size
        groups[d] = {"tokens": int(ids.size), "unique": int((c > 0).sum()), "mass": mass[d] / tot}

    cls = [target.classify(i) for i in range(V)]
    pad = np.array([c == "pad" for c in cls])
    nonlatin = np.array([c in NONLATIN for c in cls])
    util = w.copy()
    util[nonlatin] *= a.down
    hard = target.hard_keep()
    reserve = {int(i) for i in np.argsort(-w, kind="stable") if nonlatin[i] and w[i] > 0 and not pad[i]}
    reserve = set(sorted(reserve, key=lambda i: (-w[i], i))[:a.reserve])
    eligible = [i for i in range(V) if not pad[i]]
    order = sorted(eligible, key=lambda i: (i not in hard, i not in reserve, -util[i], i))
    n_obs = int(((w > 0) & ~pad).sum())
    print(f"hard-keep {len(hard)}, reserve {len(reserve)}, observed {n_obs}, eligible {len(eligible)}, pad/unused {int(pad.sum())}", flush=True)

    sizes = sorted(int(x) for x in a.sizes.split(","))
    maps = {}
    for n in sizes:
        if n > len(eligible):
            print(f"size {n}: more than the {len(eligible)} eligible tokens, skipped")
            continue
        ids = sorted(order[:n])
        maps[f"{a.prefix}_{n}"] = ids
        with open(os.path.join(a.out, f"{a.prefix}_{n}.txt"), "w") as fh:
            fh.write(f"llama-mtp-vocab-v1 {V} {n}\n" + "\n".join(map(str, ids)) + "\n")
        filled = sum(1 for i in order[:n] if w[i] == 0 and i not in hard)
        print(f"wrote {a.prefix}_{n}.txt ({filled} unobserved ids filled in id order)")

    cmp = dict(maps)
    for c in a.compare:
        k, _, p = c.partition("=")
        cmp[k] = read_map(p, V)
    for n in sizes:
        cmp[f"prefix_{n}"] = [i for i in range(min(n, V))]
    masks = {}
    for k, ids in cmp.items():
        m = np.zeros(V, bool)
        m[ids] = True
        masks[k] = m

    report = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "gguf": a.gguf, "arch": target.arch, "n_vocab": V,
              "fingerprint": f"{target.fp:016x}", "builtin_matches": [m[0] for m in builtin_matches(target.fp)],
              "groups": groups, "hard_keep": len(hard), "reserve": len(reserve), "down": a.down, "observed": n_obs,
              "held_by_domain": {}, "held_pooled": {}, "train_pooled": {}, "top_uncovered": {},
              "head_cost_est": head_cost(target, sizes), "manifest": manifest}
    # a domain can be all-train (split "train"): it has no held-out rows to score
    held = {d: arrs for d, arrs in held.items() if sum(len(x) for x in arrs) > 0}
    for d, arrs in sorted(held.items()):
        ids = np.concatenate(arrs)
        report["held_by_domain"][d] = {k: coverage(ids, m) for k, m in masks.items()}
        for k in maps:
            unc = collections.Counter(ids[~masks[k][ids]].tolist())
            report["top_uncovered"].setdefault(d, {})[k] = [
                {"id": i, "n": n, "class": cls[i], "text": target.pieces[i].decode("utf-8", "replace")} for i, n in unc.most_common(25)]
    if held:
        allh = np.concatenate([x for arrs in held.values() for x in arrs])
        report["held_pooled"] = {k: coverage(allh, m) for k, m in masks.items()}
    allt = np.concatenate([x for arrs in train.values() for x in arrs])
    report["train_pooled"] = {k: coverage(allt, m) for k, m in masks.items()}
    json.dump(report, open(os.path.join(a.out, "report.json"), "w"), indent=1, ensure_ascii=False)
    with open(os.path.join(a.out, "ranking.tsv"), "w") as fh:
        fh.write("rank\tid\tclass\tw\thard\treserve\ttext\n")
        for r, i in enumerate(order[:max(sizes) + 1024]):
            fh.write(f"{r}\t{i}\t{cls[i]}\t{w[i]:.3e}\t{int(i in hard)}\t{int(i in reserve)}\t"
                     f"{json.dumps(target.pieces[i].decode('utf-8', 'replace'), ensure_ascii=False)}\n")

    # markdown summary
    ks = list(cmp)
    lines = [f"# Draft vocab shortlist: {os.path.basename(a.gguf)}", "",
             f"arch {target.arch}, n_vocab {V}, tokenizer fingerprint {target.fp:016x}"
             + (f" (same tokenizer as built-in {', '.join(report['builtin_matches'])})" if report["builtin_matches"] else ""), "",
             "Train groups: " + ", ".join(f"{d} {g['tokens']:,} tokens / {g['unique']:,} ids, mass {g['mass']:.2f}" for d, g in groups.items()), "",
             "## Held-out coverage % (expected acceptance loss ~ 100 - coverage)", "",
             "| domain | tokens | " + " | ".join(ks) + " |", "|---|---:|" + "---:|" * len(ks)]
    for d, row in report["held_by_domain"].items():
        lines.append(f"| {d} | {row[ks[0]]['n']:,} | " + " | ".join(f"{100 * row[k]['coverage']:.2f}" for k in ks) + " |")
    if report["held_pooled"]:
        hp = report["held_pooled"]
        lines.append(f"| **pooled** | {hp[ks[0]]['n']:,} | " + " | ".join(f"**{100 * hp[k]['coverage']:.2f}**" for k in ks) + " |")
        lines += ["", "3-token windows fully covered, pooled held-out %: " + ", ".join(
            f"{k} {100 * hp[k].get('win3_full', float('nan')):.2f}" for k in ks)]
    if report["head_cost_est"]:
        lines += ["", "## Head bytes per draft step [est: weight bytes only, no launch or sampler cost]", ""]
        for h in report["head_cost_est"]:
            lines.append(f"- {h['tensor']} {h['type']} {h['n_embd']} x {h['rows']}: full {h['full_MiB']} MiB; "
                         + ", ".join(f"{n}: {v} MiB" for n, v in h["shortlist_MiB"].items()))
    open(os.path.join(a.out, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[4:]))
    print(f"\ndone in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
