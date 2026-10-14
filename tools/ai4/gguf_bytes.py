#!/usr/bin/env python3
# Private copy of frontier/BACKLOG_RUN_20261013/AIBEST/tools/gguf_bytes.py (2026-10-14).
"""AIBEST (private): per-tensor byte table of a GGUF from its header only (local file or an HTTP(S) URL fetched by range).
Tensor sizes come from the data offsets (so custom types such as AI4/EXL3 need no size table).
usage: gguf_bytes.py SRC [--hf-token-file F] [--json OUT]
prints: file bytes, token_embd bytes, head (output.weight) bytes, GPU bytes (= all tensor bytes except token_embd), params."""
import json, struct, sys, subprocess

GT = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}   # fixed-size GGUF value types


class Src:
    def __init__(self, src, tok=None):
        self.src, self.tok, self.buf, self.size = src, tok, b"", None
        if src.startswith("http"):
            self._fetch(16 << 20)
        else:
            import os
            self.size = os.path.getsize(src)
            with open(src, "rb") as f:
                self.buf = f.read(64 << 20)

    def _fetch(self, n):
        h = ["-H", f"Authorization: Bearer {open(self.tok).read().strip()}"] if self.tok else []
        out = subprocess.run(["curl", "-sSL", "-r", f"0-{n - 1}", *h, "-D", "/dev/stderr", self.src],
                             capture_output=True)
        self.buf = out.stdout
        for line in out.stderr.decode(errors="replace").splitlines():
            if line.lower().startswith("content-range:") and "/" in line:
                self.size = int(line.rsplit("/", 1)[1].strip())

    def need(self, end):
        if end > len(self.buf):
            if not self.src.startswith("http"):
                raise RuntimeError("header larger than 64 MiB")
            self._fetch(max(end + (8 << 20), 2 * len(self.buf)))


def parse(src, tok=None):
    s = Src(src, tok)
    p = 0

    def rd(fmt):
        nonlocal p
        n = struct.calcsize(fmt); s.need(p + n)
        v = struct.unpack_from("<" + fmt, s.buf, p); p += n
        return v[0] if len(v) == 1 else v

    def rstr():
        nonlocal p
        n = rd("Q"); s.need(p + n)
        v = s.buf[p:p + n].decode(errors="replace"); p += n
        return v

    def rval(t):
        nonlocal p
        if t == 8:
            return rstr()
        if t == 9:
            et = rd("I"); n = rd("Q")
            if et in GT:
                s.need(p + n * GT[et]); p += n * GT[et]; return None
            for _ in range(n):
                rval(et)
            return None
        fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}[t]
        return rd(fmt)

    assert rd("4s") == b"GGUF"
    rd("I"); nt = rd("Q"); nkv = rd("Q")
    align = 32
    for _ in range(nkv):
        k = rstr(); t = rd("I"); v = rval(t)
        if k == "general.alignment":
            align = v
    tens = []
    for _ in range(nt):
        name = rstr(); nd = rd("I"); dims = [rd("Q") for _ in range(nd)]; ty = rd("I"); off = rd("Q")
        tens.append([name, dims, ty, off])
    data0 = (p + align - 1) // align * align
    tens.sort(key=lambda r: r[3])
    out = []
    for i, (name, dims, ty, off) in enumerate(tens):
        end = tens[i + 1][3] if i + 1 < len(tens) else s.size - data0
        n = 1
        for d in dims:
            n *= d
        out.append({"name": name, "dims": dims, "type": ty, "bytes": end - off, "n": n})
    return {"file_bytes": s.size, "data_start": data0, "tensors": out}


def summary(info):
    t = info["tensors"]
    emb = sum(r["bytes"] for r in t if r["name"] == "token_embd.weight")
    head = sum(r["bytes"] for r in t if r["name"] == "output.weight")
    tot = sum(r["bytes"] for r in t)
    n_all = sum(r["n"] for r in t)
    n_emb = sum(r["n"] for r in t if r["name"] == "token_embd.weight")
    mtp = sum(r["bytes"] for r in t if ".nextn." in r["name"] or r["name"].startswith("mtp"))
    return {"file_bytes": info["file_bytes"], "tensor_bytes": tot, "token_embd_bytes": emb, "head_bytes": head,
            "gpu_bytes": tot - emb, "params_all": n_all, "params_embd": n_emb, "mtp_bytes": mtp,
            "n_tensors": len(t)}


if __name__ == "__main__":
    a = sys.argv[1:]
    tok = a[a.index("--hf-token-file") + 1] if "--hf-token-file" in a else None
    info = parse(a[0], tok)
    sm = summary(info)
    if "--json" in a:
        json.dump({"summary": sm, "tensors": info["tensors"]}, open(a[a.index("--json") + 1], "w"))
    print(json.dumps(sm))
