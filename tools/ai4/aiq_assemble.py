#!/usr/bin/env python3
"""aiq_assemble.py (AIQ3, private): build a mixed-tier GGUF from per-tier encodes, no re-encode.

  aiq_assemble.py --base B.gguf --src TAG=F.gguf [--src ...] --alloc alloc.tsv --out O.gguf

Metadata and tensor order come from --base; each tensor's type and bytes come from the tier named in alloc.tsv
(aiq_alloc.py output, columns tensor/tag; tensors not listed, or listed with the base tag, are copied from --base).
A tier can be a full model or overlays (llama-ai4-quantize --overlay-only; repeat --src TAG=FILE once per chunk file);
its files together must hold every tensor allocated to it, with the base tensor's shape. The KV "aiq3.allocation" records the per-tier tensor counts and the alloc header.
Written to O.gguf.tmp and renamed when complete; every tensor's bytes are verified against its source (size + CRC).
"""
import argparse, os, struct, sys, zlib
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
from gguf import GGUFReader  # noqa: E402

GGUF_STR, GGUF_U32 = 8, 4
CHUNK = 64 << 20


def kv_bytes(field):
    return b"".join(bytes(p.tobytes()) for p in field.parts)


def enc_str(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def read_alloc(path):
    hdr, out = "", {}
    for line in open(path):
        if line.startswith("#"):
            hdr = hdr or line[1:].strip()
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 2 or f[0] == "tensor":
            continue
        out[f[0]] = f[1]
    return hdr, out


def copy_range(fsrc, off, n, fdst):
    crc, left = 0, n
    fsrc.seek(off)
    while left:
        b = fsrc.read(min(CHUNK, left))
        if not b:
            raise SystemExit("short read")
        crc = zlib.crc32(b, crc)
        fdst.write(b)
        left -= len(b)
    return crc


def crc_range(path, off, n):
    crc, left = 0, n
    with open(path, "rb") as f:
        f.seek(off)
        while left:
            b = f.read(min(CHUNK, left))
            crc = zlib.crc32(b, crc)
            left -= len(b)
    return crc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--base-tag", default="base")
    ap.add_argument("--src", action="append", default=[], help="TAG=FILE tier source (repeatable)")
    ap.add_argument("--alloc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-verify", action="store_true", help="skip the re-read CRC check of the output")
    args = ap.parse_args(argv)

    base = GGUFReader(args.base, "r")
    paths = [args.base]
    tidx = {args.base_tag: {t.name: (0, t) for t in base.tensors}}   # tag -> name -> (file index, ReaderTensor)
    for s in args.src:   # a tag may repeat (one overlay file per layer chunk)
        tag, _, path = s.partition("=")
        if not path or tag == args.base_tag:
            raise SystemExit(f"--src wants TAG=FILE with TAG != {args.base_tag}: {s}")
        paths.append(path)
        d = tidx.setdefault(tag, {})
        for t in GGUFReader(path, "r").tensors:
            if t.name in d:
                raise SystemExit(f"{t.name} is in two files of tier {tag}")
            d[t.name] = (len(paths) - 1, t)
    hdr, alloc = read_alloc(args.alloc)
    bnames = [t.name for t in base.tensors]
    extra = set(alloc) - set(bnames)
    if extra:
        raise SystemExit(f"alloc names tensors not in the base: {sorted(extra)[:5]}")

    plan = []  # (name, tag, ReaderTensor)
    for bt in base.tensors:
        tag = alloc.get(bt.name, args.base_tag)
        if tag not in tidx:
            raise SystemExit(f"{bt.name}: tier {tag} has no --src")
        fi, st = tidx[tag].get(bt.name, (None, None))
        if st is None:
            raise SystemExit(f"{bt.name}: not in any file of tier {tag}")
        if list(st.shape) != list(bt.shape):
            raise SystemExit(f"{bt.name}: shape {list(st.shape)} in tier {tag} vs {list(bt.shape)} in base")
        plan.append((bt.name, fi, st))
    tag_of = {bt.name: alloc.get(bt.name, args.base_tag) for bt in base.tensors}
    counts = Counter(tag_of.values())

    align = base.alignment
    kvs = [f for name, f in base.fields.items() if not name.startswith("GGUF.") and name != "aiq3.allocation"]
    note = f"{hdr} | " + ", ".join(f"{t}:{c}" for t, c in sorted(counts.items()))
    kv_blob = b"".join(kv_bytes(f) for f in kvs) + enc_str("aiq3.allocation") + struct.pack("<I", GGUF_STR) + enc_str(note)
    ti_blob, off, offs = b"", 0, []
    for name, fi, st in plan:
        nd = len(st.shape)
        ti_blob += enc_str(name) + struct.pack("<I", nd) + b"".join(struct.pack("<Q", int(d)) for d in st.shape)
        ti_blob += struct.pack("<IQ", int(st.tensor_type), off)
        offs.append(off)
        off += (int(st.n_bytes) + align - 1) // align * align
    head = b"GGUF" + struct.pack("<IQQ", 3, len(plan), len(kvs) + 1) + kv_blob + ti_blob
    pad = (-len(head)) % align
    data_start = len(head) + pad
    tmp = args.out + ".tmp"
    crcs = []
    with open(tmp, "wb") as fo:
        fo.write(head + b"\0" * pad)
        files = [open(p, "rb") for p in paths]
        for (name, fi, st), o in zip(plan, offs):
            assert fo.tell() == data_start + o
            n = int(st.n_bytes)
            crcs.append(copy_range(files[fi], int(st.data_offset), n, fo))
            fo.write(b"\0" * ((-n) % align))
        for f in files:
            f.close()
    if not args.no_verify:
        r = GGUFReader(tmp, "r")
        assert len(r.tensors) == len(plan)
        for (name, fi, st), rt, c in zip(plan, r.tensors, crcs):
            if rt.name != name or rt.tensor_type != st.tensor_type or int(rt.n_bytes) != int(st.n_bytes) \
                    or crc_range(tmp, int(rt.data_offset), int(rt.n_bytes)) != c:
                raise SystemExit(f"verify failed at {name}")
    os.replace(tmp, args.out)
    tot = sum(int(st.n_bytes) for _, _, st in plan)
    print(f"wrote {args.out}: {len(plan)} tensors, {tot / (1 << 20):.1f} MiB tensor data; " +
          ", ".join(f"{t}:{c}" for t, c in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
