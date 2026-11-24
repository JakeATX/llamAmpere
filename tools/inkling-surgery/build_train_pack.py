#!/usr/bin/env python3
"""Build a clean, leak-free tokenized training pack for Inkling 2 Mini Phase 1.3a surgery.

Strictly excludes /home/jakek/inke21/ (E21 eval corpus). Uses:
  1. Real local C++/Python/docs source files from the repo (packed into 4K..8K KL sequences).
  2. Diverse synthetic long-context multi-needle / cross-file retrieval tasks at 8K..12K
     with ground-truth answer spans placed at distances d in [1500, 10500] so both
     SWA-4K (d in [1024, 4096)) and LSA (d >= 4096) receive clean retrieval gradients.
"""
from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path
from typing import Dict, List

import torch
from tokenizers import Tokenizer


def collect_source_texts(repo_root: Path, max_files: int = 300) -> List[str]:
    exts = {".cpp", ".h", ".c", ".py", ".md", ".cmake", ".sh"}
    paths: List[Path] = []
    for sub in ("src", "common", "ggml/src", "tools", "docs"):
        d = repo_root / sub
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if not p.is_file() or p.suffix not in exts:
                continue
            sp = str(p)
            if "inke21" in sp or ".git" in sp or "build" in sp:
                continue
            sz = p.stat().st_size
            if 1024 <= sz <= 120_000:
                paths.append(p)
    rng = random.Random(20261124)
    rng.shuffle(paths)
    texts = []
    for p in paths[:max_files]:
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            if len(txt) >= 1000:
                rel = p.relative_to(repo_root)
                texts.append(f"// FILE: {rel}\n{txt}\n")
        except OSError:
            continue
    return texts


def build_kl_sequences(tokenizer: Tokenizer, texts: List[str], n_seqs: int, seq_len: int) -> List[torch.Tensor]:
    all_ids: List[int] = []
    for txt in texts:
        all_ids.extend(tokenizer.encode(txt, add_special_tokens=False).ids)
        if len(all_ids) >= (n_seqs + 2) * seq_len:
            break
    seqs: List[torch.Tensor] = []
    stride = max(seq_len, (len(all_ids) - seq_len) // max(1, n_seqs))
    for i in range(n_seqs):
        start = (i * stride) % max(1, len(all_ids) - seq_len)
        chunk = all_ids[start : start + seq_len]
        if len(chunk) == seq_len:
            seqs.append(torch.tensor(chunk, dtype=torch.int32))
    return seqs


def build_long_retrieval_items(
    tokenizer: Tokenizer,
    filler_chunks: List[List[int]],
    n_items: int,
    target_len: int,
    seed: int = 20261124,
) -> List[Dict[str, torch.Tensor]]:
    """Build diverse 8K..12K multi-fact / cross-file retrieval items with exact answer-span masks.

    Places facts both in the 1.2K..3.6K mid-range (training SWA-4K) and in the 4.5K..target_len-400
    far-range (training LSA) on every item, avoiding safety-trigger words.
    """
    rng = random.Random(seed)
    items: List[Dict[str, torch.Tensor]] = []

    domains = [
        ("sensor calibration table", "calibration_code", "station"),
        ("compiler build manifest", "artifact_tag", "module"),
        ("telemetry archive log", "checksum_token", "node"),
        ("database migration ledger", "schema_revision", "shard"),
        ("routing table snapshot", "gateway_label", "region"),
    ]

    for idx in range(n_items):
        domain, field_name, entity_name = domains[idx % len(domains)]
        # Generate 4 unique alphanumeric values (2 placed in far-range >=4.5K, 2 placed in mid-range 1.2K..3.5K)
        facts = []
        for f_i in range(4):
            h = hashlib.sha256(f"{seed}:{idx}:{f_i}".encode()).hexdigest()
            ent = f"{entity_name}_{h[:4].upper()}"
            val = f"{h[4:8].upper()}-{int(h[8:12], 16) % 9000 + 1000}"
            facts.append((ent, val))

        # Assemble filler tokens of length ~target_len - 220
        filler_budget = target_len - 260
        buf: List[int] = []
        while len(buf) < filler_budget:
            ch = filler_chunks[rng.randrange(len(filler_chunks))]
            buf.extend(ch)
        buf = buf[:filler_budget]

        # Insert 2 far facts (near beginning / 25% of doc -> distance ~5K..8K) and
        # 2 mid facts (at 62% and 78% of doc -> distance ~1.6K..3.3K)
        positions = [
            int(filler_budget * rng.uniform(0.04, 0.14)),
            int(filler_budget * rng.uniform(0.22, 0.36)),
            int(filler_budget * rng.uniform(0.58, 0.68)),
            int(filler_budget * rng.uniform(0.75, 0.85)),
        ]
        # Insert from back to front so earlier indices stay valid
        for (ent, val), pos in sorted(zip(facts, positions), key=lambda x: x[1], reverse=True):
            note = f"\n# [{domain}] record for {ent}: {field_name} = {val}\n"
            note_ids = tokenizer.encode(note, add_special_tokens=False).ids
            buf[pos:pos] = note_ids

        # Trim tail slightly if insertions pushed beyond filler_budget (facts stay well inside budget)
        if len(buf) > filler_budget:
            buf = buf[:filler_budget]

        # Query all 4 facts in randomized order at the end; compute loss ONLY on the answer values
        order = list(range(4))
        rng.shuffle(order)
        prompt_ids = list(buf)
        loss_mask = [False] * len(prompt_ids)

        header = f"\n\n### {domain} lookup summary\n"
        h_ids = tokenizer.encode(header, add_special_tokens=False).ids
        prompt_ids.extend(h_ids)
        loss_mask.extend([False] * len(h_ids))

        for j in order:
            ent, val = facts[j]
            q_txt = f"The {field_name} for {ent} is "
            a_txt = f"{val}.\n"
            q_ids = tokenizer.encode(q_txt, add_special_tokens=False).ids
            a_ids = tokenizer.encode(a_txt, add_special_tokens=False).ids
            prompt_ids.extend(q_ids)
            loss_mask.extend([False] * len(q_ids))
            prompt_ids.extend(a_ids)
            loss_mask.extend([True] * len(a_ids))

        if len(prompt_ids) > target_len:
            excess = len(prompt_ids) - target_len
            prompt_ids = prompt_ids[excess:]
            loss_mask = loss_mask[excess:]

        items.append(
            {
                "ids": torch.tensor(prompt_ids, dtype=torch.int32),
                "loss_mask": torch.tensor(loss_mask, dtype=torch.bool),
            }
        )
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, default="/home/jakek/agy-inkmod/wt")
    ap.add_argument("--tokenizer", type=str, default="/mnt/data/inkling2/hf-full/tokenizer.json")
    ap.add_argument("--out", type=str, default="/home/jakek/agy-inkmod/out/phase1_3a/train_pack.pt")
    ap.add_argument("--n-kl", type=int, default=64)
    ap.add_argument("--kl-len", type=int, default=4096)
    ap.add_argument("--n-ce", type=int, default=64)
    ap.add_argument("--ce-len", type=int, default=8192)
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    texts = collect_source_texts(Path(args.repo_root))
    print(f"[pack] collected {len(texts)} clean source files (zero E21 overlap)")

    kl_seqs = build_kl_sequences(tok, texts, args.n_kl, args.kl_len)
    # Build 512-token filler chunks from second half of texts to keep KL and CE filler varied
    filler_chunks: List[List[int]] = []
    for txt in texts[::-1]:
        ids = tok.encode(txt, add_special_tokens=False).ids
        for i in range(0, len(ids) - 512, 512):
            filler_chunks.append(ids[i : i + 512])
        if len(filler_chunks) >= 400:
            break

    ce_items = build_long_retrieval_items(tok, filler_chunks, args.n_ce, args.ce_len)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kl_seqs": kl_seqs,
            "ce_items": ce_items,
            "meta": {"n_kl": len(kl_seqs), "kl_len": args.kl_len, "n_ce": len(ce_items), "ce_len": args.ce_len},
        },
        out_path,
    )
    sz_mb = out_path.stat().st_size / (1024 * 1024)
    avg_mask = sum(int(it["loss_mask"].sum().item()) for it in ce_items) / max(1, len(ce_items))
    print(f"[pack] wrote {out_path} ({sz_mb:.2f} MB) | kl={len(kl_seqs)}x{args.kl_len} ce={len(ce_items)}x{args.ce_len} avg_ce_toks={avg_mask:.1f}")


if __name__ == "__main__":
    main()
