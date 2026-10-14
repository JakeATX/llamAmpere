#!/usr/bin/env python3
"""Check two AI+ bit lists against the template's supported weight matrices.

A whole-bank SC4+1 plan is not an exact SC5 plan. Resolve a kept or omitted
output head with --reference-head-bits / --candidate-head-bits.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
from gguf import GGUFReader  # noqa: E402

BODY = r"blk\.\d+\.(attn_qkv|attn_gate|ssm_out|attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down)\.weight"


def read_plan(path, head_bits=None):
    plan = {}
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        if len(fields) not in (2, 3):
            raise ValueError(f"{path}:{line_number}: expected name, type, and optional bank")
        name, quant_type = fields[:2]
        if name in plan:
            raise ValueError(f"{path}:{line_number}: duplicate {name}")
        match = re.fullmatch(r"ai([2-6])(?:un|uo|u|s)?", quant_type)
        if match:
            bits = int(match[1])
        elif quant_type in ("q6_k", "q8_0"):
            bits = 6 if quant_type == "q6_k" else 8
        elif name == "output.weight" and quant_type == "keep" and head_bits is not None:
            bits = head_bits
        else:
            raise ValueError(f"{path}:{line_number}: unresolved type {quant_type}; set explicit head bits for keep")
        plan[name] = bits
    if head_bits is not None:
        if "output.weight" in plan and plan["output.weight"] != head_bits:
            raise ValueError(f"{path}: explicit head bits disagree with listed head")
        plan["output.weight"] = head_bits
    return plan


def compare(reference, candidate, tensors, body_layers=None):
    expected = {name for name in tensors if name == "output.weight" or (re.fullmatch(BODY, name)
                and (body_layers is None or int(name.split(".")[1]) < body_layers))}
    errors = []
    if "output.weight" not in expected:
        errors.append("template has no output.weight")
    for name in sorted(expected):
        if len(tensors[name]) != 2 or min(tensors[name]) <= 0:
            errors.append(f"template has invalid matrix shape for {name}")
    for label, plan in (("reference", reference), ("candidate", candidate)):
        for name in sorted(expected - plan.keys()):
            errors.append(f"{label} missing {name}")
        for name in sorted(plan.keys() - expected):
            errors.append(f"{label} unexpected {name}")
    mismatch = [{"tensor": name, "reference_bits": reference[name], "candidate_bits": candidate[name],
                 "parameters": math.prod(tensors[name])}
                for name in sorted(reference.keys() & candidate.keys() & expected)
                if reference[name] != candidate[name]]
    errors.extend(f"tier mismatch {m['tensor']}: {m['reference_bits']} vs {m['candidate_bits']}" for m in mismatch)
    return {"exact": not errors, "body_layers": body_layers, "expected_tensors": len(expected), "reference_tensors": len(reference),
            "candidate_tensors": len(candidate), "mismatched_tensors": len(mismatch),
            "mismatched_parameters": sum(m["parameters"] for m in mismatch), "mismatches": mismatch,
            "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--template", default=os.environ.get("AIB_TEMPLATE"))
    parser.add_argument("--reference-head-bits", type=int, choices=(2, 3, 4, 5, 6, 8))
    parser.add_argument("--candidate-head-bits", type=int, choices=(2, 3, 4, 5, 6, 8))
    args = parser.parse_args()
    if not args.template:
        parser.error("set --template or AIB_TEMPLATE to validate full tensor coverage")
    try:
        reference = read_plan(args.reference, args.reference_head_bits)
        candidate = read_plan(args.candidate, args.candidate_head_bits)
        reader = GGUFReader(args.template, "r")
        architecture_field = reader.get_field("general.architecture")
        if architecture_field is None:
            raise ValueError("template has no architecture")
        architecture = architecture_field.contents()
        blocks = reader.get_field(f"{architecture}.block_count")
        if blocks is None:
            raise ValueError("template has no block count")
        block_count = int(blocks.contents())
        nextn = reader.get_field(f"{architecture}.nextn_predict_layers")
        body_layers = block_count - (int(nextn.contents()) if nextn else 0)
        if body_layers <= 0:
            raise ValueError("template has no main-model blocks")
        tensors = {tensor.name: tuple(int(d) for d in tensor.shape) for tensor in reader.tensors}
        result = compare(reference, candidate, tensors, body_layers)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    return 0 if result["exact"] else 1


if __name__ == "__main__":
    sys.exit(main())
