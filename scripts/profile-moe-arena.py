#!/usr/bin/env python3
"""Measure Flash-Next routes on the coding, agentic, and RAG startup fixtures.

Run a server in an owned GPU-lock session with LLAMA_B1_CACHE_WARM=1 and
GGML_MOE_CACHE_PROFILE_SAVE pointing to --observed. Keep all experts on CPU
(--cpu-moe) so the profile contains the same 144 expert tensor descriptors.
The model can use either arena or mmap loading; profile identity is layout invariant.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8097')
    parser.add_argument('--fixtures', type=Path, default=Path(__file__).resolve().parents[1] / 'tests/fixtures/moe-profile/flash-next.jsonl')
    parser.add_argument('--observed', type=Path, required=True, help='server GGML_MOE_CACHE_PROFILE_SAVE path')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('PARENT_HOLDS_GPU_LOCK') != '1':
        parser.error('run inside the session holding gpu.lock with PARENT_HOLDS_GPU_LOCK=1')
    records = [json.loads(line) for line in args.fixtures.read_text().splitlines() if line.strip()]
    assert {r['category'] for r in records} == {'coding', 'agentic', 'rag'}
    prompt = '\n\n'.join('## ' + r['category'] + '\n' + r['prompt'] for r in records)

    def post(path, value):
        request = urllib.request.Request(args.url + path, json.dumps(value).encode(), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=480) as response:
            return json.load(response)

    tokens = post('/tokenize', {'content': prompt, 'add_special': True})['tokens']
    before = args.observed.stat().st_mtime_ns if args.observed.exists() else 0
    result = post('/completion', {'prompt': tokens, 'n_predict': 1, 'temperature': 1.0, 'seed': 11,
                                 'cache_prompt': False, 'stream': False, 'ignore_eos': True})
    assert args.observed.stat().st_mtime_ns > before, 'server did not persist new measured routes'
    data = args.observed.read_bytes()
    magic, tensors, entries, signature, checksum = struct.unpack_from('=5Q', data)
    assert magic == 0x31464F5250454F4D and tensors == 144 and entries == 48 * 512 * 3
    assert len(data) == 40 + entries * 8
    actual = 14695981039346656037
    for byte in data[40:]:
        actual = ((actual ^ byte) * 1099511628211) & ((1 << 64) - 1)
    assert actual == checksum
    counts = struct.unpack_from('=' + str(entries) + 'Q', data, 40)
    totals = [sum(counts[i:i + 512]) for i in range(0, len(counts), 512)]
    # The final layer may compute only output positions after row pruning.
    assert all(0 < total <= len(tokens) * 10 and total % 10 == 0 for total in totals)
    assert totals.count(len(tokens) * 10) >= 141, 'incomplete prompt route observations'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    provenance = {'fixtures_sha256': hashlib.sha256(args.fixtures.read_bytes()).hexdigest(),
                  'categories': [r['category'] for r in records], 'prompt_tokens': len(tokens),
                  'tensors': tensors, 'entries': entries, 'signature': hex(signature),
                  'count_sum': sum(counts), 'tensor_route_totals': totals, 'positive_entries': sum(c > 0 for c in counts),
                  'profile_sha256': hashlib.sha256(data).hexdigest(), 'completion_timings': result.get('timings'),
                  'math': 'original quantized weights; only routing counts are persisted'}
    args.output.with_suffix(args.output.suffix + '.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
