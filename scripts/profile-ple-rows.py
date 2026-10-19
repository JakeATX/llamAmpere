#!/usr/bin/env python3
"""Build a PLE startup row profile and replay coverage from saved token IDs.

This reads GGUF metadata only. Generated token IDs are historical evidence, not a
new inference run. A prompt-only profile is measured against held-out continuation;
a replay profile includes that continuation and is explicitly not predictive.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gguf-py'))
from gguf import GGUFReader


def ple_rows(tokens, meta):
    eos = meta['eos_token_id']
    multipliers = meta['layer_multipliers']
    previous = []
    mask = (1 << 64) - 1
    for token in tokens:
        context = [token]
        cut = False
        for step in range(1, meta['ngram_size']):
            value = previous[-step] if len(previous) >= step else -1
            cut = cut or value < 0 or value == eos
            context.append(eos if cut else value)
        indices = []
        for size in range(2, meta['ngram_size'] + 1):
            mixed = (context[0] * multipliers[0]) & mask
            for step in range(1, size):
                mixed ^= (context[step] * multipliers[step]) & mask
            for head in range((size - 2) * meta['heads_per_ngram'], (size - 1) * meta['heads_per_ngram']):
                indices.append(mixed % meta['head_vocab_sizes'][head] + meta['head_offsets'][head])
        yield indices
        previous.append(token)
        if len(previous) >= meta['ngram_size']:
            previous.pop(0)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--prompt', type=Path, required=True)
    parser.add_argument('--generated', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    reader = GGUFReader(args.model)
    meta = {name: reader.fields['qwen4exp.ple.' + name].contents() for name in (
        'eos_token_id', 'layer_multipliers', 'ngram_size', 'heads_per_ngram', 'head_vocab_sizes', 'head_offsets')}
    tensor = next(t for t in reader.tensors if t.name == 'per_layer_token_embd.weight')
    row_bytes = tensor.n_bytes // int(tensor.shape[1])
    prompt = json.loads(args.prompt.read_text())
    generated = json.loads(args.generated.read_text())
    for tokens in (prompt, generated):
        if not isinstance(tokens, list) or not all(type(t) is int and 0 <= t < 2**31 for t in tokens):
            raise ValueError('expected a list of nonnegative token IDs')
    seen = set()
    for rows in ple_rows(prompt, meta):
        seen.update(rows)
    profile = set(seen)
    missing = set()
    lookup_misses = 0
    tokens_with_misses = 0
    # Continue the prompt context across the boundary; restarting hashes EOS incorrectly.
    for i, rows in enumerate(ple_rows(prompt + generated, meta)):
        if i < len(prompt):
            continue
        absent = [r for r in rows if r not in profile]
        lookup_misses += len(absent)
        tokens_with_misses += bool(absent)
        missing.update(absent)
    args.out.mkdir(parents=True, exist_ok=True)
    def write_profile(name, rows):
        path = args.out / name
        header = f'LLAMA_PLE_ROWS_V1 {int(tensor.tensor_type)} {int(tensor.shape[0])} {int(tensor.shape[1])} {row_bytes} {len(rows)}\n'
        path.write_text(header + ''.join(f'{r}\n' for r in sorted(rows)))
        return {'path': str(path.resolve()), 'sha256': digest(path), 'rows': len(rows),
                'weight_bytes': len(rows) * row_bytes, 'index_bytes': len(rows) * 4,
                'table_bytes_avoided': tensor.n_bytes - len(rows) * row_bytes}
    result = {'method': 'CPU hash replay of historical token IDs; no new inference or speed evidence',
              'metadata': meta, 'model': str(args.model), 'table_bytes': tensor.n_bytes, 'row_bytes': row_bytes,
              'prompt': {'path': str(args.prompt), 'sha256': digest(args.prompt), 'tokens': len(prompt),
                         'row_lookups': len(prompt) * len(meta['head_offsets']), 'misses': 0},
              'generated': {'path': str(args.generated), 'sha256': digest(args.generated), 'tokens': len(generated),
                            'row_lookups': len(generated) * len(meta['head_offsets']), 'misses': lookup_misses,
                            'unique_missing_rows': len(missing), 'tokens_with_misses': tokens_with_misses},
              'prompt_profile': write_profile('prompt.rows', profile),
              'replay_profile': write_profile('replay.rows', profile | missing),
              'replay_warning': 'Replay includes known generated IDs; zero misses by construction, not evidence of future coverage. MTP rejected draft tokens may require additional rows.'}
    (args.out / 'coverage.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
