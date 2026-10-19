"""Measure teacher-forced, full-vocabulary KL with the same prompt in both arms."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def measure(post, output, prompt_n):
    ids = json.loads(Path('/mnt/models/stratamatch-home/prompt-ids.json').read_text())[:prompt_n]
    teacher = [int(x) for x in Path('/home/jake-k/qwen38-bench/kernel_cache_investigation/frontier/BACKLOG_RUN_20261015/FNFAST/teacher-tokens.txt').read_text().split()][:16]
    floor = float(np.nextafter(np.float32(0), np.float32(1)))
    identity = {'prompt_n': prompt_n, 'prompt_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(), 'tokens': teacher, 'probability_floor': floor}
    (output / 'teacher.json').write_text(json.dumps(identity, indent=2))
    rows = []
    for i, token in enumerate(teacher):
        request = dict(id_slot=0, prompt=ids + teacher[:i], n_predict=1, temperature=1.0,
                       seed=11, ignore_eos=True, cache_prompt=True, stream=False, return_tokens=True,
                       n_probs=500000, post_sampling_probs=False, top_k=40, top_p=.95, min_p=0,
                       repeat_penalty=1.0, samplers=['top_k', 'top_p', 'temperature'], logit_bias=[[token, 1000000.0]])
        response = post('/completion', request)[1]
        assert response['tokens'] == [token], 'teacher token was not forced'
        probabilities = response['completion_probabilities'][0]['top_logprobs']
        row = np.empty(len(probabilities), dtype=np.float64)
        seen = set()
        for item in probabilities:
            token_id = item['id']
            assert 0 <= token_id < len(row) and token_id not in seen
            seen.add(token_id)
            row[token_id] = item['logprob']
        assert len(seen) == len(row) and len(row) > 200000
        assert np.isfinite(row).all()
        row = np.maximum(row, np.log(floor))
        row -= np.logaddexp.reduce(row)
        rows.append(row)
        np.save(output / 'logprobs-partial.npy', np.stack(rows))
    current = np.stack(rows)
    np.save(output / 'logprobs.npy', current)
    result = {'quality_rows_completed': len(rows), 'probability_source': 'normalized full-vocabulary pre-sampling API probabilities; float32 zero values floored'}
    previous = os.environ.get('SHIP_PREVIOUS_KL')
    if previous:
        previous = Path(previous)
        assert json.loads((previous.parent / 'teacher.json').read_text()) == identity
        baseline = np.load(previous)
        assert baseline.shape == current.shape
        divergences = np.sum(np.exp(baseline) * (baseline - current), axis=1)
        assert divergences.min() >= -1e-10
        divergences = np.maximum(divergences, 0)
        result['kl_previous_to_current'] = {'mean': float(divergences.mean()), 'maximum': float(divergences.max()), 'rows': divergences.tolist()}
    return result
