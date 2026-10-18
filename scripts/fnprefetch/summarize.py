import json, math, re, sys
from pathlib import Path
import numpy as np

root = Path('/mnt/models/fnprefetch/protocol/runs')
pairs = []
last_end = -math.inf
reference = {}
evidence = {key: {'log_e': 0., 'max_log_e': 0.} for key in ['prefill_tokens_per_second', 'decode_tokens_per_second']}
for label in sys.argv[1:]:
    arms = {}
    times = {}
    for arm in ['off', 'prefetch']:
        cell = root / f'{label}-{arm}'
        result = json.loads((cell / 'result.json').read_text())
        final = result.get('final', {})
        assert result['status'] == 'complete'
        assert final['tokens_evaluated'] == 100000 and final['tokens_predicted'] == 5120
        assert abs(final['generation_settings']['temperature'] - 1) < 1e-6
        command = json.loads((cell / 'command.json').read_text())
        request = json.loads((cell / 'request.json').read_text())
        assert result['profile_provenance']==command['profile_provenance']
        profile_digest=command['profile_provenance']['sha256']
        assert len(profile_digest)==64 and command['profile_provenance']['bytes']>0
        if 'profile' in reference:assert profile_digest==reference['profile'], 'profile changed between arms or pairs'
        else:reference['profile']=profile_digest
        assert len(request['prompt']) == 100000 and request['n_predict'] == 5120
        assert request['temperature'] == 1.0
        argv = command['argv']
        assert argv[argv.index('--spec-type') + 1] == 'draft-mtp-adaptive' and '-md' in argv
        assert argv[argv.index('-ctk') + 1] == argv[argv.index('-ctv') + 1] == 'sjkvarn4'
        assert result['binary_provenance']['libraries'] and all(lib['mapped_inode_matches'] for lib in result['binary_provenance']['libraries'].values())
        samples = [json.loads(line) for line in (cell / 'samples.jsonl').read_text().splitlines()]
        request_samples = [sample for sample in samples if result['request_start_monotonic_s'] <= sample['t'] <= result['request_end_monotonic_s']]
        decode = [sample for sample in request_samples if sample['phase'] == 'decode']
        prefill = [sample for sample in request_samples if sample['phase'] == 'prefill']
        assert samples and decode
        assert all(sample['vram_mib'] <= 23552 and sample['avail_kib'] >= 10*1024**2 for sample in samples)
        assert all(abs(sample['power_limit'] - 350) < .5 for sample in samples)
        times[arm] = (samples[0]['t'], samples[-1]['t'])
        identity = {'server': result['binary_provenance']['server']['sha256'],
                    'libraries': {Path(k).name:v['sha256'] for k,v in result['binary_provenance']['libraries'].items()},
                    'profile': profile_digest, 'argv': command['argv'], 'env': command['env'], 'request': request,'runner': command['runner_provenance']['sha256']}
        if arm in reference:
            assert identity == reference[arm], f'{arm} build or settings changed'
        else:
            reference[arm] = identity
        log = (cell / 'server.log').read_text()
        hits = re.findall(r'CUDA\d+ hits=(\d+)/(\d+)', log)
        device_hits = re.findall(r'device routes: hits=(\d+) misses=(\d+)', log)
        hit_rate = None
        if device_hits:
            h, m = map(int, device_hits[-1]); hit_rate = h / (h + m) if h + m else None
        elif hits:
            h, n = map(int, hits[-1]); hit_rate = h / n if n else None
        assert hit_rate is not None and 0 <= hit_rate <= 1, 'missing or invalid expert-hit counters'
        timings = final['timings']
        assert all(math.isfinite(timings[k]) and timings[k]>0 for k in ('prompt_per_second','predicted_per_second'))
        assert (log.count('outcome=used')>0) if arm=='prefetch' else ('[moe-prefetch]' not in log), 'inactive prefetch or active off arm'
        arms[arm] = {
            'prefill_tokens_per_second': timings['prompt_per_second'],
            'decode_tokens_per_second': timings['predicted_per_second'],
            'time_to_completion_seconds': result['wall_s'],
            'first_token_seconds': result['first_token_s'],
            'prefill_power_watts': float(np.mean([sample['power_draw'] for sample in prefill])) if prefill else None,
            'decode_power_watts': float(np.mean([sample['power_draw'] for sample in decode])),
            'peak_vram_mib': result['peaks']['vram_mib'],
            'expert_hit_rate': hit_rate,
            'expert_hit_rate_scope': 'last cumulative expert-projection snapshot; target/helper not separated',
            'copies': log.count('[moe-prefetch]'),
            'used_copies': log.count('outcome=used'),
            'canceled_copies': log.count('canceled=1'),
            'server_sha256': result['binary_provenance']['server']['sha256'],
        }
    base_first = times['off'][1] < times['prefetch'][0]
    candidate_first = times['prefetch'][1] < times['off'][0]
    assert base_first or candidate_first, 'overlapping arms'
    assert min(t[0] for t in times.values()) > last_end, 'pairs must be chronological'
    last_end = max(t[1] for t in times.values())
    if pairs:
        assert base_first != pairs[-1]['baseline_first'], 'alternate AB / BA order'
    assert reference['off']['request'] == reference['prefetch']['request'], 'sampling or input differs'
    assert reference['off']['argv'][1:]==reference['prefetch']['argv'][1:], 'model arguments differ between arms'
    base_env=dict(reference['off']['env']);candidate_env=dict(reference['prefetch']['env'])
    assert base_env.pop('LLAMA_ARG_MOE_PREFETCH')=='off'
    assert candidate_env.pop('LLAMA_ARG_MOE_PREFETCH') in ('1','2','3','auto')
    assert base_env==candidate_env, 'runtime settings differ between arms'
    arms['baseline_first'] = base_first
    for metric, ev in evidence.items():
        ev['log_e'] += math.log(1.8 if arms['prefetch'][metric] > arms['off'][metric] else .2)
        ev['max_log_e'] = max(ev['max_log_e'], ev['log_e'])
    pairs.append(arms)

report = {'pairs': pairs, 'independent_pairs': len(pairs)}
if pairs:
    for metric in ['prefill_tokens_per_second', 'decode_tokens_per_second']:
        before = [pair['off'][metric] for pair in pairs]
        after = [pair['prefetch'][metric] for pair in pairs]
        report[metric] = {'baseline_mean': float(np.mean(before)), 'prefetch_mean': float(np.mean(after)),
                          'mean_gain_pct': float(100 * (np.mean(after) / np.mean(before) - 1)),
                          'one_sided_anytime_p_bound': math.exp(-evidence[metric]['max_log_e']),
                          'passes_0_01': math.exp(-evidence[metric]['max_log_e']) <= .01}
report['gate'] = 'paired sign e-process: fixed 0.9-vs-0.5 bet; ties count as losses'
output = root.parent / 'full-protocol-summary.json'
output.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
