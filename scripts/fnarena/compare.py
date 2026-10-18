"""Accept only complete, matched, interleaved 100K/5120 protocol pairs."""
import hashlib
import json
import math
from pathlib import Path
import sys


def compare(pairs):
    assert len(pairs) in (8, 16), 'two planned statistical looks only'
    identities = {}
    metrics = {name: [] for name in ('prefill', 'decode', 'ttc')}
    seen = set()
    last_end = 0
    last_candidate = None
    def read(path, arm):
        path = Path(path).resolve()
        assert path not in seen, 'reused arm'
        seen.add(path)
        root = path.parent
        value = json.loads(path.read_text())
        assert value['status'] == 'passed' and not value['violations']
        assert value['final']['tokens_evaluated'] == 100000 and value['final']['tokens_predicted'] == 5120
        assert value['resources']['peak_vram_mib'] <= 23552 and value['resources']['min_available_bytes'] >= 10 * 2**30
        assert value['expert_cache_counters']['host'], 'missing expert-hit evidence'
        request = json.loads((root / 'request.json').read_text())
        assert request['temperature'] == 1.0 and request['n_predict'] == 5120 and len(request['prompt']) == 100000
        for name in ('start', 'generation-start', 'generation-end', 'end'):
            assert json.loads((root / ('cpu-accounting-' + name + '.json')).read_text())['cpu.weight'] == '100'
        command = json.loads((root / 'command.json').read_text())
        argv = command['argv'][:]
        profile = argv[argv.index('--moe-profile') + 1] if '--moe-profile' in argv else command['env'].get('GGML_MOE_CACHE_PROFILE_LOAD')
        assert Path(profile).resolve() == root / 'expert-profile-input.bin'
        profile_hash = hashlib.sha256(Path(profile).read_bytes()).hexdigest()
        assert profile_hash == command['profile_sha256']
        bundle = Path(argv[0]).parent.parent
        manifest = json.loads((bundle / 'manifest.json').read_text())
        assert command['source_commit'] == manifest['source_commit']
        assert command['server_sha256'] == manifest['libraries']['llama-server']['sha256']
        loaded = json.loads((root / 'cold.libraries.json').read_text())
        assert len(loaded) >= 4
        for name, record in loaded.items():
            assert Path(record['path']).resolve().parent == bundle / 'bin'
            assert record['sha256'] == manifest['libraries'][name]['sha256']
        for key, setting in command['env'].items():
            if 'SKIP' in key:
                assert setting in ('0', '0.0', 'off', ''), 'lossy skip setting'
        assert '--moe-skip-tail' not in argv or argv[argv.index('--moe-skip-tail') + 1] == '0'
        if arm == 'candidate':
            assert argv[argv.index('--moe-arena') + 1] == 'on'
        for flag in ('--moe-profile', '--moe-arena'):
            if flag in argv:
                index = argv.index(flag)
                del argv[index:index+2]
        argv[0] = '<server>'
        argv[argv.index('--slot-save-path') + 1] = '<cache>'
        environment = {key: value for key, value in command['env'].items() if key not in ('GGML_MOE_CACHE_PROFILE_LOAD', 'LD_LIBRARY_PATH')}
        identity = {'commit': command['source_commit'], 'server': command['server_sha256'], 'argv': argv, 'env': environment, 'profile': profile_hash}
        if arm in identities:
            assert identity == identities[arm], 'runtime or settings changed within arm'
        identities[arm] = identity
        lease = json.loads((root / 'lease.json').read_text())
        assert 0 < lease['end_unix_s'] - lease['start_unix_s'] <= 1500
        return value, request, lease
    for index, pair in enumerate(pairs):
        previous, previous_request, previous_lease = read(pair['previous'], 'previous')
        candidate, candidate_request, candidate_lease = read(pair['candidate'], 'candidate')
        assert previous_request == candidate_request
        first, second = (previous_lease, candidate_lease) if index % 2 == 0 else (candidate_lease, previous_lease)
        assert last_end <= first['start_unix_s'] and first['end_unix_s'] <= second['start_unix_s']
        last_end = second['end_unix_s']
        for key, field in (('prefill', 'prompt_per_second'), ('decode', 'predicted_per_second')):
            metrics[key].append(candidate['final']['timings'][field] / previous['final']['timings'][field])
        metrics['ttc'].append(previous['generation_wall_s'] / candidate['generation_wall_s'])
        last_candidate = candidate
    for key in ('argv', 'env', 'profile'):
        assert identities['previous'][key] == identities['candidate'][key], 'unmatched arm configuration'
    result = {'pairs': len(pairs), 'test': 'one-sided exact sign, two planned looks with Bonferroni correction', 'metrics': {}}
    for key, ratios in metrics.items():
        wins = sum(ratio > 1 for ratio in ratios)
        p = sum(math.comb(len(ratios), count) for count in range(wins, len(ratios) + 1)) / 2**len(ratios)
        result['metrics'][key] = {'wins': wins, 'adjusted_p': min(1.0, 2 * p), 'geometric_ratio': math.exp(sum(math.log(ratio) for ratio in ratios) / len(ratios))}
    result['relative_speed_passed'] = all(result['metrics'][key]['adjusted_p'] <= .01 and result['metrics'][key]['geometric_ratio'] > 1 for key in ('prefill', 'decode')) and result['metrics']['ttc']['geometric_ratio'] >= 1
    result['last_candidate_timings'] = last_candidate['final']['timings']
    return result


if __name__ == '__main__':
    print(json.dumps(compare(json.loads(Path(sys.argv[1]).read_text())['pairs']), indent=2))
