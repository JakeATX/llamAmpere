"""Summarize one completed full run; this never declares a comparative winner."""
import hashlib
import json
import math
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
result = json.loads((root / 'result.json').read_text())
command = json.loads((root / 'command.json').read_text())
request = json.loads((root / 'request.json').read_text())
assert result['status'] == 'passed' and not result['violations'], 'incomplete or guarded run'
assert len(request['prompt']) in (8192, 100000) and request['n_predict'] == 5120
assert result['received_tokens'] == result['final']['tokens_predicted'] == 5120
assert result['final']['tokens_evaluated'] == len(request['prompt'])
timings = result['final']['timings']
assert timings['cache_n'] == 0 and timings['prompt_n'] == len(request['prompt']), 'warm prompt is not a full-prefill result'
assert request['temperature'] == 1 and request['ignore_eos']
argv = command['argv']
for index, arg in enumerate(argv):
    if arg == '--moe-skip-tail':
        assert float(argv[index+1]) == 0, 'lossy skipping is outside the lossless board'
    if arg.startswith('--moe-skip-tail='):
        assert float(arg.split('=', 1)[1]) == 0, 'lossy skipping is outside the lossless board'
for name, value in command['env'].items():
    if 'SKIP_TAIL' in name:
        assert value.lower() in ('', '0', '0.0', 'off', 'false'), 'lossy skip environment'
assert argv[argv.index('-ctk')+1] == argv[argv.index('-ctv')+1] == 'sjkvarn4'
assert argv[argv.index('--spec-type')+1].startswith('draft-mtp')
for phase in ('start', 'generation-start', 'generation-end', 'end'):
    assert json.loads((root / ('cpu-accounting-'+phase+'.json')).read_text())['cpu.weight'] == '100'
bundle = Path(argv[0]).resolve().parent.parent
manifest = json.loads((bundle / 'manifest.json').read_text())
assert manifest['source_commit'] == command['source_commit']
assert hashlib.sha256(Path(argv[0]).read_bytes()).hexdigest() == command['server_sha256'] == manifest['libraries']['llama-server']['sha256']
loaded = json.loads((root / 'cold.libraries.json').read_text())
assert len(loaded) >= 4
for name, record in loaded.items():
    path = Path(record['path']).resolve()
    assert path.parent == bundle / 'bin'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'] == manifest['libraries'][name]['sha256']
if command['profile_sha256']:
    profile = root / 'expert-profile-input.bin'
    assert Path(command['env']['GGML_MOE_CACHE_PROFILE_LOAD']).resolve() == profile
    assert hashlib.sha256(profile.read_bytes()).hexdigest() == command['profile_sha256']
lease = json.loads((root / 'lease.json').read_text())
lease_seconds = lease['end_unix_s'] - lease['start_unix_s']
assert 0 < lease_seconds <= 1500
rows = [json.loads(line) for line in (root / 'resources.jsonl').read_text().splitlines()]
assert rows and all(row['vram'] <= 23552 and row['available'] >= 10*1024**3 and abs(row['power_limit_w']-350) <= .1 for row in rows)
generation = [row for row in rows if result['request_start_s'] <= row['elapsed'] <= result['request_end_s']]
assert generation
phase_means = {}
for phase in ('prefill', 'decode'):
    samples = [row['power_w'] for row in generation if row['phase'] == phase]
    if samples:
        phase_means[phase] = {'samples': len(samples), 'mean_gpu_w': sum(samples)/len(samples)}
summary = dict(status='complete run; unpaired, no speed promotion', source_commit=command['source_commit'],
               prompt_tokens=len(request['prompt']), generated_tokens=5120,
               prefill_tps=timings['prompt_per_second'], decode_tps=timings['predicted_per_second'],
               ttc_s=result['generation_wall_s'], gpu_request_mean_w=sum(row['power_w'] for row in generation)/len(generation),
               gpu_phase_means=phase_means, expert_counters=result['expert_cache_counters'],
               peak_vram_mib=max(row['vram'] for row in rows),
               generation_peak_vram_mib=max(row['vram'] for row in generation),
               min_available_gib=min(row['available'] for row in rows)/1024**3,
               lease_s=lease_seconds, kl_previous=None, reload_checked=result['reload_checked'])
if timings.get('draft_n'):
    summary['draft_acceptance'] = timings['draft_n_accepted']/timings['draft_n']
if result['reload_checked']:
    assert result['serialized_bytes_equal'] and all(result['serialized_bytes_equal'].values()), 'saved bytes changed'
    summary['serialized_bytes_equal'] = result['serialized_bytes_equal']
    summary['reload_tokens_exact'] = result['reload_tokens_exact']
for name in ('prefill_tps', 'decode_tps', 'ttc_s', 'gpu_request_mean_w'):
    assert math.isfinite(summary[name]) and summary[name] > 0, name
(root / 'summary-auto.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
