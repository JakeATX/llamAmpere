"""Exercise evidence acceptance/refusal; synthetic timings are never board results."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

R = Path('/mnt/models/fnship/artifacts')
source = R / 'full-e3-100k-r1024-r1'
root = Path(tempfile.mkdtemp(prefix='paired-validator-', dir=R))
manifest = {'pairs': []}

def write(path, value):
    path.write_text(json.dumps(value))

for i in range(8):
    pair = {}
    for arm in ('previous', 'candidate'):
        run = root / f'{i}-{arm}'
        run.mkdir()
        for name in ('result.json', 'request.json', 'command.json', 'cold.libraries.json',
                     'cpu-accounting-start.json', 'cpu-accounting-generation-start.json',
                     'cpu-accounting-generation-end.json', 'cpu-accounting-end.json'):
            shutil.copyfile(source / name, run / name)
        shutil.copyfile(source / 'expert-profile-input.bin', run / 'expert-profile-input.bin')
        command = json.loads((run / 'command.json').read_text())
        command['env']['GGML_MOE_CACHE_PROFILE_LOAD'] = str(run / 'expert-profile-input.bin')
        command['argv'][command['argv'].index('--slot-save-path') + 1] = str(run / 'cache')
        write(run / 'command.json', command)
        value = json.loads((run / 'result.json').read_text())
        if arm == 'candidate':
            for metric in ('prompt_per_second', 'predicted_per_second'):
                value['final']['timings'][metric] *= 1.1
            value['generation_wall_s'] /= 1.1
        write(run / 'result.json', value)
        order = int((arm == 'candidate') != bool(i % 2))
        begin = 1000 + 1000*i + 400*order
        write(run / 'lease.json', {'start_unix_s': begin, 'end_unix_s': begin + 350})
        pair[arm] = str(run / 'result.json')
    manifest['pairs'].append(pair)

path = root / 'manifest.json'
write(path, manifest)
checks = []

def invoke(label, accepted, speed_passed=None):
    result = subprocess.run(['python3', str(Path(__file__).with_name('paired-speed.py')), str(path)],
                            text=True, capture_output=True)
    assert (result.returncode == 0) == accepted, (label, result.stdout, result.stderr)
    output = json.loads(result.stdout) if accepted else {'refusal': result.stderr.splitlines()[-1]}
    if speed_passed is not None:
        assert output['lossless_speed_gate_passed'] == speed_passed, label
    if accepted:
        assert output['release_throughput_targets_met'] is False, 'synthetic below-target data cannot satisfy RC targets'
    checks.append({'check': label, 'passed': True, 'output': output})

invoke('eight complete synthetic wins', True, True)

def changed_json(file, mutate, label, accepted=False, speed_passed=None):
    original = file.read_bytes()
    value = json.loads(original)
    mutate(value)
    write(file, value)
    try:
        invoke(label, accepted, speed_passed)
    finally:
        file.write_bytes(original)

first = Path(manifest['pairs'][0]['candidate']).parent
changed_json(first / 'result.json', lambda x: x.update(status='failed'), 'incomplete arm')
changed_json(first / 'result.json', lambda x: x.update(received_tokens=5119), 'short streamed output')
changed_json(first / 'result.json', lambda x: x['final']['timings'].update(predicted_per_second=float('nan')), 'nonfinite throughput')
changed_json(first / 'command.json', lambda x: x['argv'].extend(['--moe-skip-tail', '1']), 'lossy skip')
changed_json(first / 'command.json', lambda x: x['env'].update(LLAMA_MOE_SKIP_TAIL='1'), 'lossy env')
changed_json(first / 'command.json', lambda x: x.update(profile_sha256='0'*64), 'changed profile')
changed_json(first / 'cpu-accounting-generation-start.json', lambda x: x.update({'cpu.weight': '10'}), 'low CPU weight')
changed_json(first / 'lease.json', lambda x: x.update(start_unix_s=999), 'overlapping lease')
changed_json(path, lambda x: x['pairs'][1].update(candidate=x['pairs'][0]['candidate']), 'reused run')

# CPU decode gains can improve full-request TTC while leaving prefill unchanged.
originals = {}
for pair in manifest['pairs']:
    candidate = Path(pair['candidate'])
    originals[candidate] = candidate.read_bytes()
    value = json.loads(originals[candidate])
    previous = json.loads(Path(pair['previous']).read_text())
    value['final']['timings']['prompt_per_second'] = previous['final']['timings']['prompt_per_second']
    write(candidate, value)
try:
    invoke('flat prefill with improved full-request TTC', True, True)
finally:
    for file, data in originals.items():
        file.write_bytes(data)

# Higher reported kernel rates alone cannot pass the predeclared request-TTC test.
for pair in manifest['pairs']:
    candidate = Path(pair['candidate'])
    value = json.loads(candidate.read_text())
    previous = json.loads(Path(pair['previous']).read_text())
    value['generation_wall_s'] = previous['generation_wall_s']
    write(candidate, value)
try:
    invoke('unchanged request TTC prevents speed promotion', True, False)
finally:
    for file, data in originals.items():
        file.write_bytes(data)

write(root / 'checks.json', {'synthetic_only': True, 'checks': checks})
print(json.dumps({'status': 'passed', 'checks': len(checks), 'artifact': str(root / 'checks.json')}))
