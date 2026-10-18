"""Bound one arena operator/load/restart gate to a single ten-minute lease."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[2]
bundle = Path(sys.argv[1]).resolve()
cell = sys.argv[2]
assert cell.replace('-', '').isalnum()
assert os.environ.get('PARENT_HOLDS_GPU_LOCK') == '1'
started = time.monotonic()
output = root / 'artifacts' / cell
output.mkdir(parents=True, exist_ok=False)
steps = []
environment = dict(os.environ, SHIP_SERVER=str(bundle / 'bin/llama.cpp-server'),
                   SHIP_LIB=str(bundle / 'bin'), SHIP_PROMPT='2048', SHIP_PREDICT='32',
                   SHIP_RELOAD='1', SHIP_CELL=cell + '-model', SHIP_RESERVE_MB='2560')
try:
    for name, command in (
        ('operators', [sys.executable, str(root / 'scripts/fnarena/operators.py'), str(bundle), str(output / 'operators')]),
        ('model', [sys.executable, str(root / 'scripts/fnarena/gate.py'), 'flashnext']),
    ):
        remaining = int(580 - (time.monotonic() - started))
        assert remaining >= 30, 'insufficient lease for next gate'
        environment['SHIP_LIMIT'] = str(remaining - 10)
        with (output / (name + '.log')).open('w') as log:
            completed = subprocess.run(command, env=environment, stdout=log, stderr=log, timeout=remaining)
        steps.append({'name': name, 'command': command, 'returncode': completed.returncode})
        assert completed.returncode == 0, (name, completed.returncode)
    result = {'status': 'passed'}
except Exception as error:
    result = {'status': 'failed', 'error': repr(error)}
result.update(source_commit=json.loads((bundle / 'manifest.json').read_text())['source_commit'],
              elapsed_s=time.monotonic() - started, steps=steps)
(output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
sys.exit(0 if result['status'] == 'passed' else 1)
