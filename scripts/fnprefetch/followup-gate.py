import hashlib, json, os, shutil, subprocess, time
from pathlib import Path
root=Path('/mnt/models/fnprefetch')
assert os.environ['PARENT_HOLDS_GPU_LOCK']=='1'
frozen=Path((root/'logs/frozen-bin.txt').read_text().strip())
assert subprocess.run(['git','-C',str(root),'merge-base','--is-ancestor','858f25f7c',frozen.parent.name]).returncode==0,'corrected build not ready'
deadline=time.monotonic()+540
env=dict(os.environ,FN_GATE_DEADLINE=str(deadline),FN_GATE_BIN=str(frozen))
def run(script,custom=None,args=()):
 assert time.monotonic()<deadline
 subprocess.run(['/usr/bin/python3',str(Path(__file__).resolve().parent/Path(script).name),*map(str,args)],env=dict(env,**(custom or {})),check=True)
run('gpu-gate.py')
# e3 and3f public/common headers are identical; the baseline helper ABI is unchanged.
baseline_binary=root/'protocol'/'llama.cpp-prefetch-baseline-3f'
assert baseline_binary.is_file(), 'run prepare-baseline.sh inside a locked build unit first'
run('golden-gate.py',dict(FN_GOLDEN_BINARY=str(baseline_binary),FN_GOLDEN_LIB='/mnt/models/fnship/releases/3f372a1ba/bin',FN_GOLDEN_MODES='baseline',FN_GOLDEN_LABEL='baseline-3f'))
model=root/'protocol'/('llama.cpp-prefetch-golden-'+frozen.parent.name)
if not model.exists():shutil.copy2(frozen/'test-moe-prefetch-model',model)
assert hashlib.file_digest(model.open('rb'),'sha256').digest()==hashlib.file_digest((frozen/'test-moe-prefetch-model').open('rb'),'sha256').digest()
for device in (0,1):
 label=f'device{device}-'+frozen.parent.name
 run('golden-gate.py',dict(FN_GOLDEN_BINARY=str(model),FN_GOLDEN_LIB=str(frozen),FN_GOLDEN_MODES='off,1',FN_GOLDEN_LABEL=label,FN_GOLDEN_DEVICE=str(device)))
 run('compare-golden.py',dict(FN_COMPARE_MODES='off,1'),args=(root/'logs'/label,root/'logs/baseline-3f'))
 comparison=json.loads((root/'logs'/label/'golden-comparison.json').read_text())
 assert comparison['1']['copies_used']>0,('no consumed model copies',device)
 if deadline-time.monotonic()<100:raise RuntimeError('reload must use another card window')
alias=root/'protocol'/'runtimes'/frozen.parent.name/'llama.cpp-server'
alias.parent.mkdir(parents=True,exist_ok=True)
if not alias.exists():alias.symlink_to(frozen/'llama-server')
assert alias.resolve()==(frozen/'llama-server').resolve()
run('protocol/reload-gate.py',dict(FN_PREFETCH='1',FN_RELOAD_DEVICE='1',FN_RELOAD_SERVER=str(alias),FN_RELOAD_LIB=str(frozen),FN_RELOAD_CELL='reload-device-'+frozen.parent.name,FN_RELOAD_TURNS='2',FN_RELOAD_LIMIT=str(int(deadline-time.monotonic()-5))),args=('flashnext',))
(root/'logs/followup-gate-complete.json').write_text(json.dumps({'frozen':str(frozen),'remaining_seconds':deadline-time.monotonic()},indent=2))
