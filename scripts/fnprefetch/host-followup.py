import json,os,subprocess,time
from pathlib import Path
R=Path('/mnt/models/fnprefetch');S=Path(__file__).resolve().parent
assert os.environ['PARENT_HOLDS_GPU_LOCK']=='1'
frozen=Path(os.environ.get('FN_GATE_BIN',str(R/'releases/d8f23ede0/bin')));tag=frozen.parent.name
gates=json.loads((R/'logs'/('operators-'+tag)/'gates.json').read_text())
assert len(gates)==3 and all(g['rc']==0 for g in gates)
label='device0-'+tag;golden=R/'logs'/label
old=json.loads((golden/'golden-comparison.json').read_text());assert old['1']['copies_used']>0 and old['1']['state_roundtrip_exact'] and old['off']['state_roundtrip_exact']
model=R/'protocol'/('llama.cpp-prefetch-golden-'+tag);assert model.is_file()
deadline=time.monotonic()+540
env=dict(os.environ,FN_GATE_BIN=str(frozen),FN_GATE_DEADLINE=str(deadline))
def run(script,custom,args=()):
 subprocess.run(['/usr/bin/python3',str(S/script),*map(str,args)],env=dict(env,**custom),check=True)
run('golden-gate.py',dict(FN_GOLDEN_BINARY=str(model),FN_GOLDEN_LIB=str(frozen),FN_GOLDEN_MODES='2,3,auto',FN_GOLDEN_LABEL=label,FN_GOLDEN_DEVICE='0'))
run('compare-golden.py',dict(FN_COMPARE_MODES='off,1,2,3,auto'),args=(golden,R/'logs/baseline-3f'))
comparison=json.loads((golden/'golden-comparison.json').read_text())
assert all(comparison[m]['copies_used']>0 and comparison[m]['state_roundtrip_exact'] for m in ('1','2','3','auto'))
alias=R/'protocol'/'runtimes'/tag/'llama.cpp-server';alias.parent.mkdir(parents=True,exist_ok=True)
if not alias.exists():alias.symlink_to(frozen/'llama-server')
assert alias.resolve()==(frozen/'llama-server').resolve()
run('reload-gate.py',dict(FN_PREFETCH='1',FN_RELOAD_DEVICE='0',FN_RELOAD_SERVER=str(alias),FN_RELOAD_LIB=str(frozen),FN_RELOAD_CELL='reload-host-'+tag,FN_RELOAD_TURNS='2',FN_RELOAD_LIMIT=str(int(deadline-time.monotonic()-5))),args=('flashnext',))
(R/'logs/followup-gate-complete.json').write_text(json.dumps({'frozen':str(frozen),'provider':0,'flag_modes':['off','1','2','3','auto'],'device_model_certified':False,'device_model_reason':'current integration has no caller of device_prepare; direct CUDA staging/fusion/replay fixtures passed','remaining_seconds':deadline-time.monotonic()},indent=2))
