"""Advance explicit frozen builds through bounded leases, stopping at any failed gate."""
import json,os,subprocess,time
from pathlib import Path
R=Path('/mnt/models/fnprefetch');S=Path(__file__).resolve().parent
candidate=Path(os.environ['FN_GATE_BIN']).resolve();baseline=Path(os.environ['FN_BASELINE_BIN']).resolve()
tag=candidate.parent.name;baseline_tag=baseline.parent.name
assert candidate.is_dir() and baseline.is_dir()
assert os.environ.get('PARENT_HOLDS_GPU_LOCK')!='1','controller must wait outside card lock'
completion=R/'logs'/('chain-'+tag+'.json')
assert not completion.exists(),'use a new exact-build chain after a failure; preserve its evidence'
record={'candidate':str(candidate),'baseline':str(baseline),'steps':[],'status':'waiting for prerequisites'}
def read_json(path):
 try:return json.loads(path.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return None
def save():completion.write_text(json.dumps(record,indent=2))
save();deadline=time.monotonic()+28800
try:
 while True:
  gates=R/'logs'/('operators-'+tag)/'gates.json'
  tests=read_json(gates) or []
  assert not any(t['rc']!=0 for t in tests),'operator failure'
  if len(tests)==3 and (R/'protocol'/('llama.cpp-prefetch-baseline-'+baseline_tag)).is_file():break
  assert time.monotonic()<deadline,'prerequisite wait exhausted'
  time.sleep(15)
 def stage(name,script,args=()):
  record['status']=name;save()
  env=dict(os.environ,FN_RUNNER=str(S/script),FN_MEMORY_MAX='50G')
  subprocess.run(['/usr/bin/python3',str(S/'launch.py'),name+'-'+tag,*args],env=env,cwd=R,check=True)
  record['steps'].append(name);save()
 stage('model','certify.py',('model',))
 stage('reload','certify.py',('reload',))
 record['status']='waiting for matching ship regression';save()
 regression_path=Path(os.environ['FN_BASELINE_REGRESSION'])
 guard_path=regression_path.parent.parent/(regression_path.parent.name+'-guard.json')
 while True:
  regression=read_json(regression_path)
  guard=read_json(guard_path)
  if regression:
   assert regression['source_commit'].startswith(baseline_tag) and regression['status']=='passed',regression
  if guard:assert guard['returncode']==0 and guard['guard'] is None,guard
  if regression and guard:break
  assert time.monotonic()<deadline,'baseline certification wait exhausted'
  time.sleep(15)
 stage('selection','selection-gate.py')
 stage('capacity','preflight-gate.py')
 record['status']='full paired protocol';save()
 env=dict(os.environ,FN_MEMORY_MAX='50G');env.pop('FN_RUNNER',None)
 subprocess.run(['/usr/bin/python3',str(S/'run-pairs.py')],cwd=R,env=env,check=True)
 record['steps'].append('full paired protocol');record['status']='complete';save()
except BaseException as error:
 record['status']='failed';record['error']=repr(error);save();raise
