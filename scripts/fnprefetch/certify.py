"""Run one bounded host-provider certification stage against explicitly frozen builds."""
import hashlib,json,os,re,shutil,subprocess,sys,time
from pathlib import Path
R=Path('/mnt/models/fnprefetch');S=Path(__file__).resolve().parent
assert os.environ.get('PARENT_HOLDS_GPU_LOCK')=='1'
frozen=Path(os.environ['FN_GATE_BIN']).resolve();baseline=Path(os.environ['FN_BASELINE_BIN']).resolve()
assert frozen.parent.parent==R/'releases'
assert baseline.parent.parent==Path('/mnt/models/fnship/releases')
tag=frozen.parent.name;baseline_tag=baseline.parent.name
stage=sys.argv[2] if len(sys.argv)>2 else 'model'
assert stage in ('model','reload')
def identity(directory):
 return {p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in sorted(directory.iterdir()) if p.is_file() and (p.name.startswith('lib') and '.so' in p.name or p.name in ('llama-server','test-moe-prefetch-model'))}
provenance={'frozen':str(frozen),'baseline':str(baseline),'candidate_identity':identity(frozen),'baseline_identity':identity(baseline),'provider':0,'flag_modes':['off','1','2','3','auto']}
assert provenance['candidate_identity'] and provenance['baseline_identity']
deadline=time.monotonic()+540
env=dict(os.environ,FN_GATE_BIN=str(frozen),FN_GATE_DEADLINE=str(deadline))
def run(script,custom=None,args=()):
 assert time.monotonic()<deadline
 subprocess.run(['/usr/bin/python3',str(S/script),*map(str,args)],env=dict(env,**(custom or {})),check=True)
golden=R/'logs'/('device0-'+tag);model_certificate=R/'logs'/('model-certified-'+tag+'.json')
if stage=='model':
 gates=json.loads((R/'logs'/('operators-'+tag)/'gates.json').read_text())
 assert len(gates)==3 and all(g['rc']==0 for g in gates)
 baseline_binary=R/'protocol'/('llama.cpp-prefetch-baseline-'+baseline_tag)
 assert baseline_binary.is_file(),'build the baseline helper against these exact public headers first'
 run('golden-gate.py',{'FN_GOLDEN_BINARY':str(baseline_binary),'FN_GOLDEN_LIB':str(baseline),'FN_GOLDEN_MODES':'baseline','FN_GOLDEN_LABEL':'baseline-'+baseline_tag,'FN_GOLDEN_DEVICE':'0'})
 model=R/'protocol'/('llama.cpp-prefetch-golden-'+tag)
 if not model.exists():shutil.copy2(frozen/'test-moe-prefetch-model',model)
 assert hashlib.file_digest(model.open('rb'),'sha256').hexdigest()==provenance['candidate_identity']['test-moe-prefetch-model']
 run('golden-gate.py',{'FN_GOLDEN_BINARY':str(model),'FN_GOLDEN_LIB':str(frozen),'FN_GOLDEN_MODES':'off,1,2,3,auto','FN_GOLDEN_LABEL':golden.name,'FN_GOLDEN_DEVICE':'0'})
 run('compare-golden.py',args=(golden,R/'logs'/('baseline-'+baseline_tag)))
 comparison=json.loads((golden/'golden-comparison.json').read_text())
 assert comparison['off']['queued_predictions']==0
 lifecycle={}
 for mode in ('off','1','2','3','auto'):
  assert comparison[mode]['state_roundtrip_exact']
  rows=[]
  for line in (golden/f'golden-{mode}.log').read_text().splitlines():
   if '[moe-prefetch]' not in line:continue
   row=dict(re.findall(r'(\w+)=([^\s]+)',line))
   for key in ('bytes','issue_us','start_us','done_us','target_start_us','selected','canceled'):row[key]=int(row[key])
   assert row['bytes']>0 and row['issue_us']>0
   assert bool(row['canceled'])==(row['done_us']<0)
   if row['start_us']>0:assert row['start_us']>=row['issue_us']
   if row['done_us']>=0:assert row['done_us']>=row['start_us']>0
   if row['outcome']=='used':assert row['selected'] and not row['canceled'] and row['target_start_us']>0
   rows.append(row)
  if mode!='off':assert comparison[mode]['copies_used']>0
  lifecycle[mode]={'issued':len(rows),'used':sum(r['outcome']=='used' for r in rows),'completed_late':sum(r['target_start_us']>0 and r['done_us']>r['target_start_us'] for r in rows)}
 model_certificate.write_text(json.dumps(dict(provenance,comparison=comparison,lifecycle=lifecycle,complete=True),indent=2))
 print('All host flag model and copy-lifecycle checks passed; reload remains required',flush=True)
else:
 certificate=json.loads(model_certificate.read_text())
 assert all(certificate[k]==v for k,v in provenance.items()) and certificate['complete']
 alias=R/'protocol'/'runtimes'/tag/'llama.cpp-server';alias.parent.mkdir(parents=True,exist_ok=True)
 if not alias.exists():alias.symlink_to(frozen/'llama-server')
 assert alias.resolve()==(frozen/'llama-server').resolve()
 run('reload-gate.py',{'FN_PREFETCH':'1','FN_RELOAD_DEVICE':'0','FN_RELOAD_SERVER':str(alias),'FN_RELOAD_LIB':str(frozen),'FN_RELOAD_CELL':'reload-host-'+tag,'FN_RELOAD_TURNS':'2','FN_RELOAD_LIMIT':str(int(deadline-time.monotonic()-5))},('flashnext',))
 assert identity(frozen)==provenance['candidate_identity'] and identity(baseline)==provenance['baseline_identity']
 (R/'logs/followup-gate-complete.json').write_text(json.dumps(dict(provenance,model_certificate=str(model_certificate),device_model_certified=False,remaining_seconds=deadline-time.monotonic()),indent=2))
 print('Host model and two-turn fresh-process saved-cache certification passed',flush=True)
