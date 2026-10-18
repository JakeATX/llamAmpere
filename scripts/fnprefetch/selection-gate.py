import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
R=Path('/mnt/models/fnprefetch');P=R/'protocol'
assert os.environ['PARENT_HOLDS_GPU_LOCK']=='1'
gate=json.loads((R/'logs/followup-gate-complete.json').read_text())
frozen=Path(gate['frozen']); baseline=Path('/mnt/models/fnship/releases/3f372a1ba/bin')
plan=R/'protocol/selection-plan.json'
assert not plan.exists(),'selection already started; use its saved outcome'
settings=json.loads(Path('/mnt/models/fnfast/ship-d28/bench-env.json').read_text())
profile=R/'cache/selection-prompt.profile';profile.parent.mkdir(exist_ok=True)
shutil.copy2('/mnt/models/fnfast/cache/usage-c11-from-handoff5.bin',profile)
settings.update(FN_PROFILE_LOAD=str(profile),FN_RUNNER=str(Path(__file__).resolve().parent/'control.py'),FN_CELL_TIMEOUT='580',FN_LIMIT='530',FN_TOKENS='128',FN_TEMP='1',FN_RESERVE_MB='2560',FN_STACKS='0',FN_DEVICE_FUSED='0',FN_REQUIRE_SINGLE_B1='0')
runner=R/'protocol/frozen-selection-control.py'
shutil.copy2(Path(__file__).resolve().parent/'control.py',runner)
manifest={'candidate':str(frozen),'baseline':str(baseline),'runner_sha256':hashlib.file_digest(runner.open('rb'),'sha256').hexdigest(),'profile_sha256':hashlib.file_digest(profile.open('rb'),'sha256').hexdigest(),'settings':settings,'prompt_tokens':4096,'generated_tokens':128,'purpose':'development selection only; no full speed claim'}
plan.write_text(json.dumps(manifest,indent=2)); deadline=time.monotonic()+540
results=[]
def run(label,device,mode,binary):
 remaining=int(deadline-time.monotonic()-5); assert remaining>30,'selection window exhausted'
 env=dict(os.environ,**settings)
 env.update(FN_SERVER=str(binary/'llama.cpp-server'),FN_LIB=str(binary),FN_DEVICE=str(device),FN_PREFETCH=str(mode),FN_LIMIT=str(remaining))
 with (P/(label+'.diagnostic.log')).open('w') as out:
  subprocess.run(['/usr/bin/python3',str(runner),label,'1','4096'],env=env,stdout=out,stderr=out,check=True)
 path=P/'runs'/label;result=json.loads((path/'result.json').read_text())
 assert result['status']=='complete',result
 log=(path/'server.log').read_text()
 row={'label':label,'device':device,'mode':mode,'decode_tokens_per_second':result['final']['timings']['predicted_per_second'],'copies_used':log.count('outcome=used'),'path':str(path)}
 results.append(row); print(json.dumps(row),flush=True)
 return row
device=0 # Accepted host provider; optional device provider has separate readiness gates.
run('select-off-device0',device,'off',baseline)
for mode in ('1','2','3','auto'):run('select-prefetch-'+mode,device,mode,frozen)
choices=[r for r in results if r['mode']!='off' and r['copies_used']>0]
assert choices,'no consumed prefetch copies in depth selection'
winner=max(choices,key=lambda r:r['decode_tokens_per_second'])
(P/'selection-complete.json').write_text(json.dumps(dict(manifest,results=results,winner=winner),indent=2))
print('Development selection complete; full protocol still required',flush=True)
