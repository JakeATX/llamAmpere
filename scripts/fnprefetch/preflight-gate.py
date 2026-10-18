import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
R=Path('/mnt/models/fnprefetch');P=R/'protocol'
assert os.environ['PARENT_HOLDS_GPU_LOCK']=='1'
selection=json.loads((P/'selection-complete.json').read_text()); winner=selection['winner']
settings=dict(selection['settings']); settings.update(FN_RESERVE_MB='1024',FN_REQUIRE_SINGLE_B1='1',FN_TOKENS='128',FN_TEMP='1')
runner=P/'frozen-preflight-control.py';shutil.copy2(Path(__file__).resolve().parent/'control.py',runner)
deadline=time.monotonic()+540;results=[]
for arm in ('off','prefetch'):
 binary=Path(selection['baseline'] if arm=='off' else selection['candidate'])
 env=dict(os.environ,**settings)
 env.update(FN_SERVER=str(binary/'llama.cpp-server'),FN_LIB=str(binary),FN_DEVICE=str(winner['device']),FN_PREFETCH='off' if arm=='off' else winner['mode'],FN_LIMIT=str(int(deadline-time.monotonic()-5)))
 label='preflight-'+arm
 with (P/(label+'.diagnostic.log')).open('w') as out:
  subprocess.run(['/usr/bin/python3',str(runner),label,'1','100000'],env=env,stdout=out,stderr=out,check=True)
 result=json.loads((P/'runs'/label/'result.json').read_text());assert result['status']=='complete',result
 assert result['final']['tokens_evaluated']==100000 and result['final']['tokens_predicted']==128
 results.append({'arm':arm,'path':str(P/'runs'/label),'peak_vram_mib':result['peaks']['vram_mib']})
(P/'preflight-complete.json').write_text(json.dumps(dict(selection,preflight=results,preflight_runner_sha256=hashlib.file_digest(runner.open('rb'),'sha256').hexdigest()),indent=2))
print('Matched100K capacity gates passed; these are not accepted speed results',flush=True)
