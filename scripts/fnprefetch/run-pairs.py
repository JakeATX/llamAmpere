import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
P=Path('/mnt/models/fnprefetch/protocol');R=P.parent;S=Path(__file__).resolve().parent
proof=json.loads((P/'preflight-complete.json').read_text());winner=proof['winner']
runner=P/'frozen-full-control.py';assert not runner.exists(),'full protocol already started'
shutil.copy2(S/'control.py',runner)
settings=dict(proof['settings']);settings.update(FN_RUNNER=str(runner),FN_RESERVE_MB='1024',FN_REQUIRE_SINGLE_B1='1',FN_TOKENS='5120',FN_TEMP='1',FN_LIMIT='1450',FN_CELL_TIMEOUT='1480',FN_DEVICE=str(winner['device']))
manifest=dict(proof,full_settings=settings,full_runner_sha256=hashlib.file_digest(runner.open('rb'),'sha256').hexdigest())
(P/'full-frozen-plan.json').write_text(json.dumps(manifest,indent=2));labels=[]
for pair in range(1,13):
 label='pair-'+str(pair);order=('off','prefetch') if pair%2 else ('prefetch','off')
 for arm in order:
  binary=Path(proof['baseline'] if arm=='off' else proof['candidate'])
  env=dict(os.environ,**settings)
  env.update(FN_SERVER=str(binary/'llama.cpp-server'),FN_LIB=str(binary),FN_PREFETCH='off' if arm=='off' else winner['mode'])
  subprocess.run(['/usr/bin/python3',str(S/'launch.py'),label+'-'+arm,'1','100000'],cwd=P,env=env,check=True)
  result=json.loads((P/'runs'/(label+'-'+arm)/'result.json').read_text());assert result['status']=='complete',result
 labels.append(label)
 subprocess.run(['/usr/bin/python3',str(S/'summarize.py'),*labels],cwd=P,check=True)
 report=json.loads((P/'full-protocol-summary.json').read_text())
 if report['decode_tokens_per_second']['passes_0_01'] and report['prefill_tokens_per_second']['mean_gain_pct']>=0:
  report['decision']='accepted decode gain with no measured prefill regression';break
 if pair>=2 and all(p['prefetch']['decode_tokens_per_second']<=p['off']['decode_tokens_per_second'] for p in report['pairs']):
  report['decision']='futility: neither completed pair improved decode';break
else:report['decision']='not accepted within12 completed pairs'
(P/'full-decision.json').write_text(json.dumps(report,indent=2))
print(report['decision'],flush=True)
