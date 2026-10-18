"""Compare only complete, chronologically interleaved full-protocol runs."""
import hashlib, json, math, sys
from pathlib import Path
manifest=json.loads(Path(sys.argv[1]).read_text())
pairs=manifest['pairs'];n=len(pairs)
assert n in (8,16), 'predeclared checks only at 8 or 16 pairs'
metrics={key:[] for key in ('prefill','decode','ttc')}
candidate_absolute={key:[] for key in ('prefill','decode')}
identities={}; seen=set(); last_end=0

def checked(path,arm):
 path=Path(path).resolve();assert path not in seen,'a run was reused';seen.add(path)
 value=json.loads(path.read_text());root=path.parent
 assert value['status']=='passed' and not value['violations']
 assert value['final']['tokens_evaluated']==100000 and value['final']['tokens_predicted']==5120
 assert value['received_tokens']==5120,'streamed generation incomplete'
 for number in (value['generation_wall_s'],value['generation_power_mean_w'],
                value['final']['timings']['prompt_per_second'],value['final']['timings']['predicted_per_second']):
  assert math.isfinite(number) and number>0,'invalid full-protocol measurement'
 assert value['resources']['peak_vram_mib']<=23552
 for name in ('start','generation-start','generation-end','end'):
  accounting=json.loads((root/('cpu-accounting-'+name+'.json')).read_text())
  assert accounting['cpu.weight']=='100','full arm did not use CPUWeight100'
 request=json.loads((root/'request.json').read_text())
 assert request['temperature']==1.0 and request['n_predict']==5120 and len(request['prompt'])==100000
 command=json.loads((root/'command.json').read_text())
 argv=command['argv'][:];argv[argv.index('--slot-save-path')+1]='<run-cache>'
 assert argv[argv.index('-ctk')+1]=='sjkvarn4' and argv[argv.index('-ctv')+1]=='sjkvarn4'
 assert argv[argv.index('--spec-type')+1].startswith('draft-mtp')
 for i,arg in enumerate(argv):
  if arg=='--moe-skip-tail':assert float(argv[i+1])==0,'lossy expert skipping cannot win release defaults'
  if arg.startswith('--moe-skip-tail='):assert float(arg.split('=',1)[1])==0,'lossy expert skipping cannot win release defaults'
 for name,env_value in command['env'].items():
  if 'SKIP_TAIL' in name:assert env_value.lower() in ('','0','0.0','off','false'),'lossy skip environment'
 bundle=Path(argv[0]).parent.parent;frozen=json.loads((bundle/'manifest.json').read_text())
 assert command['source_commit']==frozen['source_commit']
 assert command['server_sha256']==frozen['libraries']['llama-server']['sha256']
 loaded=json.loads((root/'cold.libraries.json').read_text());assert len(loaded)>=4
 for name,record in loaded.items():
  assert Path(record['path']).resolve().parent==bundle/'bin'
  assert record['sha256']==frozen['libraries'][name]['sha256']
 profile=command['env'].get('GGML_MOE_CACHE_PROFILE_LOAD')
 profile_hash=hashlib.sha256(Path(profile).read_bytes()).hexdigest() if profile else None
 assert profile_hash==command['profile_sha256'],'profile bytes changed after the run'
 if profile:assert Path(profile).resolve()==root/'expert-profile-input.bin','profile was not frozen per run'
 normalized_env=command['env'].copy()
 if profile:normalized_env['GGML_MOE_CACHE_PROFILE_LOAD']='<frozen-profile>'
 identity=dict(source=command['source_commit'],server=command['server_sha256'],argv=argv,env=normalized_env,profile_hash=profile_hash)
 if arm in identities:assert identity==identities[arm],arm+' changed during comparison'
 else:identities[arm]=identity
 # Filesystem mtime cannot prove nonoverlapping GPU leases; use recorded UTC starts.
 timing=root/'lease.json'
 assert timing.exists(),'lease.json with start_unix_s/end_unix_s is required'
 lease=json.loads(timing.read_text());assert 0<lease['end_unix_s']-lease['start_unix_s']<=1500
 return value,request,lease

for index,pair in enumerate(pairs):
 previous,pr,pl=checked(pair['previous'],'previous')
 candidate,cr,cl=checked(pair['candidate'],'candidate')
 assert pr==cr,'paired requests differ'
 a,b=(pl,cl) if index%2==0 else (cl,pl)
 assert last_end<=a['start_unix_s'] and a['end_unix_s']<=b['start_unix_s'],'run order or overlap'
 last_end=b['end_unix_s']
 for key,field in [('prefill','prompt_per_second'),('decode','predicted_per_second')]:
  metrics[key].append(candidate['final']['timings'][field]/previous['final']['timings'][field])
  candidate_absolute[key].append(candidate['final']['timings'][field])
 metrics['ttc'].append(previous['generation_wall_s']/candidate['generation_wall_s'])
assert identities['previous']['profile_hash']==identities['candidate']['profile_hash'],'arms used different warm profiles'
result={'pairs':n,'primary_metric':'request TTC; predeclared before model pairs',
        'test':'one-sided exact sign; 2 planned looks, Bonferroni adjusted','identities':identities}
for key,ratios in metrics.items():
 wins=sum(value>1 for value in ratios)
 p=sum(math.comb(n,k) for k in range(wins,n+1))/2**n
 result[key]={'wins':wins,'one_sided_p':p,'adjusted_p':min(1,2*p),'geometric_speed_ratio':math.exp(sum(math.log(x) for x in ratios)/n),'ratios':ratios}
result['lossless_speed_gate_passed']=result['ttc']['adjusted_p']<=.01 and result['ttc']['geometric_speed_ratio']>1
result['candidate_geometric_mean_tps']={key:math.exp(sum(math.log(x) for x in values)/n) for key,values in candidate_absolute.items()}
result['release_throughput_targets_met']=result['candidate_geometric_mean_tps']['prefill']>=3170 and result['candidate_geometric_mean_tps']['decode']>=143
result['quality_and_cache_gates_required_separately']=True
print(json.dumps(result,indent=2))
