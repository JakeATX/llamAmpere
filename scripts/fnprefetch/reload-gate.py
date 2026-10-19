# Adapted from the ship integration cache gate, using this task's binary and paths.
import os,sys,json,time,subprocess,signal,threading,hashlib,struct,shutil,re,runpy
from pathlib import Path
import requests
R=Path('/mnt/models/fnprefetch'); arm='flashnext'; (R/'artifacts').mkdir(exist_ok=True); O=R/'artifacts'/os.environ.get('FN_RELOAD_CELL',sys.argv[1]);O.mkdir(exist_ok=False); C=O/'cache';C.mkdir(); port=18147; prompt_n=int(os.environ.get('FN_RELOAD_PROMPT','8192')); predict_n=int(os.environ.get('FN_RELOAD_PREDICT','128')); limit=int(os.environ.get('FN_RELOAD_LIMIT','550'))
(O/'runner.py').write_bytes(Path(__file__).read_bytes())
assert os.environ.get('PARENT_HOLDS_GPU_LOCK')=='1'
assert subprocess.check_output(['systemctl','--user','is-active','llama.cpp-sysguard'],text=True).strip()=='active'
start=time.monotonic(); done=threading.Event(); result={}; process=None; logfile=None;phase='setup'; lock=threading.Lock(); violation=[]
def stop_server():
 global process,logfile
 if process is not None and process.poll() is None:
  os.killpg(process.pid,signal.SIGTERM)
  try:process.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
 if logfile:logfile.close()
 process=None;logfile=None
class GuardViolation(RuntimeError):
 def __init__(self,row):
  self.row=row
  super().__init__('resource/time guard: '+str(row))
def check():
 avail=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
 gpu=float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
 rootfree=shutil.disk_usage('/').free;modfree=shutil.disk_usage('/mnt/models').free
 power=float(subprocess.check_output(['nvidia-smi','--query-gpu=power.draw','--format=csv,noheader,nounits'],text=True).strip())
 row=dict(power_w=power,elapsed=time.monotonic()-start,phase=phase,available=avail,vram=gpu,root_free=rootfree,models_free=modfree)
 if avail<7*1024**3 or gpu>23552 or rootfree<40*10**9 or modfree<15*10**9 or shutil.disk_usage('/mnt/overflow').free<50*10**9 or row['elapsed']>limit:raise GuardViolation(row)
 return row
def monitor():
 with (O/'resources.jsonl').open('w') as f:
  while not done.wait(.5):
   try:f.write(json.dumps(check())+'\n');f.flush()
   except Exception as e:
    if isinstance(e,GuardViolation):f.write(json.dumps(dict(e.row,guard_violation=True))+'\n');f.flush()
    violation.append(str(e))
    if process and process.poll() is None:
     try:os.killpg(process.pid,signal.SIGTERM)
     except ProcessLookupError:pass
    return
thread=threading.Thread(target=monitor);thread.start()
env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','LLAMA_','LD_LIBRARY_PATH'))}
env['LLAMA_MOE_ARENA']='off'
env.update(TMPDIR=str(R/'tmp-build'),CUDA_CACHE_PATH=str(R/'cuda-cache'),LD_LIBRARY_PATH=(R/'logs/frozen-bin.txt').read_text().strip())
model='/mnt/models/aiplus_run/finish/AI_N.gguf' if arm=='swift27b' else '/mnt/models/flash-next/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf'
base=[str(R/'llama.cpp-server'),'-m',model,'--load-mode','mmap','-ngl','999','-c','110592','--parallel','1','-b','131072','-ub','2048' if arm=='swift27b' else '12288','-t','8','-tb','8','-lv','4','-fa','on','-ctk','sjkvarn4','-ctv','sjkvarn4','--cache-ram','0','--no-cache-disk','--no-webui','--host','127.0.0.1','--port',str(port),'--slot-save-path',str(C),'--spec-type','draft-mtp-adaptive','--spec-draft-n-max','4','--spec-draft-ubatch','2048','--ctx-checkpoints','4']
if arm=='flashnext':
 base+=['-md','/mnt/models/flash-next/mtp-Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS.gguf','--cpu-moe','-fitt','1536','--moe-cache','on']
 env.update(json.loads(Path('/home/jake-k/qwen38-bench/kernel_cache_investigation/frontier/BACKLOG_RUN_20261015/FNFAST/runs/c-helper19-release-fresh-r1/command.json').read_text())['env'])
if arm=='flashnext':
 env['GGML_CUDA_MOE_CACHE_RESERVE_MB']=os.environ.get('FN_RELOAD_RESERVE_MB','2560')
 env['LLAMA_ARG_MOE_PREFETCH']=os.environ.get('FN_PREFETCH','auto')
 env['LLAMA_B1_CACHE_WARM']='1'
 env['GGML_MOE_CACHE_DEVICE']=os.environ.get('FN_RELOAD_DEVICE','0')
 env['GGML_MOE_CACHE_DEVICE_CAPTURE']=os.environ.get('FN_RELOAD_DEVICE','0')
 env['GGML_MOE_CACHE_DEVICE_STATS']=os.environ.get('FN_RELOAD_DEVICE','0')
 env['GGML_MOE_CACHE_DEVICE_FUSED']='0'
 env['GGML_MOE_CACHE_PROFILE_LOAD']=os.environ.get('FN_RELOAD_PROFILE','')
if os.environ.get('FN_RELOAD_SERVER'):
 base[0]=os.environ['FN_RELOAD_SERVER']
 if os.environ.get('FN_RELOAD_LIB'):env['LD_LIBRARY_PATH']=os.environ['FN_RELOAD_LIB']
worktree_commit=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip()
bundle_manifest=Path(base[0]).resolve().parent.parent/'manifest.json'
bundle_data=json.loads(bundle_manifest.read_text()) if bundle_manifest.exists() else {}
build_commit=bundle_data.get('source_commit',bundle_data.get('commit',worktree_commit))
(O/'command.json').write_text(json.dumps(dict(source_commit=build_commit,worktree_commit=worktree_commit,server_sha256=hashlib.sha256(Path(base[0]).read_bytes()).hexdigest(),argv=base,env={k:v for k,v in env.items() if k.startswith(('GGML_','LLAMA_','LD_LIBRARY_PATH'))}),indent=2))
def launch(tag):
 global process,logfile,phase
 stop_server();check();phase='load-'+tag
 logfile=(O/(tag+'.server.log')).open('w');process=subprocess.Popen(base,stdout=logfile,stderr=logfile,env=env,start_new_session=True)
 for _ in range(420):
  if violation:raise RuntimeError(violation[0])
  if process.poll() is not None:raise RuntimeError('server exited '+str(process.returncode))
  try:
   if requests.get(f'http://127.0.0.1:{port}/health',timeout=1).status_code==200:
    maps=Path(f'/proc/{process.pid}/maps').read_text().splitlines()
    paths=sorted({x.split(None,5)[5] for x in maps if len(x.split(None,5))==6 and '.so' in x.split(None,5)[5]})
    (O/(tag+'.maps.txt')).write_text('\n'.join(paths))
    libraries={}
    for path in paths:
     if not Path(path).name.startswith(('libggml','libllama','libmtmd')):continue
     assert not path.endswith(' (deleted)')
     assert Path(path).resolve().is_relative_to(Path(env['LD_LIBRARY_PATH']).resolve()),path
     inode=Path(path).stat().st_ino
     assert {int(x.split()[4]) for x in maps if len(x.split(None,5))==6 and x.split(None,5)[5]==path}=={inode}
     libraries[path]={'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'inode':inode}
    assert libraries
    (O/(tag+'.libraries.json')).write_text(json.dumps(libraries,indent=2));return
  except requests.RequestException:pass
  time.sleep(.5)
 raise RuntimeError('server load timeout')
def post(path,payload,allow_error=False):
 if violation:raise RuntimeError(violation[0])
 rr=requests.post(f'http://127.0.0.1:{port}'+path,json=payload,timeout=1000)
 if not allow_error:rr.raise_for_status()
 return rr.status_code,rr.json()
def action(a,allow_error=False):return post('/slots/0?action='+a,{'filename':'prompt.bin'},allow_error)
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
try:
 check();launch('cold')
 ids=json.loads(Path('/mnt/models/stratamatch-home/prompt-ids.json').read_text())[:prompt_n];assert len(ids)==prompt_n
 request=dict(id_slot=0,prompt=ids,n_predict=predict_n,temperature=1.0,seed=11,ignore_eos=True,cache_prompt=True,stream=True,return_tokens=True,top_k=40,top_p=.95,min_p=0,repeat_penalty=1.0,samplers=['top_k','top_p','temperature'])
 if arm=='flashnext':request['logit_bias']=[[248044,False],[248046,False],[248063,False],[248064,False],[248065,False]]
 (O/'request.json').write_text(json.dumps(request));phase='prefill';generated=[];t0=time.monotonic()
 with requests.post(f'http://127.0.0.1:{port}/completion',json=request,stream=True,timeout=1000) as rr,(O/'response.jsonl').open('w') as f:
  rr.raise_for_status()
  for line in rr.iter_lines():
   if violation:raise RuntimeError(violation[0])
   if not line:continue
   wire=line.decode();f.write(wire+'\n');f.flush()
   if wire.startswith('data: ') and wire[6:]!='[DONE]':
    x=json.loads(wire[6:]);generated+=x.get('tokens',[])
    if x.get('tokens'):phase='decode'
    if x.get('stop'):result['final']=x
 result['generation_wall_s']=time.monotonic()-t0
 result['request_start_s']=t0-start
 result['request_end_s']=time.monotonic()-start
 generation_log=(O/'cold.server.log').read_text(errors='replace')
 (O/'generation.server.log').write_text(generation_log)
 host_stats=re.findall(r'\[moe-cache\] CUDA0 hits=(\d+)/(\d+) \(([^)]+)\)',generation_log)
 device_stats=re.findall(r'\[moe-cache\] device routes: hits=(\d+) misses=(\d+) dispatches=(\d+)',generation_log)
 result['expert_cache_counters']=dict(host=host_stats[-1] if host_stats else None,device=device_stats[-1] if device_stats else None,coverage='last logged snapshot before generation completed; cumulative within this process')
 assert result['final']['tokens_predicted']==predict_n and len(generated)==predict_n, 'generation count'
 assert result['final']['tokens_evaluated']==prompt_n,'prompt count'
 (O/'generated-tokens.json').write_text(json.dumps(generated))
 result['turns']=[]
 for turn in range(1,int(os.environ.get('FN_RELOAD_TURNS','1'))):
  tool_text='\nTool result: The previous calculation completed. Review the result, identify the next useful check, and explain the next action in detail.\n'
  tool_ids=post('/tokenize',{'content':tool_text,'add_special':False})[1]['tokens']
  ids=ids+generated+tool_ids
  phase='turn-'+str(turn+1)
  turn_request=dict(request,prompt=ids,stream=False)
  turn_start=time.monotonic()
  response=post('/completion',turn_request)[1]
  assert response['tokens_predicted']==predict_n and len(response['tokens'])==predict_n
  result['turns'].append(dict(turn=turn+1,wall_s=time.monotonic()-turn_start,response=response))
  (O/('turn-'+str(turn+1)+'.json')).write_text(json.dumps(response))
  generated=response['tokens']
 phase='save';result['save']=action('save')[1]
 hashes={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in C.iterdir()};result['saved_files']=hashes
 continuation=dict(request,prompt=ids+generated,n_predict=64,temperature=0,stream=False)
 phase='live-continuation';result['live_continuation']=post('/completion',continuation)[1]
 (O/'live-continuation.json').write_text(json.dumps(result['live_continuation']))
 assert result['live_continuation']['tokens_predicted']==64 and len(result['live_continuation']['tokens'])==64,'live continuation count'
 if env.get('GGML_MOE_CACHE_DEVICE')=='1':
  audit=runpy.run_path(str(Path(__file__).resolve().with_name('device-evidence.py')))['device_evidence']
  result['cold_device_evidence']=audit((O/'cold.server.log').read_text())
 launch('restart');phase='restore';result['restore']=action('restore')[1]
 assert result['save']['n_saved']==result['restore']['n_restored'],'restored count'
 result['resave']=post('/slots/0?action=save',{'filename':'roundtrip.bin'})[1]
 result['serialized_bytes_equal']={p.name:digest(p)==digest(C/p.name.replace('prompt.bin','roundtrip.bin',1)) for p in C.glob('prompt.bin*')}
 assert all(result['serialized_bytes_equal'].values()),'serialized bytes differ after reload'
 for p in C.glob('roundtrip.bin*'):p.unlink()
 phase='reload-continuation';result['reload_continuation']=post('/completion',continuation)[1]
 (O/'reload-continuation.json').write_text(json.dumps(result['reload_continuation']))
 assert result['reload_continuation']['tokens_predicted']==64 and len(result['reload_continuation']['tokens'])==64,'reload continuation count'
 result['reload_tokens_exact']=result['live_continuation']['tokens']==result['reload_continuation']['tokens']
 if env.get('GGML_MOE_CACHE_DEVICE')=='1':result['restart_device_evidence']=audit((O/'restart.server.log').read_text())
 # CPU/GPU rounding may change sampled words; report the comparison without rejecting valid math.
 assert result['reload_continuation']['timings']['prompt_n'] <= len(continuation['prompt'])-result['save']['n_saved']+1,'saved prefix reprocessed'
 # Corrupt only our sidecar. Both binding and MTP payload errors must reject the restore without stopping the server.
 phase='malformed-sidecar';result['negative_checks']=[];sidecar=C/'prompt.bin.ckpt'
 positions=[('target-digest',12,1),('draft-digest',44,1)]
 with sidecar.open('rb') as f:
  magic,version,count=struct.unpack('<III',f.read(12));assert version==4
  if count:positions.append(('invalid-checkpoint-position',84,4))
  f.seek(76)
  for _ in range(count):
   f.seek(16,1)
   for _ in range(3):
    size=struct.unpack('<Q',f.read(8))[0];f.seek(size,1)
  spec_size=struct.unpack('<Q',f.read(8))[0];spec_start=f.tell()
  if spec_size>=36:positions.append(('nonfinite-MTP',spec_start+32,4))
 for name,offset,size in positions:
  action('erase')
  with sidecar.open('r+b') as f:
   f.seek(offset);original=f.read(size);f.seek(offset);f.write(struct.pack('<i',2147483647) if name=='invalid-checkpoint-position' else struct.pack('<f',float('nan')) if size==4 else bytes([original[0]^1]))
  try:
   code,body=action('restore',True);result['negative_checks'].append(dict(name=name,http=code,response=body));assert code>=400,(name,body)
   assert requests.get(f'http://127.0.0.1:{port}/health',timeout=3).status_code==200
  finally:
   with sidecar.open('r+b') as f:f.seek(offset);f.write(original)
 result['final_valid_restore']=action('restore')[1]
 assert {p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in C.iterdir()}==hashes,'saved files changed'
 result['status']='passed'
except Exception as e:result.update(status='failed',error=repr(e))
finally:
 stop_server();done.set();thread.join();result['elapsed_s']=time.monotonic()-start;result['violations']=violation
 rows=[json.loads(x) for x in (O/'resources.jsonl').read_text().splitlines()]
 generation_rows=[x for x in rows if result.get('request_start_s',float('inf')) <= x['elapsed'] <= result.get('request_end_s',float('-inf'))]
 result['generation_power_mean_w']=sum(x['power_w'] for x in generation_rows)/len(generation_rows) if generation_rows else None
 result['generation_peak_vram_mib']=max((x['vram'] for x in generation_rows),default=0)
 result['resources']=dict(peak_vram_mib=max((x['vram'] for x in rows),default=0),min_available_bytes=min((x['available'] for x in rows),default=0),min_root_free=min((x['root_free'] for x in rows),default=0),min_models_free=min((x['models_free'] for x in rows),default=0))
 (O/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('live_continuation','reload_continuation','final')}),flush=True)
sys.exit(0 if result['status']=='passed' else 1)
