import os,sys,json,time,subprocess,signal,threading
from pathlib import Path
import requests,ctypes,runpy,shutil,hashlib
def request_special_token_policy():
 value=os.environ.get('FN_BAN_ROLE_START','0')
 assert value in ('0','1'), 'FN_BAN_ROLE_START must be 0 or 1'
 tokens=[248044,248046,248063,248064,248065]
 if value=='1':tokens.append(248045)
 return {'name':'legacy-plus-role-start-ban' if value=='1' else 'legacy-special-token-bans',
         'ban_role_start':value=='1','banned_token_ids':tokens}
class DiagnosticComplete(Exception):
 pass
HERE=Path('/mnt/models/fnprefetch/protocol')
cell=sys.argv[1];mtp=sys.argv[2]=='1';depth=int(sys.argv[3]);switch=sys.argv[4] if len(sys.argv)>4 else ''
out=HERE/'runs'/cell;out.mkdir(parents=True,exist_ok=False)
assert os.environ.get('PARENT_HOLDS_GPU_LOCK')=='1'
assert shutil.disk_usage('/').free >= 40*10**9, 'root disk below 40 GB'
assert shutil.disk_usage('/mnt/models').free >= 15*10**9, 'model disk below 15 GB'
assert subprocess.check_output(['systemctl','--user','is-active','llama.cpp-sysguard'],text=True).strip()=='active'
assert not list(Path('/home/jake-k/qwen38-bench/.locks').glob('PRIORITY_*'))
cfg=json.loads(Path('/home/jake-k/qwen38-bench/kernel_cache_investigation/frontier/BACKLOG_RUN_20261015/FNCONFIG_HUNT/runs/confirm-best-r2/command.json').read_text())
env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','LLAMA_'))}
env.update(cfg['env']);env['LLAMA_ARG_MOE_PREFETCH']=os.environ.get('FN_PREFETCH','off');env['CUDA_CACHE_PATH']='/mnt/models/fnprefetch/cuda-cache';Path(env['CUDA_CACHE_PATH']).mkdir(exist_ok=True);env['TMPDIR']='/mnt/models/fnprefetch/tmp-build';env['LD_LIBRARY_PATH']=os.environ.get('FN_LIB',Path('/mnt/models/fnprefetch/logs/frozen-bin.txt').read_text().strip())
env['LLAMA_HOST_PHASES']='1'
env['LLAMA_MOE_ARENA']='off'
env['LLAMA_B1_CACHE_WARM']=os.environ.get('FN_WARM','1')
for source,target in [('FN_CACHE_STATS','GGML_CUDA_MOE_CACHE_STATS'),('FN_QSA_POOL_RANGE','LLAMA_QSA_POOL_RANGE'),('FN_MTP_UPDATE_KV_ONLY','LLAMA_MTP_UPDATE_KV_ONLY'),('FN_GDN_R16_COLS','GGML_CUDA_GDN_R16_COLS'),('FN_B1_LAYER_RESERVE','LLAMA_B1_LAYER_RESERVE'),('FN_MMQ_ID_CAP','GGML_CUDA_MMQ_ID_CAP'),('FN_RESERVE_MB','GGML_CUDA_MOE_CACHE_RESERVE_MB'),('FN_DEVICE_CAPTURE','GGML_MOE_CACHE_DEVICE_CAPTURE'),('FN_PROFILE_SAVE','GGML_MOE_CACHE_PROFILE_SAVE'),('FN_PROFILE_LOAD','GGML_MOE_CACHE_PROFILE_LOAD'),('FN_DEVICE','GGML_MOE_CACHE_DEVICE'),('FN_DEVICE_FUSED','GGML_MOE_CACHE_DEVICE_FUSED'),('FN_RELEASE','LLAMA_MOE_CACHE_RELEASE_PROMPT'),('FN_SHARED_COMPUTE','LLAMA_SHARED_COMPUTE'),('FN_DRAFT_LEAN','LLAMA_MTP_DRAFT_COMPUTE_LEAN'),('FN_DEVICE_STATS','GGML_MOE_CACHE_DEVICE_STATS'),('FN_B1_FFN','LLAMA_B1_FFN'),('FN_B1_FFN_CHECK','LLAMA_B1_FFN_CHECK'),('FN_B1_FFN_MAX_ROWS','LLAMA_B1_FFN_MAX_ROWS'),('FN_B1_FFN_MMID','LLAMA_B1_FFN_MMID'),('FN_B1_FFN_ROUTED_ROWS','LLAMA_B1_FFN_ROUTED_ROWS')]:
 if source in os.environ:env[target]=os.environ[source]
if 'FN_FA_PREFILL_NCOLS2' in os.environ:
 assert os.environ['FN_FA_PREFILL_NCOLS2'] in ('1','2','8'), 'FN_FA_PREFILL_NCOLS2 must be 1, 2 or 8'
 env['GGML_CUDA_FA_PREFILL_NCOLS2']=os.environ['FN_FA_PREFILL_NCOLS2']
for source,target,allowed in [('FN_NORM_CAST_B1_256','GGML_CUDA_NORM_CAST_B1_256',('0','1')),('FN_GDN_R16_WARPS','GGML_CUDA_GDN_R16_WARPS',('1','2','4')),('FN_GENERATION_PROFILE_PRIORITY','GGML_MOE_CACHE_GENERATION_PROFILE_PRIORITY',('0','1'))]:
 if source in os.environ:
  assert os.environ[source] in allowed, source+' has an unsupported value'
  env[target]=os.environ[source]
for source,target in [('FN_GENERATION_PROFILE_SAVE','GGML_MOE_CACHE_GENERATION_PROFILE_SAVE'),('FN_GENERATION_PROFILE_LOAD','GGML_MOE_CACHE_GENERATION_PROFILE_LOAD')]:
 if source in os.environ:
  profile_path=Path(os.environ[source]).resolve()
  assert profile_path.is_relative_to('/mnt/models/fnprefetch') and profile_path.parent.is_dir(), source+' must be under /mnt/models/fnprefetch with an existing parent'
  if source.endswith('_LOAD'):assert profile_path.is_file(), source+' must exist'
  env[target]=str(profile_path)
for source,target in [('FN_DEVICE_ADAPT_INTERVAL','GGML_MOE_CACHE_DEVICE_ADAPT_INTERVAL'),('FN_DEVICE_ADAPT_MB','GGML_MOE_CACHE_DEVICE_ADAPT_MB')]:
 if source in os.environ:
  value=os.environ[source]
  assert value.isascii() and value.isdecimal() and 1 <= int(value) <= 1024, source+' must be an integer from 1 through 1024'
  env[target]=value
if 'FN_MMQ_ID_JAVG' in os.environ:
 assert os.environ['FN_MMQ_ID_JAVG'] in ('0','1'), 'FN_MMQ_ID_JAVG must be 0 or 1'
 env['GGML_CUDA_MMQ_ID_JAVG']=os.environ['FN_MMQ_ID_JAVG']
if 'FN_MTP_OUTPUT_SHRINK' in os.environ:
 assert os.environ['FN_MTP_OUTPUT_SHRINK'] in ('0','1'), 'FN_MTP_OUTPUT_SHRINK must be 0 or 1'
 env['LLAMA_MTP_OUTPUT_SHRINK']=os.environ['FN_MTP_OUTPUT_SHRINK']
if 'FN_MTP_EH_BATCH' in os.environ:
 assert os.environ['FN_MTP_EH_BATCH'] in ('0','1'), 'FN_MTP_EH_BATCH must be 0 or 1'
 env['LLAMA_MTP_EH_BATCH']=os.environ['FN_MTP_EH_BATCH']
if 'FN_MTP_COMPAT_REUSE' in os.environ:
 assert os.environ['FN_MTP_COMPAT_REUSE'] in ('0','1'), 'FN_MTP_COMPAT_REUSE must be 0 or 1'
 env['LLAMA_MTP_COMPAT_REUSE']=os.environ['FN_MTP_COMPAT_REUSE']
if 'FN_MTP_UBATCH_REUSE' in os.environ:
 assert os.environ['FN_MTP_UBATCH_REUSE'] in ('0','1'), 'FN_MTP_UBATCH_REUSE must be 0 or 1'
 env['LLAMA_MTP_UBATCH_REUSE']=os.environ['FN_MTP_UBATCH_REUSE']
if 'FN_MTP_CPU_TIMING' in os.environ:
 assert os.environ['FN_MTP_CPU_TIMING'] in ('0','1'), 'FN_MTP_CPU_TIMING must be 0 or 1'
 env['LLAMA_MTP_CPU_TIMING']=os.environ['FN_MTP_CPU_TIMING']
if 'FN_CACHE_PHASE_STATS' in os.environ:
 assert os.environ['FN_CACHE_PHASE_STATS'] in ('0','1'), 'FN_CACHE_PHASE_STATS must be 0 or 1'
 env['LLAMA_CACHE_PHASE_STATS']=os.environ['FN_CACHE_PHASE_STATS']
cmd=cfg['argv'];cmd[0]=os.environ.get('FN_SERVER','/mnt/models/fnprefetch/llama.cpp-server')
if '--load-mode' not in cmd:cmd+=['--load-mode','mmap']
assert cmd[cmd.index('--load-mode')+1]=='mmap'
cmd[cmd.index('--port')+1]='8095'
for flag,key in [('-ub','FN_UB'),('-b','FN_BATCH'),('--spec-draft-ubatch','FN_DRAFT_UB')]:
 if os.environ.get(key):cmd[cmd.index(flag)+1]=os.environ[key]
cache_dir=Path('/mnt/models/fnprefetch/cache')/cell;cache_dir.mkdir(parents=True,exist_ok=True)
if os.environ.get('FN_CACHE_TEST') or os.environ.get('FN_SAVE_PATH_ONLY'):cmd+=['--slot-save-path',str(cache_dir)]
def file_provenance(path):
 path=Path(path).resolve();digest=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
 stat=path.stat()
 return {'path':str(path),'sha256':digest.hexdigest(),'bytes':stat.st_size,'inode':stat.st_ino,'device':stat.st_dev}
command_manifest={'argv':cmd,'env':{k:v for k,v in env.items() if k.startswith(('GGML_','LLAMA_'))},'runtime_overrides':{k:v for k,v in os.environ.items() if k.startswith('FN_')},'ld_library_path':env.get('LD_LIBRARY_PATH'),'server_provenance':file_provenance(cmd[0]),'special_token_policy':request_special_token_policy(),'runner_provenance':file_provenance(__file__)}
if env.get('GGML_MOE_CACHE_PROFILE_LOAD'):
 profile=Path(env['GGML_MOE_CACHE_PROFILE_LOAD']).resolve()
 assert profile.is_relative_to('/mnt/models/fnprefetch'), 'use an owned frozen profile copy'
 command_manifest['profile_provenance']=file_provenance(profile)
(out/'command.json').write_text(json.dumps(command_manifest,indent=2))
print('CHECKED ARGV '+json.dumps(cmd),flush=True)
assert '--cpu-moe-draft' not in cmd and cmd[cmd.index('-t')+1]=='8' and cmd[cmd.index('--moe-cache')+1]=='on'
server_executable=str(Path(cmd[0]).resolve())
launch=cmd
if os.environ.get('FN_GDB'):
 launch=['gdb','-q','-batch','-ex','set pagination off','-ex','run','-ex','thread apply all bt 12','-ex','quit','--args']+cmd
nsys='/mnt/models/stratamatch-home/fnhunt-tools/nsight_systems-linux-x86_64-2026.3.2.476-archive/target-linux-x64/nsys'
if os.environ.get('FN_PROFILE'):
 env['FN_EXPERT_PROFILE']='1'
 env['TMPDIR']='/mnt/models/fnprefetch/tmp-build';Path(env['TMPDIR']).mkdir(exist_ok=True)
 launch=[nsys,'launch','--session-new=fnfast-'+cell,'--trace=cuda,nvtx','--resolve-symbols=false','--cuda-graph-trace=node','--wait=primary']+cmd
log=(out/'server.log').open('w');p=subprocess.Popen(launch,env=env,stdout=log,stderr=log,start_new_session=True,preexec_fn=lambda:ctypes.CDLL(None).prctl(0x59616d61,ctypes.c_ulong(-1).value,0,0,0))
# A profiler may put the actual server in a new process group. The systemd cell
# cgroup, not the wrapper's process group, is the ownership boundary.
cell_cgroup_rel=Path('/proc/self/cgroup').read_text().strip().split('::')[1].lstrip('/')
cell_cgroup=Path('/sys/fs/cgroup')/cell_cgroup_rel
assert cell_cgroup.name.startswith('llama.cpp-fnprefetch-') and cell_cgroup.name.endswith('.service'), 'not in an owned benchmark unit'
def cell_processes():
 members=set()
 for listing in [cell_cgroup/'cgroup.procs', *cell_cgroup.rglob('cgroup.procs')]:
  try:members.update(int(pid) for pid in listing.read_text().split())
  except FileNotFoundError:pass
 return members

def is_cell_server(pid):
 try:
  group=Path(f'/proc/{pid}/cgroup').read_text().strip().split('::')[1].lstrip('/')
  return (group==cell_cgroup_rel or group.startswith(cell_cgroup_rel+'/')) and str(Path(f'/proc/{pid}/exe').resolve())==server_executable
 except (FileNotFoundError,ProcessLookupError,PermissionError):return False

def server_pid(required=False):
 matches=[pid for pid in cell_processes() if is_cell_server(pid)]
 if len(matches)>1:raise RuntimeError('multiple server processes in owned cell: '+repr(matches))
 if matches:return matches[0]
 if required:raise RuntimeError('actual server process not found in owned cell')
 return None

def signal_owned(sig):
 # Signal the actual server once; its profiler wrapper exits when it does.
 members=cell_processes()
 targets={pid for pid in members if is_cell_server(pid)}
 if not targets and p.pid in members:targets.add(p.pid)
 for pid in targets:
  try:os.kill(pid,sig)
  except ProcessLookupError:pass

def cleanup_owned_processes():
 # All remaining children in this dedicated systemd unit were started by this
 # cell. Exclude the controller and its ancestors so results can still be saved.
 excluded={os.getpid()};ancestor=os.getppid()
 while ancestor>1:
  excluded.add(ancestor)
  try:ancestor=int(Path(f'/proc/{ancestor}/stat').read_text().rsplit(')',1)[1].split()[1])
  except (FileNotFoundError,ProcessLookupError):break
 def remaining():return cell_processes()-excluded
 for sig,delay in [(signal.SIGTERM,3),(signal.SIGKILL,1)]:
  for pid in remaining():
   try:os.kill(pid,sig)
   except ProcessLookupError:pass
  deadline=time.monotonic()+delay
  while remaining() and time.monotonic()<deadline:time.sleep(.1)
 return sorted(remaining())

def memory_snapshot(cgroup, engine=None):
 snapshot={'t':time.monotonic()}
 for filename in ['memory.current','memory.swap.current','memory.events','memory.pressure','memory.stat','cpu.stat','cpu.pressure','cpu.max','cpu.weight']:
  try:
   content=(cgroup/filename).read_text()
   if filename in ['memory.current','memory.swap.current']:
    snapshot[filename]=int(content)
   elif filename in ('memory.pressure','cpu.pressure'):
    snapshot[filename]={line.split()[0]:{k:float(v) if k!='total' else int(v) for k,v in (field.split('=') for field in line.split()[1:])} for line in content.splitlines()}
   elif filename in ('cpu.max','cpu.weight'):
    snapshot[filename]=content.strip()
   else:
    fields={k:int(v) for k,v in (line.split() for line in content.splitlines())}
    snapshot[filename]=fields if filename in ('memory.events','cpu.stat') else {k:v for k,v in fields.items() if k in ['pgfault','pgmajfault','workingset_refault_file','workingset_activate_file','pgscan','pgsteal']}
  except (OSError,ValueError) as exc:snapshot[filename]={'unavailable':str(exc)}
 if engine is not None:
  try:
   fields=Path(f'/proc/{engine}/stat').read_text().rsplit(')',1)[1].split()
   snapshot['process_faults']={'pid':engine,'minor':int(fields[7]),'major':int(fields[9])}
  except (OSError,ValueError):pass
 return snapshot

def memory_interval(before, after):
 interval={'start_monotonic_s':before['t'],'end_monotonic_s':after['t'],'elapsed_s':after['t']-before['t']}
 for key in ['memory.events','memory.stat','process_faults','cpu.stat']:
  left,right=before.get(key,{}),after.get(key,{})
  if key=='process_faults' and left.get('pid')!=right.get('pid'):continue
  interval[key+'_delta']={k:right[k]-v for k,v in left.items() if k!='pid' and isinstance(v,int) and isinstance(right.get(k),int) and right[k]>=v}
 interval['memory.pressure_total_us_delta']={kind:after['memory.pressure'][kind]['total']-values['total'] for kind,values in before.get('memory.pressure',{}).items() if isinstance(values,dict) and 'total' in values and 'total' in after.get('memory.pressure',{}).get(kind,{})}
 return interval

profile_thread=None
stop=threading.Event();phase='load';start=time.monotonic();result={'special_token_policy':command_manifest['special_token_policy']};hz=os.sysconf('SC_CLK_TCK')
memory_boundaries={'load_start':memory_snapshot(cell_cgroup,server_pid())}
def monitor():
 with (out/'samples.jsonl').open('w') as f:
  while not stop.wait(.5):
   try:
    avail=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,power.limit,power.draw','--format=csv,noheader,nounits'],text=True).strip().split(',')
    actual_engine=server_pid()
    engine=actual_engine if actual_engine is not None else p.pid
    rss=int(next(l.split()[1] for l in Path(f'/proc/{engine}/status').read_text().splitlines() if l.startswith('VmRSS:')))
    tasks={}
    for d in Path(f'/proc/{engine}/task').glob('*'):
     try:
      s=(d/'stat').read_text().rsplit(')',1)[1].split();tasks[d.name]=int(s[11])+int(s[12])
     except FileNotFoundError:pass
    f.write(json.dumps({'t':time.monotonic(),'phase':phase,'engine_pid':engine,'engine_role':'server' if actual_engine is not None else 'wrapper','rss_kib':rss,'tasks':tasks,'avail_kib':avail,'vram_mib':float(gpu[0]),'power_limit':float(gpu[1]),'power_draw':float(gpu[2]),'memory':memory_snapshot(cell_cgroup,actual_engine)})+'\n');f.flush()
    single_b1_failures=('legacy one-pass layer reservation failed before applying prompt state','cache loan made no physical progress; using normal prefill','cache loan made no logical or physical progress; using normal prefill','layer-sized B1 fallback uses 512-token normal chunks')
    if os.environ.get('FN_REQUIRE_SINGLE_B1')=='1' and any(marker in (out/'server.log').read_text(errors='replace') for marker in single_b1_failures):
     result['guard']='required single-pass B1 reservation failed';signal_owned(signal.SIGTERM);return
    if shutil.disk_usage('/').free<40*10**9 or shutil.disk_usage('/mnt/models').free<15*10**9 or shutil.disk_usage('/mnt/overflow').free<50*10**9 or avail<10*1024**2 or float(gpu[0])>23552 or float(gpu[1])!=350 or time.monotonic()-start>int(os.environ.get('FN_LIMIT','1140')):
     result['guard']='disk/RAM/VRAM/power/time guard';signal_owned(signal.SIGTERM);return
   except (FileNotFoundError,ProcessLookupError):
    if p.poll() is not None:return
   except Exception as exc:
    result['guard']='monitor failure: '+repr(exc)
    signal_owned(signal.SIGTERM)
    return
thread=threading.Thread(target=monitor);thread.start()
def terminate_cell(signum, frame):
 raise RuntimeError('cell interrupted by signal '+str(signum))
signal.signal(signal.SIGTERM, terminate_cell)
signal.signal(signal.SIGINT, terminate_cell)
try:
 time.sleep(.2)
 cg=Path('/sys/fs/cgroup')/Path(f'/proc/{p.pid}/cgroup').read_text().strip().split('::')[1].lstrip('/')
 (out/'cpu-settings.json').write_text(json.dumps({'affinity':sorted(os.sched_getaffinity(p.pid)),**{n:(cg/n).read_text() if (cg/n).exists() else 'unavailable' for n in ['cpu.weight','cpu.max','cpuset.cpus.effective','memory.max']}}))
 for i in range(900):
  if p.poll() is not None:raise RuntimeError('server exited '+str(p.returncode))
  try:
   if requests.get('http://127.0.0.1:8095/health',timeout=1).status_code==200:break
  except requests.RequestException:pass
  time.sleep(1)
 else:raise RuntimeError('load timeout')
 print('FIRST SERVER LOG LINES\n'+ '\n'.join((out/'server.log').read_text().splitlines()[:35]),flush=True)
 startup=(out/'server.log').read_text()
 assert 'SJ-KVaRN' in startup and 'n_threads = 8' in startup
 assert 'offloaded 50/50 layers to GPU' in startup
 assert 'successfully fit params to free device memory' in startup
 assert '[B1] layer-major prefill ready' in startup
 assert '[moe-cache] configured: mode=on' in startup
 if 'profile_provenance' in command_manifest:
  expected=command_manifest['profile_provenance']
  assert file_provenance(expected['path'])==expected, 'profile changed during model load'
  result['profile_provenance']=expected
 if os.environ.get('FN_TEACHER_TOKENS'):
  assert 'n_layer=48 n_layer_nextn=0 teacher=1' in startup
  assert 'n_layer=48 n_layer_nextn=1 teacher=0' in startup
 print('B1/FIT/CACHE CHECK\n'+'\n'.join(l for l in startup.splitlines() if any(k in l.lower() for k in ['[b1]','moe-cache','fit:','buffer size'])),flush=True)
 engine=server_pid(required=True)
 running_server=Path(f'/proc/{engine}/exe').stat()
 if (running_server.st_ino,running_server.st_dev)!=(command_manifest['server_provenance']['inode'],command_manifest['server_provenance']['device']):raise RuntimeError('server executable changed between launch and health check')
 result['server_process']={'pid':engine,'executable':server_executable,'cgroup':cell_cgroup_rel,'wrapper_pid':p.pid}
 maps=Path(f'/proc/{engine}/maps').read_text().splitlines()
 library_maps=[(line.split(None,5)[5],int(line.split()[4])) for line in maps if len(line.split(None,5))==6 and '.so' in line.split(None,5)[5]]
 libraries=sorted({name for name,inode in library_maps})
 (out/'loaded-libraries.txt').write_text('\n'.join(libraries))
 provenance={}
 for library in libraries:
  if Path(library).name.startswith(('libggml','libllama','libmtmd')):
   if library.endswith(' (deleted)'):raise RuntimeError('loaded task library was deleted: '+library)
   if not Path(library).resolve().is_relative_to(Path(env['LD_LIBRARY_PATH'].split(':')[0]).resolve()):raise RuntimeError('loaded project library is outside the selected build: '+library)
   info=file_provenance(library)
   mapped={inode for name,inode in library_maps if name==library}
   info['mapped_inode_matches']=mapped=={info['inode']}
   if not info['mapped_inode_matches']:raise RuntimeError('loaded library changed on disk: '+library)
   provenance[library]=info
 command_manifest['library_provenance']=provenance
 build_manifest=Path(env['LD_LIBRARY_PATH'].split(':')[0])/'commands.json'
 if build_manifest.is_file():command_manifest['build_commands_provenance']=file_provenance(build_manifest)
 (out/'command.json').write_text(json.dumps(command_manifest,indent=2))
 result['binary_provenance']={'server':command_manifest['server_provenance'],'libraries':provenance}
 if 'build_commands_provenance' in command_manifest:result['binary_provenance']['build_commands']=command_manifest['build_commands_provenance']
 if os.environ.get('FN_PROFILE') and os.environ.get('FN_TRACE_PHASE')=='prompt':
  phase='prompt-trace-start'
  with (out/'nsys-control.log').open('w') as nf:
   subprocess.run([nsys,'start','--session=fnfast-'+cell,'--sample=none','--cpuctxsw=none','--output='+str(out/'prompt')],stdout=nf,stderr=nf,env=env,timeout=25,check=True)
 ids=json.loads(Path(os.environ.get('FN_PROMPT_IDS','/mnt/models/stratamatch-home/prompt-ids.json')).read_text());prompt=ids[:depth] if depth else ids[-16:]
 def stacks():
  time.sleep(3)
  for i in range(8):
   if p.poll() is not None:return
   with (out/f'stack-{i}.txt').open('w') as f:subprocess.run(['gdb','-q','-batch','-ex','set pagination off','-ex','thread apply all bt 8','-ex','thread 1','-ex','frame 2','-ex','info args','-ex','info locals','-p',str(server_pid(required=True))],stdout=f,stderr=f,timeout=15)
   time.sleep(1)
 if os.environ.get('FN_RESTORE_FROM'):
  phase='initial-cache-restore'
  restored=requests.post('http://127.0.0.1:8095/slots/0?action=restore',json={'filename':'prompt.bin'},timeout=180);restored.raise_for_status();result['initial_cache_restore']=restored.json()
 if os.environ.get('FN_NEGATIVE'):
  def post_json(url,payload,timeout=180):
   response=requests.post(url,json=payload,timeout=timeout);response.raise_for_status();return response.json()
  phase='negative-resume-tests'
  runpy.run_path(str(Path('/home/jake-k/qwen38-bench/kernel_cache_investigation/frontier/BACKLOG_RUN_20261015/FNFAST/resume-negative.py')))['run_negative_cases'](post_json,'http://127.0.0.1:8095',cache_dir,out)
  result['status']='complete';result['negative_resume_tests']=json.loads((out/'resume-negative.json').read_text())
  raise DiagnosticComplete()
 payload={'prompt':prompt,'n_predict':int(os.environ.get('FN_TOKENS','512')),'temperature':float(os.environ.get('FN_TEMP','0')),'seed':int(os.environ.get('FN_SEED','11')),'ignore_eos':True,'cache_prompt':False,'stream':True,'return_tokens':True}
 if True:payload.update(top_k=40,top_p=.95,min_p=0,repeat_penalty=1.,samplers=['top_k','top_p','temperature'],logit_bias=[[i,False] for i in command_manifest['special_token_policy']['banned_token_ids']])
 if os.environ.get('FN_CACHE_TEST') or os.environ.get('FN_RESTORE_FROM'):payload['cache_prompt']=True
 (out/'request.json').write_text(json.dumps(payload));phase='prefill';t0=time.monotonic();result['request_start_monotonic_s']=t0
 memory_boundaries['before_request']=memory_snapshot(cell_cgroup,server_pid())
 with requests.post('http://127.0.0.1:8095/completion',json=payload,stream=True,timeout=1160) as r,(out/'response.jsonl').open('w') as f:
  r.raise_for_status()
  for line in r.iter_lines():
   if not line:continue
   s=line.decode();f.write(json.dumps({'t':time.monotonic(),'line':s})+'\n');f.flush()
   if s.startswith('data: ') and s[6:]!='[DONE]':
    o=json.loads(s[6:])
    if o.get('content') or o.get('tokens'):
     if phase != 'decode':
      result['first_token_s']=time.monotonic()-t0
      memory_boundaries['first_token']=memory_snapshot(cell_cgroup,server_pid())
      prompt_log=(out/'server.log').read_text()
      if 'profile_provenance' in command_manifest:
       expected=command_manifest['profile_provenance']
       assert '[moe-cache] profile loaded: '+expected['path']+',' in prompt_log and 'sampled weight identity verified' in prompt_log, 'profile did not load at generation handoff with model identity verification'
       assert file_provenance(expected['path'])==expected, 'profile changed at generation handoff'
      b1_rates=[l for l in prompt_log.splitlines() if '[B1] done:' in l]
      print('B1 TIMING CHECK\n'+'\n'.join(b1_rates),flush=True)
      assert depth < 100000 or b1_rates or os.environ.get('FN_RESTORE_FROM'), '100K timed prompt did not use B1'
      if os.environ.get('FN_PROFILE'):
       def capture():
        if os.environ.get('FN_TRACE_PHASE')!='prompt':time.sleep(2)
        with (out/'nsys-control.log').open('a') as f:
         if os.environ.get('FN_TRACE_PHASE')!='prompt':
          subprocess.run([nsys,'start','--session=fnfast-'+cell,'--sample=none','--cpuctxsw=none','--output='+str(out/'decode')],stdout=f,stderr=f,env=env,timeout=20)
          time.sleep(6)
         subprocess.run([nsys,'stop','--session=fnfast-'+cell],stdout=f,stderr=f,env=env,timeout=35)
       profile_thread=threading.Thread(target=capture);profile_thread.start()
     if phase!='decode' and os.environ.get('FN_STACKS','1')=='1':threading.Thread(target=stacks,daemon=True).start()
     phase='decode'
    if o.get('stop'):result['final']=o
 result['request_end_monotonic_s']=time.monotonic()
 result['wall_s']=result['request_end_monotonic_s']-t0
 result['status']='complete' if 'final' in result else 'failed'
 if 'final' in result and result['final'].get('tokens_predicted')!=payload['n_predict']:raise RuntimeError('generated-token count mismatch')
 if 'final' in result and result['final'].get('tokens_evaluated')!=len(prompt):raise RuntimeError('prompt-depth mismatch')
 if os.environ.get('FN_CACHE_TEST') and 'final' in result:
  phase='cache-save'
  saved=requests.post('http://127.0.0.1:8095/slots/0?action=save',json={'filename':'prompt.bin'},timeout=180);saved.raise_for_status();result['cache_save']=saved.json()
  erased=requests.post('http://127.0.0.1:8095/slots/0?action=erase',json={},timeout=60);erased.raise_for_status()
  restored=requests.post('http://127.0.0.1:8095/slots/0?action=restore',json={'filename':'prompt.bin'},timeout=180);restored.raise_for_status();result['cache_restore']=restored.json()
  phase='cache-reload-decode';again=dict(payload,stream=False)
  rr=requests.post('http://127.0.0.1:8095/completion',json=again,timeout=900);rr.raise_for_status();result['cache_reload_final']=rr.json()
  first_ids=[]
  for line in (out/'response.jsonl').read_text().splitlines():
   wire=json.loads(line)['line']
   if wire.startswith('data: ') and wire[6:]!='[DONE]':first_ids+=json.loads(wire[6:]).get('tokens',[])
  result['cache_exact']=first_ids==result['cache_reload_final'].get('tokens')
  result['cache_first_divergence'] = next((i for i,(a,b) in enumerate(zip(first_ids,result['cache_reload_final'].get('tokens',[]))) if a != b), None)

except DiagnosticComplete:pass
except Exception as e:result.update(status='failed',error=repr(e))
finally:
 if 'profile_provenance' in command_manifest:
  expected=command_manifest['profile_provenance']
  if file_provenance(expected['path'])!=expected:result.update(status='failed',error='profile changed during benchmark')
 if 'request_start_monotonic_s' in result:result.setdefault('request_end_monotonic_s',time.monotonic())
 if profile_thread is not None:profile_thread.join(55)
 stop.set();thread.join()
 memory_boundaries['before_cleanup']=memory_snapshot(cell_cgroup,server_pid())
 result['memory_measurement']={'boundaries':memory_boundaries,'intervals':{name:memory_interval(memory_boundaries[left],memory_boundaries[right]) for name,left,right in [('startup','load_start','before_request'),('prompt','before_request','first_token'),('response','first_token','before_cleanup'),('whole_cell','load_start','before_cleanup')] if left in memory_boundaries and right in memory_boundaries},'note':'Cgroup event/PSI/fault counters are cumulative; intervals use boundary differences. Streaming first-token boundary includes network buffering. Response runs through controller cleanup and includes any auxiliary reload requests. Periodic samples preserve progression.'}
 try:
  result['cgroup_memory']={n:(cg/n).read_text() for n in ['memory.current','memory.peak','memory.swap.current','memory.events']}
 except Exception:pass
 result['wrapper_returncode_before_cleanup']=p.poll()
 signal_owned(signal.SIGTERM)
 if p.poll() is None:
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:signal_owned(signal.SIGKILL);p.wait()
 result['remaining_cell_processes']=cleanup_owned_processes()
 result['wrapper_returncode_after_cleanup']=p.poll()
 if result['remaining_cell_processes']:result.update(status='failed',guard='owned cell cleanup left processes')
 stop.set();thread.join();log.close()
 samples=[json.loads(l) for l in (out/'samples.jsonl').read_text().splitlines()];ds=[s for s in samples if s['phase']=='decode']
 if len(ds)>3:ds=ds[1:-2]
 if len(ds)>1:
  elapsed=ds[-1]['t']-ds[0]['t'];result['decode_thread_busy_pct']={tid:round(100*(ticks-ds[0]['tasks'].get(tid,ticks))/hz/elapsed,2) for tid,ticks in ds[-1]['tasks'].items()};result['decode_sample_s']=elapsed
 result['peaks']={'server_rss_kib':max((s['rss_kib'] for s in samples if s.get('engine_role')=='server'),default=0),'vram_mib':max((s['vram_mib'] for s in samples),default=0),'min_memavailable_kib':min((s['avail_kib'] for s in samples),default=0)}
 (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if result.get('status') != 'complete':
 sys.exit(1)
