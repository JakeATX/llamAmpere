import hashlib, json, os, shutil, signal, subprocess, time
from pathlib import Path
root=Path('/mnt/models/fnprefetch')
assert os.environ.get('PARENT_HOLDS_GPU_LOCK')=='1'
assert subprocess.check_output(['systemctl','--user','is-active','llama.cpp-sysguard'],text=True).strip()=='active'
config=json.loads(Path('/home/jake-k/qwen38-bench/kernel_cache_investigation/frontier/BACKLOG_RUN_20261015/FNCONFIG_HUNT/runs/confirm-best-r2/command.json').read_text())
frozen_file=root/'logs/frozen-bin.txt'
frozen=Path(frozen_file.read_text().strip()) if frozen_file.is_file() else root/'build/bin'
binary=Path(os.environ.get('FN_GOLDEN_BINARY',str(frozen/'test-moe-prefetch-model')))
lib=Path(os.environ.get('FN_GOLDEN_LIB',str(frozen)))
base=[str(binary),'-m',config['argv'][config['argv'].index('-m')+1],'--load-mode','mmap','-c','8192','-ngl','999','--cpu-moe','--moe-cache','on','-ctk','sjkvarn4','-ctv','sjkvarn4','-b','2048','-ub','512','-t','8','-fa','on','-fitt','1536','-lv','4']
label=os.environ.get('FN_GOLDEN_LABEL','legacy')
if label=='baseline-e3':
 unit='llama.cpp-fnprefetch-baseline-golden-e3.service'
 assert Path('/proc/self/cgroup').read_text().strip().endswith('/'+unit)
 subprocess.run(['systemctl','--user','set-property','--runtime',unit,'CPUWeight=100'],check=True)
mode_dir=root/'logs'/label
mode_dir.mkdir(exist_ok=True)
def identity(path):
 path=path.resolve()
 return {'path':str(path),'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest()}
(mode_dir/'provenance.json').write_text(json.dumps({'binary':identity(binary),'libraries':[identity(p) for p in sorted(lib.glob('lib*.so'))]},indent=2))
start=time.monotonic()
deadline=float(os.environ.get('FN_GATE_DEADLINE',start+540))
modes=os.environ.get('FN_GOLDEN_MODES','off,1,2,3,auto').split(',')
for mode in modes:
 assert mode in ('off','1','2','3','auto','baseline')
 env=dict(os.environ,**config['env'], FN_GOLDEN_IDS=str(root/'protocol/golden-ids.bin'),FN_GOLDEN_LOGITS=str(mode_dir/f'golden-{mode}.f32'),CUDA_CACHE_PATH=str(root/'cuda-cache'),TMPDIR=str(root/'tmp-build'), LD_LIBRARY_PATH=str(lib))
 env['GGML_MOE_CACHE_DEVICE']=os.environ.get('FN_GOLDEN_DEVICE','0')
 env['LLAMA_B1_CACHE_WARM']='1'
 env['GGML_CUDA_MOE_CACHE_RESERVE_MB']='2560'
 with (mode_dir/f'golden-{mode}.log').open('w') as log:
  command=base+([] if mode=='baseline' else ['--moe-prefetch',mode])
  process=subprocess.Popen(command,env=env,stdout=log,stderr=log,start_new_session=True)
  runtime=None
  while process.poll() is None:
   if runtime is None:
    maps=Path(f'/proc/{process.pid}/maps').read_text().splitlines()
    paths=sorted({x.split(None,5)[5] for x in maps if len(x.split(None,5))==6 and Path(x.split(None,5)[5]).name.startswith(('libggml','libllama','libmtmd'))})
    if any(Path(p).name.startswith('libggml-cuda') for p in paths):
     runtime={}
     for path in paths:
      assert not path.endswith(' (deleted)') and Path(path).resolve().is_relative_to(lib.resolve()),path
      inode=Path(path).stat().st_ino
      assert {int(x.split()[4]) for x in maps if len(x.split(None,5))==6 and x.split(None,5)[5]==path}=={inode}
      runtime[path]={**identity(Path(path)),'inode':inode}
     (mode_dir/f'golden-{mode}.libraries.json').write_text(json.dumps(runtime,indent=2))
   available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
   used=float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
   if available<10*1024**2 or used>23552 or time.monotonic()>deadline or any(shutil.disk_usage(path).free<floor*10**9 for path,floor in [('/',40),('/mnt/models',15),('/mnt/overflow',50)]):
    os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=5);raise RuntimeError('resource/time guard')
   time.sleep(.5)
 assert process.returncode==0,(mode,process.returncode)
 assert runtime,'loaded CUDA project libraries were not verified'
 assert 'roundtrip=exact' in (mode_dir/f'golden-{mode}.log').read_text()
print('Fixed-token logits and exact saved-state round trips completed: '+','.join(modes),flush=True)
