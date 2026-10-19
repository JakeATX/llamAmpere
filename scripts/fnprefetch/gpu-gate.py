import json, os, signal, shutil, subprocess, time
from pathlib import Path
root=Path('/mnt/models/fnprefetch')
binary_dir=Path(os.environ.get('FN_GATE_BIN',(root/'logs/frozen-bin.txt').read_text().strip()))
assert os.environ.get('PARENT_HOLDS_GPU_LOCK')=='1'
assert subprocess.check_output(['systemctl','--user','is-active','llama.cpp-sysguard'],text=True).strip()=='active'
env=dict(os.environ, GGML_MOE_CACHE_TEST_TMPDIR=str(root/'tmp-build'), CUDA_CACHE_PATH=str(root/'cuda-cache'), LD_LIBRARY_PATH=str(binary_dir))
commands=[
 [str(binary_dir/'test-backend-ops'),'-b','CPU','-o','MUL_MAT_ID','-p','type_a=iq3_xxs,.*m=512,n=1,k=256,'],
 [str(binary_dir/'test-moe-cache')],
 [str(binary_dir/'test-backend-ops'),'-b','CUDA0','-o','MUL_MAT_ID','-p','type_a=iq3_xxs,.*m=512,n=1,k=256,'],
]
if os.environ.get('FN_GATE_SKIP_CACHE')=='1':commands=[commands[0],commands[2]]
output_dir=root/'logs'/('operators-'+binary_dir.parent.name)
output_dir.mkdir(exist_ok=True)
started=time.monotonic();deadline=float(os.environ.get('FN_GATE_DEADLINE',started+540));results=[]
for i,cmd in enumerate(commands):
 with (output_dir/f'gate-{i}.log').open('w') as out:
  p=subprocess.Popen(cmd,env=env,stdout=out,stderr=out,start_new_session=True)
  while p.poll() is None:
   available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
   memory=float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
   if available<10*1024**2 or memory>23552 or time.monotonic()>deadline or any(shutil.disk_usage(path).free<floor*10**9 for path,floor in [('/',40),('/mnt/models',15),('/mnt/overflow',50)]):
    os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=5);raise RuntimeError('resource or time guard')
   time.sleep(.5)
 text=(output_dir/f'gate-{i}.log').read_text()
 if Path(cmd[0]).name=='test-moe-cache':
  assert 'cache-prefetch-router: OK' in text and 'FAIL' not in text
  assert 'cache-arena-stride: OK' in text, 'real padded-stride fixture did not run'
  for kind in ('legacy','device','fused'):
   assert text.count('cache-prefetch-staging-'+kind+': OK')==2, 'both packed and strided staging paths must pass'
 else:
  import re
  counts=re.findall(r'(\d+)/(\d+) tests passed',text)
  assert counts and all(int(a)==int(b) and int(b)>0 for a,b in counts),text[-2000:]
 results.append({'argv':cmd,'rc':p.returncode})
 (output_dir/'gates.json').write_text(json.dumps(results,indent=2))
 assert p.returncode==0
print('GPU gates passed',flush=True)
