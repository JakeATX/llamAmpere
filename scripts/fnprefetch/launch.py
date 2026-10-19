import os,sys,subprocess
from pathlib import Path
os.environ.update(XDG_RUNTIME_DIR='/run/user/1000',DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus')
cell=sys.argv[1]
limit=int(os.environ.get('FN_CELL_TIMEOUT','580'))
assert 0 < limit <= 1490
memory_max=os.environ.get('FN_MEMORY_MAX','46G')
assert memory_max in ('46G','50G'), 'FN_MEMORY_MAX must be 46G or 50G; global memory guard remains active'
if Path(os.environ.get('FN_RUNNER',str(Path(__file__).with_name('control.py')))).name=='gpu-gate.py' and 'FN_GATE_BIN' not in os.environ:
 os.environ['FN_GATE_BIN']=Path('/mnt/models/fnprefetch/logs/frozen-bin.txt').read_text().strip()
with (Path('/mnt/models/fnprefetch/protocol')/(cell+'.unit.log')).open('w') as f:
 rc=subprocess.call(['flock','/home/jake-k/qwen38-bench/.locks/gpu.lock','systemd-run','--user','--wait','--unit=llama.cpp-fnprefetch-'+cell,'-p','CPUWeight=100','-p','MemoryMax='+memory_max,'-p','TimeoutStopSec=10','-p','LimitCORE=0','-p','WorkingDirectory='+'/mnt/models/fnprefetch/protocol','env','PARENT_HOLDS_GPU_LOCK=1',*[k+'='+v for k,v in os.environ.items() if k.startswith('FN_')],'timeout','--signal=TERM','--kill-after=10',str(limit),'/usr/bin/python3',str(Path(os.environ.get('FN_RUNNER',str(Path(__file__).with_name('control.py')))).resolve()),*sys.argv[1:]],stdout=f,stderr=subprocess.STDOUT)
print(cell,rc)
sys.exit(rc)
