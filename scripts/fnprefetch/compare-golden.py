import json, os, struct, sys
import numpy as np
from pathlib import Path
root=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/mnt/models/fnprefetch/logs/legacy')
def load(mode, directory=root):
 path=directory/f'golden-{mode}.f32'
 with path.open('rb') as f:
  vocab,rows=struct.unpack('<ii',f.read(8))
 data=np.memmap(path,dtype='<f4',offset=8,shape=(rows,vocab))
 assert np.isfinite(data).all()
 return data
base=load('off');results={}
baseline=load('baseline',Path(sys.argv[2])) if len(sys.argv)>2 else None
for mode in os.environ.get('FN_COMPARE_MODES','off,1,2,3,auto').split(','):
 data=load(mode);assert data.shape==base.shape
 divergences=[]
 for a,b in zip(base,data):
  a=a.astype(np.float64);b=b.astype(np.float64)
  a-=np.logaddexp.reduce(a);b-=np.logaddexp.reduce(b)
  divergences.append(float((np.exp(a)*(a-b)).sum()))
 log=(root/f'golden-{mode}.log').read_text()
 results[mode]={'mean_kl_nats':float(np.mean(divergences)),'max_kl_nats':max(divergences),
                'max_abs_logit_difference':float(np.max(np.abs(base-data))),
                'queued_predictions':log.count('[moe-prefetch]'),'completed_copies':log.count('canceled=0'),'canceled_predictions':log.count('canceled=1'),'copies_used':log.count('outcome=used'),
                'state_roundtrip_exact':'roundtrip=exact' in log}
 if baseline is not None:
  assert baseline.shape==data.shape
  kl=[]
  for a,b in zip(baseline,data):
   a=a.astype(np.float64);b=b.astype(np.float64)
   a-=np.logaddexp.reduce(a);b-=np.logaddexp.reduce(b)
   kl.append(float((np.exp(a)*(a-b)).sum()))
  results[mode]['previous_build_mean_kl_nats']=float(np.mean(kl))
  results[mode]['previous_build_max_kl_nats']=max(kl)
(root/'golden-comparison.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
