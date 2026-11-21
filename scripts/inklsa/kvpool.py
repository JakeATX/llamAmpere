#!/usr/bin/env python3
"""KV block pool sim for INKLSA host body (P2/P3): vote picks (block-mean index, B, keep) on the dumped runs of 64
consecutive queries per visit; LRU pool of P blocks per visit; reports misses (blocks fetched over PCIe) per token."""
import numpy as np, os, sys, json
D='/mnt/data/inklsa/dump/doc128k'; NH,NKV,HD=24,8,128; B=64; W=1024; S=128
scale=1/np.sqrt(HD)
out={}
for il in [5,11,17,29,35,41,47,23]:
    K=np.fromfile(f'{D}/K.L{il}.f16',dtype=np.float16).reshape(-1,NKV,HD)
    Q=np.fromfile(f'{D}/Q.L{il}.f32',dtype=np.float32).reshape(-1,NH,HD)
    P=np.fromfile(f'{D}/pos.L{il}.i64',dtype=np.int64)
    nb=K.shape[0]//B
    mk=K[:nb*B].astype(np.float32).reshape(nb,B,NKV,HD).mean(1)      # [nb,8,128]
    runs=[np.where((P//32768==r)&(P%32768>=32768-64))[0] for r in range(4)]
    for keep in (4096,8192,16384):
        nk=keep//B
        for pool in (nk, 2*nk, 4*nk):
            miss=[];
            for idx in runs:
                if len(idx)<2: continue
                lru={}; t=0; m=[]
                for qi in idx:
                    p=int(P[qi]); b_lo=S//B; b_hi=max(b_lo,(p-W+1)//B)
                    q=Q[qi].reshape(NKV,3,HD)
                    s=np.einsum('bgd,ghd->ghb',mk[b_lo:b_hi],q).reshape(NH,-1)*scale
                    s=np.exp(s-s.max(1,keepdims=True)); s/=s.sum(1,keepdims=True)
                    pick=b_lo+np.argsort(-s.sum(0))[:nk]
                    new=[b for b in pick if b not in lru]
                    for b in pick: lru[b]=t
                    if len(lru)>pool:
                        for b,_ in sorted(lru.items(),key=lambda x:x[1])[:len(lru)-pool]: del lru[b]
                    if t>0: m.append(len(new))
                    t+=1
                miss.append((int(P[idx[0]])//1024, float(np.mean(m))))
            out[(il,keep,pool)]=miss
            print(il,keep,pool,[(f,round(x,1),round(x/nk,3)) for f,x in miss],flush=True)
json.dump({str(k):v for k,v in out.items()},open('/mnt/data/inklsa/runs/kvpool.json','w'))
