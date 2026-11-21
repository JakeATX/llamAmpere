import json, collections, statistics as st, sys
rows=[json.loads(l) for l in open(sys.argv[1])]
order=[5,11,17,29,35,41,47,23]
mid=collections.defaultdict(list); sel=collections.defaultdict(list)
for r in rows:
    if r.get('kind')=='mid' and r['W']==1024: mid[(r['il'],r['bucket'])].append(r)
    elif r.get('kind')=='sel': sel[(r['il'],r['W'],r['B'],r['budget'],r['sel'])].append(r)
print('visit bucket n mid_share sink c50 c90 c99 n_mid')
for il in order:
    for b in ['0-32K','32-64K','64-96K','96-128K']:
        v=mid.get((il,b))
        if not v: continue
        m=lambda f: st.mean(x[f] for x in v)
        print(il,b,len(v),'%.3f %.3f %.0f %.0f %.0f %.0f'%(m('mid_share'),m('sink'),m('c50'),m('c90'),m('c99'),m('n_mid')))
print('\nW1024 B64: KL(miss) err per visit')
for bud in ['k2048','k4096','k8192','k16384','f0.05','f0.1','f0.25']:
    for il in order:
        out=[]
        for s in ['oracle','oracle_sh','vote','mean','quest']:
            v=sel.get((il,1024,64,bud,s))
            if v: out.append('%s %.3f(%.3f)e%.3f'%(s,st.mean(x['kl'] for x in v),st.mean(x['miss'] for x in v),st.mean(x['err'] for x in v)))
        print(bud,il,' | '.join(out))
print('\nvote KL by W/B')
for il in order:
    o=[]
    for W in (1024,4096):
        for B in (64,128):
            for bud in ('k4096','k16384'):
                v=sel.get((il,W,B,bud,'vote'))
                if v: o.append('W%d B%d %s %.3f'%(W,B,bud,st.mean(x['kl'] for x in v)))
    print(il,' ; '.join(o))
for kind in ('loop','step','step16'):
    v=[r for r in rows if r.get('kind')==kind]
    by=collections.defaultdict(list)
    for r in v: by[(r.get('other',r.get('il')),r['keep'])].append(r['overlap'])
    print(kind, {k:round(st.mean(x),3) for k,x in sorted(by.items())})
