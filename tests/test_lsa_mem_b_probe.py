#!/usr/bin/env python3
"""Small offline mutations of the runtime checker; no model or device work."""
import ast
import copy
import csv
import json
import math
import re
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

SRC = Path(__file__).resolve().parents[1] / 'scripts/lsa_mem_b_probe.py'
tree = ast.parse(SRC.read_text())
with tempfile.TemporaryDirectory(prefix='lsa-memB-check-') as directory:
    root = Path(directory)
    ns = dict(csv=csv,json=json, math=math, re=re, time=SimpleNamespace(time=lambda:1010),
              collect_only=False,p=SimpleNamespace(O=root,B='/fixture'), FOOTER='LSA MEMB COMPLETE ')
    ns['config'] = lambda *a:SimpleNamespace(model='fixture',ctx=110592,port=18449,slot_dir='/slots',env=['LLAMA_LSA_MODE=key'])
    ns['xver'] = SimpleNamespace(arms=SimpleNamespace(server_cmd=lambda *a:['llama-server','-m','fixture']))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ('check_cuda', 'contract', 'resources', 'expected_headers', 'read_log', 'stop', 'check_server'):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(SRC), 'exec'), ns)
    samples=''.join(json.dumps(dict(epoch=1000+i,ram_kib=9*2**20,vram_mib=17000))+'\n' for i in range(11))
    resource_receipt=dict(started=1000,ended=1010,samples=11,first=1000,last=1010)
    def requests(name):
        if name in ('a0','b0','a1','b1'):
            return [dict(prompt_n=98000,predicted_n=64,prompt_len=98000,tokens=64),
                    dict(prompt_n=208,predicted_n=512,prompt_len=98000,tokens=512)]
        return [dict(prompt_n=100000,predicted_n=1,prompt_len=100000,tokens=0)] + [
            dict(prompt_n=2 if i==0 else 4,predicted_n=1,prompt_len=100001+4*i,tokens=0) for i in range(128)]
    def footer(name):
        return ns['FOOTER']+json.dumps(dict(arm=name,pid=123,rc=0,
            resolves=ns['contract'](name,requests(name)), requests=requests(name),resources=resource_receipt))+'\n'
    def valid(name):
        text='\n'.join(ns['expected_headers'](name))+'\n'
        if name in ('b0','b1','f16_20_stage'):
            slots,records,stride=458,831,91136
            text+=f'LSA RESIDENT: 128-cell record slots, records={slots} heads=4 ({slots*stride/2**20:.1f} MiB per layer)\n'
            text+=f'LSA RESIDENT: stage of {records} records ({records*stride/2**20:.1f} MiB) for records past the slots\n'
            n=ns['contract'](name,requests(name))
            for layer in range(3,64,4):
                text+=f'LSA resident layer {layer} (K3/V2 record pages): {slots} of {records} records on the card (55.0%), {slots*stride/2**20:.1f} MiB slots vs {records*stride/2**20:.1f} MiB body, {records*stride/2**20:.1f} MiB pinned\n'
                for resolve in range(16,n+1,16):
                    text+=f'LSA STAGE: layer {layer} resolves={resolve} requested=42 capacity={records} clock_over_slots=42 host_records=0\n'
                    text+=f'LSA RESIDENT: layer {layer} resolves={resolve} picked={resolve*500} fetched={resolve*10} over_slots=0 slots={slots} record_bytes={stride}\n'
                text+=f'LSA RESIDENT: layer {layer} slots={slots} records, resolves={n} picked={n*500} fetched={n*10} over_slots=0, per resolve: 0.87 MiB fetched + 0.00 MiB read from host past the slots\n'
                text+=f'LSA RESIDENT: layer {layer} stage={records} records, staged={(n-1)*42} stage_full=0, per resolve: {(n-1)*42*stride/2**20/n:.2f} MiB copied to the stage, 0.00 MiB still read from host\n'
        return text+footer(name)
    pos=neg=0
    def check(name,text,passes=False,resource_text=samples):
        global pos,neg
        ns['resident']=name in ('b0','b1','f16_20_stage')
        (root/f'{name}.server.log').write_text(text)
        sample_path=root/f'{name}.resources.jsonl'
        if resource_text is None:
            sample_path.unlink(missing_ok=True)
        else:
            sample_path.write_text(resource_text)
        try:
            ns['read_log'](name)
        except (AssertionError,ValueError,TypeError,KeyError,IndexError,ZeroDivisionError,OSError):
            assert not passes, (name,'valid rejected')
            neg+=1
        else:
            assert passes,(name,'false PASS')
            pos+=1
    for name in ('a0','b0','a1','b1','f16_20_nores','f16_20_stage'):
        text=valid(name)
        check(name,text,True)
        for bad in ('',text[:-1],text[:len(text)//2],text.replace('cmd: ','other: ',1),
                    text.replace('explicit env: ','other env: ',1),text+footer(name),
                    text.replace(text.splitlines()[0],'cmd: ',1),
                    text.replace(text.splitlines()[1],'explicit env: ',1),
                    text.replace(ns['FOOTER'],text.splitlines()[1]+'\n'+ns['FOOTER']),
                    text.replace('"rc": 0','"rc": 1'),text.replace('"rc": 0','"rc": -11'),
                    text.replace('"requests": [','"requests": [ {"x":0},',1),
                    text.replace('"resolves": ', '"resolves": "',1),
                    text.replace('"predicted_n": 1,','"predicted_n": 0,',1) if 'f16' in name else text.replace('"tokens": 512','"tokens": 511')):
            check(name,bad)
        for sample_text in (None,'',samples[:-1],'{bad json}\n',samples.splitlines()[0]+'\n',
                            samples.replace('9437184','5242879',1),samples.replace('17000','23553',1),
                            samples.replace('"epoch": 1000','"epoch": 900',1),
                            samples+samples,samples.replace('"epoch": 1000','"epoch": NaN',1)):
            check(name,text,resource_text=sample_text)
        for row in requests(name):
            if row['prompt_n']==208:
                for replay in (0,1,8,207,209,98000):
                    check(name,text.replace('"prompt_n": 208',f'"prompt_n": {replay}'))
        for diagnostic in ('CUDA error: late failure','GGML_ASSERT(false)','terminate called','Segmentation fault',
                           'LSA STAGE','LSA RESIDENT','LSA ST','LSA RESIDENT: layer 3 over_slots=1',
                           'LSA STAGE: layer 3 host_records=1'):
            check(name,text.replace(ns['FOOTER'], diagnostic+'\n'+ns['FOOTER']))
        if name in ('b0','b1','f16_20_stage'):
            rows=text.splitlines(keepends=True)
            indices=[i for i,row in enumerate(rows) if re.search(r'LSA (?:STAGE|RESIDENT: layer)',row)]
            # Remove, duplicate and truncate every diagnostic, at every sampling position.
            for i in indices:
                check(name,''.join(rows[:i]+rows[i+1:]))
                check(name,''.join(rows[:i]+[rows[i],rows[i]]+rows[i+1:]))
                check(name,''.join(rows[:i]+[rows[i][:max(15,len(rows[i])//2)]+'\n']+rows[i+1:]))
            for label in ('LSA RESIDENT: stage of ', 'LSA RESIDENT: 128-cell record slots,'):
                line=next(row for row in rows if row.startswith(label))
                check(name,text.replace(line,'',1))
                check(name,text.replace(line,line+line,1))
            for old,new in [('host_records=0','host_records=1'),('over_slots=0','over_slots=1'),
                            ('stage_full=0','stage_full=1'),('requested=42','requested=832'),
                            ('capacity=831','capacity=830'),('layer 3 ','layer 2 '),
                            ('resolves=16 ','resolves=32 '),('slots=458','slots=457'),
                            ('records=458','records=457'),('0.00 MiB read','1.00 MiB read'),
                            ('0.00 MiB still','1.00 MiB still'),('record_bytes=91136','record_bytes=91135')]:
                check(name,text.replace(old,new,1))
            n=ns['contract'](name,requests(name))
            for staged in (0,1,(n-1)*5,n*831):
                # Change the printed rate too: accounting, not a stale rate, must fail.
                old=f'staged={(n-1)*42} stage_full=0, per resolve: {(n-1)*42*91136/2**20/n:.2f}'
                new=f'staged={staged} stage_full=0, per resolve: {staged*91136/2**20/n:.2f}'
                check(name,text.replace(old,new,1))
            for clock in (0,41,43,832,999999):
                check(name,text.replace('clock_over_slots=42',f'clock_over_slots={clock}',1))
            check(name,text.replace('requested=42','requested=0',1))
            # All overflow fields agree, but the read set cannot fit in the slots.
            bad=text.replace('requested=42','requested=0').replace('clock_over_slots=42','clock_over_slots=0')
            bad=re.sub(r'staged=\d+ stage_full=0, per resolve: [0-9.]+',
                       'staged=0 stage_full=0, per resolve: 0.00',bad)
            check(name,bad)
            check(name,text.replace('0.87 MiB fetched','0.00 MiB fetched',1))
            check(name,text.replace('MiB copied to the stage','MiB copied to the stage EXTRA',1))
            check(name,text.replace('resolves=574 ','resolves=128 ',1) if name!='f16_20_stage' else text.replace('resolves=128 picked=64000','resolves=127 picked=64000',1))
            # Old false PASS: eight samples + terminal at 128 for a 576-output arm.
            if name in ('b0','b1'):
                short=valid('f16_20_stage').replace('"f16_20_stage"',f'"{name}"')
                short=short[:short.rfind(ns['FOOTER'])]+footer(name)
                check(name,short)
    cuda_rows=[['0','fixture','CUDA0','FLASH_ATTN_EXT_SJKVARN_TRELLIS',f'n_q={q},nh=24','test','1','1'] for q in (1,8)]
    def cuda_check(rows,passes):
        with (root/'cuda.csv').open('w') as out:
            csv.writer(out).writerows(rows)
        try: ns['check_cuda'](root/'cuda.csv')
        except (AssertionError,AttributeError): assert not passes
        else: assert passes
    cuda_check(cuda_rows,True)
    for bad in ([],cuda_rows[:1],[cuda_rows[0]]*2,
                [[*r[:3],'FLASH_ATTN_EXT',*r[4:]] for r in cuda_rows],
                [[*r[:6],'0','1'] for r in cuda_rows]):
        cuda_check(bad,False)
    # The stop wrapper must never emit a completion footer on error or partial requests.
    stop_positive=stop_negative=0
    for code,partial,breach,dead,missing in ((0,False,False,False,False),(1,False,False,False,False),
            (-11,False,False,False,False),(-15,False,False,False,False),(-9,False,False,False,False),
            (0,True,False,False,False),(0,False,True,False,False),(0,False,False,True,False),
            (0,False,False,False,True)):
        path=root/'a0.server.log';path.write_text('cmd: test\nexplicit env: []\n')
        obj=SimpleNamespace(proc=SimpleNamespace(pid=123,returncode=code),mem_b_name='a0',
                            mem_b_requests=requests('a0')[:-1] if partial else requests('a0'),mem_b_completed=False,
                            mem_b_started=1000,monitor=SimpleNamespace(live=not dead))
        obj.monitor.is_alive=lambda:obj.monitor.live
        (root/'a0.resources.jsonl').write_text('' if missing else samples)
        def base_stop(self):
            self.monitor.live=False
            if breach: raise RuntimeError('guard/timeout')
        ns['base_stop']=base_stop
        try: ns['stop'](obj)
        except (AssertionError,RuntimeError):
            assert code!=0 or partial or breach or dead or missing
            assert ns['FOOTER'] not in path.read_text()
            stop_negative+=1
        else:
            assert code==0 and not partial and not breach and not dead and not missing
            stop_positive+=1
    ns['collect_only']=True
    obj=SimpleNamespace(proc=SimpleNamespace(pid=123,returncode=0),mem_b_name='a0',
                        mem_b_requests=requests('a0')[:-1],mem_b_completed=False,mem_b_started=1000,
                        monitor=SimpleNamespace(live=True))
    obj.monitor.is_alive=lambda:obj.monitor.live
    ns['base_stop']=lambda self:setattr(self.monitor,'live',False)
    (root/'a0.server.log').write_text('cmd: fixture\nexplicit env: []\n')
    (root/'a0.resources.jsonl').write_text(samples)
    ns['stop'](obj)
    assert ns['FOOTER'] in (root/'a0.server.log').read_text()
    ns['collect_only']=False
    ns['base_check']=lambda self, require_listener:True
    for alive in (True,False):
        obj=SimpleNamespace(monitor=SimpleNamespace(is_alive=lambda:alive))
        try: ns['check_server'](obj)
        except AssertionError: assert not alive
        else: assert alive
    print(f'AST; {pos} positive/{neg} negative log cases; shutdown {stop_positive} clean/{stop_negative} failure cases PASS; GPU/runtime UNRUN')

if len(sys.argv) > 1:
    job=Path(sys.argv[1]).read_text()
    assert '-o FLASH_ATTN_EXT_SJKVARN_TRELLIS --output csv' in job
    print('Frozen I16 operation selection PASS')
