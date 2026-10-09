import json, os, signal, subprocess, sys, time, urllib.request
N=os.path.dirname(os.path.abspath(__file__)); PORT=8303
S="/home/jake-k/TheTom-llama-cpp-turboquant-qwen38-merge/build-sm86/bin/llama-server"; M="/home/jake-k/qwen38-bench/models/atx4xs/Qwen3.8-27B-ATX-4-XS.gguf"
def start(extra, log):
    cmd=[S,"-m",M,"--host","127.0.0.1","--port",str(PORT),"-t","8","-tb","8","-c","16384","-b","4096","-ub","1024","-ngl","99","-fa","on","-ctk","q8_0","-ctv","turbo3",
         "--parallel","1","--jinja","--no-webui","--fit","off","--cache-prompt","--ctx-checkpoints","2","--checkpoint-min-step","512","--verbose",
         "--spec-type","draft-mtp","--spec-draft-n-max","3","--spec-draft-p-min","0.45","--spec-draft-type-k","q8_0","--spec-draft-type-v","turbo3"]+extra
    env=dict(os.environ); env["GGML_Q8_TURBO3_MMA_FUSED"]="1"
    p=subprocess.Popen(cmd,stdout=open(log,"w"),stderr=subprocess.STDOUT,env=env,start_new_session=True)
    for _ in range(300):
        try:
            if json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health",timeout=2).read()).get("status")=="ok": return p
        except Exception: time.sleep(1)
    raise SystemExit("server not ready")
def stop(p): os.killpg(p.pid,signal.SIGTERM); p.wait(timeout=60)
def chat(msgs):
    body={"messages":msgs,"max_tokens":48,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}
    r=json.loads(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}),timeout=600).read())
    t=r.get("timings",{}); return r["choices"][0]["message"]["content"], t.get("prompt_n"), t.get("cache_n")
def conv(seed):
    filler=("Reference note %d: the quick brown fox jumps over the lazy dog near the riverbank while the farmer counts sheep. "%seed)*140
    return [{"role":"user","content":filler+"\n\nIn one sentence, what animal jumps in note %d?"%seed}]
A,B,C=conv(1),conv(2),conv(3)
mode=sys.argv[1]; CRAM=sys.argv[2] if len(sys.argv)>2 else "400"
extra=["--cache-ram",CRAM]+(["--cache-disk-path",f"{N}/pcdisk","--cache-disk-limit","4096"] if mode=="disk" else [])
p=start(extra,f"{N}/disk_tier_{mode}_{CRAM}.log")
try:
    a1,pn,cn=chat(A); print("A1 prompt_n",pn,"cache_n",cn); A=A+[{"role":"assistant","content":a1},{"role":"user","content":"Now say it again in five words."}]
    b1,pn,cn=chat(B); print("B1 prompt_n",pn,"cache_n",cn)
    c1,pn,cn=chat(C); print("C1 prompt_n",pn,"cache_n",cn)
    a2,pn,cn=chat(A); print("A2 (return to A) prompt_n",pn,"cache_n",cn,"->",repr(a2[:80]))
    json.dump({"a2":a2,"A":A},open(f"{N}/disk_tier_{mode}_A2.json","w"))
finally: stop(p)
if mode=="disk":
    print("--- restart: index should list spilled entries; B should restore from disk")
    p=start(extra,f"{N}/disk_tier_restart_{CRAM}.log")
    try:
        b2,pn,cn=chat(B+[{"role":"assistant","content":b1},{"role":"user","content":"Now say it again in five words."}]); print("B2 after restart prompt_n",pn,"cache_n",cn,"->",repr(b2[:80]))
    finally: stop(p)
    print("--- control: A2 with no interruption")
    p=start(["--cache-ram","0"],f"{N}/disk_tier_control.log")
    try:
        chat(A[:1]); a2c,pn,cn=chat(A); print("A2 control prompt_n",pn,"cache_n",cn,"->",repr(a2c[:80])); print("MATCH" if a2c==a2 else "MISMATCH")
    finally: stop(p)
