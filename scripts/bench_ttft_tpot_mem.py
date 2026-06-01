import json,os,signal,subprocess,sys,time,urllib.request,threading
import psutil
MODEL="mlx-community/Qwen3.6-35B-A3B-4bit"
ARLE_BIN="target/release/metal_serve"
LENS=[128,256,512,1024,2048,4096,8192,12288]
OUT_TOKENS=int(os.environ.get("OUT_TOKENS","256"))  # steady TPOT is rate, 256 enough; set 4096 for full
FLOOR_GB=1.2
RES="/tmp/bench_ttft_tpot_mem.json"
log=open("/tmp/bench_ttft_tpot_mem.log","a",buffering=1)
def L(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",file=log,flush=True); print(m,flush=True)
def avail(): return psutil.virtual_memory().available/1024**3
def used():  return psutil.virtual_memory().used/1024**3
st={"die":False,"proc":None,"sample":False,"peak_rss":0.0,"peak_used":0.0}
def watchdog():
    while not st["die"]:
        if avail()<FLOOR_GB:
            st["die"]=True; L("!! WATCHDOG kill")
            p=st["proc"]
            if p:
                try: os.killpg(os.getpgid(p.pid),signal.SIGKILL)
                except: pass
            return
        if st["sample"]:
            p=st["proc"]
            if p:
                try:
                    pr=psutil.Process(p.pid)
                    rss=pr.memory_info().rss
                    for c in pr.children(recursive=True):
                        try: rss+=c.memory_info().rss
                        except: pass
                    st["peak_rss"]=max(st["peak_rss"],rss/1024**3)
                except: pass
            st["peak_used"]=max(st["peak_used"],used())
        time.sleep(0.5)
def make(n): return f"doc {n} "+("word "*n)+"\nContinue:"
def wait_ready(port,timeout=200):
    t0=time.time()
    while time.time()-t0<timeout:
        if st["die"]: return False
        try: urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models",timeout=3); return True
        except: pass
        p=st["proc"]
        if p and p.poll() is not None: return False
        time.sleep(2)
    return False
def probe(port,n):
    body=json.dumps({"model":MODEL,"prompt":make(n),"max_tokens":OUT_TOKENS,"temperature":0,"stream":True}).encode()
    req=urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",data=body,headers={"Content-Type":"application/json"})
    t0=time.time(); ts=[]
    with urllib.request.urlopen(req,timeout=600) as r:
        for raw in r:
            s=raw.decode("utf-8","ignore").strip()
            if s.startswith("data:") and s[5:].strip()!="[DONE]":
                try: ch=json.loads(s[5:].strip())["choices"][0]
                except: continue
                if ch.get("text"): ts.append(time.time()-t0)
    if len(ts)<3: return None
    ttft=ts[0]
    iv=[(ts[i]-ts[i-1])*1000 for i in range(1,len(ts))]
    rest=iv[1:]                          # steady TPOT: drop token1->2 (prefill tail)
    tpot=sum(rest)/len(rest) if rest else None
    return ttft,tpot,len(ts)
def run(name,launch,port):
    L(f"--- {name}: launch (avail={avail():.1f}GB) ---")
    st["peak_rss"]=0.0; st["peak_used"]=0.0
    base_used=used()
    p=launch(); st["proc"]=p
    res={}
    try:
        if not wait_ready(port): L(f"{name}: not ready"); return res
        L(f"{name}: ready avail={avail():.1f}GB load_used_delta={used()-base_used:.1f}GB")
        probe(port,128)  # warmup
        for n in LENS:
            if st["die"]: break
            st["peak_rss"]=0.0; st["peak_used"]=0.0; st["sample"]=True
            r=probe(port,n)
            st["sample"]=False
            if not r: L(f"{name} n={n}: <3 tok"); continue
            ttft,tpot,nt=r
            res[n]={"ttft_s":ttft,"tpot_ms":tpot,"peak_rss_gb":round(st["peak_rss"],2),"peak_used_gb":round(st["peak_used"],2),"ntok":nt}
            L(f"{name} n={n:6d}: TTFT={ttft:.3f}s TPOT={tpot:.1f}ms peak_rss={st['peak_rss']:.1f}GB sys_used={st['peak_used']:.1f}GB")
    finally:
        st["sample"]=False
        if p.poll() is None:
            try: os.killpg(os.getpgid(p.pid),signal.SIGTERM)
            except: pass
            for _ in range(25):
                if p.poll() is not None: break
                time.sleep(1)
            if p.poll() is None:
                try: os.killpg(os.getpgid(p.pid),signal.SIGKILL)
                except: pass
        st["proc"]=None
        for _ in range(40):
            if avail()>24: break
            time.sleep(2)
        L(f"{name}: stopped avail={avail():.1f}GB")
    return res
L("="*56); L(f"BENCH TTFT/TPOT/MEM out_tokens={OUT_TOKENS} avail={avail():.1f}GB")
threading.Thread(target=watchdog,daemon=True).start()
out={"model":MODEL,"lens":LENS,"out_tokens":OUT_TOKENS,"arle":{},"mlx":{}}
def arle_launch():
    return subprocess.Popen([ARLE_BIN,"--model-path",MODEL,"--port","8881","--max-running-requests","1","--max-batch-tokens","4096"],
        stdout=open("/tmp/bench_arle.log","w"),stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,RUST_LOG="warn"))
out["arle"]=run("ARLE",arle_launch,8881)
def mlx_launch():
    return subprocess.Popen([sys.executable,"-m","mlx_lm","server","--model",MODEL,"--port","8882","--host","127.0.0.1"],
        stdout=open("/tmp/bench_mlx.log","w"),stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ))
if not st["die"]:
    out["mlx"]=run("mlx-lm",mlx_launch,8882)
st["die"]=True
json.dump(out,open(RES,"w"),indent=2)
L(f"DONE -> {RES}")
