import http.client, json, sys, time
port=int(sys.argv[1]); out_path=sys.argv[2]
cases=[
    ("warmup16", "请用中文简短说明彩虹为什么有多种颜色。", 16),
    ("decode64", "Write a concise paragraph about why GPU communication overhead matters for mixture-of-experts inference.", 64),
    ("math", "Calculate 123 + 287. Answer with only the number.", 16),
]
def iter_sse_chunks(raw):
    decoder=json.JSONDecoder(); buf=""
    for line in raw.splitlines():
        if not line.startswith("data: "): continue
        data=line[6:]
        if data.strip()=="[DONE]": break
        buf += data
        while buf:
            try:
                obj,end=decoder.raw_decode(buf)
            except json.JSONDecodeError:
                break
            yield obj
            buf=buf[end:].lstrip()
results={}
for name,prompt,max_tokens in cases:
    body=json.dumps({"model":"DeepSeek-V4-Flash","messages":[{"role":"user","content":prompt}],"max_tokens":max_tokens,"temperature":0,"stream":True}, ensure_ascii=False).encode("utf-8")
    conn=http.client.HTTPConnection("127.0.0.1", port, timeout=240)
    t0=time.time(); conn.request("POST","/v1/chat/completions",body=body,headers={"Content-Type":"application/json"})
    resp=conn.getresponse(); raw=resp.read().decode("utf-8", errors="replace"); t1=time.time()
    if resp.status != 200:
        results[name]={"status":resp.status,"elapsed_s":t1-t0,"raw":raw[:4096]}
        print(name, json.dumps(results[name], ensure_ascii=False), flush=True)
        continue
    first=None; text=[]; chunks=0; content_chunks=0
    for obj in iter_sse_chunks(raw):
        chunks += 1
        delta=obj.get("choices", [{}])[0].get("delta", {})
        if "content" in delta:
            if first is None: first=t0  # full body was read already; TTFT unavailable with stdlib buffered read
            piece=delta["content"]
            text.append(piece); content_chunks += 1
    s="".join(text)
    post_first_tok_s=None
    # This buffered stdlib path cannot measure TTFT; report e2e decode throughput from requested max_tokens.
    if t1 > t0:
        post_first_tok_s=max_tokens/(t1-t0)
    results[name]={"status":resp.status,"elapsed_s":t1-t0,"ttft_s":None,"chunks":chunks,"content_chunks":content_chunks,"post_first_tok_s_estimate":post_first_tok_s,"text":s,"chars":len(s)}
    print(name, json.dumps(results[name], ensure_ascii=False), flush=True)
open(out_path,"w").write(json.dumps(results, ensure_ascii=False, indent=2)+"\n")
