import json, time, requests, sys
BASE="http://127.0.0.1:18161/v1/chat/completions"
CASES=[("warmup16","请用中文简短说明彩虹为什么有多种颜色。",16),("decode64","Write a concise paragraph about why GPU communication overhead matters for mixture-of-experts inference.",64),("math","Calculate 123 + 287. Answer with only the number.",16)]
def parse_data_chunks(resp):
    decoder=json.JSONDecoder(); buf=""; resp.encoding="utf-8"
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data: "): continue
        data=raw[6:]
        if data.strip()=="[DONE]": break
        buf += data
        while buf:
            try: obj,end=decoder.raw_decode(buf)
            except json.JSONDecodeError: break
            yield obj; buf=buf[end:].lstrip()
out={}
for name,prompt,max_tokens in CASES:
    payload={"model":"DeepSeek-V4-Flash","messages":[{"role":"user","content":prompt}],"max_tokens":max_tokens,"temperature":0,"stream":True}
    t0=time.time(); first=None; text=[]; chunks=0; status=None
    with requests.post(BASE,json=payload,stream=True,timeout=180) as r:
        status=r.status_code; r.raise_for_status()
        for obj in parse_data_chunks(r):
            chunks+=1; delta=obj.get("choices",[{}])[0].get("delta",{})
            if "content" in delta:
                if first is None: first=time.time()
                text.append(delta["content"])
    t1=time.time(); s="".join(text)
    out[name]={"status":status,"elapsed_s":t1-t0,"ttft_s":None if first is None else first-t0,"chunks":chunks,"text":s,"chars":len(s)}
    print(name, json.dumps(out[name], ensure_ascii=False), flush=True)
open(sys.argv[1],"w").write(json.dumps(out,ensure_ascii=False,indent=2))
