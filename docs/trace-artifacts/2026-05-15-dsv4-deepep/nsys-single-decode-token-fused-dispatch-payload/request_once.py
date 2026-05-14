import http.client, json, sys, time
port=int(sys.argv[1]); label=sys.argv[2]; max_tokens=int(sys.argv[3])
prompt="用两个字形容彩虹。"
payload=json.dumps({"model":"DeepSeek-V4-Flash","messages":[{"role":"user","content":prompt}],"max_tokens":max_tokens,"temperature":0,"stream":True,"stream_options":{"include_usage":True}}, ensure_ascii=False).encode()
conn=http.client.HTTPConnection("127.0.0.1", port, timeout=1200)
start=time.time(); first=None; out=[]; usage=None; status=None
conn.request("POST","/v1/chat/completions",body=payload,headers={"Content-Type":"application/json"})
resp=conn.getresponse(); status=resp.status
decoder=json.JSONDecoder(); buf=""
while True:
    line=resp.readline()
    if not line: break
    text=line.decode("utf-8","replace").strip()
    if not text or text.startswith(":") or not text.startswith("data: "): continue
    data=text[6:]
    if data=="[DONE]": break
    buf += data
    while buf:
        try: chunk,end=decoder.raw_decode(buf)
        except json.JSONDecodeError: break
        buf=buf[end:].lstrip()
        if chunk.get("usage"): usage=chunk["usage"]
        for choice in chunk.get("choices",[]):
            content=(choice.get("delta") or {}).get("content") or ""
            if content:
                if first is None: first=time.time()
                out.append(content)
conn.close()
print(json.dumps({"label":label,"status":status,"max_tokens":max_tokens,"elapsed_s":time.time()-start,"ttft_s":None if first is None else first-start,"output":"".join(out),"usage":usage}, ensure_ascii=False, indent=2))
