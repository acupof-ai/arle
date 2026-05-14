import http.client, json, sys, time
port=int(sys.argv[1]); label=sys.argv[2]

def request(prompt, max_tokens, ignore_eos=False):
    payload=json.dumps({
        "model":"DeepSeek-V4-Flash",
        "messages":[{"role":"user","content":prompt}],
        "max_tokens":max_tokens,
        "temperature":0,
        "ignore_eos":ignore_eos,
        "stream":True,
        "stream_options":{"include_usage":True},
    }, ensure_ascii=False).encode()
    conn=http.client.HTTPConnection("127.0.0.1", port, timeout=1200)
    start=time.time(); first=None; output=[]; usage=None; status=None; err=None
    try:
        conn.request("POST","/v1/chat/completions",body=payload,headers={"Content-Type":"application/json"})
        resp=conn.getresponse(); status=resp.status
        while True:
            line=resp.readline()
            if not line: break
            text=line.decode("utf-8","replace").strip()
            if not text or text.startswith(":") or not text.startswith("data: "): continue
            data=text[6:]
            if data=="[DONE]": break
            chunk=json.loads(data)
            if chunk.get("usage"): usage=chunk["usage"]
            for choice in chunk.get("choices",[]):
                content=(choice.get("delta") or {}).get("content") or ""
                if content:
                    if first is None: first=time.time()
                    output.append(content)
    except Exception as exc:
        err=repr(exc)
    finally:
        conn.close()
    total=time.time()-start
    ttft=None if first is None else first-start
    completion=(usage or {}).get("completion_tokens")
    decode_window=None if ttft is None else max(total-ttft,0)
    return {
        "label": label,
        "status": status,
        "error": err,
        "ttft_s": None if ttft is None else round(ttft,4),
        "total_s": round(total,4),
        "usage": usage,
        "post_first_decode_tok_s": round((completion-1)/decode_window,2) if completion and completion>1 and decode_window and decode_window>0 else None,
        "output": "".join(output)[:500],
    }

cases=[]
cases.append(("warmup16", "You are benchmarking decoding speed. Continue with short plain English words separated by spaces. Seed words: alpha beta.", 16, True))
cases.append(("decode64", "You are benchmarking decoding speed. Continue with short plain English words separated by spaces. Keep going until the token limit. Seed words: alpha beta gamma delta.", 64, True))
cases.append(("math", "Calculate 17 * 23 + 19. Return only the final integer.", 16, False))
results=[]
for name,prompt,max_tokens,ignore in cases:
    label=name
    results.append(request(prompt,max_tokens,ignore))
print(json.dumps({"case":sys.argv[2],"results":results}, ensure_ascii=False, indent=2))
