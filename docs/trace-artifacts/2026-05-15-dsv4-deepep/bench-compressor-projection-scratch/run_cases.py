import http.client, json, time, sys
port = int(sys.argv[1]); out_path = sys.argv[2]
cases = [
    ("warmup16", "请用中文简短说明彩虹为什么有多种颜色。", 16),
    ("decode64", "Write a concise paragraph about why GPU communication overhead matters for mixture-of-experts inference.", 64),
    ("math", "Calculate 123 + 287. Answer with only the number.", 16),
]
def iter_sse(resp):
    decoder = json.JSONDecoder(); buf = ""
    while True:
        line = resp.readline()
        if not line:
            break
        s = line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not s.startswith("data: "):
            continue
        data = s[6:]
        if data.strip() == "[DONE]":
            break
        buf += data
        while buf:
            try:
                obj, end = decoder.raw_decode(buf)
            except json.JSONDecodeError:
                break
            yield obj
            buf = buf[end:].lstrip()
results = {}
for name, prompt, max_tokens in cases:
    body = json.dumps({"model":"DeepSeek-V4-Flash","messages":[{"role":"user","content":prompt}],"max_tokens":max_tokens,"temperature":0,"stream":True}, ensure_ascii=False).encode("utf-8")
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    t0 = time.perf_counter(); conn.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type":"application/json"})
    resp = conn.getresponse(); headers_t = time.perf_counter()
    text=[]; chunks=0; content_chunks=0; first_content=None; last_obj=None
    if resp.status == 200:
        for obj in iter_sse(resp):
            last_obj = obj; chunks += 1
            delta = obj.get("choices", [{}])[0].get("delta", {})
            piece = delta.get("content")
            if piece is not None:
                if first_content is None:
                    first_content = time.perf_counter()
                text.append(piece); content_chunks += 1
        t1 = time.perf_counter(); s="".join(text)
        ttft = None if first_content is None else first_content - t0
        post_first = None
        if first_content is not None and t1 > first_content:
            post_first = max(0, content_chunks - 1) / (t1 - first_content)
        e2e = max_tokens / (t1 - t0) if t1 > t0 else None
        results[name] = {"status": resp.status, "headers_s": headers_t - t0, "elapsed_s": t1 - t0, "ttft_s": ttft, "chunks": chunks, "content_chunks": content_chunks, "post_first_content_chunks_per_s": post_first, "requested_tokens_per_s_e2e": e2e, "text": s, "chars": len(s), "last_obj": last_obj}
    else:
        raw = resp.read().decode("utf-8", errors="replace")
        t1 = time.perf_counter()
        results[name] = {"status": resp.status, "elapsed_s": t1 - t0, "raw": raw[:4096]}
    print(name, json.dumps(results[name], ensure_ascii=False), flush=True)
open(out_path, "w").write(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
