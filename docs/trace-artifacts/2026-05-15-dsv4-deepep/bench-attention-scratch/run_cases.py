import json, sys, time, urllib.request
port = int(sys.argv[1])
out = sys.argv[2]
cases = [
  ("warmup16", "用中文简短解释彩虹为什么有多种颜色。", 16, False),
  ("decode64", "You are benchmarking decoding speed. Continue with short plain English words separated by spaces. Keep going until the token limit. Seed words: alpha beta gamma delta.", 64, True),
  ("math", "Calculate 17 * 23 + 19. Return only the final integer.", 32, False),
  ("writing", "用一句中文写长上下文推理性能优化发布说明，提到吞吐、流式输出和 trace。", 24, False),
]

def run(label, prompt, max_tokens, ignore_eos):
    payload = {
        "model": "DeepSeek-V4-Flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": ignore_eos,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter(); first = None; chunks = 0; content_chunks = 0; parts = []; usage = None; err = None; status = None
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            status = resp.status
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line or line.startswith(":") or not line.startswith("data: "):
                    continue
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                chunks += 1
                item = json.loads(data)
                if item.get("usage"):
                    usage = item["usage"]
                for choice in item.get("choices", []):
                    text = (choice.get("delta") or {}).get("content") or ""
                    if text:
                        if first is None:
                            first = time.perf_counter()
                        content_chunks += 1
                        parts.append(text)
    except Exception as exc:
        err = repr(exc)
    elapsed = time.perf_counter() - t0
    ttft = None if first is None else first - t0
    comp = (usage or {}).get("completion_tokens")
    post = None
    if comp and comp > 1 and ttft is not None and elapsed > ttft:
        post = (comp - 1) / (elapsed - ttft)
    return {
        "status": status,
        "error": err,
        "elapsed_s": elapsed,
        "ttft_s": ttft,
        "chunks": chunks,
        "content_chunks": content_chunks,
        "usage": usage,
        "post_first_tok_s": post,
        "requested_tok_s_e2e": (max_tokens / elapsed) if elapsed > 0 else None,
        "text": "".join(parts),
        "chars": len("".join(parts)),
    }
summary = {}
for case in cases:
    label = case[0]
    res = run(*case)
    summary[label] = res
    print(label, json.dumps(res, ensure_ascii=False), flush=True)
with open(out + "/summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
    f.write("\n")
