import json
import os
import sys
import time
import urllib.request

port = int(sys.argv[1])
out = sys.argv[2]
max_tokens = int(sys.argv[3])
model = os.environ.get("MODEL_NAME", "DeepSeek-V4-Flash")
prompt = os.environ["PROMPT"]
payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": max_tokens,
    "temperature": 0,
    "stream": False,
}
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
t0 = time.perf_counter()
with urllib.request.urlopen(req, timeout=240) as resp:
    body = resp.read()
elapsed = time.perf_counter() - t0
parsed = json.loads(body)
result = {
    "status": 200,
    "elapsed_s": elapsed,
    "usage": parsed.get("usage"),
    "text": parsed["choices"][0]["message"]["content"],
}
with open(out, "w", encoding="utf-8") as f:
    f.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False))
