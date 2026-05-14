import http.client
import json
import sys
import time

port = int(sys.argv[1])
label = sys.argv[2]
max_tokens = int(sys.argv[3])
prompt = sys.argv[4] if len(sys.argv) > 4 else "用两个字形容彩虹。"
body = json.dumps({
    "model": "/root/DeepSeek-V4-Flash",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": max_tokens,
    "temperature": 0,
    "stream": False,
}, ensure_ascii=False).encode("utf-8")
conn = http.client.HTTPConnection("127.0.0.1", port, timeout=180)
start = time.time()
conn.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
resp = conn.getresponse()
raw = resp.read()
elapsed = time.time() - start
text = raw.decode("utf-8", errors="replace")
try:
    data = json.loads(text)
except Exception:
    data = {"raw": text[:4096]}
output = ""
try:
    output = data["choices"][0]["message"]["content"]
except Exception:
    pass
print(json.dumps({
    "label": label,
    "status": resp.status,
    "max_tokens": max_tokens,
    "elapsed_s": elapsed,
    "output": output,
    "usage": data.get("usage"),
    "response": data if resp.status != 200 else None,
}, ensure_ascii=False, indent=2))
if resp.status != 200:
    sys.exit(1)
