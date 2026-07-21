#!/usr/bin/env python3
import json
import urllib.request
import sys

paragraph = (
    "The quick brown fox jumps over the lazy dog. This sentence is used as a standard "
    "test for typing and font displays. It contains every letter of the alphabet at "
    "least once. The fox is quick and brown, while the dog is lazy and slow. They "
    "coexist in a field where the fox hunts and the dog rests. "
)
prompt = (paragraph * 60) + "\n\nPlease summarize the text above in one sentence."

payload = {
    "model": "default",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 50,
    "temperature": 0.0,
}

port = sys.argv[1] if len(sys.argv) > 1 else "8080"
url = f"http://127.0.0.1:{port}/v1/chat/completions"

print(f"prompt chars: {len(prompt)}")
print(f"estimated tokens: ~{len(prompt) // 4}")

req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read())
        print("STATUS: OK")
        print("response:", json.dumps(body, indent=2)[:2000])
except urllib.error.HTTPError as e:
    print(f"STATUS: HTTP {e.code}")
    print(e.read().decode()[:2000])
    sys.exit(1)
except Exception as e:
    print(f"STATUS: ERROR {type(e).__name__}: {e}")
    sys.exit(1)
