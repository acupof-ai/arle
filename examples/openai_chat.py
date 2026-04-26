#!/usr/bin/env python3
import json
import os
import urllib.request

base_url = os.environ.get("ARLE_BASE_URL", "http://127.0.0.1:8000")
payload = {
    "messages": [{"role": "user", "content": "Hello from ARLE"}],
    "max_tokens": 64,
}

request = urllib.request.Request(
    f"{base_url}/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urllib.request.urlopen(request, timeout=120) as response:
    print(response.read().decode("utf-8"))
