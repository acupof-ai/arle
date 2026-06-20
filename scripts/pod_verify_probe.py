#!/usr/bin/env python3
# Pipeline verify probe: needle x3 (correctness) + tok/s. Usage: probe.py <port>
import json, sys, time, urllib.request
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18293
BASE = f"http://127.0.0.1:{PORT}"

def mid():
    try:
        return json.load(urllib.request.urlopen(BASE + "/v1/models", timeout=10))["data"][0]["id"]
    except Exception:
        return "x"
M = mid()

def comp(p, mt, to=180):
    b = json.dumps({"model": M, "prompt": p, "max_tokens": mt, "temperature": 0}).encode()
    r = urllib.request.Request(BASE + "/v1/completions", b, {"content-type": "application/json"})
    t = time.time(); resp = json.load(urllib.request.urlopen(r, timeout=to)); dt = time.time() - t
    txt = resp["choices"][0]["text"]
    ct = resp.get("usage", {}).get("completion_tokens") or 0
    return txt, ct, dt

# ready-retry INSIDE the probe (no fragile nested-shell curl) — up to ~120s
ready = False
for _ in range(40):
    try:
        comp("Hi", 1, to=10); ready = True; break
    except Exception:
        time.sleep(3)
if not ready:
    print("NOT_READY", flush=True); sys.exit(4)
print("READY", flush=True)
# NEEDLE x3 — planted fact in a ~1.5k-token context (csa_select must retrieve it)
secret = "ZQ7K9X"
filler = "The regional weather report notes mild conditions and light wind. " * 120
ctx = filler[:len(filler)//2] + f" CRITICAL FACT: the vault access code is {secret}. " + filler[len(filler)//2:]
ok = 0
for _ in range(3):
    txt, _, _ = comp(ctx + "\nQ: What is the vault access code?\nA: The vault access code is", 10)
    if secret in txt:
        ok += 1
print(f"NEEDLE {ok}/3 secret={secret}", flush=True)
# tok/s — predictable decode (ms/step anchor)
_, ct, dt = comp("Count from one to three hundred, one integer per line:\n1\n2\n3\n4", 256)
print(f"TOKS {ct/max(dt,1e-3):.2f} ct={ct} dt={dt:.2f} ms_per_step={1000*dt/max(ct,1):.2f}", flush=True)
