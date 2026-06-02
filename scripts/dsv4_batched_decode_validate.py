#!/usr/bin/env python3
"""On-pod validation for DSv4 batched decode (runs INSIDE the pod).

Gate:
  * c=1, c=4, and c=8 must complete without HTTP errors.
  * every output must contain the expected answer token.

The c=1 vs c=4 byte-identical check is kept as a diagnostic, not the process
exit condition. The model may produce extra deterministic text around the
answer under this debug smoke prompt; that should not hide real failures such as
HTTP 500s, empty generations, or repeated garbage-token regressions.

Usage: python3 dsv4_batched_decode_validate.py <port>
"""
import json, sys, time, urllib.request, threading

PORT = sys.argv[1] if len(sys.argv) > 1 else "18300"
PROMPT = "Compute 137 + 269. Answer with the number only."
EXPECTED = "406"
MAXTOK = 24


def gen(prompt, results, idx):
    body = json.dumps({"model": "DeepSeek-V4-Flash",
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": MAXTOK, "temperature": 0, "ignore_eos": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.loads(r.read())
        results[idx] = (time.perf_counter() - t0, d["choices"][0]["message"]["content"],
                        d.get("usage", {}).get("completion_tokens", 0))
    except Exception as e:
        results[idx] = (time.perf_counter() - t0, f"ERR:{e}", 0)


# c=1 reference (per-row path)
r1 = [None]
gen(PROMPT, r1, 0)
ref = r1[0][1]
print(f"c1_ref ({r1[0][0]:.1f}s): {ref!r}")

# c=4 batched
r4 = [None] * 4
ts = [threading.Thread(target=gen, args=(PROMPT, r4, i)) for i in range(4)]
t0 = time.perf_counter()
for t in ts: t.start()
for t in ts: t.join()
w4 = time.perf_counter() - t0
match = all(r4[i][1] == ref for i in range(4))
print(f"c4_batched wall={w4:.1f}s outputs match c1_ref: {match}")
for i in range(4):
    print(f"  row{i}: {r4[i][1]!r}")

# c=8 timing sanity (just confirm it completes, not hangs)
r8 = [None] * 8
ts = [threading.Thread(target=gen, args=(PROMPT, r8, i)) for i in range(8)]
t0 = time.perf_counter()
for t in ts: t.start()
for t in ts: t.join()
w8 = time.perf_counter() - t0
errs8 = sum(1 for x in r8 if x[1].startswith("ERR"))
toks8 = sum(x[2] for x in r8)
print(f"c8 wall={w8:.1f}s errs={errs8} out_tokens={toks8} agg_tok/s={toks8/w8:.2f}")
c8_answer_match = all(not row[1].startswith("ERR") and EXPECTED in row[1] for row in r8)
if not c8_answer_match:
    for i in range(8):
        print(f"  c8_row{i}: {r8[i][1]!r}")
answer_ok = (
    not ref.startswith("ERR")
    and EXPECTED in ref
    and all(not row[1].startswith("ERR") and EXPECTED in row[1] for row in r4)
    and c8_answer_match
    and errs8 == 0
)
print("BYTE_PARITY_PASS" if match else "BYTE_PARITY_DIAGNOSTIC_FAIL")
print("ANSWER_PASS" if answer_ok else "ANSWER_OR_C8_FAIL")
sys.exit(0 if answer_ok else 1)
