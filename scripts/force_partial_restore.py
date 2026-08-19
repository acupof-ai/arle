#!/usr/bin/env python3
"""Force PARTIAL prefix restores (prefix-attach: restored < matched) on demand.

Mechanism (all file:line in crates/):
  infer-cuda/src/executor.rs:54     SIDECAR_SNAPSHOT_STRIDE_PAGES = 512
  infer-cuda/src/executor.rs:43     SUPPORTED_PAGE_SIZE = 16   -> stride = 8192 tokens
  infer-cuda/src/executor/qwen35.rs:363-384  restore probes [matched_len] then
                                             8192*k descending; first hit wins
  infer-cuda/src/executor/qwen35.rs:252-276  a finished request publishes sidecars at
                                             {8192*k < prompt_len} U {align16(prompt_len)}
                                             U {align16(prompt_len+generated)}
  infer-core/src/lib.rs:1264-1269   prompt-boundary publish (this is why "resend the whole
                                    prompt + a suffix" ALWAYS restores fully)
  infer-core/src/prefix.rs:115-124  attach_cap = len-1, then the last-block pop
  infer-core/src/planner.rs:83-107  tail chunking: 2048 then lcm(page,restore_align)-aligned
                                    remainder; restore_align == 1 for Qwen35 (executor.rs:628)
  infer-cuda/src/ops/quant_linear.rs:750,754 FP4 Marlin claims M <= QWEN_MARLIN_MAX_M

Run:  python3 force_partial_restore.py                 # full sweep, sequential
      python3 force_partial_restore.py --only A        # one case
      python3 force_partial_restore.py --repeat 3
Watch: grep -E 'prefix-attach|\[marlin\]|engine step failed' <server.log>
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

ENDPOINT = "http://127.0.0.1:18191/v1/completions"

# --- constants mirrored from the tree; change only if the server was started with
# --- a non-default --chunked-prefill-size.
PAGE = 16                 # SUPPORTED_PAGE_SIZE (infer-cuda/src/executor.rs:43)
STRIDE = 512 * PAGE       # 8192 (infer-cuda/src/executor.rs:54)
CHUNK = 2048              # chunked_prefill_size
MARLIN_MAX_M = 1024       # QWEN_MARLIN_MAX_M (quant_linear.rs:750)

# (name, publisher prompt len, divergence position, trigger prompt len)
# X = 31860 -> matched = align16(31860) = 31856, restored = 8192*3 = 24576.
# L_B controls the tail: M_a = align16(L_B) - 32768, M_b = L_B % 16.
CASES = [
    # exact shape of the two observed crashes: 4x2048 + 288 + 3
    ("A", 33059, 31860, 33059),
    # bigger Marlin tail, 8-row hits the m_block_size_8 kernel family
    ("B", 33059, 31860, 33768),
    # tail at the Marlin ceiling; L*-cut splits 1024 into 1008 + 16
    ("C", 33059, 31860, 33792),
    # last marlin sub-chunk 48 -> thread_m_blocks = 3
    ("D", 33059, 31860, 33077),
    # tail is exactly one marlin sub-chunk of 64 -> thread_m_blocks = 4
    ("E", 33059, 31860, 32840),
    # partial restore whose big tail row is NOT Marlin (1488 > 1024): separates
    # "multi-chunk prefill" from "Marlin tail" as the crash discriminator
    ("F", 33059, 31860, 34268),
    # control: shares the WHOLE publisher prompt -> matched lands exactly on the
    # prompt-boundary sidecar -> FULL restore. This is what the previous attempt
    # was doing, and why it produced 0 partial restores.
    ("CTRL", 33059, 33059, 34258),
]


# ---------------------------------------------------------------- token streams
def stream(seed, n):
    """Deterministic token ids in [1000, 90000) - ordinary Qwen BPE pieces, no
    special tokens (Qwen specials are >= 151643)."""
    out = []
    x = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    for _ in range(n):
        x = (x * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        out.append(1000 + ((x >> 33) % 89000))
    return out


# ---------------------------------------------------------------- prediction
def a16(x):
    return x // PAGE * PAGE


def sidecar_positions(prompt_len, generated):
    pos = {k * STRIDE for k in range(1, prompt_len // STRIDE + 1) if k * STRIDE < prompt_len}
    pos.add(a16(prompt_len))
    pos.add(a16(prompt_len + generated))
    return pos


def predict(p_a, x, l_b, generated=1):
    side = sidecar_positions(p_a, generated)
    matched = a16(x)
    if matched > l_b - 1:                       # prefix.rs:115-124 last-block trim
        matched -= PAGE
    if matched in side:
        restored = matched
    else:
        hits = [b for b in range(STRIDE, matched + 1, STRIDE) if b in side]
        restored = max(hits) if hits else 0
    rows, s = [], restored
    while s < l_b:                              # planner.rs:83-107
        c = min(l_b - s, CHUNK)
        aligned = (s + c) - ((s + c) % PAGE)
        if aligned > s:
            c = aligned - s
        rows.append((s, s + c))
        s += c
    lstar = a16(l_b - 1)
    fw = []
    for s, e in rows:                           # qwen35.rs:2497-2553
        cuts = []
        if s > 0 and s % STRIDE == 0:
            cuts.append(s)
        t = (s // STRIDE + 1) * STRIDE
        while t < e:
            cuts.append(t)
            t += STRIDE
        if e >= l_b and lstar > 0 and s <= lstar < e and lstar not in cuts:
            cuts.append(lstar)
        cuts.sort()
        cur = s
        for c in cuts:
            if c > cur:
                fw.append(c - cur)
                cur = c
        fw.append(e - cur)
    return {
        "sidecars": sorted(side),
        "matched": matched,
        "restored": restored,
        "partial": restored < matched,
        "remainder": l_b - restored,
        "forward_m": fw,
        "marlin_m": [m for m in fw if 0 < m <= MARLIN_MAX_M],
    }


# ---------------------------------------------------------------- http
def post(ids, max_tokens=1, timeout=1800):
    body = json.dumps({
        "prompt": ids,                # token-id array -> fed verbatim (schema.rs:105-115)
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 0,
        "ignore_eos": True,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        ENDPOINT, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read()[:400].decode('utf8', 'replace')}",
                "wall": time.time() - t0}
    except Exception as e:                                    # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}", "wall": time.time() - t0}
    usage = payload.get("usage") or {}
    return {"prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "wall": time.time() - t0}


# ---------------------------------------------------------------- driver
def run_case(name, p_a, x, l_b, seed):
    pred = predict(p_a, x, l_b)
    print(f"\n=== case {name}  P_A={p_a} X={x} L_B={l_b} (seed {seed})")
    print(f"    predict: matched={pred['matched']} restored={pred['restored']} "
          f"partial={pred['partial']} remainder={pred['remainder']}")
    print(f"    predict: forward M={pred['forward_m']}  marlin M={pred['marlin_m']}")

    base = stream(seed, p_a)
    alt = stream(seed + 777_777, l_b)
    assert x <= p_a and x < l_b, "divergence must be inside both prompts"
    assert base[x] != alt[x], "streams collide at the divergence token; bump the seed"
    trigger = base[:x] + alt[x:l_b]

    r1 = post(base)
    print(f"    publisher : {r1}")
    if r1.get("error"):
        return False
    if r1.get("prompt_tokens") != p_a:
        print(f"    WARN: server reports prompt_tokens={r1.get('prompt_tokens')} != {p_a}")

    r2 = post(trigger)
    print(f"    trigger   : {r2}")
    if r2.get("error"):
        print("    ^ if this is the crash, the server log holds the [marlin] line")
        return False
    return True


def main():
    global ENDPOINT
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="run one case by name (A..F, CTRL)")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--endpoint", default=ENDPOINT)
    args = ap.parse_args()
    ENDPOINT = args.endpoint

    cases = [c for c in CASES if args.only is None or c[0] == args.only]
    if not cases:
        sys.exit(f"no case named {args.only}")
    print(f"endpoint {ENDPOINT}  stride={STRIDE} chunk={CHUNK} page={PAGE} "
          f"marlin_max_m={MARLIN_MAX_M}")
    print("expect one 'prefix-attach: ... matched=31856 restored=24576' per non-CTRL case")
    seed = 20260819
    for rep in range(args.repeat):
        for name, p_a, x, l_b in cases:
            if not run_case(name, p_a, x, l_b, seed):
                print(f"\nSTOPPED at case {name} rep {rep}")
                return 1
            seed += 1          # fresh content per case: no cross-case radix reuse
    print("\nall cases completed without a server error")
    return 0


if __name__ == "__main__":
    sys.exit(main())
