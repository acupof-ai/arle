#!/usr/bin/env python3
"""DSv4 c-sweep harness: concurrent serving lanes that previously crashed the
engine (mixed prefill/decode plans). Per-request correctness probe via distinct
prompts with expected substrings (cross-slot contamination check).

Usage: dsv4_c_sweep.py <arm-label> [c1,c2,...] [max_tokens] [repeats]
"""
import json
import statistics
import sys
import time
import threading
import urllib.request

PORT = 18188
BASE = f"http://127.0.0.1:{PORT}"
MODEL = "DeepSeek-V4-Flash"

# Distinct prompts: each row decodes different content; expected substring
# gates cross-slot contamination. temperature=0.
PROBES = [
    ("The capital of France is", "Paris"),
    ("The capital of Japan is", "Tokyo"),
    ("The capital of Italy is", "Rome"),
    ("The capital of Germany is", "Berlin"),
    ("The capital of Spain is", "Madrid"),
    ("The capital of Russia is", "Moscow"),
    ("The capital of Egypt is", "Cairo"),
    ("The capital of Canada is", "Ottawa"),
    ("The capital of Australia is", "Canberra"),
    ("The capital of Brazil is", "Bras"),
    ("The capital of India is", "Delhi"),
    ("The capital of China is", "Beijing"),
    ("The capital of Greece is", "Athens"),
    ("The capital of Portugal is", "Lisbon"),
    ("The capital of Austria is", "Vienna"),
    ("The capital of Norway is", "Oslo"),
]


def one_request(prompt, max_tokens, timeout=900):
    body = json.dumps({
        "model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0, "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read().decode())
        t1 = time.perf_counter()
    except Exception as e:
        return (0, time.perf_counter() - t0, "", repr(e))
    usage = payload.get("usage", {}) or {}
    ct = usage.get("completion_tokens") or 0
    text = ""
    try:
        text = payload["choices"][0].get("text", "")
    except Exception:
        pass
    return (ct, t1 - t0, text, None)


def server_alive():
    try:
        with urllib.request.urlopen(f"{BASE}/v1/models", timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


def concurrent_lane(c, max_tokens, stagger_s, repeat_idx=0):
    """c concurrent requests; stagger_s>0 staggers arrivals so later requests
    PREFILL while earlier ones DECODE — the exact mixed-plan crash shape."""
    label = f"c={c} stagger={stagger_s}s"
    print(f"=== Concurrent lane {label} repeat={repeat_idx} max_tokens={max_tokens} ===", flush=True)
    results = [None] * c

    def worker(i):
        prompt, expect = PROBES[i % len(PROBES)]
        ct, wall, text, err = one_request(prompt, max_tokens)
        ok_text = expect.lower() in text.lower()
        results[i] = {"i": i, "tok": ct, "wall": wall, "err": err,
                      "expect": expect, "expect_hit": ok_text,
                      "head": text[:60]}

    t0 = time.perf_counter()
    threads = []
    for i in range(c):
        th = threading.Thread(target=worker, args=(i,))
        th.start()
        threads.append(th)
        if stagger_s > 0 and i < c - 1:
            time.sleep(stagger_s)
    for th in threads:
        th.join()
    window = time.perf_counter() - t0

    total_tok = sum(r["tok"] for r in results if r and not r["err"])
    n_ok = sum(1 for r in results if r and not r["err"])
    n_hit = sum(1 for r in results if r and not r["err"] and r["expect_hit"])
    for r in results:
        if r is None:
            continue
        status = "ERR " + str(r["err"]) if r["err"] else (
            "HIT" if r["expect_hit"] else f"MISS(want {r['expect']})")
        rate = r["tok"] / r["wall"] if r["wall"] > 0 else 0
        print(f"  req {r['i']}: {r['tok']} tok {r['wall']:.2f}s "
              f"{rate:.2f} tok/s [{status}] {r['head']!r}", flush=True)
    agg = total_tok / window if window > 0 else 0
    alive = server_alive()
    print(f"  LANE RESULT {label}: ok={n_ok}/{c} expect_hit={n_hit}/{n_ok} "
          f"total_tok={total_tok} window={window:.2f}s agg={agg:.2f} tok/s "
          f"server_alive={alive}", flush=True)
    return {"c": c, "stagger": stagger_s, "repeat": repeat_idx, "ok": n_ok, "hit": n_hit,
            "total_tok": total_tok, "window": window, "agg_tok_s": agg,
            "alive_after": alive}


def summarize_repeats(lanes):
    grouped = {}
    for lane in lanes:
        key = (lane["c"], lane["stagger"])
        grouped.setdefault(key, []).append(lane)
    summary = []
    for (c, stagger), rows in sorted(grouped.items()):
        aggs = [r["agg_tok_s"] for r in rows if r["ok"] == c and r["alive_after"]]
        if aggs:
            item = {
                "c": c,
                "stagger": stagger,
                "n": len(aggs),
                "median_agg_tok_s": statistics.median(aggs),
                "min_agg_tok_s": min(aggs),
                "max_agg_tok_s": max(aggs),
                "spread": max(aggs) - min(aggs),
            }
        else:
            item = {"c": c, "stagger": stagger, "n": 0, "median_agg_tok_s": None, "spread": None}
        summary.append(item)
        print("  REPEAT SUMMARY " + json.dumps(item), flush=True)
    return summary


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    cs = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "2,4,8").split(",")]
    max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 96
    repeats = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    print(f"########## C-SWEEP ARM={arm} cs={cs} max_tokens={max_tokens} repeats={repeats} ##########",
          flush=True)
    if not server_alive():
        print("server not up", flush=True)
        sys.exit(2)
    # Warmup (DeepGEMM JIT etc.)
    one_request(PROBES[0][0], 8)
    lanes = []
    for repeat_idx in range(repeats):
        for c in cs:
            # Burst arrivals (pure decode batching after prefills drain)...
            lanes.append(concurrent_lane(c, max_tokens, 0.0, repeat_idx))
            if not lanes[-1]["alive_after"]:
                print("SERVER DIED — aborting sweep", flush=True)
                print("JSON_SUMMARY=" + json.dumps({"arm": arm, "lanes": lanes}), flush=True)
                return
            # ...and staggered arrivals (forces mixed prefill+decode plans).
            lanes.append(concurrent_lane(c, max_tokens, 1.0, repeat_idx))
            if not lanes[-1]["alive_after"]:
                print("SERVER DIED — aborting sweep", flush=True)
                print("JSON_SUMMARY=" + json.dumps({"arm": arm, "lanes": lanes}), flush=True)
                return
    summary = summarize_repeats(lanes)
    print("JSON_SUMMARY=" + json.dumps(
        {"arm": arm, "repeats": repeats, "lanes": lanes, "summary": summary}
    ), flush=True)


if __name__ == "__main__":
    main()
