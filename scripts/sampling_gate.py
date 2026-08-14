#!/usr/bin/env python3
"""End-to-end sampling-parameter gate against a live serve.

One greedy zero-penalty reference, then one arm per sampling parameter; each
penalty arm must produce decoded text that diverges from the reference, the
logit_bias +100 arm must diverge at/near char 0 and be dominated by the biased
token, and two ordinary follow-ups must answer non-empty (the TP>1 relay
failure answers 200 and then kills every rank). /v1/stats supplies the
spec-engagement counters and product_binary_sha256 for path identity.
Method: docs/experience/wins/2026-08-14-sampling-penalties-verified-on-both-runtimes.md
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

PROMPT = "List ten common fruits, one per line, then explain which is your favorite and why."


def http_json(url, body=None, timeout=240):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.load(r)


def decoded_text(response):
    # Thinking models answer in reasoning_content and leave content empty;
    # reading only content reports a healthy serve as a dead one.
    m = response["choices"][0]["message"]
    return (m.get("reasoning_content") or "") + (m.get("content") or "")


def first_divergence(ref, out):
    n = min(len(ref), len(out))
    for i in range(n):
        if ref[i] != out[i]:
            return i
    return None if len(ref) == len(out) else n


def dominance(out):
    toks = out.split()
    if not toks:
        return 0.0
    top = max(set(toks), key=toks.count)
    return toks.count(top) / len(toks)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("port")
    ap.add_argument("--biased-token", type=int, default=73233)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    ask = {
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0,
        "max_tokens": 128,
    }
    if args.model:
        ask["model"] = args.model

    arms = [
        ("reference", {}),
        ("repetition_penalty", {"repetition_penalty": 1.5}),
        ("frequency_penalty", {"frequency_penalty": 1.0}),
        ("presence_penalty", {"presence_penalty": 1.0}),
        ("logit_bias", {"logit_bias": {str(args.biased_token): 100.0}}),
    ]

    rows = []
    ok = True
    ref_text = None
    try:
        for name, extra in arms:
            st, r = http_json(f"{base}/v1/chat/completions", {**ask, **extra})
            out = decoded_text(r)
            if name == "reference":
                ref_text = out
                passed = st == 200 and len(out) > 0
                rows.append((name, st, "-", f"len={len(out)}", passed))
            elif name == "logit_bias":
                div = first_divergence(ref_text, out)
                dom = dominance(out)
                passed = st == 200 and div is not None and div <= 2 and dom >= 0.5
                rows.append((name, st, div, f"dom={dom:.2f}", passed))
            else:
                div = first_divergence(ref_text, out)
                passed = st == 200 and div is not None
                rows.append((name, st, div, f"len={len(out)}", passed))
            ok &= passed

        # Liveness: the relay failure this gate exists for returns 200 on the
        # biased request and then kills every rank — judge two more on text.
        for i in range(2):
            st, r = http_json(f"{base}/v1/chat/completions", ask)
            out = decoded_text(r)
            passed = st == 200 and len(out) > 0
            rows.append((f"liveness[{i}]", st, "-", f"len={len(out)}", passed))
            ok &= passed
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        print(f"sampling_gate: cannot reach serve at {base}: {e}", file=sys.stderr)
        return 1

    stats_line = "stats: unavailable"
    try:
        _, j = http_json(f"{base}/v1/stats", timeout=10)
        body = j.get("body", j) if isinstance(j, dict) else j
        if isinstance(body, str):
            body = json.loads(body)
        stats_line = (
            f"stats: chains={body.get('chains')} drafted={body.get('drafted')} "
            f"accepted={body.get('accepted')} "
            f"product_binary_sha256={body.get('product_binary_sha256')}"
        )
    except Exception as e:
        stats_line = f"stats: unavailable ({e})"

    print(f"{'arm':<20} {'http':>4} {'diverge@':>8} {'detail':<12} verdict")
    for name, st, div, detail, passed in rows:
        print(f"{name:<20} {st:>4} {str(div):>8} {detail:<12} {'PASS' if passed else 'FAIL'}")
    print(stats_line)
    print(f"gate: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
