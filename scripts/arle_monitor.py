#!/usr/bin/env python3
"""Real-time ARLE serve monitor.

Polls /v1/stats and /metrics, prints a compact dashboard of load and
throughput. Ctrl-C to exit.
"""

import argparse
import json
import time
import urllib.request


def fetch(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.read().decode()


def fetch_stats(base):
    return json.loads(fetch(f"{base}/v1/stats"))


def fetch_metrics(base):
    """Parse Prometheus text format into a dict of {name: value}."""
    out = {}
    for line in fetch(f"{base}/metrics").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        # name{labels} value
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        name_and_labels, value = parts
        name = name_and_labels.split("{", 1)[0]
        try:
            out[name] = float(value)
        except ValueError:
            pass
    return out


def fmt(n, width=8):
    if isinstance(n, float):
        if abs(n) >= 1000:
            return f"{n:,.0f}".rjust(width)
        return f"{n:.2f}".rjust(width)
    return str(n).rjust(width)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://10.37.2.27:8000")
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args()

    base = args.url.rstrip("/")
    prev = None
    print(f"ARLE monitor — {base}  (Ctrl-C to exit)\n")

    try:
        while True:
            try:
                stats = fetch_stats(base)
                metrics = fetch_metrics(base)
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] fetch error: {e}")
                time.sleep(args.interval)
                continue

            sched = stats.get("scheduler", {})
            tp = stats.get("throughput", {})
            spec = stats.get("spec_decode", {})
            kv = stats.get("kv_system", {})

            active = sched.get("active_requests", 0)
            queue = sched.get("queue_depth", 0)
            kv_free = sched.get("kv_free_pages", 0)
            kv_total = kv_free + kv.get("resident_pages", 0) + kv.get("host_demoted_pages", 0)
            kv_used_pct = (1 - kv_free / kv_total) * 100 if kv_total else 0

            gen = tp.get("generated_tokens", 0)
            steps = tp.get("steps", 0)
            decode_us = tp.get("decode_forward_busy_micros", 0)
            decode_steps = tp.get("decode_forward_steps", 0)

            now = time.time()
            tok_per_s = None
            if prev and prev["gen"] is not None:
                dt = now - prev["t"]
                d_tok = gen - prev["gen"]
                if dt > 0:
                    tok_per_s = d_tok / dt

            prev = {"t": now, "gen": gen}

            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}]")
            print(f"  load:   active={active}  queue={queue}  kv_used={kv_used_pct:.1f}% ({kv_total - kv_free}/{kv_total} pages)")
            print(f"  kv:     resident={kv.get('resident_pages', 0)}  host_demoted={kv.get('host_demoted_pages', 0)}  disk={kv.get('disk_pages', 0)}")
            print(f"  output: tokens={gen}  steps={steps}  decode_avg={gen / (decode_us/1e6):.1f} tok/s" if decode_us else f"  output: tokens={gen}  steps={steps}")
            if tok_per_s is not None:
                print(f"  rate:   {tok_per_s:.1f} tok/s (last {args.interval:.0f}s)")
            if spec.get("available"):
                print(f"  spec:   dspark chains={spec.get('chains', 0)}  accept_rate={spec.get('accept_rate', 'n/a')}")
            else:
                print(f"  spec:   off")
            print()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
