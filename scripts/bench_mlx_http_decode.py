#!/usr/bin/env python3
"""Fair decode A/B companion: measure mlx-lm decode/TTFT over its OWN HTTP server
with the SAME client-side SSE timing used for ARLE, so the decode comparison is
apples-to-apples (identical transport overhead on both sides).

Merges an "mlx_http" block into /tmp/mlx_arle_sweep.json. Sequential + watchdog —
only ONE 19 GB model resident (the ARLE/MLX phases already finished and freed).
"""
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request

import psutil

MODEL = "mlx-community/Qwen3.6-35B-A3B-4bit"
PORT = 8850
HOST = "127.0.0.1"
LENS = [128, 256, 512, 1024, 2048, 4096, 8192, 12288]
OUT_TOKENS = 64
FLOOR_GB = 1.0
RESULTS = "/tmp/mlx_arle_sweep.json"
LOG = "/tmp/mlx_arle_sweep.log"

_logf = open(LOG, "a", buffering=1)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, file=_logf)
    print(line, flush=True)


def avail_gb():
    return psutil.virtual_memory().available / 1024 / 1024 / 1024


_watch = {"proc": None, "abort": False, "stop": False}


def watchdog():
    while not _watch["stop"]:
        if avail_gb() < FLOOR_GB:
            _watch["abort"] = True
            log(f"!!! WATCHDOG_ABORT: available={avail_gb():.1f}GB — killing model")
            p = _watch["proc"]
            if p is not None and p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception:
                    pass
            return
        time.sleep(2)


def wait_ready(timeout=240):
    url = f"http://{HOST}:{PORT}/v1/models"
    t0 = time.time()
    while time.time() - t0 < timeout:
        if _watch["abort"]:
            return False
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        if _watch["proc"] and _watch["proc"].poll() is not None:
            return False
        time.sleep(2)
    return False


def probe(n):
    # /v1/completions (raw prompt) — identical request format to the ARLE probe.
    prompt = "word " * n
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": OUT_TOKENS, "temperature": 0, "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"http://{HOST}:{PORT}/v1/completions",
        data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    ts = []
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                ch = json.loads(payload)["choices"][0]
            except Exception:
                continue
            if ch.get("text"):
                now = time.time()
                if ttft is None:
                    ttft = now - t0
                ts.append(now)
    # Steady-state TPOT from token 2 onward (drop the token1->2 prefill-tail
    # interval) — same fix as bench_mlx_vs_arle_sweep.py so both sides match.
    if not ts:
        return ttft, None, 0
    intervals = [(ts[i] - ts[i - 1]) * 1000.0 for i in range(1, len(ts))]
    rest = intervals[1:]
    dtps = (1000.0 / (sum(rest) / len(rest))) if rest else None
    return ttft, dtps, len(ts)


def main():
    log("=" * 64)
    log(f"MLX-HTTP decode A/B  avail={avail_gb():.1f}GB")
    threading.Thread(target=watchdog, daemon=True).start()
    res = {}
    log("MLX-HTTP: launching mlx_lm.server")
    env = dict(os.environ)
    p = subprocess.Popen(
        [sys.executable, "-m", "mlx_lm", "server", "--model", MODEL,
         "--port", str(PORT), "--host", HOST],
        stdout=open("/tmp/mlx_server.log", "w"), stderr=subprocess.STDOUT,
        start_new_session=True, env=env)
    _watch["proc"] = p
    try:
        if not wait_ready():
            log("MLX-HTTP: server not ready — check /tmp/mlx_server.log")
            return
        log(f"MLX-HTTP: ready (avail={avail_gb():.1f}GB). warmup...")
        probe(128)
        for n in LENS:
            if _watch["abort"]:
                break
            ttft, dtps, out = probe(n)
            res[n] = {"ttft_s": ttft, "decode_tps": dtps, "out_tokens": out}
            log(f"MLX-HTTP n={n:6d}: TTFT={ttft:.3f}s decode={dtps:.1f}tok/s "
                f"out={out} avail={avail_gb():.1f}GB")
    finally:
        if p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                pass
            for _ in range(20):
                if p.poll() is not None:
                    break
                time.sleep(1)
            if p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception:
                    pass
        _watch["stop"] = True
        out_json = {}
        if os.path.exists(RESULTS):
            out_json = json.load(open(RESULTS))
        out_json["mlx_http"] = {str(k): v for k, v in res.items()}
        out_json["watchdog_abort"] = out_json.get("watchdog_abort", False) or _watch["abort"]
        with open(RESULTS, "w") as f:
            json.dump(out_json, f, indent=2)
        log(f"MLX-HTTP DONE. merged mlx_http into {RESULTS} (abort={_watch['abort']})")


if __name__ == "__main__":
    main()
