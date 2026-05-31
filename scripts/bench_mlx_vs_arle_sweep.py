#!/usr/bin/env python3
"""Safe sequential MLX-vs-ARLE c=1 prefill/decode sweep on the canonical Metal
model (Qwen3.6-35B-A3B-4bit, ~19 GB) — for a 48 GB Apple Silicon box.

SAFETY (the box already hung once from two 19 GB models co-resident):
  * Only ONE 19 GB model is ever resident. The ARLE metal_serve process is
    fully terminated and its memory confirmed reclaimed BEFORE the MLX phase
    loads its own copy.
  * A background watchdog thread polls available RAM every 2 s; if it drops
    below FLOOR_GB it kills the ARLE child process group (or aborts the MLX
    phase) immediately and writes a WATCHDOG_ABORT marker.
  * c=1 only (Metal local focus), bounded output tokens, bounded max-seq-len.

Outputs JSON to RESULTS and a human log to LOG. Run in the background and tail
the log; the script protects the Mac on its own.
"""
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
import urllib.request

import psutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "mlx-community/Qwen3.6-35B-A3B-4bit"
BIN = os.path.join(ROOT, "target", "release", "metal_serve")
PORT = 8849
HOST = "127.0.0.1"
LENS = [128, 256, 512, 1024, 2048, 4096, 8192, 12288]
OUT_TOKENS = 64
MAX_SEQ_LEN = 16384
# Steady-state available with ONE 19 GB model on this 48 GB box is ~2.3 GB
# (KV is tiny — 2 GQA heads, ~1.25 GB even at 16k). FLOOR is purely a catastrophe
# backstop for the 2-model case (which the sequential design already prevents);
# set well below the proven-safe one-model operating point so it never false-trips.
FLOOR_GB = 1.0
MLX_MIN_AVAIL_GB = 18.0  # need ~20 GB free to load the MLX copy
RESULTS = "/tmp/mlx_arle_sweep.json"
LOG = "/tmp/mlx_arle_sweep.log"

_logf = open(LOG, "a", buffering=1)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, file=_logf)
    print(line, flush=True)


def avail_gb():
    return psutil.virtual_memory().available / 1024 / 1024 / 1024


_REQ = [0]


def make_prompt(n):
    """Build a ~n-token prompt with a UNIQUE nonce prefix per request.

    The server's radix/prefix cache matches from the *start* of the prompt, so a
    unique first token forces a full (uncached) prefill every time and makes the
    lengths non-nested. This defeats the prefix-cache trap that voided the first
    eb14f29e A/B (identical repeated prompts reported physically-impossible
    prefill tok/s). Used by BOTH the ARLE and MLX probes for a fair comparison.
    """
    _REQ[0] += 1
    nonce = f"{_REQ[0]:06d}{random.randint(0, 1_000_000_000):010d}"
    return f"session {nonce} transcript begins now " + ("word " * n)


# ---- watchdog -------------------------------------------------------------
_watch = {"proc": None, "abort": False, "stop": False}
BASELINE_AVAIL = 0.0  # available RAM before any model loads (set in main)


def watchdog():
    while not _watch["stop"]:
        g = avail_gb()
        if g < FLOOR_GB:
            _watch["abort"] = True
            log(f"!!! WATCHDOG_ABORT: available={g:.1f}GB < {FLOOR_GB}GB — killing model")
            p = _watch["proc"]
            if p is not None and p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception as e:
                    log(f"watchdog killpg failed: {e}")
            return
        time.sleep(2)


# ---- ARLE phase (HTTP server) --------------------------------------------
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
            log("metal_serve exited during boot")
            return False
        time.sleep(2)
    return False


def arle_probe(n):
    """One streaming /v1/completions request. Returns (ttft_s, decode_tps, out).

    metal_serve serves /v1/completions (raw prompt) — same endpoint the
    eb14f29e A/B used. Streaming chunks carry choices[0].text.
    """
    prompt = make_prompt(n)
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": OUT_TOKENS,
        "temperature": 0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"http://{HOST}:{PORT}/v1/completions",
        data=body, headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = first_tok_t = last_tok_t = None
    out = 0
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except Exception:
                continue
            if chunk.get("choices", [{}])[0].get("text"):
                now = time.time()
                if ttft is None:
                    ttft = now - t0
                    first_tok_t = now
                last_tok_t = now
                out += 1
    decode_tps = None
    if out > 1 and last_tok_t and first_tok_t and last_tok_t > first_tok_t:
        decode_tps = (out - 1) / (last_tok_t - first_tok_t)
    return ttft, decode_tps, out


def run_arle():
    res = {}
    env = dict(os.environ)
    env["RUST_LOG"] = env.get("RUST_LOG", "warn")
    log(f"ARLE: launching metal_serve (avail={avail_gb():.1f}GB)")
    # metal_serve flags: c=1 via --max-running-requests 1; --max-batch-tokens
    # 4096 matches the eb14f29e A/B (largest prefill chunk that fits); context
    # length comes from the model (max_position_embeddings=262144). Auto wired
    # limit (no --wired-limit-bytes) = the canonical c=1 default.
    p = subprocess.Popen(
        [BIN, "--model-path", MODEL, "--port", str(PORT),
         "--max-running-requests", "1", "--max-batch-tokens", "4096"],
        stdout=open("/tmp/arle_serve.log", "w"), stderr=subprocess.STDOUT,
        start_new_session=True, env=env,
    )
    _watch["proc"] = p
    try:
        if not wait_ready():
            log("ARLE: server not ready — skipping ARLE phase")
            return res
        log(f"ARLE: ready (avail={avail_gb():.1f}GB). warmup...")
        try:
            arle_probe(128)
        except Exception as e:
            log(f"ARLE: warmup failed (non-fatal): {e!r}")
        for n in LENS:
            if _watch["abort"]:
                log("ARLE: aborted by watchdog")
                break
            try:
                ttft, dtps, out = arle_probe(n)
            except Exception as e:
                log(f"ARLE n={n:6d}: probe error {e!r}")
                continue
            res[n] = {"ttft_s": ttft, "decode_tps": dtps, "out_tokens": out}
            dstr = f"{dtps:.1f}" if dtps else "n/a"
            tstr = f"{ttft:.3f}" if ttft else "n/a"
            log(f"ARLE n={n:6d}: TTFT={tstr}s decode={dstr}tok/s "
                f"out={out} avail={avail_gb():.1f}GB")
    finally:
        log("ARLE: terminating server")
        if p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                pass
            for _ in range(30):
                if p.poll() is not None:
                    break
                time.sleep(1)
            if p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception:
                    pass
        _watch["proc"] = None
        # confirm the 19 GB is reclaimed before MLX loads (back near baseline)
        target = BASELINE_AVAIL - 4
        for _ in range(60):
            g = avail_gb()
            if g > target:
                log(f"ARLE: memory reclaimed (avail={g:.1f}GB, target>{target:.1f})")
                break
            time.sleep(2)
    return res


# ---- MLX phase (in-process) ----------------------------------------------
def run_mlx():
    res = {}
    if _watch["abort"]:
        log("MLX: skipped (prior watchdog abort)")
        return res
    g = avail_gb()
    if g < MLX_MIN_AVAIL_GB:
        log(f"MLX: not enough RAM to load model safely (avail={g:.1f}GB < {MLX_MIN_AVAIL_GB}) — skipping")
        return res
    log(f"MLX: loading model (avail={g:.1f}GB)")
    from mlx_lm import load, stream_generate
    model, tok = load(MODEL)
    log(f"MLX: loaded (avail={avail_gb():.1f}GB). warmup...")

    def probe(n):
        prompt = make_prompt(n)
        msgs = [{"role": "user", "content": prompt}]
        ptoks = tok.apply_chat_template(msgs, add_generation_prompt=True)
        last = None
        for r in stream_generate(model, tok, ptoks, max_tokens=OUT_TOKENS):
            last = r
        return last

    probe(128)
    for n in LENS:
        if _watch["abort"]:
            log("MLX: aborted by watchdog")
            break
        r = probe(n)
        prompt_tokens = getattr(r, "prompt_tokens", None)
        prompt_tps = getattr(r, "prompt_tps", None)
        gen_tps = getattr(r, "generation_tps", None)
        prefill_s = (prompt_tokens / prompt_tps) if (prompt_tokens and prompt_tps) else None
        res[n] = {
            "prefill_s": prefill_s,
            "decode_tps": gen_tps,
            "prompt_tokens": prompt_tokens,
            "prompt_tps": prompt_tps,
        }
        log(f"MLX  n={n:6d}: prefill={prefill_s:.3f}s decode={gen_tps:.1f}tok/s "
            f"ptoks={prompt_tokens} avail={avail_gb():.1f}GB")
    del model
    return res


def main():
    global BASELINE_AVAIL
    BASELINE_AVAIL = avail_gb()
    log("=" * 64)
    log(f"START sweep  total_RAM={psutil.virtual_memory().total/1024**3:.0f}GB "
        f"avail={BASELINE_AVAIL:.1f}GB FLOOR={FLOOR_GB}GB")
    log(f"LENS={LENS} OUT_TOKENS={OUT_TOKENS}")
    wt = threading.Thread(target=watchdog, daemon=True)
    wt.start()
    # PHASE = arle | mlx | both. Single-phase runs MERGE into an existing
    # results file so the two 19 GB models are never benched in one process.
    phase = os.environ.get("ARLE_SWEEP_PHASE", "both").lower()
    out = {"model": MODEL, "lens": LENS, "out_tokens": OUT_TOKENS,
           "arle": {}, "mlx": {}, "watchdog_abort": False}
    if phase != "both" and os.path.exists(RESULTS):
        try:
            out.update(json.load(open(RESULTS)))
        except Exception:
            pass
    log(f"PHASE={phase}")
    try:
        if phase in ("arle", "both"):
            out["arle"] = run_arle()
        if phase in ("mlx", "both"):
            out["mlx"] = run_mlx()
    except Exception as e:
        log(f"ERROR: {e!r}")
    finally:
        _watch["stop"] = True
        out["watchdog_abort"] = _watch["abort"]
        with open(RESULTS, "w") as f:
            json.dump(out, f, indent=2)
        log(f"DONE. results -> {RESULTS}  watchdog_abort={_watch['abort']}")


if __name__ == "__main__":
    main()
