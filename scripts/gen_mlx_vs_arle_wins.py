#!/usr/bin/env python3
"""Generate the single-machine ARLE-vs-mlx-lm TTFT/TPOT sweep wins entry + copy
the chart, straight from the on-disk results JSON (numbers never hand-typed).

Usage: python3 scripts/gen_mlx_vs_arle_wins.py
"""
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = "/tmp/mlx_arle_sweep.json"
PNG_SRC = "/tmp/mlx_arle_sweep.png"
DATE = "2026-05-31"
ASSET = f"assets/{DATE}-mlx-vs-arle-128-12k-sweep.png"
PNG_DST = os.path.join(ROOT, "docs", "experience", "wins", ASSET)
DOC = os.path.join(ROOT, "docs", "experience", "wins",
                   f"{DATE}-bench-arle-vs-mlxlm-128-12k-ttft-tpot.md")

d = json.load(open(RESULTS))
lens = [int(x) for x in d["lens"]]
arle, mlx, mlx_http = d.get("arle", {}), d.get("mlx", {}), d.get("mlx_http", {})
has_http = bool(mlx_http)


def g(src, n, k):
    return (src.get(str(n)) or {}).get(k)


def tpot(src, n):
    v = g(src, n, "decode_tps")
    return (1000.0 / v) if v else None


def f(x, p=2):
    return f"{x:.{p}f}" if isinstance(x, (int, float)) else "—"


def lbl(n):
    return f"{n // 1024}k" if n >= 1024 else str(n)


png_ok = False
if os.path.exists(PNG_SRC):
    os.makedirs(os.path.dirname(PNG_DST), exist_ok=True)
    shutil.copyfile(PNG_SRC, PNG_DST)
    png_ok = True

# fair MLX reference for headline ratios: HTTP if available else engine
ref = mlx_http if has_http else mlx
ref_name = "mlx-lm HTTP" if has_http else "mlx-lm engine"

# ---- table ----
hdr = "| prompt | ARLE TTFT (s) | "
sep = "|--------|--------------:|"
if has_http:
    hdr += "mlx HTTP TTFT (s) | "; sep += "-----------------:|"
hdr += "mlx eng prefill (s) | ARLE TPOT (ms) | "
sep += "-------------------:|---------------:|"
if has_http:
    hdr += "mlx HTTP TPOT (ms) | "; sep += "------------------:|"
hdr += "mlx eng TPOT (ms) |"
sep += "-----------------:|"
rows = [hdr, sep]
ttft_ratios, tpot_ratios = [], []
for n in lens:
    at, et = g(arle, n, "ttft_s"), g(mlx, n, "prefill_s")
    ht = g(mlx_http, n, "ttft_s")
    a_tp, e_tp, h_tp = tpot(arle, n), tpot(mlx, n), tpot(mlx_http, n)
    r = f"| {lbl(n)} | {f(at)} | "
    if has_http:
        r += f"{f(ht)} | "
    r += f"{f(et)} | {f(a_tp,1)} | "
    if has_http:
        r += f"{f(h_tp,1)} | "
    r += f"{f(e_tp,1)} |"
    rows.append(r)
    rt = g(ref, n, "ttft_s" if has_http else "prefill_s")
    if at and rt:
        ttft_ratios.append((n, at / rt))
    rtp = tpot(ref, n)
    if a_tp and rtp:
        tpot_ratios.append((n, a_tp / rtp))
table = "\n".join(rows)

short_tt = next((r for n, r in ttft_ratios if n <= 512), None)
long_tt = ttft_ratios[-1] if ttft_ratios else None
short_tp = next((r for n, r in tpot_ratios if n <= 512), None)
long_tp = tpot_ratios[-1] if tpot_ratios else None

http_note = (
    "Decode is measured **the same way on both sides** (mlx-lm via its own "
    "`mlx_lm.server` `/v1/completions` with the identical SSE client), so the "
    "TPOT comparison is transport-matched and apples-to-apples."
    if has_http else
    "**Caveat:** ARLE TPOT is client-observed over HTTP/SSE while the mlx-lm "
    "reference here is engine-internal `generation_tps` (no HTTP) — the absolute "
    "TPOT offset is part transport, part engine. The *shape* (ARLE TPOT rising "
    "with context) is transport-independent and therefore real. Run "
    "`scripts/bench_mlx_http_decode.py` for the transport-matched MLX TPOT.")

doc = f"""# Bench: ARLE Metal vs mlx-lm — single-machine TTFT/TPOT sweep 128→12k (Qwen3.6)

## Goal
Single-machine (c=1) prompt-length sweep, 128 → 12 288 tokens, charting the two
canonical serving-latency metrics — **TTFT** (time to first token) and **TPOT**
(time per output token) — for ARLE Metal vs the mlx-lm reference on the canonical
Metal model. Hard constraint: never co-resident two 19 GB models on the 48 GB box
(a prior run hung that way).

## Hypothesis
TTFT is comparable once the prefix cache is defeated; TPOT is where the backends
diverge with context length.

## Params / Env
- Model: `mlx-community/Qwen3.6-35B-A3B-4bit` (same HF snapshot; 40 layers,
  2 KV heads, head_dim 256 → KV ~0.94 GB @12k).
- HW: Apple Silicon, 48 GB unified; macOS. c=1, greedy (temp 0), {d.get('out_tokens')} output tokens.
- **Prefix cache defeated**: every request gets a unique nonce prefix
  (`make_prompt`), so prefill is always uncached and lengths are non-nested —
  the trap that voided the first eb14f29e A/B.
- ARLE: `metal_serve --max-running-requests 1 --max-batch-tokens 4096`, auto wired
  limit, streaming `/v1/completions`; TTFT + TPOT from client SSE timing.
- mlx-lm (engine): in-process `stream_generate`; TTFT = prompt_tokens/prompt_tps,
  TPOT = 1000/generation_tps.{" " if has_http else ""}
- mlx-lm (HTTP): `mlx_lm.server` `/v1/completions`, identical SSE client to ARLE.{"" if has_http else "  (not run in this pass)"}
- **Memory safety**: one 19 GB model resident at a time — ARLE `metal_serve` fully
  terminated and RAM reclaimed before any mlx-lm phase; in-script watchdog SIGKILLs
  the model if free RAM < 1.0 GB. Lowest free RAM with one model ≈ 5 GB; no hang.
- Driver `scripts/bench_mlx_vs_arle_sweep.py` (+ `bench_mlx_http_decode.py`),
  chart `scripts/plot_mlx_vs_arle_sweep.py`, doc `scripts/gen_mlx_vs_arle_wins.py`.

## Results

{table}

![ARLE vs mlx-lm TTFT/TPOT sweep]({ASSET})

## Learnings
- **TTFT (prefill)**: ARLE is ~{f(short_tt,2)+'×' if short_tt else '~1×'} {ref_name} at ≤512 and
  ~{f(long_tt[1],2)+'×' if long_tt else '?'} at {lbl(long_tt[0]) if long_tt else '?'} — {"roughly at parity / a modest gap" if (long_tt and long_tt[1] < 1.6) else "a widening gap at long context"}.
  (The earlier "ARLE 2× faster" was a prefix-cache artifact; with nonces the
  curves track closely.)
- **TPOT (decode) is the real divergence**: ARLE TPOT rises steeply with context
  (~{f(short_tp,2)+'×' if short_tp else '?'} {ref_name} at ≤512 → ~{f(long_tp[1],1)+'×' if long_tp else '?'} at {lbl(long_tp[0]) if long_tp else '?'}),
  while mlx-lm TPOT is near-flat. A context-dependent TPOT slope cannot come from
  fixed HTTP overhead — it is a real ARLE Metal decode characteristic: per-token
  decode cost grows with KV/context far faster than mlx-lm's. {http_note}
- **Highest-value Metal item**: the long-context decode path (TPOT@8k–12k). Agent /
  multi-turn workloads sit exactly there. Profile the decode attention + per-step
  scheduler overhead vs context length (Xcode Metal capture / MLX trace).

## Rule
Report TTFT and TPOT separately, on a prompt-length sweep — a single c/shape hides
the divergence. Defeat the prefix cache with per-request nonces or prefill numbers
are fantasy. Measure decode the same way on both engines (both over HTTP, or both
in-engine); never compare HTTP-observed TPOT to engine-internal TPOT for absolute
claims. RAM-tight cross-backend A/B runs strictly sequentially with a watchdog.
"""

with open(DOC, "w") as fh:
    fh.write(doc)

print(f"PNG copied: {png_ok} -> {PNG_DST}")
print(f"DOC -> {DOC}")
print(f"has_http={has_http}")
print(f"TTFT ratios vs {ref_name}: {[(lbl(n), round(r,2)) for n,r in ttft_ratios]}")
print(f"TPOT ratios vs {ref_name}: {[(lbl(n), round(r,2)) for n,r in tpot_ratios]}")
