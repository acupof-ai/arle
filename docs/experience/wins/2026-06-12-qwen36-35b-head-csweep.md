# Qwen3.6-35B-A3B CUDA TP=1 HEAD health c-sweep — no regression, c=8 first coverage

**Date:** 2026-06-12. **Backend:** CUDA, Qwen3.6-35B-A3B (67 GB bf16), H20
GPU 0, TP=1, `--num-slots 8`. **Binary:** `36b12bc4` runtime content (built
07:19 UTC same day). **Goal:** user-directed health check of the 35B lane at
HEAD — the 2026-06-11 licensed stack (batched decode + decode-band MoE
kernels + rows==1 whole-step graph, all default-ON) plus every runtime
commit landed since the `9e37bc77` baseline build: EOS honor `ea71e060`,
hd128 prefill wgmma fix `930cb9d3`, KV budget clamps `c7fe1aea`/`75530149`,
paged_kv_table lift `72c00fd4`, quant-KV dispatch (opt-in) `d9f930b6`..
`36b12bc4`, shared MoE scratch `10ab4530`.

## Result

Aggregate decode tok/s, 256-token essay completions ×3 reps after 1 warmup
rep per c (B=1..8 same-binary, same-shell, side-by-side; per-stream token
counts all exactly 256 — the EOS fix does not truncate this workload):

| c | batched (default) | seq (`ARLE_QWEN35_BATCHED_DECODE=0`) | batched Δ vs seq | 06-11 baseline (batched) | Δ vs baseline |
|---|---|---|---|---|---|
| 1 | 93.5 | 93.9 | — (identical path) | 91.97 | +1.7% |
| 2 | 152.3 | 94.3 | +62% | 141.59 | +7.6% |
| 4 | 207.5 | 95.0 | +118% | 185.38 | +11.9% |
| 8 | **255.6** | 93.6 | **+173%** | — (new coverage) | — |

- **No regression anywhere**; every c-point is at or above the 06-11
  baseline. Cross-day caveat applies (different build, box variance — the
  06-11 entry itself flagged this): the load-bearing same-session pair is
  batched-vs-seq; the vs-baseline column is health, not a new license.
- **c=8 first-ever datapoint: 255.6 tok/s aggregate** (2.73× single-stream,
  +23% over c=4). 8 slots boot clean: KV budget marker
  `free 29307MB, per_slot 2621MB (K+V 2560MB + gdr 60MB + conv 1MB)` —
  8×2.62 GB = 21 GB KV inside the 29.3 GB post-weights headroom, the
  machine-derived clamp (`c7fe1aea`) sizing correctly.
- c=8 rep1 ran 205.8 before settling at 256.3/254.9 — one ramp rep at a new
  concurrency level even after the warmup rep; reps 2–3 agree to 0.5%.
- Sequential arm is flat ~94 at every c (by construction) — note it now
  rides the new MoE decode kernels too (default-ON), hence far above the
  06-11 old-kernel sequential row (40.x).
- Per-stream ITL at c=8: 255.6/8 ≈ 32 tok/s/stream vs 93.5 single — the
  aggregate-vs-ITL tradeoff the #60 license gate (TTFT AND ITL AND output)
  must still adjudicate per shape; this entry is throughput health, not a
  scheduling-default change.
- Needle gate at HEAD (`GATE_PROFILE=generic MODEL=Qwen3.6-35B-A3B RAW=1
  lever_gate.sh q36_35b_head`, first 35B ladder — this IS the envelope):
  **len 2000/8000 exact=3 DET** (long-context paged-KV retrieval precise);
  len 446 partial ×3 DET (verbose answer ran out the token budget mid-digit
  — "The secret access code is 73829[1]"); len 115/300 miss ×3 DET — the
  model spontaneously emits `<think>` and burns the budget even under
  RAW=1 (decoded output is coherent thinking prose, NOT corruption; same
  artifact class as the 0.6B chat-template think-burn). No runtime
  correctness flag: retrieval is exact wherever the budget reaches the
  answer, and all classes are deterministic ×3.

## Single-stream decode composition (why 93.5 tok/s with ~3B active)

10.7 ms/token at HEAD sits inside the 06-11 decode-band formula prediction
(9–11 ms, measured 10.9 at `9e37bc77`) — the speed IS the licensed band of
the current kernel stack, not a regression. Against the physical floor:
~3B active ≈ 6 GB bf16 weight-read/token ÷ 4 TB/s H20 HBM ≈ 1.5 ms
(~670 tok/s); measured = ~14% of roofline. The gap is the structural B=1
GEMV tax: each token's ~6 GB fragments into hundreds of small per-layer
kernels (40 × {norms, qkv/GDR scan+conv, router, 9 expert GEMVs, down} +
the 1.0 GB lm_head GEMV over the 248k vocab), serially dependent, each too
small to saturate HBM — the same effect class the 06-11 nsys pinned (old
grouped kernels: 76.6% GPU time at 3% HBM efficiency) before the rework
brought it into the formula band. Direct evidence the GPU is starved, not
saturated: same binary at c=8 does 2.73× aggregate. Levers that move B=1
per `feedback_b1_decode_gpu_bound`: speculative decode (GEMV→small-GEMM),
W4 weight quant (bytes ÷4) — overhead-shaving is a wash. A HEAD per-kernel
nsys refresh is **deferred**: the attempt died at boot (`--duration=240`
expired inside the 67 GB load; script fixed, no duration cap now) and the
box was re-occupied by the restored DSv4 serve before a retry window.

## Context

User-directed ("你来测试 35B的吧") while the 8×H20 pod's DSv4 MTP serve was
parked idle; serve snapshotted (`serve_specon.restore`), killed with user
approval for the duration of the sweep; ckl relaunched it himself
(`arle-serve-allreduce`, tmux `serve`, 08:18 UTC) before the trace-retry
window. Harness: `/tmp/q36_sweep.sh` (pod),
c concurrent `/v1/completions` curls, aggregate = Σ completion_tokens /
wall; prompts share a base essay prefix but diverge per stream (no full
radix dedup).

## Rule

- A HEAD that accumulated N runtime commits since the last bench entry gets
  ONE health c-sweep before new optimization work piles on top — cheap
  (≈25 min incl. two 67 GB boots) and it converts "should still be fine"
  into a dated table the next regression hunt can diff against.
