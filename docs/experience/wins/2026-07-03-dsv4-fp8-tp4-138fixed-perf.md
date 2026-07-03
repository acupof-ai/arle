# DSv4-Flash-FP8 TP=4 decode perf — first valid long-context baseline (post-#138), via /v1/stats

> Status: baseline measurement, 2026-07-03, 8×H20 pod (115.190.184.36),
> DeepSeek-V4-Flash-FP8, TP=4/EP=4 on **GPUs 0-3 only** (4-7 left free),
> `--spec-type mtp` (MTP-on, production default), greedy T=0.
> **Measured via `/v1/stats` + a python3-stdlib client, NOT guidellm** — the
> guidellm `pip install` is network-blocked on the pod (a prior session hung 40
> min on it), so this is the round-7 stats-trace method. First DSv4
> long-context (2048-in / 256-out) perf baseline.

## SLO-shape probed?  N (partial)

batch ≥ 4 met (measured c=4/8/16); **prefill 2013 < 4096 and prompt 2013 < 8K**,
so this is a *first baseline at 2048-in/256-out*, not an SLO-PASS. It cannot
drive a default-flag-flip. A full SLO run (≥4096 prefill / ≥8K prompt) is the
next step once guidellm (or an equivalent long-prompt driver) is available.

## Roofline check

Deferred — accept uncertainty. This is a baseline characterization, not an
optimization claim (no X% delta asserted). A decode-GEMV roofline needs the
per-token active-MoE-param count + KV-read bytes on DSv4-Flash-FP8, which was
not instrumented here. Next step for an optimization pass: `ncu` on the decode
step to get achieved GB/s vs H20 HBM peak (~4.0 TB/s).

## Goal

- Establish the first correctness-gated DSv4-Flash-FP8 TP=4 decode-latency /
  throughput baseline at a realistic long-context shape (~2048-in / 256-out),
  MTP-on, so future kernel/scheduler work has a wall-clock reference.

## Hypothesis

- With `num_slots=4`, useful concurrency caps at 4; MTP roughly doubles
  tokens/step; full-wall aggregate would be prefill-bound at this prompt length.

## Environment

- **Backend:** cuda
- **Model:** DeepSeek-V4-Flash-FP8 (294 GB FP8 weights, 46 shards)
- **Hardware:** 8×NVIDIA H20 (97,871 MiB each), driver 535.161.08, CUDA 12.9
  (V12.9.86); sglang cu129 container. **Pinned to GPUs 0-3** (`CUDA_VISIBLE_DEVICES=0,1,2,3`);
  GPUs 4-7 left free (GPU 4 carried a foreign Qwen3.6-27B serve — untouched).
- **Parallelism:** TP=4 / EP=4, 4 worker ranks (rank0 owns output).
- **Serve config** (from `/v1/stats` + serve-log `EngineLoadConfig`): `num_slots=4`,
  `total_pages=8192`, `page_size=16`, `chunked_prefill_size=64`,
  `mem_fraction_static=0.9`, `kv_cache_dtype=auto`, `kv_dram=Fraction(0.5)`,
  `INFER_DSV4_MAX_SEQ_LEN=5120`.
- **MTP:** `--spec-type mtp`, `mtp_draft_tokens=2`, `mtp_draft_topk=1`
  (`depth=2 topk=1 draft_rows=2 verify_rows=3`).
- **loader prefetch:** `294.0 GB across 46 shards in 13.0s (22.62 GB/s, 16 threads)`
  (rank0; ranks 22.0–22.6 GB/s).
- **Binary provenance (caveat):** running serve = `/host/arle-dsv4-snap/arle`,
  built 2026-07-03 13:29 UTC, embeds git sha `59807616` which is **untraceable**
  in both local and pod git (a #138-campaign WIP; the pod build tree
  `/host/arle-build` has since been reset to HEAD `954d9905`, the #138 root-cause
  fix — the task's floor). Commit-exact provenance is therefore NOT proven.
  What IS proven: the binary was built 3 h *after* `954d9905` committed
  (10:24 UTC), and it passes the #138 correctness gate directly — a ~2014-token
  prompt returns **coherent, on-topic text** (189 tok, EOS'd early), no
  token-0/NaN collapse. The commit floor is a proxy for that gate; the gate
  passes on measured output.
- **Profiling state:** OFF.

## Method

- python3-stdlib client (`urllib`, `threading`), run pod-side against
  `127.0.0.1:8000` (no tn-tunnel latency in the timings). Every prompt gets a
  unique nonce → cold prefill (defeats prefix cache; confirmed `d_pref≈2015`/req).
- arle streams SSE in **coarse buffered flushes** (a 12-tok gen arrived in one
  chunk; on 256-tok gens `first_chunk ≈ total`) → per-token stream timing is
  unusable. TTFT/TPOT therefore use the **two-request method**: cold
  `max_tokens=1` latency = TTFT (prefill + 1 decode step); cold `max_tokens=256`
  latency = L256; `TPOT = (L256 − TTFT)/(256−1)`. Both cold, equal prompt length
  → prefill cancels, isolating decode.
- `/v1/stats` `throughput.{steps,generated_tokens,prefill_tokens}` deltas are the
  authoritative service trace; MTP accept from the serve log's cumulative
  `[dsv4-mtp] accept_total / reject_total`.
- c=high: N concurrent requests (barrier-synced), a 0.4 s `/v1/stats` sampler
  thread → peak windowed (2 s) decode tok/s isolates the steady decode phase
  from the prefill ramp.

## Results — c=1 (the primary anchor)

Prompt 2013 tok, output 256 tok (exact, via stats), greedy. 3 cold reps each.

| metric | value |
|---|---:|
| **TTFT** (cold prefill + 1 tok) | **1.420 s** (reps 1.420 / 1.420 / 1.440) |
| L256 total (cold) | 8.795 s median (8.027 / 8.795 / 9.508) |
| **TPOT** = (L256−TTFT)/255 | **28.9 ms / committed-token** |
| **decode throughput** = 255/(L256−TTFT) | **34.6 tok/s** |
| e2e throughput = 256/L256 | 29.1 tok/s |
| MTP committed tokens/step (Δgen/Δsteps) | **2.07** (reps 1.88 / 2.06 / 2.31) |
| MTP accept rate (Δacc/(Δacc+Δrej)) | ~55% (reps 46% / 55% / 68%) |
| implied ms per decode step | ~60 ms (28.9 ms × 2.07) |
| prefill rate (2013 tok / TTFT) | ~1.4k tok/s |

MTP folds acceptance into the 28.9 ms: ~2.07 committed tok/step ≈ 2× the
no-spec 1.0 tok/step, i.e. MTP ~halves the effective ms/token vs a
hypothetical 1-tok/step at the same ~60 ms/step.

## Results — c=high (num_slots=4 is the ceiling)

Same 2048-in/256-out shape, N barrier-synced concurrent cold requests.
Aggregate from `/v1/stats` deltas is authoritative; per-req effective assumes
256 out tok (some requests EOS early on this prompt → `d_gen` < N×256).

| N | peak windowed decode tok/s | full-wall agg tok/s | TTFT-under-load median (max) s | per-req completion med (max) s | per-req eff tok/s (med) | MTP tok/step | accept |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | — (34.6 decode) | 29.1 | 1.42 (1.44) | 8.80 (9.51) | 29.1 | 2.07 | 55% |
| 4 | **51.8** | 35.9 | 5.68 (5.68) | 28.3 (28.5) | 9.05 | 7.94 | 59% |
| 8 | 45.8 | 31.4 | 10.5 (11.2) | 38.7 (57.2) | 6.63 | 6.37 | 47% |
| 16 | 49.9 | 33.1 | 17.3 (22.3) | 63.0 (110.0) | 4.08 | 7.31 | 52% |

Raw stats deltas: c4 `d_gen=1024 d_steps=129`; c8 `d_gen=1796 d_steps=282`;
c16 `d_gen=3648 d_steps=499`. All 4 slots decode in lockstep (tok/step ÷ active
slots ≈ 1.8–2.0/req, matching c=1).

## Problems

- **guidellm unavailable** — `pip install` network-blocked on the pod; measured
  via `/v1/stats` + stdlib client instead (round-7 method). No guidellm HTML.
- **Buffered SSE** defeats streaming TTFT — worked around with the 2-request
  method (documented above).
- **Early EOS** on the repetitive nonce-prompt makes per-req effective tok/s at
  c=8/16 an over-estimate (uses 256); the stats-`d_gen` aggregate is the sound
  number.
- **Binary commit untraceable** (`59807616`) — see provenance caveat; correctness
  gated functionally instead.

## Learnings

- **`num_slots=4` is a hard concurrency wall.** Peak decode throughput plateaus
  at **~50 tok/s** (c4 51.8, c8 45.8, c16 49.9). Beyond N=4 the system only
  queues: TTFT scales ~linearly with the number of concurrent cold prefills
  (1.42 → 5.68 → 10.5 → 17.3 s for N=1/4/8/16) and per-req completion blows out
  (8.8 → 63 s median at N=16) with **no aggregate-throughput gain**.
- **Batch-4 gives only ~1.45× over single-stream** (34.6 → ~50 tok/s) — weak
  MoE batch scaling, expected with MTP `verify_rows=3` per slot (12 model rows /
  step at batch 4).
- **Full-wall aggregate is prefill-bound at this shape** (~31–36 tok/s for all N):
  every request pays a full cold 2048-tok prefill and prefill serializes across
  the batch, so the decode ceiling never shows up in the end-to-end number.
- **MTP ≈ 2.07 committed tok/step (~55% accept)** at c=1 — a real ~2× decode-step
  efficiency, folded into the 28.9 ms TPOT.

## Δ vs baseline

- **First valid DSv4 long-context (2048/256) perf baseline** — no prior snapshot
  to diff. Prior DSv4 wins were short-shape kernel micro-benches
  (`2026-05-12-bench-dsv4-fp8-*`); this is the first end-to-end serve-level
  latency/throughput reference post-#138.

## Notes

- Cleanup: serve killed by exact PID at end of session (prior session torn down);
  GPUs 0-3 → 0; GPU 4 foreign Qwen serve untouched; `tn hold --stop`.
- Follow-ups: (1) SLO-shape run at ≥4096 prefill / ≥8K prompt; (2) raise
  `num_slots` (>4) and re-measure the decode ceiling under real batching;
  (3) `ncu` decode-GEMV roofline for an optimization pass; (4) a
  provenance-clean rebuild from `954d9905` if a commit-exact number is needed.
