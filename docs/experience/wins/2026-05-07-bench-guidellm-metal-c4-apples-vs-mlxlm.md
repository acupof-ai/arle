# Bench — ARLE-Metal vs mlx-lm true c=4 apples-to-apples (Qwen3.5-0.8B-MLX-4bit)

## Goal

Re-measure ARLE-Metal vs mlx-lm at **the same client AND server
concurrency** (c=4 → c=4) so the previous c-sweep's TTFT signal isn't
confused by ARLE's server-side queue. The previous run
(`2026-05-07-bench-guidellm-metal-c-sweep-m4pro.md`) sent client c=16
to an ARLE server capped at 4, leaving 12 requests queued and inflating
ARLE's TTFT to 13 s — which read as "ARLE prefill is slow" but was
actually queue overhead. This run pins both sides to c=4.

## Hypothesis

If the previous run's ARLE TTFT was queue-driven, ARLE's actual
prefill should be competitive or better than mlx-lm at matched c=4.
ITL is unchanged either way (the previous c=4 ITL of 19 ms was already
queue-free).

## Params

- Workload: 4096-in / 256-out, concurrent, rate=4, max-seconds=30
- Model: `models/Qwen3.5-0.8B-MLX-4bit` (same instance for both)
- Tokenizer: same path as `--processor`
- ARLE: `metal_serve --max-running-requests 4 --max-batch-tokens 512`
  (defaults; same binary used in the previous c-sweep)
- mlx-lm: 0.31.2, defaults
- guidellm: 0.6.0 invoked via `scripts/bench_guidellm.sh
  --concurrencies 4 --max-seconds 30 --data
  prompt_tokens=4096,output_tokens=256`

## Env

- Host: Apple M4 Pro (20 cores), macOS darwin 25.3.0, ~36 GB unified
- ARLE commit: `e607646` (errata in M_e.1) + `bbc484c` (scheduler flag)
- ARLE feature set: `--no-default-features --features metal`
- KV dtype: BF16 (no quant)
- Single host, sequential server runs (servers swapped between cells)

## Results

| Metric | ARLE-Metal c=4 | mlx-lm c=4 | Δ (ARLE vs mlx-lm) |
|---|---:|---:|---:|
| TTFT mean | 1.49 s | 4.56 s | **-67% (3.06× faster)** |
| TTFT p50 | **1.20 s** | 4.51 s | **-73% (3.77× faster)** |
| TTFT p99 | 4.07 s | 4.74 s | -14% |
| ITL p50 | 19.34 ms | **7.17 ms** | +169% (2.70× slower) |
| ITL p95 | 19.96 ms | 7.61 ms | +162% |
| Output tok/s | 147.5 | **196.1** | -25% (1.33× slower aggregate) |
| Conc p50 | 4 | 4 | matched |
| total tok/s | 2,507 | 3,334 | -25% |

(`req/s actual` was identical: 0.533 — both backends sustained the
same request rate; the throughput delta is purely per-token.)

## Problems

1. **The morning's c-sweep narrative was wrong** about where the gap
   came from. The c=16 ARLE entry showed "13 s TTFT" which I read as
   prefill-bottlenecked; in fact server cap=4 + client c=16 left 12
   requests queued, and the TTFT measurement folded queue wait into
   "time-to-first-token". With matched c=4, ARLE's TTFT is **3.8×
   faster** than mlx-lm. ARLE's chunked prefill + decode-priority
   scheduler interleave (locked by `decode_priority_holds_under_c4_
   mixed_traffic` in tick 2) is real and works.
2. **Single-stream decode IS slower in ARLE Metal**, separate from
   any padding effect. At c=4 the per-batch left-padding is near zero
   (all four prompts ~ same length) — yet ITL is still 2.70× slower
   than mlx-lm. This means the morning's claim that "the 3× gap is
   100% in the kernel-architecture axis (left-pad)" was incomplete.
   There are **two** kernel issues:
   - left-pad collapse at c≥8 (the c-sweep findings stand)
   - per-token decode kernel itself running 2.7× slower than mlx-lm
     even at c=1/c=4 with no padding — root cause not yet
     identified
3. The decode kernel gap is the bigger structural issue. Even if M_e.1
   commits 1–6 close the left-pad gap perfectly, ARLE's c=4 ITL would
   still sit at ~19 ms vs mlx-lm's 7 ms. That's a ~270% gap in the
   per-token decode that paged-KV does NOT address.

## Learnings

- Add **matched-c apples-to-apples** as a default bench protocol step
  whenever comparing backends, not just c-sweep with one side capped.
- The decode-kernel-itself gap is now the leading concern. Hypotheses
  to investigate next bench window:
  - MLX `fast::scaled_dot_product_attention` is invoked with extra
    `eval()` / `item()` boundaries in ARLE (`Metal scheduler is on
    the hot path` per `infer/src/backend/metal/AGENTS.md` "Metal vs
    CUDA mental model" — eager boundaries are scheduling boundaries).
  - mlx-lm fuses RoPE + RMSNorm + matmul through `mx.compile`'d
    submodules; ARLE's per-step compiled-step path may not get the
    same fusion (the per-driver `Qwen3StepDriver` builds graphs each
    step in some modes).
  - Per-step Python-side overhead in mlx-lm is amortized by JIT;
    ARLE's Rust per-step calls cross the C++ bridge and incur fixed
    per-call cost. Measure with `mlx::active_memory_bytes` deltas
    and the existing `metric.set_memory_bytes` trace at `runtime.rs:
    2879` for a per-tick cost estimate.
- M_e.1 plan should grow a **commit 0** at the head: profile single-
  stream decode under MLX instruments / metal capture and identify
  the dominant cost. Without that, paged-KV (commits 1–6) closes one
  axis only.
- TTFT win is genuine and worth surfacing in M6 / M_e gauntlet
  reporting. ARLE's prefill story is healthy — chunked + decode-
  priority + the scheduler runtime all working as intended.
- ELI Layer 1 latency context: the smoke-test `curl /v1/chat/completions`
  in the previous bench session hit HTTP 200 in 70 ms total; given
  ARLE's 1.2 s TTFT at 4096-token prompts, the HTTP layer overhead
  is ~5% even at long prompts. Layer 1 is end-to-end fast.

## Action items (post-bench)

1. **Profile single-stream decode (c=1, 128-in/2048-out) on ARLE
   vs mlx-lm under matched conditions.** Use metal capture or `mlx
   instruments` to identify the dominant per-token cost. Adds an
   M_e.0 plan entry.
2. **Update the morning gap-analysis doc** with the corrected gap
   composition (TTFT win + ITL gap + scaling collapse — three
   distinct axes, not one).
3. **Keep M_e.1 plan as written** for the left-pad collapse track,
   but explicitly note that it does NOT address the per-token decode
   gap; that's a separate workstream.

## Reproduce

```bash
# ARLE side
RUST_LOG=warn target/release/metal_serve \
  --model-path models/Qwen3.5-0.8B-MLX-4bit \
  --port 8000 --bind 127.0.0.1 \
  --max-running-requests 4 --max-batch-tokens 512

# mlx-lm side
mlx_lm.server --model models/Qwen3.5-0.8B-MLX-4bit \
  --port 8001 --host 127.0.0.1

# Bench (run sequentially against each)
PATH=$HOME/.local/bin:$PATH ./scripts/bench_guidellm.sh metal-c4-arle-true \
  --concurrencies 4 --max-seconds 30 \
  --data 'prompt_tokens=4096,output_tokens=256' \
  --target http://localhost:8000 \
  --model Qwen3.5-0.8B-MLX-4bit \
  --processor /path/to/code/agent-infer/models/Qwen3.5-0.8B-MLX-4bit

PATH=$HOME/.local/bin:$PATH ./scripts/bench_guidellm.sh metal-c4-mlxlm \
  --concurrencies 4 --max-seconds 30 \
  --data 'prompt_tokens=4096,output_tokens=256' \
  --target http://localhost:8001 \
  --model /path/to/code/agent-infer/models/Qwen3.5-0.8B-MLX-4bit \
  --processor /path/to/code/agent-infer/models/Qwen3.5-0.8B-MLX-4bit
```

Raw artefacts: `bench-output/2026-05-07-metal-c4-{arle-true,mlxlm}/`

## Cross-references

- Previous c-sweep (the run whose narrative this corrects):
  [`2026-05-07-bench-guidellm-metal-c-sweep-m4pro.md`](2026-05-07-bench-guidellm-metal-c-sweep-m4pro.md)
- Plan whose §3 commit-4 acceptance numbers must be updated (paged-KV
  closes left-pad, NOT the per-token gap):
  [`docs/plans/M_e1-metal-paged-kv-hot-path.md`](../../plans/M_e1-metal-paged-kv-hot-path.md)
- Master gap analysis to retroactively reconcile:
  [`docs/projects/2026-05-07-metal-world-first-gap-analysis.md`](../../projects/2026-05-07-metal-world-first-gap-analysis.md)
- Decode-priority scheduler invariant that powers the TTFT win:
  test `decode_priority_holds_under_c4_mixed_traffic` in
  `infer/src/backend/metal/scheduler.rs` (locked tick 2, commit
  `199a0a8`/rebased).
