# 2026-05-07 · Cross-request prefill batching IS working — TTFT gap is elsewhere

## Goal

Test the M3.8 hypothesis: does ARLE actually batch prefill across
requests, or does the scheduler send one request's chunk per step
serially?

If batching works: M3.8's "implement cross-request prefill" plan
is unnecessary; the TTFT gap to vLLM lives elsewhere.
If not: M3.8 implementation work is justified.

## Hypothesis

After source-code survey
([`2530ad6`](../../plans/M3.8-batched-prefill-cross-request.md)
correction), code paths in `qwen3/prefill.rs` (lines 295, 415, 640)
appear to support batched prefill, and `prefill_max_requests`
defaults to None (unbounded). So batching SHOULD happen
structurally — but B.1.2's failed TTFT result and longctx 8k/c=4
slowness suggest it might not be filling the batch width.

## Command

ARLE server with a configuration designed to expose batched
prefill: c=8 capacity, single-chunk requests (prompt 2048 = 1
full chunk):

```bash
target/release/infer --port 8000 --max-seq-len 5120 --num-slots 16 \
  --max-prefill-tokens 16384
```

Bench:
```bash
guidellm benchmark run --target http://localhost:8000 \
  --profile concurrent --rate 8 \
  --data 'prompt_tokens=2048,...,output_tokens=64,...' \
  --max-seconds 30 --warmup 5 --random-seed 20260416
```

## Results

### Server log step breakdowns (the actual evidence)

```
Step  1: plan=prefill prefill=471568us batch=1 active=1   (only req 0 admitted yet)
Step  2: plan=prefill prefill=252160us batch=8 active=8   (ALL 8 BATCHED, 252ms total)
Step 3+: plan=split   prefill=787-789ms batch=8 active=8  (mixed prefill+decode steps)
```

**Step 2 is the breakthrough**: 8 requests' prefill processed in
ONE step. 8 reqs × 2048 tokens = **16,384 tokens of prefill in
252 ms** = **15.4 µs/token batched prefill**.

### Per-token comparison

| Backend | Workload | per-token prefill |
|---|---|---:|
| **ARLE batched prefill (this bench)** | c=8 × 2048 tok | **15.4 µs/tok** |
| vLLM aggregated TTFT/8k | c=4 × 8192 tok | 290 µs/tok |
| ARLE (longctx 8k/c=4 measured TTFT) | c=4 × 8192 tok | ~600 µs/tok |

**ARLE batched prefill is ~19× faster per-token than vLLM's
aggregated number when the batch is fully filled** at single-chunk
shape. The question is why ARLE longctx 8k (multi-chunk) shows
600 µs/tok.

### Mixed-plan tax — the next bottleneck signal

After all 8 prefills complete, the scheduler runs `plan=split`
steps that mix decode (8 active rows) with prefill (the
just-completed prompts now in decode). These steps take ~787 ms
each — **3× slower than pure prefill at the same batch=8**.

Pure prefill at batch=8 = 252 ms.
Split at batch=8 = 787 ms.

The 535 ms (3.1× factor) overhead comes from the mixed-step
machinery itself. This is consistent with the M3.6 Phase 1 trace
finding that `step_mixed_kernel_launch` averages 5.2 ms per
launch (vs 1.0 ms for pure decode).

## Headline finding

**Cross-request prefill batching works.** When the scheduler can
fill the batch width, it does — 8 requests × 2048 tokens batch
into one 252 ms prefill step.

**M3.8's implementation tasks (Phase 1.1 / 1.2) are NOT needed.**
The cross-request packing in `qwen3/prefill.rs` is exercised and
working as designed.

The TTFT gap to vLLM at longctx 8k/c=4 (ARLE 4961 vs vLLM 2367)
must come from one of:

1. **Multi-chunk serialization** within a single request:
   8k prompt = 4 chunks. Does ARLE batch chunk-1 of 4 requests
   into one step, then chunk-2 of all 4 into the next? Or does
   it process all 4 chunks of req 0 serially before starting
   req 1? Need to test with longctx 4k/c=4 (single-chunk
   requests at long-ctx) vs longctx 8k/c=4 (4-chunk requests).

2. **Mixed-plan tax** at split steps:
   787 ms / 252 ms = 3.1× slower per step. With 8k prompts × 4
   chunks × c=4, even if batched chunk-by-chunk, the steps
   transition to split as soon as the first chunks complete and
   start emitting decode tokens. The mixed-step overhead then
   compounds.

3. **vLLM doesn't have ARLE's mixed-step problem** — vLLM's
   continuous batching design lets it stay in pure-prefill mode
   longer (until enough requests are decode-ready) AND uses
   chunked prefill for very long prompts so each request's
   prefill is broken into smaller atomic units that mix more
   gracefully with decode.

## Implications

| Track | Pre-experiment status | Post-experiment update |
|---|---|---|
| M3.8 Phase 1 (cross-request prefill plumbing) | proposed | **CANCEL — already implemented and working** |
| M3.8 Phase 0 (config bump) | proposed | not needed — `prefill_max_requests=None` already unbounded |
| **NEW: M3.8 v2 — fix mixed-step tax** | n/a | **HIGH PRIORITY** — split-plan is 3× slower than pure prefill at same batch; investigate why and reduce |
| **NEW: longctx 4k/c=4 bench** | n/a | medium — characterize multi-chunk vs single-chunk at long-ctx to isolate (1) vs (2) |
| M_b.2 (FP8 prefill TileLang) | proposed | LOWER priority — kernel-axis is not the gap (15.4 µs/tok shows kernel is fast when batched) |
| F4-Big | optional | optional |

## Bench Status

- Bench artifact: `bench-output/2026-05-07-prefill-batching-test/`
  (gitignored).
- Server log captured the breakdowns (`/tmp/prefill-test.log`).
- nsys trace skipped (server log gave the answer directly; nsys
  has been unreliable at low-conc workloads in this iteration).

## Rule

- **Source-code survey + targeted log inspection > nsys trace**
  for verifying scheduler behavior at low-concurrency. The
  step-breakdown log lines have all the data needed; nsys 2025.6
  doesn't reliably populate at c=4 prefill-heavy workloads.
- **A single direct experiment beats two plan iterations**.
  M3.8 v1 wrote a 156-line plan to add cross-request batching;
  this 30-min experiment shows it already works. The 156-line
  plan was wasted (corrected in 2530ad6 + now this entry).
  Future: when a hypothesis can be tested in <60 min of bench
  + log, do that BEFORE writing the plan.
- **Mixed prefill+decode (`split` plan) has a 3× efficiency tax**.
  This is a NEW finding worth its own milestone.
