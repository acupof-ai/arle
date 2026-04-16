# Qwen3.5 DFlash correctness + varlen batch fixes — bench snapshot

## Context

Three correctness fixes landed together to unblock Qwen3.5 DFlash and repair
concurrent decode on Metal:

1. **DFlash tape correctness.** Per-step drain of GDR tapes + sticky-state
   reset in C++ step/prefill entry + bf16 cast of `g`/`k` at tape record time
   (see commit `4db4fe9`).
2. **Varlen attn-mask dtype.** MLX ≥ 0.32 rejects f32 additive masks on bf16
   SDPA — `build_varlen_decode_mask` now emits bf16 directly.
3. **Packed-batch KV grow.** `extend_kv_cache` hardcoded `batch=1` in its
   zero-pad, which panicked as soon as the packed-decode cache (batch>1) hit
   its capacity boundary. Batch dim now inherited from the existing cache.

**Hardware.** M4 Max 40-core GPU, 64GB unified memory, macOS 25.3.
**Model.** `mlx-community/Qwen3.5-4B-MLX-4bit` (target) + `z-lab/Qwen3.5-4B-DFlash` (draft).
**Server.** `./target/release/metal_serve`, default flags, `--port 8000`.
**Prompt.** `"Write a 200-word story about a robot chef who enters a cooking competition."`
**Decode budget.** 256 `max_tokens`, temperature 0 (greedy).
**Measurement.** Wall-clock around `curl`, `jq` for `usage.completion_tokens`,
repeated 3× for single + N-way parallel curl aggregation.

## Baseline (no DFlash)

| Workload | tokens | elapsed | throughput |
|----------|--------|---------|------------|
| single req (run 1) | 256 | 3.45s | **74.2 tok/s** |
| single req (run 2) | 256 | 3.42s | **74.9 tok/s** |
| single req (run 3) | 256 | 3.42s | **74.8 tok/s** |
| 4× concurrent     | 1024 | 6.58s | **155.6 tok/s** (2.1× over single) |
| 8× concurrent     | 2048 | 12.96s | **158.1 tok/s** (plateau) |

Concurrent scaling holds to ~4-wide, flat beyond — consistent with the packed
batch decode path doing real work across rows.

## DFlash (block_size=16, draft Qwen3.5-4B-DFlash)

| Workload | tokens | elapsed | throughput |
|----------|--------|---------|------------|
| single req (run 1) | 256 | 15.26s | **16.8 tok/s** |
| single req (run 2) | 256 | 16.79s | **15.2 tok/s** |
| single req (run 3) | 256 | 16.81s | **15.2 tok/s** |
| 4× concurrent     | 1024 | 67.41s | **15.2 tok/s** (serial — no batching) |
| 8× concurrent     | 2048 | 134.32s | **15.2 tok/s** (serial) |

DFlash blocks logged ≈ 345 for ~1536 generated tokens → ~4.45 accepted per
block → acceptance ≈ **28%**. Reference target was 58% (see `dflash_mlx`);
our verify_16 latency per block is ~230ms.

## Read

- **Correctness:** ✅ DFlash produces coherent, deterministic output matching
  baseline on the same greedy prompt.
- **Baseline throughput:** unchanged (~75 tok/s single, ~155 tok/s 4-way).
- **Concurrent decode regression fixed:** `extend_kv_cache` batch-dim bug
  crashed the scheduler the moment the packed cache rolled past a
  `KV_CACHE_CHUNK` boundary. Now stable under 4× and 8× concurrency.
- **DFlash speedup:** **regression — 5× slower than baseline, no concurrency
  benefit.** Two causes stacked:
  1. Low acceptance (~28% vs reference 58%) — indicates the draft model is
     disagreeing often. Likely hidden-state capture or draft-context
     alignment.
  2. verify_16 on a 4-bit target model is ~4.6× verify_1 in the reference
     measurements; without acceptance ≥ 50% the 16-token block doesn't pay
     for itself.
  3. DFlash requests are **not batched across concurrent sessions** — each
     runs serially through `qwen35_dflash_speculative_block`. The packed
     batch path (`execute_qwen35_packed_decode_batch`) only covers non-DFlash
     decode.

## Follow-ups (NOT in this commit)

- Acceptance investigation: compare logit trajectories between DFlash draft
  and target on the same prefix to find where the draft is diverging early.
- DFlash concurrency: teach the scheduler to pack multiple DFlash requests
  into one verify forward (batch-axis packing over the 16-token block).
- Alternative: run DFlash only when a single request is in flight,
  auto-fallback to packed decode when ≥2 sessions queue up.

## Rule

**Ship correctness first, speed second.** DFlash output is right now, the
scheduler no longer crashes under concurrency, CI is green — this is the
stable floor we measure the next round of tuning against. Do not claim a
DFlash "speedup win" until acceptance > 50% on the target eval set.

## Raw commands

```bash
./target/release/metal_serve --model-path mlx-community/Qwen3.5-4B-MLX-4bit \
  [--dflash-draft-model z-lab/Qwen3.5-4B-DFlash] --port 8000

curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" -d '{
    "model":"qwen35",
    "messages":[{"role":"user","content":"Write a 200-word story about a robot chef who enters a cooking competition."}],
    "max_tokens":256,"temperature":0
  }' | jq -r '.usage.completion_tokens'
```
