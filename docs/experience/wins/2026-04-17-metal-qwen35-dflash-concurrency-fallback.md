# Qwen3.5 DFlash auto-fallback on concurrent decode

## Context

Yesterday's DFlash correctness snapshot
([`2026-04-17-metal-qwen35-dflash-correctness-bench.md`](2026-04-17-metal-qwen35-dflash-correctness-bench.md))
recorded a 10× regression for concurrent DFlash workloads:

| Mode          | Baseline (no DFlash) | DFlash |
|---------------|---------------------|--------|
| 4× concurrent | 155.6 tok/s         | 15.2 tok/s (serial) |
| 8× concurrent | 158.1 tok/s         | 15.2 tok/s (serial) |

Root cause: each DFlash request's 16-token verify block (~230ms, ~28% accept)
runs through `execute_decode_single`, fully serializing the scheduler. With
4–8 concurrent sessions every request was queued behind every other request.

## What worked

Permanent DFlash downgrade on concurrency. Two gates:

1. **Admission-time** (`runtime.rs::admit_request`): new request skips
   DFlash init when `active.is_empty()` is false. Saves the full-prompt
   single-shot DFlash prefill cost — the request would have been downgraded
   on its first decode step anyway.

2. **Per-step** (`runtime.rs::execute_decode_batch`): when `open.len() >= 2`,
   call `MetalRequestState::disable_dflash()` on every row before the
   DFlash/non-DFlash partition. The partition then empties naturally and
   everyone joins the packed batch.

The downgrade is **one-way**. DFlash's `target_hidden` capture needs a fresh
hidden-state capture during prefill — once packed decode advances the KV
cache without capturing, DFlash can't resume. That's fine because packed
decode is strictly faster whenever concurrency ≥ 2.

Per-request draft KV cache (`ContiguousKvState` for the DFlash draft model)
is dropped at `disable_dflash()`, freeing GPU memory. Main target KV
(`cpp_state.kv_flat`) is untouched — it's the same buffer packed decode uses.

## Bench

**Hardware.** M4 Max 40-core GPU, 64GB unified memory, macOS 25.3.
**Model.** `mlx-community/Qwen3.5-4B-MLX-4bit` + `z-lab/Qwen3.5-4B-DFlash`.
**Server.** `./target/release/metal_serve --dflash-draft-model z-lab/Qwen3.5-4B-DFlash --port 8000`.
**Prompt.** `"Write a 200-word story about a robot chef who enters a cooking competition."`
**Decode budget.** 256 `max_tokens`, temperature 0.

| Workload | tokens | elapsed | throughput | vs baseline |
|----------|--------|---------|------------|-------------|
| single req (run 1) | 256 | 19.53s | **13.1 tok/s** | 17% (DFlash still active) |
| single req (run 2) | 256 | 20.22s | **12.7 tok/s** | 17% |
| single req (run 3) | 256 | 19.49s | **13.1 tok/s** | 17% |
| 4× concurrent     | 1024 | 7.45s | **137.4 tok/s** | 88% |
| 8× concurrent     | 2048 | 15.92s | **128.7 tok/s** | 81% |

Single-session unchanged — DFlash still runs, still ~28% accept, still 5×
slower than baseline. That's Track 2's problem.

Concurrent 4× and 8× jumped **9× and 8.5×** respectively versus yesterday's
serial DFlash. The ~15% remaining gap vs pure baseline is the first
request's DFlash prefill + 1-2 speculative blocks before the second request
arrives and triggers the downgrade.

## Read

- **No concurrent regression.** DFlash sessions no longer block each other;
  under load they all run through packed decode at batched throughput.
- **First-request locality.** When admitted alone, a request still gets
  DFlash — preserved for the future where DFlash acceptance actually pays.
- **No mid-stream recovery.** Once downgraded, a session stays on packed
  decode even if others finish and leave it solo. Acceptable because
  re-initializing DFlash would require a fresh prefill with hidden capture,
  which is slower than just continuing to generate.

## Rule

**DFlash is a single-session optimization — never let it block a batch.**
Speculative decode pays off only when (a) acceptance is high enough and
(b) no other session is waiting. When concurrency arrives, fall back to
the fastest decode you have (packed batch for Metal Qwen3.5), and don't
try to toggle back. Mid-stream re-init is more expensive than the gains.

## Follow-ups

- **Track 2**: DFlash acceptance ~28% → ≥50%. Single-session DFlash is
  still a 5× regression vs baseline because the draft model diverges early.
  Diff draft-vs-target logit trajectories on a shared prefix to find the
  divergence point.
- **Track 3**: Batched DFlash verify across sessions (only worth doing
  once Track 2 proves DFlash pays single-session).

## Raw commands

```bash
./target/release/metal_serve --model-path mlx-community/Qwen3.5-4B-MLX-4bit \
  --dflash-draft-model z-lab/Qwen3.5-4B-DFlash --port 8000

bash /tmp/bench_q35.sh "DFlash-enabled server (Track-1 downgrade)"
```
