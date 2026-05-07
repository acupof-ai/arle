# Bench — c=4 phase-timing baseline (Task #16) — 2026-05-07

## Goal

Profile per-phase wall-time inside `decode_qwen35_packed_batch` (the
real c≥2 hot path per
[2026-05-07 audit](../errors/2026-05-07-three-layer-audit-miss-c4-real-path-is-packed-batch.md))
to falsify or confirm the hypothesis from
[`docs/research/2026-05-07-mlx-ecosystem-survey-c4-itl-gap.md`](../../research/2026-05-07-mlx-ecosystem-survey-c4-itl-gap.md):
sync `eval(&[&sampled])` immediately after `async_eval(eval_refs)`
within the same step is the dominant cost at c≥2, scaling with batch.

## Hypothesis

- `sample_us` (sample build + sync eval wait) > 50% of step time at c=4
- This justifies oMLX-C (multi-step pipelining: overlap sample readback
  with next-step forward, mirroring mlx-lm `GenerationBatch._step` —
  `mlx_lm/generate.py:1320-1378`)

## Params

- Binary: `target/release/metal_serve` rebuilt at commit `7a54050`
- Model: `models/Qwen3.5-0.8B-MLX-4bit` (Qwen3.5-0.8B 4-bit MLX)
- `--max-running-requests 8` (slot cap; matters because default 4 caps
  the c=4 working set)
- Env: `INFER_PHASE_TIMING=1` `RUST_LOG=info`
- Workload: 4 concurrent POSTs to `/v1/chat/completions`, prompt
  "Count from 1 to 50 with a one-word adjective for each number."
  `max_tokens=64 temperature=0.0`. Same prompt body each request,
  varying suffix to defeat any prefix caching.
- Driver: `/tmp/c4_smoke.sh` (4× backgrounded curl + wait)

## Env

- macOS Darwin 25.3.0 / Apple Silicon (binary built on this host)
- MLX 0.31.1 (per `crates/mlx-sys/CMakeLists.txt`)
- No `INFER_DETERMINISTIC`, no `INFER_KV_POOL` (pool-dual-write path
  inactive — `pool_dual_write_us=0` confirms)

## Results — at batch=4 (c=4 hot path), 23 steps

| Phase | avg μs | p50 μs | p99 μs | % of step |
|---|---:|---:|---:|---:|
| `prep` (host MlxArray builds) | 159 | 154 | 200 | 2.4% |
| `build_graph` (`step_batch_packed`) | 421 | 416 | 491 | 6.4% |
| **`async_eval_kickoff`** | **1563** | **1523** | 2487 | **23.7%** |
| **`sample`** (sample + sync eval wait) | **4438** | **4298** | 8361 | **67.4%** |
| `pool_dual_write` | 0 | 0 | 0 | 0% |
| **`total`** | **6583** | **6409** | 10281 | 100% |

→ avg ITL/token at c=4 ≈ 6.6 ms / 152 tok/s.

(NOT 19 ms as cited in earlier audit notes — that earlier number must
have come from an older binary or different workload. Re-baselined
here at commit `7a54050`. Current ARLE c=4 ITL is much closer to
mlx-lm 7 ms than the audit chain feared.)

## Problems / observations

1. **`sample` dominates** (67.4%, 4.3 ms). Confirms hypothesis. Within
   `sample`: `gpu_sample_token_batched(&logits, …)` builds an argmax
   graph, then `eval(&[&sampled])` blocks for both the forward graph
   AND the sample to finish.
2. **`async_eval_kickoff` is non-trivial** (23.7%, 1.5 ms). MLX's
   `async_eval` should return immediately after enqueueing — the
   1.5 ms here suggests it's also doing graph compilation /
   command-buffer encoding for the just-built `eval_refs` (logits +
   `packed_kv_flat[]` slice_updates + `packed_gdr_flat[]`). Not free.
3. **`build_graph`** (6.4%, 421 μs) is the C++ FFI call into
   `step_batch_packed`. Pure graph build, no GPU kernels yet.
4. **`pool_dual_write` is 0** because the smoke run did not exercise
   the kv_pool path (no `INFER_KV_POOL` env). When a future bench
   enables the pool, this phase's contribution can be measured
   separately.

Combined `sample + async_kick = 6001 μs = 91.1%` of every step is
host-blocked or graph-encoding work. **Multi-step pipelining can
overlap most of this with the next step's forward**, not eliminate it.

## Learnings

1. **Profile-first discipline pays.** The 19 ms claim from earlier
   tickets was wrong; the actual c=4 ITL is 6.6 ms — within shouting
   distance of mlx-lm. Without this profile, oMLX-C would have aimed
   at a phantom 12 ms gap and overshot.
2. **The remaining gap to mlx-lm 7 ms is small** (≈ −0.4 ms / step
   below today's 6.6 ms). oMLX-C's value is therefore not "close a
   12 ms gap" but "drop the host-block tax so c=8 / c=16 scale
   linearly instead of paying 4.4 ms × N tax per step."
3. **Re-aim the goal**: instead of "match mlx-lm at c=4", target
   "scale flat to c=16+" — that's where multi-step pipelining
   compounds (the 4.4 ms sample wait grows with batch when batched
   sampler has more rows to extract via .item()).
4. The 1.5 ms `async_eval_kickoff` is a separate finding — worth a
   follow-up to understand why MLX's async_eval is doing 1.5 ms of
   work; possibly a custom-op graph optimization step we can amortize.

## What worked / Rule

- `INFER_PHASE_TIMING=1` env-gated logger from
  [`request_state.rs:2635-2650`](../../infer/src/backend/metal/request_state.rs)
  delivered the data with one `metal_phase_timing` line per decode
  step — exactly the format
  `feedback_path_probe_before_perf_claim.md` calls for. Bench grep:
  `grep "metal_phase_timing" /tmp/metal_serve_phase_timing.log` →
  parse via Python.
- **Rule for next perf change**: any oMLX-C diff or kernel cutover
  re-runs this exact smoke and produces a delta entry citing this
  baseline.

## Next

- Implement oMLX-C (Task #22) — multi-step async pipelining via
  pre-step sampled-MlxArray persistence on `Qwen35PackedDecodeBatch`,
  mirroring mlx-lm `GenerationBatch._next_tokens`/`_current_tokens`
  pattern. Target: drop `sample` from 4.4 ms → ≤ 1.5 ms (the
  `async_eval_kickoff` floor).
- Investigate `async_eval_kickoff` 1.5 ms — separate MLX-side work
  unit if confirmed amortizable.

## References

- Phase logger landing commit:
  [`7a54050 feat(metal): INFER_PHASE_TIMING=1 …`](../../../infer/src/backend/metal/request_state.rs)
- Hypothesis source:
  [`docs/research/2026-05-07-mlx-ecosystem-survey-c4-itl-gap.md`](../../research/2026-05-07-mlx-ecosystem-survey-c4-itl-gap.md)
- mlx-lm pattern (subagent-verified):
  `mlx_lm/generate.py:1320-1378` (`GenerationBatch._step`)
- Driver path-probe audit:
  [`2026-05-07-three-layer-audit-miss-c4-real-path-is-packed-batch.md`](../errors/2026-05-07-three-layer-audit-miss-c4-real-path-is-packed-batch.md)
