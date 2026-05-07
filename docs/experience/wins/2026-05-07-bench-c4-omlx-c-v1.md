# Bench — c=4 oMLX-C v1 (multi-step async pipelining) — 2026-05-07

## Goal

Apply the oMLX-C design from
[`docs/plans/M_e1-omlx-c-multi-step-pipelining.md`](../../plans/M_e1-omlx-c-multi-step-pipelining.md)
to drop `sample_us` from 4.4 ms (67.4% of step time) to ≤ 1.5 ms by
overlapping prev-step sampled-token host readback with current-step
forward+sample kernels. Mirrors mlx-lm `GenerationBatch._step`
(`mlx_lm/generate.py:1320-1378`).

## Hypothesis

- `sample_us` drops to ≤ 1.5 ms (the `async_eval_kickoff` floor)
- `total_us` drops below 5 ms p50 at c=4

## Params

- Binary: `target/release/metal_serve` rebuilt at this commit
- Model: `models/Qwen3.5-0.8B-MLX-4bit`
- Env: `INFER_PHASE_TIMING=1 INFER_OMLX_C=1 RUST_LOG=info`
- Same workload as
  [`2026-05-07-bench-c4-phase-timing-baseline.md`](2026-05-07-bench-c4-phase-timing-baseline.md):
  4 concurrent POSTs, 64 max_tokens, temperature=0.0,
  `--max-running-requests 8`.

## Results — at batch=4 (c=4 hot path), 25 pipelined steps

| Phase | baseline μs | oMLX-C v1 μs | Δ |
|---|---:|---:|---:|
| `prep` (host MlxArray builds + take()) | 159 | **4233** | **+4074** ⚠️ |
| `build_graph` (`step_batch_packed`) | 421 | 463 | +42 |
| `async_eval_kickoff` | 1563 | 1614 | +51 |
| `sample` (eval(prev_sampled) + extract + record) | 4438 | **0** | **−4438** ✅ |
| `pool_dual_write` | 0 | 0 | 0 |
| **`total`** | **6583** | **6312** | **−271 (−4.1%)** |

Path probe confirmed firing:
```
metal_path_probe: oMLX-C pipelined step FIRED (decode_qwen35_packed_batch_pipelined)
```

## Problems / observations

1. **The pipelining DID work where it was supposed to.** `sample` phase
   (eval(prev_sampled) + extract + record) collapsed from 4.4 ms to
   essentially zero. This validates the core thesis: by the time a
   pipelined call's `eval(prev_sampled)` runs, the GPU has finished
   the previous step's forward+sample kernels (kicked off at end of
   prev call) — eval is essentially a host-side memcpy.

2. **But `prep` regressed by 4 ms.** The work didn't disappear; it
   migrated to the start of the pipelined function. The most likely
   cause: `step_batch_packed`'s C++ FFI body has an internal
   `mlx::eval` barrier (e.g. for GPU memory allocation, embedding
   lookup needing the integer token value, or an internal command
   buffer commit). When the input MlxArray is the previously-async-
   eval'd `prev_sampled`, the first read of its values forces eval
   completion of the prev-step chain. THIS chain-completion was
   previously charged to the `sample` phase (because the input was a
   from_slice_i32 host-built array, no GPU dependency, but the
   sample's eval forced the WHOLE step's kernel chain to complete).
   With pipelining, the chain completion happens earlier in the
   function — at the input boundary — and is captured in `prep`.

3. **Net effect: 4% total ITL improvement.** Real, but far from the
   3× I'd hoped for in the design plan. The sample-phase elimination
   is genuine win; the prep regression is a constant-time cost that
   can only be removed by getting the C++-side eval barrier out of
   the input boundary.

4. **No correctness issues.** All 644 infer unit tests pass. Smoke
   produces correct generations.

## Learnings

1. **The eval barrier is INSIDE step_batch_packed**, not in the
   sample phase. Pipelining moves where the wait happens but the
   total wait time is dominated by the C++ FFI's internal eval.
   Next step is to instrument the C++ side
   (`mlx_qwen35_model.cpp:qwen35_compiled_step_batch_packed`) to
   find the internal sync barrier.

2. **The host-side overlap CAN work** — between async_eval at end
   of call N and the input access at start of call N+1, the host
   does:
   - Scheduler's process_token + finish_or_requeue (~hundreds of μs)
   - try_decode_qwen35_packed_batch dispatch (~tens of μs)
   - This call's prep until the input access
   That host time was enough to finish the GPU work in ~most of the
   `sample` phase budget. So eval(prev_sampled) is fast — the GPU
   is already done.

3. **The C++-internal eval barrier is the new bottleneck.** Likely
   suspects (need to verify):
   - `mx::compile`'d functions inside the model forward (some steps
     may force eval at function boundaries)
   - `KVCache` slice_update internals
   - Embedding lookup that may need the integer value of the input
     token

4. **Rule of thumb confirmed:** measure phase timing AFTER landing a
   change, not before, because the work *moves* rather than
   *disappears* when you optimize one path.

## What worked / Rule

- Path probe (`OMLX_C_PIPELINE_PROBE`) confirmed the new path fires
  exactly as designed.
- Feature gate (`INFER_OMLX_C=1`) means production sees no change
  unless flipped on.
- Phase timing with separate `metal_phase_timing_pipelined` tag
  isolated this branch from baseline measurements.
- Numerical correctness preserved (tests + smoke).

## Rule

When pipelining work across function-call boundaries, ALWAYS
re-instrument both halves of the pipeline. The sync wait may move
inside the FFI boundary you assumed was free.

## Next

- **oMLX-C v2 — investigate the C++-side eval barrier.** Add C++
  high-resolution timing inside `qwen35_compiled_step_batch_packed`
  to identify which sub-call costs the 4 ms. If it's
  `mx::compile`'d code, consider seeding the cache or using
  `shapeless=true` more aggressively.
- **Alternative**: sink the loop into C++ entirely (mirror
  `mlx_qwen35_model.cpp:3231-3325` c=1 pattern at c≥2). Pros: full
  control over eval boundaries. Cons: scheduler reactivity. Move
  to next-tick design discussion.
- Keep oMLX-C v1 as DEFAULT-OFF feature gate. Flip ON when v2
  closes the prep regression.

## References

- Baseline:
  [`2026-05-07-bench-c4-phase-timing-baseline.md`](2026-05-07-bench-c4-phase-timing-baseline.md)
- Design:
  [`docs/plans/M_e1-omlx-c-multi-step-pipelining.md`](../../plans/M_e1-omlx-c-multi-step-pipelining.md)
- Pattern source: mlx-lm `generate.py:1320-1378`
- Implementation:
  [`infer/src/backend/metal/request_state.rs`](../../../infer/src/backend/metal/request_state.rs)
  (`decode_qwen35_packed_batch_pipelined`)
