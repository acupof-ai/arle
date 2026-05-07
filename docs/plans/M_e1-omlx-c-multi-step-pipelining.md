# M_e.1 oMLX-C — multi-step async pipelining for c≥2 packed-batch decode

**Owner:** ckl · **Status:** designed 2026-05-07, awaiting implementation tick
**Track:** Metal / scheduler · **Blocked by:** ✅ Task #16 baseline captured

## Goal

Drop `sample_us` (4.4 ms p50, 67.4% of step time at c=4) to ≤ 1.5 ms by
overlapping sampled-token host readback with next-step forward graph
build/dispatch. Apply mlx-lm's `GenerationBatch._step` pattern
(`mlx_lm/generate.py:1320-1378`) to ARLE's `decode_qwen35_packed_batch`
([`request_state.rs:2417`](../../infer/src/backend/metal/request_state.rs)).

## Baseline (the floor we are trying to lift)

From [`docs/experience/wins/2026-05-07-bench-c4-phase-timing-baseline.md`](../experience/wins/2026-05-07-bench-c4-phase-timing-baseline.md):

| Phase | avg | % |
|---|---:|---:|
| prep | 159μs | 2.4% |
| build_graph | 421μs | 6.4% |
| async_eval_kickoff | 1563μs | 23.7% |
| **sample (sample + sync eval)** | **4438μs** | **67.4%** |
| total | 6583μs | 100% |

## The mlx-lm pattern (verified)

```python
def _step(self):
    self._current_tokens = self._next_tokens          # promote
    inputs = self._current_tokens                     # this step's input = last sample
    logits = self.model(inputs[:, None], cache=...)
    sampled = sampler(logits)
    self._next_tokens = sampled                       # stash for next step
    mx.async_eval(self._next_tokens, ...)             # kick off NEXT step's host data
    mx.eval(inputs, self._current_logprobs)           # wait for THIS step's input
    return inputs.tolist()                            # return last-step's host ints
```

Each `_step` call:
- Forward-passes step K using step K-1's sampled tokens (NO from_slice_i32 host conversion)
- Samples step K → MlxArray
- async_eval the new sample (so next call's eval has data ready)
- eval the inputs (step K-1's sample, async_eval'd by previous call) — host readback runs while step K's forward kernels are still on the GPU stream
- Returns step K-1's tokens

## ARLE design

### Add to `Qwen35PackedDecodeBatch` ([`request_state.rs:773`](../../infer/src/backend/metal/request_state.rs))

```rust
pub(crate) struct Qwen35PackedDecodeBatch<'a> {
    /// ...existing fields...
    /// oMLX-C: previous step's sampled-token MlxArray, kicked off via
    /// `async_eval` at the end of the previous decode step. The current
    /// step uses this directly as `step_batch_packed`'s input (skipping
    /// host-side from_slice_i32) AND eval()s it after firing the new
    /// step's `async_eval` so host readback overlaps with the new
    /// step's forward kernels. None until the first decode step has
    /// completed.
    prev_sampled: Option<MlxArray>,
}
```

`retain_rows()` MUST also gather along axis 0 of `prev_sampled` to keep
shape aligned when the batch shrinks.

### Refactor `decode_qwen35_packed_batch`

Pseudocode for the new control flow:

```rust
fn decode_qwen35_packed_batch(...) -> Result<Vec<u32>> {
    let phase_timing = ...; let t0 = ...;

    // (1) Determine THIS step's input array.
    let token_arr = match batch.prev_sampled.take() {
        Some(prev) => prev,                          // pipeline path
        None => {                                    // bootstrap: convert state.last_token
            let toks: Vec<i32> = states.iter().map(|s| s.last_token? as i32).collect();
            MlxArray::from_slice_i32(&toks, &[states.len() as i32])
        }
    };

    // (2) Forward + KV update (UNCHANGED).
    let logits = cpp_model.step_batch_packed(&token_arr, ...);

    // (3) Sample → MlxArray (NOT host-extracted yet).
    let new_sampled = if can_batch_sample {
        gpu_sample_token_batched(&logits, params)
    } else {
        // fallback per-row sample, concat into one [B] MlxArray
        ...
    };

    // (4) Kick off next step's host data via async_eval.
    let mut eval_refs: Vec<&MlxArray> = Vec::with_capacity(/* +1 */);
    eval_refs.push(&new_sampled);
    eval_refs.push(&logits);
    eval_refs.extend(batch.packed_kv_flat.iter());
    eval_refs.extend(batch.packed_gdr_flat.iter());
    async_eval(&eval_refs);

    // (5) Eval THIS step's input (= prev step's sampled OR bootstrap arr)
    //     to extract host int32s.
    eval(&[&token_arr]);
    let extracted: Vec<u32> = token_arr.as_slice_i32().iter().map(|&t| t as u32).collect();

    // (6) Record state updates from `extracted`.
    batch.batch_cache_len += 1;
    for (row_idx, (state, token)) in states.iter_mut().zip(&extracted).enumerate() {
        state.driver.cache_len = batch.batch_cache_len - batch.left_padding[row_idx];
        state.driver.kv_capacity = batch.kv_capacity;
        state.record_sampled_token(*token)?;
    }

    // (7) Stash new_sampled for next call's pipeline path.
    batch.prev_sampled = Some(new_sampled);

    // (8) Pool dual-write (UNCHANGED) — operates on packed_kv_flat which
    //     was already async_eval'd above. flush() forces the slice_update
    //     chain so it doesn't grow.
    if any state has kv_pool { ... pool.write_kv(...); pool.flush(); }

    // Phase timing log (UNCHANGED).
    ...

    Ok(extracted)
}
```

### What the contract looks like to the caller

**It does NOT change.** Each call still returns "the tokens generated by
this step's forward pass" — the only difference is *what arithmetic
happens around the `eval()`*. The pipelining is purely internal: the
tokens we return are what the CURRENT step's forward+sample produced,
extracted via `eval(&[&token_arr])`. The "magic" is that
`token_arr = prev_sampled` was already async_eval'd at the end of the
previous call, so by the time `eval()` runs in step K, the GPU work
for step K-1's sample is done and `eval()` returns nearly immediately
— the wall-time saved is the difference between (a) waiting for both
"sample-graph build + sample-kernel + host-readback" inline and (b)
just "host-readback" when the rest already finished overlapped with
step K's forward graph build.

Wait — that contract has a subtle issue: **step K's forward consumed
`token_arr` (= step K-1's sample) as input**. So what step K returns
IS step K-1's tokens. The TOKEN that K's forward+sample produced is
in `new_sampled`, not yet host-extracted, and will be returned by
step K+1's call.

**Resolution:** the caller (scheduler) must accept that the function
returns "the tokens used as input THIS step (which are step K-1's
sampled)" rather than "tokens just sampled this step". Functionally:

- Bootstrap call: `token_arr` = `state.last_token` (prefill output).
  Step 1 forward+sample produces `new_sampled` (held). Returns
  `state.last_token` values, which the scheduler already knows. So
  bootstrap returns "echo of last_token" — scheduler can ignore or
  no-op.
- Step 2: `token_arr` = step 1's `new_sampled`. Step 2 forward+sample
  produces step 2's `new_sampled`. Returns step 1's tokens. **First
  useful generation.**
- Step N: returns step N-1's tokens.

The scheduler runs one extra tick to drain `prev_sampled` at the end
of the request. Same pattern as mlx-lm's `GenerationBatch.next()`
running one extra `_step()` past the user's `max_tokens`.

### Edge cases

1. **Stop tokens** — caller checks the returned `Vec<u32>` for stop
   tokens. Same as today. The "extra step" past stop is wasted but
   bounded (≤ 1 step / request).
2. **Batch shrinkage / `retain_rows`** — the `prev_sampled` MlxArray
   must be `take_axis(prev_sampled, &index_arr, 0)`'d alongside
   `packed_kv_flat` and `packed_gdr_flat`. Add to the existing
   `retain_rows` body.
3. **Sample-fallback row mode** (`!qwen35_can_batch_sample`) — concat
   the per-row sampled MlxArrays into one `[B]` MlxArray before
   `async_eval`, otherwise we lose the single-array overlap. mlx-lm
   does the same: `mx.concatenate(all_samples, axis=0)`.
4. **First-call detection** — `prev_sampled.is_none()` is the bootstrap
   flag. After step 1 it is always `Some`.
5. **`pool_dual_write`** — operates on `packed_kv_flat` which IS
   already async_eval'd. Order of `eval(&[&token_arr])` vs pool
   `write_kv()` matters: write_kv reads from `packed_kv_flat` so its
   slice/reshape lazy graph extends what was async_eval'd. Pool's
   `flush()` force-evals the pool tensors — that work overlaps with
   `eval(&[&token_arr])` host readback.

## Implementation steps (next tick)

1. **Add `prev_sampled` field** + initialize to `None` everywhere
   `Qwen35PackedDecodeBatch` is constructed.
2. **Patch `retain_rows`** to gather `prev_sampled` along axis 0.
3. **Refactor `decode_qwen35_packed_batch` body** per pseudocode
   above; keep all phase-timing checkpoints intact.
4. **Add a path probe** `oMLX_C_PIPELINE_PROBE` that fires once when
   the pipeline path is taken (i.e. `prev_sampled.is_some()`),
   labeled `"oMLX-C pipelined step"`.
5. **Build & unit test** — existing
   `qwen35_packed_decode_…` tests must still pass numerically.
6. **Bench**: re-run `c4_smoke.sh` with `INFER_PHASE_TIMING=1`,
   compare `sample_us` before vs after. Acceptance:
   - `sample_us` drops to ≤ 1.5 ms (the `async_eval_kickoff` floor)
   - `total_us` drops below 5 ms p50 at c=4 (vs current 6.4 ms)
   - Numerical parity with the sync path on a deterministic seed.
7. **Wins entry**:
   `docs/experience/wins/2026-05-07-bench-c4-omlx-c-pipelining.md`
   citing this baseline.

## Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | Numerical drift if pool dual-write reads from packed_kv_flat AFTER `prev_sampled` consumes some lazy graph state. | Pool ops happen AFTER `async_eval` of eval_refs that include packed_kv_flat — same as today. No new dependency cycle. |
| R2 | `retain_rows` forgets to gather `prev_sampled` → shape mismatch crash. | Add a `debug_assert!(prev_sampled.shape()[0] == batch_size())` at start of decode. |
| R3 | Scheduler-level: caller's "extra tick" past max_tokens. | Document and verify the existing scheduler's max-token check is at scheduler boundary, not function boundary. Probably already OK — same pattern as the c=1 path's max_new_tokens loop. |
| R4 | `async_eval` of new_sampled crashes M4 driver if the chain depends on un-evaluated state. | mlx-lm does this unconditionally on M4 per subagent verification (`mlx_lm/generate.py:1369`); the chain we're firing has all its inputs already lazy-built and depends only on the immediately-prior `step_batch_packed` outputs. Same shape as ARLE's c=1 pattern at `mlx_qwen35_model.cpp:3274` which already runs on M4. |

## References

- Baseline:
  [`docs/experience/wins/2026-05-07-bench-c4-phase-timing-baseline.md`](../experience/wins/2026-05-07-bench-c4-phase-timing-baseline.md)
- Hypothesis source (subagent ROI ranking):
  [`docs/research/2026-05-07-mlx-ecosystem-survey-c4-itl-gap.md`](../research/2026-05-07-mlx-ecosystem-survey-c4-itl-gap.md)
- mlx-lm canonical pattern: `mlx_lm/generate.py:1320-1378`
  (`GenerationBatch._step`)
- ARLE c=1 prior art (sink-loop-into-C++ alternative, deliberately
  not chosen for c≥2 to keep scheduler reactivity):
  [`mlx_qwen35_model.cpp:3231-3325`](../../crates/mlx-sys/src/mlx_qwen35_model.cpp)
- oMLX deep-dive (paged-KV scope correction; this plan composes with
  oMLX-A flush-sync):
  [`docs/plans/M_e1-omlx-deep-dive-strategy-correction.md`](M_e1-omlx-deep-dive-strategy-correction.md)
- Audit trail (path-probe rule, c≥2 hot path identification):
  [`docs/experience/errors/2026-05-07-three-layer-audit-miss-c4-real-path-is-packed-batch.md`](../experience/errors/2026-05-07-three-layer-audit-miss-c4-real-path-is-packed-batch.md)
