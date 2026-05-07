# M_e.11 — Port omlx residency-set hygiene to Rust scheduler

**Owner:** ckl · **Status:** designed 2026-05-08 (subagent), awaiting impl tick
**Track:** Metal scheduler stability · **Predecessor:** M_e.4 SwiGLU compile-fusion

## Goal

Port omlx commit `6bda6781` (2026-05-06): periodic `mx.clear_cache()`
cadence on the decode hot path to prevent IOGPU residency-set abort
on long-context (>4096 generated tokens) workloads. macOS aborts
inside `-[IOGPUMetalResidencySet addAllocation:]` once the residency
set hits ~4096 entries because every `mx.random.categorical` →
`gumbel` → `uniform` allocates a fresh tiny scalar on the GPU.

## Reference (omlx)

`omlx/scheduler.py` (commit `6bda6781`, 2026-05-06):

```python
# Every 1024 decoded tokens:
mx.synchronize(generation_stream)
mx.clear_cache()
```

The cadence is conservative (1024 tokens between clears). Beyond a
certain threshold, the residency-set ceiling forces a full sync
anyway — the periodic clear converts a hard abort into a deterministic
sync.

## ARLE state

Two scheduler entry points need coverage. Codex review (2026-05-08)
flagged that v1 of this plan only mentioned the c≥2 path and missed
the c=1 path:

- **C++ `qwen35_compiled_generate`** (`crates/mlx-sys/src/mlx_qwen35_model.cpp:3399-3404`):
  has `clear_cache()` every 256 generated tokens, gated behind
  `use_qwen35_cpp_clear_cache()`. **More aggressive than omlx.** BUT
  this is the end-to-end "compile-and-run" generate function used
  by certain offline paths, NOT the per-step entry the scheduler
  calls.
- **C++ `qwen35_compiled_step_session` / `_paged`** (the c=1
  scheduler path that runs on every per-step tick): **NO clear_cache
  call.** Long single-request generations on the scheduler are
  unprotected.
- **C++ `qwen35_compiled_step_batch_packed`** (the c≥2 packed-batch
  scheduler path): same — **NO clear_cache call.**
- **Rust scheduler decode loop**
  (`infer/src/backend/metal/runtime.rs::execute_qwen35_packed_decode_batch`,
  `try_decode_qwen35_packed_batch`): does NOT periodic-clear.
  Has `clear_metal_cache()` at specific KV_CACHE_CHUNK boundaries
  (line 2487 of request_state.rs) but those are capacity-grow-
  driven, not residency-set-driven.

→ **All three scheduler paths (c=1 step_session, c=1 step_session_paged,
c≥2 step_batch_packed) are unprotected at >4096 generated tokens.**

## What the port looks like

Add a per-batch generated-token counter to `Qwen35PackedDecodeBatch`
or maintain it in the runtime dispatcher. After every N generated
tokens (per row × steps or aggregate), call `clear_metal_cache()`.

Option A — per-batch counter:
```rust
// In Qwen35PackedDecodeBatch:
generated_tokens_since_clear: u32,

// In decode_qwen35_packed_batch after sample:
batch.generated_tokens_since_clear += states.len() as u32;
if batch.generated_tokens_since_clear >= 1024 * states.len() as u32 {
    clear_metal_cache();
    batch.generated_tokens_since_clear = 0;
}
```

Option B — env-gated knob:
```rust
// INFER_METAL_CLEAR_CACHE_EVERY_N_TOKENS=1024 (default 0 = off)
```

For v1, **Option A with a fixed 1024-token cadence** matches omlx's
proven setting. Env-gate later if it conflicts with any workload.

## Why the existing C++-path clear isn't sufficient

The C++-path `clear_cache()` at `mlx_qwen35_model.cpp:3399` only fires
inside `qwen35_compiled_step_session` (single-stream c=1 path). The
c≥2 packed-batch path doesn't go through that loop — it goes through
`qwen35_compiled_step_batch_packed`, which has no clear_cache call.
So at c≥2 with long generation, the residency-set ceiling is
unprotected.

Bench evidence: ARLE's own `2026-04-30-longctx-32k-128k-leadership.md`
references long-context targets but the reproduce path needs >4096
generated tokens at c=4 to actually trip this. Today's eli e2e bench
(64 max_tokens × 11 turns = ~704 tokens) is well below the threshold.

## Acceptance bench

`scripts/bench_guidellm.sh longctx-32k-c4-decode-only --workload longctx-32k`
with `max_tokens=8192` (above the 4096 threshold). Predicate:

- Without M_e.11: server should abort at ~4096 generated tokens with
  IOGPU-residency-set assert. (Reproduce the bug first.)
- With M_e.11: server completes the bench without abort; aggregate
  ITL p50 unchanged (clear_cache is fast on Metal — ~10ms one-shot
  vs ~25ms per step).

## Composability

- **oMLX-C v3** (host pipelining): orthogonal — clear_cache is between
  pipelined steps, doesn't disturb the prev_sampled handoff.
- **auto-wired-limit**: complementary — auto-wired-limit pins weights;
  clear_cache prevents per-step scratch allocations from accumulating.
- **M_e.4 SwiGLU compile**: orthogonal.
- **INFER_MOE_TOP_K=N**: orthogonal.

## Risks

| ID | Risk | Mitigation |
|----|------|------------|
| R1 | clear_cache forces a synchronization point that breaks oMLX-C v3 pipelining | omlx's pattern uses `mx.synchronize(generation_stream)` BEFORE clear_cache; ARLE could match — synchronize after the prev async_eval of step N completes, then clear_cache between N and N+1's encode. The prev_sampled handoff still works (its stream just needs to be drained before clear). |
| R2 | 1024-token cadence is too aggressive for short-generation workloads (chat at max_tokens=64 never trips the bug, just pays the clear cost) | Skip the clear if `generated_tokens_since_clear < threshold`. Current eli workload (64 max_tokens × 11 turns = 704 tokens) wouldn't trigger any clear. |
| R3 | Per-batch counter complexity in the dispatch path | One u32 field, one increment, one if check. Simpler than the existing pool dual-write logic. |

## Implementation steps (covers ALL three scheduler paths)

1. **c≥2 packed-batch path** (`Qwen35PackedDecodeBatch`):
   - Add `generated_tokens_since_clear: u32` field.
   - After each successful step in `decode_qwen35_packed_batch`,
     increment by `states.len() as u32`.
   - When count >= 1024, `clear_metal_cache()` + reset counter.
2. **c=1 single-stream path** (`Qwen35StepDriver`):
   - Add `generated_tokens_since_clear: u32` field.
   - After each `run_cpp_step` / `run_cpp_step_paged`, increment by 1.
   - Same threshold + clear.
3. **Path probe**: `M_E11_RESIDENCY_CLEAR_PROBE` once-fire log on
   first triggered clear (one per scheduler path = three probes).
4. **Bench**:
   - **c=1 reproduce path**: long-generation single-stream with
     max_tokens=8192 against `metal_serve --max-running-requests 1`.
     Without M_e.11: should abort at ~4096. With M_e.11: completes.
   - **c=4 reproduce path**: `scripts/bench_guidellm.sh
     longctx-32k-c4-decode-only --workload longctx-32k`,
     max_tokens=8192. Same predicate.
   - **Steady-state regression check**: c=1/c=4/c=8 sweep at
     max_tokens=64 (today's eli workload) — clear cadence shouldn't
     fire (well below 1024 tokens), so ITL should be unchanged
     within ±2%.

## References

- omlx commit `6bda6781`:
  https://github.com/jundot/omlx/commit/6bda6781
- ARLE existing C++ clear:
  [`crates/mlx-sys/src/mlx_qwen35_model.cpp:3399-3404`](../../crates/mlx-sys/src/mlx_qwen35_model.cpp)
- ARLE Rust scheduler decode loop:
  [`infer/src/backend/metal/runtime.rs`](../../infer/src/backend/metal/runtime.rs)
  + [`request_state.rs::decode_qwen35_packed_batch`](../../infer/src/backend/metal/request_state.rs)
- Long-context bench reference:
  [`docs/experience/wins/2026-04-30-longctx-32k-128k-leadership.md`](../experience/wins/2026-04-30-longctx-32k-128k-leadership.md)
