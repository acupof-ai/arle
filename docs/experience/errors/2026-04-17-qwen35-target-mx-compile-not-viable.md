# Qwen3.5 target-forward mx::compile is not viable without cache-as-input redesign

## Context

After the block-verify win
([2026-04-17-metal-qwen35-dflash-block-verify.md](../wins/2026-04-17-metal-qwen35-dflash-block-verify.md))
left DFlash single-session at −35% vs plain decode on 4-bit Qwen3.5-4B,
the next hypothesis was that wrapping the target forward / verify path
in `mx::compile()` would close the gap. The reference `dflash-mlx`
implementation does compile parts of its forward, so it seemed
reachable.

An exploratory pass wired a `QWEN35_COMPILE=1` gate through the C++
model: new `compiled_verify_fn`, `ensure_compiled_verify`,
`build_compiled_inputs`, `compiled_forward_impl`, `compiled_verify_impl`
and helpers that boxed `cache_pos`, `batch_size`, `seq_len` as int32
rank-0 `array` inputs so the traced graph could read them.

## Root cause (why it was wrong)

Two independent problems showed up on review:

**P1 — `cache_pos` baked into the trace.** `scalar_array_to_i32()`
called `value.item<int32_t>()` from inside the compiled lambda. `item()`
materializes on the host during tracing, so values like `cache_pos`
became trace-time constants rather than dynamic graph inputs. With
`QWEN35_COMPILE=1`, the second compiled step would have written to the
first step's cache slot and corrupted generation.

**P2 — hidden-capture bookkeeping drifted from actual outputs.** The
new FFI returned `current_captured_hidden_count = capture_layer_ids.size()`
instead of counting what was actually pushed to `prev_outputs`. Any
duplicate or out-of-range id in the capture config (the DFlash config
is loaded verbatim) silently over-reports, and `qwen35_get_captured_hidden()`
starts throwing `captured hidden index out of range` instead of
capping gracefully.

Underlying problem: **MLX's `slice_update(cache, new_k, {0, 0, cache_pos, 0})`
requires `cache_pos` as a compile-time `int`, not a runtime tensor.**
The reference `dflash-mlx` sidesteps this by passing the *active KV
prefix* as a graph input and using `concat(prefix, new_kv)` — cache_pos
never appears in the traced function. That is a substantially larger
refactor than the `QWEN35_COMPILE` wiring tried to do, and it changes
the Rust-side cache contract for every call site.

## Fix

Reverted the compile wiring:

- Removed `use_qwen35_compile()`, `make_i32_scalar`,
  `scalar_array_to_i32`.
- Removed fields `compiled_verify_fn`, `is_verify_compiled`,
  `compiled_verify_seq_len`, `compiled_verify_capture_layer_ids`,
  `current_captured_hidden_count`.
- Removed `build_compiled_inputs`, `compiled_forward_impl`,
  `compiled_verify_impl`, `can_use_compiled_{forward,verify}`,
  `ensure_compiled_verify`, `restore_compiled_verify_side_effects`,
  `execute_forward`, `execute_block_verify_forward`.
- Restored `qwen35_get_captured_hidden_count` to derive the count from
  `prev_outputs.size()` (pre-change behavior).
- Dropped the `QWEN35_COMPILE`-gated branch in
  `qwen35_dflash_speculative_block`; `QWEN35_DFLASH_PROFILE=1` alone
  now drives the block_verify profile window.

Kept the clean parts of the refactor:

- `ForwardContext` / `ForwardArtifacts` structs — thread forward state
  and side-effect sinks explicitly through `full_attn_step`,
  `gdr_step`, and `forward_impl` instead of reading mutable member
  vars. No behavior change, smaller core functions, prerequisite for
  any future compile work.
- `contains_layer_id` helper.
- `Qwen35BlockVerifyProfileWindow` (10-block sliding average for
  block_verify latency).

## Numbers

Not measured — the compile path crashed (P1) or produced wrong output
before a stable bench was possible.

Separately, the reference `dflash-mlx` benchmark at
`benchmarks/qwen35-results.md` reports ~56.7ms verify; ours is ~54ms
today. Compile is not the bottleneck for verify cost.

## Rule

**`mx::compile(shapeless=true)` fuses elementwise ops; it does not
absorb position-dependent indexing.** Any input whose value enters the
graph via `item<T>()` becomes a trace-time constant. If the hot path
writes `cache[..., cache_pos, ...] = new_kv`, compile requires a
redesign that passes the active prefix as a graph input and uses
`concat` — not a wrapper that forwards the raw cache + position.

**Corollary:** before scoping a compile experiment on a stateful
forward, audit every `item()`, every `slice_update` with a runtime
position, and every captured mutable member — they are the actual
boundary, not the function signature.

**Corollary:** The reference implementation's pattern is not a free
drop-in. If the reference compiles `forward(active_kv_prefix, new_x)`
and we compile `forward(full_cache, cache_pos, new_x)`, they are not
the same graph. Match the graph shape, not just the API surface.

## Follow-ups

- Draft-side compile already works (single 5-layer forward, stateless
  w.r.t. KV — passes active prefix as graph input). Ported in
  `crates/mlx-sys/src/mlx_dflash_draft_model.cpp`, gated behind
  `DFLASH_DRAFT_CPP=1`.
- Target-side compile remains open. Path forward is cache-as-input
  design mirroring the reference's `whole_verify` — estimated multi-day
  refactor touching the Rust KV contract. Not on the critical path
  while single-session DFlash is still −35% vs plain decode on 4-bit.
- The ForwardContext refactor is the prerequisite; when we come back
  to this, the graph function can take a `ForwardContext` + prefix
  slices and avoid member-var capture entirely.
