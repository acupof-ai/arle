# Metal DFlash Qwen3.5 — batched verify roadmap

**Status**: Layer 1 landed 2026-04-17 (single-forward intra-request
verify).  Layer 2 / Layer 3 pending benchmarks.
**Scope**: Apple Silicon Metal backend, Qwen3.5-4B hybrid model,
`qwen35_dflash_speculative_block`.

---

## 1. Motivation

Single-request Qwen3.5 DFlash decode measured **15.2 tok/s** vs
**74.8 tok/s** baseline before 2026-04-17 — a 5× regression.  Root
cause and Layer-1 fix are in
[`docs/experience/errors/2026-04-17-metal-dflash-qwen35-verify-serial-loop.md`](../experience/errors/2026-04-17-metal-dflash-qwen35-verify-serial-loop.md).

This plan tracks the remaining two layers needed for DFlash to be a
net throughput win on Metal.

---

## 2. Layers

### Layer 1 — single-forward intra-request verify (done, 2026-04-17)

- **C++**: `qwen35_compiled_verify_block` — one forward at
  `seq_len = block_size` emits logits `[1, S, V]`, GDR tapes `[1, S,
  Hv, Dv]` (one per GDR layer, plus matching k/g/qkv), capture hidden
  `[1, S, H]` (one per capture layer).
- **Rust**: `CppQwen35Model::verify_block` wrapper + rewritten
  `qwen35_dflash_speculative_block` that slices axis 1 for partial
  accept and runs one `mlx_tape_replay` per GDR layer.

**Expected win** (to be measured on Mac):  `S × {set_packed_kv +
schedule + eval + tape/hidden drain + host sync}` collapses to one of
each.  Estimated 3–5× decode throughput recovery for single-request
DFlash, bringing Qwen3.5 DFlash ≈ Qwen3 DFlash (which already did this).

**Acceptance** (Mac, `cargo test --release --no-default-features
--features metal -- --test-threads=1`):
- `qwen35_dflash_baseline_regression` matches pre-change outputs
  bit-for-bit when acceptance rate is 100%.
- `guidellm` snapshot at Qwen3.5-4B DFlash ≥ 2.5× Layer-0 (serial
  loop) throughput on Apple M4 Pro.

### Layer 2 — cross-request packed verify (planned)

**Problem**:  Layer 1 batches positions *within* one DFlash block for
one request.  On a multi-request scheduler tick, each slot still runs
its own verify forward — no cross-request batching for the speculative
path.

**Scheduler wiring — already done, just downgraded**:
DFlash already dispatches from the scheduler runtime
(`backend/metal/runtime.rs::execute_decode_batch:1040-1056`), not a
separate runtime.  The "legacy serial runtime" phrasing in
`backend/metal/AGENTS.md` invariant #6 is outdated — DFlash runs via
`execute_decode_single` **inside** `run_metal_scheduler_runtime`.  What
actively blocks B>1 verify today is a **policy choice**, not missing
wiring: when `open.len() >= 2` the scheduler **permanently** disables
DFlash on every row (`runtime.rs:1040-1045`) so speculative rows join
the packed decode path.  Rationale at the time: per-request serial
verify was 230 ms/block at 28% accept, worse than 25 ms packed decode.
Layer 2 flips that policy by making packed verify cheap enough to keep
DFlash on under concurrency.  Decomposes into four sub-pieces that
**must land in order**:

**2a — varlen `mlx_tape_replay_varlen` kernel** (in flight, 2026-04-18)
- Kernel + FFI accept `steps: mlx_array([B], int32)` instead of scalar
  `int steps`.  Keeps scalar `mlx_tape_replay` alongside for the
  single-row callsite.
- Unblocks (but does not depend on) 2b/2c/2d.
- Acceptance: gdr.rs `test_tape_replay_varlen_matches_scalar` passes;
  bit-identical output when per-row steps are uniform.

**2b — `qwen35_compiled_verify_block_batched` C++ FFI**
- New C++ FFI mirroring `qwen35_compiled_verify_block` but accepting
  `[B, S_padded]` tokens, `attn_mask: [B, 1, S_padded, key_len]`
  additive, `rope_offsets: int32[B]`, per-row `cache_lens: int32[B]`.
- Reuses the `step_batch_packed` plumbing
  (`crates/mlx-sys/src/mlx_qwen35_model.cpp`) that already lands
  varlen plain-decode with per-row RoPE offsets.
- Emits logits `[B, S_padded, V]`, GDR tapes `[B, S_padded, Hv, Dv]`
  (one set per GDR layer), capture hidden `[B, S_padded, H]`.
- Acceptance: `cargo test --release --features metal -- --test-threads=1
  qwen35_compiled_verify_block_batched` — B=1 run is bit-identical to
  existing `qwen35_compiled_verify_block`.

**2c — Lift the `open.len() >= 2` downgrade + packed verify dispatch**
- Remove (or gate behind a config toggle) the permanent DFlash disable
  at `runtime.rs:1040-1045` so DFlash rows survive into a multi-row tick.
- Add a DFlash-speculative dispatch variant to
  `execute_qwen35_packed_decode_batch` — when every open row has DFlash
  enabled, each row runs its own draft forward (CPU-batchable since the
  draft model is small), then all B draft blocks pack into one
  `qwen35_compiled_verify_block_batched` call (Layer 2b).
- Mixed-mode (some DFlash rows, some plain-decode rows): cleanest route
  is to run the two groups in parallel via MLX lazy eval — DFlash rows
  through the batched verify, plain rows through the existing
  `step_batch_packed`.  Alternative: auto-enroll plain rows with a
  stub draft (all-mask tokens) so everyone rides the same forward; only
  viable if acceptance-cost tradeoff holds under zero accept.  Pick
  after Layer 2b benchmarks show the per-row verify cost.
- Per-row rollback: `qwen35_rollback_to_accepted` already loops per GDR
  pair; swap its scalar `mlx_tape_replay` for `mlx_tape_replay_varlen`
  (Layer 2a) and thread `accepted_inputs: Vec<usize>` through.
- Acceptance: `metal_serve --dflash-draft-model …` with concurrency 4
  keeps DFlash on; the Track-1 auto-downgrade becomes a configurable
  opt-in fallback (e.g. `--dflash-concurrency-off`) instead of the
  default hard-off.

**2d — Layer 2 wire-up**
- `qwen35_dflash_speculative_block` grows a batched sibling or replaces
  its inner verify call with a batched one when `B > 1`.
- Tape replay uses `mlx_tape_replay_varlen` with per-row
  `accepted_inputs[b]`.
- Partial-accept rollback is per-row: KV trim + GDR state restore
  happen row-by-row.
- Acceptance: `guidellm` sweep at concurrency ≥ 8 shows ≥ 2× throughput
  over Layer 1 on mixed-length prompts.

**Trip wires** (carry forward from the original plan + new discoveries):
- Per-row RoPE offsets for verify positions: the existing varlen RoPE
  path (`fast::rope` with int32[B] offsets) works for S=1 plain decode;
  confirm it handles S>1 where each column within a row needs a
  different absolute position (`cache_lens[b] + s`).  If `fast::rope`'s
  array-offset signature only accepts scalar-per-row (no per-column
  offset), the kernel path may need a broadcast trick: feed
  `base_offsets[b]` and let the kernel's internal `t_axis` add the
  column index — verify in isolation before 2b.
- GDR state shape under B>1: today `target_gdr_flat` is flat-indexed
  per-layer.  Packed version needs `[B, Hv, Dv, Dk]` rooted at the
  batch axis.  `qwen35_set_tape_mode` / `qwen35_set_capture_layers`
  toggle applies to the whole C++ model — capture is all-rows or no-rows,
  which is fine since every DFlash slot needs tapes.
- Single-session single-request workloads get **nothing** from Layer 2
  (B=1 already).  The single-session regression on Qwen3.5-4B-4bit
  documented in `wins/2026-04-17-metal-qwen35-dflash-block-verify.md`
  is out of scope for this layer; it requires attacking the GDR
  time-axis recurrence directly (Layer 3 candidate, or orthogonal
  algorithmic work on `gated_delta_tape_kernel`).

### Layer 3 — cross-slot speculative scheduling (planned)

**Problem**:  Even with Layer 2 packing, slot lifecycles are
independent: one slot's DFlash block completes in N_block × verify
latency, another slot may have already finished a block and be idle
waiting to join the next tick.

**Approach**: unify DFlash into the continuous-batching scheduler tick
rather than running it as a per-slot inner loop.  Each tick:

1. For each active slot, run its draft model (independent; can be
   CPU-batched since draft is small).
2. Pack all slots' draft blocks into one verify forward.
3. Decompose results per slot and route to sampling + KV advance.

Requires scheduler-side invariants: slots must agree on `block_size`
or pad to max.  Draft cache per slot stays independent.

**Acceptance**: `scheduler` unit tests pass with DFlash enabled.
Steady-state throughput under concurrency 16 ≥ 1.5× Layer 2.

---

## 3. Out of scope

- CUDA DFlash — separate backend, different set of issues
  (`FlashInfer` prefill already handles `seq_len > 1`; the regression
  was Metal-only).
- Draft-model architecture changes (block-diffusion vs chain-of-thought
  drafts).  Orthogonal to verify batching.
- Acceptance-rate tuning (temperature, top-k, draft-target alignment).
  Our measurements show acceptance is already high; the bottleneck
  was verify cost.

---

## 4. Timeline (soft)

| Layer | ETA | Owner |
|-------|-----|-------|
| 1 (intra-request single forward) | **Done 2026-04-17** | Claude |
| 2a (varlen `mlx_tape_replay`) | **In flight 2026-04-18** | Codex |
| 2b (`verify_block_batched` FFI) | After 2a | Codex |
| 2c (DFlash scheduler integration) | After 2b | Claude (direction) + Codex (impl) |
| 2d (packed verify wire-up) | After 2c | Codex |
| 3 (cross-slot scheduling) | After 2d; may collapse into 2c | TBD |
