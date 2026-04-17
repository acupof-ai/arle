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

**Approach**: extend the mlx-lm `BatchKVCache` varlen pattern (already
used by the scheduler for plain decode —
`infer/src/backend/metal/AGENTS.md` §7) to verify.

- Pack B requests' draft blocks into one `[B, S_padded]` token tensor
  with left-padding + additive causal mask + per-row RoPE offsets.
- Tape shape becomes `[B, S_padded, Hv, Dv]`; slice each row by its
  own `accepted_inputs[b]` before replay.
- Replay kernel must accept per-batch `steps[b]` — upgrade
  `mlx_tape_replay` to take `steps: array(B)` (Metal kernel already
  iterates `for (int t = 0; t < T; ++t)`, needs a per-row T bound).

**Trip wires**:
- Per-row RoPE offsets for verify positions (`cache_pos[b] + 0 ..
  cache_pos[b] + S`).  The existing varlen RoPE path in `forward.rs`
  handles a single offset per row — verify needs a per-row *range*
  which for contiguous S tokens collapses to broadcasting
  `cache_pos[b]` and letting the kernel offset each column.  Verify
  that the existing `fast::rope` array-offset path handles this.
- Partial-accept mask rebuild between verify and next prefill: rows
  with different `accepted_inputs[b]` need different trim amounts in
  both KV (full-attn) and GDR state (replay with per-row `steps[b]`).

**Acceptance**: `guidellm` sweep at concurrency ≥ 8 shows ≥ 2×
throughput over Layer 1 on mixed-length prompts.

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
| 2 (cross-request packed verify) | Next available slot, after Layer 1 Mac benchmarks | TBD |
| 3 (cross-slot scheduling) | After Layer 2 | TBD |
