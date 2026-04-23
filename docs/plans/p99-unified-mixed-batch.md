# P99 Unification — Mixed-Batch Scheduler & Paged-KV Everywhere

> Status (2026-04-23): **Historical master plan.** This file still explains the broader mixed-batch / paged-prefill motivation, but it is not the current source of truth for CUDA decode alignment against SGLang `main`. Use [`2026-04-23-cuda-decode-sglang-alignment.md`](2026-04-23-cuda-decode-sglang-alignment.md) for the current decode-focused plan.

**Status:** active master plan (2026-04-17)
**Supersedes:** `paged-kv-prefill.md` (Phase 3b/3c/4), `qwen35-single-graph-prefill.md`,
`flashinfer-planned-prefill.md`, `scheduler-gpu-cpu-overlap.md`.
**Complements (does not replace):** `qwen35-sglang-parity.md` (this is its P0/P1 scaffolding).

---

## Why

`docs/experience/wins/2026-04-17-sglang-parity-systematic-analysis.md` measured
the current gaps against SGLang with `num_slots=64`:

| Metric      | Qwen3-4B     | Qwen3.5-4B    |
|-------------|--------------|---------------|
| ITL mean    | within 3%    | **we win 3-8%** |
| TTFT mean   | ~50% slower  | ~60% slower   |
| Peak tput   | ~18% lower   | ~47% lower    |
| P99 TTFT    | 4-8× worse   | 6-12× worse   |

Decode is competitive. The P99/TTFT/peak-tput gap lives in **prefill admission
and P+D fusion** — the same place SGLang `--enable-mixed-chunk` and vLLM V1's
two-pass token budget solve.

The root blocker is **half-state**: four plan docs each landed ~30% and started
stepping on each other. Concretely:

1. Qwen3 has both the new paged-prefill path (`process_all_layers_batch_paged`)
   **and** the old contiguous+scatter path (`decode_batch_with_prefill`
   → `prefill_attention_prep_dual_write_cuda`). Mixed P+D uses the old path, so
   P+D fusion doesn't get the paged benefits.
2. Qwen3.5 and GLM4 never migrated to paged at all — the full-attn layers still
   hit a 512-token chunk ceiling because `CONTIGUOUS_KV_TOKENS` can't be lifted
   while they read from contiguous scratch.
3. The scheduler's `step()` already does launch/decode-overlap correctly, but
   `step_decode_launch_mixed` only fires when **Qwen3** is in the batch — the
   dispatch wants a uniform MixedBatch contract, not per-model specials.
4. `flashinfer-planned-prefill` (single-request TTFT) and
   `qwen35-single-graph-prefill` (Qwen3.5 full-forward graph) both need the
   paged-KV migration as a prerequisite. They stalled waiting for it.

This plan sequences the fix so each phase unblocks the next and leaves no
half-states on main.

---

## Is the architecture unifiable?

**Yes at the scheduler/trait contract layer, no at the forward() implementation
layer.** That answers the user's question directly — here's where the line is:

### What IS unified (and stays unified)

- **`InferenceEngine`** — one HTTP/engine contract across CUDA + Metal.
- **`ModelForward`** — trait every model plugs into. The scheduler dispatches
  through it; models are backend/architecture-specific but interchangeable at
  the scheduler's level.
- **`MixedBatch` contract (new, §Phase 2)** — one packed-token shape
  (`qo_indptr`, `kv_indptr`, per-row `slot_idx`/`kv_len`/`sample`) covers
  pure-decode, pure-prefill, and P+D mixed. Same struct for every model on
  CUDA; Metal has its own equivalent built from the same row-level info
  (left-pad + additive mask).
- **Paged KV pool** — single pool, single `page_size` per dtype, same indexing
  kernels for every model on CUDA.
- **Scheduler** — slot lifecycle, prefix cache, preemption, admission — all
  backend/model-agnostic.

### What is NOT and cannot be unified

- **`forward()` body.** Qwen3 (HD128), Qwen3.5 (HD256 + linear-attn DeltaNet),
  GLM4 (HD128 + QKV bias). Each has its own op-tape. That's the whole point of
  the `ModelForward` trait — architectural differences live here.
- **Quantization KV paths.** INT8 KV has no FlashInfer wrapper and uses a
  project-specific kernel that assumes `qo_len=1`. Paged-KV is BF16 first;
  INT8/FP8 stay on their custom path until we write paged INT8 kernels
  (out of scope).
- **Qwen3.5 linear-attn layers.** DeltaNet is a chunkwise recurrent kernel
  (`gated_delta_rule_prefill_chunkwise`). It assumes contiguous chunks and
  doesn't share the packed-token shape. Varlen packing affects only the 8
  full-attn layers.
- **Metal packing.** MLX bridge uses left-pad + additive mask + per-row RoPE
  offsets (see `crates/mlx-sys/AGENTS.md` §7). Same logical `MixedBatch` info,
  different physical layout.

So the final architecture: **one scheduler, one dispatch contract, N model
forward()s.** New models plug in by implementing `ModelForward`; they inherit
the scheduler's P+D fusion, prefix cache, preemption, graph capture policy,
and paged-KV pool automatically. They do *not* inherit Qwen3's kernel layout —
they provide their own.

---

## Phases

### Phase 1 — Kill the half-states (this commit + next)

**Goal:** every model that can use paged-KV prefill does. CONTIGUOUS_KV_TOKENS
ceases to cap any model in the default path. No contiguous + dual-write path
left on the default prefill.

**1A — Qwen3.5 paged prefill (this commit, HD256 full-attn layers):**
- Add `paged_prefill_plan_hd256: Mutex<Option<BatchPrefillPagedPlan>>` to
  `Qwen35Model`.
- Add `process_all_layers_batch_paged_35(hidden, start_pos, pool, slot)` in
  `infer/src/model/qwen35/prefill.rs`. For `FullAttention` layers:
  `ops::prefill_attention_hd256_paged_batch`. For `LinearAttention` layers:
  existing `prefill_linear_attention` (recurrent, no KV change).
- Override `forward_prefill_with_pool` and `prefill_uses_paged_pool() → true`
  in `infer/src/model/qwen35/forward.rs`.
- Scheduler already handles the rest — `prefill_uses_paged_pool` lifts the
  CONTIGUOUS_KV_TOKENS cap via `prefill_chunk_size()` (already in
  `infer/src/scheduler/cuda/core.rs:884`).

**1B — GLM4 paged prefill (follow-up, HD128):**
- Scoped out of the first commit. GLM4 adds two complications on top of the
  Qwen3 pattern: (a) QKV bias must broadcast-add after the projections and
  before the paged prep kernel; (b) GLM-4's RoPE shape needs to be reconciled
  with `prefill_attention_paged_prep_cuda` (it currently bakes in the HD128
  full-rotate layout Qwen3 uses). Both are contained but not trivial.
- Approach when we take it: apply `add_bias_batch_into` to q/k/v, then call
  `ops::prefill_attention_paged_batch` with dummy-identity q_norm/k_norm
  (same trick the current GLM4 code already uses for the contiguous path).
  Compare numerics against the existing contiguous path before deleting it.
- Until then GLM4 keeps its existing contiguous+scatter path — it works, it
  just doesn't get the >512-token chunk win.

**1C — Drop Qwen3's legacy dual-write path from mixed-batch (follow-up):**
- `batch_decode::decode_batch_with_prefill` currently calls
  `prefill_attention_prep_dual_write_cuda`. Once Phase 2 lands a true paged
  mixed-batch kernel, remove this call site. Until then, Qwen3 P+D fusion
  still uses the dual-write (correct but slower); this is logged, not hidden.

**Verification for Phase 1:**
- `cargo check -p infer --no-default-features --features cuda,no-cuda` (here, Mac).
- CUDA box: `cargo test --release --test e2e` (Qwen3), `cargo test --release
  --test e2e_qwen35`, GLM4 e2e, then `scripts/bench_guidellm.sh paged-all`
  snapshot vs. the Apr 17 baseline. Expected: Qwen3.5 TTFT gap closes
  substantially once 512-token cap lifts.

### Phase 2 — MixedBatch dispatch contract (follow-up PR)

**Goal:** one struct, one scheduler path, one kernel shape for pure-decode,
pure-prefill, and P+D. Eliminates `step_decode_launch` vs
`step_decode_launch_mixed` vs `step_prefill_chunk` as separate code paths.

**Contract (`infer/src/scheduler/mixed_batch.rs`):**
```rust
pub struct MixedBatch {
    pub tokens:        Vec<u32>,         // packed, length = total_tokens
    pub positions:     Vec<u32>,         // per-token absolute position
    pub qo_indptr:     Vec<i32>,         // prefix sum of per-row qo_len
    pub kv_indptr:     Vec<i32>,         // prefix sum of per-row kv_len_in_pages
    pub slot_indices:  Vec<usize>,       // one per row (not per token)
    pub sample_rows:   Vec<bool>,        // which rows need sampling
    pub kv_len:        Vec<i32>,         // per-row absolute KV length
}
```
- Pure decode: `qo_len=1` for every row, `sample_rows=true`.
- Pure prefill (single): one row, `qo_len=N`, sample the last token.
- Mixed: decode rows first, prefill rows last — same struct.

**Changes:**
- `ModelForward::forward_batch(&self, batch: &MixedBatch, states: &mut [State],
  pool: &PagedKVPool) -> Result<Logits>`. Default impl delegates to existing
  decode/prefill for backward compat during migration.
- Models that support true mixed (Qwen3 full-attn, Qwen3.5 full-attn only,
  GLM4) implement `forward_batch` to run one FlashInfer
  `BatchPrefillWithPagedKVCacheDispatched` call with the per-row varlen shape.
- Scheduler `step()` builds one `MixedBatch` per step; drops
  `pending_mixed_prefill_idx` and `step_decode_launch_mixed` specials.

### Phase 3 — vLLM-style two-pass token budget (follow-up PR)

**Goal:** deterministic admission that matches vLLM V1's behavior. Prevents
the admission-thrashing seen under `max_num_seqs=500` load.

- Introduce `max_num_batched_tokens` (= prefill+decode token budget per step).
- Introduce `long_prefill_token_threshold` (chunk caps the per-request prefill
  slice once running decode count exceeds it).
- Pass 1: score requests → Pass 2: fit into budget while respecting KV
  admission (full-ISL reservation, Phase 4 §4).
- Replace the ad-hoc `max_prefills` ladder in `execution.rs:90-98` with the
  two-pass result.

### Phase 4 — Scheduler admission & preemption parity (follow-up PR)

- **Full-ISL KV reservation** (SGLang `scheduler_reserve_full_isl`,
  vLLM V1 `--kv-cache-memory-bytes`): admit a request only if the pool has
  pages for its full maximum context. Preemption-storm fix.
- **Preemption policy:** recompute-only with "retract most-generated"
  heuristic — matches V1. Already partial; formalize + test.
- **Prefix-cache hit admission bonus:** count prefix-hit tokens against the
  budget only once (vLLM `num_computed_tokens_no_spec`).

### Phase 5 — Graph capture strategy (follow-up PR)

- **Qwen3.5 full-forward graph** (ex `qwen35-single-graph-prefill.md`):
  capture all 32 layers in one graph for the uniform-decode batch. Requires
  `MixedBatch` contract landed; graph captures only the fixed-shape
  uniform-decode slice.
- **FA3 `AttentionCGSupport::ALWAYS` approximation** for mixed batches of the
  same `num_actual_tokens`: cache graph by total-token count, not by batch
  layout. Avoids eager-mode fallback on mixed.
- **Planned-prefill workspace reuse** (ex `flashinfer-planned-prefill.md`):
  the shared `BatchPrefillPagedPlan` from Phase 1 already does this.
  Reconfirm, then delete the standalone plan doc.

---

## Adding a new model (post-Phase 2)

1. Implement `ModelForward` with `forward_batch(&MixedBatch, ...)`.
2. If your architecture has HD128 or HD256 attention with paged KV: use
   `ops::prefill_attention_paged_batch` (HD128) or
   `ops::prefill_attention_hd256_paged_batch` (HD256) — both already batched
   over a varlen shape.
3. If your architecture has recurrent/linear layers (Mamba, DeltaNet, RWKV):
   implement the per-layer kernel yourself, but keep the outer `forward_batch`
   signature. State lives in the model's `State` associated type (see
   `Qwen35State.recurrent_state` for the template).
4. Set `prefill_uses_paged_pool() → true` unless your KV is custom (INT8 etc).
5. Register in `model.rs` dispatch. Scheduler handles everything else.

---

## Non-goals

- INT8/FP8/TurboQuant paged-prefill (separate project — needs new kernel).
- Cross-request attention (not on the roadmap).
- Speculative decoding integration (has its own plan).
- Metal parity for `MixedBatch` internal layout — the logical contract is
  shared but Metal's physical packing stays left-pad+mask per
  `backend/metal/AGENTS.md` §7.

---

## Success criteria (Phase 1 acceptance)

- `cargo check -p infer --no-default-features --features cuda,no-cuda` clean.
- `cargo test --release --test e2e` and `e2e_qwen35` green on CUDA box.
- `scripts/bench_guidellm.sh paged-phase1` shows:
  - Qwen3.5 `prefill_chunk_size()` logs ≥ 2048 under load (cap lifted).
  - Qwen3.5 TTFT mean improves ≥ 20% vs 2026-04-17 snapshot.
  - No regression in Qwen3 ITL or Qwen3 TTFT.
- Qwen3.5 `forward_prefill_with_pool` + `prefill_uses_paged_pool() → true`
  wired through the scheduler (lifts the 512-token cap for Qwen3.5).
- GLM4 untouched in Phase 1 (tracked as 1B follow-up).
