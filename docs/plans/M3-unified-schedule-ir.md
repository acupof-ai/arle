# M3 — Unified Scheduler Decision IR

> Sub-plan of [`backend-unification.md`](backend-unification.md) §M3 (Week 3-4).
> Goal: collapse the two parallel decision-shape enums (CUDA `ScheduleDecision`
> + Metal `MetalLogicalServePlan`) into one logical IR consumed by both
> backends, so any future scheduling/policy improvement is written once.

## 0. Current state — two parallel shapes

### 0.1 CUDA side
`infer/src/scheduler/batch.rs:66` (used only by `scheduler/tests.rs`; the
production path under `scheduler/cuda/core.rs` does its own decision
inline):

```rust
pub enum ScheduleDecision {
    DecodeBatch(DecodeBatch),     // req_ids, input_ids, block_tables
    PrefillBatch(PrefillBatch),   // req_ids, input_ids, seq_lens, block_tables
    Mixed { decode, prefill },
    Idle,
}
```

The CUDA scheduler **does not actually emit `ScheduleDecision`** — it
inlines an equivalent dispatch in `scheduler/cuda/core.rs::step()` +
`scheduler/cuda/runtime/scheduler_loop.rs`. That production decoder is
where the spec-decode and sparse-KV branches live (`spec_path.rs`,
`scheduler/cuda/decode.rs`, `prefill.rs`). The `BatchScheduler` enum
above is the *original* design that fell out of sync; M3 either revives
it or supersedes it.

### 0.2 Metal side
`infer/src/backend/metal/plan.rs:104`:

```rust
pub struct MetalLogicalServePlan {
    pub decode_rows: Vec<MetalLogicalDecodeRow>,        // row_index, req_id, input_token, logical_offset
    pub prefill_rows: Vec<MetalLogicalPrefillRow>,      // row_index, req_id, input_tokens, prompt_start/end/len, logical_start/end
    pub batch_shape: MetalLogicalBatchShape,            // decode_rows / prefill_rows / total_rows / *_tokens / scheduled_tokens
}
```

Metal scheduler emits this. Decode rows are 1-token; prefill rows carry
chunks. Logical offsets are packed-varlen indices (left-padding +
additive mask runtime). Backend-specific.

### 0.3 Field-level diff

| Field | CUDA | Metal | Convergence note |
|---|---|---|---|
| request id | `req_id: RequestId` | `req_id: RequestId` | identical |
| decode input token | `input_ids: Vec<u32>` (per-row) | `input_token: u32` | trivially the same flattened `Vec<u32>`, no source change |
| prefill chunk | `input_ids: Vec<u32>` for the chunk | `input_tokens: Vec<u32>` | identical |
| prefill seq state | `seq_lens: Vec<usize>` | `prompt_start/end/len` + `logical_start/end` | Metal carries more (packed-varlen needs absolute offsets) |
| KV access | `block_tables: Vec<Vec<BlockId>>` (paged) | implicit (MLX cache holds it) | **divergent** — see §1.2 |
| batch shape stats | absent | `MetalLogicalBatchShape` | metal-specific, easy to lift to common |

## 1. Unified IR design

### 1.1 Shape

```rust
// infer/src/scheduler/plan.rs (new top-level file, no cuda/metal cfg gates)

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LogicalDecodeRow {
    pub row_index: usize,
    pub req_id: RequestId,
    pub input_token: u32,
    /// Absolute KV offset for this request's KV state at this step.
    /// CUDA derives `block_tables[req_id]` from `req_id` at lowering time;
    /// Metal uses the packed-varlen offset directly.
    pub logical_kv_offset: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LogicalPrefillRow {
    pub row_index: usize,
    pub req_id: RequestId,
    pub input_tokens: Vec<u32>,
    /// Absolute prompt span being chunked, in the request's prompt sequence.
    pub prompt_start: usize,
    pub prompt_end: usize,
    pub prompt_len: usize,
    /// Logical KV span this chunk will write into.
    /// CUDA: `(block_table_lookup, block_table_lookup + (prompt_end - prompt_start))`.
    /// Metal: packed-varlen position.
    pub logical_kv_start: usize,
    pub logical_kv_end: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LogicalBatchShape {
    pub decode_rows: usize,
    pub prefill_rows: usize,
    pub total_rows: usize,
    pub decode_tokens: usize,
    pub prefill_tokens: usize,
    pub scheduled_tokens: usize,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LogicalServePlan {
    pub decode_rows: Vec<LogicalDecodeRow>,
    pub prefill_rows: Vec<LogicalPrefillRow>,
    pub batch_shape: LogicalBatchShape,
    /// Spec-decode row tags: each entry indexes into `decode_rows` and lists
    /// (draft_token_count, draft_tokens, draft_logits_handle?). Empty when
    /// no row is using speculation this tick.
    pub spec_rows: Vec<LogicalSpecDecodeRow>,
    /// Sparse-KV self-spec view tags (MagicDec). Per-row, optional.
    pub sparse_views: Vec<LogicalSparseDraftView>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LogicalSpecDecodeRow {
    pub decode_row_index: usize,
    pub draft_tokens: Vec<u32>,
    pub draft_mode: DraftMode,  // reuse existing enum from scheduler/types.rs
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LogicalSparseDraftView {
    pub decode_row_index: usize,
    /// Block ids the request's draft attention may read from
    /// (radix-cache backed, pre-validated by SchedulerSignals).
    pub allowed_block_ids: Vec<BlockId>,
    pub recent_window_tokens: usize,
}
```

### 1.2 Why `logical_kv_offset` instead of CUDA's `block_tables`

CUDA's per-row `Vec<BlockId>` is lookup-keyed by `req_id` against
`request.kv_indptr`/`kv_indices` already on the scheduler. The IR can
expose `logical_kv_offset` (a single integer = "where the next token's
KV lives") and let the CUDA lowering re-derive `block_tables` at
dispatch time from `req_id` + the on-scheduler `paged_kv_pool`. Metal
uses the offset directly. **This keeps the IR backend-neutral without
losing CUDA's paged-KV model** — paging is a backend concern, not a
scheduler concern.

### 1.3 Spec-decode + sparse-KV in the IR (CUDA-only fields, but backend-neutral shape)

CUDA's spec-decode (`spec_path.rs`) and MagicDec sparse-KV
(`spec_sparse_kv_enabled`) currently live in CUDA-private state on the
scheduler request struct (`spec_acceptance_tracker`,
`spec_decode_disabled`). M3 lifts the **scheduler decision** "this
decode row will run in spec-decode mode and these are the proposed draft
tokens" into the IR as `LogicalSpecDecodeRow` / `LogicalSparseDraftView`.
The Metal lowering ignores those fields until M_c (hybrid + spec
rollback) wires Metal-side spec — they are present-but-unused on Metal.

### 1.4 Backend lowering API

```rust
// infer/src/scheduler/plan.rs

pub trait LogicalPlanLowering {
    /// Run one logical plan on the backend. Returns per-row outputs in
    /// `LogicalServePlan.decode_rows[i].row_index` order so the scheduler
    /// can re-key results back to RequestId.
    fn execute(&mut self, plan: &LogicalServePlan) -> Result<LogicalStepOutput>;
}

#[derive(Debug, Default)]
pub struct LogicalStepOutput {
    pub generated_tokens: Vec<(usize /* row_index */, GeneratedToken)>,
    pub finished_rows: Vec<(usize, FinishReason)>,
    /// Spec-decode acceptance counts per spec-decode row, ordered by
    /// `LogicalSpecDecodeRow.decode_row_index`.
    pub spec_acceptance: Vec<(usize, usize /* accepted */, usize /* drafted */)>,
}
```

CUDA implements `LogicalPlanLowering` against its existing
`forward_decode_batch` / `forward_prefill` paths. Metal implements it
against the existing `run_metal_scheduler_runtime`. **Neither backend's
internals get rewritten in M3**; M3 is purely about the IR boundary.

## 2. Migration path

| Step | Files touched (estimate) | Owner | Days |
|---|---|---|---|
| S1. Land `infer/src/scheduler/plan.rs` with the IR types + `LogicalPlanLowering` trait. CPU-only. Add round-trip CPU tests. | 2 (1 new + 1 mod export) | Codex | 0.5 |
| S2. Replace `infer/src/backend/metal/plan.rs` with a re-export from `scheduler/plan.rs`. Migrate `MetalLogicalDecodeRow` → `LogicalDecodeRow` and `MetalLogicalServePlan` → `LogicalServePlan` everywhere in `backend/metal/`. | ~12 (mostly Metal callsites) | Codex | 1 |
| S3. Make CUDA `Scheduler<M>::step()` emit a `LogicalServePlan` *alongside* the existing inline dispatch (gate behind a trace logging flag). Verify diff in CPU log between the two for a fixed seed scenario. | 2 (`scheduler/cuda/core.rs`, `runtime/scheduler_loop.rs`) | Codex | 1 |
| S4. Implement `LogicalPlanLowering` for CUDA — reuse current dispatch paths inside the trait method. Replace the inline dispatch with `lowering.execute(plan)`. Keep spec-decode / sparse-KV inline branches gated by `LogicalSpecDecodeRow` / `LogicalSparseDraftView` populated state. | 4 (`scheduler/cuda/{core, decode, prefill, spec_path}.rs`) | Codex | 2 |
| S5. Implement `LogicalPlanLowering` for Metal — wraps `run_metal_scheduler_runtime`. | 2 (`backend/metal/{runtime, scheduler}.rs`) | Codex | 1.5 |
| S6. Delete `infer/src/scheduler/batch.rs::ScheduleDecision` + the old `MetalLogicalServePlan` shim once both backends consume the unified IR. Verify `scheduler/tests.rs` either (a) deleted as redundant or (b) ported to test the unified IR. | 3 | Codex | 0.5 |
| S7. wins entry. Bench: e2e + greedy_consistency + Metal smoke (pending-remote on Linux runner). | 1 | Claude | 0.5 |

**Total: ~26 files, ~6.5 days.** Manageable in M3's Week 3-4 budget.

## 3. Boundary cases — what stays inline (NOT lifted to IR)

These are explicitly NOT M3 scope; they are backend-private optimizations
the IR caller doesn't need to reason about:

- **CUDA Graph capture** for decode batches (`decode_batch_graph_body`) —
  caller passes the plan, lowering decides graph vs eager. Stays
  CUDA-private.
- **TileLang AOT kernel selection** — purely backend.
- **MLX lazy graph build** on Metal — purely backend.
- **Paged-KV admission** (T1/T2 fetch, prefix-cache lookup) — runs
  *before* the scheduler emits a `LogicalServePlan`, not inside it.
- **Per-request RAII / KV refcount** — scheduler-side, stays in
  `infer/src/scheduler/cuda/core/state_types.rs` and Metal-side
  `MetalKVPool` respectively.

## 4. Risks + retreat

- **R1 — CUDA's inline dispatch carries scheduler-side state that's
  awkward to thread through a plan struct** (e.g. mutable
  `spec_acceptance_tracker` per-request, prefix-cache hit ledgers).
  Retreat: keep state on `Scheduler<M>` request structs; the plan only
  carries *decisions*, not state pointers. The lowering implementation
  reads/writes scheduler state as it does today.
- **R2 — Metal's packed-varlen requires `logical_kv_offset` to be a
  packed-buffer offset, while CUDA wants block_id**.
  Retreat: §1.2 already addresses; CUDA lowering re-derives
  `block_tables` from `req_id`. The IR carries the
  *backend-neutral position* (= how many tokens of KV exist for this
  request).
- **R3 — Spec-decode tests (`spec_decode_correctness`,
  `magicdec_self_spec_integration`) regress under the new IR**.
  Mitigation: S3 is shadow-mode (emit IR alongside, do not switch). S4
  flips the switch only after CPU diff is empty for a fixed seed +
  prompt set (target test added in S1).
- **R4 — Performance regression on CUDA hot path**.
  Mitigation: M3 is structurally a renaming + indirection through
  trait. Hot path is unchanged. Run `bench_guidellm.sh --quick` before
  and after S4; tolerate ≤1% throughput delta.

## 5. Definition of Done

- `cargo test --release -p infer --features cuda --test e2e` passes.
- `INFER_DETERMINISTIC=1 cargo test --release -p infer --features cuda --test greedy_consistency` passes (B=1≡B=3, byte-exact baseline).
- `cargo test --release -p infer --features cuda --test spec_decode_correctness` passes (4 ok).
- `cargo check -p infer --no-default-features --features metal,no-cuda` passes (Metal Rust typecheck on Linux).
- `infer/src/scheduler/plan.rs` is the only place where decode/prefill row schemas are defined; old shape definitions are deleted.
- wins entry recording the file delta + bench Δ%.

## 6. Open questions for next manager review

- **Q1**: Should `LogicalServePlan` also carry `scheduler_signals` (admission policy state)?
  Current proposal: NO — admission policy operates on the scheduler request structs;
  the plan is the *output* of admission, not its input.
- **Q2**: Should the IR support more than two phases (e.g. spec-verify as a third phase)?
  Current proposal: NO — spec-verify is a sub-mode of decode (carried in `LogicalSpecDecodeRow` annotations on existing decode rows). Adding a phase would needlessly fan out the lowering trait.

## 7. References

- Parent: [`backend-unification.md`](backend-unification.md) §M3
- Companion: [`M_b-tilelang-fused-draft-verify-kernel.md`](M_b-tilelang-fused-draft-verify-kernel.md)
- Existing CUDA decision shape: `infer/src/scheduler/batch.rs::ScheduleDecision`
- Existing Metal decision shape: `infer/src/backend/metal/plan.rs::MetalLogicalServePlan`
- Production CUDA path that bypasses the above: `infer/src/scheduler/cuda/core.rs::step`
