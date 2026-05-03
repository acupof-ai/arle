# B2 — Multi-Prefill-Per-Tick for Metal Scheduler

**Status:** `planned — 待落地`. Successor to B1 (commit `669d9c4`) which closed the Qwen3.5 in-memory prefix-publish silent-miss but left the Phase 3 acceptance gate (warm TTFT p99 ≤ 4s, prefix_hit_rate ≥ 0.7) unmet because the Metal scheduler still emits ≤ 1 prefill row per tick.

**Owner:** unassigned (next Track A commit per [`2026-05-03-bench-metal-qwen35-prefix-publish-fix.md`](../experience/wins/2026-05-03-bench-metal-qwen35-prefix-publish-fix.md) handoff).

**Created:** 2026-05-03.

---

## 1. Goal + Acceptance Gates

### Goal

Lift the legacy `MetalScheduleStep` invariant of "at most one prefill row per tick" so that, under heavy concurrent admission (c=16 W3 trace), multiple Qwen3.5 requests can pack their prompts into one batched prefill forward through the existing C++ Qwen3.5 step bridge. The bottleneck quoted in the wins entry: cold p50 = 22 s for 1024-token prompts at c=16 → effective system prefill ≈ 745 tok/s vs. mlx-lm Qwen3.5-4B M4 Pro ceiling ≈ 5 000 tok/s. Roughly 7× headroom on cold/tool-call TTFT.

### Acceptance gates (re-stated from the wins entry)

| gate | source | current | target |
|---|---|---:|---:|
| W3 warm TTFT p99 (ms) | `2026-05-02-bench-agent-load-a1-session-affinity-admission.md` | 29 380 (post-B1) | **≤ 4 000** |
| W3 prefix_hit_rate (post B1+B1.5) | wins/2026-05-03 §Δ | 30.4 % | **≥ 0.7** (jointly with B1.5 per-session dedup; B2 alone cannot move hit-rate, but unblocks the floor) |
| P1 c=1,2,4,8 sweep regression | `bench_guidellm.sh metal-b1-agent-short` | within ±5 % | **stay within ±5 %** |
| Greedy-consistency test | `infer/tests/greedy_consistency.rs` | passing | **passing** |
| `cargo test --release --no-default-features --features metal --lib` | local | 614 / 0 / 24 | **≥ 614 / 0 / 24** |

### Explicit non-goals

- Not changing `MetalSchedulerConfig::max_batch_tokens` defaults — the existing 512-token tick budget is the right pool, B2 just stops paying it on a single row.
- Not introducing a separate "chunked prefill" mode — packed prefill across rows is the architecturally cheaper lever (see §11). True per-row chunking (split one >budget prompt across ticks at fixed chunk size N<512) is a follow-up.
- Not touching the Qwen3 Rust path. Qwen3 already has a packed-prefill `try_mixed_batch` path at `infer/src/backend/metal/request_state.rs:1296-1374` that could be generalized later but is independently scoped.

---

## 2. Current Single-Prefill Flow — How `≤1 prefill row` Is Enforced

### 2a. The hard gate

The cap is enforced in exactly one place: [`infer/src/backend/metal/scheduler.rs:160-181`](../../infer/src/backend/metal/scheduler.rs).

```rust
fn from_logical_plan(plan: MetalLogicalServePlan) -> Self {
    debug_assert!(
        plan.prefill_rows.len() <= 1,
        "legacy MetalScheduleStep supports at most one prefill row"
    );
    ...
    let prefill = plan.prefill_rows.first().map(|row| MetalPrefillChunk { ... });
    Self { decode, prefill, plan }
}
```

`MetalScheduleStep` itself stores `prefill: Option<MetalPrefillChunk>`, not `Vec<MetalPrefillChunk>` (`scheduler.rs:140-145`). The logical plan layer (`plan.rs`) already supports `prefill_rows: Vec<MetalLogicalPrefillRow>` (see `plan.rs:104-126` and `MetalLogicalBatchShape` aggregation at `plan.rs:81-100`), so the legacy DTO is genuinely the only gate left between the planner and the runtime.

### 2b. The planner only ever picks one

`MetalScheduler::build_prefill_row` (`scheduler.rs:379-442`) is wired to return `Option<MetalLogicalPrefillRow>` (singular), and `build_logical_plan` wraps that in a 0-or-1 `prefill_rows` Vec at `scheduler.rs:329-333`. The planner already rotates — at most one prefilling request is "in flight" at a time:

- `find_prefilling_request` (`scheduler.rs:461-474`) finds the existing in-progress prefill (if any).
- Otherwise `admit_next_waiting_request` (`scheduler.rs:476-489`) pulls the head of `waiting`.
- `prefill_chunk_budget(decode_count)` (`scheduler.rs:203-205`) gives the prefill row whatever's left of `max_batch_tokens` after decode.

### 2c. The runtime assumes singular prefill

Everywhere downstream of the scheduler step, the singularity is structurally baked in:

- `run_metal_scheduler_runtime` reads `step.prefill.is_some()` for metrics (`runtime.rs:1260-1264`).
- `guard_schedule_step` matches on `(step.decode, step.prefill)` as `(Option<…>, Option<…>)` (`runtime.rs:1356-1416`).
- `execute_mixed_batch` takes a single `prefill_req_id` + `prefill_budget` (`runtime.rs:1496-1665`).
- `execute_prefill_chunk` takes a single `req_id` + `budget` (`runtime.rs:1835-1947`).

### 2d. Why the cap exists (legacy reasoning)

The C++ Qwen3.5 step path (`crates/mlx-sys/src/mlx_qwen35_model.cpp:2081-2147`, `2330-2379`) was originally designed around a single live session: `qwen35_session_begin` → `qwen35_compiled_prefill_session` → `qwen35_compiled_step_session`. The MLX bridge holds exactly one `m->session_kv_caches` and one `m->session_gdr_states` set at a time. The Rust side `Qwen35CppState` (`request_state.rs:3680-3713`) reflects that with `session_active: bool`. The "one prefill at a time" rule kept the implementation simple: prefill drains every other request's session before running (`drain_other_qwen35_cpp_sessions` at `runtime.rs:1949-1971`), runs its own prefill in the now-exclusive C++ session, and decodes peel back into batched `qwen35_compiled_step_batch_packed`.

In other words: the bridge has a per-session stateful resource (`session_*`), which forces serialization of any prefill whose KV is going to live in that session. A "second prefill row" today would either (a) contend on `session_active` or (b) require a session-less batched prefill primitive that doesn't yet exist on the C++ side.

### 2e. Other downstream assumptions to audit

- **GDR state.** `Qwen35DFlashState::target_hidden` is captured per request via `with_qwen35_capture_layers` (see `request_state.rs:4280-4291`). A multi-prefill row design must not mix capture layers across requests. The B1 P1 lesson on hybrid GDR state (wins/2026-05-03 §Learnings) is the same surface here: GDR can't be silently mixed across rows without losing per-request consistency.
- **Per-row max-tokens accounting.** Each request's KV is pre-allocated to `prompt_len + max_new_tokens` (see `Qwen35StepDriver::new` `request_state.rs:3749-3754`). Packing prefill rows that have different `kv_capacity` requires the same left-padding + admit-rows path that decode already uses (`request_state.rs:873-1010`). The B1 P2 lesson — the cache-eviction `snapshot_footprint(...) = max(token_count, kv_capacity)` accounting — applies symmetrically: a packed-prefill batch's resident footprint must charge against `max_batch_tokens` correctly, not against the smallest row's `prompt_len`.
- **16-block alignment.** `METAL_PREFIX_BLOCK_SIZE = 16` (`runtime.rs:251`) is the published-snapshot block alignment from B1 and matches the paged-pool page size on CUDA. A multi-prefill packed row layout must keep `cache_len` divisible by 16 at publish time (the B1 v2 fix locked this in for the in-memory tier).

---

## 3. Proposed Multi-Prefill Design

### 3a. `MetalSchedulePlan` shape

Replace `MetalScheduleStep::prefill: Option<MetalPrefillChunk>` with a small Vec carrying ≥0 prefill rows. The plumbing already exists on the planner side via `MetalLogicalServePlan::prefill_rows: Vec<…>`.

```rust
// scheduler.rs (sketch — design only, not committed)
pub struct MetalScheduleStep {
    pub decode: Option<MetalDecodeBatch>,
    pub prefill: Vec<MetalPrefillChunk>,   // was Option<…>; up to N per tick
    plan: MetalLogicalServePlan,
}
```

The `debug_assert!(prefill_rows.len() <= 1, …)` at `scheduler.rs:162` deletes. The runtime-side match at `runtime.rs:1356-1416` collapses to one branch with two helpers:

- `guard_packed_prefill_batch(prefill_rows, …)` — 1+ rows.
- `guard_decode_batch(…)` — unchanged.
- `guard_mixed_batch(…)` — generalized to take `prefill_rows: &[MetalPrefillChunk]` (the existing single-prefill mixed path keeps working as the N=1 case; for N≥2 the C++ batched-prefill primitive lands first — see §3b).

`build_prefill_row` becomes `build_prefill_rows`. The token budget changes from "what's left after decode" to "what's left after decode, divided across up to `max_concurrent_prefill_rows` rows" — concretely: a new `MetalSchedulerConfig::max_prefill_rows: usize` (default 4 to match existing CUDA `prefill_max_requests` default; CUDA scheduler agents.md §5 calls this out as a budget knob).

### 3b. C++ Qwen3.5 bridge — packed batched prefill

The C++ side already has a packed batched **decode** primitive: `qwen35_compiled_step_batch_packed` (`mlx_qwen35_model.cpp:2253-2328`) which accepts a packed `[batch, n_kv_layers, …]` KV layout, a packed `[batch, n_gdr_layers, …]` GDR layout, an additive `attn_mask`, and per-row `rope_offsets`. This is exactly the shape we need for prefill, only with `current_seq_len = chunk_len` instead of `1`.

Add one symmetric primitive:

```c
// crates/mlx-sys/src/mlx_qwen35_model.cpp (sketch)
int32_t qwen35_compiled_prefill_batch_packed(
    void* model,
    mlx_array* token_ids,           // int32 [batch, max_chunk_len]
    int32_t batch_size,
    int32_t max_chunk_len,
    const int32_t* cache_pos_arr,   // i32[batch] — per-row write cursor
    const int32_t* prompt_len_arr,  // i32[batch] — per-row valid length
    mlx_array** packed_kv_caches,   // [batch, n_kv_heads, kv_capacity, head_dim] x n_kv
    int32_t n_kv,
    mlx_array** packed_gdr_states,
    int32_t n_gdr,
    mlx_array* attn_mask,           // [batch, 1, max_chunk_len, key_len]
    mlx_array* rope_offsets,        // i32[batch] — per-row starting RoPE position
    mlx_array** out_logits,         // [batch, 1, vocab] — last-token only
    mlx_array** out_packed_kv_caches,
    mlx_array** out_packed_gdr_states
) { ... }
```

Why this is a small surface change:

- The `current_batch_size` / `current_seq_len` / `current_attn_mask` / `current_rope_offsets` fields on `Qwen35CompiledModel` (see usages at `mlx_qwen35_model.cpp:2272-2295`) already control the forward path; setting them per call already drives `step_batch_packed`'s shape selection. A new entry point reusing those fields for `seq_len > 1` keeps the model-graph code path unchanged in spirit.
- `last_logits_only` is already wired through `use_qwen35_cpp_prefill_last_logits_only()` (single-prefill path at `mlx_qwen35_model.cpp:2102`). The packed variant does the same — last-logit gather per row given the per-row valid length.

The Rust-side wrapper in `infer/src/backend/metal/qwen35.rs` adds `prefill_batch_packed` next to `step_batch_packed` (see existing pattern at `qwen35.rs:1794-1852` for shape).

### 3c. Per-row mask + RoPE — reuse §7 of metal AGENTS.md

The varlen packed-decode pattern (`AGENTS.md` §7) is exactly the pattern a packed-prefill needs, with a **2D mask** instead of a 1D one:

- `attn_mask` becomes `[batch, 1, max_chunk_len, key_len]` (vs `[batch, 1, 1, key_len]` for decode). The same `build_varlen_decode_mask` helper in `mlx.rs` generalizes to `build_varlen_prefill_mask(left_padding, max_chunk_len, batch_cache_len)` — additive `-inf` on padded query positions and on padded key columns.
- `rope_offsets` stays `int32[batch]`, where row `b`'s logical position range is `[rope_offsets[b], rope_offsets[b] + chunk_len_b)`. The C++ side derives the per-row position grid by adding the offset to a `[chunk_len]` arange. **Critically**: per the docs/experience/errors/2026-04-16 retrospective, even same-length batches must use array-offset RoPE because MLX 0.31.1's scalar-offset path silently drops `B>0` rows on `[B, H, S=1, D]`. The same defensive approach for `S>1` is the right default — array-offset always.

### 3d. Memory budget — don't blow `cache_mem`

The B1 P2 fix locked `cache_mem` to 1.0 GB (down from buggy 9.0 GB) by accounting `max(token_count, kv_capacity)` per snapshot (`runtime.rs:290-293`). Multi-prefill in one tick must not regress that:

- The packed prefill batch's transient `[batch, n_kv_heads, max_chunk_len, head_dim]` working set is per-tick, not retained — it does not enter the snapshot pool. ✓
- Each row's KV/GDR is the same per-request `kv_capacity = prompt_len + max_new_tokens` it would have allocated under the singleton path. Total resident = sum across active rows, identical to today's c=16 case. ✓
- The publish path (`MetalLivePrefixRuntime::publish_prompt_prefix`, `runtime.rs:400-405` → `Qwen35StepDriver::export_qwen35_live_prefix_snapshot`, `request_state.rs:3460-…`) drains ONE session at a time. With multi-prefill, publish has to drain all the rows that just transitioned out of `Prefill` → `Decode`. **The current per-tick "publish if a token was emitted" flow at `runtime.rs:1927-1934` already handles N=1 correctly; the N≥2 generalization is a `for prefill_request in just_finished_rows { publish_prompt_prefix }` loop.**

The risk surface around the pool sizing constant `METAL_PREFIX_POOL_MULTIPLIER = 64` (`runtime.rs:258`) is unchanged: working-set cardinality is bounded by `max_running_requests`, not by prefill-rows-per-tick, so the W3 64-warm-session pool sizing carries over.

### 3e. Interaction with B1 prefix-snapshot publish

The B1 publish call site (`execute_prefill_chunk`, `runtime.rs:1927-1934`) fires only after the terminal prefill chunk emits its first decode token. With multi-prefill packed batches, several rows can transition `Prefill → Decode` in the same tick. The N=1 path is unchanged (it happens to be the new path's degenerate case). For N≥2:

1. The packed C++ call returns last-logits per row.
2. Each row that was `terminal_prompt = true` (chunk reached `prompt_len`) gets its sampled token via `gpu_sample_token` over its row of the logits batch.
3. For each such row, `prefix_runtime.publish_prompt_prefix(row)` runs — same single-session drain → snapshot path B1 fixed.

There is no "one big snapshot for the batch": the in-memory cache is keyed by `prompt_tokens` per request, not by batch identity. **The B1 v2 correctness rule (snapshot at exactly the live `cache_len`, never at a truncated `aligned_len`) is preserved row-by-row.**

### 3f. Mixed prefill+decode interaction

Today: `execute_mixed_batch` (`runtime.rs:1496-1665`) packs a singleton prefill row with the decode batch through `MetalRequestState::try_mixed_batch` (Qwen3-only — Qwen3.5 returns `Ok(None)`, see `request_state.rs:1257-1287`). For B2 we have two clean choices:

- **Option A — split.** Decode rows go through `qwen35_compiled_step_batch_packed`; prefill rows go through the new `qwen35_compiled_prefill_batch_packed`; both within the same tick but back-to-back, not co-batched. Two C++ entries per tick. Simpler, recommended for first commit.
- **Option B — true mixed.** Pack decode (`seq_len = 1`) and prefill rows (`seq_len = chunk_len`) into a single batched call. Requires the C++ side to support varlen *query* length within one batch — that's a meaningful new shape (existing `step_batch_packed` is `seq_len = 1` flat). Costs another C++ entry plus mask/RoPE generalization. Out of scope for B2; document as a B2.5 follow-up.

**Recommendation: Option A.** Decode keeps using `step_batch_packed`; prefill rows go through the new `prefill_batch_packed`; the scheduler emits both in the same `MetalScheduleStep`; the runtime executes them sequentially in one tick. This preserves the "decode-first" priority semantics that `MetalScheduleStep`'s docstring spells out (`scheduler.rs:138-145`) while delivering the multi-prefill throughput win.

---

## 4. CUDA-side Reference

The CUDA scheduler already does multi-prefill-per-tick. Useful shape references (do **not** copy code; learn the architecture):

- **Plan-level multi-row prefill enum.** `infer/src/scheduler/cuda/execution.rs:30-37` — `StepPlan::Prefill(Vec<PrefillCandidate>)`, `Split(Vec<…>)`, `Mixed(Vec<…>)`. Note that `Vec<PrefillCandidate>` carries N≥1 — Metal's `MetalScheduleStep.prefill: Vec<MetalPrefillChunk>` is the symmetric move.
- **Token+page budget feasibility.** `cuda/execution.rs:94-162` (`PrefillBudget::can_schedule` / `reserve`) shows how CUDA gates each candidate against a shared step token budget AND a shared page budget. Metal doesn't have a paged pool, so the page budget collapses; only the token budget remains. CUDA's `PrefillBudget::from_scheduler_for_decode_slots` is a cleaner factoring than Metal's current ad-hoc `prefill_chunk_budget(decode_count)` (`scheduler.rs:203-205`) and is worth mirroring.
- **Candidate scoring/selection.** `cuda/execution.rs:164-208` — `score_prefill_candidates` + `select_prefill_candidates` + `cap_prefill_candidates_by_tokens`. Stable-rank by queue order; reserve until budget is exhausted; cap a final mixed-prefill subset by `mixed_prefill_token_budget`. Metal's planner can lift the same shape; the only Metal-specific twist is `max_running_requests` (decode + prefilling combined) instead of CUDA's separate `prefill_max_requests` knob.
- **Batched prefill request shape.** `cuda/prefill.rs:480-547` (`prepare_prefill_batch`) — collects per-slot `(slot_idx, tokens_chunk, total_tokens, next_progress)`. The Metal symmetric collection is `Vec<MetalLogicalPrefillRow>`.
- **Async/sync split.** `cuda/prefill.rs:649-718` shows sync vs async dispatch. Metal can stay synchronous for B2 (MLX is lazy-eval'd anyway; `eval`/`async_eval` boundaries already serve the same role — see `request_state.rs:3574-3581`).
- **Model-side multi-row entrypoint.** `infer/src/model.rs:435-457` — `ModelForward::forward_prefill_batch` takes `&[PrefillBatchRequest<'_>]`, falls back to a per-request loop when the model can't truly batch. Qwen3.5's CUDA implementation (`infer/src/model/qwen35/forward.rs:429-453`) overrides with a real batched paged-prefill kernel. The Metal C++ `prefill_batch_packed` is the symmetric override on the Metal side.

---

## 5. Phased Implementation — 3 Commits, ≤ 4 Files Each, Each Bench-Clean

### Commit 1 — `feat(metal): generalize MetalScheduleStep to ≥0 prefill rows`

**Goal:** Plumbing-only. The DTO change. No behavior change at runtime; first commit must remain N=1 in practice and bench-clean against the B1 baseline.

**Files (4):**

- `infer/src/backend/metal/scheduler.rs` — drop `debug_assert`; `MetalScheduleStep.prefill: Vec<MetalPrefillChunk>`; rename `build_prefill_row` → `build_prefill_rows` (returns 0-or-1 entry vec); update `from_logical_plan`; update unit tests.
- `infer/src/backend/metal/plan.rs` — no change (already supports `Vec<…>`).
- `infer/src/backend/metal/runtime.rs` — `guard_schedule_step` matches on `(decode, prefill_rows.as_slice())`; `execute_*` helpers that take a singular prefill take `&[MetalPrefillChunk]` of length 1; metrics keep counting `step.prefill.len()`.
- `infer/src/backend/metal/runtime/` (no new file — keep the existing flat module layout per CLAUDE.md §Code conventions).

**Verify:** `cargo test --release --no-default-features --features metal --lib`; `bench_guidellm.sh metal-b2-c1-noop` against `metal-b1-agent-short` baseline → expect numerical noise only (±5 %). Wins entry: `2026-05-?-bench-metal-b2-c1-plumbing.md`.

### Commit 2 — `feat(mlx-sys,metal): qwen35_compiled_prefill_batch_packed C++ entry + Rust wrapper`

**Goal:** Land the C++ batched-prefill primitive plus its Rust wrapper. Still no scheduler-side multi-prefill — this commit is provably regression-clean because nothing calls the new entry yet.

**Files (4):**

- `crates/mlx-sys/src/mlx_qwen35_model.cpp` — add `qwen35_compiled_prefill_batch_packed` next to `qwen35_compiled_step_batch_packed`; reuse `current_seq_len` / `current_attn_mask` / `current_rope_offsets` plumbing, write `last_logits_only=true` inside.
- `crates/mlx-sys/src/lib.rs` — add the FFI declaration after `qwen35_compiled_prefill` (line 743).
- `infer/src/backend/metal/qwen35.rs` — add `prefill_batch_packed` Rust wrapper next to `step_batch_packed` (line 1794). Mirror the `step_batch_packed` shape — same packed KV/GDR contract, same array-offset RoPE.
- `infer/src/backend/metal/mlx.rs` — generalize `build_varlen_decode_mask` to `build_varlen_prefill_mask(left_padding, max_chunk_len, key_len)` returning `[B, 1, max_chunk_len, key_len]` additive mask. Existing decode helper continues to work as the `max_chunk_len = 1` degenerate case.

**Verify:** `cargo test --release --no-default-features --features metal --lib` (a new unit test on the entry — small fake-graph round-trip is fine). Bench: not applicable — no hot-path consumer yet, this is a `pending-remote`-style entry: declare it exempt in commit body per CLAUDE.md §Benchmarks (no runtime hot path touched). The next commit pays the bench debt.

### Commit 3 — `feat(metal): scheduler emits N prefill rows; runtime dispatches via packed batch`

**Goal:** Wire the new C++ entry into the runtime. The actual win commit.

**Files (4):**

- `infer/src/backend/metal/scheduler.rs` — `MetalSchedulerConfig::max_prefill_rows` (default 4); `build_prefill_rows` now scans the waiting queue + in-progress prefill rows and returns up to `max_prefill_rows` with the shared step token budget split among them via a `PrefillBudget`-style helper modelled on `cuda/execution.rs:94-162`.
- `infer/src/backend/metal/runtime.rs` — `execute_prefill_packed_batch(&[MetalPrefillChunk], …)` replaces the singleton `execute_prefill_chunk` (single-row case becomes N=1 in the new helper); `drain_other_qwen35_cpp_sessions` is called once per tick before the packed call (not per row); per-row publish loop after the call; `execute_mixed_batch` keeps Option A (sequential decode-batch + prefill-batch within one tick — see §3f).
- `infer/src/backend/metal/request_state.rs` — new `MetalRequestState::try_prefill_packed_batch(states: &mut [&mut Self], chunk_lens: &[usize]) -> Result<Option<Vec<PrefillChunkResult>>>` that constructs the packed prefill batch (left-pad rows by `cache_len` delta to a shared `batch_cache_len` cursor exactly like `admit_rows` at `request_state.rs:873-1010`), calls the new C++ entry, splits last-logits per row, and runs `gpu_sample_token` per terminal-row. Mirrors the contract of `try_decode_qwen35_packed_batch` (line ~1464).
- `infer/src/backend/metal/qwen35.rs` — minimal additions for any helper the request-state side needs that doesn't already exist (e.g. extracting one row's logits from a packed `[batch, 1, vocab]`).

**Verify:** `cargo test --release --no-default-features --features metal --lib` + `cargo test --release --no-default-features --features metal --test e2e_qwen35`. Bench: full W3 trace replay (`scripts/bench_agent_trace.py --workload agent-w3-short-multiturn`), P1 c=1,2,4,8 sweep regression check. Acceptance gate target: warm TTFT p99 ≤ 4 s. Wins entry: `2026-05-?-bench-metal-b2-multi-prefill.md`.

---

## 6. Risk Register

| risk | source | mitigation |
|---|---|---|
| **GDR state corruption across packed prefill rows.** Each row carries an independent recurrent-accumulator state; mixing rows requires the C++ side to keep `gdr_states` as a packed `[batch, …]` tensor with no cross-row attention. | B1 P1 lesson: hybrid recurrent state cannot be silently truncated/mixed (`wins/2026-05-03 §Learnings`). | Reuse the `step_batch_packed` GDR contract verbatim — it already keeps per-row GDR independent (see `mlx_qwen35_model.cpp:2310-2317`). Add a single-row vs batched parity test (run prompt P alone vs. P padded into a 4-row batch with 3 dummy rows; logits on row 0 must match within bf16 noise). |
| **Per-row `kv_capacity` / `max_new_tokens` mismatch.** Different requests have different `kv_capacity`; packing must bump every row to `max(kv_capacity)` or it underflows during decode. | B1 P2 lesson: cache eviction under-accounting (`wins/2026-05-03 §Learnings`). | Reuse the `Qwen35PackedDecodeBatch::ensure_capacity_for_states` pattern (`request_state.rs:810-833`). The packed-prefill batch picks the row with the largest `kv_capacity` and grows the others to match. **Pre-size accounting:** the per-tick prefill token budget already caps total work at `max_batch_tokens`; the per-row `kv_capacity` extension is independent of token budget and must not be charged against it. |
| **16-block alignment for snapshot publish.** Sub-block prompts already skip publish (B1 v2 fix). With multi-prefill, the row that finishes off-boundary must not corrupt the publish loop. | B1 v1 → v2 lesson: GDR vs. KV consistency at snapshot time (`wins/2026-05-03 §Learnings`). | The publish call in commit 3 is a per-row loop; rows whose `cache_len` is sub-block fall out of the publish branch the same way they do today. Preserved by construction — no new logic. |
| **`drain_other_qwen35_cpp_sessions` race.** Today this drains every other request's session before a single-row prefill. With multi-prefill, draining must happen once per tick, not once per row, or rows in the same tick fight for the session. | B1 P1 silent-correctness pattern (drain or be silently dead). | Move the drain call to the top of `execute_prefill_packed_batch`, run once for the whole tick. Add a `debug_assert!(!any_other_session_active)` at the entry to the C++ packed call. |
| **Mixed-batch path (Option A) drops the existing Qwen3-only `try_mixed_batch`.** The current Qwen3 mixed-decode-+-prefill packed path at `request_state.rs:1296-1374` would still run when the prefill row count is exactly 1 and rows are Qwen3. | (no past incident — design risk) | Leave `try_mixed_batch` untouched; the new packed-prefill-batch path is Qwen3.5-only for B2. The Qwen3 side keeps its existing `Option<MetalPrefillChunk>` semantics until a B2.5 generalizes mixed packing. The DTO is now `Vec<…>` but Qwen3-only paths take the `len == 1` slice. |
| **`max_prefill_rows` default chosen wrong.** Too low → no win; too high → memory pressure on M4 Pro 48 GB. | (no past incident — tuning risk) | Default `max_prefill_rows = 4` (matches CUDA `prefill_max_requests`-class defaults). Wins entry sweeps {2, 4, 8} and reports the elbow. **Don't bump beyond 8 in B2;** if the bench shows residual headroom there, ship it as B2.1. |
| **Bench reproducibility on Mac.** B2 is intrinsically Metal — `--features metal` only — and the W3 replay needs >13-min wall on M4 Pro 48 GB. | CLAUDE.md §Benchmarks: bench-or-`pending-remote`. | Local M4 Pro 48 GB run is the canonical bench host. Cite the same host as B1 for delta comparability. |

---

## 7. Open Questions for the User

These are decisions outside what the planner can settle alone:

1. **Hit-rate vs. correctness on first commit.** Commit 1 (plumbing-only) and commit 2 (C++ entry, no consumer) are both 0-impact on warm TTFT. **Commit 3 is the entire perceived win** — there is no half-state where a partial commit lands a partial improvement. Is that acceptable, or should commit 3 be split further (e.g. "ship with `max_prefill_rows = 2` first, raise to 4 in a follow-up after one bench cycle")?

2. **C++ vs. Rust boundary for the packed prefill entry.** Two viable shapes:
   - **(a) New C++ entry `qwen35_compiled_prefill_batch_packed`** (recommended, §3b). Reuses the existing forward graph; one new ~50-line C++ function + Rust wrapper.
   - **(b) Rust-side N×scalar `prefill_session` calls in a loop**, surrounded by one shared `eval` boundary. Zero C++ change, but it loses the GPU-batching win — every row launches its own MLX graph. This is the "cheaper to ship, doesn't actually solve B2" path. **Recommend rejecting** unless C++ build cost is a hard blocker.

3. **Reuse the existing varlen packed-decode pipeline, or land a parallel prefill pipeline?** The text above proposes a parallel `qwen35_compiled_prefill_batch_packed` rather than passing `seq_len > 1` into the existing `qwen35_compiled_step_batch_packed`. Reasons: (i) `step_batch_packed`'s `current_seq_len = 1` is asserted in several places (`mlx_qwen35_model.cpp:2278`); (ii) prefill returns last-logits-only by convention, decode returns next-token logits — different shapes; (iii) DFlash uses tape-mode + capture-layers during prefill (see `request_state.rs:4280-4291`), and folding that into the decode entry confuses the contract. Acceptable to land as a separate entry?

4. **Is true mixed batching (Option B in §3f) required to hit the gate?** Option A (sequential decode-then-prefill within one tick) preserves the "one prefill phase per tick" cost relative to today, just amortized over N prefill rows. If 4× headroom isn't enough to hit warm TTFT p99 ≤ 4 s on W3, Option B becomes the next lever — but it's a B3-class commit, not a B2 fix.

5. **`max_prefill_rows` config exposure.** Land it as a hardcoded constant (like the B1 `METAL_PREFIX_POOL_MULTIPLIER`) or as a CLI flag from the start? B1.5/B1.6 follow-ups already plan to expose `--max-prefix-pool-tokens`; piggy-backing `--metal-max-prefill-rows` is cheap.

---

## 8. Composition

- Stacks **on top of** B1 (`669d9c4` — wins/2026-05-03). Without B1 the publish path is silent and warm-TTFT is unmeasurably bad regardless of B2; B2 alone moves cold-TTFT but not warm.
- **Independent of** B1.5 (per-session snapshot dedup) and B1.6 (pool-capacity CLI flag), but composes cleanly: B2 fixes the prefill-floor on cold/cache-miss turns, B1.5 raises hit-rate so fewer turns hit that floor.
- **Does not touch** Qwen3 (Rust-path) prefill or the DFlash speculative decode runtime. Both stay on their current single-prefill-row contract until a follow-up B2.5 generalizes.
- **Unblocks** a future B3 (true mixed packing, Option B in §3f) by getting the packed-prefill C++ entry into the bridge.

---

## 9. Pointers

- B1 result + B2 statement: [`docs/experience/wins/2026-05-03-bench-metal-qwen35-prefix-publish-fix.md`](../experience/wins/2026-05-03-bench-metal-qwen35-prefix-publish-fix.md).
- Metal varlen pattern: [`infer/src/backend/metal/AGENTS.md`](../../infer/src/backend/metal/AGENTS.md) §7.
- CUDA reference architecture: [`infer/src/scheduler/cuda/execution.rs:30-208`](../../infer/src/scheduler/cuda/execution.rs).
- Snapshot semantics + GDR state correctness: [`docs/experience/errors/2026-04-16-metal-varlen-rope-blocker.md`](../experience/errors/2026-04-16-metal-varlen-rope-blocker.md), wins/2026-05-03 §Learnings.
- No-half-states discipline: `feedback_no_half_states.md` (referenced from CLAUDE.md §Editing).
- Bench spec: [`docs/bench-and-trace-spec.md`](../bench-and-trace-spec.md).

---

## 10. Acceptance Criteria

- All three commits land independently bench-clean (commit 1 noise-band regression check; commit 2 declared exempt with rationale in commit body; commit 3 W3 + P1 sweep + e2e_qwen35).
- `cargo clippy --release -p infer --no-default-features --features metal -- -D warnings` clean across the series.
- Final wins entry shows W3 warm TTFT p99 ≤ 4 s (or, if not met, an honest gap analysis pointing at Option B / B3 as the next lever).
- The plan-to-bench cross-link is updated in this file's "status" header from `planned` → `landed: <commit-sha>`.
- B1.5 and B1.6 follow-ups explicitly cited in the wins entry as un-shipped (so reviewers know the hit-rate gate isn't B2's responsibility).

---

## 11. Is B2 Actually Two Levers? — Recommendation

Yes — and they should phase as **multi-prefill first, chunked prefill second**:

- **Lever 1 (this plan, B2):** N prefill rows per tick, each row covering its remaining prompt up to a per-row share of the tick token budget. **GPU batching win.** The 7× headroom number in the B2 statement (745 → 5000 tok/s) measures this lever specifically — the M4 Pro mlx-lm ceiling assumes an MLX batched-prefill graph, not chunking.
- **Lever 2 (defer, B2.5):** Chunked prefill — split a single >budget prompt across multiple ticks at fixed chunk size, so a 4 k-token prompt doesn't monopolize the tick. This is the *latency* lever (helps tail TTFT under contention) but does not raise *throughput*.

The two are independent. B2's W3 trace has 1 024±32 token prompts and `max_batch_tokens = 512` default — every prompt today already fits in ~2 chunks. Chunking is already implicitly happening via the existing `prefill_chunk_budget` cursor (`scheduler.rs:203-205`); what's missing is *parallelism across rows*, not *more chunks per row*. **Land lever 1 first; revisit chunking only if W3 wins-entry shows residual TTFT contention from one big-prompt request blocking decode rows.**
