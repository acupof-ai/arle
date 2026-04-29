# Runtime Resource Architecture — design pass

> Date: 2026-04-29.  Owner: runtime / scheduler.  Status: design (no code).
> Inputs: [`projects/2026-04-29-perf-bug-roundup.md`](../projects/2026-04-29-perf-bug-roundup.md),
> [`projects/2026-04-29-scheduler-pipeline-map.md`](../projects/2026-04-29-scheduler-pipeline-map.md),
> [`projects/2026-04-29-throughput-gap-analysis.md`](../projects/2026-04-29-throughput-gap-analysis.md),
> [`errors/2026-04-29-bf16-shadow-mixed-architectural-dead-end.md`](../experience/errors/2026-04-29-bf16-shadow-mixed-architectural-dead-end.md).

K1–K10 share one root cause: **GPU resources have no unified ownership
model with cleanup-on-error**.  Slot, KV pages, FlashInfer workspace,
recurrent state and the prefill context each have their own free path;
none fire on `?`-propagation.  This design replaces ad-hoc cleanup with
one ownership graph (A) + one budget gate (B) + one attention contract
(C) + one auto-config entry point (D).

---

## Area A — Resource ownership model (RAII-style)

### Current state

Each per-request GPU resource has a separate free path:

- Slot: `Scheduler.active: Vec<Option<ActiveRequest>>`
  (`infer/src/scheduler/cuda/core.rs:132`); freed by `take()` in
  `cleanup()` (`scheduler_loop.rs:283-353`).
- KV pages: `paged_kv_pool.free_slot(slot_idx)`
  (`scheduler_loop.rs:310`, `decode.rs:167`); `finish_slot` itself does
  NOT free pages (`core.rs:803-816`) — only `cleanup()` does, and only
  when phase is `Phase::Finished` AND `slot_has_pending_gpu_work` is
  false (`scheduler_loop.rs:285,288-291`).
- FlashInfer / mixed workspace: lazy via `ensure_mixed_buffers`
  (`infer/src/model/qwen3/batch_decode.rs:350,524`), can fail at
  `crates/cuda-kernels/src/flashinfer.rs:516,521`.
- Prefill context: `Scheduler.prefill_ctx` (`core.rs:144-145`); cleared
  on `complete_prefill_batch` (`prefill.rs:664`); on error left
  allocated and slot is `finish_slot`-only
  (`prefill.rs:583-592,602,621,639,666`).

K7 follows directly: `forward_prefill_batch` returns OOM →
`finish_prefill_batch_error` → `finish_slot` only — `ActiveRequest`
stays in `active[slot_idx]` until `cleanup()` clears it, but if pending
state lingers, the slot stays occupied indefinitely and new requests
return empty completions.

### Proposed design

`RequestResources` RAII guard owns every per-request allocation; Drop
releases everything atomically, including on `?` propagation.

```rust
// infer/src/scheduler/cuda/request_resources.rs (new)
pub(super) struct RequestResources<'a> {
    slot:        SlotLease<'a>,
    kv_pages:    KvPageLease<'a>,
    workspace:   Option<WorkspaceLease<'a>>,
    recurrent:   Option<RecurrentStateLease<'a>>,    // hybrid models
    prefill_ctx: Option<PrefillContextLease<'a>>,
}
// Drop order (reverse-decl): prefill_ctx → workspace → recurrent →
// kv_pages → slot.  slot must outlive kv_pages because free_slot reads
// page indices from the slot row.

pub(super) struct SlotLease<'a> { idx: usize, table: &'a SlotTable }
impl Drop for SlotLease<'_> { /* table.release(self.idx) */ }
```

Invariants:

1. `Scheduler.active[idx]` is `Some` iff a `SlotLease { idx }` is
   alive — compile-time, no `take()`-races.
2. Every `?` in the per-request pipeline carries the guard; early
   return runs Drop before the next scheduler tick.
3. The OOM error path collapses to `return Err(e)`; no bespoke
   `finish_prefill_batch_error` cleanup.

Migration target paths: `prefill.rs:583-671`, `decode.rs:113-178`,
`scheduler_loop.rs:283-353` (`cleanup()` becomes "drop completed
guards"; the `slot_has_pending_gpu_work` hedge goes away because the
guard is only released after readback consumes it).

### Migration order

1. **A1.** `SlotLease` + `SlotTable` over the existing `active` Vec —
   pure refactor, no semantic change.  Unblocks A2/A3.
2. **A2.** `KvPageLease` held by `ActiveRequest`; `paged_kv_pool.free_slot`
   moves into Drop.  **K7 closes here**, before A3/A4.
3. **A3.** `PrefillContextLease` (still server-shared internally; lease
   asserts "in use" so error paths can't leave half-state).
4. **A4.** `WorkspaceLease` for mixed/FlashInfer — server-lifetime
   today, leak benign; lands last.

### Tests

- `tests/oom_injection.rs` (new): inject N `flashinfer_alloc` failures,
  replay 100 admissions, assert `pool.free_count() == initial` and
  `active.iter().all(Option::is_none)`.  K7 regression gate.
- Property test in `core/tests.rs`: after every step,
  `count(SlotLease alive) == count(active.is_some())`.
- Drop-ordering unit test using shim leases that log to a
  `Vec<&'static str>`.

### Risk

- Borrowck across `&mut self` boundaries — model forwards take
  `&mut self`; introduce a `scheduler.split_borrow() -> (&mut Pool,
  &mut SlotTable, …)` helper.  Verify with `cargo check -p infer
  --release` clean before promoting A1.
- Drop ordering: slot must outlive its pages.  Reverse-decl order
  in the struct above encodes it.  Drop-order test catches
  regressions.

---

## Area B — Single capacity-budget admission with **lazy reservation**

### Critical principle (added 2026-04-29 per user feedback)

**Reserve at use-time, not at admission-time.** A request entering the
queue should NOT pre-claim its full
`(prompt_tokens + max_tokens) × bytes_per_token` of KV pool. That's
the bug today: every admitted request reserves its worst-case max-ISL
upfront, so 16 admitted requests claim 16 × 4352 × bytes_per_token
even though most are sitting idle waiting for their prefill turn.
This wastes pool, blocks would-be admittees, and is the source of
the "tune `chunked_prefill_size` to avoid OOM" workaround in the
2026-04-29 KV-quant matrix wins entry — which the user correctly
flagged as **NOT** a chunked-prefill issue, it's an
eager-reservation issue.

The right model:

- **Admission** checks only "is there a slot + can the next *step*
  fit?" — a small immediate cost, not the full lifetime cost.
- **Per-step reservation** grows the request's KV claim only as
  the step actually appends tokens. Decode reserves +1 token's
  worth; chunked prefill reserves +chunk_size's worth.
- **Backpressure** at step-time: if the next step's reservation
  would exceed the pool, the scheduler defers (admits another
  decode-only step from a different request, or stalls and waits
  for a finished request to free pages).
- **Hard 503** only when even a *minimum-step* admission would
  OOM the pool — i.e., the system is genuinely overloaded.

This is exactly SGLang's PagedAttention model: pages are appended
on demand by `alloc_tokens` per step, never pre-claimed for the
remaining max-tokens.

### Current state

Admission **eagerly reserves** the full ISL footprint at admit-time
(`scheduler_loop.rs:216-222`):

```rust
admission_budget.reserve_target(estimated_request_target(
    slot_idx,
    req.prompt_tokens.len(),
    req.max_tokens,           // ← full output cap pre-claimed
    req.reusable_prefix_len,
));
```

and again in `can_reserve_full_isl` (`admission.rs:42-53`) which
gates new admits on the FULL prompt+max_tokens fitting RIGHT NOW.
This is the "Reserve at admission" model the user is asking us to
abandon.

Other state:

- Slot availability check (`scheduler_loop.rs:185-263`).
- Per-step prefill page budget (`PrefillBudget::from_scheduler`,
  `execution.rs:81-117`).
- `runtime_workspace` computed once at boot
  (`construction.rs:135-143`), warned if exceeds headroom
  (`construction.rs:158-167`), never re-checked.
- Mixed workspace, recurrent state, and FlashInfer plan allocs
  happen inside `forward_*` calls and can OOM.
- `submit` returns 503 only on bounded waiting queue
  (`max_waiting_requests`, `types.rs:107`,
  `http_server/handlers.rs:247-252`).

### Proposed design

```rust
pub(super) struct GpuCapacityBudget {
    kv_pool_tokens:        usize,
    workspace_bytes:       usize,
    slot_count:            usize,
    recurrent_state_bytes: usize,
}
pub(super) struct RequestReservation {
    kv_tokens:             usize,
    prefill_workspace:     usize,
    recurrent_state:       usize,
    slots:                 usize,
}
```

Invariants:

- `GpuCapacityBudget` is the **only** truth for "can we admit?".
  Mutated atomically by `RequestResources::Drop` (Area A) and
  `assign_slots`.
- Admission calls `budget.try_reserve(reservation)` before
  `admit_waiting_candidate`; on failure the request stays on the
  waiting queue.
- `SchedulerHandle::submit` exposes `would_admit(reservation)` so the
  HTTP layer can return 503 *before* the admission tick.  This is
  SGLang's `prefill_max_requests` (`types.rs:106`) made
  memory-aware.

New `ModelForward` accessors (`infer/src/model.rs:231`):

```rust
fn estimate_prefill_workspace(&self, prompt_tokens: usize) -> usize { 0 }
fn recurrent_state_per_slot(&self) -> usize { 0 }
```

Defaults preserve current behavior; Qwen3.5 hybrid overrides
`recurrent_state_per_slot` (the "Alloc chunk_state failed" term in the
c8-fp8 wins entry).  `scheduler_runtime_workspace_bytes`
(`model.rs:489`) stays but is now consumed by the budget rather than
logged once.

### Migration order

1. **B0.** Replace `can_reserve_full_isl` (eager) with
   `can_admit_minimum_step(prompt_tokens, prefix_len)` (lazy).
   The minimum step for a new admit is its first prefill chunk
   (`min(chunked_prefill_size, prompt_tokens − prefix_len)`).
   Drops the `max_tokens` arg entirely from the admission gate.
   Closes the eager-reservation bug independently of A/B/C.
2. **B1.** Add the two trait methods + Qwen3.5 implementation.
   Independent.
3. **B2.** Construct `GpuCapacityBudget` in
   `Scheduler::with_config` from existing
   `construction.rs:135-167` numbers.
4. **B3.** Wire per-step `try_reserve(step_delta)` into the
   step planner (`execution.rs::plan_step`); refunds in
   `RequestResources::Drop` (depends on A2). Each step's reservation
   is `Σ(prefill_chunk_tokens) + Σ(decode_step_tokens) +
   Σ(workspace_for_active_kernels)` — concrete and small, not
   speculative.
5. **B4.** `would_admit` on the handle; gate `submit_request` for
   503 only when even `can_admit_minimum_step` fails (depends on
   B0 + B3).

**B0 alone closes the immediate "tune chunked_prefill_size to avoid
OOM" workaround.** Once a new admit reserves only its first chunk
instead of full max-ISL, 16 admits cost ~`16 × chunked_prefill_size
× bytes_per_token` upfront (vs ~`16 × 4352 × bytes_per_token`
today), an 8.5× drop.

B1+B2 ship next as plumbing; B3 shifts the per-step semantics — gate
via the bench step from area D before promotion.

### Tests

- **System**: launch N requests with `N × per_request_kv_tokens >
  pool_capacity`.  Assert every excess returns HTTP 503, never empty
  completions, never OOM.  Sits next to `e2e_qwen35`.
- **Unit**: `try_reserve` round-trips; failed `try_reserve` does not
  mutate.  Property test.
- **Bench wrapper**: tighten `scripts/bench_guidellm.sh` to fail runs
  where `request_totals.errored == 0` AND
  `streaming_iterations.successful.mean == 0` (already filed K6).

### Risk

- Workspace estimates are imprecise — FlashInfer workspace varies with
  `qo_indptr`/`kv_indptr` shape (`flashinfer.rs:516`).  Use a
  conservative upper bound; accept under-utilization.  Verify with the
  bench above before shipping.
- Hybrid chunk-state isn't per-slot static — but is fixed at model
  load time.  Safe to charge per-slot.

---

## Area C — KV-format unification at the attention layer

### Current state

[`throughput-gap-analysis.md` lines 131-148](../projects/2026-04-29-throughput-gap-analysis.md) quantifies the matrix:

| Phase / Format | BF16 | FP8 | INT8 |
|---|---|---|---|
| Prefill paged | TileLang HD128 / FlashInfer | + commit_to_fp8 | + commit_to_int8 |
| Single-token decode | Triton AOT | `decode_attention_fp8` | `decode_attention_int8` |
| Batched decode | `flashinfer_tc_run_layer` | `decode_attention_fp8` | `decode_attention_int8` |
| **Mixed (varlen)** | `flashinfer_tc_run_layer` | **MISSING** | **MISSING** |

K2 gate at `infer/src/model/qwen3/forward.rs:579-586` rejects non-BF16
→ `StepPlan::Split` (`execution.rs:343`).  The varlen FP8 kernel
landed at
[`crates/cuda-kernels/csrc/attention/decode_attention_varlen_fp8.cu:78`](../../crates/cuda-kernels/csrc/attention/decode_attention_varlen_fp8.cu)
+ FFI `crates/cuda-kernels/src/ffi/attention.rs:541` but is **not
wired** into `decode_batch_with_prefill`
(`infer/src/model/qwen3/batch_decode.rs:481-483` early-returns on
non-BF16).  INT8 varlen kernel does **not exist**
(`crates/cuda-kernels/src/kv_quant.rs:382-440` is per-row only) — that
is kernel work.

### Proposed design

```rust
// infer/src/ops/attention.rs — extend
pub trait BatchAttention {
    fn attend_varlen(
        &self,
        q: VarlenQ<'_>,                  // packed [decode rows | prefill rows]
        kv_pool: &PagedKVPool,
        format: KVFormat,
        layer_idx: usize,
        out: &mut DeviceSlice<f16>,
    ) -> Result<()>;
}
```

Each KV format implements **one** varlen attention kernel.  Plan
dispatch (`StepPlan::Mixed/Split/Pure`) becomes orthogonal — all call
`attend_varlen`.  The 16-cell matrix collapses to 4 (one per format).

### Migration order

1. **C1.** Wire existing `decode_attention_varlen_fp8` into
   `decode_batch_with_prefill`; lift the K2 gate to `BF16 | FP8E4M3`.
   Closes K2 for FP8.  No kernel work — the kernel is in tree.
2. **C2.** Extract the dispatch into a `BatchAttention` impl on
   `Qwen3`.  Pure refactor; makes the contract explicit.
3. **C3.** **[KERNEL WORK]** Template the varlen FP8 kernel for INT8
   (per-page scale loads).  Multi-day + ncu pass.  Closes K2 for INT8.
4. **C4.** **[KERNEL WORK]** HD256 variant for Qwen3.5 full-attention
   layers — the c8-fp8 wins entry's "Path & K2 status" blocker.
   Multi-day.

C1 ships independent of A/B/D; same plan as
`throughput-gap-analysis.md` lines 106-128.  C3/C4 are the only new
kernel work in this design.

### Tests

- **Correctness**: per-format pass through `infer/tests/e2e.rs` (after
  K5 fix to drop byte-exact assertion); 4-token greedy must produce
  coherent text.
- **Plan distribution**: c=16/60s with FP8 KV asserts `>0` Mixed
  steps via the `plan_label` counter from
  `pipeline-map.md` §6.
- **Numerical**: regen golden JSONs in `infer/test_data/` after each
  format wires up (standing rule).

### Risk

- C1 read-after-write ordering: BF16 reads `k_data` directly
  (`paged_kv.rs:1170-1178`); FP8 must `quantize_scatter_kv_fp8_range`
  between prep and attention (`paged_kv.rs:1556-1574`).  The
  [bf16-shadow dead-end](../experience/errors/2026-04-29-bf16-shadow-mixed-architectural-dead-end.md)
  is the cautionary tale — verify with a 4-token smoke before
  benching.
- HD256 INT8 is the long pole.  Qwen3.5 will not reach Mixed parity
  with Qwen3 until C4.  Flagged separately.

---

## Area D — Model+format+hardware auto-config

### Current state

Three independent helpers compute pieces:

- `auto_num_slots` (`infer/src/main.rs:547-611`) — slots from
  `(gpu_total × frac − weights) / per_slot_bytes`.  Ignores workspace,
  recurrent state, expected concurrency.
- `pick_chunked_prefill_size_for_hbm` (`types.rs:144-152`) — HBM tier,
  format-blind.
- `compute_max_seq_len` (`core.rs:826-883`) — assumes BF16-style cost.

Qwen3.5 c=16 OOMs at defaults: nothing accounts for the 640 MB
HD256 FlashInfer workspace + per-slot recurrent state.  Hand-tuning
required `slots=8 + frac=0.70 + chunk=512`.

### Proposed design

```rust
// infer/src/scheduler/runtime_profile.rs (new)
pub struct RuntimeProfile {
    pub max_slots:           usize,
    pub mem_fraction_static: f64,
    pub chunked_prefill:     usize,
    pub max_prefill_tokens:  usize,
    pub max_seq_len:         usize,
}

impl RuntimeProfile {
    pub fn resolve<M: ModelForward>(
        model: &M,
        kv_format: KVFormat,
        hardware: HardwareTier,        // gpu_total_bytes + sm
        overrides: RuntimeOverrides,   // user CLI flags
    ) -> Self;
}
```

Resolver consumes `model.kv_cache_bytes_per_token()` (`model.rs:266`),
`scheduler_runtime_workspace_bytes` (`model.rs:489`),
`recurrent_state_per_slot` and `estimate_prefill_workspace` (Area B),
`pick_chunked_prefill_size_for_hbm` (`types.rs:144`), and the format's
bytes-per-token multiplier.

Rules:

- Hybrid: subtract `recurrent_state_per_slot × max_slots` from KV
  budget before sizing slots.
- Quantized **without Mixed** (pre-C3): `chunked_prefill = 512`.
- Quantized **with Mixed** (post-C3): HBM-table value.
- BF16 with Mixed: HBM-table value.
- Always emit the existing
  "Scheduling envelope (resolved | SGLang-equiv)" log line
  (`construction.rs:200-214`), now from one source.

### Migration order

1. **D1.** New trait methods (= B1; ship once).
2. **D2.** `RuntimeProfile::resolve`; `main.rs:234-249` calls it.
   `auto_num_slots` becomes a thin wrapper for transitional callers.
3. **D3.** `runtime_defaults` (`types.rs:162-170`) takes the profile;
   verify
   [`wins/2026-04-29-bench-guidellm-c16fixed-fp8.md`](../experience/wins/2026-04-29-bench-guidellm-c16fixed-fp8.md)
   numbers reproduce without manual flags.

D1+D2 land before C3 so the "no Mixed" branch picks the right chunk
size.  Post-C3, flip the quantized arm to HBM-table chunk size.

### Tests

- **Unit**: `resolve` over `(model, format, hardware)` — assert
  Qwen3-4B/L4/FP8 → slots=16, chunk=512; Qwen3.5-4B/L4/FP8 → slots=8,
  frac=0.70, chunk=512; Qwen3-4B/L4/BF16 (post-C3) → slots=auto,
  chunk=2048.
- **Integration**: server boot with no flags, single request, assert
  no OOM in log + boot envelope matches the unit-test profile.

### Risk

- Silent regression — wrong default ships if the resolver picks badly.
  Mitigation: every `resolve` call emits the resolved profile; tighten
  K10 wins-promotion gate to assert boot envelope matches the
  wins-entry header.
- Override drift — keep policy explicit: user CLI flag wins, no
  env-var escape hatches (per `types.rs:114-116`).

---

## Cross-area landing order

Two parallel tracks:

- **Track 1 — robustness (closes K7, future-proofs):**
  A1 → A2 → B1+B2 → B3 → B4.  K7 regression test from A.4 gates A2
  promotion.
- **Track 2 — perf (closes K2):** C1 → D1+D2 → D3.  C1 is the largest
  single tok/s lever (~+45 tok/s expected, throughput-gap-analysis
  line 56).

C3 + C4 (kernel work) sit downstream of both tracks; the only
multi-day items.  Everything else is structural plumbing.

## Out of scope

- K1 lazy `qkv_proj` build — orthogonal optimization.
- K8 per-phase TTFT histogram — observability,
  [`bench-tracing-patch-2026-04-29.md`](bench-tracing-patch-2026-04-29.md).
- K9 INT8 dequant ITL — falls out of C3.
- K5/K6/K10 bench gates — `docs/bench-and-trace-spec.md`.
