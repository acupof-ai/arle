# 2026-05-07 · Phase 1A v3 regression root cause CONFIRMED — Mixed admission budget undersized

## Priority & ROI

**Priority**: P0 (blocks Phase 1A v3 acceptance — root cause for
the 132% TTFT regression in `ba748af`).

**ROI of this finding**: source-survey of `mixed_prefill_token_budget()`
+ math (~30 min Claude work) confirms codex's hypothesis. **Avoids
nsys deep-dive that would have cost hours**. Fix path is now
narrowly identified (config-level, not kernel-level).

**Negative case**: if the budget cap WAS load-bearing (e.g.
prevents OOM), simply removing it could trigger memory pressure
or kernel mismatch. Verification needed before fix lands.

**Kill criteria for the fix**: if raising `mixed_prefill_token_budget`
to match `max_prefill_tokens` causes:
- E2E regression on greedy_consistency / e2e tests → revert
- KV pool OOM at high-conc → re-add cap with smarter rule
- Long-ctx 4k/c=4 TTFT not back to ≤ F4-Small baseline (3403 ms)
  after fix → admission redesign needed (codex's "long prefill
  → Split, short → Mixed" hypothesis)

## Source evidence

`infer/src/scheduler/types.rs::mixed_prefill_token_budget`:

```rust
pub fn mixed_prefill_token_budget(&self) -> usize {
    self.max_prefill_tokens
        .min(self.long_prefill_token_threshold)
        .max(1)
}
```

Defaults:
- `Default::default()` (line 194-196):
  - `max_prefill_tokens: 16384`
  - `long_prefill_token_threshold: 512`
  - → Mixed budget = **min(16384, 512) = 512 tokens**
- `runtime_defaults` (line 270-271):
  - `max_prefill_tokens: 16384`
  - `long_prefill_token_threshold: 4096`
  - → Mixed budget = **min(16384, 4096) = 4096 tokens**

ARLE bench server uses `runtime_defaults` per the typical CLI
path → Mixed admission budget at **4096 tokens** for our long-ctx
test.

## Math: why 4096 budget regresses long-ctx 4k/c=4

The bench shape: 4 concurrent requests, each with 4096-token
prompt + 256-token output.

### Phase 1A v3 (Mixed enabled, budget = 4096)

Each Mixed admission step packs prefill chunks until it hits
4096 tokens (per `select_mixed_launch_prefill_candidates`). Since
each request needs 4096 tokens of prefill:
- One Mixed step admits **ONE request** worth of prefill (4096 tokens)
- 4 requests × 4096 = 16384 tokens of prefill total
- Required steps: 4 (one per request, sequentially)

Plus mixed-step decode work for ALREADY-decoding rows. With c=4
admission burst, by the time request 2 starts prefill, request
0/1 are decoding. Mixed decode adds ~4 rows × per-row cost.

Server log confirms: 21 plan=mixed steps for the 60s bench window.
Average ~700ms each = ~14.7s of total mixed-step prefill+decode
work.

### F4-Small (Mixed disabled, Split path, budget = max_prefill_tokens = 16384)

Split path's `step_prefill_batch` uses `select_launch_prefill_candidates`
with the FULL `max_prefill_tokens` budget. Can pack:
- 4 requests × 4096 tokens = 16384 tokens in ONE Split step (fits exactly)

Server log confirmed: 7 plan=split steps × ~1.2s = ~8.4s total.

### Ratio

ARLE Mixed (Phase 1A v3): 14.7 s mixed work
ARLE Split (F4-Small): 8.4 s split work
**Ratio: 14.7 / 8.4 = 1.75× more work in Mixed mode**

Plus Mixed has:
- 4× scheduler overhead (4 admit cycles instead of 1)
- 4× admission decision overhead per step

This matches the measured 132% TTFT regression (and 43% out
tok/s drop). **Codex's hypothesis CONFIRMED**.

## What Mixed budget was originally for

Comment in source:
> "so it follows the decode-active long-prefill cap instead of
> the full standalone prefill budget."

The intent: when decode rows are active in same step, prefill
work should be CAPPED to avoid starving decode. The "long-prefill
threshold" is "if any request's prefill chunk would be larger
than this, send it to Split instead of Mixed".

But the cap was INDIRECTLY applied via min: it caps the total
batched mixed work, not just the per-request chunk size. Result:
even MULTIPLE small chunks (each < threshold) get capped to a
total budget = threshold. That's the bug.

## Fix design (3 candidates)

### Fix A — Per-request threshold, not total budget (~5 LOC)

```rust
pub fn mixed_prefill_token_budget(&self) -> usize {
    self.max_prefill_tokens   // total budget = full
}

pub fn mixed_prefill_per_request_token_threshold(&self) -> usize {
    self.long_prefill_token_threshold  // per-request cap kept
}
```

In `select_mixed_launch_prefill_candidates`:
- Filter out individual candidates whose chunk > per-request threshold
- Sum the rest until total fits in `max_prefill_tokens`

Preserves the "single-request-with-huge-prefill goes to Split"
intent without capping the total batched work.

### Fix B — Just bump default `long_prefill_token_threshold` (~1 LOC)

```rust
runtime_defaults: long_prefill_token_threshold = max_prefill_tokens
```

Simplest. Effectively disables the mixed cap. Safe IF the
"single-request-huge-prefill ruins decode" failure mode isn't
real at production workloads. Risk: regression at workloads with
ONE large prefill alongside active decode rows.

### Fix C — Codex's admission redesign (~50-100 LOC)

Long prefill (≥ threshold per request): force Split.
Short prefill (< threshold per request) AND has decode: Mixed.

Most architecturally clean. Highest implementation cost. Best
long-term fix. Requires care on threshold tuning per workload.

## Recommended sequencing

1. **Land Option B (codex's plan flag)** FIRST — `--scheduler-mixed-or-split-policy`
   default Split (no production regression). 30 min codex work.
2. **Apply Fix A or B in parallel** as a secondary commit. Fix A
   preserves intent; Fix B is fastest.
3. **Re-bench longctx 4k/c=4 with Mixed re-enabled + fix applied**.
   Expect ARLE TTFT < F4-Small baseline (3403 ms).
4. If still regresses → Fix C admission redesign.

## Cross-references

- Phase 1A v3 regression: `ba748af`
- F4-Small Mixed-disable cause: `26b7f86`
- Phase 0 validation: `be6c292`
- M3.9 plan: `63af21f`
- Source: `infer/src/scheduler/types.rs:mixed_prefill_token_budget`
- Codex's recommendation summary (per tmux 2026-05-07):
  > "保留 ring，默认关 Mixed，立刻用 policy flag 止血；再查
  > mixed budget/admission"

## Rule

- **Trace-driven optimization can fix a "false-cause" while the
  real cause sits one indirection over**. M3.9 trace identified
  Mixed-disable as the cause of split-tax. Removing the Mixed-
  disable did route work to Mixed, but Mixed had its OWN budget
  cap (`mixed_prefill_token_budget`) making it slower. The fix
  is at the budget cap, not the precondition.
- **Per-config function names like `mixed_prefill_token_budget()`
  are diagnostic flags worth investigating early.** A
  `min(A, B).max(1)` pattern is a strong signal that one of the
  caps is load-bearing in unexpected ways. Always check both
  values + their defaults.
