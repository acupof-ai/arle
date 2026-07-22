# DSv4 MTP/DSpark high-concurrency regression: serial draft + B>1 disable

> Status: Fixed 2026-07-19 (`13426a8de` + `7a8c0bdd4`). MTP c16 recovered +31%; DSpark B>1 dispatch + budget fixed.

## Context

Production-all-on benchmark (`45dd64bd2`, 4×H20 TP=4/EP=4, `bench-prompts-64.jsonl`
~2.8k tok, 120 s/point, max_tokens 256):

| c | Base | MTP (before fix) | MTP Δ | MTP (after fix `7a8c0bdd4`) | MTP Δ |
|---|-----:|----:|------:|----:|------:|
| 1  | 38.0 | **46.2** | **+21.6%** | **47.0** | **+23.7%** |
| 4  | 74.6 | 70.2 | -5.9% | 71.3 | -4.4% |
| 8  | 123.7 | 72.0 | -41.8% | 79.4 | -35.8% |
| 16 | 195.7 | 69.7 | **-64.4%** | 91.4 | **-53.3%** |

MTP accept_rate 0.704 (c1). DSpark `spec_d` tokens = 0 (prompt router
`--dspark-max-prompt-tokens 64` routes all >64-tok prompts to no-spec).

## Root Cause

### MTP: serial draft generation, no batching

`spec_decode.rs:399-433` — `spec_step_batched` drafts per-slot sequentially:
```rust
for s in 0..n {
    draft_chain(s, ...)  // spec_decode.rs:590-619 loops depth × mtp_forward_level(m=1)
}
```

Each `mtp_forward_level` (`dsv4.rs:6613`) is a **full forward pass**
(embedding + rms_norm + e_proj/h_proj + MoE) processing **m=1 token**.

At c16, depth=4: **16 slots × 4 levels = 64 sequential full forwards**,
each with launch overhead and no GEMM batching. The verify phase
(`spec_decode.rs:440`, one batched `forward_decode_batch_verify` over all
chains) is amortized, but the serial draft phase dominates.

Scheduler single-step sync (`infer-core/src/lib.rs:620-693`): all active
requests form one `forward_decode_batch` → one `spec_step_batched`. The
entire 16-slot draft+verify+commit must finish before step N+1 begins.
No pipelining of draft(N+1) behind verify(N).

**Fix direction**: batch the draft phase — one `mtp_forward_level` call
processing N slots × depth tokens (reshape to [N×depth, hidden] GEMM),
instead of N×depth sequential m=1 calls.

### DSpark: weights reduce slots + B>1 disables speculative

1. `executor/dsv4.rs:510-547`: DSpark draft weights load **before**
   `kv_budget_plan`. Less free VRAM → `affordable` slots → `num_slots`
   clamped down → fewer concurrent requests at c16.

2. `executor/dsv4.rs:1922-1935`: in `forward_decode_batch_inner`, if
   **any** row in the batch is DSpark-eligible, **all** rows are marked
   ineligible, then the batched MTP path requires `self.dspark.is_none()`
   (false) → falls through to normal batched decode. **Zero speculative
   benefit at B>1.**

3. `executor/dsv4.rs:699-772`: `load_dspark_exec` allocates per-slot `df`
   (latent KV, `max_seq_len`) + `attn_states` (one per draft stage,
   `draft_span = max_seq_len + block_size`) for **all** `num_slots`,
   **after** `kv_budget_plan` already fixed slot count. Unbudgeted VRAM
   pressure.

**Fix direction**: (a) only mark the eligible row ineligible, not all;
(b) implement batched DSpark verify for B>1; (c) include per-slot DSpark
runtime in `kv_budget_plan`.

## Fix Applied (`13426a8de` + `7a8c0bdd4`)

### MTP: batched draft phase
`spec_decode.rs:442-478` — `spec_step_batched` now runs `depth` batched
`mtp_forward_level` calls (one row per slot) instead of `N×depth` serial
m=1 calls. `mtp_forward_level` accepts `slot_ids[]` + `positions[]` so
attention targets each row's own KV cache.

Result: c16 69.7 → 91.4 (+31.1%). c8 72.0 → 79.4 (+10.3%).

### DSpark: B>1 dispatch + budget
`executor/dsv4.rs:1920-1935` — eligible rows dispatch individually via
`forward_decode_row`; the rest keep batched lanes. No more "one eligible
→ whole batch disabled".

`executor/dsv4.rs:545-566` — DSpark per-slot runtime (latent KV + draft
attention states) is now counted in `kv_budget_plan` via
`extra_per_slot_bytes`, so `num_slots` doesn't over-commit.

## Remaining Gap

MTP c16 (91.4) is still 47% of base (195.7). The draft phase is now
batched, but:
- Verify phase runs all chains in one forward (already batched, but adds
  `depth+1` rows per slot vs 1 row for plain decode)
- Commit phase is per-slot serial (`spec_decode.rs:501-520`)
- Scheduler single-step sync: all slots finish before next step

MTP's value remains c1-only (+23.7%); at c4+ the batch already provides
throughput and MTP adds overhead. Not a default-flip candidate.

## Rule

- Speculative decode gains are c1-only until draft generation is batched.
- Before `13426a8de`, DSpark was effectively disabled at B>1. That dispatch defect is fixed; later `13fe251cb` also batches anchor + target verify. Concurrency wins still require measured A/B because draft work remains costly and c=8 was later −7.6% vs valid no-spec.
- Benchmark speculative configs with the production workload (long
  prompts), not synthetic short-prompt sets.
