# DSv4 MTP/DSpark high-concurrency regression: serial draft + B>1 disable

> Status: Active — root cause identified 2026-07-19, fix pending

## Context

Production-all-on benchmark (`45dd64bd2`, 4×H20 TP=4/EP=4, `bench-prompts-64.jsonl`
~2.8k tok, 120 s/point, max_tokens 256):

| c | Base | MTP | MTP Δ | DSpark | DSpark Δ |
|---|-----:|----:|------:|-------:|---------:|
| 1  | 38.0 | **46.2** | **+21.6%** | 38.1 | +0.3% |
| 4  | 74.6 | 70.2 | -5.9% | 74.3 | -0.4% |
| 8  | 123.7 | 72.0 | -41.8% | 121.9 | -1.5% |
| 16 | 195.7 | 69.7 | **-64.4%** | 117.6 | **-39.9%** |

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

## Rule

- Speculative decode gains are c1-only until draft generation is batched.
- DSpark is effectively disabled at B>1 by current logic; do not claim
  DSpark throughput wins on concurrency >1.
- Benchmark speculative configs with the production workload (long
  prompts), not synthetic short-prompt sets.
