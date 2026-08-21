# Batched MTP verify — approach, 2026-08-21

> Status: Scoped, not started. Gated on the c=16/32 decode profile: the payoff
> depends on what fraction of the step is weight GEMM at serving concurrency.

## Why

Measured on the 32K agent chain, one binary, 0 errors, 32/32 at every point
(`/host/arle-runs/specab-20260821/`):

| c | ms per committed token, no spec | with MTP d=2 | Δ |
|---:|---:|---:|---:|
| 1 | 20.50 | **11.94** | **−41.8%** |
| 4 | 8.67 | 8.63 | noise |
| 8 | 6.98 | 6.97 | noise |
| 16 | 6.20 | 6.20 | noise |
| 32 | 5.86 | 5.85 | noise |

**Above c=1 speculation never runs.** The `/v1/stats` chain-counter delta is
exactly 0 for every c≥4 and `tok/decode-step` matches the control to four
significant digits, so the three arms are literally the same code path and the
±1.5% spread is this bench's noise floor.

The cause is structural, at `executor/qwen35.rs:2365-2372`:

```rust
let batched = kind == SpecKind::Dspark && self.paged_kv_bf16() && spec_compatible
    && decode_rows.iter().all(|r| r.params.is_greedy());
let gate = match batched { true => spec_max_batch(), false => 1 };
```

MTP is pinned to `gate = 1` because its branch is a serial per-row loop
(`for row in decode_rows { self.mtp_decode_row(row, host_kv)? }`). Raising
`--spec-max-batch` cannot move it. Batched DSpark is not an alternative here:
`/data00` has DSpark drafts for DeepSeek-V4-Flash and Qwen3.6-27B only, and this
checkpoint's spec head is `model_mtp.safetensors`.

The rows themselves are close to free. At c=1 the d=2 arm presents **2.97 rows
for +10% step time** (20.50 → 22.57 ms), which is the flat-Marlin kernel
measurement showing up end-to-end
([errors/2026-08-21](../experience/errors/2026-08-21-sm90-collective-loses-below-m32.md):
0.0629 ms at M=1, 4 and 8 alike).

Depth is settled: d=4 is not better than d=2 (2.03 vs 1.89 tok/step while the
accept rate falls 44.7% → 25.9%), so the target is **d=2 batched**.

## The machinery already exists

`dspark_decode_batch` (`qwen35.rs:1656`) is the three-phase shape this needs, and
only its first phase is DSpark-specific:

| phase | DSpark today | for MTP |
|---|---|---|
| draft | `dspark_draft_blocks` over N slots, one forward | **new**: batched MTP draft, `depth` levels x one forward per level |
| pack | `DsparkChain { out, slot, start, row0, chain, partial_ctx }` into a flat `chains: Vec<u32>` | reuse — an MTP chain is a linear DSpark chain |
| snapshot | per-slot linear state via `batched_copy` as the rollback base | reuse |
| verify | `dspark_verify_forward(batch, chains, total_rows, host_kv)` — takes chain data, not DSpark state | reuse |
| commit | per-slot accept path, ring restore, trunk truncate | reuse the accept, drop the ctx-ring parts |

DSv4 already runs the batched form of exactly this
(`executor/spec_decode.rs:382 spec_step_batched`): per-slot ring capture, then
`depth` batched `mtp_forward_level` calls, then **one** batched verify, then
per-slot accept. That is the reference to follow.

## Plan

Two files.

1. `crates/infer-cuda/src/executor/qwen35.rs`
   - Rename `DsparkChain` to `SpecChain`. It already carries no DSpark-specific
     field; `partial_ctx` is a ctx-ring flag MTP leaves false.
   - Lift the scratch and spec-slot access `dspark_verify_forward` reaches
     through `self.dspark` into a parameter, so an MTP batch can drive it.
   - Add `mtp_decode_batch`: partition seeded from unseeded rows (unseeded keep
     the existing per-row `mtp_warm_decode_row`, which is a plain decode), draft
     the seeded rows batched, pack, verify once, then accept per slot.
   - Gate: `let gate = match kind { SpecKind::Mtp => spec_max_batch(), _ => ... }`.
2. The qwen35 MTP draft module — add the batched draft level, one forward over N
   rows per level. This is the only genuinely new piece; the MTP head is a single
   transformer layer, so N rows is a plain batched decode.

## What would kill it

The c=1 win is 1.72x on the decode phase, but this workload is 154:1 prefill to
decode, so it dilutes to +21.6% end-to-end. At higher concurrency the arithmetic
runs the other way and the decode profile decides it:

- KV read per decode step is **32 KB per token of context** (16 full-attention
  layers of 64, `num_key_value_heads=4`, `head_dim=256`, FP8). At 32K that is
  1.07 GB per sequence per step, so **34.4 GB at c=32 against 20.0 GB of
  weights**.
- A verify does **not** multiply the KV read — the chain attends over the same
  context with `d` more query rows. So if the step is KV-bound at serving
  concurrency, batched spec is close to free there too and the win is larger
  than the Marlin arithmetic suggests.
- If instead the step is still weight-GEMM-bound, Marlin is superlinear past
  M=16 (0.0880 ms at M=16, 0.1731 at M=32, 0.3177 at M=64), and d=2 at c=16
  presents M=48 — into the expensive region. Then the gate should open only up
  to a measured row ceiling, not to `spec_max_batch`.

Either way the gate wants a row-count ceiling rather than a request-count one,
since `M = b*(d+1)` is what the kernel sees.

## Verification

`scripts/needle_gate.py` x3 same-config (`RAW=1 TEMPLATE=qwen3_nonthink`, read
the SUMMARY line — it always exits 0) plus `scripts/lever_gate.sh`, against the
no-spec envelope on the same binary. Then the 32K chain at c=1,4,8,16,32 against
the arm-A control above, which already has its noise floor measured at ±1.5%.
