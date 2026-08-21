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

## Expected payoff, re-derived against the decode profile

The first estimate was built on "Marlin is 68.3% of decode". At serving
concurrency it is 13.9% and attention is 80.6%
([errors/2026-08-21](../experience/errors/2026-08-21-decode-profile-taken-at-the-wrong-batch.md)),
which changes the arithmetic in this lever's favour.

**A verify does not multiply the KV read.** `forward_tokens_verify`
(`qwen35_forward.rs:928`) runs with `seq_len = tokens.len()`, i.e. a
prefill-shaped forward: the chain's `depth + 1` query rows attend over one KV
pass. The c=1 measurement confirms it — 2.97 rows cost +10% of step time
(20.50 → 22.57 ms), where tripling the 24% attention share would have cost +48%.

So a batched d=2 verify leaves the 80% roughly where it is and pays only on the
weight GEMM, which is superlinear past M=16 but small:

| | c=16 | c=32 |
|---|---:|---:|
| attention (unchanged, KV read once) | 78.6 ms | 153.9 ms |
| Marlin at M=b vs M=3b | 13.6 → **37.7** | 23.8 → **59** |
| `gdr_decode` (per query row) | 3.4 → 10 | 6.1 → 18 |
| other | 1.4 | 1.8 |
| **step** | 99.1 → **~128** | 188.2 → **~231** |
| tokens per step at 1.89 accept | 16 → 30.2 | 32 → 60.5 |
| **ms per committed token** | 6.20 → **4.23** | 5.88 → **3.85** |
| | **−32%** | **−35%** |

Marlin's M=48 and M=96 costs are the measured sweep, not an extrapolation:
0.0880 ms at M=16, 0.2442 at M=48, 0.3177 at M=64.

## What would kill it

The c=1 win is 1.72x on the decode phase, but this workload is 154:1 prefill to
decode, so it dilutes to +21.6% end-to-end. At higher concurrency the arithmetic
runs the other way and the decode profile decides it:

- The table assumes the batched verify packs N sequences as a varlen batch with
  each sequence's KV read once. If it instead presents `N * (depth + 1)` rows to
  the decode attention path, attention triples and the lever dies outright. This
  is the first thing the implementation must prove, with a kernel-time capture
  and not with a step-time delta.
- `gdr_decode_batch_kernel` is per query row and does triple. It is 3.5% today,
  so it costs about 7 ms at c=16 — already in the table.
- The gate wants a row-count ceiling, not a request-count one: `M = b*(d+1)` is
  what Marlin sees, and it is superlinear past M=16.

## Verification

`scripts/needle_gate.py` x3 same-config (`RAW=1 TEMPLATE=qwen3_nonthink`, read
the SUMMARY line — it always exits 0) plus `scripts/lever_gate.sh`, against the
no-spec envelope on the same binary. Then the 32K chain at c=1,4,8,16,32 against
the arm-A control above, which already has its noise floor measured at ±1.5%.
