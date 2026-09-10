# Multi-chain spec verify uses the parity-validated chunked FlashQLA GDR — CUDA, 2026-09-10

> Status: **pending-remote** — host routing + pure geometry predicate land in
> this change; the numeric c=8 acceptance A/B runs on an H20 (CUDA) and is
> recorded below, not on this Mac.

## Goal

DSpark spec decode on the Qwen3.8-27B hybrid target: draft acceptance ~13% at
c=1 but **0% at c=8**. Batch size must not switch the GDR (linear-attention)
recurrence to a different kernel than the c=1 verify path.

## Hypothesis (leading, pending GPU confirmation)

The uniform multi-chain spec verify (8 chains × block rows) advances every
slot's GDR/conv recurrent state with `gdr_prefill_recurrent_varlen_raw` via
`advance_linear_conv_gdr_batched` — a varlen gated-delta recurrent kernel whose
**only callers are batched spec verify and its own partial-accept replay**
(`qwen35_attention.rs`); it has never run a parity-gated normal path. c=1 verify
(one chain) instead uses the FlashQLA chunked recurrence (`gdr_fq_cumsum/kkt/fwd`)
that chunked prefill and c=1 verify use and that passed parity.

Established by reading code (facts): the two paths use different GDR kernels,
and the varlen one has never been on a parity-gated path. **Suspected, not yet
proven**: that this kernel difference causes the c=8 0% acceptance — a
different multi-token recurrent reduction for the same linear layers could
corrupt the hybrid layers' verify logits so in-block draft tokens mismatch while
the full-attention anchor/bonus row stays correct (which would explain why
committed-output parity can pass while acceptance reads 0). The precondition is
also unconfirmed: c=8 must actually be running DSpark on a BF16 full-attention
KV pool. Both are decided by the GPU A/B below.

Treatment: when the FlashQLA chunked kernel is the active GDR recurrence, route
the uniform multi-chain branch through the **same per-slot FlashQLA path** c=1
uses, instead of the unvalidated varlen batch. The varlen one-launch kernel
stays only for geometries chunked FlashQLA does not cover (flag off, non-sm90,
or non-AOT (Hk,H) heads).

## Why "same kernel" is the fix (not re-deriving the varlen kernel)

Correctness here means *the batched verify's recurrent state equals the c=1 /
decode recurrent state*. Re-deriving `gdr_prefill_recurrent_varlen_raw` to a
locally-plausible reduction would create a second unvalidated core. Routing
both paths through the one parity-gated FlashQLA kernel makes batch size a pure
loop-bound (same kernel, same reduction order per slot), so there is no new
numerics to validate beyond "the c=1 kernel still runs for c=N slots". Conv1d's
equivalent batched/per-row split is a plain causal conv with no recurrent
reduction-order sensitivity; it is left on the varlen batched kernel and is a
separate, lower-suspect item.

## Change

- `infer-model/src/qwen35.rs`: pure `fq_geometry_supported(key_heads,
  value_heads, key_head_dim, value_head_dim)` — the single AOT-geometry
  predicate (hd128, (Hk,Hv) ∈ {(16,32),(16,48)}), with a host unit test
  (`fq_geometry_matches_aot_instantiations`) that runs in the CPU test lane.
- `infer-cuda/src/qwen35_attention.rs`:
  - one helper `gdr_chunked_fq_active(row_len)` (flag + geometry + sm90) is the
    single predicate for both verify and replay;
  - the per-row `use_fq_chunked` dispatch uses it;
  - the uniform-multi-chain **verify** branch falls through to the per-row
    FlashQLA loop when it is set; `advance_linear_conv_gdr_batched` (varlen)
    runs only when chunked FlashQLA is not active.
- `infer-cuda/src/qwen35/dspark.rs`: the partial-accept **rollback replay**
  applies the same availability predicate (`gdr_fq_available`, independent of
  row count) — when FlashQLA is available it loops the per-slot
  `replay_linear_only`, including the all-reject case where every slot advances
  one row (k=0 → single-token decode kernel); otherwise
  `replay_linear_only_batched` (varlen). Routing on availability rather than
  max row length is what closes the all-reject 0%-acceptance case.
- No new kernel; the executor (`qwen35.rs`) is untouched.

### Deliberate perf tradeoff

The correct path is a per-slot loop over three kernels (conv + FQ cumsum/kkt +
FQ fwd), so at c=8 a verify step and a rollback replay each go from ~one varlen
launch to **3×B per-slot launches** — correctness bought with launch count.
Record **c=8 verify-step latency** as an A/B metric; if the extra launches cost
materially, the follow-up is a *batched FlashQLA* kernel (same validated
reduction, pointer-table batched), never re-enabling the unvalidated varlen
recurrence.

## Parameters / GPU A/B (pending-remote)

Run on an H20 with a **BF16 full-attention KV pool** — under FP8/INT8 KV the
DSpark batch gate does not admit batched DSpark (`decide_decode`,
`infer-plan`), so confirm in the boot log that c=8 is on a BF16 pool and DSpark
is actually engaging before treating any number as this gate.

A/B (compare per-chain **verify argmax / accept counts**, not just committed
tokens):
1. Baseline separation: `ARLE_QWEN35_GDR_CHUNKED=0`, c=8 — verify uses the
   non-chunked recurrent kernel; isolates FlashQLA-vs-varlen as the variable.
2. This fix with chunked on (default), c=8 — expected acceptance back to the
   c=1 order of magnitude (~10%+), and verify argmax for in-block draft rows
   matching across the one-chain-vs-multi-chain runs.

## Rule (pending the H20 A/B)

A batched shortcut that swaps a stateful recurrence for a different kernel is
not the same operation as the single-row path — gate batching on using the
*same validated kernel*, not on sharing one launch. The varlen recurrent kernel
being spec-only (never on a parity-gated decode/prefill path) is the
**code-established fact that makes it the leading suspected cause**; the H20
A/B decides whether it actually produces the c=8 0% acceptance.
