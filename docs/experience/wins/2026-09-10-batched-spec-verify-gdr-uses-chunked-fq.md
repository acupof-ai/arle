# One chunked FlashQLA GDR path for every multi-row advance — CUDA, 2026-09-11

> Status: **pending-remote** — the deletion lands as draft PR #300; merge gate
> is the H20 GPU batch (TP2/4/8 needle ladder ×3 + lever gate,
> `flashqla_chunk_parity` at all four shards). No CUDA on the authoring Mac.

## Goal

Delete the second GDR (gated-delta-rule) recurrence. #292 moved batched spec
verify and DSpark rollback replay onto per-slot chunked FlashQLA whenever it
was available; #327 added the attn_tp=2 (8,24) and attn_tp=4 (4,12) shards;
#329 added attn_tp=8 (2,6). On the default path — sm90 with FlashQLA cubins
and the flag on — every multi-row advance at every TP then took chunked
FlashQLA, and the batched varlen kernels were reached only by the fallback
configurations:

- `--qwen35-gdr-chunked false` (CLI flag; there is no env override —
  `ARLE_QWEN35_GDR_CHUNKED` was removed 2026-07-10, see docs/environment.md),
- no FlashQLA cubins / non-sm90 hosts (A100 and older),
- an unsupported (Hg,H) geometry.

`qwen35_attention.rs:1968` took `advance_linear_conv_gdr_batched` in those
cases and `dspark.rs:1685` took `replay_linear_only_batched`. #300 deletes
the batched kernels anyway: on the default path nothing changes, and the
fallback goes from one batched conv+GDR launch per layer to B per-slot
launches of the single-sequence recurrent prefill. The cost is recorded
below as a pending-remote A/B; the gain is one recurrence to validate.

Deleted kernels:

- `conv1d_prefill_varlen_cuda` (crates/cuda-kernels/csrc/recurrent/conv1d.cu)
- `gated_delta_rule_prefill_recurrent_varlen_cuda`
  (crates/cuda-kernels/csrc/recurrent/gated_delta_rule.cu)

The single-sequence `gated_delta_rule_prefill_recurrent_cuda` /
`conv1d_prefill_cuda` stay: they are the fallback kernel the per-slot loop
now calls, and the training (autograd) forward.

## What changed

- Kernels: deleted both varlen extern wrappers (the shared per-slot
  `conv1d_prefill_kernel` and the recurrent prefill kernel are retained; only
  the batched pointer-table wrappers went).
- FFI + Rust wrappers: deleted the two externs
  (`ffi/recurrent.rs`), `conv1d_prefill_varlen_raw` and
  `gdr_prefill_recurrent_varlen_raw` (`cuda-kernels/src/recurrent.rs`).
- Serving (`infer-cuda`): deleted `advance_linear_conv_gdr_batched` and
  `replay_linear_only_batched`; `LinearCore::Rows` now unconditionally loops
  the per-slot advance (chunked FlashQLA for `len>1` where supported,
  single-token decode for one-row slots). DSpark rollback
  (`qwen35/dspark.rs` `dspark_rollback_batch`) loops `replay_linear_only`.
- Staging: deleted `Qwen35ReplayTables`/`ReplayLayout` and the four
  per-slot table constants (`qwen35_spec_state.rs`), the executor's
  `replay_tables` field, and the four `batch_*` LinearAttnScratch buffers
  plus `LINEAR_BATCH_MAX_LEN` (`qwen35_workspace.rs`).
- Gate: `gdr_varlen_parity.rs` replaced by `flashqla_chunk_parity.rs`. Two
  families share one f64 host recurrence: (a) the chunked FlashQLA pipeline
  conv1d→gdr_fq_prep→cumsum→kkt→fwd vs f64 at all four shards
  (16,48)/(8,24)/(4,12)/(2,6), lengths 5/17/64, output + final-state teeth;
  (b) the single-sequence recurrent fallback (the same
  `conv1d_prefill_cuda` followed by `gated_delta_rule_prefill_recurrent_cuda`)
  at the same shards and lengths, with conv-output, rebuilt-ring, GDR-output
  and final-state comparators and four teeth of its own. Before #300 this
  fallback kernel had no numeric gate (gdr_decode_parity covers decode only).
- Registry/plan: `qwen35.linear_attention` gate row now lists
  `gdr_decode_parity` + `flashqla_chunk_parity`; the varlen implementations
  are recorded as deleted in the coverage audit.

Net: 17 files, +606/−1628 (net −1022).

## Parameters / GPU batch (pending-remote)

Merge is blocked on the batch that exercises both the unchanged default path
and the changed fallback path:

1. `flashqla_chunk_parity` positive + `--negative-control` at all four
   shards: six teeth — FlashQLA output/state and recurrent
   conv/ring/output/state — each trips only its own family, exit 0,
   `NEGATIVE CONTROL OK`.
2. Correct inference at attn_tp=2/4/8 (default path, chunked FlashQLA):
   `scripts/needle_gate.py` needle ladder ×3 same-config and
   `scripts/lever_gate.sh`, against the pre-#300 envelope; routing here is
   unchanged from #327/#329.
3. Fallback cost, H20 standing in for non-sm90: DSpark c=8 spec verify with
   the serve flag `--qwen35-gdr-chunked false` (no env override exists),
   baseline (pre-#300: one batched conv+recurrent launch per layer) vs
   treatment (B per-slot launches of the single-sequence recurrent prefill).
   Measure verify-step ITL and chain acceptance; numerics are covered by
   family (b). The treatment arm must PROVE the fallback engaged: the kernel
   profile emits `linear/gdr_recurrent` multi-row events (the chunked arm
   emits `linear/gdr_fq` — qwen35_attention.rs `profile_op` names), and the
   run log must show them before any number counts. The design decision
   stands; this records the fallback's launch-count cost.

## Rule

A/B (compare per-chain **verify argmax / accept counts**, not just committed
tokens):
1. Baseline separation: serve with `--qwen35-gdr-chunked false`, c=8 —
   verify uses the non-chunked recurrent kernel; isolates
   FlashQLA-vs-varlen as the variable. (There is no env var for this
   switch; `ARLE_QWEN35_GDR_CHUNKED` never existed — only the CLI flag.)
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

A second kernel kept "just in case" behind a router is a third state to
validate: close every shard of the replacement's geometry first (#327/#329
each shipped with its parity gate), gate the replacement AND the kernel that
remains as the fallback, then delete the redundant middle path in one change.
Deleting a batched fast path changes fallback latency — record the
flag-off A/B as a cost instead of pretending the path had no caller.
