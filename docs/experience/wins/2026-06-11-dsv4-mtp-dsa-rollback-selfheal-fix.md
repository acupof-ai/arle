# DSv4 MTP spec decode — DSA compressed-KV rollback gap fixed (self-heal, SGLang-aligned); non-batched now passes the needle gate

## Context

MTP (depth-1 self-speculative) decode is already serve-wired in `executor.rs`
(`forward_decode_tokens`: `mtp_forward` draft → `forward_tokens_verify` →
accept `[draft,bonus]` / reject `truncate+restore_spec_rollback`), opt-in via
`ARLE_DSV4_SPEC_DECODE=1`. It was validated **byte-identical +61% on 2026-06-08**.
Re-validating on the current tree (`1e0f05e1`, post efficiency-tranche +
official-DSA rewrite) for productionization, it was **BROKEN**: every domain
(问答/创作/代码/客服) decoded **degenerate garbage** (loops, `或或或`, `\ \ \`),
while plain decode (spec off) was fully coherent — so a rewrite since 6-08
regressed the spec path. (ckl's hint: "KV + 位置编码 + 历史 CSA 得算出来存下来"
+ "看 sglang"; the perf table measured before this was garbage-speed, void.)

## Two bugs found (root-caused)

1. **DSA compressed-KV rollback gap (this fix).** `dsv4_dsa_official_enabled()`
   defaults ON; CompressedSparse layers write the official DSA key cache every
   decode (`csa_select_official`, `official.packed_rows = indexer_rows_after`,
   attention.rs:5438). But `capture/restore_rollback_snapshot` cover
   sw/fp8/compressor/indexer **only — NOT `dsa_official`**, and
   `truncate_decode_len`→`advance_decode_len` roll back `compressor`/`indexer`
   `compressed.seq_len` but leave `packed_rows` at its speculative value. A
   rejected draft that crossed a compression boundary leaves a stale draft key
   in `dsa_key_cache` and a desynced counter → corrupt top-k selection →
   accumulating degeneration. Classic partial-fix gap: the post-6-08 official-DSA
   rewrite added `dsa_official` without extending the rollback's mutated-buffer
   enumeration (rollback had only recently been patched for compressor/sw/fp8).

2. **Batched 2-token verify (separate, still open).** With
   `ARLE_DSV4_MTP_BATCHED_VERIFY=1` (default) the `[pending,draft]` batched
   verify garbles even **accepted** tokens (问答 86% accept still fully garbage);
   `=0` (per-token verify) fixes 问答 → a forward-compute bug in the 2-token
   batched path (the "known col1 bug" now hitting col0; likely the compressed
   positions for 2 tokens aren't boundary-floored). Tracked separately.

## What worked — SGLang frozen-KV discipline

Dispatched a study of SGLang's DeepSeek-V3.2 DSA + nextn/MTP. Verdict: SGLang
ships **zero** compressed-cache snapshot/restore. The compressed write target is
a **deterministic function of committed `seq_len`** (`write_pos =
((seq_len-1)//ratio)*ratio`, boundary-floored, written only at `seq_len %
ratio == 0`); on reject it rolls back **only `seq_len` + frees dense KV**, and
the compressed cache **self-heals** next step (re-pack overwrites). The
compressor ring is sized larger in spec mode (16/256 vs 8/128) so a rejected
draft never aliases a live committed slot. → **Fix is (B) self-heal, not (A)
snapshot/restore.**

Applied (attention.rs `truncate_decode_len`):
```rust
self.advance_decode_len(mode, ratio, total_len);
if let Some(dsa) = &mut self.dsa_official {
    let compressed_rows = total_len / ratio.max(1);
    dsa.packed_rows = dsa.packed_rows.min(compressed_rows); // self-heal, no snapshot
}
```
Clamp `packed_rows` DOWN to the rolled-back compressed-row count so the next
real decode re-packs (overwrites) the stale draft key from the restored indexer
KV. (Why not freeze the verify writes: that loses accept-amortization + the
within-batch dependency + depth-K; the DRAFT is already frozen — `mtp_forward`
is SW-only. SGLang mutates-then-self-heals, doesn't freeze the verify.)

## Validation (gate, not eyeballed samples)

- **garbage → coherent**: non-batched 客服 went from `\ \ \` to a coherent
  customer-service reply.
- **Needle correctness gate PASS** (`dsv4_lever_gate.sh`, lengths 115/446/2000 ×2,
  spec-fix nobatch vs baseline): **zero misses, zero garbage, within the baseline
  envelope** at every length (115: spec exact=2 vs baseline exact=1/partial=1; the
  `738731/738741` partials appear in baseline too = MoE non-determinism floor).
- **Discipline caught**: an eyeballed single 客服 sample looked like a "residual
  spec bug" (looping); the rigorous gate showed it was the **MoE-non-determinism
  confound** the project warns about (`feedback_correct_inference_not_baseline_identity`).

## State / Rule

- Non-batched MTP spec decode is **CORRECT** with this fix (needle gate pass).
  **Perf is NOT yet a win**: non-batched verify = 2 forwards/step → ~wash vs
  plain decode; the +61% requires the **batched** verify, which is bug #2 (open).
  So MTP stays opt-in until #2 is fixed; default flip un-licensed.
- **Rule**: any new device-state buffer added to a forward path (here
  `dsa_official`) MUST be enumerated into the spec/rollback mutated-buffer list
  in the SAME change (`feedback_design_to_implementation_detail`). And gate spec
  correctness on the **needle ladder + same-config-twice**, never an eyeballed
  greedy sample (MoE non-determinism).
