# Chunked FlashQLA GDN on the attn_tp=2/4 shards — cuda, 2026-09-11

> Status: pending-remote

## Goal

One recurrence kernel for every multi-row GDN advance on the attn_tp=2 (local
shard (8,24)) and attn_tp=4 ((4,12)) paths. The single geometry predicate
`gdr_chunked_fq_active` / `gdr_fq_available` is read by three call sites, so
widening it moves ALL of them off the hand-written recurrent varlen kernel:

- ordinary prompt **prefill** — `qwen35_attention.rs` advance_linear_conv_gdr
  (`fq_supported = self.gdr_chunked_fq_active(seq_len)`) and the uniform
  batched multi-row advance (the uniform-length check ~`:1969`);
- batched spec **verify** multi-row advance;
- DSpark rollback **replay** — `qwen35/dspark.rs:1685` routes per-slot replay
  on `gdr_fq_available`.

The goal is deliberately ONE path for every multi-row advance (verify-only
routing would need a separate predicate; it is not introduced). After the
(2,6)/attn_tp=8 follow-up, #300 can delete the varlen path for every TP.

## What changed

The (24,8) and (12,4) flashqla AOT instantiations (all five phases) and their
`gdr_fq_*_h24g8_cuda` / `gdr_fq_*_h12g4_cuda` externs already existed in
`kernels.toml` and `ffi/recurrent.rs`; nothing compiled a new cubin. Two serve
hard-codes excluded the shards:

- `infer-model/src/qwen35.rs` `fq_geometry_supported` matched only (16,32)/(16,48);
  widened to also accept (8,24)/(4,12). (2,6) stays rejected (no AOT row).
- `infer-cuda/src/qwen35_attention.rs` symbol match added the h24g8 and h12g4
  cumsum/kkt/fwd triples; `gdr_fq_prep` is geometry-generic.

After this, every multi-row GDN advance — prefill, verify, replay — takes
chunked FlashQLA at attn_tp=2/4. At attn_tp=8 all three still take varlen;
that is the recorded gap before #300 can remove varlen entirely.

`gdr_varlen_parity`'s FlashQLA-vs-varlen cross-check is extended to
(16,48)/(8,24)/(4,12) at lengths 5/17/64 with per-geometry max diff.

## Hypothesis

The two implementations run the same GDN math; `gdr_varlen_parity` already
anchors both to one f64 host recurrence at these geometries, so outputs and
final states agree to within the chunked path's bf16-intermediate tolerance.
Three independent consequences, each decided by its own pending-remote check:

1. **Prefill correctness** — because the DEFAULT prefill path at attn_tp=2/4
   changes, the correct-inference gate must pass: `needle_gate.py` +
   `lever_gate.sh`, needle ladder ×3 against the baseline envelope. A failure
   blocks the change regardless of perf.
2. **Prefill perf** — matched TTFT / prefill tok/s A/B at attn_tp=2 on long
   prompts (baseline recurrent, treatment chunked FQ). This decides whether
   chunked prefill is wall-clock neutral/better; a material regression would
   force either a fix or an explicit prefill-vs-verify predicate.
3. **DSpark acceptance** — the matched acceptance-rate A/B decides the #292
   question (13% c=1 / 0% c=8 gap attributed to varlen as a suspicion only):
   unchanged gap with numerically-equal paths discharges the varlen suspicion
   (#300 then stands on simplicity alone); a moved gap or a numeric divergence
   locates the cause in the recurrent/chunked difference.

## Parameters

- Kernel bundle hash: changes because the serving bundle links the h24g8/h12g4
  flashqla cubins into the dispatch set — record `KERNEL_BUILD_ID` on baseline
  and treatment.
- Numeric gate: `gdr_varlen_parity` at (16,48)/(8,24)/(4,12), lengths
  5/17/64, both paths vs f64 anchor; record the VARLEN-vs-FQ maxdiff per
  geometry/length.
- Correctness: `scripts/needle_gate.py` and `scripts/lever_gate.sh` at
  attn_tp=2, needle ladder ×3 vs the baseline envelope (same prompt/model).
- Prefill A/B: attn_tp=2, long-prompt set, matched concurrency/prompts,
  measure TTFT and prefill tok/s, baseline (recurrent) vs treatment (chunked).
- DSpark A/B: same model/prompt/concurrency, DSpark spec verify on, attn_tp=2;
  chain-acceptance rate c=1 and c=8; trials per the bench spec.
- Metrics: needle/lever pass, TTFT Δ%, prefill tok/s Δ%, acceptance-rate Δ.

## Environment

- Host / GPU: `<1×H20 sm90, pod — pending-remote>`
- Driver / CUDA: `<pending-remote>`
- Model / dtype: `<ThinkingCap/Qwen3.6-27B FP8 — pending-remote>`
- TP / slots: attn_tp=2 and attn_tp=4 runs
- Server flags: `--qwen35-gdr-chunked` (default on), DSpark spec verify on

## Result

pending-remote: H20 bundle rebuild (bundle hash change), numeric
`gdr_varlen_parity` PASS at the new geometries with maxdiff numbers, the
needle/lever correct-inference gate, the prefill TTFT A/B, and the DSpark
acceptance A/B. Decisions per check as listed under Hypothesis.

## Rule

When an AOT geometry exists in the manifest but the serve geometry predicate
hard-codes a smaller set, the predicate — not a new kernel — is the gap; and a
single predicate that widens a recurrence path moves EVERY caller (prefill,
verify, replay), so gate numerics, correct inference, and prefill wall-clock
before flipping the router.

