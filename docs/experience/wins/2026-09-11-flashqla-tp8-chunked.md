# Chunked FlashQLA GDN on the attn_tp=8 (2,6) shard — cuda, 2026-09-11

> Status: pending-remote
>
> Update (#300, same day): `gdr_varlen_parity` was renamed
> `flashqla_chunk_parity` after the varlen half was deleted; run the
> FlashQLA-vs-f64 checks below under the new name.

## Goal

Close the last TP gap so the recurrent varlen GDN kernel has no production
caller. #327 routed every multi-row advance (ordinary prompt prefill, batched
spec verify, DSpark rollback replay) through chunked FlashQLA at global
(16,48), attn_tp=2 (8,24) and attn_tp=4 (4,12); attn_tp=8 still fell back to
varlen because the local shard (Hg,H)=(2,6) had no AOT instantiation. This
adds the (2,6) cubins and routes the shard. After it lands #300 can delete
varlen for every TP size.

## What changed

- `crates/cuda-kernels/kernels.toml`: five new flashqla rows `q_heads=6,
  kv_heads=2` (fq_cumsum/kkt/fwd/prepare_h/bwd), symbols
  `gdr_fq_*_h6g2`, mirroring the h24g8/h12g4 rows. sm90, `gate="flashqla"`.
- `crates/cuda-kernels/src/ffi/recurrent.rs`: the five `*_h6g2_cuda` externs;
  the generated `FLASHQLA_GDR_TABLE` picks them up from the manifest.
- `infer-model/src/qwen35.rs` `fq_geometry_supported`: add (2,6); unit test
  now asserts (2,6) supported and keeps (1,3) negative.
- `infer-cuda/src/qwen35_attention.rs`: (2,6) fq_fns match arm selects the
  h6g2 cumsum/kkt/fwd symbols (prep is geometry-generic).
- `gdr_varlen_parity` `FQ_XCHECK_GEOMS`: add (2,6), so the new cubins are
  gated against the f64 anchor and against varlen at lengths 5/17/64 in the
  same PR that introduces them (no unvalidated-cubin window).

Template feasibility: all five flashqla kernels are per-head CTAs; H appears
only in the grid extent and `bh // (H//Hg)`, and the only head-count guard is
`h % hg == 0` (`tools/tilelang/flashqla_gdr.py:2078`). (2,6) has H/Hg=3, the
same group ratio as the shipping (16,48)/(8,24)/(4,12) geometries; no
minimum-heads-per-CTA or H-divisible tile blocks it. (Authoring Mac has no
nvcc/GPU, so compile is part of the pending-remote bundle build.)

## Hypothesis

The (2,6) instantiation runs the same GDN math as the other H/Hg=3
geometries; `gdr_varlen_parity` anchors both paths to one f64 recurrence, so
outputs and final states agree within the chunked path's bf16 tolerance at
attn_tp=8, and correct inference is unchanged.

## Parameters

- Bundle hash: changes — five new sm90 h6g2 cubins link into the bundle;
  record the new `KERNEL_BUILD_ID` vs the #327 build.
- Numeric gate: `gdr_varlen_parity` at (2,6) lengths 5/17/64, FQ + varlen vs
  f64 anchor; record VARLEN-vs-FQ maxdiff.
- Correct inference at attn_tp=8: `scripts/needle_gate.py` and
  `scripts/lever_gate.sh`, needle ladder ×3 against the baseline envelope —
  hard block, because the attn_tp=8 default prefill path changes.
- Baseline: #327 build (attn_tp=8 takes varlen); treatment: this lane
  (attn_tp=8 takes chunked FQ); same model/prompts/concurrency.

## Environment

- Host / GPU: `<8-GPU H20 sm90, pod — pending-remote>`
- Driver / CUDA: `<pending-remote>`
- Model / dtype: `<ThinkingCap/Qwen3.6-27B FP8 — pending-remote>`
- TP / slots: attn_tp=8
- Server flags: `--qwen35-gdr-chunked` (default on)

## Result

pending-remote: h6g2 bundle build + hash, numeric `gdr_varlen_parity` PASS at
(2,6), and the attn_tp=8 needle/lever gate. On pass varlen has no production
caller and #300 is unblocked to remove it for every TP.

## Rule

Every new AOT geometry ships in the same change as its host parity gate;
close the last shard of a fallback before the PR that deletes the fallback is
opened.
