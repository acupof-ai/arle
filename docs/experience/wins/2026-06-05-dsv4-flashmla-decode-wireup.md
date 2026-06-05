# DSv4-Flash decode: FlashMLA fused kernel wired up (correct), occupancy ncu pending

**Date:** 2026-06-05. **Backend:** CUDA, DSv4-Flash FP8 TP=8/EP=8, 8×H20.
**Status:** wired + correctness-verified, **gated** (`ARLE_DSV4_FLASHMLA_DECODE=1`,
scalar bf16 default kept). Spec:
[`2026-06-05-flashmla-sparse-decode-already-vendored-wireup-spec.md`](../../research/2026-06-05-flashmla-sparse-decode-already-vendored-wireup-spec.md).

## Context

The deep-read found the FlashMLA fused sparse-decode kernel + shim + FP8-KV pack +
selector were **already vendored in ARLE's tree, just unwired** — the 3 scalar
SW/CSA/HCA kernels were the anti-pattern (ncu: SM 1-3%, tiny B=1 grid). The job was
runtime wire-up, not a port (`先用最好的再自己写` —
[[feedback_no_closed_door_solutions]]).

## What Worked

In `attention.rs`/`dsv4.rs`/`loader.rs`/`tp.rs`: reverted the scalar split-KV
stepping-stone; un-gated the DSv4 FP8 KV arena; added the `attn_sink_f32` mirror +
TP BF16 all-gather for Q → FlashMLA repack/slice; env-gated eager decode dispatch
through `arle_flashmla_sm90_sparse_decode_fwd`; kept scalar bf16 as the reference
fallback; disabled the decode graph when FlashMLA is on.

**Verified (8×H20):** pod CUDA build passed; **symbol gate** —
`arle_flashmla_sm90_sparse_decode_fwd` present in `libkernels_cuda.a` (the archive
check, since the final binary `strings` didn't expose it — the
`2026-05-28-...precond-fail` trap avoided); 2-tok smoke `[11111, 603]`; **16-tok
exact oracle match**. Decode (16-tok smoke): **27.811 tok/s vs 25.5 baseline
(+9%)**.

## Honest read

+9% is the smoke number and **modest** — the wire-up added a per-step TP Q
all-gather + FP8-KV pack that partly offset the occupancy win. The occupancy proof
(ncu SM 1-3% → full) and the steady-state perf A/B are **not done** — this is
correctness/reachability, not a final perf claim. Finalizing it is gated by the
velocity problem (every A/B reloads the 149 GB model ~112 s); the next step is a
resident A/B harness (load once, run variants) before the ncu/perf pass.

## Rule

Wiring a vendored fused kernel can be net-modest at first if the wire-up adds new
per-step ops (TP all-gather, FP8 pack) that offset the kernel win — the occupancy
ncu, not the end-to-end tok/s, is what licenses the kernel; the surrounding ops are
the next optimization. Commit the gated, correctness-verified wire-up first; prove
the occupancy + trim the wrapper ops second.
