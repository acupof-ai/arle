# qwen4_exp batched prefill: forward_prompt lands bit-exact at full scale, and the gate fires on two real bugs

## Context / Goal

Prefill was the decode loop per prompt token: a 2048-token prompt cost ~2048 ×
85 ms ≈ 3 minutes. Target: >= 100 tok/s at chunk 256, measured, with
prefill-then-decode EQUAL to the per-token loop (a prefill that diverges from
decode is a wrong prefill).

## What Worked

**The chunked forward (`forward_prompt`, `model_qwen4_exp.rs`).** Chunks of
`T` tokens (default 256, `ARLE_QWEN4_PREFILL_CHUNK`):

- Linear attention and PLE: the existing kernels' `seq_len = T` modes
  (`qwen35_ssm_conv` / `qwen35_gated_delta_net` / `qwen4_ple_gate` /
  `qwen4_ple_conv`) against the SAME resident state buffers decode advances.
  The HF→GGUF head maps became four chunk-wide token-major map buffers — one
  `qwen4_block_perm` dispatch per tensor per layer instead of `4T`.
- Full attention: batched q/k norms + RoPE, strided KV pack into the planes at
  `[start_pos..start_pos+T]`, ONE causal-masked flash dispatch per layer.
- Hyper-connections and MoE: per token within the chunk (accepted v1 — no
  batched HC kernel; per-token expert routing). The MoE ids fence batches:
  ONE read of `T × top_k` ids per (layer, chunk) — the same fence count per
  chunk that decode pays per token. One n-gram pool gather and one embed seed
  per chunk.
- Dense projections, TWO routes: the DEFAULT records the same per-token GEMV
  dispatches decode records (bit-exact by construction — proven below); the
  opt-in coopmat lane (`ARLE_QWEN4_PREFILL_GEMM=1`) records ONE GEMM per
  weight via a new `MmCmBf16` variant (bf16-bit A decoded through
  `TO_FLOAT_TYPE`, PLAIN F16 B through a two-macro `TO_FLOAT_TYPE_B` vendor
  seam that defaults to upstream behavior everywhere else; staging =
  `f16_kv_pack`, like every other coopmat GEMM). Device proof
  `device_mm_coopmat_bf16.rs`: worst 1.7e-6 of term magnitude vs a
  bf16(A)×f16(B) reference, including offset-bound slab/arena bindings.

**The gate (`tests/qwen4_prefill.rs`) fired on two real bugs.** A truncated
4-layer model (linear / linear+PLE / linear / full — every stage class) in the
`SubsetF32` residency, 24 real-checkpoint tokens through `forward_token` one
at a time, reset, replayed through `forward_prompt` at chunk widths 7 (uneven
multi-chunk) and 24. Compared: final logits, every linear layer's GDN S + conv
ring, the PLE ring, the full layer's KV rows, `seq_len`.

1. **An in-place `add.comp` race (subset scale, max rel 1.2e3).** The
   per-stage table localized it in one read (layer 0 state and the PLE ring
   bit-exact, everything downstream of the PLE residual add wrong): the chunk
   path added `h += ple_out` through `add.comp` with `dst` aliasing `src0`,
   and that shader's workgroups cover overlapping index ranges (each thread
   handles `idx` and `idx+256`) — benign into a distinct buffer, a
   read-after-write race in place. Decode never aliases through it; the fix
   is decode's own alias-safe weighted-accum. After: **max rel 0.000e0,
   bit-exact, both chunk widths.**
2. **Staged GEMM activations flip the full-scale argmax — for bf16 AND f16
   staging.** The first cut staged B as bf16 (2^-8; upstream's single
   `TO_FLOAT_TYPE` forces it): full-scale parity read max rel 1.227e3, ~1.2
   absolute logit drift, argmax flipped. An `ARLE_VK_DISABLE_COOPMAT=1`
   ablation pinned the whole divergence on the GEMM arm: the GEMV fallback at
   full scale is **bit-exact (0.000e0)**. The `TO_FLOAT_TYPE_B` seam then cut
   staging to f16 (2^-11) — at 4 layers drift drops 8.07 → 2.01 max rel
   exactly as precision predicts, but at 48 layers the full-scale drift READ
   2.6 absolute and flipped the argmax AGAIN: the compounding perturbation
   crosses expert-selection boundaries in the 512-expert routers, after which
   drift saturates on O(1) expert flips and no longer scales with the seed
   rounding. Conclusion adopted: on this model no sub-f32 activation staging
   equals decode, so the GEMM lane is opt-in and the DEFAULT prefill is the
   decode-identical GEMV route. The new
   `gemm_route_drift_stays_in_the_f16_envelope` test (subset +
   `ARLE_QWEN4_SUBSET_DENSE=bf16`, a 20-second repro of the full dense tier)
   pins the lane's envelope at 4.0: honest f16 staging (2.0) passes; a
   bf16-staging regression (8.1) or a re-vendor clobbering the seam
   (O(1e2+)) fails.

Mutation check (this round): a `c <= limit` → `c < limit` mutation of the
causal mask fails the gate at max rel 2.2e2, localized by the table to the
attention output alone (states bit-exact). Reverted, gate green.

## Full scale, measured

`ARLE_QWEN4_PREFILL=1 ARLE_QWEN4_PREFILL_PARITY=1`, HybridExperts residency,
512 synthetic tokens, Performance power mode, same sitting:

| route | chunk 64 | chunk 256 | 24-token prefill-vs-decode parity |
| --- | ---: | ---: | --- |
| decode loop (baseline) | 11.8 tok/s | 11.8 tok/s | — |
| **default (GEMV, ships)** | **31.0 tok/s** | **30.3 tok/s** | **max rel 0.000e0, max abs 0.000e0, argmax equal** |
| opt-in GEMM lane | 49.9 tok/s | 60.8 tok/s | max rel 1.4e3, abs 2.6, argmax FLIPS (198→271; decode top-2 gap 0.19) |

The 100 tok/s bar is NOT met this round, and the per-stage wall says exactly
why: `pf.moe.ids_fence` — the once-per-(layer, chunk) flush-and-read of the
router ids, which drains ALL GPU work recorded for that layer — is 14.8 s of
the 16.9 s chunk-256 wall (87%) on the default route and 7.0 of 8.4 s (83%)
on the GEMM lane. What it drains is the accepted v1 bottleneck: per-token
NVFP4 expert GEMVs (~74k dispatches re-streaming the active expert bytes per
token; on the GEMM lane the residual ~62 tok/s ceiling ≈ active-expert bytes
/ ~205 GB/s). The dense route choice only moves the smaller dense share
(31 → 61 tok/s).

Next levers, in order of measured leverage:

1. **Grouped / indirect expert dispatch** (explicitly out of scope this
   round): batch each expert's tokens into one GEMM — or read ids on device
   via indirect dispatch, killing all 48 fences per chunk. This attacks the
   87%.
2. **An exact fast dense lane**: two-GEMM f16 residual split
   (`x = hi + r`, both staged f16, summed f32 — recovers ~2^-22 activation
   precision) could let the coopmat lane meet the equality bar; must be
   re-measured against the gate, not assumed.

## Rule

- A kernel whose workgroups cover overlapping index ranges (llama.cpp's
  `add.comp` family: `num_iter > 1`) must never run with the output aliasing
  an input — duplicate writes of identical values become read-after-write
  races the moment they stop being identical. In-place accumulates go through
  a one-thread-per-element kernel.
- **Sub-f32 activation staging does not "average out" over a deep MoE stack —
  it saturates.** Drift that scales with staging precision at 4 layers
  (8.07 vs 2.01) stops scaling by 48 layers (1.2 vs 2.6 absolute, both
  argmax-flipping): once the perturbation crosses a top-k routing boundary,
  a different expert runs and the divergence is O(1) regardless of the seed
  rounding. Judge numeric shortcuts at full depth against a behavioral
  invariant; a small per-stage table entry proves nothing about the shipped
  configuration.
- Ablate before debugging numerics: one `ARLE_VK_DISABLE_COOPMAT=1` run
  split "the GEMM arm is wrong" from "the chunk machinery is wrong"
  bit-exactly, in one load.
- An equivalence gate wants matched STATE, not just matched logits: the
  per-stage state comparison (ring exact, downstream wrong; states exact,
  logits wrong) localized both bugs and the mutation without a debug print.
