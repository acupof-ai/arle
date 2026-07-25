# Qwen3.6 DSpark block draft — LICENSED, 2.4× plain decode (short-ctx greedy)

> Status: Licensed 2026-07-10 (short-ctx greedy, backbone-only z-lab draft).
> Remaining before the OPD-rollout claim: long-ctx A/B + prefix-restore gap.

## Context

`--spec-type dspark --mtp-draft-model <dir>` (plan:
2026-07-09-dspark-dflash-spec-decode-qwen36).
Verify/rollback substrate reused from the MTP lane; drafter = z-lab DFlash
5-layer block-16 (backbone-only; markov/confidence heads pending AEON/FR
checkpoints). H20, Qwen3.6-27B-FP8, GPU-idle, greedy 500-tok probes.

## Optimization ledger (one variable per round, ARLE_DSPARK_PHASE)

| round | draft | verify | total step | tok/s | change |
|---|---|---|---|---|---|
| baseline | 63.2 | 177.2 | 248.7 ms | 17.0 | as-landed dd79c713e |
| TILE==B gemv tweak | 63.3 | 201.8 | 273.3 | 15.5 | **regression, reverted** |
| fix A | 63.5 | 72.3 | 144.1 | 29.1 | `QWEN_FP8_DEEPGEMM_DENSE_MIN_M` 64→16 |
| fix B | **7.7** | **24.9** | **36.2** | **104–108** | bf16 gemm small-N GEMV loop 16→4 + full lane routing (7f84a3371) |

Plain decode: 43.6 tok/s (control unregressed). **Net 2.4×** at accept
3.3–3.4/16 (think-block); acceptance 2.8–5.4 across prompts.

## Root cause (attributed, not inferred)

Every dominant phase ≈ single-row decode cost × 16: the quant dispatch ran
B=16 as 16 sequential GEMVs (dense_ffn 5.7 ms/tok FLAT at M=8/16/32) while the
DeepGEMM lane costs 0.203 ms/tok at M=800 — 28× cheaper. Same class in the
bf16 gemm fallback (N≤16 GEMV loop re-streams the weight N times; lm_head
16-row 21.6 ms → ~1 ms). Fix = lane routing, not kernel tuning — the TILE==B
attempt regressed +13.9% and was reverted.

## Gates

- Needle ×3 exact (`MAGENTA-TIGER-42`), nonce-busted prompts, spec path.
- Self-consistency ×3: coherent tails, spec path.
- Plain-decode control: 43.6 tok/s, unchanged.
- DeepGEMM-vs-GEMV greedy flips are deterministic per path (expected numerics;
  byte-identity is not the gate per correct-inference doctrine).
- Needle with max_tokens=24 false-failed 3/3 — think preamble eats tiny
  budgets; gate with max_tokens ≥ 700 on think-enabled serving.

## Open before the OPD-rollout claim

1. **Prefix-restore gap**: a prefix-cache-hit request has no draft ctx
   features → silently degrades to plain decode. OPD rollout hits ~91% —
   DSpark is near-inert there until taps are sidecar'd or partial-ctx drafting
   lands. This is the decisive item.
2. Long-ctx (20–45K) A/B: draft attn (4.9 ms, now 64% of draft) and
   gdr_recurrent scale with ctx; short-ctx 2.4× will compress.
3. Markov (+AEON) / confidence (+FR-converted) head rounds; MTP-d2 arm.
4. Small-M DeepGEMM still ~4–5× above the M=800 per-token floor
   (dense_ffn 26 ms profiled @16) — next verify wall if needed.
5. `chunked_prefill_size` clamped to ≥2048 on CUDA Qwen (loaded.rs:1796,
   audit QW-KV-07) — recorded, untouched.
