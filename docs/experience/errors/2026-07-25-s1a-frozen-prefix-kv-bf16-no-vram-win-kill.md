# Frozen prefix-KV bf16 (Stage 1a) delivers no VRAM win on Qwen3.6-27B — KILLED

## Context

Full-chain bf16 tape plan (`docs/plans/2026-07-25-fullchain-bf16-tape.md`)
targets moving the agent-OPD writeback OOM wall 30K→60K by storing retained
buffers bf16. §7 chose **Stage 1a = frozen prompt-prefix K/V** (`PrefixKv.k/v`,
qwen35.rs:1665) as the minimum first A/B, on the premise it is "the largest
single retained buffer." Shipped S1a (commit `2738a5b90`): `quantize_frozen_to_bf16`
(Backend) + `quantize_frozen_bf16` (TensorStore), gated on `--tape-precision bf16`.
Verified on the H20 pod against ThinkingCap-Qwen3.6-27B-FP8 (r16 α32 attention-qv,
`--writeback-frozen-prompt-kv true`, deterministic ob172 fixed-length replay so
both arms share seq lengths).

## Root Cause

**The premise is false for this architecture.** `PrefixKv.k/v` is not the largest
retained buffer — it is ~400× smaller than the buffer that sets the peak.

Qwen3.6-27B: 64 layers, only **16 full-attention layers** (interval 4), n_kv=4,
head_dim=256 → PrefixKv f32 = **0.13 MB/token**. The other 48 layers are GDN
linear-attention; their forward prefix-capture transient (`la preact/v/output`)
is what dominates. At seq1024 (prefix 960 + 64 gen), the *only* length that fits,
the allocator grew **+52.7 GB** for the 960-token prefix capture — S1a acts on a
0.13 GB buffer against a 52 GB transient.

Measured A/B, peak VRAM at the backward writeback (GPU 1, `ARLE_OPD_VRAM_TRACE=1`):

| arm | post_backward MiB | loss |
|-----|-----|-----|
| fp32 | 81225 | 0.000155 |
| bf16 | 81513 | 0.000155 |
| **Δ** | **+288** | **0** |

**bf16 is +288 MiB HIGHER, not lower.** The f32→bf16 quantize double-buffers
(bf16 allocated while f32 still resident, then f32 dropped); that +320 MiB
transient exceeds the 63 MiB it shrinks. And the OOM wall did not move: every
seq ≥ 2048 OOM'd in the **forward** prefix-capture (upstream of the backward),
on both precisions.

Mechanism itself is correct — loss byte-identical (0.000155 both arms), needle
5/5 lengths 3/3 DET, fp32 default + Metal byte-identical. The plumbing works; it
points at the wrong buffer.

## Fix

Skip Stage 1a as a VRAM lever — but keep the shipped code: `--tape-precision`
config (S0) + the quantize/widen mechanism (S1a) are the substrate S1b/S2 reuse,
and S1a is a correct no-op at the fp32 default. The peak is set by the
**forward linear-attention capture transient**, so the win lives in Stage 1b
(retained activations, specifically the GDN `la_*` forward buffers over 48
layers), not the frozen K/V. Re-scope the plan: §7 Stage-1a is rejected; the
first buffer worth attacking is the `la preact/v/output` chain.

Second, the quantize double-buffer must not add a transient larger than it saves
— when S1b quantizes activations, free the f32 source before/as the bf16 is
allocated (or quantize in place), else every store site pays the same +320 MiB
penalty that sank S1a.

## Rule

- **Measure the buffer that sets the peak before optimizing a named buffer.** A
  plan-level "largest single retained buffer" judgment (§7) was wrong by 400×
  because it reasoned from a dense-attention mental model, not this MoE+GDN
  config's layer mix (16 full-attn of 64). Probe `allocator_retained_delta` /
  the VRAM trace first; don't infer the dominant buffer from architecture prose.
- **A store-time dtype downcast that double-buffers can cost more than it saves.**
  Net VRAM = (buffer shrink) − (transient f32+bf16 coexistence). For a small
  buffer the transient dominates and the flip is a net loss. Quantize in place or
  free-then-alloc.
- `--tape-precision` is **train-only**; the serve/rollout path never reads it, so
  a needle gate certifies build inference correctness, not an fp32/bf16 serve diff.
