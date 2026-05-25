# FP8 KV step-1 catastrophic divergence — reproduced + deferred to Phase 3

## Context

`docs/plans/2026-05-25-kv-precision-parity-framework.md` Phase 2 ran the new
per-precision parity harness (`infer/tests/kv_precision_parity.rs`) on L4 /
Qwen3-4B BF16 / 4-prompt × 64-token and 8-prompt × 256-token configs. The
audit reproduces the 2026-05-02 / 2026-05-05 FP8 KV bug exactly:

| Config | FP8 `mean_match` | First divergence |
|---|---:|---|
| 2 prompts × 32 tokens | 1.0000 | (sample too small — false-pass) |
| 4 prompts × 64 tokens | 0.0156 | prompt 0, step 1 |
| 8 prompts × 256 tokens | 0.0039 | prompt 0, step 1 |

`mean_match = 1/64 = 0.0156` or `1/256 = 0.0039` means every prompt outputs
the same token as BF16 at step 0 (prefill's last logit) and **diverges at
the very first decode step** — identical signature to the 2026-05-02
"token-1 divergences 30/32" reading.

## Root cause (status: hypothesis only)

Step 0 logit is the prefill kernel's last position output, computed against
the BF16 working buffer — agrees with the BF16 reference. Step 1 is the
first true decode that reads K/V from the FP8 paged pool for all prior
positions (the prefill rows quantized by `finalize_paged_prefill_kv_layer`
+ the new decode write). Divergence at step 1 isolates the failure to one of:

1. `finalize_paged_prefill_kv_layer` writes FP8 rows whose dequantized
   values drift far from the BF16 reference.
2. `decode_attention_fp8` reads scales / rows with wrong indexing
   (the scales-layout `[row_idx * num_kv_heads + kv_head]` vs the read
   path at `decode_attention_quantized.cu:381` matches in source survey,
   but the audit has not yet confirmed bit-identity of the durable bytes).
3. The per-token-per-head scale recomputation between prefill-time
   quantization and decode-time quantization desynchronizes attention's
   key/query interaction.

These remain hypotheses pending the diagnostic work listed below. The
2026-05-05 errors entry already enumerated the right next steps; no one has
landed them.

## Fix — deferred to Phase 3

This session does not attempt a numerical fix. The harness exposes the bug
deterministically; the next session should:

1. Add a unit test under `crates/cuda-kernels/tests/` that constructs a
   known BF16 K/V tensor, calls `quantize_scatter_kv_fp8_range` then
   `dequantize_paged_kv_fp8_to_hnd`, and compares L1 / L2 / L∞ delta vs the
   source BF16 across realistic Qwen3 head configs (8 KV heads × 128 head_dim).
2. Independently, run a single-prompt prefill on L4 in both BF16 and FP8
   modes; dump the durable FP8 K/V bytes + scales for the prefill rows at
   layer 0, layer 31; compare the dequantized values against the BF16 K/V
   for the same rows. This isolates whether the bug is in the quantizer
   or the consumer.
3. If (1) and (2) pass, instrument `decode_attention_fp8` with a row-by-
   row attention-score readback at layer 0 step 1 and compare against the
   BF16 attention scores for the same prompt. This isolates whether the
   bug is in the kernel's scale handling vs the per-token offset math.

Audit gate: `gate_trajectory: None` (report-only) until trajectory match
≥ 0.95.

## Operational fallback — auto-default

`auto` is no longer FP8. `infer/src/main.rs::kv_mode_candidates` now emits
`[BF16]` only; FP8 is opt-in via `--kv-cache-dtype fp8` with the
divergence behavior called out in the CLI help. Until Phase 3 lands, FP8
must be regarded as **for memory experiments only, not for correctness-
sensitive workloads**.

## Rule

- A hypothesis with 24 days of failed patches is not "almost fixed". The
  2026-05-02 / 2026-05-05 / 2026-05-12 (six FP8 KV optimization kills)
  sequence shows iterative band-aids without a parity gate keep producing
  same-shape regressions. Until the parity gate (this session's harness)
  passes for FP8, no FP8 optimization should land.
- Short smokes lie. The 2-prompt × 32-token smoke gave FP8 `mean_match = 1.0`
  (false-pass) because greedy argmax happens to agree on the first 32 tokens
  for short prompts. The 256-token horizon is where the bug surfaces. Any
  future FP8 KV smoke must use ≥ 64 tokens; the 32-token cap is banned.

## Cross-refs

- Plan: [`docs/plans/2026-05-25-kv-precision-parity-framework.md`](../../plans/2026-05-25-kv-precision-parity-framework.md)
- Wins: [`docs/experience/wins/2026-05-26-kv-precision-parity-framework-tq4-routing-fix.md`](../wins/2026-05-26-kv-precision-parity-framework-tq4-routing-fix.md)
- Prior FP8 KV bug:
  - [`2026-05-02-qwen3-fp8-kv-numerical-tier1-fail.md`](2026-05-02-qwen3-fp8-kv-numerical-tier1-fail.md)
  - [`2026-05-05-fp8-kv-tier1-still-fail.md`](2026-05-05-fp8-kv-tier1-still-fail.md)
- INT8 long-decode drift (a co-discovered, lower-priority issue): same
  parity audit at 8 × 256 shows INT8 `mean_match = 0.8901` with prompt 1
  diverging at step 242. Short-decode (≤ 64 tokens) passes the 0.99 gate.
  Tracked here for next-session investigation; not blocked on this entry.
