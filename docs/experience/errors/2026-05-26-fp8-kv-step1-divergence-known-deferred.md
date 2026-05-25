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

## Root cause — narrowed to dispatch / wiring (2026-05-26)

Two production-layout roundtrip diagnostics now live in
`crates/cuda-kernels/src/kv_quant.rs` (`fp8_scatter_qwen3_production_layout_diagnostic`,
`fp8_paged_quantize_qwen3_production_layout_diagnostic`). Both exercise
the actual FP8 kernels called from the migration path
(`quantize_scatter_kv_fp8_range`) and the prefill-finalize / per-decode-step
path (`quantize_paged_kv_fp8`) at Qwen3-4B layout (num_kv_heads=8,
head_dim=128, 64 tokens) with realistic ±6 outliers and N(0, 2)
fill. Result on L4 (sm_89):

| Kernel | max_abs_err | mean_abs_err | max_rel_err | scale range |
|---|---:|---:|---:|---|
| `quantize_scatter_kv_fp8_range` | 0.109 | 0.022 | 21% | [0.0123, 0.0134] |
| `quantize_paged_kv_fp8` | 0.113 | 0.022 | 32% | [0.0123, 0.0134] |

Both within the expected FP8 E4M3 precision envelope. **The kernels are
not the bug.** The 0.4% trajectory match in the end-to-end audit must come
from dispatch or wiring upstream of these kernels:

1. `prefill_token_rows` passed to `finalize_paged_prefill_kv_layer` could
   address the wrong rows (off-by-page, off-by-slot).
2. `last_token_indices` in the per-decode-step write could address a
   different row than the `kv_indices` the decode-attention kernel reads.
3. K vs V scale pointers could be swapped at a higher level.
4. Layer-index propagation: layer N quantize could write to layer M's
   pool slot.
5. Mixed-batch vs pure-decode dispatch (`decode_attention_varlen_fp8` vs
   `decode_attention_fp8`) could mis-route for the first decode step
   following prefill in the same scheduler tick.

Phase 3 next-step refinement (replacing the 2026-05-05 list, which assumed
the kernel needed fixing):

1. Add an in-process integration test that boots Qwen3-4B in FP8 mode,
   runs prefill on a fixed short prompt, dequantizes the FP8 paged pool
   for the prefill rows, and compares against the same prompt's BF16
   prefill K/V layer-by-layer. Expect divergence at layer 0 if migration
   indices are wrong; expect drift at deeper layers if layer-state
   propagation is wrong.
2. If (1) reports all layers clean, instrument decode step 1 only:
   dequantize the FP8 pool's read region at decode-attention entry and
   diff against the BF16 mode's K/V cache view of the same positions.
3. ❌ Tried (2026-05-26, reverted): gate FP8 through the same
   contiguous-BF16-prefill path TurboQuant uses by excluding
   `KVFormat::FP8E4M3` from the `page_size == 16` whitelist in
   `scheduler/cuda/prefill.rs`. **Did not recover parity** — FP8 then
   diverged at step 0 (worse than original step 1) because the legacy
   non-paged CUDA prefill kernel is not bit-identical to the TileLang
   HD128 paged prefill kernel that BF16 uses; greedy argmax flips on the
   numerical diff alone. Conclusion: must keep FP8 on the paged path
   (same kernel as BF16) for any step-0 parity hope, and find the actual
   wiring bug in the FP8-specific finalize / quantize / decode call sites
   instead of trying to route around it.

The two diagnostic tests now serve as regression gates for the kernels
themselves — any future FP8 kernel change must keep them green.

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
