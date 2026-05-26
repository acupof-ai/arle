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

## Root cause — narrowed to decode-step path (2026-05-26)

A third diagnostic, `infer/tests/kv_fp8_prefill_logit_parity.rs`, drives
`forward_token_logits` on the same 16-token input in BF16 and FP8 KV
modes and compares the per-position logit vector (vocab=151936). On L4
sm_89:

```
fp8_vs_bf16_prefill_logits_parity:
  max_abs=0.000000  mean_abs=0.000000  max_rel=0.000000
  argmax_bf16=16  argmax_fp8=16  argmax_match=true
  top1_bf16_val=17.7500  top1_fp8_val=17.7500
```

**Bit-identical.** The entire prefill compute path — TileLang HD128 paged
prefill, BF16 work buffer attention, last-position logit extraction — is
shape- and dispatch-equivalent across both modes. The
`finalize_paged_prefill_kv_layer` quantize-after-attention step runs after
the logit is computed, so even if it wrote wrong bytes it would not show
up here.

The audit's catastrophic divergence at step 1 must therefore live in the
per-decode-step path, which is the only code that differs once we move
past the prefill boundary:

- Per-decode-step `quantize_paged_kv_fp8` writes the new token's K/V
  into FP8 paged storage using `last_token_indices` for the destination
  row. Off-by-row here would write to the wrong page slot.
- `decode_attention_fp8` reads prefill rows (written by
  `finalize_paged_prefill_kv_layer`) + the new row using `kv_indices` and
  `kv_last_page_len`. Off-by-row OR a stale-snapshot of `kv_indices`
  would read the wrong tokens.
- Per-token-per-head scale layout in the paged pool (the kernel writes
  `scales[row * num_kv_heads + kv_head]`; the decode-attention kernel
  reads `K_scales[row * num_kv_heads + kv_head]`). Source survey shows
  these agree, but a hidden stride or layer-index miswire would only
  surface in production.

Two production-layout kernel roundtrips already certify the kernel
correctness in isolation (`crates/cuda-kernels/src/kv_quant.rs`
`fp8_scatter_qwen3_production_layout_diagnostic`,
`fp8_paged_quantize_qwen3_production_layout_diagnostic`). Both exercise
the actual FP8 kernels at Qwen3-4B layout (num_kv_heads=8, head_dim=128,
64 tokens) with realistic ±6 outliers and N(0, 2) fill.

A third diagnostic, `fp8_kernel_pair_decode_attention_diagnostic`, now
wires the production kernel pair end-to-end (production
`quantize_paged_kv_fp8` writes → production `decode_attention_fp8` reads)
over a deterministic Qwen3-4B-shaped GQA 2:1 workload (num_q_heads=4,
num_kv_heads=2, head_dim=128, 32 KV tokens / 2 pages) and compares the
GPU attention output against a dequantize-then-host-compute reference.
Result on L4 sm_89: `max_abs_err=0.003784`, `mean_abs_err=0.000523` —
within BF16 truncation noise. The kernel pair, the kv_meta / kv_indices
layout interpretation, and the per-(token, head) scale plumbing are all
clean when the dispatch fields match the kernel's contract.

**CUDA Graph ruled out** (2026-05-26): re-ran the audit with
`INFER_TEST_CUDA_GRAPH=0` after teaching the harness to honor the
env. FP8 `mean_match = 0.0156`, `first_div = step 1` — bit-for-bit
identical to the graph-on result. Graph-capture write-order is not
the bug.

**Production-state runtime dump captured (2026-05-26 A100 sm_80, graph
off)** via `quant_debug_dump_fp8_state` in
`infer/src/model/qwen3/batch_decode.rs`, env-gated on `INFER_FP8_DEBUG=1`,
fires on layer 0 only. Pool sizing is correct (`pool_layers=36
pool_kv_dim=1024 k_data_len=807927808 bytes per layer`,
`k_scales_len=6311936 floats per layer`). The decode-step writes
update `last_token_indices` correctly between steps
(`4035 → 4036 → 4037 → 4038`) and the FP8 byte payload + scales
vary per row — the per-step quant kernel is doing its job.

**The smoking-gun signal**: with KV_PARITY_PROMPTS=1 against a 50+ BPE
token prompt, the first real decode step writes to `row=4035 =
page_252 * 16 + 3`, implying the slot's accumulated `seq_len = 4`
when `build_last_indices` was called — only **3 prefill rows
materialized** out of the 50+ the prompt expanded into. INT8 and
BF16 with the same prompt produce parity-correct sequences, so the
gap is FP8-specific to either `prefill_forward_paged_batch` (the
paged-prefill kernel sequence for FP8 chunks) or
`finalize_paged_prefill_kv_layer` (the FP8 finalize that quantizes
work → durable FP8 pool). The decode-step path is innocent; the
prefill path silently drops most of the tokens or fails to publish
their KV into the pool, leaving the decode-step attention to read
mostly-zero / dummy rows for positions 3..N.

Next instrumentation point: dump prefill_token_rows length and
pool.seq_lens[slot] before/after `finalize_paged_prefill_kv_layer`
to confirm where the prefill rows are getting lost. The kernel-
clean and dispatch-symmetry findings still hold for the decode-step
path; the failure is upstream in prefill.

**Structural symmetry confirmed**: INT8 mode and FP8 mode use the
exact same paged-pool buffers and dispatch shapes — both route the
prep K/V to `pool.k_ptr(layer)` (which is the shared `k_work` for
both quantized formats), both call `quantize_paged_kv_<x>` with the
same `last_token_indices` / `prefill_token_rows` plumbing, both call
`decode_attention_<x>` with the same `kv_indices` / `kv_meta`. The
only on-the-wire difference is the kernel name. INT8 passes parity,
FP8 fails — so the bug is in some FP8-specific behavior the kernel
parity tests have not yet exercised.

**Qwen3.5 hybrid audit (2026-05-26, V100 sm_70) — inverts the
signal**: ran the same harness adapted for `Qwen35Model` on V100
(L4 dense-only box offline). With `KV_PARITY_MAX_TOKENS=32`,
`KV_PARITY_PROMPTS=2`:

- `bf16   mean_match=1.0000`  ✓
- `int8   mean_match=0.0469`  ✗ catastrophic step 1
- `fp8    mean_match=0.8125`  ⚠ slow drift (first divergence step 20)
- `tq4    mean_match=0.0000`  ✗ kernel error

INT8 catastrophic stems from `CUDA_ERROR_NOT_SUPPORTED` at
`qwen35 decode full-attention layer_idx=3 full_idx=0` —
`decode_attention_int8_partial_kernel` uses `__pipeline_memcpy_async`
(`cp.async`, SM 8.0+), which Volta (sm_70) does not implement. So
INT8 on V100 is an architectural sm-coverage gap, not a numerical
parity bug. (`crates/cuda-kernels/csrc/attention/decode_attention_quantized.cu:142-148`.)

FP8 only drifting at step 20 on Qwen3.5/V100, instead of the
catastrophic step-1 divergence seen on Qwen3-4B/L4, is the strongest
signal yet: **the Qwen3-dense FP8 step-1 catastrophe is Qwen3-dense-
specific, not shared-kernel-pair-specific**. The 4-prompt × 64-token
follow-up audit (2026-05-26, V100) reinforces this — Qwen3.5 FP8 is
**prompt-dependent drift**, not catastrophic at any consistent step:

| Prompt | match | first divergence |
|---|---:|---|
| 0 | 0.5781 | step 37 |
| 1 | 0.9219 | (late or none) |
| 2 | 1.0000 | none |
| 3 | 0.1406 | (very early) |

Mean 0.6602 with no consistent first-divergence step is the
fingerprint of an honest FP8 precision-floor drift, not a wiring
bug. Compare with Qwen3-dense where every prompt diverges at step 1
with identical fingerprint — a structural dispatch break, not
precision noise. The Qwen3.5 full-
attention dispatch indexes the paged pool with `full_idx` (subset
index over the 8 full-attention layers) while Qwen3-dense uses
`layer_idx` over all 32 layers. The bug surface narrows to whatever
diverges between the two dispatch sites in
`infer/src/model/qwen3/batch_decode.rs` vs
`infer/src/model/qwen35/batch_decode.rs`, with the kernels themselves
already certified clean.

**Conclusion**: the audit's step-1 catastrophic divergence is in
scheduler-side runtime dispatch wiring of the values fed to these
kernels, not in any FP8 CUDA kernel. Remaining suspect surface:

1. The per-decode-step `last_token_indices` build (`tilelang.rs`
   `last_token_scratch` → `last_token_indices` H2D + `build_last_indices`
   in `paged_kv.rs`). If the new token's destination row is wrong, the
   per-step quantize writes to the wrong slot and `decode_attention_fp8`
   reads stale bytes there.
2. The `kv_indices` snapshot (`tilelang.rs` `indices_scratch` → H2D vs
   incremental-update GPU kernel). At decode step 1 the page list grows
   to include the newly allocated page — if the snapshot is stale, the
   attention misses the new token's page or reads from an evicted one.
3. The K vs V scale-pointer ordering across the per-layer
   `finalize_paged_prefill_kv_layer` + per-step paths — a swap would
   make every attention score multiplicatively wrong, exactly the
   "catastrophic from step 1" failure shape.
4. Layer-index propagation across `pool.k_data_ptr(layer)` /
   `pool.k_scales_ptr(layer)` between prefill finalize and decode steps —
   if layer L's decode reads layer M's pool data, every layer mixes
   wrong K/V. Result on L4 (sm_89):

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
