# W4A8 substrate output is GARBAGE — 100% token diff vs BF16, accuracy bug blocks default-on

> **Critical finding** discovered by Claude-added greedy_consistency test
> `test_w4a8_vs_bf16_token_diff`. Reframes W4A8 substrate LAND
> [`e61d26e`] from "TTFT win + ITL eager penalty" to "**fast but
> producing garbage**". The bench numbers measured token rates of
> garbage output; performance characterization is unaffected, but
> the production-default flip blocker chain is now:
>
> 1. **Accuracy fix** (W4A8 quantize OR kernel OR dispatch — root cause TBD)
> 2. Then graph capture hoist (`62e75ee` plan)
> 3. Then default-on flip
>
> Both prerequisites are codex-substrate work. W4A16 Marlin remains the
> only production-viable W4 path.

## Test that caught it

`infer/tests/greedy_consistency.rs::test_w4a8_vs_bf16_token_diff` (added
this session): runs greedy decode on the prompt "The capital of France is"
with both checkpoints, max_tokens=32, deterministic GEMM enabled. Asserts
token-level diff ≤ 25% threshold.

## Result

```
prompt: "The capital of France is"

BF16 baseline:
  " Paris. The capital of Germany is Berlin. The capital of Italy is Rome.
   The capital of Spain is Madrid. The capital of Portugal is Lisbon.
   The capital"

W4A8 Marlin:
  ".........11.1.11111111 baudaskan1 baud111askan11"

First divergence: idx=0 (bf16=12095 ≈ "Paris", w4a8=13 ≈ ".")
Diff: 100% (matched first 0/32 tokens)
```

Test FAILED at threshold 25%; actual diff 100% means the W4A8 path produces
completely unrelated output. Not a quantization-precision drift — it's a
**fundamental correctness bug**.

## Reframing W4A8 substrate LAND

Wins entry `e61d26e` reported:
- TTFT 1633.6 ms (-36% vs W4A16 ✅)
- ITL 19.23 ms (+63% vs W4A16, eager-mode penalty ⚠)

Both are **performance numbers measured on garbage output**. The kernel
runs (no crash, no NaN), produces deterministic tokens, but the tokens
are unrelated to the prompt's true continuation.

The performance characterization is still useful — TTFT -36% does
suggest the FP8 mma compute path engages, even if the output values are
wrong. The kernel is "computing the wrong thing fast".

**Updated phase 8 verdict** (supersedes `e61d26e`): W4A8 substrate
**KILL pending accuracy fix**, not "LAND deferred". Production should
NOT route to W4A8 at any default level until output matches BF16 to
within < 5% diff (loose) or ideally < 1% (strict).

## Root cause hypotheses (ranked by likelihood)

1. **Quantize script `/tmp/quantize_qwen3_w4a8.py` scale extraction wrong**
   - Per-channel scale (s1) and per-group scale (s2) may be miscomputed
   - 252 linear tensors all wrong → no token can produce sensible output
   - Verify: load a known-good W4A8 checkpoint from elsewhere, compare scales

2. **`marlin_w4a8_kernel.cu` dequant logic wrong**
   - Codex's 987-LOC kernel may have an off-by-bit or sign error
   - Verify: smoke test with known-input/known-output unit test

3. **`w4a8_activation_quant.cu` BF16→INT8 wrong**
   - 59-LOC activation quantizer may scale activations incorrectly
   - Verify: round-trip activation quant/dequant on known input

4. **Dispatch wrapper `run_marlin_w4a8` arg ordering / scale plumbing**
   - `linear.rs` Rust wrapper may pass wrong A/B/C/D or s1/s2/s3 args
   - Verify: trace one linear layer's GEMM args

5. **Weight loader scale tensor wiring** (`weight_loader.rs`)
   - The W4A8-specific scale tensors `marlin_w4a8_s_channel` and
     `marlin_w4a8_s_group` may load with wrong dtype or layout
   - Verify: dump scale tensor stats, compare BF16 dynamic range

## What to do BEFORE re-bench

1. Add a **per-tensor numerical sanity check** to weight_loader.rs that
   logs scale tensor statistics (min/max/mean) for the first few layers.
   Mismatch with BF16 weight dynamic range → quantize bug; match → kernel/dispatch bug.

2. Run a single linear-layer **unit test**: feed known input through one
   W4A8 linear (e.g., q_proj on layer 0) and compare output magnitude to
   BF16 q_proj on the same input. If magnitude differs by > 10×, scale
   loading is wrong.

3. **DO NOT** flip production default to W4A8 under any circumstance
   until `test_w4a8_vs_bf16_token_diff` passes ≤ 5% diff.

## Action — commit failing test (track regression)

The test at `test_w4a8_vs_bf16_token_diff` fails today; this is the
gate that blocks W4A8 default-on. Keeping the test failing is correct
behavior — once fix lands, this same test passes and unblocks the flip.

Skill v1.3.0 rule (anti-pattern #6 license-on-evidence-of-reuse):
performance bench measures the kernel pipeline ran; it does NOT verify
the kernel computed the right thing. Greedy_consistency vs reference
is the necessary semantic gate; substrate LAND must include both
performance + correctness.

## Brief to codex (queued)

W4A8 substrate has accuracy bug. Performance + correctness gates are
SEPARATE; neither alone is sufficient for default-on flip. Graph
capture hoist (`62e75ee`) addresses ITL but does NOT fix accuracy.

## Cross-references

- W4A8 substrate LAND wins entry (now reframed): [`docs/experience/wins/2026-05-08-w4a8-marlin-prod-bench-mixed-outcome.md`](../wins/2026-05-08-w4a8-marlin-prod-bench-mixed-outcome.md) (`e61d26e`)
- W4A8 graph capture hoist plan: [`docs/plans/M_quant-w4a8-graph-capture-hoist.md`](../../plans/M_quant-w4a8-graph-capture-hoist.md) (`62e75ee`)
- Codex W4A8 substrate: `crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu` (987 LOC) + `w4a8_activation_quant.cu` (59 LOC) (`a019a0e`)
- Quantize script (uncommitted): `/tmp/quantize_qwen3_w4a8.py` (codex authored)
- ARLE dispatch wrapper: `infer/src/ops/linear.rs::run_marlin_w4a8`
- Skill v1.3.0: anti-pattern #6 (license on real reuse) + skill rule #2 (measure binding constraint, not only proxy)
- Failing test commit: this entry's companion commit
- W4A16 Marlin license bench (correctness verified): [`f6f3af3`](../wins/2026-05-08-m_quant-w4a16-marlin-bench.md)

## Rule

W4 substrate LAND has **two gates**, not one:

1. **Performance gate**: matched A/B vs BF16 baseline at production-default
   KV (skill v1.2.0 isolation-motive callout)
2. **Correctness gate**: token-level diff vs BF16 ≤ 5% (loose) or ≤ 1%
   (strict, default-on flip)

A "fast garbage" outcome is worse than slow correct output. Document
both gates' status; do NOT report only the performance gate as
"substrate LAND".

The W4A8 wins entry `e61d26e` violated this rule by reporting bench
numbers without running the correctness gate first. This errors entry
issues the correction.
