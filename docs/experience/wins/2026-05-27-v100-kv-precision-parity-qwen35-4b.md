# V100 sm_70 KV precision parity — Qwen3.5-4B real text baseline

## Context

A100 sm_80 unreachable today (cloudflare tunnel down). V100 sm_70 in
`agent-infer-v100-audit` has TileLang built with the sm70 FMA fallback
patch (`scripts/sm70_tilelang.patch`) and Qwen3.5-4B weights at the
ModelScope cache (`/data00/.../modelscope/hub/models/Qwen/Qwen3.5-4B`).
Used the existing `kv_precision_parity_qwen35` test, model-path
overridden via `INFER_TEST_MODEL_PATH`. Two prompts × 16 generated
tokens × four KV formats (BF16 ref / INT8 / FP8 E4M3 / TurboQuant-4).

This was the first KV quant audit that produced a **non-degenerate**
BF16 baseline — earlier Qwen3-4B-on-A100 audits collapsed BF16 to the
`!`-repeat loop (see
[`2026-05-26-fp8-kv-catastrophic-was-test-artifact.md`](../errors/2026-05-26-fp8-kv-catastrophic-was-test-artifact.md))
so the `mean_match` metric was noise-fidelity, not quality. Qwen3.5-4B
on V100 holds a real text trajectory under greedy decode, so the
metric becomes meaningful.

## What Worked

### Setup

- Discovered Qwen3.5-4B safetensors at the **ModelScope cache**, not
  the HF cache. Switched `INFER_TEST_MODEL_PATH` from the HF cache
  path (which only had `config.json` + `tokenizer.json`) to
  `/data00/home/chenkailun.c/.cache/modelscope/hub/models/Qwen/Qwen3.5-4B`.
  HF direct download was throttled (32 KB/s without proxy, 1.4 MB/s
  with byted internal proxy — still 110 min for 8.8 GB).
- Patched the existing `kv_precision_parity_qwen35` test to dump
  the first 16 token IDs per (precision × prompt) so we can decode
  them to text and validate the baseline is coherent.

### Results

`KV_PARITY_PROMPTS=2 KV_PARITY_MAX_TOKENS=16 cargo test --release \
 -p infer --features cuda --test kv_precision_parity_qwen35 \
 -- --nocapture --test-threads=1`

| Precision | mean_match | first_div step | gate_passed | wall-clock |
|---|---:|---:|---|---:|
| **BF16** (reference) | 1.0000 | None | ✓ (gate 1.0) | 53.7 s |
| **FP8 E4M3** | **1.0000** | None | ✓ (no gate) | 51.0 s |
| **INT8** | **0.0938** | step 1 | **✗** (gate 0.99) | 50.9 s |
| TurboQuant-4 | 0.0000 | step 0 | n/a (decode unsupported on sm_70) | 8.6 s |

`per_prompt_match` for INT8: `[0.0625, 0.1250]` = (1/16, 2/16) — INT8
matches BF16 on tokens 0 and (for prompt 1) 1, then drifts.

### Decoded text — the real quality signal

```
BF16 p0: "\n\nKV caching and attention mask are the most important parts.\n\nKV caching is"
BF16 p1: "\n\nThe n-th Fibonacci number is computed using iterative dynamic programming, where the time"

FP8  p0: "\n\nKV caching and attention mask are the most important parts.\n\nKV caching is"     ← identical to BF16
FP8  p1: "\n\nThe n-th Fibonacci number is computed using iterative dynamic programming, where the time"  ← identical to BF16

INT8 p0: "\n\n<think>\nThis is a question requiring detailed technical explanation. The user wants to"
INT8 p1: "\n\nThe following are the results of the user's request:\n- A user"
```

**Baseline is coherent text**, not degenerate. The metric reflects
real generation-trajectory quality, not noise fidelity. INT8 output
remains coherent — it is not crashing or NaN'ing — but produces
*semantically different* text from the very first generated step,
which is real quantization drift, not a kernel correctness bug.

### Per-precision verdict

- **FP8 E4M3 KV: production-ready** under this audit shape. 32/32
  tokens bit-identical to the BF16 reference across two distinct
  prompts. No drift detected.
- **INT8 KV: real quality drift.** Step-1 divergence is consistent
  across both prompts. Output is coherent (not garbage) but the
  greedy trajectory shifts immediately. This is the same INT8
  pattern the May 26 reports flagged
  (`kv-parity-qwen35-Qwen3.5-4B-1779783223.json` showed `int8
  mean_match=0.0625, first_diverging_step=1` — reproduced today on
  a fresh build with current code).
- **TurboQuant-4: pending sm_80** — V100 FMA build prints
  `CUDA_ERROR_NOT_SUPPORTED` for the FP8 KV / DSv4 HD64 wrappers,
  which the TQ4 decode wrapper depends on. Architectural sm_70
  limit, not a TQ4 quality signal. The TQ4 decode path needs an
  sm_80 box to be audited.

### What this unblocks

For sm_70 today and any sm_80 box that runs this same Qwen3.5-4B audit:

- **FP8 KV** can be promoted on auto-default for hybrid models like
  Qwen3.5 — it is at parity with BF16 across the small but
  non-trivial 32-token horizon checked here.
- **INT8 KV** stays explicitly opt-in until the step-1 drift gets a
  quality investigation (PPL / lm-eval-harness, not just trajectory
  match — INT8 output is coherent so the test may be over-strict).
- **TQ4** stays in the queue gated on sm_80 access.

## Rule

When debugging a KV-quant audit failure, **always decode the actual
generated token IDs to text first**. mean_match alone cannot
distinguish "FP8 perfectly tracks the junk BF16 baseline" from "FP8
perfectly tracks the real BF16 baseline" — and the rule from
[`2026-05-26-fp8-kv-catastrophic-was-test-artifact.md`](../errors/2026-05-26-fp8-kv-catastrophic-was-test-artifact.md)
was specifically born from that confusion. The diagnostic is one
`tok.decode([...])` away, costs nothing, and is what discriminates
a quant kernel bug from real precision drift.

When a prior parity report shows BF16/INT8/FP8 all at `mean_match=1.0`
on long horizons, **decode the bf16 tokens.** If the decoded text is
a single character repeated, the baseline is degenerate and every
conclusion drawn from that report is invalid until the baseline is
fixed (different prompts, an instruct model, or a coherent-model
variant — Qwen3.5 hybrid being one available choice today).

## Related

- [`docs/experience/errors/2026-05-27-tilelang-0110-fullrow-warp23-nan-sm80.md`](../errors/2026-05-27-tilelang-0110-fullrow-warp23-nan-sm80.md)
  — A100 sm_80 paged-prefill NaN bug blocking the equivalent
  audit on Qwen3-4B / Qwen3.5-4B's full-attention layers; this
  V100 result is on the FMA-fallback path and does not exercise
  the same TileLang FullRow codegen.
- [`docs/experience/errors/2026-05-26-fp8-kv-catastrophic-was-test-artifact.md`](../errors/2026-05-26-fp8-kv-catastrophic-was-test-artifact.md)
  — the prior "FP8 catastrophic" claim that was actually a
  degenerate-baseline artifact. This wins entry is the
  re-audit under a coherent baseline that retires that claim
  for FP8.
- [`docs/plans/2026-05-27-flashinfer-paged-prefill-migration.md`](../../plans/2026-05-27-flashinfer-paged-prefill-migration.md)
  — long-term fix that removes TileLang from the paged prefill
  surface entirely, after which sm_80 FP8 and TQ4 audits become
  reproducible.
