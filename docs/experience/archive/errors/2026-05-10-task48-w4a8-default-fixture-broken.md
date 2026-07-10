# Task #48 — W4A8 Accuracy Gate Used Broken Default Fixture

## Context

`greedy_consistency::test_w4a8_vs_bf16_token_diff` was dispatched as a
regression bisect after the default run reported an 84.4% token diff against
BF16:

```text
BF16: " Paris. The capital of Germany is Berlin. ..."
W4A8: " Paris. The capital of France is Paris. ..."
W4A8 vs BF16: matched first 5/32 tokens, diff 84.4%
```

The initial candidate commits were `09ae5a5`, `c44788f`, and `35fc3cf`.

## Root Cause

Behavioral A/B showed the failure predates the candidate set:

```text
09ae5a5^ (43bda9c): matched first 5/32 tokens, diff 84.4%
```

The test defaulted to `infer/models/Qwen3-4B-W4A8-marlin`, the old naive W4A8
checkpoint. That fixture is known-broken. Two calibrated variants were tested:

```text
Qwen3-4B-GPTQ-W4A8-marlin: "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
Qwen3-4B-GPTQ-W4A8-marlin: matched first 0/32 tokens, diff 100.0%

Qwen3-4B-GPTQ-W4A8-zpfix: matched first 32/32 tokens, diff 0.0%
```

The earlier note that `Qwen3-4B-GPTQ-W4A8-marlin` was the correct calibrated
fixture was incomplete; the qzeros-fixed checkpoint is the usable one.

## Fix

- Changed W4A8 defaults in `infer/tests/greedy_consistency.rs` and
  `infer/tests/e2e.rs` to `Qwen3-4B-GPTQ-W4A8-zpfix`.
- Tightened `test_w4a8_vs_bf16_token_diff` from the temporary 25% threshold
  back to the documented 1% default-on gate.
- Kept `INFER_TEST_W4A8_MODEL_PATH` override support for explicit fixture
  testing.

## Rule

When a quantized-model test fails, bisect the fixture before bisecting kernels.
Known-broken local checkpoints must not be test defaults. Accuracy gates should
encode the real ship threshold, not a lenient investigation threshold.
