# 2026-04-09 · Scheduler Paged Decode Greedy Output Divergence

## Context

BF16 model through the scheduler (paged FlashInfer batch decode) produces different
greedy output compared to the single-request engine (Triton decode attention).

Example: "The capital of France is" →
- Single-request engine: "Paris. Paris is the capital..." (matches E2E baselines)
- Scheduler path: "Paris Agreement, adopted in 2015..." (coherent but different)

## Root Cause

Two independent issues documented in `forward.rs:351-356`:

1. **Numerical divergence**: Triton and FlashInfer attention kernels produce different
   BF16 results (rounding, accumulation strategies). For greedy sampling, even 1-2 ULP
   differences in logits change argmax → different token → cascading divergence.

2. **KV cache coherency**: If B=1 goes through contiguous path, new KV tokens are written
   only to contiguous cache, not the paged pool. When batch grows to B>1, stale pool data
   is read. (Mitigated by always using paged path when pool is active.)

## Impact

- All weight formats (BF16, W8, W4) are affected equally
- Output is coherent English but diverges from baselines
- Quality measured by PPL is equivalent (same model, same weights)
- Only affects greedy decode (temperature>0 sampling is stochastic anyway)

## Fix Options

1. **Regenerate E2E baselines** against scheduler path (simplest)
2. **Use FlashInfer for single-request** path too (architectural change)
3. **Accept divergence** as known behavior (greedy is deterministic per-path)

## Rule

When testing model quality, use the same inference path as production.
E2E baselines should match the scheduler path, not the single-request engine.
