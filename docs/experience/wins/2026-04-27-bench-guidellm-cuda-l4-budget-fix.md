# cuda-l4 budget fix — c=16 / 4096-in / 256-out: 88 → 122 tok/s (+39%)

## Context

Diagnostic chain:
- 2026-04-27 Phase 2 KV swap-out attempt reverted entirely after vLLM
  V1 / SGLang precedent research showed swap is structurally dead. See
  [`errors/2026-04-27-kv-swap-out-deleted-following-vllm-v1-sglang.md`](../errors/2026-04-27-kv-swap-out-deleted-following-vllm-v1-sglang.md).
- The c=16 long-prompt `peak_active=6` cap remained — that's a
  budget/scheduling/kernel-quality problem, not a preemption-strategy
  problem. SGLang reaches higher concurrency on the same workload
  WITHOUT swap (139 tok/s baseline, our 88 tok/s pre-fix).
- Direct code-trace diagnostic identified two waste sites:
  - `MixedBatchBuffers::logits` allocated `vocab × max_total_tokens`
    when only `vocab × max_batch_size` is ever consumed. ~1.25 GB
    waste at c=16 / 4096-in.
  - `estimated_request_target_tokens` reserved `prompt + max_tokens`
    upfront per admission instead of admitting on prefill cost only
    (SGLang-style). For 4096-prompt / 256-max the savings are modest
    (~5%/slot), but it's the right policy.

## What Worked

Two surgical fixes (commit `af042d1c`):

**Fix A — right-size mixed-batch logits**: in-place compaction of
`mixed.normed` with raw `cuMemcpyDtoDAsync_v2` (forward-order copies
are aliasing-safe; `src_i ≥ b + i = dst_i`), then run output projection
over only kept rows. `mixed.logits` shrinks `vocab × max_tokens` →
`vocab × max_batch_size`. Estimator updated to match.

**Fix B — admit on prompt-only**: `estimated_request_target_tokens`
no longer adds `max_tokens`. Decode pages allocate lazily inside
`step_decode_launch::allocate_decode_tokens`. Mid-step OOM is handled
by `retract_decode_to_fit` (already wired from both decode + mixed
launch paths). `clipped_max_new_tokens_estimate` stays for the
running-decode reservation path in execution.rs.

Two further fixes were attempted, codex review caught both as P1, and
both were reverted before commit:

- Sizing `prefill_activation` for `chunked_prefill_size` (4096) instead
  of `prefill_budget_tokens` (16384): wrong — multi-row mixed prefill
  packs `Σ chunks` per step, capped at `max_prefill_tokens`. Codex P1.
- Dropping `MIXED_FLOAT_WORKSPACE_BYTES` from HD256 (640 MB) to HD128
  default (256 MB) for HD128 models: wrong — FlashInfer's
  `batch_prefill_tmp_v` actually needs ~283 MB at c=16 / 4112 tokens.
  Server crashed with `Buffer overflow ... allocator.h:49`.

## Bench

`scripts/bench_guidellm.sh cuda-l4-budget-fix --fast` (profile=concurrent
rate=16, data=4096-in/256-out, max-seconds=30, random-seed=20260416).

Server config: `--num-slots 16 --max-seq-len 4608 --mem-fraction-static
0.94 --chunked-prefill-size 4096`. L4 / Qwen3-4B BF16. Pinned to
HEAD `af042d1c`.

| metric                        | pre-fix (`93e3798d`) | post-fix (`af042d1c`) | Δ      |
|-------------------------------|---------------------:|----------------------:|-------:|
| TokenKVPool budget            | 4.0 GB               | 5.2 GB                | +1.2GB |
| TTFT p50 (ms)                 | 4065                 | 5429                  | +34%   |
| TTFT p99 (ms)                 | 20675                | 24910                 | +20%   |
| ITL p50 (ms)                  | 48.6                 | 54.6                  | +12%   |
| out tok/s                     | 88.0                 | **122.1**             | **+39%** |
| req/s                         | 0.20                 | 0.27                  | +33%   |
| peak_active                   | 6                    | **8**                 | **+33%** |
| peak_kv_util                  | 96.2%                | 98.2%                 |        |
| samples (30s)                 | 70                   | 84                    | +20%   |
| failed requests               | 0                    | 0                     |        |

ITL +12% / TTFT +34% are the cost of running 33% more concurrent
decoders on the same memory-bandwidth-bound L4 — total throughput
dominates these per-request latencies for any mixed-load workload.
The bench's c=16 admission burst sees longer TTFT because each new
admission queues behind more prefilling slots, but the bench's
samples-per-30s also rises 20%.

vs SGLang head-to-head (2026-04-26, 139 tok/s @ c=16): ratio improves
0.474 → 0.879. Closing the gap remains kernel/admission/batching
work, not preemption-strategy work.

## Rule

**Trace allocation order before re-architecting.** The original
diagnostic blamed admission policy ("we need swap"). The real culprit
was that `runtime_workspace` (4.9 GB on 23 GB box) was eating the
KV pool. Two estimator/allocator mismatches accounted for ~1.7 GB of
that. Future "we can't fit more concurrent requests" investigations
should start from `nvidia-smi` snapshots taken at well-defined points
in the boot sequence, not from preemption-strategy proposals.

## Cross-references

- Code: commit `af042d1c` (3 files, +143/-27)
- Pre-fix baseline bench: `bench-output/2026-04-27-cuda-l4-phase2-recompute/`
- Post-fix bench: `bench-output/2026-04-27-cuda-l4-fix2only/`
- Bench env drift note (re: 88 vs SGLang 139 baseline):
  [`memory/project_bench_env_drift_2026-04-20.md`](../../../memory/MEMORY.md)
- Phase 2 closure (sets context for why budget fixes were the next move):
  [`errors/2026-04-27-kv-swap-out-deleted-following-vllm-v1-sglang.md`](../errors/2026-04-27-kv-swap-out-deleted-following-vllm-v1-sglang.md)
