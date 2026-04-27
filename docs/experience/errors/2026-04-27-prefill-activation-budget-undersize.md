# Prefill activation budget — under-reservation attempt caught by codex review

## Context

The 2026-04-26 SGLang head-to-head
([`vs-sglang-c1-c16`](../wins/2026-04-26-bench-guidellm-cuda-l4-vs-sglang-c1-c16.md))
showed `infer` HEAD's KV pool sized at **4.0 GB** vs SGLang's **13.3 GB**
on Qwen3-4B / L4 24 GB at the same `--mem-fraction-static 0.94`. The
server log confirmed:

```
TokenKVPool budget: 4.0 GB (contiguous=0.0 GB, runtime_workspace=4.9 GB, fraction=94%)
TokenKVPool BF16:   data=110.7 MB/layer
```

`runtime_workspace=4.9 GB` was eating most of the budget that should
have gone to KV. With only ~28K KV tokens, the c=16 trace logged
`peak_active=5, peak_waiting=11, peak_kv_util=98%` — 11 of 16 client
requests sat queued because the pool physically couldn't hold them.

## Root Cause (proposed)

The Qwen3 estimator
(`infer/src/model/qwen3/forward.rs::scheduler_runtime_workspace_bytes`)
reserves **prefill_activation = prefill_activation_dims × prefill_budget_tokens × 2**
where `prefill_budget_tokens = max_prefill_tokens = 16384`.

Reading the runtime, `PrefillBuffers` is allocated lazily per chunk by
`prefill_buffers(seq_len)?` and freed when the prefill batch finishes.
Async prefill keeps at most ONE `PendingPrefill` in flight
(`SchedulerRuntime::pending_prefill: Option<_>`). So the *appearance*
was that peak transient activation = one chunk = 4096 tokens, and we
were over-reserving 12288 tokens × 49664 dims × 2 B = **1.2 GB**.

Hypothesis: re-size `prefill_activation` against
`chunked_prefill_size = 4096` instead of `max_prefill_tokens = 16384`,
recover 1.2 GB into the KV pool, close 16-30 pp of the SGLang
throughput gap.

## What I tried

Three-line patch:

1. Add `prefill_chunk_tokens: usize` to `SchedulerRuntimeWorkspaceBudget`.
2. Pass `config.chunked_prefill_size` from `core.rs` into the budget.
3. In Qwen3's estimator, use `prefill_chunk_tokens` instead of
   `prefill_budget_tokens` when computing `prefill_activation`.

The pool grew **4.0 → 5.2 GB (+30 %)**, KV tokens 28K → 36K. Bench at
matched flags showed:

| conc | TTFT p50 (was → now) | tok/s (was → now) | Δ tok/s vs sglang (was → now) |
|---|---|---|---|
|  4 | 2354 → 2987 ms (+27 %) | 53.31 → **72.75** | −28.8 % → **−2.8 %** (parity) |
|  8 | 3838 → 5207 ms (+36 %) | 66.94 → **98.67** | −38.8 % → **−9.8 %** |
| 16 | 16357 → 21202 ms (+30 %) | 65.92 → **89.0** | −52.6 % → **−36.0 %** |

Throughput jumped 35-47 % at high concurrency. TTFT regressed because
the bigger pool admits more concurrent → more queue contention.

## Codex caught the bug

`codex review --uncommitted` flagged:

> [P1] **Reserve activation space for the whole prefill batch** —
> `infer/src/model/qwen3/forward.rs:199-201`. When a scheduler step
> selects more than one prefill row, which is allowed by the defaults
> (`prefill_max_requests = None`, `chunked_prefill_size = 4096`,
> `max_prefill_tokens = 16384`), Qwen3 packs all selected rows and
> allocates `PrefillBuffers` with `packed_tokens.len()`, not with a
> single row's chunk length. Sizing this reservation from only
> `prefill_chunk_tokens` therefore underestimates peak activation
> memory for multi-row prefill batches and gives that memory to the
> KV pool, so concurrent long-prefill runs can still OOM mid-prefill
> even though pool sizing claimed the workspace was reserved.

Verifying in `infer/src/model/qwen3/prefill.rs:330-336`:

```rust
let total_tokens = requests.iter().map(|req| req.tokens.len()).sum();
let mut packed_tokens = Vec::with_capacity(total_tokens);
for req in requests {
    packed_tokens.extend_from_slice(req.tokens);
}
…
let mut bufs = self.prefill_buffers(packed_tokens.len())?;
```

`packed_tokens.len()` IS the across-request sum, not a single chunk.
With `prefill_max_requests = None`, the scheduler can pack up to 4
requests × 4096 tokens = 16384 tokens into one `PrefillBuffers` =
1.6 GB transient allocation. My patch only reserved 0.4 GB. Under a
burst of concurrent long prefills the standalone path would OOM.

The throughput numbers above are real but were taken on a workload
(c=1..16 mixed-batch steady-state) where the standalone path's
multi-row packing rarely triggers — almost every prefill rode the
mixed-batch path through `MixedBatchBuffers` instead. That doesn't
make the reservation correct; it makes the bug latent.

## Fix

**Reverted the patch.** Keeping the original `prefill_budget_tokens`
based estimate is the safe answer. Recovering the 1.2 GB requires
addressing the actual driver instead:

1. **Bound the runtime packing** — cap `packed_tokens.len()` at
   `chunked_prefill_size` in `step_prefill_batch_async`, and drop
   excess requests back to the queue. Then the activation budget can
   safely shrink to `chunked_prefill_size`. Risk: degrades multi-row
   packed prefill throughput on the standalone path.
2. **Default `prefill_max_requests = 1` when mixed-batch is enabled**
   — same outcome as (1) but as a config default. Less invasive.
3. **Allocate `PrefillBuffers` with `chunked_prefill_size` and reuse
   it** — make the buffer persistent and capped. The current
   per-call `prefill_buffers(seq_len)?` is the source of the
   variability.

(2) is the cleanest single knob. (3) requires a kernel-level audit to
confirm `seq_len` is only ever read off the buffer's actual fill, not
its capacity.

## Rule

**Estimator over-reservations are NOT free.** A "looks safe to leave
the budget conservative" attitude leaves a 1.2 GB hole on a 24 GB
GPU — directly translates to a 30+ % throughput regression vs SGLang.

When sizing a workspace estimate, trace the actual runtime allocation
end-to-end and prove the *worst case* matches the estimate. Don't size
to an obvious upper bound (`max_prefill_tokens`) without confirming
the runtime ever actually allocates that much in one buffer; conversely
don't shrink to an "obvious" minimum (`chunked_prefill_size`) without
confirming the runtime never allocates more.

The verification path: read the call-site (`prefill_buffers(seq_len)?`),
trace `seq_len` upstream (`packed_tokens.len()`), trace the upstream
(`requests` list × `req.tokens.len()`), and find the scheduler invariant
that bounds the sum. In our case the bound is across-request
(`max_prefill_tokens`), not per-request (`chunked_prefill_size`).

## Cross-references

- SGLang head-to-head that exposed the gap:
  [`wins/2026-04-26-bench-guidellm-cuda-l4-vs-sglang-c1-c16.md`](../wins/2026-04-26-bench-guidellm-cuda-l4-vs-sglang-c1-c16.md)
- Verification report:
  [`reviews/2026-04-26-l4-scheduler-and-tilelang-verification.md`](../../reviews/2026-04-26-l4-scheduler-and-tilelang-verification.md)
- The 2026-04-26 commit `df2d3e8e fix(scheduler): align mixed
  workspace budget` is the symmetric fix on the *mixed* side that
  caught a similar over-reservation. The pattern is real and worth
  watching for elsewhere in the estimator.
