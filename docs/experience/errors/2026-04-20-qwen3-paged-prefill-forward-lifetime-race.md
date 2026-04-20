# Qwen3 paged-prefill chunking crashed because forward-owned metadata died too early

## Context

While starting Phase 1.1 of the overnight SGLang-alignment plan, `Qwen3-4B`
with paged prefill regressed at `--chunked-prefill-size 2048`:

- server: `target/release/infer --model-path infer/models/Qwen3-4B --num-slots 10 --max-seq-len 5120 --mem-fraction-static 0.88 --chunked-prefill-size 2048 --max-prefill-tokens 16384`
- workload: `scripts/bench_guidellm.sh ... --profile synchronous --data prompt_tokens=4096,output_tokens=256`
- observed prompt length on the wire: `4097` tokens (`2048 + 2048 + 1`)

Without extra fencing, request 1 consistently died on the second paged-prefill
chunk and poisoned the CUDA context:

```text
Request 1: prefill chunk 4096/4097 tokens
thread '<unnamed>' panicked at infer/src/ops/norm.rs:231:14:
rms_norm_batched_cuda failed: DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, ...)
```

## Root Cause

The regression was not a math bug in `chunked_prefill_size=2048`. It was a
forward-lifetime bug in the paged-prefill path.

`Qwen3Model::process_all_layers_batch_paged()` creates three classes of objects
that are only valid for the duration of the forward:

- the shared `BatchPrefillPagedPlan` workspace (`page_locked_workspace`,
  `int_workspace`, `plan_info`)
- the per-forward device page table (`slot_page_indices`)
- the per-forward device indptr buffers held by `PagedPrefillForward`

The function enqueued all paged-prefill kernels and returned immediately,
dropping the per-forward `CudaSlice`s and unlocking the shared plan before the
compute stream had necessarily consumed them. At `chunked_prefill_size=4096`
the bug often stayed hidden because many prompts fit in one big chunk. At
`2048`, the same request immediately schedules a second paged-prefill chunk, so
the next `PrefillPlan` reuses the same host/device workspace while the prior
chunk's kernels are still reading it.

The decisive diagnostic was `CUDA_LAUNCH_BLOCKING=1`: with the same
`4097 -> 2048 + 2048 + 1` workload, two consecutive requests completed their
second chunk cleanly. Making launches synchronous removed the crash, which
isolates the bug to metadata/workspace lifetime across asynchronous stream
execution.

## Fix

Turn the paged-prefill forward itself into the synchronization boundary.

- `infer/src/model/qwen3/prefill.rs`
  - after the per-layer paged-prefill loop, call `self.ctx.sync()?` before
    returning so the shared plan workspace and per-forward device metadata stay
    alive until the compute stream drains
- `infer/src/model/qwen35/prefill.rs`
  - apply the same contract to the latent HD256 paged-prefill path so it does
    not reintroduce the same bug when re-enabled

This is intentionally a safety fix, not the final throughput shape. The next
phase can replace the hard stream sync with a more explicit ownership/fencing
scheme if needed, but the correctness contract is now unambiguous.

## Validation

Rebuilt with:

```bash
cargo build --release -p infer --bin infer
```

Then reran the same local reproducer **without** `CUDA_LAUNCH_BLOCKING=1`:

```bash
RUST_LOG=info target/release/infer \
  --model-path infer/models/Qwen3-4B \
  --port 8000 \
  --num-slots 10 \
  --max-seq-len 5120 \
  --mem-fraction-static 0.88 \
  --chunked-prefill-size 2048 \
  --max-prefill-tokens 16384

scripts/bench_guidellm.sh p11-sync-fixed \
  --target http://127.0.0.1:8000 \
  --model Qwen3-4B \
  --processor infer/models/Qwen3-4B \
  --profile synchronous \
  --max-seconds 12
```

Result:

- 2 completed requests
- both requests reached `prefill chunk 4096/4097 tokens`
- no illegal-address panic
- headline metrics: `TTFT p50 729.5 ms`, `TTFT p99 735.3 ms`,
  `ITL p50 35.31 ms`, `ITL p99 35.32 ms`, `out tok/s 27.31`

Artifacts: `bench-output/2026-04-20-p11-sync-fixed/`

## Rule

If a CUDA forward owns host-pinned planning buffers or temporary device
metadata used by asynchronously launched kernels, its lifetime must extend to
stream completion. "Enqueued" is not "finished". Either:

- keep ownership until `ctx.sync()`, or
- introduce an explicit event/fence scheme that proves the next reuse happens
  after the prior kernels have consumed the buffers.
