# First-Batch Cold-Start Scheduler Mode — guidellm trace, CUDA L4, 2026-04-30

## Goal

- Reduce first-window TTFT for bursty c=16 long-prompt traffic by preventing decode from interleaving with the initial prefill drain when the server has just become ready.

## Hypothesis

- The Qwen3.5 first window is hurt by mixed scheduling: early completed requests start decode while sibling 4096-token prompts are still queued for prefill, so p50 TTFT waits on a long prefill/decode interleave instead of a compact prefill drain.

## Profile Evidence

Existing HEAD trace samples before this change:

| trace | first active samples | mixed | prefill-only | decode-only | prefill share | decode share |
|---|---:|---:|---:|---:|---:|---:|
| `bench-output/2026-04-29-headline-arle-qwen3-fp8/service_stats_trace.jsonl` | 20 | 5 | 8 | 7 | 91.9% | 8.1% |
| `bench-output/2026-04-29-headline-arle-qwen3-bf16/service_stats_trace.jsonl` | 20 | 0 | 12 | 8 | 88.1% | 11.8% |
| `bench-output/2026-04-29-headline-arle-qwen35-fp8/service_stats_trace.jsonl` | 20 | 9 | 5 | 6 | 30.8% | 70.1% |
| `bench-output/2026-04-29-arle-qwen35-bf16kv-planfix/service_stats_trace.jsonl` | 20 | 7 | 6 | 7 | 58.9% | 40.7% |

Qwen3.5 shows the bad pattern directly: with `prefill_queue=15`, the trace already has `decode_rows=1 prefill_rows=1`, and later decode phase grows while prefill rows remain queued.

## Reference Check

- SGLang `scheduler.py` prefers `get_new_batch_prefill()` before decode. Mixed chunked prefill is gated by `enable_mixed_chunk` and requires an existing running decode batch.
- This ARLE change keeps existing mixed scheduling after cold-start; it only adds a first-batch drain gate for the no-active-decode startup state.

## Change

- Add scheduler `FirstBatchMode` state.
- Enter when a tick has prefill candidates and no runnable decode rows.
- While active, schedule prefill-only batches even after early requests become decode-ready.
- Exit when no prefill candidates remain, logging duration, total prefill rows/tokens, decode-ready count, and queue state.

## Verification

- `cargo fmt --all`
- `CUDA_HOME=/usr/local/cuda CARGO_TARGET_DIR=/tmp/arle-target CARGO_HOME=/tmp/arle-cargo-home ZIG=/tmp/zig-0.15.2/zig-x86_64-linux-0.15.2/zig cargo check -p infer --release --no-default-features --features cuda`
- Attempted targeted unit test with `cargo test -p infer --lib --release --no-default-features --features cuda first_batch_mode_records_prefill_work_until_finish`; interrupted before execution because release LTO stayed in test binary link for multiple minutes. The test remains in-tree for the next full test pass.

## Results

Status: implementation verified by compile only. Full c=16 / 4096 / 256 / 120s guidellm rerun is intentionally pending because the same worktree currently contains uncommitted A-track fp8 KV numerical-drift experiments in CUDA/model files. Running bench before that tranche is resolved would contaminate TTFT numbers.

## Learnings

- First-window profiling must distinguish steady mixed scheduling from cold-start admission. Qwen3.5 needed a startup drain gate; Qwen3 bf16 did not show the same interleave.

## Artefacts

- Before traces:
  - `bench-output/2026-04-29-headline-arle-qwen3-fp8/service_stats_trace.jsonl`
  - `bench-output/2026-04-29-headline-arle-qwen3-bf16/service_stats_trace.jsonl`
  - `bench-output/2026-04-29-headline-arle-qwen35-fp8/service_stats_trace.jsonl`
  - `bench-output/2026-04-29-arle-qwen35-bf16kv-planfix/service_stats_trace.jsonl`

## Follow-Up Bench

- Rerun the four ARLE headline rows after the fp8 KV diagnostic tranche is either fixed or reverted from the worktree:
  - Qwen3-4B fp8/bf16
  - Qwen3.5-4B fp8/bf16
