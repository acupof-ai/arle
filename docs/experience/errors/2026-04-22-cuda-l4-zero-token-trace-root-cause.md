# CUDA L4 zero-token requests at `c8/c16` are dominated by prefill OOM, with a small tail of benchmark cutoff aborts

## Context

- Added request-level scheduler trace hooks for:
  - `scheduler.admit`
  - `scheduler.launch_prefill` / `scheduler.launch_decode` / `scheduler.launch_mixed`
  - `scheduler.decode_readback`
  - `request.finish` / `request.zero_token_finish`
  - terminal causes such as `prefill_batch_failed`
- Rebuilt `infer` and reran the same L4/Qwen3-4B serving envelope used by the invalid `c8/c16` benches:
  - `--num-slots 16`
  - `--max-seq-len 4608`
  - `--mem-fraction-static 0.94`
  - `--chunked-prefill-size 4096`
  - `--max-prefill-tokens 16384`
  - `--enable-mixed-chunk true`
- Diagnostic artefacts:
  - `c8` benchmark: `bench-output/2026-04-22-infer-qwen3-4b-l4-c8-trace-local-zero-token-run2`
  - `c8` server log: `bench-output/2026-04-22-infer-qwen3-4b-l4-c8-trace-local-zero-token/infer.log`
  - `c16` benchmark: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-trace-local-zero-token-run2`
  - `c16` server log: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-trace-local-zero-token/infer.log`
- The JSON trace spine still did **not** persist files under `--trace-output-path` during local smoke; the root cause below was derived from the new scheduler-side cause hooks plus server logs.

## Root Cause

Two paths produce the bad high-concurrency runs:

1. **Dominant path: first-chunk prefill OOM.**
   - `c8`: `248` zero-token completions, `245` of them have a directly preceding `prefill batch failed` OOM.
   - `c16`: `1445` zero-token completions, `1437` of them have a directly preceding `prefill batch failed` OOM.
   - The failure text is consistently:
     - `Alloc failed: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")`
     - or `FlashInfer float_workspace alloc failed: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")`
   - This means the scheduler is still admitting more prompt-prefill work than the L4 can hold under the `4096-in / 256-out` workload and current envelope.

2. **Minor tail path: benchmark cutoff aborts after chunk 1, before first output token.**
   - `c8`: requests `246/247/248`
   - `c16`: requests `1438..1445`
   - These requests show:
     - admission
     - `prefix MISS`
     - `chunked prefill starting (4097 effective tokens, chunk_size=4096)`
     - `prefill chunk 4096/4097 tokens` or equivalent progress
     - then `done: 0 tokens`
   - They have **no** preceding OOM log, and they occur at the very end of the benchmark window, which is consistent with client-side cutoff / benchmark window expiry while the request is still between the last prefill chunk and the first sampled token.

So the `c8/c16` invalidity is not primarily a decode/readback metrics bug. The dominant failure is **prefill admission overrunning GPU headroom**; benchmark-window cutoff only explains the small tail of remaining zero-token requests.

## Fix

- Treat the serving regression as an **admission / memory-envelope bug**, not a `guidellm` rendering issue.
- Tighten high-concurrency first-chunk admission on L4 so the scheduler cannot launch a prefill batch that immediately OOMs.
- Keep terminal-cause separation between:
  - internal failure: `prefill_batch_failed`
  - external cutoff: client / benchmark window abort before first token
- Follow up separately on trace persistence: the request-level hooks are in the runtime, but the current file reporter path still needs one more fix before `--trace-output-path` produces usable JSON artefacts.

## Rule

- When `zero-token` completions line up almost 1:1 with `prefill batch failed` OOMs, classify the run as a scheduler memory-envelope regression.
- Do not lump end-of-window client cutoffs into the same bucket; keep the internal OOM path and the benchmark cutoff tail separate.
