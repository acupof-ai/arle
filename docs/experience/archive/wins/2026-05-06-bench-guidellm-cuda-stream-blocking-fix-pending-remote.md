# 2026-05-06 · `complete_stream` blocking fix — bench pending remote

## Goal
Verify that aligning the CUDA `RequestHandleInferenceEngine::complete_stream`
to the metal/cpu blocking contract does not regress sustained throughput
or TTFT. The fix only changes return-timing semantics (block until
finish-marker), not the kernel path.

## Hypothesis
Bench numbers should match the most recent CUDA Qwen3-4B baseline within
the spec's ±2% noise envelope; the fix runs on the same hot decode path
and only adds a blocking forwarder around the existing scheduler tx.

## Status
**pending-remote.** This workspace has neither `guidellm` (the bench
wrapper extra in `pyproject.toml`'s `[bench]` group) installed nor a
prior local baseline checked in for this host (RTX 4070 Ti SUPER /
sm_89). The fix runs locally and passes the e2e Phase 1+2 path; Phase 3
still trips on the open
[`2026-04-13-batched-decode-high-concurrency.md`](../errors/2026-04-13-batched-decode-high-concurrency.md)
divergence which is a separate kernel-level issue, not introduced here.

When a CUDA host with guidellm is available, run
`scripts/bench_guidellm.sh stream-block-fix-2026-05-06` against
`docs/experience/wins/2026-05-05-bench-tilelang-phase0-pending-remote.md`'s
expected baseline window (or the latest CUDA Qwen3-4B win on file) and
fold the Δ% row into this entry.

## Params (when run remote)
- Backend: CUDA `--features cuda`
- Model: Qwen/Qwen3-4B (BF16)
- num_slots: 4
- Profile: guidellm sweep (canonical params per
  `docs/plans/guidellm-integration.md` §3)

## Fix
`infer/src/server_engine/request_handle_engine.rs`:
`complete_stream` now forwards through an internal channel and
`blocking_recv`s until the finish-marker delta lands, mirroring the
metal/cpu contract. See
[`2026-05-06-cuda-complete-stream-async-violates-contract.md`](../errors/2026-05-06-cuda-complete-stream-async-violates-contract.md)
for the original race write-up.

## Why no bench number yet
Local environment lacks `guidellm` (would need to extend pip extras
install in this venv). The change is a contract-alignment fix on the
return path — it does not touch kernels, scheduling decisions, or
buffer layouts, so a perf delta is not expected. Recorded as
pending-remote per the §Benchmarks rule rather than silently skipped.
