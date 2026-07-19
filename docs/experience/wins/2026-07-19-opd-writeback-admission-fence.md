# OPD writeback admission fence — CUDA, 2026-07-19

> Status: pending-remote — local control-flow gates pass; multi-round H20 verification remains.

## Goal

Prevent requests from entering an OPD serve engine while writeback has released its KV pool, without hanging shutdown or admitting pre-quiesce backlog after resume.

## Hypothesis

Quiescing at submission intake, cancelling the pre-existing backlog, and reopening admission only after KV-pool restoration removes the round-2 late-admit crash without changing ordinary serving.

## Parameters

- Treatment: `EngineMode::Quiesced` at the infer-server intake boundary.
- Writeback guard: trajectories above `--max-update-seq` are dropped before behavior-logprob recomputation.
- Local gates: infer-core, train, arle CPU release-fast tests; CUDA/no-CUDA infer-api and infer-cuda checks; clippy and rustfmt.
- Remote gate: one multi-round Agent-OPD run with a submission queued at quiesce, one over-limit trajectory, KV release/reacquire, and controlled early shutdown.

## Environment

- Local host: Apple Silicon; CUDA code typechecked with `CUDARC_CUDA_VERSION=12080` and `cuda,no-cuda`.
- Remote target: 8× H20, Qwen3.6-27B-W4A16 student, exact source and binary receipts.

## Results

Local tests prove the device-neutral state transitions compile and existing suites remain green. No wall-clock result is claimed: this path changes OPD writeback control flow, and the binding acceptance case requires a real multi-round CUDA run.

Required H20 assertions:

1. `active_requests` reaches zero after quiesce without a 60-second stale-counter wait.
2. Requests already queued before quiesce are cancelled; requests arriving afterward remain deferred.
3. KV restoration failure leaves admission quiesced and returns an error.
4. Successful restoration resumes exactly once and the next rollout prefills normally.
5. Closing the frontend while quiesced joins the engine thread without a hang.
6. An over-limit trajectory never enters `capture_rollout_logprobs`.

Raw artifacts: pending remote run.

## Problems

The repository has no GitHub Actions runner labelled `gpu,cuda`; CUDA acceptance cannot run in hosted CI. The dedicated H20 run remains mandatory before changing this status to Shipped.

## Learnings

`pending-remote`. The correctness boundary is submission intake plus successful resource restoration, not scheduler admission alone. No performance claim is licensed until the multi-round H20 case completes.
