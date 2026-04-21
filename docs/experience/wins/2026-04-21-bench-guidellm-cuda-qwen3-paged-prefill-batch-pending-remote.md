# CUDA Qwen3 paged-prefill batch bench stub — pending remote

**Status:** `pending-remote`  
**Plan anchor:** [`docs/plans/paged-prefill-followups-2026-04-18.md`](../../plans/paged-prefill-followups-2026-04-18.md)  
**Change scope:** `infer/src/ops/attention.rs`, `infer/src/model/qwen3/{forward,prefill,weights}.rs`, `infer/src/model/qwen35/{prefill,weights}.rs`

## Goal

- **Type:** regression
- Record the required post-change CUDA regression bench for the Qwen3 batched paged-prefill forward path after replacing the batch-size-1 FlashInfer plan/run contract with packed varlen prefill metadata.

## Hypothesis

- Qwen3 paged prefill should preserve or slightly improve long-prompt TTFT/throughput under the canonical CUDA sweep because multiple queued prefills can now share one FlashInfer paged-prefill execution shape per forward instead of forcing the single-request contract in model-owned code.

## Command

```bash
scripts/bench_guidellm.sh cuda-qwen3-paged-prefill-batch
```

Canonical params are locked by [`docs/plans/guidellm-integration.md`](../../plans/guidellm-integration.md).

## Environment

- **Backend:** `cuda`
- **Model:** `Qwen/Qwen3-4B`
- **Hardware:** pending remote CUDA machine
- **Commit:** `45dc8ad` plus local uncommitted paged-prefill batch refactor
- **Feature set:** `cargo build -p infer --release --bin infer`
- **Non-default flags / env vars:** pending remote machine selection
- **Server launch:** `scripts/start_infer.sh Qwen/Qwen3-4B 8000`

## Results

- Pending remote execution.
- Local verification on this workstation reached only `ZIG=/tmp/zig-tool/zig-x86_64-linux-0.15.2/zig cargo check -p infer --no-default-features --features cuda,no-cuda`.

## Problems

- This workspace cannot run the required CUDA serving benchmark locally, so no trustworthy GuideLLM numbers were collected here.
- Qwen3.5 was updated only to the new packed paged-prefill metadata/plan contract; its scheduler-visible paged-prefill path remains disabled pending separate hybrid recurrent-state batching work and should be benchmarked only when that path is re-enabled.

## Learnings

- Runtime CUDA refactors that alter plan/workspace shape still need a dated benchmark stub when local hardware cannot execute the serving lane.
- Keeping the Qwen3 active path and the Qwen3.5 latent path on the same packed paged-prefill metadata contract reduces follow-up integration risk, but only Qwen3 is currently bench-relevant.

## Delta vs baseline

- **Baseline:** [2026-04-21-bench-guidellm-cuda-l4-c16-unified-single-plan.md](./2026-04-21-bench-guidellm-cuda-l4-c16-unified-single-plan.md)
- **Delta table:** pending remote execution

## Follow-up

- Run the canonical CUDA GuideLLM sweep on the next available NVIDIA host against `Qwen/Qwen3-4B`.
- If the scheduler lands multi-request prefill wiring on top of this refactor, record a second dated snapshot against the same baseline instead of editing this stub.
