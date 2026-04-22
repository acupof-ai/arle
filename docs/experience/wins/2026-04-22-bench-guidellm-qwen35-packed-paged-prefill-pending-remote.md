# Qwen3.5 packed paged prefill — guidellm sweep, cuda, 2026-04-22

**Status:** `pending-remote`  
**Plan anchor:** [`docs/plans/2026-04-22-sglang-gap-closure-execution.md`](../../plans/2026-04-22-sglang-gap-closure-execution.md)  
**Change scope:** `infer/src/model/qwen35/forward.rs`, `infer/src/model/qwen35/prefill.rs`, `infer/src/model/qwen35/prefill_buffers.rs`, `infer/src/model/qwen35/weights.rs`, `infer/src/ops.rs`

## Goal

- Regression / validation: record the required CUDA benchmark stub for the
  Qwen3.5 model-side change that replaces the scheduler-visible paged-prefill
  batch override's per-request replay with one real packed multi-request
  paged-prefill forward.

## Hypothesis

- High-concurrency TTFT should improve over the prior batch-override baseline
  because `Qwen3.5` now packs the full paged-prefill forward once per batch
  instead of looping over requests inside `forward_prefill_batch_with_pool()`.
- The biggest gain should appear on mixed-length and multi-request loads where
  the hybrid linear-attention layers previously serialized conv1d + GDR replay
  per request after scheduler-side admission had already grouped the work.

## Command

```bash
scripts/bench_guidellm.sh cuda-qwen35-packed-paged-prefill \
  --target http://<remote-host>:8000 \
  --model Qwen/Qwen3.5-4B \
  --processor models/Qwen3.5-4B \
  --trace-interval-ms 1000
```

Invoked via: `scripts/bench_guidellm.sh cuda-qwen35-packed-paged-prefill [--target URL] [--model NAME] [--processor PATH] [--trace-interval-ms N]`

## Environment

- **Backend:** `cuda`
- **Model:** `Qwen/Qwen3.5-4B`
- **Hardware:** `pending-remote`
- **Commit:** `pending-remote`
- **Feature set:** `cargo build -p infer --release --no-default-features --features cuda`
- **Non-default flags / env vars:** `pending-remote`
- **Server launch:** `pending-remote`

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh cuda-qwen35-packed-paged-prefill`

## Results — sweep headline table

Pending remote CUDA run on a benchmark host.

## Problems

- This machine is not a usable CUDA bench host, so the canonical `guidellm`
  sweep could not run in this turn.
- Local `cargo test -p infer --release --no-default-features --features cuda,no-cuda --lib`
  still fails to link on macOS because the CUDA symbols are unavailable in this
  environment.
- Local `cargo clippy -p infer --release --no-default-features --features cuda,no-cuda -- -D warnings`
  is still blocked by the existing unrelated `manual_is_multiple_of` lint in
  `crates/cuda-kernels/src/paged_kv.rs`.

## Learnings

- The scheduler-visible batch override was only a partial win; Qwen3.5 still
  needed a real packed model-side paged-prefill path to avoid re-serializing
  hybrid-layer prefill after shared paged-KV admission.
- Packed recurrent launch surfaces are enough for Qwen3.5's hybrid layers when
  the model supplies canonical packed metadata and per-request recurrent-state
  pointer arrays.

## Δ vs baseline

- **Baseline:** [Qwen3.5 paged-prefill batch override — guidellm sweep, cuda, 2026-04-22](./2026-04-22-bench-guidellm-qwen35-paged-prefill-batch-override.md)
- Delta table: `pending-remote`

## Artefacts

- Raw: `pending-remote`
- CSV: `pending-remote`
- HTML: `pending-remote`
- Service trace (before): `pending-remote`
- Service trace (during): `pending-remote`
- Service trace (after): `pending-remote`
- Service trace (summary): `pending-remote`

## Notes

- What changed in the code since baseline:
  - `forward_prefill_batch_with_pool()` now dispatches one packed paged-prefill
    model path instead of replaying per-request paged-prefill forwards
  - paged-prefill metadata is now batch-shaped (`qo_indptr`, `kv_indptr`,
    `kv_last_page_len`, `start_pos`) and reused across the whole packed
    forward
  - hybrid linear-attention layers now call the packed conv1d and packed GDR
    launch surfaces through one canonical batch path
  - paged-prefill logits for the packed batch path are written into
    `state.base.prefill_logits` per request
- Suspected cause of any regression: n/a until the remote CUDA sweep runs
- Follow-ups:
  - run the canonical remote CUDA sweep and fill in the delta table versus the
    batch-override baseline
  - if the delta is smaller than expected, inspect full-attention prep
    overhead and recurrent scratch sizing before touching scheduler policy
