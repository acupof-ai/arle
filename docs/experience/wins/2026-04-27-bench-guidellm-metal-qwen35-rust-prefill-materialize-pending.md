# Qwen3.5 Metal Rust prefill materialize guard — guidellm sweep, pending, 2026-04-27

## Goal

- Regression-check the Rust scalar prefill fallback after forcing periodic MLX graph materialization for long prompts.

## Hypothesis

- Periodic `eval` + cache clear every 32 prompt tokens should cap graph growth in fallback mode without changing token output semantics. It should not affect the normal C++ prefill path because the guard only runs when `cpp_model` is absent.

## Command

```bash
scripts/bench_guidellm.sh metal-qwen35-rust-prefill-materialize
```

Invoked via: pending dedicated fallback run.

## Environment

- **Backend:** metal
- **Model:** Qwen3.5-family configuration that intentionally disables or lacks the C++ model path
- **Hardware:** pending Apple Silicon bench host
- **Commit:** pending
- **Feature set:** `cargo build --release --no-default-features --features metal,no-cuda`
- **Non-default flags / env vars:** fallback-triggering model/config required
- **Server launch:** pending

## Canonical Params

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh metal-qwen35-rust-prefill-materialize`

## Results

- Status: `pending`
- Local verification planned with the regular Metal compile and unit-test gates.

## Problems

- This path is now a fallback-only path for Qwen3.5-family Metal. A representative fallback-triggering model/config needs to be selected before a meaningful guidellm comparison can be recorded.

## Learnings

- Rust fallback prefill should periodically force MLX graph realization instead of accumulating an unbounded per-token graph over long prompts.

## Delta vs Baseline

- **Baseline:** pending fallback baseline selection.
- **Delta table:** pending.

## Artefacts

- Raw: pending
- CSV: pending
- HTML: pending
- Service trace: pending

## Notes

- What changed in code since baseline: Rust scalar prefill now materializes logits, KV cache arrays, and recurrent state every 32 prompt tokens when `weights.cpp_model.is_none()`.
- Suspected cause of any regression: extra materialization cadence in short prompts or excessive cache churn.
- Follow-ups: run a dedicated fallback guidellm sweep once the fallback model/config is pinned.
