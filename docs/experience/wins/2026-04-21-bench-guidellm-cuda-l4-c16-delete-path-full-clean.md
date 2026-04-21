# guidellm sweep cuda-l4-c16-delete-path-full-clean — guidellm sweep, cuda-l4-c16-delete-path-full-clean, 2026-04-21

> Template for canonical guidellm bench wins. Copy this file when
> `scripts/bench_guidellm.sh` runs, fill the placeholders, commit. Never
> edit an existing wins entry — always create a new dated one and diff
> against the prior. `scripts/bench_throughput.py` is legacy only.
> Canonical params are locked in
> [`docs/plans/guidellm-integration.md`](../../plans/guidellm-integration.md) §3.

## Goal

- Regression-check a clean canonical c16 sweep after deleting the temporary mixed/prefill gate and one-chunk cap, to verify whether those prohibition-style heuristics were the source of the throughput drop.

## Hypothesis

- Removing the extra gate should return CUDA behavior to the earlier queued-chunk-reserve baseline, but may still remain below the stronger `headroom-retract` result if the real bottleneck is serialized long-prefill work.

## Command

```bash
scripts/bench_guidellm.sh <backend-label> \
  [--target http://localhost:8000] \
  [--model Qwen/Qwen3-4B] \
  [--processor models/Qwen3-4B]
```

Invoked via: `scripts/bench_guidellm.sh <backend-label> [--target URL] [--model NAME] [--processor PATH]`

Actual invocation:

```bash
./scripts/bench_guidellm.sh \
  cuda-l4-c16-delete-path-full-clean \
  --model Qwen/Qwen3-4B \
  --processor /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
```

## Environment

- **Backend:** cuda
- **Model:** Qwen/Qwen3-4B
- **Hardware:** NVIDIA L4, 23034 MiB VRAM, driver 580.82.07
- **Commit:** `40e392b` plus local uncommitted CUDA scheduler changes
- **Feature set:** `ZIG=/tmp/zig-tool/zig-x86_64-linux-0.15.2/zig cargo build --manifest-path infer/Cargo.toml --release --bin infer`
- **Non-default flags / env vars:** `GUIDELLM__MP_CONTEXT_TYPE`; server flags `--num-slots 16 --max-seq-len 4608 --mem-fraction-static 0.94 --chunked-prefill-size 4096 --max-prefill-tokens 16384 --enable-mixed-chunk true`
- **Server launch:** `target/release/infer --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c --port 8000 --num-slots 16 --max-seq-len 4608 --mem-fraction-static 0.94 --chunked-prefill-size 4096 --max-prefill-tokens 16384 --enable-mixed-chunk true`

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh <backend-label>`

## Results — sweep headline table

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| sync | 754.1 | 771 | 36.26 | 36.39 | 25.9 | 0.083 |
| throughput | 15230.5 | 38790.5 | 101.46 | 138.54 | 69.78 | 0.267 |
| 0.10624999999999998r/s | 855.5 | 868.5 | 39.56 | 39.68 | 26.79 | 0.1 |
| 0.12916666666666665r/s | 848.5 | 898.7 | 40.56 | 40.69 | 31.47 | 0.117 |
| 0.1520833333333333r/s | 849.8 | 865.9 | 41.31 | 43.61 | 36.1 | 0.133 |
| 0.175r/s | 859.2 | 1415.4 | 44.77 | 47.16 | 40.26 | 0.15 |
| 0.19791666666666663r/s | 848.7 | 860 | 45.6 | 48.32 | 44.6 | 0.167 |
| 0.22083333333333327r/s | 858.5 | 878.6 | 46.83 | 49.05 | 49.64 | 0.183 |
| 0.24374999999999997r/s | 857.7 | 883.3 | 52.23 | 52.92 | 52.24 | 0.2 |
| 0.2666666666666666r/s | 868.7 | 893.8 | 51.66 | 53.88 | 57.12 | 0.217 |


## Problems

- This clean run is still below the best known `headroom-retract` baseline: throughput-leg output throughput fell from `73.21` to `69.78 tok/s`, and the highest constant-load leg fell from `71.30` to `57.12 tok/s`.
- Server logs remained dominated by serialized long-prefill windows. Typical hot-path steps were `decode≈14 ms, prefill≈720-760 ms`, which kept active decode concurrency low even when the sweep rate rose.
- The throughput leg still built a deep queue instead of reaching a healthy c16 steady state: `TTFT p50` rose to `15.2 s`, and the run logged `2,093,056` incomplete input tokens on that phase.

## Learnings

- Deleting the temporary gate/cap does not recover the `headroom-retract` result. It reproduces the earlier queued-chunk-reserve behavior almost exactly, so the real loss is deeper than that heuristic.
- The controlling bottleneck is still scheduler structure, not benchmark tooling: long 4096-token prefill chunks serialize on the same stream and starve decode, so the system spends most of its time in prefill-heavy steps instead of maintaining a wide running batch.
- For GuideLLM itself, this run reinforced an important interpretation rule: headline tables use `successful.*` metrics, while partial/in-flight work is tracked separately as `incomplete.*`. Short or overloaded runs can therefore understate steady-state generation if too few requests fully complete.

## Δ vs baseline

- **Baseline:** [2026-04-21-bench-guidellm-cuda-l4-c16-sglang-align-headroom-retract.md](./2026-04-21-bench-guidellm-cuda-l4-c16-sglang-align-headroom-retract.md)
- **Delta table** (only when a prior snapshot exists — else "first run"):

| metric | baseline | now | Δ% |
|---|---|---|---|
| TTFT p99 @ synchronous | 768.2 ms | 771.0 ms | +0.4% |
| out tok/s @ throughput leg | 73.21 | 69.78 | -4.7% |
| out tok/s @ highest constant leg | 71.30 | 57.12 | -19.9% |

## Artefacts

- Raw: `/content/workspace/agent-infer/bench-output/2026-04-21-cuda-l4-c16-delete-path-full-clean/benchmarks.json`
- CSV:  `/content/workspace/agent-infer/bench-output/2026-04-21-cuda-l4-c16-delete-path-full-clean/benchmarks.csv`
- HTML: `/content/workspace/agent-infer/bench-output/2026-04-21-cuda-l4-c16-delete-path-full-clean/benchmarks.html`

## Notes

- What changed in the code since baseline: removed the temporary mixed/decode-active prefill gate and removed the per-step one-chunk token cap, leaving the budgeted queued-chunk-reserve path as the single prefill admission path again.
- Suspected cause of any regression: the gating experiment was not the fundamental issue. The persistent regression versus `headroom-retract` comes from long serialized prefill work and insufficient overlap, not from the presence or absence of that temporary gate.
- Follow-ups: refactor the execution path so mixed decode and prefill share one semantic batch path instead of “mixed one slot + extra serial prefills”, then re-run the same c16 sweep from a fresh server.
