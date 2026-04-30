# longctx-32k Phase 1 baseline anchor — guidellm sweep, 0464fb3e, 2026-04-30

## Goal

- Aggregate the required three pre-patch Phase 1 P1.1 longctx-32k baseline
  runs at commit `0464fb3e` before designing or scanning a scheduler patch.

## Hypothesis

- The three-run anchor should show c=1 near the known parity point and c=4
  unstable or far below the SGLang c=4 reference of `16.27` output tok/s.

## Command

```bash
for r in 1 2 3; do
  WORKLOAD=longctx-32k scripts/bench_guidellm.sh \
    "longctx-32k-phase1-baseline-r${r}-0464fb3e" \
    --target http://127.0.0.1:8000 \
    --model Qwen3-4B \
    --processor infer/models/Qwen3-4B
done
```

## Environment

- **Backend:** CUDA
- **Model:** Qwen3-4B
- **Hardware:** NVIDIA L4, 23034 MiB VRAM, driver 580.82.07, nvcc
  `/usr/local/cuda/bin/nvcc` reports 12.8
- **Commit:** `0464fb3e`
- **Feature set:** `cargo build -p infer --release --features cuda`
- **Server flags:** `--kv-cache-dtype fp8 --num-slots 16 --max-seq-len 131072
  --mem-fraction-static 0.85 --max-num-batched-tokens 16384
  --max-prefill-tokens 16384 --schedule-policy fcfs`

## Canonical params

- Workload: `longctx-32k`
- `--profile concurrent`
- `--data prompt_tokens=32768,prompt_tokens_stdev=1,prompt_tokens_min=32768,prompt_tokens_max=32768,output_tokens=256,output_tokens_stdev=1,output_tokens_min=256,output_tokens_max=256`
- `--max-seconds 300`
- `--random-seed 20260416`
- `--rate 1,4`

## Results — three-run anchor

| run | c1 out tok/s | c1 TTFT p50 | c4 out tok/s | c4 TTFT p50 | c4 successful requests | c4 completed out | c4 incomplete out | validity |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| r1 | 10.00 | 12491.0 ms | 0.90 | 39264.8 ms | 1 | 256 | 510 | valid |
| r2 | 9.80 | 12497.5 ms | 8.07 | 100862.3 ms | 3 | 768 | 10 | valid |
| r3 | 9.50 | 12500.4 ms | 0.10 | n/a | 0 | 0 | 38 | invalid: c4 no successful requests |
| mean | 9.77 | 12496.3 ms | 3.02 | n/a | 1.33 | 341.3 | 186.0 | mixed |
| sample stddev | 0.25 | 4.8 ms | 4.39 | n/a | 1.53 | 391.0 | 280.9 | n/a |

## Results — vs SGLang reference

| metric | ARLE anchor | SGLang reference | ratio |
|---|---:|---:|---:|
| c4 output tok/s, three-run mean | 3.02 | 16.27 | 0.186x |
| c4 output tok/s, best run | 8.07 | 16.27 | 0.496x |
| c4 output tok/s, worst run | 0.10 | 16.27 | 0.006x |

## Scheduler Signals

- All runs showed mixed decode+prefill activity and `split=0`.
- r3 failed GuideLLM validation because c=4 produced no completed request in
  300s, while service after-snapshot still had 3 active/running requests.
- Plan-label counters are process-cumulative when the same server is reused;
  use r3 before/after deltas and per-run trace shapes, not absolute counter
  means, for scheduler interpretation.

## Problems

- c=4 has a high-variance failure edge: one run completed three c=4 requests,
  one completed one, and one completed none.
- The mean c=4 output throughput is only `0.186x` of the pinned SGLang
  reference, and the best run is still under half of SGLang.
- TTFT/tail behavior is not healthy even when c=4 output throughput appears
  higher: r2 reached `100862.3 ms` c4 TTFT p50.

## Learnings

- The patch acceptance gate cannot use a single c=4 run; it needs at least
  mean throughput plus validity and c4 completion checks.
- A scheduler patch should be rejected if it only increases output tokens by
  letting TTFT/tail or incomplete requests degrade.
- Future benchmark entries should restart the server between independent runs
  or record counter deltas from `service_stats_before.txt` to
  `service_stats_after.txt`.

## Δ vs baseline

- **Baseline:** this file is the pre-patch baseline anchor for Phase 1 P1.1.

| metric | baseline | now | Δ% |
|---|---|---|---|
| c1 output tok/s mean | n/a | 9.77 | n/a |
| c4 output tok/s mean | n/a | 3.02 | n/a |
| c4 valid runs | n/a | 2/3 | n/a |
| c4 no-success runs | n/a | 1/3 | n/a |

## Artefacts

- r1: `docs/experience/wins/2026-04-30-bench-guidellm-longctx-32k-phase1-baseline-r1-0464fb3e.md`
- r2: `docs/experience/wins/2026-04-30-bench-guidellm-longctx-32k-phase1-baseline-r2-0464fb3e.md`
- r3: `docs/experience/wins/2026-04-30-bench-guidellm-longctx-32k-phase1-baseline-r3-0464fb3e.md`
- Raw r1: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r1-0464fb3e/benchmarks.json`
- Raw r2: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r2-0464fb3e/benchmarks.json`
- Raw r3: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r3-0464fb3e/benchmarks.json`

## Notes

- Code changed since baseline: none. This anchor is pinned to pre-patch commit
  `0464fb3e`; the rejected headroom patch had already been reverted on `main`.
- Next step: design the Phase 1.5 scheduler patch from the SGLang admission
  research and ARLE gap table, then run the required headroom/mixed-mode scan
  against this anchor.
