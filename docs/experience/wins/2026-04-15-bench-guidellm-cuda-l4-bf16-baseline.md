# guidellm sweep cuda-l4-bf16-baseline — 2026-04-15 (first canonical guidellm run)

First canonical `scripts/bench_guidellm.sh` run on the remote L4 box after the
wrapper bug fixes (`--backend openai_http` + `--processor models/Qwen3-4B`) and
the `OpenAiChatContent` parts-array deserializer landed. Baseline for every
future guidellm win on this hardware.

## Context

- **Backend:** cuda
- **Model:** Qwen3-4B (bf16, local `models/Qwen3-4B`)
- **Hardware:** NVIDIA L4 (23034 MiB HBM, driver 580.82.07), SM89
- **Commit:** 40151f3 (post-pull, post content-parts deserializer fix)
- **Feature set:** `cargo build --release -p infer` (default features = cuda)
- **Non-default flags / env vars:** none (defaults: kv_cache_dtype=bf16,
  cuda_graph=on, auto_num_slots=7 on 9.2 GB of available HBM headroom).
- **Server launch:** `target/release/infer --model-path models/Qwen3-4B` on
  port 8000, 7 slots, max_seq_len=4096.

## Canonical params (DO NOT CHANGE PER-RUN)

```
guidellm benchmark \
  --target http://localhost:8000 \
  --model <model> \
  --profile sweep \
  --data  prompt_tokens=1024,output_tokens=256 \
  --max-seconds 60 \
  --random-seed 20260416 \
  --output-dir /content/workspace/agent-infer/bench-output/2026-04-15-cuda-l4-bf16-baseline/ \
  --outputs json,csv,html
```

Invoked via: `scripts/bench_guidellm.sh <backend-label>`

## Results — sweep headline table

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| sync | 191.6 | 196.2 | 34.09 | 34.15 | 28.91 | 0.1 |
| throughput | 24365.1 | 58253.2 | 41.82 | 44.25 | 132.87 | 0.583 |
| 0.16041666666666668r/s | 360.3 | 376.1 | 35.61 | 35.63 | 38.95 | 0.15 |
| 0.22083333333333338r/s | 357.1 | 370.3 | 36.95 | 36.98 | 51.71 | 0.2 |
| 0.28125000000000006r/s | 361.9 | 372.9 | 37.47 | 37.54 | 64.5 | 0.25 |
| 0.34166666666666673r/s | 362.2 | 382.3 | 38.96 | 39 | 76.93 | 0.283 |
| 0.4020833333333334r/s | 358.7 | 377.9 | 40.37 | 40.42 | 88.67 | 0.333 |
| 0.46250000000000013r/s | 365.8 | 385.6 | 41.09 | 41.12 | 101.08 | 0.383 |
| 0.5229166666666668r/s | 363.2 | 385.3 | 42.6 | 42.64 | 113.71 | 0.433 |
| 0.5833333333333335r/s | 366 | 386.1 | 44.1 | 44.15 | 124.88 | 0.483 |


## Artefacts

- Raw: `/content/workspace/agent-infer/bench-output/2026-04-15-cuda-l4-bf16-baseline/benchmarks.json`
- CSV:  `/content/workspace/agent-infer/bench-output/2026-04-15-cuda-l4-bf16-baseline/benchmarks.csv`
- HTML: `/content/workspace/agent-infer/bench-output/2026-04-15-cuda-l4-bf16-baseline/benchmarks.html`

## Delta vs previous snapshot

First canonical guidellm run — no prior snapshot under this tool. Historical
`scripts/bench_throughput_sweep.py` wins (`2026-04-14-bench-peak-throughput.md`
and `2026-04-15-bench-hbm-peak-throughput.md`) reported ~132 out tok/s sustained
at saturation with 7 slots under the same flags. Guidellm's `throughput` phase
hits **132.87 out tok/s** — dead-on.

## Notes

- **sync @ 1024/256:** 28.91 out tok/s is the single-stream ceiling for the
  1024-prompt / 256-output shape. Matches the 30.5 tok/s tok-per-sec figure
  from [`2026-04-14-bench-peak-throughput.md`](./2026-04-14-bench-peak-throughput.md)
  scaled by the 1 % difference between output-only and prompt-inclusive tok/s.
- **TTFT p50 @ sync:** 191 ms. 1024-token prefill on L4 with cuda_graph + FlashInfer
  HD128 dispatch.
- **ITL scaling:** ITL p50 grows linearly from 34.09 ms (sync) to 44.1 ms at
  saturation — the expected batch-size-induced slowdown as decode rows share
  one iteration. Matches the 36 ms / batch=1 → 45 ms / batch=7 profile from
  prior peak-throughput wins.
- **Throughput phase TTFT p99 = 58.3 s:** this is the queue-wait artefact of
  guidellm's unbounded `throughput` profile — it floods the server until the
  concurrency constraint fires, and TTFT then includes queue time for the
  tail of the flood. Do NOT treat this as a real prefill regression.
- **No regressions vs baseline.** First canonical guidellm snapshot — future
  runs compare against this file.

## Follow-ups

- Run int8 counterpart (`./scripts/bench_guidellm.sh cuda-l4-int8`) for the
  Qwen3-4B int8 KV delta vs this bf16 baseline.
- After the coordinator real-byte-path batch lands
  ([`../plans/tiered-kv-cache-coordinator-real-byte-path.md`](../../plans/tiered-kv-cache-coordinator-real-byte-path.md)),
  re-run this label and diff against this baseline — the acceptance doc
  requires steady-state regression ≤ 3 %.
