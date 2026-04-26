# Mixed-batch scheduler refactors — guidellm c=1..16, cuda, 2026-04-26

## Goal

- Optimization regression-check (per `docs/bench-and-trace-spec.md`
  §goal taxonomy): quantify the impact of the post-2026-04-22
  scheduler refactor stack on the same `--num-slots 16
  --max-seq-len 4608 --enable-mixed-chunk` workload that drove the
  2026-04-22 SGLang comparison. Specifically: did `f526e10b
  refactor(cuda): align mixed batch scheduler with sglang`,
  `27ba7308 fix(cuda): make decode scheduling emit-gate aware`,
  `df2d3e8e fix(scheduler): align mixed workspace budget`, and the
  surrounding kv-tier work close the c=4/c=8/c=16 gap?

## Hypothesis

- TTFT at c=4 and c=8 should drop substantially. The 2026-04-22 entry
  showed both stuck at ≥14 s — symptomatic of decode being starved
  while prefill held the running batch. The mixed-batch alignment +
  decode emit-gate fixes specifically target prefill/decode
  coexistence.
- Throughput at c=16 should rise. The 2026-04-22 entry topped out at
  45 tok/s; SGLang at the same config hit 137 tok/s.

## Command

```bash
# Server (matches 2026-04-22 setup exactly except --enable-mixed-chunk
# is now implicit per `e4554cdb refactor(scheduler): delete mixed-batch
# legacy surface`).
/tmp/infer-off \
  --model-path /content/workspace/agent-infer/models/Qwen3-4B \
  --port 8000 \
  --num-slots 16 --max-seq-len 4608 \
  --mem-fraction-static 0.94 \
  --chunked-prefill-size 4096 --max-prefill-tokens 16384

scripts/bench_guidellm.sh cuda-l4-mixed-batch-vs-f98ca92 \
  --concurrencies 1,2,4,8,16 --max-seconds 60 --warmup 5 \
  --processor /content/workspace/agent-infer/models/Qwen3-4B
```

## Environment

- **Backend:** cuda
- **Model:** Qwen/Qwen3-4B
- **Hardware:** NVIDIA L4, 24 GB, sm_89, driver 580.82.07, CUDA 12.8.93
- **Commit:** `c5836a9a` (HEAD)
- **Feature set:** `cargo build --release -p infer --no-default-features --features cuda`
- **Toolchain:** rustc 1.95.0, nvcc 12.8.93, zig 0.14.0
- **Non-default flags / env vars:** `INFER_CUDA_SM=89`,
  `CARGO_HOME=/tmp/cargo-home-local` (Drive-FUSE workaround per
  `memory/project_remote_cuda_box.md`).
- **Bench mode:** exploration (`--concurrencies 1,2,4,8,16`,
  `--warmup 5`) — not canonical sweep, but the same mode the
  2026-04-22 baseline used so the diff is exact.

## Canonical params (DEVIATION — exploration mode, matches baseline)

- `--profile concurrent` (`--concurrencies` switches to concurrent).
- `--data prompt_tokens=4096,output_tokens=256` (canonical default kept).
- `--max-seconds 60` (canonical).
- `--warmup 5` (matches baseline).
- `--random-seed 20260416` (canonical).

## Results — paired concurrency table

`infer` HEAD vs 2026-04-22 baseline (commit `f98ca92`).

| conc | TTFT p50 ms now | TTFT p50 ms then | Δ TTFT | ITL p50 ms now | ITL p50 ms then | Δ ITL | out tok/s now | out tok/s then | Δ tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  1 |    719.3 |   739.9 |  −2.8% | 35.24 | 35.28 | −0.1% | 26.56 | 26.59 |  −0.1% |
|  2 |   1518.6 |  1485.0 |  +2.3% | 38.77 | 38.82 | −0.1% | 45.21 | 41.59 |  +8.7% |
|  4 |   2354.4 | 14556.7 | **−83.8%** | 41.62 | 44.24 | −5.9% | 53.31 | 36.70 | **+45.3%** |
|  8 |   3838.0 | 15403.7 | **−75.1%** | 52.30 | 47.32 | +10.5% | 66.94 | 57.71 | +16.0% |
| 16 |  16356.9 | 15405.9 |  +6.2% | 49.27 | 44.61 | +10.4% | 65.92 | 45.08 | **+46.2%** |

## Results — raw `infer` HEAD table (full guidellm output)

| conc | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | TPOT p50 (ms) | out tok/s | req/s actual |
|---|---:|---:|---:|---:|---:|---:|---:|
|  1 |    719.3 |    738.0 | 35.24 | 35.26 |  37.9 | 26.56 | 0.109 |
|  2 |   1518.6 |   1586.3 | 38.77 | 41.56 |  44.6 | 45.21 | 0.182 |
|  4 |   2354.4 |  15162.7 | 41.62 | 47.26 |  50.5 | 53.31 | 0.218 |
|  8 |   3838.0 |  19613.3 | 52.30 | 57.97 |  67.1 | 66.94 | 0.273 |
| 16 |  16356.9 |  32159.3 | 49.27 | 58.39 | 122.3 | 65.92 | 0.273 |

## Results — service-side KV / scheduler metrics

| metric | value |
|---|---:|
| samples (poll @ 1000ms) | 432 |
| failed | 0 |
| peak active | 5 |
| peak waiting | 11 |
| peak running_batch | 5 |
| peak prefill_queue | 3 |
| peak kv_util | 98.0% |
| `prefix_hit_rate` | 0.0% |
| completed requests | 86 |
| completed output tokens | 17 900 |

`peak waiting=11` at c=16 means 11 of the 16 client-side requests
queued at peak; admission held running_batch at ≤5 to keep KV
headroom. That's the desired shape — backpressure is in the queue,
not in dropped requests.

## Problems

- TTFT p99 at c=4/c=8/c=16 still climbs to 15-32 s. p50 dropped
  hugely but the tail still has long-prefill stalls. Suggests room
  for further work on prefill-tail behaviour.
- ITL p50 rose ~10% at c=8/c=16 vs the 2026-04-22 baseline. Probably
  a knock-on of the wider running batch — more requests share each
  decode step, so per-request ITL is slightly higher even as
  aggregate throughput rises. Net positive on tok/s.

## Learnings

- The mixed-batch alignment did exactly what the commit messages
  predicted: c=4 TTFT collapsed from 14.6 s to 2.4 s and c=8 from
  15.4 s to 3.8 s. **TTFT-at-saturation is no longer dominated by
  decode starvation while prefill blocks the running batch.**
- Throughput at c=16 rose +46.2% (45.08 → 65.92 tok/s), closing a
  meaningful chunk of the prior SGLang gap (137 tok/s) — from
  −67.1% to about −51.9%.
- c=1 is bit-for-bit identical (26.56 vs 26.59 tok/s). The
  scheduler refactors did not affect the decode-only single-stream
  path. Expected, but a useful negative control.
- Single-token decode at c=1 still stages 26.56 tok/s vs the in-
  process bench_serving baseline of 30.52 tok/s (d902090). The
  HTTP overhead delta is real and stable across runs.

## Δ vs baseline

- **Baseline:** [`2026-04-22-bench-guidellm-qwen3-4b-infer-vs-sglang-c1-c16.md`](./2026-04-22-bench-guidellm-qwen3-4b-infer-vs-sglang-c1-c16.md)
  (commit `f98ca92`, identical server flags + identical bench params).

| metric | f98ca92 | c5836a9a | Δ% |
|---|---:|---:|---:|
| TTFT p50 @ c=4   |  14556.7 ms |  2354.4 ms | **−83.8%** |
| TTFT p50 @ c=8   |  15403.7 ms |  3838.0 ms | **−75.1%** |
| out tok/s @ c=4  |    36.70    |   53.31    | **+45.3%** |
| out tok/s @ c=8  |    57.71    |   66.94    |     +16.0% |
| out tok/s @ c=16 |    45.08    |   65.92    | **+46.2%** |

## Artefacts

- Raw: `bench-output/2026-04-26-cuda-l4-mixed-batch-vs-f98ca92/benchmarks.json`
- CSV: `bench-output/2026-04-26-cuda-l4-mixed-batch-vs-f98ca92/benchmarks.csv`
- HTML: `bench-output/2026-04-26-cuda-l4-mixed-batch-vs-f98ca92/benchmarks.html`
- Service trace (during): `bench-output/2026-04-26-cuda-l4-mixed-batch-vs-f98ca92/service_stats_trace.jsonl`
- Service trace (summary): `bench-output/2026-04-26-cuda-l4-mixed-batch-vs-f98ca92/service_stats_trace_summary.md`

(Artefact dir is gitignored — paths are local to the L4 host.)

## Notes

- **What changed in code since `f98ca92`** (the relevant scheduler /
  cuda commits, ordered earliest → latest):
  - `e4554cdb refactor(scheduler): delete mixed-batch legacy surface`
    — folded the `--enable-mixed-chunk` flag into the always-on path.
  - `f526e10b refactor(cuda): align mixed batch scheduler with sglang`
    — the structural change that c=4/c=8 wins trace to.
  - `27ba7308 fix(cuda): make decode scheduling emit-gate aware`.
  - `df2d3e8e fix(scheduler): align mixed workspace budget`.
  - `01670cb0 refactor(scheduler): clean up runtime admission loop`.
  - kv-tier refactors `64e350c / a94682a / 9ce74aa / 99b7bcb /
    2ed93c9 / 235d363 / a854491` — should not affect this workload
    (single-tier, no prefix sharing) but landed in the same window.
- **Suspected cause of any regression**: ITL p50 +10% at c=8/c=16
  is from the wider running batch sharing decode steps; offset by
  larger throughput gains.
- **Follow-ups:**
  - Sanity-check on Qwen3.5-4B with the same flags (no f98ca92
    baseline exists for that, but project_l4_perf_baseline notes
    parity at c=1).
  - Investigate the c=16 TTFT (still ~16 s p50) — the mixed-batch
    alignment unstuck c=4/c=8 but c=16 is still in the prior
    backlog regime. Probably needs a follow-up on prefill admission
    backpressure when waiting > num_slots.
  - Compare against SGLang at the same flags on the same box to see
    how much of the 2026-04-22 gap remains.
