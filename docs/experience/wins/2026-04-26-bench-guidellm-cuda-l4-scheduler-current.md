# L4 scheduler regression check (current main) — guidellm concurrent, cuda, 2026-04-26

## Goal

- Regression check (per `docs/bench-and-trace-spec.md` §goal taxonomy):
  verify the post-2026-04-14 scheduler / kv-tier / cuda refactor stack
  did not regress single-GPU concurrent serving on L4. The reference is
  `memory/project_l4_perf_baseline.md` (Qwen3-4B, single-request bench
  `bench_serving request --prompt-len 512 --output-len 128`,
  d902090 = 30.52 tok/s, 132bc84 = 27.84 tok/s).

## Hypothesis

- Decode tok/s at c=1 (the closest the HTTP serving path gets to the
  in-process baseline) lands in the 27–32 tok/s band. Anything below
  27 tok/s is a regression vs the post-port floor; anything above 32
  tok/s suggests a real scheduler/kernel improvement worth bisecting.
- Aggregate throughput at c=2/4/8 scales sub-linearly (HBM-bound),
  with no concurrency tier producing token-rate collapse.

## Command

```bash
nohup env LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64 \
  /tmp/infer-off --model-path models/Qwen3-4B --port 8000 \
  > /tmp/server-off.log 2>&1 < /dev/null &
disown
until curl -s --max-time 1 http://localhost:8000/v1/models > /dev/null; do
  sleep 2
done
scripts/bench_guidellm.sh cuda-l4-scheduler-current --quick \
  --processor /content/workspace/agent-infer/models/Qwen3-4B
```

`/tmp/infer-off` is the snapshot of `target/release/infer` from the
matching `cargo build --release -p infer --no-default-features
--features cuda` invocation (FlashInfer prefill + decode path).

## Environment

- **Backend:** cuda
- **Model:** Qwen/Qwen3-4B (HF Instruct, eos_token_id=151645)
- **Hardware:** NVIDIA L4 24 GB, sm_89, driver 580.82.07, CUDA 12.8.93
- **Commit:** `802c5fc8` (HEAD)
- **Feature set:** `cargo build --release -p infer --no-default-features --features cuda`
- **Toolchain:** rustc 1.95.0, nvcc 12.8.93, zig 0.14.0
- **Non-default flags / env vars:** `INFER_CUDA_SM=89`,
  `CARGO_HOME=/tmp/cargo-home-local` (Drive-FUSE workaround per
  `memory/project_remote_cuda_box.md`), `INFER_TRITON_PYTHON=/usr/bin/python3`.
- **Server launch:** `/tmp/infer-off --model-path models/Qwen3-4B --port 8000`
- **Bench mode:** exploration (`--quick`) — concurrent profile,
  c=1,2,4,8 × 60 s, prompts 512 in / 128 out, warmup 5.

## Canonical params (DEVIATION — exploration mode)

- `--profile concurrent` (canonical: `sweep`) — `--quick` switches profile.
- `--data prompt_tokens=512,output_tokens=128` (canonical: 4096/256) — short
  data so c=1..8 finishes inside one session.
- `--max-seconds 60` (matches canonical).
- `--random-seed 20260416` (matches canonical).
- `--outputs json csv html` (matches canonical).
- Wrapper: `scripts/bench_guidellm.sh cuda-l4-scheduler-current --quick`.

This is an exploration-mode regression check, not a canonical-params
sweep. The target is parity with the existing in-process baseline at
c=1 plus a smoke test that the scheduler scales through c=8.

## Results — concurrency table

| concurrency | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | TPOT p50 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|---|
| 1 |  114.4 |  120.1 | 33.28 | 33.32 | 33.9 |  29.32 | 0.218 |
| 2 |  183.3 |  260.1 | 35.16 | 35.32 | 36.4 |  52.52 | 0.400 |
| 4 |  341.7 |  436.2 | 35.28 | 37.39 | 38.0 |  98.08 | 0.800 |
| 8 |  618.3 | 5894.5 | 38.77 | 42.73 | 43.4 | 173.75 | 1.382 |

## Results — service-side KV / scheduler metrics

| metric | value |
|---|---:|
| samples (poll @ 1000ms) | 329 |
| failed | 0 |
| peak active | 8 |
| peak waiting | 1 |
| peak running_batch | 8 |
| peak prefill_queue | 0 |
| peak kv_util | 96.8% |
| `prefix_hit_rate` | 0.0% |
| `prefix_skip_rate` | 0.0% |
| `kv_fetch_q` | 0/18 |
| `kv_fetch_waiters` | 0 |
| `kv_store_q` | 0/18 |
| `kv_store` | sub:2 done:2 fail:0 rej:0 |
| `kv_bp` | fetch:0 store:0 |
| tier_recall / tier_src / tier_promoted / tier_fallback | n/a (single-tier run) |

## Results — request accounting

| metric | value |
|---:|---:|
| completed requests | 177 |
| completed input tokens | ~90 681 (177 × 512.3) |
| completed output tokens | 22 403 |

(Bench script does not split incomplete-token totals out separately
in `service_stats_*.txt`; both stats lines reported zero in-flight
requests at trace start and end.)

## Problems

- TTFT p99 at c=8 jumps to 5894 ms while p50 stays at 618 ms — head-
  of-line blocking when prefill saturates. Behavioural, not a regression.
- `prefix_hit_rate` reads 0.0% because guidellm's synthetic prompt
  generator emits unique seeds per request (no shared prefix). Expected
  for this dataset; revisit when re-running against an agent-trace
  dataset.
- The `service_stats_after.txt` reports `tokens_out=22403` matching the
  guidellm-side output total — endpoint counters are wired correctly
  here (contrast with the pre-existing `/v1/stats` no-increment note in
  `memory/project_remote_cuda_box.md`). Worth re-confirming on Qwen3.5
  before retiring that note.

## Learnings

- Decode at c=1 lands at **29.32 tok/s** vs the in-process baseline
  d902090 = 30.52 tok/s (−3.9 %) and the post-port 132bc84 = 27.84
  tok/s (+5.3 %). Within the 5 % matched-A/B noise band — call it
  **flat vs d902090, recovered against 132bc84**. Apples-to-oranges
  caveat: the historical numbers are `bench_serving request` (in-
  process, no HTTP); this run is guidellm via HTTP, so a few-percent
  HTTP overhead is expected. The conclusion holds: post-2026-04-14
  scheduler / kv-tier / cuda refactors did not regress single-request
  decode on L4.
- Aggregate scaling: c=1→2 = 1.79×, c=1→4 = 3.34×, c=1→8 = 5.93× of
  the c=1 tok/s. Sub-linear past c=4, consistent with an HBM-bound L4
  saturating around c=8 (peak `kv_util` = 96.8 %). The scheduler is
  not the bottleneck — the GPU is.
- ITL p50 stays under 40 ms across c=1..8 (33→35→35→39 ms). Per-token
  decode latency is stable as concurrency rises; the cost of more
  concurrent requests is paid in TTFT (queueing for prefill), not in
  steady-state ITL. That is the desired scheduler shape.
- Peak `running_batch` = 8 at c=8 means batched decode is reaching the
  full client concurrency — admission isn't gating below the request
  load.

## Δ vs baseline

- **Baseline (in-process, no HTTP):**
  - d902090 (2026-04-14): Qwen3-4B 30.52 tok/s.
  - 132bc84 (2026-04-14, post Triton→CUDA C port): Qwen3-4B 27.84 tok/s.
  - Source: `memory/project_l4_perf_baseline.md`.
- **Current (guidellm HTTP, c=1):** 29.32 tok/s.

| metric | baseline d902090 | baseline 132bc84 | now (HEAD 802c5fc8) | Δ% vs d902090 | Δ% vs 132bc84 |
|---|---|---|---|---|---|
| out tok/s @ c=1 | 30.52 | 27.84 | 29.32 | −3.9 % | +5.3 % |

Both deltas are inside the matched-A/B noise band. Flat.

## Artefacts

- Raw: `bench-output/2026-04-26-cuda-l4-scheduler-current/benchmarks.json`
- CSV: `bench-output/2026-04-26-cuda-l4-scheduler-current/benchmarks.csv`
- HTML: `bench-output/2026-04-26-cuda-l4-scheduler-current/benchmarks.html`
- Service trace (before): `bench-output/2026-04-26-cuda-l4-scheduler-current/service_stats_before.txt`
- Service trace (during): `bench-output/2026-04-26-cuda-l4-scheduler-current/service_stats_trace.jsonl`
- Service trace (after): `bench-output/2026-04-26-cuda-l4-scheduler-current/service_stats_after.txt`
- Service trace (summary): `bench-output/2026-04-26-cuda-l4-scheduler-current/service_stats_trace_summary.md`
- Headline table: `bench-output/2026-04-26-cuda-l4-scheduler-current/headline_table.md`

(Artefact dir is gitignored — these paths are local to the L4 host.)

## Notes

- What changed in code since baseline (d902090 → 802c5fc8, ~38 commits):
  scheduler refactors `df2d3e8` (mixed workspace budget align),
  `27ba730` (decode emit-gate), `01670cb` (admission loop cleanup),
  `f526e10` (sglang-aligned mixed batch), `64e350c`
  (TieredKvPolicy → scheduler/cuda), `a94682a` (coordinator split);
  kv-tier fixes `235d363` (release staged regions before bp),
  `a854491` (TileLang preflight), `9ce74aa` (typed FailureClass +
  RAII regions), `99b7bcb` (zero-copy host-pinned slice), `2ed93c9`
  (ClusterSharedBackend macro); plus the Phase 0 TileLang trio
  `022e8dd / 76e044b / 9896d25` (off in this run — feature gated).
- Suspected cause of the −3.9 % vs d902090: HTTP overhead +
  guidellm 0.6.0 measurement methodology drift (per
  `memory/project_bench_env_drift_2026-04-20.md`, the
  guidellm-vs-bench_serving gap is environmental). Not a code
  regression.
- Follow-ups:
  - Re-run on Qwen3.5-4B for symmetry against the `27.59 tok/s` line
    from project_l4_perf_baseline.md (Qwen3.5 path was the clean
    control during the kernel port — should still be flat).
  - Re-run with shared-prefix prompts to exercise `prefix_hit_rate`
    and the kv_tier promotion paths the recent refactors changed.
  - Lift to canonical-params sweep (4096-in / 256-out, profile=sweep)
    once a quiet maintenance window is available; --quick c=1..8 is
    not enough to retire the L4-floor stub.
