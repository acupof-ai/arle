# guidellm sweep cuda-l4-int8 — 2026-04-15 (int8 KV counterpart to bf16 baseline)

Int8 KV counterpart to
[`2026-04-15-bench-guidellm-cuda-l4-bf16-baseline.md`](./2026-04-15-bench-guidellm-cuda-l4-bf16-baseline.md).
Same hardware, same commit, same canonical guidellm sweep params; only the
`--kv-cache-dtype int8` flag differs. This quantifies the bf16→int8 sweep
headline delta on L4/SM89 for Qwen3-4B at the 1024/256 workload.

## Context

- **Backend:** cuda
- **Model:** Qwen3-4B (bf16 weights, int8 KV cache)
- **Hardware:** NVIDIA L4 (23034 MiB HBM, driver 580.82.07), SM89
- **Commit:** 40151f3 (post-pull, M0.4 page_size=16 + kTargetBlocksPerSm=32
  int8 decode kernel tuning already landed)
- **Feature set:** `cargo build --release -p infer` (default features = cuda)
- **Non-default flags / env vars:** `--kv-cache-dtype int8` (vs the bf16 baseline
  which ran with defaults). Auto-sizer now sees ~1.9× the per-slot headroom
  and provisions **14 slots** (vs bf16 baseline's 7) at the same max_seq_len=4096.
- **Server launch:** `target/release/infer --model-path models/Qwen3-4B --kv-cache-dtype int8`
  on port 8000, 14 slots, max_seq_len=4096.

## Canonical params (DO NOT CHANGE PER-RUN)

```
guidellm benchmark \
  --target http://localhost:8000 \
  --model <model> \
  --profile sweep \
  --data  prompt_tokens=1024,output_tokens=256 \
  --max-seconds 60 \
  --random-seed 20260416 \
  --output-dir /content/workspace/agent-infer/bench-output/2026-04-15-cuda-l4-int8/ \
  --outputs json,csv,html
```

Invoked via: `scripts/bench_guidellm.sh <backend-label>`

## Results — sweep headline table

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| sync | 189.9 | 196.3 | 33.48 | 33.56 | 29.34 | 0.117 |
| throughput | 29115 | 58507.4 | 180.63 | 189.59 | 47.06 | 3.467 |
| 0.5354166666666667r/s | 347.7 | 376.4 | 37.11 | 97.94 | 21.77 | 0.517 |
| 0.9541666666666666r/s | 359.6 | 378 | 41 | 41.08 | 26.63 | 0.95 |
| 1.3729166666666666r/s | 349.1 | 373 | 48.5 | 263.56 | 24.46 | 1.367 |
| 1.7916666666666665r/s | 356 | 373.8 | 55.93 | 56.22 | 43.67 | 1.75 |
| 2.2104166666666667r/s | 344.2 | 365.2 | 66.77 | 67.06 | 31.18 | 2.167 |
| 2.6291666666666664r/s | 349.4 | 369.2 | 77.26 | 83.29 | 43.13 | 2.567 |
| 3.047916666666666r/s | 460.6 | 498 | 105.51 | 110.95 | 34.35 | 2.967 |
| 3.4666666666666663r/s | 492.8 | 500.4 | 141.78 | 143.17 | 25.21 | 3.383 |


## Artefacts

- Raw: `/content/workspace/agent-infer/bench-output/2026-04-15-cuda-l4-int8/benchmarks.json`
- CSV:  `/content/workspace/agent-infer/bench-output/2026-04-15-cuda-l4-int8/benchmarks.csv`
- HTML: `/content/workspace/agent-infer/bench-output/2026-04-15-cuda-l4-int8/benchmarks.html`

## Delta vs previous snapshot

Baseline: [`2026-04-15-bench-guidellm-cuda-l4-bf16-baseline.md`](./2026-04-15-bench-guidellm-cuda-l4-bf16-baseline.md)
(same commit, same hardware, only the kv-cache dtype differs).

| metric | bf16 baseline | int8 now | Δ |
|---|---|---|---|
| TTFT p50 @ sync | 191.6 ms | 189.9 ms | **−0.9 %** (noise) |
| ITL p50 @ sync | 34.09 ms | 33.48 ms | **−1.8 %** (noise) |
| out tok/s @ sync | 28.91 | 29.34 | +1.5 % |
| out tok/s @ throughput | **132.87** | **47.06** | **−64.6 %** 🚨 |
| req/s @ throughput | 0.583 | 3.467 | +4.95× |
| ITL p50 @ throughput | 41.82 ms | 180.63 ms | **+4.3×** |
| num_slots (auto) | 7 | 14 | +2× (expected — int8 KV is ~half the bytes) |

## Notes

- **Short context (sync phase) is at parity.** 189.9 ms TTFT and 33.48 ms ITL
  at batch=1 are both within noise of bf16. Confirms the M0.4 + splits=32 +
  SMEM-tile work closed the short-context int8 gap.
- **The throughput phase is where int8 loses.** Guidellm's throughput profile
  floods the server; at batch>7 the int8 decode kernel's worse scaling
  profile dominates. ITL climbs from 33 ms (batch=1) to 180 ms (batch≈14) —
  more than 5× the slowdown the bf16 kernel shows (34 ms → 42 ms from
  batch=1→7). This is the same residual gap documented in
  [`2026-04-15-bench-longseq-int8-splits32.md`](./2026-04-15-bench-longseq-int8-splits32.md)
  and root-caused in
  [`../../research/kv-quant-decode-industry-survey.md`](../../research/kv-quant-decode-industry-survey.md):
  no tensor-core path for int8 quant decode on SM89.
- **Aggregate throughput regression: −64.6 %.** Int8 quant lets us run 2×
  the slots but each batched iteration costs ~4× more, so aggregate output
  throughput goes down, not up. For this 1024/256 workload on L4, **int8
  KV is not a win vs bf16.**
- **Per-request (sync) is neutral to slightly positive.** Int8's HBM bandwidth
  savings exactly offset the extra register dequant work at batch=1.

## Follow-ups

- **Keep bf16 as the default kv-cache-dtype on L4.** The int8 flag makes
  sense when (a) VRAM headroom matters more than throughput or (b) per-request
  latency is the only metric. At saturation it's a throughput regression.
- Re-run int8 at a **longer output** (2k / 4k) shape — the bf16 bottleneck
  shifts from HBM to compute as decoded tokens accumulate, which could flip
  the sign of the delta.
- The **coordinator real byte path** batch (T1 spill to host pinned pool)
  is what would make the extra int8 slots useful — it lets the server hold
  many more warm sessions without pushing more concurrent decode load onto
  the slow int8 kernel. Plan:
  [`../../plans/tiered-kv-cache-coordinator-real-byte-path.md`](../../plans/tiered-kv-cache-coordinator-real-byte-path.md).
- Do not change the canonical int8 default-recommendation framing in
  `docs/support-matrix.md` — the industry survey research already flagged
  int8 as "advanced flag, not the default".
