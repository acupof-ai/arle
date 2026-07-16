# Rolling performance baselines

> Status: Active

One table per config fingerprint. Screening compares new runs against the
champion row — no second arm. Rules:

1. **Effect > ~10% (2× the measured cross-session drift band)**: rolling
   baseline verdict is valid; update the champion row, archive the binary.
2. **Effect inside the drift band (±3% measured 2026-07-16: same binary
   lineage, different session + GPU set): never kill on ambiguity — every
   stable positive gain is kept.** Escalate to a same-shell A/B against the
   archived champion binary (≥3 trials/arm, median + range) to resolve sign.
3. **Fingerprint change re-anchors**: any change to model, TP/EP, GPU set,
   serve flags, num_slots line, dataset, driver/CUDA invalidates the row —
   re-measure the champion before comparing.
4. **Anchor audit**: every ~5 accepted updates (and before any default flip),
   one A/B against the oldest archived binary bounds accumulated drift.

## DSv4-Flash-FP8 · 4×H20 GPUs 0–3 · TP=4/EP=4 · eager · port 8000

Champion: `2e635eda3` (grid-parallel FP32 compressor), build sha256
`fd568375fc06…`, pod archive `/host/arle-build/arle-armB-gridpar`.
Serve fingerprint: `arle serve --backend cuda --model-path
/host/DeepSeek-V4-Flash-FP8 --port 8000`, slot line `256 clamped to 2,
per_slot 9618MB, budget 20840MB`. Datasets: rates = `bench-prompts.jsonl`
(20×~3352 tok, 60 s); c1/c32 = `bench-prompts-64.jsonl` (64 varied, 120 s).
guidellm concurrent, seed 20260416. Measured 2026-07-16
([entry](experience/wins/2026-07-16-dsv4-fp32-compressor-grid-parallel.md)).

| point | ok | total tok/s | TTFT p50/p99 ms | ITL p50/p99 ms |
|---|---:|---:|---|---|
| rate 1 | 20 | 4116.8 | 521 / 3541 | 21.47 / 21.99 |
| rate 4 | 20 | 5339.4 | 1467 / 2295 | 75.80 / 81.24 |
| rate 8 | 20 | 5604.5 | 3921 / 4584 | 69.73 / 74.75 |
| rate 16 | 20 | 5481.2 | 6698 / 9686 | 72.61 / 77.71 |
| var-c1 | 35 | 853.6 | 3019 / 3082 | 21.83 / 22.35 |
| var-c32 | 41 | 1007.5 | 51842 / 87832 | 160.38 / 166.73 |

Pure-prefill anchor (c1, no queueing): ~3352-tok prompt / 521 ms TTFT ≈
**6434 tok/s prefill**.

Archived arms (pod): `arle-armA-serialprobe` (60be54d9a-equivalent, serial
probe), `arle-armB-gridpar` (2e635eda3). Raw:
`/host/arle-build/bench-output/2026-07-16-{fp32par,fp32serialA}-*`.
