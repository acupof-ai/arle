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

Champion: `672b8ac08` (grid-parallel FP32 compressor + slot hoist), build
`--features cuda,nccl`, pod `target/release/arle` (2026-07-16 08:11 build).
Serve fingerprint: `arle serve --backend cuda --model-path
/host/DeepSeek-V4-Flash-FP8 --port 8000`, slot line `256 clamped to 59,
per_slot 338MB, budget 20584MB, shared comp capacity 84736 tokens`.
Datasets: rates = `bench-prompts.jsonl` (20×~3352 tok, 60 s); c1/c32 =
`bench-prompts-64.jsonl` (64 varied, 120 s). guidellm concurrent, seed
20260416. Measured 2026-07-16
([entry](experience/wins/2026-07-16-dsv4-fp32-scratch-hoist-slots.md)).

| point | ok | total tok/s | out tok/s | TTFT p50 ms | ITL p50 ms |
|---|---:|---:|---:|---:|---:|
| rate 1 | 20 | 4514 | 21.4 | 446 | 21.5 |
| rate 4 | 20 | 6663 | 31.7 | 1270 | 56.8 |
| rate 8 | 20 | 7985 | 37.9 | 2612 | 48.6 |
| rate 16 | 20 | 8130 | 38.6 | 5397 | 75.2 |
| var-c1 | 35 | 850 | 4.9 | 3031 | 21.9 |
| var-c32 | 59/101.5 s | — (accounting caveat) | — | 32630 | 2079 |

**KNOWN CRASH at c32**: `HostPagedKvPool out of pages` is fatal at ~101 s
([errors](experience/errors/2026-07-16-dsv4-c32-hostpagedkvpool-fatal.md)) —
c32 rows are pre-crash; fix pending before this regime is production-safe.

Pure-prefill anchor (c1, no queueing): ~3352-tok prompt / 446 ms TTFT ≈
**7516 tok/s prefill**.

Archived arms (pod): `arle-armA-serialprobe` (serial probe),
`arle-armB-gridpar` (2e635eda3, per_slot 9618MB/2 slots). Raw:
`/host/arle-build/bench-output/2026-07-16-{fp32par,fp32serialA,fp32slots}-*`.
