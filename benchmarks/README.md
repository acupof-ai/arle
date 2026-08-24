# Benchmark snapshots

Committed, distilled bench results — the single source of truth for the headline
numbers shown on the [README front door](../README.md#performance). Raw native
artefacts (json / csv / html per run) stay in the gitignored `bench-output/`;
this folder keeps only the distilled headline metrics, version-controlled, each
row traceable to a dated [`docs/experience/wins/`](../docs/experience/wins/) entry.

**No floating numbers rule:** a perf figure quoted anywhere in the repo as a
*current* capability must resolve to a snapshot here (or a wins entry a snapshot
links to). Stale micro-benches from deleted code paths get archived, not quoted.

## Current canonical numbers

### Metal — Apple M4 Pro (48 GB), single user, c=1

512-in / 128-out · temp=0 · median of 6 · build `4ea77e11` · decode = single-stream generation rate.
Snapshot: [`snapshots/2026-06-14-metal-m4pro-ladder.json`](snapshots/2026-06-14-metal-m4pro-ladder.json) ·
wins: [2026-06-14-bench-metal-m4pro-local-model-ladder](../docs/experience/wins/2026-06-14-bench-metal-m4pro-local-model-ladder.md)

| Model · Metal 4-bit | Decode | TPOT | TTFT |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 317.8 tok/s | 3.15 ms | 168.5 ms |
| Qwen3.5-4B | 84.1 tok/s | 11.89 ms | 820.3 ms |
| Qwen3.5-9B | 50.0 tok/s | 20.01 ms | 1448.8 ms |
| Qwen3.6-35B-A3B · MoE (~3B active) | 85.3 tok/s | 11.73 ms | 1231.0 ms |

**Not served (fail closed)** — every other locally-cached checkpoint was attempted and rejected at validation; the Metal serve path is **Qwen3.5/3.6 family** only:

| Model | Why it fails closed |
|---|---|
| Qwen3.6-35B-A3B-**MTP**-4bit | weight-prefix mismatch — the MTP draft head changes the layout (`could not detect Qwen3.5 text weight prefix`) |
| z-lab Qwen3.5-4B-DFlash · Qwen3.6-35B-A3B-DFlash | draft-only checkpoints, no standard tokenizer to load |
| Qwen2.5-0.5B / 1.5B-bf16, Llama-3.2-1B-bf16, Qwen3-0.6B | non-Qwen3.5 family — `R3a Metal executor requires Qwen3.5 layer_types` (the 1.5B hung in init, killed at 120s) |

### CUDA — DeepSeek-V4-Flash, 8×H20 (TP=8 / EP=8, FP8 MoE)

Recorded from the 2026-06-13 → 06-14 decode campaign wins entries (no local CUDA;
not re-measured here — provenance is the linked wins):

| Metric | Value | Source |
|---|---:|---|
| B=1 decode | 53.3 tok/s (42.7 ms/forward-step, ~2.3 tok/step MTP) | [d2-chain-fold](../docs/experience/wins/2026-06-13-dsv4-mtp-d2-chain-fold-53.md) |
| B=1 prefill | 23 ms | [decode-6ms-FINAL](../docs/experience/wins/2026-06-08-dsv4-decode-6ms-FINAL-consolidated.md) |
| c=8 batched lane | 45.6 → 67.6 tok/s (+48%) | [batched-flashmla-phaseA](../docs/experience/wins/2026-06-14-dsv4-batched-flashmla-decode-phaseA.md) |

## Snapshot index

| File | Date | Backend · hardware | What it captures |
|---|---|---|---|
| [snapshots/2026-06-14-metal-m4pro-ladder.json](snapshots/2026-06-14-metal-m4pro-ladder.json) | 2026-06-14 | Metal · M4 Pro | 0.8B→35B serve-path decode/TPOT/TTFT ladder (**current**) |
| [snapshots/2026-04-01-metal-qwen3-0.6b-baseline.json](snapshots/2026-04-01-metal-qwen3-0.6b-baseline.json) | 2026-04-01 | Metal · (monolith-era) | Qwen3-0.6B step-driver baseline — **archived**, pre-rewrite, superseded by the ladder above |

## Adding a snapshot

- **Metal local ladder:** `bash scripts/bench_local_metal_all.sh` (serves each
  locally-cached MLX model in turn, c=1, strictly serial), then distil the
  `RESULT:` lines into a dated `snapshots/<date>-<label>.json` and update the
  tables above.
- **Canonical serving benchmark** (CUDA/Metal, produces a wins entry):
  `python3 scripts/bench_throughput.py ... --output bench-output/<label>/bench`
  — raw JSON and CSV land in `bench-output/`;
  copy the headline table into a snapshot here and link the wins entry.
