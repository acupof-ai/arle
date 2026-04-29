# TileLang ON vs OFF — guidellm c=16, CUDA L4, 2026-04-29

## Goal

Diagnosis: measure the actual L4 sm89 impact of default `tilelang-attn` versus
FlashInfer prefill fallback under the current scheduler and fp8 KV fixes.

## Hypothesis

TileLang should mainly affect long prefill. If the scheduler is decode- or
admission-bound at c=16, TileLang ON may not beat OFF despite being enabled in
the default `cuda` feature set.

## Command

TileLang ON build:

```bash
CARGO_TARGET_DIR=/tmp/arle-target CARGO_HOME=/tmp/arle-cargo-home \
CUDA_HOME=/usr/local/cuda ZIG=/tmp/zig-0.15.2/zig-x86_64-linux-0.15.2/zig \
cargo build -p infer --features cuda --release
cp /tmp/arle-target/release/infer /tmp/arle-target/release/infer-tilelang-on
```

TileLang OFF build used a temporary local-only edit removing `tilelang-attn`
from `infer/Cargo.toml`'s `cuda = [...]`, then restored the file after copying
the binary:

```bash
CARGO_TARGET_DIR=/tmp/arle-target CARGO_HOME=/tmp/arle-cargo-home \
CUDA_HOME=/usr/local/cuda ZIG=/tmp/zig-0.15.2/zig-x86_64-linux-0.15.2/zig \
cargo build -p infer --features cuda --release
cp /tmp/arle-target/release/infer /tmp/arle-target/release/infer-tilelang-off
git restore infer/Cargo.toml
```

Server launch, per run:

```bash
/tmp/arle-target/release/infer-tilelang-{on,off} \
  --model-path infer/models/<model> \
  --port 8000 \
  --num-slots 16 \
  --max-seq-len 8192 \
  --kv-cache-dtype fp8
```

Bench command, per run:

```bash
scripts/bench_guidellm.sh <label> \
  --target http://localhost:8000 \
  --model <Qwen/Qwen3-4B|Qwen/Qwen3.5-4B> \
  --processor infer/models/<model> \
  --profile concurrent \
  --concurrencies 16 \
  --max-seconds 120
```

Short-qlen regression:

```bash
scripts/bench_guidellm.sh arle-qwen3-tilelang-on-short32 \
  --target http://localhost:8000 \
  --model Qwen/Qwen3-4B \
  --processor infer/models/Qwen3-4B \
  --profile concurrent \
  --concurrencies 16 \
  --data 'prompt_tokens=32,prompt_tokens_stdev=1,prompt_tokens_min=32,prompt_tokens_max=32,output_tokens=16,output_tokens_stdev=1,output_tokens_min=16,output_tokens_max=16' \
  --max-seconds 30
```

## Environment

- Backend: ARLE CUDA
- Hardware: NVIDIA L4, 23.66 GB VRAM, driver 580.82.07
- Commit: `2deacc96` plus the fp8 KV fix commits already on `main`
- Feature set ON: `cuda` default, includes `tilelang-attn`
- Feature set OFF: local bench-only binary with `cuda` excluding
  `tilelang-attn`; no Cargo feature change committed
- KV dtype: fp8 paged KV, BF16 contiguous prefill cache
- SGLang baseline: reused
  `docs/experience/wins/2026-04-29-bench-guidellm-sglang-align-c16.md`

## Results

| model | attention path | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | out tok/s | total tok/s | GPU SM% | KV pool | peak KV util | completed in/out | incomplete in/out |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| Qwen3-4B | TileLang ON | 13299.4ms | 13586.5ms | 72.55ms | 93.29ms | 138.01 | 2346.71 | 100.0 | 148,256 tokens / 11.5 GB | 97.0% | 262,208 / 16,384 | 0 / 0 |
| Qwen3-4B | TileLang OFF | 12435.9ms | 13282.9ms | 72.44ms | 84.52ms | 143.50 | 2440.13 | 100.0 | 148,256 tokens / 11.5 GB | 97.0% | 258,111 / 16,128 | 4,097 / 256 |
| Qwen3.5-4B | TileLang ON | 12395.9ms | 14336.3ms | 60.78ms | 97.87ms | 151.03 | 2568.04 | 99.6 | 398,624 tokens / 8.2 GB | 82.3% | 262,208 / 16,384 | 65,536 / 310 |
| Qwen3.5-4B | TileLang OFF | 11278.1ms | 13830.9ms | 64.00ms | 97.32ms | 156.19 | 2655.80 | 99.4 | 398,624 tokens / 8.2 GB | 82.3% | 262,208 / 16,384 | 65,536 / 360 |

Delta, TileLang ON vs OFF:

| model | out tok/s | TTFT p50 | ITL p50 | reading |
|---|---:|---:|---:|---|
| Qwen3-4B | -3.8% | +6.9% slower | +0.2% slower | FlashInfer path wins this c=16 shape |
| Qwen3.5-4B | -3.3% | +9.9% slower | -5.0% faster | TileLang improves steady decode-adjacent ITL but loses more in TTFT / total throughput |

SGLang baseline rows, reused without rerun:

| model | backend | out tok/s | TTFT p50 | ITL p50 | source |
|---|---|---:|---:|---:|---|
| Qwen3-4B | SGLang | 133.74 | 8375.7ms | 86.20ms | `2026-04-29-bench-guidellm-sglang-align-c16.md` |
| Qwen3.5-4B | SGLang | 151.06 | 7559.2ms | 75.77ms | `2026-04-29-bench-guidellm-sglang-align-c16.md` |

## Short-Qlen Regression

TileLang ON, Qwen3-4B, c=16 / prompt 32 / output 16 / 30s:

| TTFT p50 | ITL p50 | out tok/s | total tok/s | peak active | peak prefill_queue | peak KV util | accounting |
|---:|---:|---:|---:|---:|---:|---:|---|
| 161.1ms | 39.95ms | 341.50 | 1045.86 | 16 | 0 | 0.5% | 21,120 input / 10,240 output completed; 0 errors |

`rg -n "NaN|nan|!!!!!!"` over the server log and GuideLLM log found no hits.
A direct short completion also did not reproduce the old `"!!!!!!"` output
shape. The text was not used as a quality signal; this gate only checks the
old short-qlen NaN-token regression.

## Status

- `tilelang-attn` remains in the default `cuda` feature after `47bad713`.
- The old short-qlen NaN issue stays fixed in this run.
- `tilelang-decode-hd256` is still broken at build time on sm89. Error entry:
  `docs/experience/errors/2026-04-29-tilelang-decode-hd256-sm89-build.md`.
- Based on these numbers, TileLang prefill is not a win for the current L4
  c=16 scheduler envelope. Keep it enabled only if another target shape or
  profile demonstrates a real gain; otherwise consider reverting default
  `tilelang-attn` in a separate decision commit.

## Problems

- Runs use the requested fixed-concurrency 120s shape, so the wrapper marks
  them exploration mode instead of seeding a canonical sweep entry.
- The Qwen3.5 rows have incomplete tail requests at the 120s stop boundary;
  completed output tokens are still equal across ON/OFF.
- Qwen3 OFF had one incomplete request, which makes its raw completed token
  accounting slightly lower than ON. The throughput gap still favours OFF.
- No NaN-token regression appeared in the short prompt run.

## Learnings

- At c=16 / 4096 / 256 on L4, the scheduler envelope is not helped by the
  TileLang HD128 prefill path. FlashInfer prefill gives better TTFT and total
  output throughput on both Qwen3 and Qwen3.5.
- Qwen3.5's prefill-row cap makes TTFT sensitive to long-prefill latency, so
  a prefill kernel must win clearly to offset serialized admission.
- Short-prompt correctness should remain a separate gate from long-prompt
  throughput; long benches did not catch the original `qlen < 64` bug.

## Artefacts

- Qwen3 ON: `bench-output/2026-04-29-arle-qwen3-tilelang-on/`
- Qwen3 ON server: `bench-output/2026-04-29-arle-qwen3-tilelang-on-server/server.log`
- Qwen3 ON dmon: `bench-output/2026-04-29-arle-qwen3-tilelang-on-dmon/gpu_dmon.csv`
- Qwen3 OFF: `bench-output/2026-04-29-arle-qwen3-tilelang-off/`
- Qwen3 OFF server: `bench-output/2026-04-29-arle-qwen3-tilelang-off-server/server.log`
- Qwen3 OFF dmon: `bench-output/2026-04-29-arle-qwen3-tilelang-off-dmon/gpu_dmon.csv`
- Qwen3.5 ON: `bench-output/2026-04-29-arle-qwen35-tilelang-on/`
- Qwen3.5 ON server: `bench-output/2026-04-29-arle-qwen35-tilelang-on-server/server.log`
- Qwen3.5 ON dmon: `bench-output/2026-04-29-arle-qwen35-tilelang-on-dmon/gpu_dmon.csv`
- Qwen3.5 OFF: `bench-output/2026-04-29-arle-qwen35-tilelang-off/`
- Qwen3.5 OFF server: `bench-output/2026-04-29-arle-qwen35-tilelang-off-server/server.log`
- Qwen3.5 OFF dmon: `bench-output/2026-04-29-arle-qwen35-tilelang-off-dmon/gpu_dmon.csv`
- Short-qlen: `bench-output/2026-04-29-arle-qwen3-tilelang-on-short32/`
- Short-qlen server: `bench-output/2026-04-29-arle-qwen3-tilelang-on-short-server/server.log`

## Delta vs Baseline

This entry is a paired ON/OFF diagnosis rather than a new production baseline.
Use the TileLang OFF row as the within-entry baseline for the delta table above.
