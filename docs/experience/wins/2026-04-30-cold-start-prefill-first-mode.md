# Cold-Start Prefill-First Mode

## Goal

Diagnosis + optimization: remove decode/prefill interleaving from the first
c=16 cold-start burst before judging whether startup prefill warmup is needed.

## Hypothesis

For a fresh burst with no runnable decode rows, early decode work from the first
completed prefill should not run until the rest of the burst has finished
prefill. The first window should show contiguous `plan=prefill` steps and a
single switch back to decode/mixed after the prefill queue drains.

## Reference

- SGLang documents chunked prefill as the knob that splits large prefills and
  can mix prefill/decode with `--enable-mixed-chunk`, so ARLE should make the
  mixed choice explicit instead of applying it during a cold-start burst.
- vLLM exposes `max_num_partial_prefills` and `max_long_partial_prefills`; its
  scheduler treats concurrent partial prefill as a separate policy surface from
  the token budget.

## Command

Pre-fix profile on `c0f4ca8f`:

```bash
CUDA_HOME=/usr/local/cuda /tmp/arle-target/release/infer \
  --model-path infer/models/Qwen3.5-4B \
  --port 8000 \
  --num-slots 16 \
  --max-seq-len 8192 \
  --chunked-prefill-size 2048 \
  --max-prefill-tokens 16384 \
  --kv-cache-dtype fp8

./scripts/bench_guidellm.sh 2026-04-30-ttft-profile-head-qwen35-fp8 \
  --target http://localhost:8000 \
  --model Qwen/Qwen3.5-4B \
  --processor infer/models/Qwen3.5-4B \
  --fast \
  --trace-interval-ms 200
```

Post-fix exploration:

```bash
CUDA_HOME=/usr/local/cuda /tmp/arle-target/release/infer \
  --model-path infer/models/Qwen3.5-4B \
  --port 8000 \
  --num-slots 16 \
  --max-seq-len 8192 \
  --chunked-prefill-size 2048 \
  --max-prefill-tokens 16384 \
  --kv-cache-dtype fp8

./scripts/bench_guidellm.sh 2026-04-30-firstbatch-qwen35-fp8 \
  --target http://localhost:8000 \
  --model Qwen/Qwen3.5-4B \
  --processor infer/models/Qwen3.5-4B \
  --fast \
  --trace-interval-ms 200
```

Canonical 120s matrix on final HEAD:

```bash
./scripts/bench_guidellm.sh 2026-04-30-head2-<model>-<kv> \
  --target http://127.0.0.1:8000 \
  --model <Qwen/Qwen3-4B|Qwen/Qwen3.5-4B> \
  --processor infer/models/<Qwen3-4B|Qwen3.5-4B> \
  --concurrencies 16 \
  --max-seconds 120 \
  --trace-interval-ms 1000
```

## Environment

- GPU: NVIDIA L4, 23 GiB VRAM
- CUDA: `/usr/local/cuda`
- Commit: `ecb2fe3c` initial first-batch mode,
  `16f7aef8` cohort admission follow-up
- Features: `infer --no-default-features --features cuda`
- Models: `infer/models/Qwen3-4B`, `infer/models/Qwen3.5-4B`
- Scheduler: `num_slots=16`, `max_seq_len=8192`,
  `chunked_prefill_size=2048`, `max_prefill_tokens=16384`,
  `kv_cache_dtype=<fp8|bf16>`

## Results

### Profile Evidence

| Run | First-window behavior | Result |
|---|---|---|
| pre-fix `c0f4ca8f` | `plan=decode+prefill` starts while requests 2-16 still have unfinished prompt chunks | Mixed ticks spend ~410-440ms in decode and ~3.3ms in prefill launch, so early decode blocks the rest of first-burst TTFT. |
| initial post-fix `ecb2fe3c` | First c=16 burst runs contiguous `plan=prefill` for Qwen3.5 fp8 | `first-batch prefill drain complete: elapsed=13503ms rows=48 tokens=65552 decode_rows=16`. |
| follow-up `16f7aef8` | Keeps the cold-start cohort admission-open and records all queued prefills until the waiting queue drains | Qwen3.5 bf16 now reaches `rows=48 tokens=65552 decode_rows=16` before any `decode+prefill`. |

### Exploration Bench

| model | kv | mode | tok/s | TTFT p50 | ITL p50 | status |
|---|---|---|---:|---:|---:|---|
| Qwen3.5-4B | fp8 | pre-fix 30s | n/a | n/a | n/a | server exited before any completed request; trace still captured the bad interleaving. |
| Qwen3.5-4B | fp8 | first-batch 30s | 190.27 output tok/s | 13199.0ms | 52.71ms | clean completion, 16 complete + 16 incomplete requests. |
| Qwen3.5-4B | fp8 | prior headline 120s | 151.76 output tok/s | 13273.2ms | 57.14ms | historical reference from `2026-04-29-bench-guidellm-cuda-l4-headline-summary.md`. |

Raw artefacts:

- Pre-fix trace:
  `bench-output/2026-04-30-2026-04-30-ttft-profile-head-qwen35-fp8/`
- Pre-fix server:
  `bench-output/2026-04-30-ttft-profile-head-qwen35-fp8-server/server.log`
- Post-fix trace:
  `bench-output/2026-04-30-2026-04-30-firstbatch-qwen35-fp8/`
- Post-fix server:
  `bench-output/2026-04-30-firstbatch-qwen35-fp8-server/server.log`

### Canonical Matrix

Clean-HEAD c=16 / 4096 prompt / 256 output / 120s reruns after the cohort
follow-up:

| model | kv | tok/s | TTFT p50 | ITL p50 | GPU SM avg | KV pool tokens | peak KV util | first cold drain | status |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| Qwen3-4B | fp8 | 139.19 | 12628.1ms | 78.06ms | 75.1% | 144,000 | 98.4% | 12.821s, 48 rows, 65,552 tokens, 16 decode rows | valid |
| Qwen3-4B | bf16 | 106.64 | 12137.2ms | 101.29ms | 77.1% | 78,160 | 99.7% | 12.753s, 48 rows, 65,552 tokens, 16 decode rows | valid, 1 incomplete at cutoff |
| Qwen3.5-4B | fp8 | 151.96 | 13040.7ms | 52.56ms | 74.8% | 393,696 | 84.3% | 13.452s, 48 rows, 65,552 tokens, 16 decode rows | valid |
| Qwen3.5-4B | bf16 | 161.60 | 13165.1ms | 54.14ms | 72.9% | 249,136 | 80.6% | 13.451s, 48 rows, 65,552 tokens, 16 decode rows | valid |

Delta vs the 2026-04-29 headline summary:

| model | kv | tok/s Δ | TTFT p50 Δ | ITL p50 Δ |
|---|---|---:|---:|---:|
| Qwen3-4B | fp8 | +0.7% | -4.6% | +7.6% |
| Qwen3-4B | bf16 | +27.0% | -5.9% | -1.4% |
| Qwen3.5-4B | fp8 | +0.1% | -1.8% | -8.0% |
| Qwen3.5-4B | bf16 | +7.5% | +4.3% | -15.5% |

FP8 vs BF16 on final HEAD:

| model | tok/s Δ | TTFT p50 Δ | ITL p50 Δ | note |
|---|---:|---:|---:|---|
| Qwen3-4B | +30.5% | +4.0% slower | -22.9% | fp8 remains the throughput/ITL winner. |
| Qwen3.5-4B | -6.0% | -0.9% | -2.9% | bf16 wins throughput; fp8 wins latency slightly but still has the numerical-default blocker from the fp8 KV spot-check. |

## Problems

- This mode removes cold-start phase interference but does not by itself close
  the Qwen3.5 TTFT gap. The post-fix exploration TTFT p50 is effectively flat
  against the prior headline number.
- The first pre-fix 30s run exited before completion. The server log still
  captured the scheduling defect, but the run cannot be used as a throughput
  baseline.
- The initial `ecb2fe3c` implementation sealed the cohort too early for
  Qwen3.5 bf16 because admission could still be draining while a single
  prefill-cap request had already produced a decode row. Follow-up
  `16f7aef8` keeps the cohort open until waiting admission drains and tracks
  the full `prefill_queue`, not only candidates selected in the current tick.

## Learnings

- Qwen3.5 c=16 TTFT is dominated by the time to finish all prefill chunks. The
  fixed first-batch mode makes the invariant explicit, but the measured TTFT
  remains ~13s because the contiguous prefill drain itself takes ~13.4s.
- The scheduler needs explicit lifecycle logs for special modes; the new
  `first-batch prefill drain started/complete` lines made the profile
  unambiguous.
- The strongest win is ITL, not TTFT. Qwen3.5 bf16 ITL improved 15.5% vs the
  prior headline, and Qwen3.5 fp8 improved 8.0%.

## Delta vs Baseline

| comparison | tok/s Δ | TTFT p50 Δ | ITL p50 Δ | note |
|---|---:|---:|---:|---|
| post-fix 30s vs prior headline Qwen3.5 fp8 120s | +25.4% | -0.6% | -7.8% | Different duration; use only as directional smoke. |
| final Qwen3.5 fp8 120s vs prior headline | +0.1% | -1.8% | -8.0% | Final same-duration comparison. |
| final Qwen3.5 bf16 120s vs prior headline | +7.5% | +4.3% | -15.5% | First-batch ordering improves decode cadence, not cold prefill TTFT. |

## Artefacts

- Qwen3 fp8: `bench-output/2026-04-30-2026-04-30-head2-qwen3-fp8/`
- Qwen3 fp8 server: `bench-output/2026-04-30-2026-04-30-head2-qwen3-fp8-server/server.log`
- Qwen3 bf16: `bench-output/2026-04-30-2026-04-30-head2-qwen3-bf16/`
- Qwen3 bf16 server: `bench-output/2026-04-30-2026-04-30-head2-qwen3-bf16-server/server.log`
- Qwen3.5 fp8: `bench-output/2026-04-30-2026-04-30-head2-qwen35-fp8/`
- Qwen3.5 fp8 server: `bench-output/2026-04-30-2026-04-30-head2-qwen35-fp8-server/server.log`
- Qwen3.5 bf16: `bench-output/2026-04-30-2026-04-30-head2-qwen35-bf16/`
- Qwen3.5 bf16 server: `bench-output/2026-04-30-2026-04-30-head2-qwen35-bf16-server/server.log`
