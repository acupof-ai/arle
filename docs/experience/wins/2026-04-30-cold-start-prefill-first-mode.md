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

Canonical 120s matrix is pending below and will be filled from clean HEAD.

## Environment

- GPU: NVIDIA L4, 23 GiB VRAM
- CUDA: `/usr/local/cuda`
- Commit: runtime change committed with this entry
- Features: `infer --no-default-features --features cuda`
- Model: `infer/models/Qwen3.5-4B`
- Scheduler: `num_slots=16`, `max_seq_len=8192`,
  `chunked_prefill_size=2048`, `max_prefill_tokens=16384`,
  `kv_cache_dtype=fp8`

## Results

### Profile Evidence

| Run | First-window behavior | Result |
|---|---|---|
| pre-fix `c0f4ca8f` | `plan=decode+prefill` starts while requests 2-16 still have unfinished prompt chunks | Mixed ticks spend ~410-440ms in decode and ~3.3ms in prefill launch, so early decode blocks the rest of first-burst TTFT. |
| post-fix | First c=16 burst runs contiguous `plan=prefill`; no `decode+prefill` before drain completion | `first-batch prefill drain complete: elapsed=13503ms rows=48 tokens=65552 decode_rows=16`. |

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

Pending clean-HEAD c=16 / 4096 / 256 / 120s reruns:

| model | kv | tok/s | TTFT p50 | ITL p50 | status |
|---|---|---:|---:|---:|---|
| Qwen3-4B | fp8 | pending | pending | pending | pending |
| Qwen3-4B | bf16 | pending | pending | pending | pending |
| Qwen3.5-4B | fp8 | pending | pending | pending | pending |
| Qwen3.5-4B | bf16 | pending | pending | pending | pending |

## Problems

- This mode removes cold-start phase interference but does not by itself close
  the Qwen3.5 TTFT gap. The post-fix exploration TTFT p50 is effectively flat
  against the prior headline number.
- The first pre-fix 30s run exited before completion. The server log still
  captured the scheduling defect, but the run cannot be used as a throughput
  baseline.

## Learnings

- Qwen3.5 c=16 TTFT is dominated by the time to finish all prefill chunks, not
  just decode/prefill interleaving. Removing early decode is still the right
  scheduler invariant for the first burst, but the next TTFT work should look
  at prefill compute and prefill first-time planning/autotune.
- The scheduler needs explicit lifecycle logs for special modes; the new
  `first-batch prefill drain started/complete` lines made the profile
  unambiguous.

## Delta vs Baseline

| comparison | tok/s Δ | TTFT p50 Δ | ITL p50 Δ | note |
|---|---:|---:|---:|---|
| post-fix 30s vs prior headline Qwen3.5 fp8 120s | +25.4% | -0.6% | -7.8% | Different duration; use only as directional smoke. |

