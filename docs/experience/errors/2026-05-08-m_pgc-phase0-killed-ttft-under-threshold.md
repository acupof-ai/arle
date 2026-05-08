# M_pf-Graph Phase 0 Killed — TTFT Under License Threshold

## Context

M_pf-graph Phase 0 tested an opt-in Qwen3 prefill CUDA Graph path:

- `INFER_PREFILL_GRAPH=1`
- one 2048-token prefill bucket
- eager fallback preserved
- prefill admission clamped to one 2048-token request

Correctness gates passed before the benchmark:

- `cargo check --release -p infer --features cuda`
- `cargo clippy --release -p infer --features cuda -- -D warnings`
- `INFER_PREFILL_GRAPH=1 cargo test --release -p infer --features cuda --test e2e`
- `INFER_PREFILL_GRAPH=1 cargo test --release -p infer --features cuda --test greedy_consistency`

The user-provided launch shape used auto KV cache mode, which resolves to FP8 on this machine and would force `kv-format` eager fallback because Phase 0 only supported BF16 paged KV. To measure the graph path rather than fallback, the license run used explicit BF16:

```bash
INFER_PREFILL_GRAPH=1 CUDA_HOME=/opt/cuda TORCH_CUDA_ARCH_LIST=8.9 \
  ./target/release/infer \
    --model-path infer/models/Qwen3-4B \
    --port 8000 \
    --num-slots 8 \
    --max-seq-len 5120 \
    --kv-cache-dtype bf16
```

Bench command:

```bash
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
  scripts/bench_guidellm.sh m_pgc-phase0-2048bucket-c4 \
  --concurrencies 4 --max-seconds 120 --warmup 10 \
  --data 'prompt_tokens=4096,prompt_tokens_stdev=1,prompt_tokens_min=4096,prompt_tokens_max=4096,output_tokens=256,output_tokens_stdev=1,output_tokens_min=256,output_tokens_max=256'
```

Raw artifacts:

- `bench-output/2026-05-08-m_pgc-phase0-2048bucket-c4/benchmarks.csv`
- `bench-output/2026-05-08-m_pgc-phase0-2048bucket-c4/headline_table.md`
- `bench-output/2026-05-08-m_pgc-phase0-2048bucket-c4/service_stats_trace_summary.md`
- `bench-output/2026-05-08-m_pgc-phase0-2048bucket-c4/guidellm.log`
- `bench-output/2026-05-08-m_pgc-phase0-2048bucket-c4/command.txt`

## Results

License threshold from `docs/plans/M_pf-graph-prefill-capture.md`: proceed only if TTFT p50 improves by at least 10% over ARLE pre-Phase0, with a strong proceed at 25%.

| Engine | TTFT p50 | TTFT mean | TTFT p99 | ITL p50 | out tok/s | E2E mean | conc p50 | Rank |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ARLE pre-Phase0 (`786a20a`) | 1976.4 ms | n/a | n/a | 19.27 ms | 153.83 | n/a | 4 | control |
| ARLE Phase0 graph opt-in | 1961.2 ms | 1956.8 ms | 1997.3 ms | 25.58 ms | 122.95 | 8.48 s | 4 | killed |
| vLLM s8 (`d13d2b3`) | 1177 ms | n/a | n/a | 19.4 ms | 159.1 | n/a | 4 | reference |
| SGLang 0.5.11 (`12c4c86`) | 972.9 ms | 1117 ms | n/a | 19.44 ms | 164.3 | 6.29 s | 4 | #2 |

Delta:

| Comparison | TTFT p50 delta | ITL p50 delta | out tok/s delta |
|---|---:|---:|---:|
| Phase0 vs ARLE pre-Phase0 | -0.8% | +32.7% | -20.1% |
| Phase0 vs SGLang | +101.6% slower | +31.6% | -25.2% |
| Phase0 vs world-first target (`<=748 ms`) | +162.2% over target | n/a | n/a |

TTFT std for the single scout run was 25.7 ms, about 1.3% of mean, so the run was stable enough for the kill decision.

## Evidence

Resolved scheduling envelope:

```text
Scheduling envelope (resolved | SGLang-equiv): max_num_batched_tokens=16384 | 16384, chunked_prefill_size=2048 | 2048, max_prefill_tokens=2048 | 16384, mem_fraction_static=0.85 | 0.85, max_slots=8 | (n/a — SGLang has no fixed cap)
```

Service trace:

```text
Peak active: 4
Peak running_batch: 4
Peak prefill_queue: 3
Plan labels: idle=68941, decode=3577, prefill=179, split=0, mixed=0
Peak kv_util: 100.0%
Prefix hit rate: peak 0.0%, q75 0.0%
```

Graph log counts from `/tmp/infer-mpgc-phase0.log`:

```text
prefill graph capture key: 30
prefill graph fallback reason=token-count: 59
prefix cache pressure fallback: 14

15 PrefillGraphKey { token_count: 2048, start_pos: 0, num_pages: 128, page_size: 16 }
15 PrefillGraphKey { token_count: 2048, start_pos: 2048, num_pages: 256, page_size: 16 }
```

There were no `graph capture failed` warnings.

## Root Cause

Phase 0 did not remove enough launch overhead to matter for the real 4097-token request shape. The implementation captured valid 2048-token chunks, but the workload still had three prefill parts per request: 2048 + 2048 + 1. The final 1-token tail fell back to eager, and the graph cache held only one key at a time, so the runtime recaptured when alternating between `start_pos=0` and `start_pos=2048` across request groups.

The opt-in clamp also serialized prefill admission to one 2048-token request at a time. The trace shows 179 `prefill` scheduler plans for the 120s window, so Phase 0 traded potential launch savings for more scheduler-level prefill steps. On this 16 GiB GPU, forcing BF16 to exercise graph support also increased KV pressure relative to the production auto-FP8 baseline, causing prefix-cache pressure fallbacks during the run.

## Fix

Kill Phase 0. The runtime diff was restored and not committed.

Do not promote `INFER_PREFILL_GRAPH=1` from this substrate. A future graph attempt needs, at minimum:

- multi-key or multi-bucket graph caching instead of a single last-key cache
- support for the exact long-context request decomposition, including tails or a scheduler shape that avoids tails
- FP8 paged-KV graph support or a separate apples-to-apples BF16 baseline
- no default prefill envelope clamp that serializes c=4 long-context traffic
- nsys evidence that CUDA launch overhead drops before re-running the throughput gate

## Rule

For CUDA Graph optimizations, do not license on "capture exists" evidence. The license gate must prove reuse on the real request shape and must not introduce scheduler envelope changes that dominate the measured effect.
