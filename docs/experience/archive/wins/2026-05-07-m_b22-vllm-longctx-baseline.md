# M_b.2.2 — vLLM longctx 4k/8k baseline before TileLang split-KV

## Priority & ROI

**Priority**: P0 baseline for M_b.2.2 task #13. The planned
TileLang HD128 BF16 split-KV work is only justified if the long-context
gap is anchored against a same-shape #2 engine baseline.

**ROI basis**: vLLM is the current local reference for longctx
4k/c=4 and 8k/c=4. M_b.2.2 acceptance should compare ARLE after
split-KV against these numbers and the P0' ARLE baseline.

**Negative case**: if M_b.2.2 does not beat the ARLE baseline, or
does not materially narrow the vLLM TTFT/throughput gap, the
split-KV hypothesis is wrong for this shape and should be abandoned
or moved to an errors entry.

**Kill criteria for M_b.2.2**: wall-time / GuideLLM numbers after the
TileLang split-KV patch must improve vs P0' baseline. If the patch is
not faster, do not ship the runtime route.

## Goal

Baseline: capture vLLM longctx 4k/c=4 and 8k/c=4 before starting
TileLang HD128 BF16 split-KV implementation.

## Hypothesis

vLLM should reproduce the earlier local references:

- 4k/c=4 TTFT near 1177 ms
- 8k/c=4 TTFT near 2367 ms

These numbers define the post-M_b.2.2 target gap.

## Command

Server:

```bash
scripts/vllm_serve_control.sh --max-num-seqs 8 --max-model-len 12288
```

4k/c=4:

```bash
PATH=.venv/bin:$PATH scripts/bench_guidellm.sh m_b22-vllm-longctx-4k-c4 \
  --concurrencies 4 --max-seconds 60 --warmup 10 \
  --data 'prompt_tokens=4096,prompt_tokens_stdev=1,prompt_tokens_min=4096,prompt_tokens_max=4096,output_tokens=256,output_tokens_stdev=1,output_tokens_min=256,output_tokens_max=256'
```

8k/c=4:

```bash
PATH=.venv/bin:$PATH scripts/bench_guidellm.sh m_b22-vllm-longctx-8k-c4 \
  --concurrencies 4 --max-seconds 60 --warmup 10 \
  --data 'prompt_tokens=8192,prompt_tokens_stdev=1,prompt_tokens_min=8192,prompt_tokens_max=8192,output_tokens=256,output_tokens_stdev=1,output_tokens_min=256,output_tokens_max=256'
```

## Environment

- **Backend:** vLLM OpenAI server
- **Model:** Qwen3-4B BF16, `infer/models/Qwen3-4B`
- **Hardware:** RTX 4070 Ti SUPER 16 GiB, CUDA 13.2, sm_89
- **vLLM launch:** `--max-num-seqs 8 --max-model-len 12288
  --dtype bfloat16 --kv-cache-dtype fp8 --attention-backend TRITON_ATTN
  --gpu-memory-utilization 0.85`
- **Commit:** `c219434`
- **GuideLLM:** fixed concurrency, c=4, 60 s, 10 s warmup,
  seed `20260416`

## Results

| workload | TTFT p50 | TTFT p99 | ITL p50 | ITL p95 | out tok/s | total tok/s | in tok/s | total in | total out | req/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4k/c=4 | 1174.9 ms | 1426.9 ms | 18.84 ms | 20.59 ms | 159.17 | 2706.49 | 2826.97 | 135201 | 8448 | 0.64 |
| 8k/c=4 | 2361.5 ms | 5017.5 ms | 26.71 ms | 30.33 ms | 104.74 | 3456.89 | 3839.35 | 180246 | 5632 | 0.44 |

Service-side stats are `n/a` because vLLM does not expose ARLE's
`/v1/stats` telemetry. GuideLLM request accounting is the source of
truth for this baseline.

## Delta vs ARLE Baselines

| workload | ARLE baseline | vLLM now | delta |
|---|---:|---:|---:|
| 4k/c=4 TTFT p50 | 1976.4 ms (`p0prime-default-split-c4`) | 1174.9 ms | vLLM 1.68x faster |
| 4k/c=4 out tok/s | 153.83 | 159.17 | vLLM +3.5% |
| 8k/c=4 TTFT p50 | 4961 ms (`c63c31c` F4-Small) | 2361.5 ms | vLLM 2.10x faster |
| 8k/c=4 out tok/s | 92.2 | 104.74 | vLLM +13.6% |

## Problems

- The first attempt ran 4k and 8k under one shell with two
  back-to-back wrapper calls. The 4k run completed, then the 8k run
  hit the wrapper's stale serial lock. I removed
  `bench-output/.bench_guidellm.lock`, restarted the same vLLM
  configuration, and reran only 8k/c=4.
- 4k/c=4 had incomplete accounting at the end of the fixed 60 s
  window (`12291` input tokens, `756` output tokens incomplete).
  Completed-request latency and throughput remain comparable to the
  earlier vLLM reference.
- vLLM server telemetry is unavailable, so this entry cannot report
  plan labels, KV utilization, or queue depth.

## Learnings

- The 4k/c=4 local vLLM reference is stable: TTFT p50 `1174.9 ms`,
  matching the prior `~1177 ms` note.
- The 8k/c=4 reference is stable: TTFT p50 `2361.5 ms`, matching the
  prior `2367 ms` control.
- M_b.2.2 should be judged first by ARLE-vs-ARLE speedup, then by
  narrowing these vLLM gaps. A pure code-unification split-KV patch
  that is not faster fails the current P0 goal.

## Artefacts

- 4k raw: `bench-output/2026-05-07-m_b22-vllm-longctx-4k-c4/`
- 4k server log:
  `bench-output/server-logs/2026-05-07T19-38-39-m_b22-vllm-baseline.log`
- 8k raw: `bench-output/2026-05-07-m_b22-vllm-longctx-8k-c4/`
- 8k server log:
  `bench-output/server-logs/2026-05-07T19-40-43-m_b22-vllm-longctx-8k-c4.log`

## Rule

For M_world1 work, competitor baselines must land before runtime
patches. If a patch cannot beat the current ARLE baseline after the
competitor baseline is known, do not ship it for strategic reasons.
