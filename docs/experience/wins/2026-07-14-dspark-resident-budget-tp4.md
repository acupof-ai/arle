# DSv4 DSpark resident-weight budget — H20 TP=4

> Status: Shipped

## SLO-shape probed? N — 32-token prompt, batch 16

This licenses the DSpark TP=4 concurrency fix, not a default flip. Long-prompt
routing remains governed by the existing 64-token crossover measurement.

## Roofline check

Deferred: this change removes a false host-side VRAM reserve and a TP-unsafe
dispatch path; it does not change a GPU kernel. Wall-clock GuideLLM is the
license. Kernel roofline is unchanged.

## Goal

Replace DSpark's checkpoint-size KV reserve with resident VRAM truth and recover
TP=4 high-concurrency throughput without changing c=1 behavior.

## Hypothesis

The 19.9 GB/rank reserve is false because routed draft experts are already EP4.
Loading draft weights before KV planning should expose the measured 4,960 MB/rank
footprint, raise slots above one, and let the existing target batch path scale.

## Command

```bash
INFER_TP_SIZE=4 INFER_CUDA_DEVICES=3,4,5,6 \
  arle serve --backend cuda \
  --model-path /host/DeepSeek-V4-Flash-FP8 \
  --spec-type dspark \
  --mtp-draft-model /host/DeepSeek-V4-Flash-DSpark-draft-fp8 \
  --dspark-max-prompt-tokens 64 --comm-backend nccl --port 8799 \
  --max-running-requests 16 --max-prompt-tokens 8448 \
  --max-total-tokens 9216 --kv-dram 50% \
  --kv-disk /host/arle-dspark-kv-l3 --kv-disk-limit 100GiB

GUIDELLM__MP_CONTEXT_TYPE=forkserver guidellm benchmark run \
  --target http://127.0.0.1:8799 \
  --model /host/DeepSeek-V4-Flash-FP8 \
  --processor /host/DeepSeek-V4-Flash-FP8 \
  --backend openai_http \
  --backend-kwargs '{"validate_backend":"/v1/models","request_format":"/v1/completions"}' \
  --profile concurrent --rate 1 --rate 4 --rate 8 --rate 16 \
  --data /host/dspark_natural_32in_128out.jsonl \
  --max-seconds 60 --random-seed 20260416 \
  --disable-console-interactive \
  --outputs benchmarks.json --outputs benchmarks.csv
```

## Environment

- Backend/model: CUDA, DeepSeek-V4-Flash FP8 + DSpark FP8 draft
- Hardware: 4x H20 96 GB, TP=4, GPUs 3–6
- CUDA/driver: CUDA 12.9, driver 535.161.08
- Binary SHA-256: `2dfac846a5c2b09459fa95d87317cc32bb6f0130cc7fd8fe36155cd324e83f00`
- GuideLLM 0.6.0; 20 natural prompts, 32 input + 128 output tokens, greedy
- L2: 50% deployment DRAM, resolved to about 27.56 GB/rank
- L3: 100 GiB deployment cap, resolved to 25 GiB/rank
- Profiling off

## Results — checkpoint and VRAM truth

| Item | Measured |
|---|---:|
| Draft checkpoint | 18.555 GiB |
| Routed experts | 18.001 GiB |
| Other tensors | 567.61 MiB |
| Resident draft delta/rank | 4,960 MiB |
| Post-trim free VRAM/rank | 17,053 MiB |
| Per-slot state | 450 MiB |
| Affordable/final slots | 34 / 33 |
| Previous final slots | 1 |

The loader already keeps only `local_expert_start..local_expert_end()` and uses
the target TP shards for large attention projections. The remaining plausible
new shards total about 185 MiB/rank: shared expert 54 MiB, `main_proj` 36 MiB,
and Markov/vocabulary heads 94.7 MiB. Each adds a new collective or changes sum
order, so none was licensed.

## Results — fixed-concurrency GuideLLM

Throughput is completed output tokens divided by each measured run duration.
Every row completed 20/20 requests with zero errors and 2,560 output tokens.

| c | duration s | req/s | TTFT p50/p99 ms | ITL p50/p99 ms | request p50/p99 s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 56.845 | 0.352 | 243.88 / 261.78 | 21.09 / 23.16 | 2.895 / 3.193 |
| 4 | 31.974 | 0.626 | 509.17 / 705.55 | 46.42 / 48.11 | 6.399 / 6.591 |
| 8 | 21.217 | 0.943 | 966.22 / 1285.82 | 50.20 / 54.60 | 7.342 / 7.709 |
| 16 | 18.097 | 1.105 | 2168.85 / 2169.82 | 76.87 / 83.83 | 11.932 / 11.933 |

c=4 repeated: 20/20 complete,
zero errors, and zero prefix hits.

## Results — service counters

| Counter | Value |
|---|---:|
| Peak active requests | 16 |
| Peak queue depth | 0 |
| L2 host pages after sweep | 149 |
| L3 disk pages after sweep | 0 |
| Spec drafted/accepted during sweep | 1,136 / 885 (77.9%) |
| Lockstep stalls | 0 |

L3 was attached on every rank but not filled: this short unique-prompt workload
used only 149 host pages. The result therefore measures enabled-tier overhead,
not an L3-hit speedup.

## Correctness

The pre-bench curl returned 16 coherent tokens:
` of Rayleigh scattering. The sky is blue because of Rayleigh scattering. The sky is`.

## Problems

- The first B>1 implementation ran DSpark rows sequentially. TP ranks diverged
  in collective order and stalled at coordinator tick 1092. It was deleted.
- DSpark has no TP-safe batched verify lane. A request that joins B>1 is marked
  ineligible for the rest of its lifetime and uses the existing target batch
  path; this prevents stale draft reuse after the batch shrinks.
- A final rebuild retry hit an unrelated TileLang AOT internal error for
  `tilelang_batch_prefill_paged_hd64_q16_kv1_run` (14 planned vs 15 pipeline
  stages). The published source matches the previously successful pod build and
  measured binary; no TileLang code was changed.
- Local CUDA/no-CUDA type-check passed. Clippy `-D warnings` is blocked outside
  this diff by one unused variable in `qwen35.rs` and two
  `manual_is_multiple_of` findings in `quant_linear.rs`. Two earlier pod release
  builds of this exact runtime change passed.

## Cold-load evidence

| State | Base prefetch | Draft prefetch |
|---|---:|---:|
| Cold local disk | 294.0 GB / 1561.8 s = 0.19 GB/s | 19.9 GB / 168.6 s = 0.12 GB/s |
| Warm page cache | 294.0 GB / 35.5 s = 8.28 GB/s | 19.9 GB / 1.2 s = 17.05 GB/s |

Only rank zero prefetched. The cold 26-minute base read is the measured local
disk floor, not the former four-rank 1.65 TB amplification.

## Learnings

- Budget from synchronized resident state, not checkpoint bytes. Quantization,
  EP/TP sharding, and load-time scratch make file size a bad VRAM estimator.
- A per-row loop is not a TP batch: collective order must remain identical on
  every rank. Fall back to one proven batched path until batched verify exists.
- Draft sharding is already doing the high-value work. Do not add collectives to
  save 185 MiB/rank when deleting a false 19.9 GB reserve recovers 32 slots.

## Delta vs baseline

Baseline: same binary predecessor, model, dataset, seed, and flags except the
resident-weight budget and L2/L3 flags. It was limited to one active slot.

| c | baseline TTFT p50 ms | now TTFT p50 ms |
| ---: | ---: | ---: |
| 1 | 234.20 | 243.88 |
| 4 | 8466.96 | 509.17 |
| 8 | 17614.49 | 966.22 |
| 16 | 27221.12 | 2168.85 |

## Artefacts

- Baseline: `/host/arle-dspark-bench/bench-output/2026-07-14-dspark-budget-before-c1-4-8-16/`
- Final: `/host/arle-dspark-bench/bench-output/2026-07-14-dspark-resident-budget-l2-l3-c1-4-8-16/`
- c=4 repeat: `/host/arle-dspark-bench/bench-output/2026-07-14-dspark-resident-budget-l2-l3-c4-repeat/`
- Server log: `/host/dspark_final_server.log`
