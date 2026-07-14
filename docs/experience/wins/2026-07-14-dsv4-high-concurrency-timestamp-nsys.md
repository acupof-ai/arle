# DSv4 high-concurrency nsys — B16 raises saturated throughput 33%, 80% of launch time overlaps GPU

## Goal

**Diagnosis.** Measure c=8/c=16 throughput and prove the host/GPU order of the TP=4 layer-major decode path with timestamps.

## Hypothesis

Higher concurrency improves aggregate throughput through GPU batch efficiency; most host launch time overlaps GPU execution.

## Command

The server command, dataset, and seed match the [resident-budget benchmark](2026-07-14-dspark-resident-budget-tp4.md). Each fixed-concurrency run used 5 warmups and 20 measured requests under a CUDA/NVTX/OSRT capture:

```bash
ARLE_DSV4_NVTX=1 INFER_TP_SIZE=4 INFER_CUDA_DEVICES=3,4,5,6 \
  arle serve --backend cuda \
  --model-path /host/DeepSeek-V4-Flash-FP8 \
  --spec-type dspark \
  --mtp-draft-model /host/DeepSeek-V4-Flash-DSpark-draft-fp8 \
  --dspark-max-prompt-tokens 64 --comm-backend nccl --port 8799 \
  --max-running-requests 16 --max-prompt-tokens 8448 \
  --max-total-tokens 9216 --kv-dram 50% \
  --kv-disk /host/arle-dspark-kv-l3 --kv-disk-limit 100GiB

nsys launch --session-new dspark-c${c} -t cuda,nvtx,osrt -- arle ...
nsys start --session dspark-c${c}
guidellm benchmark run --profile concurrent --rate "$c" \
  --data /host/dspark_natural_32in_128out.jsonl \
  --max-seconds 60 --random-seed 20260416 ...
nsys stop --session dspark-c${c}

python3 scripts/analyze_nsys_overlap.py trace.sqlite > overlap.json
```

The analyzer uses `dsv4/embed` and `dsv4/lm_head_sample_batched` NVTX timestamps as tick boundaries. It joins CUDA Runtime launches to kernels by process and `correlationId`, then unions intervals before measuring GPU busy, host launch, and their intersection.

## Environment

- Backend/model: CUDA, DeepSeek-V4-Flash FP8 + DSpark FP8 draft
- Hardware: 4x H20 96 GB, TP=4, GPUs 3–6
- CUDA toolkit/driver: 12.9 / 535.161.08
- Nsight Systems: 2026.3.1.157
- Binary SHA-256: `2dfac846a5c2b09459fa95d87317cc32bb6f0130cc7fd8fe36155cd324e83f00`
- GuideLLM 0.6.0; 20 natural prompts, 32 input + 128 output tokens, greedy
- L2/L3 enabled: c8 resolved L2 to 12.12 GB/rank; c16 resolved L2 to zero because host `MemAvailable` fell below the reserve; L3 was 25 GiB/rank
- Capture starts after engine-ready, so weight prefetch and model load are excluded

## Results — request wall clock

Every row completed 20/20 requests, produced 2,560 output tokens, returned coherent text, and had zero errors.

| c | nsys out tok/s | no-nsys out tok/s | observer tax | TTFT p50/p99 ms | ITL p50/p99 ms | request p50/p99 s |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 102.18 | 120.66 | -15.3% | 1003.55 / 2897.36 | 59.10 / 69.95 | 8.510 / 10.425 |
| 16 | 120.38 | 141.46 | -14.9% | 3749.96 / 3751.30 | 88.63 / 103.07 | 15.006 / 15.008 |
| delta | +17.8% | **+17.2%** | +0.4 pp | +273.7% | +50.0% | +76.3% |

The observer tax is matched within 0.4 pp, so the relative nsys comparison is usable. Absolute throughput comes from the no-nsys benchmark.

## Results — timestamp critical path

The 20-request workload drains in waves. c8 contained 381 batched ticks: B3/B4/B7/B8 = 12/115/11/243. c16 contained 257: B2/B3/B4/B15/B16 = 5/8/117/6/121. Thus aggregate c16 includes one B16 wave followed by a B4 tail; it is not a pure saturated-B16 number.

Rank-0 p50, contiguous ticks below 150 ms; percentage rows use summed intervals:

| Metric | full B8 | full B16 | B8→B16 |
|---|---:|---:|---:|
| Tick wall | 56.444 ms | 84.619 ms | +49.9% |
| GPU kernel busy union | 42.112 ms | 66.714 ms | +58.4% |
| GPU idle | 14.332 ms | 17.905 ms | +24.9% |
| Host launch API union | 14.746 ms | 26.269 ms | +78.2% |
| Launch/GPU intersection | 11.238 ms | 21.149 ms | +88.2% |
| Host launch outside GPU busy intervals | 3.485 ms | 5.174 ms | +48.5% |
| Launch time covered by GPU | 75.84% | 79.60% | +3.76 pp |
| Launch outside GPU busy / tick wall | **6.30%** | **6.27%** | -0.03 pp |
| Implied saturated throughput | 141.7 tok/s | 189.1 tok/s | **+33.4%** |

`B / tick_wall` is the timestamp-derived full-batch rate under nsys, not completed-request throughput. It explains why doubling batch does not double throughput: B16 does twice the tokens in 1.50x the wall.

The correlation join matched 1,413,336 c8 and 1,027,817 c16 rank-0 kernels. Kernel start followed launch end for 99.9880% and 99.9873%. Queue delay p50/p90/p99 was 5.869/8.649/23.725 ms at c8 and 4.756/10.089/12.513 ms at c16. The host therefore queues most work ahead of execution; summed CUDA API time is not the same as non-overlapped wall.

This proves temporal overlap, not counterfactual speedup. A launch hidden under an earlier kernel may still gate its dependent kernel and the tick end. Only a B16 graph/preallocation A/B or a dependency-aware critical-path analysis can measure recoverable host time; 6.27% is **not** an optimization ceiling.

The repeated timestamp order is:

```text
embed
  -> 43 x (batched mHC attention -> all-gather -> FlashMLA
           -> attention all-reduce -> grouped MoE -> MoE all-reduce)
  -> per-row head HC -> lm_head -> sample
  -> next scheduler tick
```

The final head is still a row loop at `crates/infer-cuda/src/dsv4.rs:3097`. Its full-batch NVTX host range p50 was 10.945 ms at B8 and 15.810 ms at B16: +44.4%, still sub-linear versus 2x rows.

## Results — GPU work mix

These are nsys kernel-duration sums across all ranks, not additive critical wall:

| Bucket | c8 share | c16 share |
|---|---:|---:|
| Grouped MoE SwiGLU + down | 27.9% | **31.1%** |
| GEMV family | about 20.2% | about 22.6% |
| NCCL all-reduce + all-gather | 14.2% | 12.7% |
| DeepGEMM pack/SwiGLU quantize | 6.1% | 6.2% |

NVTX GPU projection for batched MLA rose from 0.567 to 0.681 ms/layer average (+20.1%); its median rose from 0.501 to 0.571 ms (+14.0%). MoE is the largest measured GPU-work bucket, but only an operator A/B can turn that work share into recoverable request wall.

## Service counters and cache

The measured load added 2,576 generated tokens at c8 and c16 after including one correctness request. Prefix reuse was zero: 21 misses, zero resident/L2/L3 hits, zero promotions, and zero fetch wait. c8 ended with 40 host-demoted pages; c16 ended with 40 disk pages because L2 resolved to zero. Cache-tier differences therefore did not accelerate either decode path, but this workload does not measure cache-hit benefit.

DSpark chains increased by only 9 at c8 and 8 at c16, while the high-concurrency trace contained 381 and 257 batched target ticks. This matches the implementation: B>1 marks DSpark slots ineligible and enters the target layer-major batch path (`executor/dsv4.rs:1648-1660`, `1725-1735`). The trace measures target batching, not batched DSpark verification.

## Problems

- GuideLLM used only 20 requests. Full B16 still has 121 consecutive ticks, but aggregate throughput includes the B4 drain wave. A sustained service number needs at least 80 requests.
- c8 had L2 capacity while c16 did not. Zero cache hits/promotions/fetch wait remove it from this decode comparison; cache-reuse throughput remains unmeasured.
- Two startup probes were discarded: unsupported `--cpuctxsw`, then a physical/logical CUDA device ordinal mismatch. Both failed before capture. The recorded runs use physical GPUs 3–6 and the same binary.
- nsys adds about 15% wall overhead. It licenses attribution, not absolute production throughput.

## Learnings

- **High concurrency works:** no-nsys aggregate output rises 17.2%; saturated timestamp throughput rises 33.4% from B8 to B16.
- **Timestamp overlap is descriptive, not causal:** about 80% of launch time overlaps GPU work, and launch outside GPU-busy intervals remains 6.3% of both full-batch ticks. This disproves “all launch time is exposed,” but does not license or kill graph/preallocation.
- **The next measured wall is GPU batching efficiency, led by grouped MoE work.** License-or-kill grouped-MoE or head batching with a same-binary operator A/B; kernel share alone is not a license.
- **Benchmark occupancy explicitly.** Client concurrency is not batch size. Report the NVTX-derived batch histogram and separate saturated ticks from drain waves.

## Delta vs baseline

The prior c=1 timestamp model measured 0.313 ms launch time outside GPU-busy intervals per token, 1.19% wall. Full B16 measures 5.174 ms per 16-token tick, 0.323 ms/token, nearly identical per token. This is an overlap invariant; causal graph benefit remains unmeasured at B16.

## Artifacts

- c8: `/host/arle-dspark-nsys/2026-07-14-c8/`
- c16: `/host/arle-dspark-nsys/2026-07-14-c16/`
- Each directory contains `trace.nsys-rep`, `trace.sqlite`, GuideLLM JSON/CSV, correctness output, server log, `/v1/stats`, nsys summaries, `overlap.json`, and binary SHA-256.
