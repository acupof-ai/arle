# DSv4 SM90 MegaMoE TP4

> Status: Shipped — opt-in licensed on 4x H20; default unchanged.

## Goal

Use DeepGEMM PR #323 directly for DSv4 routed experts without PyTorch or a
second kernel implementation.

## Hypothesis

The fused dispatch, L1, L2, and combine path improves TP4 wall-clock throughput
once each replicated TP batch is token-sharded before dispatch.

## Parameters

- Model: `/host/DeepSeek-V4-Flash-FP8`, TP=4 on GPUs 3-6
- Input/output: 1,024/256 tokens, seed `20260416`
- Concurrency: 1, 4, 8, 16; 120 seconds each
- Baseline: `ARLE_DSV4_MOE_TRANSPORT=allreduce`
- Candidate: `ARLE_DSV4_MOE_TRANSPORT=mega_moe`
- L1 KV: `mem_fraction_static=0.9`; L2: 50% host DRAM; L3: off
- Binary SHA256: `16a2dd6a30d64e333a082991e938c18e5c6558573239ed7eb963e0e38e5f98e1`
- Upstream PR head: `9e3afe91cb145ddfa0b18ae874a11dbb449e16a9`

GuideLLM needed a tokenizer-only processor view because installed Transformers
does not recognize `deepseek_v4`; tokenization still used the checkpoint's
unchanged `tokenizer.json` and `tokenizer_config.json`.

## Environment

- 4x H20, SM90, 96 GB each
- Driver 535.161.08, CUDA 12.9, NCCL 2.27.3
- 1.9 TiB host RAM
- Final incremental release build: 2m12s
- Final serving binary: 73,735 MB weights/rank, 186 slots

## Correctness

The allreduce and final token-sharded MegaMoE paths both decoded `Paris` for the
same greedy request. The independent PR reference passed at the exact Flash
shape with `calc_diff=0.0006 < 0.07`; upstream accuracy passed 28/28 cases.

## Results

Final same-binary sweep:

| c | Path | TTFT ms | ITL ms | Output tok/s | Total tok/s | Req/s | Errors |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | allreduce | 1,432.8 | 21.90 | 36.93 | 184.77 | 0.142 | 0 |
| 1 | MegaMoE | 951.5 | 19.95 | 42.72 | 213.79 | 0.158 | 0 |
| 4 | allreduce | 3,995.9 | 52.32 | 59.98 | 289.73 | 0.200 | 0 |
| 4 | MegaMoE | 2,267.3 | 45.52 | 72.92 | 337.43 | 0.250 | 0 |
| 8 | allreduce | 6,365.6 | 69.44 | 86.95 | 405.37 | 0.283 | 0 |
| 8 | MegaMoE | 3,571.8 | 52.81 | 121.20 | 563.34 | 0.408 | 0 |
| 16 | allreduce | 20,674.5 | 76.16 | 117.51 | 588.02 | 0.275 | 0 |
| 16 | MegaMoE | 9,220.2 | 74.06 | 126.15 | 575.77 | 0.433 | 0 |

| c | Output tok/s delta | TTFT delta | ITL delta |
|---:|---:|---:|---:|
| 1 | +15.7% | -33.6% | -8.9% |
| 4 | +21.6% | -43.3% | -13.0% |
| 8 | +39.4% | -43.9% | -24.0% |
| 16 | +7.4% | -55.4% | -2.8% |

Fresh-server fixed-c16 control:

| Path | TTFT ms | ITL ms | Output tok/s | Total tok/s | Req/s | Prefix hit peak |
|---|---:|---:|---:|---:|---:|---:|
| allreduce | 21,330.1 | 76.60 | 79.97 | 400.18 | 0.267 | 0.0% |
| MegaMoE | 9,393.0 | 74.44 | 148.74 | 697.44 | 0.467 | 46.0% |

Fixed-c16 output throughput improved 86.0%, but faster completion created more
prefix-reuse opportunities. The cache-independent decode claim is therefore the
2.8% ITL improvement; 86.0% is the measured whole-service result with L2/prefix
enabled, not a pure kernel claim.

## Nsight Systems timeline

The pre-sharding c16 trace captured 6,836 MegaMoE launches. CUDA runtime and GPU
timestamps joined by process plus correlation id show host/GPU overlap, not a
host serialization bottleneck:

- Host CUDA launch time overlapped GPU work by 87.1-89.3% across four ranks.
- MegaMoE launch API p50: 4.1 us; kernel p50/p99: 372/639 us.
- API-end to kernel-start queue p50/p99: 10.78/28.51 ms.
- Three MegaMoE launches, nine BF16 all-reduces, and eight all-gathers exceeded
  10 ms; maxima were 107.6/143.0/96.8 ms.

The trace killed the original implementation: TP replicated the same tokens on
all ranks, while PR #323 assumes rank-owned tokens. Sharding contiguous token
rows before dispatch and reducing owned outputs removed 4x duplicate expert
work. The output scratch is allocated once; the hot loop allocates nothing.

## Component evidence

| Tokens/rank | MegaMoE us | TFLOPS | HBM GB/s |
|---:|---:|---:|---:|
| 1 | 103.0 | 3.4 | 1,467 |
| 4 | 262.5 | 4.8 | 2,015 |
| 8 | 322.1 | 6.7 | 2,190 |
| 16 | 455.4 | 9.8 | 2,821 |

At 16 tokens/rank, the optional DeepEP plus grouped-FP8 control took
1.54-1.58 ms versus 0.46-0.52 ms fused. This is component evidence only; the
wall-clock tables above license serving.

## Cold-load result

Only rank 0 physically prefetched the base checkpoint. The measured load was
294.0 GB across 46 shards in 1,445.4 seconds on the cold disk; ranks 1-3 reported
zero physical read bytes and reused page cache. This removes the old TP4
1.65-TB read amplification. A warm page-cache prefetch took 7.5-15.4 seconds.

## Problems

- The canonical 4,096/256 sweep is invalid above c=4 on this KV envelope. At
  c=8 the minimum-rank 47,936-token pool exhausted and the worker exited after
  repeated preemption. c=16 needs about 69,632 active tokens before retained
  prefix pages, so no code path can satisfy that shape with this pool.
- L3 was intentionally off. Synthetic unique prompts do not license SSD recall;
  enable it only for a repeated-prefix workload with measured disk hits.
- The nsys report is pod-local and intentionally not committed:
  `/host/arle-megamoe-t1/bench-output/2026-07-15-megamoe-c16-nsys/fused.nsys-rep`.

## Learnings

PR #323 is directly usable, but its distributed-token contract is DP-like, not
TP-replicated. Kernel microbench wins were real and still insufficient: only
the token-sharded, fixed-concurrency serving A/B licensed the path.

## Delta vs baseline

Opt-in MegaMoE is licensed for H20 TP4. Defaults stay on allreduce until a
second production shape and repeated-run variance gate pass.
