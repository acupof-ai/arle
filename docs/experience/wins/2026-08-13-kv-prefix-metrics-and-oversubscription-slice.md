# KV prefix metrics and oversubscription slice — CUDA, 2026-08-13

> Status: pending-remote

## Goal

Measure 32K-agent throughput while reducing whole-slot park/promote frequency,
and make prefix-cache counters report the tokens actually restored.

## Hypothesis

A configurable minimum decode slice preserves the existing scheduling policy
while allowing fewer whole-slot transfers. Recording raw, licensed, and
restored prefix lengths at their actual decision points removes false hits when
Qwen3.6 has no restorable sidecar boundary.

## Parameters

```bash
python3 scripts/bench_throughput.py \
  --url http://127.0.0.1:<port> \
  --model ThinkingCap-Qwen3.6-27B-FP8 \
  --prompts-jsonl bench-agent-32k-2x10.jsonl \
  --concurrency-grid 2 \
  --requests-per-concurrency 20 \
  --max-tokens 256 \
  --seed 20260416 \
  --timeout-seconds 900 \
  --output <evidence>/bench
```

- Arms: `--kv-oversubscription-min-slice 8` and `32`
- Common flags: BF16 KV, `--max-running-requests 1`,
  `--kv-oversubscription`, `--max-total-tokens 65536`
- Prompt target: 32K tokens, two unique sessions, ten strict-prefix turns each
- Completion tokens: fixed 256 with `ignore_eos=true`
- Trials: three per arm, sequential and counterbalanced

## Environment

- Host / GPU: DevOps H20 pod / one isolated NVIDIA H20 96 GB
- Model / dtype: ThinkingCap-Qwen3.6-27B-FP8 / BF16 KV
- Source parent: `448104810`; the six changed files match current main
- Candidate binary SHA256: `da2fb8224edf0bb5c8fa5628f4914d7aa2f67e8621444964ab3d37ce994c4fd6`
- Kernel bundle SHA256: `4cd9da81dfa624cd1becaa1416c56458b6edc36994b1f30cac836b52feb167b4`

## Results

Prefix correctness is complete:

| common prefix | raw / licensed blocks | restored tokens | hit tokens / pages / hits | prefill tokens |
|---:|---:|---:|---:|---:|
| 8191 | 511 / 511 | 0 | 0 / 0 / 0 | 8223 |
| 8192 | 512 / 512 | 8192 | 8192 / 512 / 1 | 32 |

The 8191-token miss increments both `reuse_miss` and `fallback_recompute` once.
Needle retrieval passed 3/3 at 512, 4096, 8192, and 12000 tokens.

The first A/B used an invalid prompt shape: measured prompt p50 was 36159,
10.35% above the 32K target. It is diagnostic only:

| slice | completed | errors / empty | output tok/s | wall s | park / promote | TTFT p50 / p99 s | ITL p50 / p99 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 20 / 20 | 0 / 0 | 0.941 | 5440.23 | 620 / 620 | 5.084 / 22.57 | 18.06 / 18.35 |
| 32 | 20 / 20 | 0 / 0 | 3.878 | 1320.26 | 140 / 140 | 5.102 / 22.72 | 18.06 / 18.57 |

Slice 32 reduced transfers by 77.4% and improved diagnostic throughput by
312.1%, with TTFT p99 +0.7% and ITL p99 +1.2%.

The corrected workload has tokenizer-measured prompt min/p50/max
30142/33002/35849, p50 +0.71% from target, SHA256
`d5ead063d578ece68d3fc2f2dee831541449cf8a8dd716d4e6acfaa547792206`.
Formal trials are blocked: independent GPU2 and GPU7 attempts were externally
SIGKILLed after launch. Helper signal tracing saw no signal to the helper; the
ARLE logs had no graceful-shutdown line; there was no OOM, Xid, or host-memory
pressure. The sender is unknown. No formal performance result is reported.

Raw evidence is under `/host/arle-kvfix-ops-448104810/evidence/` on the pod.

## Problems

At 32K context, one whole-slot park or promote takes about four seconds. The
8-token baseline therefore needs about 90 minutes per 20-request trial.
Parallel arms would contend for host DRAM bandwidth and invalidate the
comparison. A corrected-shape Q8 attempt reached 14/20 with zero request errors
before an external termination; it is invalid and excluded.

## Learnings

PASS for prefix metric correctness and model correctness. The scheduler default
remains 8. Repeat the corrected-shape, three-trial matched A/B only after the
pod can preserve a GPU process for the full trial.
