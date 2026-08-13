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

Performance A/B is running. Raw evidence is under
`/host/arle-kvfix-ops-448104810/evidence/` on the pod.

## Problems

At 32K context, one whole-slot park or promote takes about four seconds. The
8-token baseline therefore needs about one hour per 20-request trial. Parallel
arms would contend for host DRAM bandwidth and invalidate the comparison, so
the trials run sequentially.

## Learnings

PASS for prefix metric correctness and model correctness. The scheduler default
remains 8 until the matched A/B resolves throughput and tail-latency impact.
