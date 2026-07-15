# DSv4 MegaMoE decoded-case correctness failure

> Status: Active — performance passes; correctness blocks use.

## Goal

Measure MegaMoE against allreduce with fixed concurrency and no prefix reuse.

## Hypothesis

Changing only the MoE transport preserves greedy output while improving
wall-clock throughput.

## Parameters

- Binary: `91d105f3f618`, SHA256 `16a2dd6a30d64e333a082991e938c18e5c6558573239ed7eb963e0e38e5f98e1`
- Baseline: `ARLE_DSV4_MOE_TRANSPORT=allreduce`
- Treatment: `ARLE_DSV4_MOE_TRANSPORT=mega_moe`
- Workload: 20 natural 32-token prompts, 128 output tokens, greedy decode
- Concurrency: 1, 4, 8, 16; 16 requests per point; seed `20260416`
- KV: L1 90%, L2 50% host DRAM, L3 off; zero prefix hits

## Environment

- 4x NVIDIA H20, TP=4 on GPUs 3-6
- Driver 535.161.08; CUDA 12.9; NCCL 2.27.3
- Model: `/host/DeepSeek-V4-Flash-FP8`
- Server: 16 slots, 400 KV pages

## Results

| c | allreduce tok/s | MegaMoE tok/s | delta | TTFT p50 delta | ITL p50 delta | E2E p50 delta |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 43.00 | 48.25 | +12.2% | -33.1% | -10.0% | -11.3% |
| 4 | 81.08 | 95.39 | +17.7% | -46.3% | -13.3% | -15.1% |
| 8 | 139.03 | 184.83 | +32.9% | -42.2% | -25.0% | -24.6% |
| 16 | 177.20 | 256.56 | +44.8% | -48.3% | -29.0% | -30.9% |

All 128/128 requests completed without transport errors. Prefix hits were zero.
MegaMoE failed the decoded-output gate on one c=1 request: it entered a
`2.2.2...` attractor while allreduce continued coherent text. Five direct
replays reproduced the MegaMoE failure five times.

## Problems

The earlier single `Paris` smoke proved reachability, not output preservation.
This run has one trial and 16 requests per point, so it licenses neither a
default flip nor a variance claim. The deterministic failing case is sufficient
to block use.

## Learnings

KILL the current opt-in license, not the optimization. MegaMoE has a material
wall-clock win, but distributed FP8 routing or reduction must match allreduce on
the failing token sequence before more throughput tuning.

## Artifacts

- `/host/arle-megamoe-t1/bench-output/2026-07-15-native-allreduce-c1-16-r1/benchmarks.{json,csv}`
- `/host/arle-megamoe-t1/bench-output/2026-07-15-native-megamoe-c1-16-r1/benchmarks.{json,csv}`
- `/host/arle-megamoe-t1/logs/mega-native-ab.log`
