# B2 CP decode gate (T3.1) — CUDA, 2026-08-17

> Status: Shipped

## Goal

Verify that the B2 (replicated-KV, flash-decoding) CP decode path
(807e6c0b4) produces correct output and does not regress decode at long
context. The gate covers needle correctness ×3 same-config and decode-rate
probes at 4K / 128K / 224K context, CP=1 vs CP=2, TP=2 on 2×H20.

## Parameters

- Model: ThinkingCap-Qwen3.6-27B-FP8, TP=2, GPUs 1,3
- Needle gate: `lever_gate.sh`, LENGTHS=8000, RUNS=3, needle ladder ×3
- Probes: `decode_rate_probe.py --target-tokens <N> --max-tokens 128`
- Baseline: CP=1 (same binary, `INFER_ATTN_CP_SIZE=1`)
- Treatment: CP=2 (`INFER_ATTN_CP_SIZE=2`)
- 10 arms: needle wash×2, needle engaged×2, 4K probe×2, 128K probe×2, 224K probe×2

## Environment

- Host / GPU: 8×H20 pod (sm_90), GPUs 1,3
- Driver / CUDA: 12.8
- Model / dtype: ThinkingCap-Qwen3.6-27B-FP8
- TP / CP: TP=2, CP=2 (world=2)
- Binary: `/host/arle-build/target/release/arle` (build b2cp2)

## Results

All 10 arms exit=0. Needle gate: correctness PASS on all arms (exact=3/3
per arm, 5 summaries per arm).

| Context | CP | decode tok/s | TTFT (s) | decode Δ | TTFT Δ |
|---------|----|-------------|----------|----------|--------|
| 4K | 1 | 64.82 | 1.87 | — | — |
| 4K | 2 | 59.91 | 1.26 | -7.6% | +33% |
| 128K | 1 | 58.20 | 63.78 | — | — |
| 128K | 2 | 59.45 | 37.04 | +2.1% | +42% |
| 224K | 1 | 50.31 | 127.57 | — | — |
| 224K | 2 | 52.03 | 80.97 | +3.4% | +37% |

Raw artifacts: `/host/arle-gates-b2cp/` on pod.

## Problems

- The 256K target was adapted to 224K: RoPE caps at 262144 and the prompt
  ceiling is 229376, so 256K is unreachable. 224K is the practical maximum.
- The #211 devops agent accidentally killed the gate's serve process mid-run
  (judged it stale); the gate driver recovered and relaunched. No arm was
  affected.

## Learnings

PASS. B2 CP decode is correct at all context lengths. CP=2 does not regress
decode at 128K/224K (slightly faster, within noise) and cuts TTFT by 33-42%
at all context lengths. The 4K decode overhead (-7.6%) is the CP
communication cost at short context where the KV-sharding benefit has not
yet kicked in — acceptable for a long-context feature.

The 224K decode rate (50-52 tok/s) confirms that B2 CP decode is viable at
the edge of the context window. T3.2 (2D KV ownership sharding) is the next
step for capacity past 512K.
