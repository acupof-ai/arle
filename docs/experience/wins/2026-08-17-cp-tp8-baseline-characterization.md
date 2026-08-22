# CP baseline at TP=8: no speedup, 17% throughput regression — CUDA, 2026-08-17

> Status: Confirmed

## Goal

Characterize CP=1 vs CP=2 end-to-end on ThinkingCap-Qwen3.6-27B-FP8 at TP=8
across all production axes: throughput, decode rate, TTFT, needle correctness,
and GSM8K accuracy. This is the clean-baseline re-run after the L2 tier budget
fix (0088e37e5); the prior 1.67x TTFT "speedup" was an artifact of that bug.

## Parameters

- Model: ThinkingCap-Qwen3.6-27B-FP8 (dense, 64 layers = 16 full-attn + 48 GDN,
  hidden=5120, intermediate=17408, FP8 e4m3)
- TP=8, CP=1 vs CP=2, world=8, 8×H20 pod (sm_90)
- Binary: `/host/arle-build/target/release/arle` (build 4bcefcb57)
- Throughput: `bench_throughput.py`, 64 synthetic prompts (535 tok), 128 decode
  tokens, concurrency 1/4/8/16/32, 64 requests per concurrency
- Decode: `decode_rate_probe.py --target-tokens 128000 --max-tokens 128`
- Needle: `lever_gate.sh`, lengths 446/1000/2000/4000/8000, ×3 runs
- GSM8K: `arle_capability_eval.py --tasks gsm8k --n-samples 50 --n-shots 8
  --concurrency 8 --gsm8k-max-tokens 10240`
- KV dtype: bf16, max-total-tokens 160000

## Environment

- Host / GPU: 8×H20 pod (sm_90), all 8 GPUs
- Driver / CUDA: 12.8
- Server flags: `--kv-cache-dtype bf16 --max-total-tokens 160000`

## Results

### Throughput (synthetic, 535-tok prompt, 128 decode)

| concurrency | CP=1 TTFT p50 | CP=2 TTFT p50 | CP=1 ITL p50 | CP=2 ITL p50 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 67 ms | 71 ms | 9.4 ms | 11.4 ms |
| 4 | 178 ms | 191 ms | 34.1 ms | 45.1 ms |
| 8 | 229 ms | 266 ms | 74.4 ms | 90.2 ms |
| 16 | 457 ms | 543 ms | 148.3 ms | 180.3 ms |
| 32 | 845 ms | 972 ms | 296.9 ms | 360.2 ms |

CP=2: TTFT and ITL are both 5-22% worse at every concurrency level.

### Decode at 128K context (single request, 128 decode tokens)

| CP | TTFT (s) | decode tok/s | TTFT Δ | decode Δ |
|---:|---:|---:|---:|---:|
| 1 | 45.4 | 78.7 | — | — |
| 2 | 47.6 | 61.2 | +4.9% | -22.2% |

### TTFT at 128K (separate probe, 2 runs each)

| CP | run 1 (s) | run 2 (s) | avg (s) |
|---:|---:|---:|---:|
| 1 | 44.1 | 42.0 | 43.0 |
| 2 | 42.3 | 42.3 | 42.3 |

CP=2 TTFT is 0.98x of CP=1 — within noise, no speedup.

### Needle correctness

| CP | lengths tested | exact | partial | miss |
|---:|---|---:|---:|---:|
| 1 | 446/1000/2000/4000/8000 ×3 | 15/15 | 0 | 0 |
| 2 | 446/1000/2000/4000/8000 ×3 | 15/15 | 0 | 0 |

All arms output the correct needle value. Correctness is not affected by CP.

### GSM8K (50 samples, 8-shot, concurrency 8)

| CP | accuracy | correct/total | invalid | wall (s) |
|---:|---:|---:|---:|---:|
| 1 | 0.980 | 49/50 | 0 | 428.3 |
| 2 | 0.960 | 48/50 | 0 | 560.1 |

The 1-sample gap is within noise (CI95: [0.895, 0.996] vs [0.865, 0.989]).
CP=2 wall time is 31% higher, consistent with the throughput regression.

Raw artifacts: `/host/arle-runs/` on pod (tp-cp{1,2}-bench.json,
decode-cp{1,2}-probe.log, cp{1,2}-needle.log, eval-cp{1,2}-gsm8k.log).

## Root cause

CP=2 at TP=8 provides zero compute or memory benefit and adds NCCL overhead.
Three mechanisms compound:

1. **Per-rank GEMM parity.** `attn_tp = tp / cp`. At TP=8: CP=1 → attn_tp=8,
   CP=2 → attn_tp=4. Per-rank GEMM shard: hidden/8 vs hidden/4 × seq/2 —
   identical FLOPs per rank. CP does not reduce GEMM load.

2. **Per-rank attention parity.** Heads per rank = H/attn_tp. CP=1: H/8 heads
   over full seq. CP=2: H/4 heads over seq/2 (B2 KV sharding). Same KV bytes
   per rank. The B2 decode path (threshold 8192 tokens) engages at 128K but
   cannot reduce per-rank work.

3. **NCCL overhead.** CP=2 adds 16 layers of KV all-gather (full-attention) +
   48 layers of GDN state relay per step. At TP=8 the attn all-reduce already
   spans 8 ranks; CP adds a second communication axis without reducing the
   first.

The Amdahl limit for CP=2 on this model: 75% of layers are GDN (linear, no
sequence parallelism). Theoretical max speedup = 1/(0.75 + 0.25/2) = 1.14x.
The measured 0.98x is within the overhead band of that ceiling.

**Why TP=2 showed +33-42% TTFT (b2-cp-decode-gate):** at TP=2, CP=2 collapses
attn_tp from 2 to 1, eliminating the attention all-reduce entirely and halving
attention FLOPs per rank. That benefit does not generalize to TP=8 where
attn_tp only halves 8→4.

**Why the old 1.67x existed:** the L2 tier budget bug (0088e37e5) allocated
50% DRAM per rank without dividing by TP world, causing 8x over-allocation at
TP=8. This inflated CP=1 TTFT (70.8s vs 43.0s clean). CP=2 was less affected
(attn_tp=4, 4x over-allocation), creating a false speedup.

## Learnings

CP=2 at TP=8 on this model is a net negative: -17% throughput, -22% decode,
no TTFT benefit, +31% eval wall time. Accuracy is unaffected (GSM8K 49/50 vs
48/50, within noise). The CP feature is correct (needle 15/15, GSM8K 48/50) but
provides no performance value at TP=8. CP is beneficial only when attn_tp
collapses to 1 (TP=2) or for capacity (replicated KV doubles effective KV
cache). The 1.67x speedup in prior entries was a bug artifact, not a real CP
gain.

The throughput ceiling (~107 tok/s at c≥4) is set by Marlin W8A16 GEMM (52%
of decode step) + per-layer NCCL all-reduce, not by attention. CP cannot move
this ceiling.
