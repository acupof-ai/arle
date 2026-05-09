---
title: SKELETON — Path B Phase 1 Substep 1.1 dequant.h port wins entry (codex fill on PASS)
date: 2026-05-10
type: wins-skeleton
status: pending-codex-bench-result
---

# SKELETON — Path B Phase 1 Substep 1.1 dequant.h port

> Pre-drafted skeleton for codex to fill in once Substep 1.1
> greedy_consistency PASS + bench A/B numbers land. Saves a tick of
> doc work after the PASS comes through.
>
> **Codex: rename this file to remove `SKELETON-` prefix and fill in
> the `[fill]` placeholders below using raw bench output (per skill
> v1.10.0 #28: quote literal output, NOT memory recall).**

## Context

Path B-Phase2' Phase 0 P0.A KILLED `67f18b9` + `61c9666` (W4 decode
HBM-bound, FP8 mma is wrong lever). Path B Phase 1 is the surviving
low-risk -3-8% ITL fallback per `e59beb5` survey + `61c9666` revised
priority (P2).

This wins entry covers **Substep 1.1 only**: port vLLM-current
`marlin/dequant.h` (609 LOC) into ARLE as `marlin_dequant.h` (Strategy
Hybrid per `24be401` scope note + `70b4d7b` audit). Substep 1.2
atomic_add reduce opt-in lands as a separate wins entry.

## What worked

[fill: codex describe the hybrid-strategy choice (single-file
verbatim shim) + greedy_consistency PASS evidence + bench A vs bench
B Δ% + key TFLOPS / ITL / TTFT numbers from raw output]

Suggested fill structure:

### 1. Strategy choice (validate hybrid was correct)

[fill: confirm hybrid (single-file marlin_dequant.h with vllm shim
namespace) compiled cleanly + future-cherry-pick friendly]

### 2. Greedy_consistency gate (NUMERICAL PASS)

```bash
$ cargo test --release --features cuda --test greedy_consistency
[fill: PASS output verbatim — number of test cases passed,
duration, any noteworthy lines]
```

### 3. Bench A — baseline (current path before dequant.h port)

```bash
$ scripts/bench_guidellm.sh path-b-p1-baseline \
    --concurrencies 4 --max-seconds 120 --warmup 10 \
    --data 'prompt_tokens=4096,...,output_tokens=256,...'
[fill: TTFT/ITL p50/p95/p99 from /v1/stats engine_ttft_us, NOT
guidellm client TTFT — guidellm is broken with INFER_PREFILL_GRAPH=1
per e8d82b0]
```

### 4. Bench B — treatment (new dequant.h, no atomic)

```bash
$ INFER_MARLIN_ATOMIC_REDUCE=0 \
  scripts/bench_guidellm.sh path-b-p1-newdequant ... [same params as A]
[fill: TTFT/ITL numbers + Δ% vs A]
```

### 5. License-or-kill against e59beb5 gates

| Metric | Baseline (A) | Treatment (B) | Δ% | License gate | Verdict |
|--------|--------------|---------------|----|--------------|---------|
| ITL p50 (engine_ttft_us-equivalent) | [fill] ms | [fill] ms | [fill] | ≥ -3% | [LICENSE/KILL] |
| ITL p95 | [fill] ms | [fill] ms | [fill] | regression < +5% | [LICENSE/KILL] |
| TTFT p50 | [fill] ms | [fill] ms | [fill] | regression < +2% | [LICENSE/KILL] |
| Throughput tok/s | [fill] | [fill] | [fill] | ≥ -2% | [LICENSE/KILL] |
| greedy_consistency | n/a | PASS | n/a | required | [PASS/FAIL] |

[fill: overall LICENSE / KILL verdict + 1-2 sentence reasoning]

## Rule (sediment for skill v1.11.0 candidate if applicable)

[fill: any anti-pattern observation worth promoting to skill catalog —
e.g. "vLLM Apache 2.0 cherry-pick into ARLE works cleanly with
namespace shim + macro define pattern; future ports follow this
template" OR "FasterTransformer-derived dequant constants are
byte-identical across Marlin variants — spot-check is sufficient
for greedy_consistency low-risk gate"]

## Cross-references

- Phase 1 brief: `/tmp/codex-brief-phase1-dequant.txt` (sent prior tick)
- Phase 1 substep breakdown: `docs/research/2026-05-10-path-b-phase-1-vllm-marlin-port-execution-ready.md` (e59beb5)
- Scope note + dependency map: `docs/research/2026-05-10-phase1-dequant-port-scope-note.md` (24be401)
- Pre-build audit (CLEAN): `docs/research/2026-05-10-phase1-substep1.1-codex-impl-audit-clean.md` (70b4d7b)
- Pre-staged upstream: `/tmp/upstream-marlin/dequant.h`, `/tmp/upstream-marlin/marlin_dtypes.cuh`
- Phase 0 KILL context: `docs/research/2026-05-10-phase0a-decode-kill-architectural-implication.md` (61c9666)
- ARLE Marlin substrate: `crates/cuda-kernels/csrc/gemm/marlin_kernel.cu` (~828 LOC post-port = 844 - 16)
- New file: `crates/cuda-kernels/csrc/gemm/marlin_dequant.h` (651 LOC)
- vLLM upstream Apache 2.0: https://github.com/vllm-project/vllm/blob/main/csrc/quantization/marlin/dequant.h

## Codex pickup directive when bench done

1. Replace `[fill]` placeholders above with raw bench + greedy output
2. Rename file: remove `SKELETON-` prefix → `2026-05-10-path-b-phase1-substep1.1-dequant-port.md`
3. Commit: `docs(wins): Path B Phase 1 Substep 1.1 dequant.h port — [LICENSE/KILL outcome]`
4. If LICENSE: PushNotification user with headline numbers
5. If KILL: PushNotification user + open errors entry (rename to
   docs/experience/errors/) + propose next pickup (Strategy B stripped
   port? OR skip Phase 1 entirely?)

## 状态 (post-fill, codex update)

[fill: brief status — Phase 1 Substep 1.1 LICENSED / KILLED, ready for
Substep 1.2 atomic_add OR pivot to alternative]
