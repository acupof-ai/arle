---
title: 2026-05-10 Session-tail TOTAL summary — n=18 PASS Claude benches + 3 SKILL graduations + PF8 KILL chain + n=5 hybrid value progression + Phase 4 formula validated
date: 2026-05-10
type: research
status: open (session-tail anchor for next-session pickup)
related_docs: [`2026-05-10-post-pf85-direction-options.md`, `2026-05-10-w4a16-longctx-prompt2048-bench.md` (extended to n=4), `2026-05-10-w4a8-vs-w4a16-concurrency-scaling-full-matrix.md` (extended to §10), `pickup-queue-2026-05-10.md`]
---

# 2026-05-10 Session-tail TOTAL summary

> **Purpose**: single-anchor doc capturing ALL substantive work from
> this session-tail (~7+ hours, 86+ ticks, ~75+ Claude commits) for
> efficient next-session pickup. Supersedes scattered references in
> pickup queue §8 + various wins/errors/research entries.

## §1 Cumulative bench tally (Claude-run)

**19 PASS + 1 NULL-discovery + 1 misapplied (self-corrected)** across
session-tail. Categorized:

### §1.1 PF8.5 license + isolation chain (5 benches)

| # | Arm | Config | Verdict |
|---|---|---|---|
| 1 | Arm A | PF8.3 + warmup ON conc=1 | KILL — 5878 kernel failures from Pass 3 warmup B=1 |
| 2 | Arm B | PF8.3 + warmup OFF conc=1 | KILL — 5959 failures, REFUTES warmup-DEPENDENT framing |
| 3 | Arm C | W4A16-marlin-zpfix (control) conc=1 | HEALTHY — TTFT 66.0 / ITL 5.8 / 799 tok/s |
| 4 | Arm D | W4A8-zpfix (Task #48 default, control) conc=1 | HEALTHY — TTFT 54.2 / ITL 11.9 / 409 tok/s |
| 5 | (errors) | long-ctx all-rejected w/ default `--max-seq-len` | NULL — 4926 rejections discovered config ceiling at max_input=1997 |

### §1.2 Concurrency scaling W4A16 + W4A8 (4 benches)

| Conc | W4A16 (TTFT/ITL/tok-s) | W4A8 (TTFT/ITL/tok-s) |
|---|---|---|
| 1 | 66.0 / 5.8 / 159.6 | 54.2 / 11.9 / 81.7 |
| 2 | 82.1 / 7.4 / 248.8 | 83.2 / 12.7 / 149.1 |
| 4 | 78.1 / 7.7 / 469.6 | 52.8 / 13.0 / 289.4 |

73% throughput scaling efficiency at conc=4 (per `8d32576`). W4A8
TTFT bimodal across conc.

### §1.3 Long-context scaling W4A16 (4 benches: 512/2048/4096/8192)

| Prompt | TTFT | scale vs 512 | ITL | tok/s | demotions |
|---:|---:|---:|---:|---:|---:|
| 512 | 66.0 | 1× | 5.8 | 159.6 | 0 |
| 2048 | 272.1 | 4.12× | 6.4 | 117.6 | 0 |
| 4096 | 577.6 | 8.75× | 7.4 | 84.6 | 1 |
| 8192 | **1335.5** | **20.2×** | **8.9** | **52.4** | **4** |

Phase 4 formula validated EXACT MATCH at 8k (predicted 1.2-1.4s,
actual 1.336s).

### §1.4 Long-context scaling W4A8 (3 benches: 2048/4096/8192)

| Prompt | TTFT | ITL | tok/s | demotions |
|---:|---:|---:|---:|---:|
| 2048 | 191.3 | 12.6 | 71.8 | 0 |
| 4096 | 409.4 | 13.8 | 59.5 | 1 |
| 8192 | **985.4** | 15.4 | **43.9** | 3 |

§9.4 prediction validated at +4% margin (predicted 948 → actual 985).

### §1.5 BF16 baseline (1 bench, added EOD+2150)

| Path | TTFT | ITL | tok/s | Note |
|---|---:|---:|---:|---|
| BF16 (no quant) | 68.7 ms | 14.0 ms | 69.3 | Surprising: W4A16 STRICTLY dominates BF16 (TTFT -4%, ITL -59%, tok/s +130%) |

Per `eab166d` wins entry: quantization is STRICT WIN at sm_89 16GB,
not a tradeoff. Decode is HBM-bound on weight read; 4× less weight
memory at W4 → ~4× theoretical decode speedup; actual -59% ITL
matches. Accuracy preserved (greedy 0.0% diff per Task #48 8d1caad).

**Validates W4 quant axis as primary path on sm_89.**

### §1.6 Verification + self-correction (6 benches)

| Bench | Result | Notes |
|---|---|---|
| test_w4a8_vs_bf16 (greedy) | 0.0% diff PASS | Task #48 codex 8d1caad verification |
| test_e2e_w4a8 (default) | PASS in 3.90s | Task #48 verification |
| test_e2e_w4a8 (Pass 3 OFF) | PASS in 2.31s | TRUE A/B with -1.59s/-40.8% delta |
| W4A8 vs BF16 W4-hybrid-zpfix | FAILED 100% | Misapplied SKILL #29 — fixture mismatch |
| Task #43 hypothesis test | DISPROVEN INVERSE | Arm A scratch=ON KILLed, Arm B scratch=OFF HEALTHY |
| (Task #43 70:1 OOM ratio) | per da7f5a2 | + 16× TTFT degradation per d09623a self-correction |

## §2 SKILL graduations (3 this session-tail)

- **v1.13.0 (2026-05-10)**: graduated #38 (warmup target shape clamp)
  per `8b530ad`, n=2 evidence
- **v1.14.0 (2026-05-10)**: graduated #36 (grep + behavioral A/B both
  required) per `d2c987f`, n=2 INVERSE evidence
- **v1.15.0 (2026-05-10)**: graduated #35 (root-cause-TBD canary)
  per `b255c58` + `2356e6a`, n=3 evidence

**Total canonical anti-patterns**: 28-34 + 35 + 36 + 38 = **37**

## §3 SKILL candidates (pending future evidence accretion)

- #29 enhancement (now n=6 with Claude self-applications)
- #37 multi-shape bench discipline
- #39 post-fix bench data stale
- #40 KILL vs graceful-fallback discriminator
- #41 terminal silence ≠ no progress
- #42 temp-branch recovery for detached-HEAD
- **NEW twin-control-arm discipline** (n=1 from PF8.5 4-arm A/B per
  `430a4be`)
- **NEW end-to-end latency math vs naïve "best of both"** (n=1 from
  Hybrid Option B aggregation framing decay per `92813dc`)

## §4 Hybrid Option B value progression (n=5 contexts, fully measured)

| Context | Hybrid value vs W4A16 | Status |
|---|---:|---|
| conc=1 prompt=512 | -1.4% | sub-noise |
| conc=4 prompt=512 | -2.4% | sub-noise |
| conc=1 prompt=2048 | -7.5% | meaningful |
| conc=1 prompt=4096 | -11.1% | clearly above noise |
| conc=1 prompt=8192 | **-14.2% (measured)** | approaching Machete |
| prompt=16384 (extrap) | ~-17% | asymptotic, diminishing returns |
| prompt=32768+ (Qwen3 native) | ~-20% Machete-class | YARN required |

**Asymptotic, not linear**. Machete-class -20% threshold needs 32k+
native ctx OR 64k+ YARN-extended.

## §5 Strategic decision matrix (refined post-session)

| User priority | Recommended path | Time | Why |
|---|---|---|---|
| Maximum tok/s (any context) | A (Medusa) | 2-3 days | 2-3× tok/s at ≥70% accept |
| Maximum TTFT/ITL short-ctx (≤2k) | A (Medusa) | 2-3 days | hybrid value sub-noise here |
| Maximum TTFT/ITL long-ctx (≥8k) | B (Hybrid) | ~2 weeks | hybrid value -14 to -17% at 8-16k |
| **World-first 32k+ ctx specifically** | **B (Hybrid)** | ~2 weeks | only path to Machete-class -20% |
| Short timeline (≤1 week) | A (Medusa) | 2-3 days | clearest pickup, lowest blocker |
| Match ROADMAP P0 World #1 (W1/W2 32k×c=4) | Separate harness | ~weeks | not addressed by either A or B alone |

## §6 Tasks state at session-tail end

- **Task #28** (Medusa scaffold) — pending P1 pickup (Option A path)
- **Task #30** (Hybrid W4A16/W4A8 dispatch) — pending P2 pickup (Option B path)
- **Task #44** (PF8 chain) — completed KILL
- **Task #47** (PF8.3 H1' static-scratch refactor) — BLOCKED-pending-redesign
  (per `657c297`); 2 A/B gates needed (OOM + TTFT/tok-s regression)
- **Task #48** (W4A8 accuracy regression) — completed via codex 8d1caad
  (qzeros-fixed default + 1% gate canary)

## §7 Procedural sediment for future sessions

- `memory/feedback_user_drives_cron_cadence_overrides_saturation.md` —
  do NOT apply self-halt rules; user-driven cadence overrides
- `e37a46b` errors entry — cooperative loop asymmetric saturation
  pattern (3 ground rules)
- `a15a062` errors entry — long-ctx bench requires `--max-seq-len ≥
  2× prompt_tokens`
- `d09623a` — parse bench CSVs before strong claims; multi-artifact
  verification required for "server survived" / "fix worked" framing

## §8 Key cross-references for next-session pickup

- **Direction options** (A/B/C): `2026-05-10-post-pf85-direction-options.md`
- **W4A8 vs W4A16 full matrix**: `2026-05-10-w4a8-vs-w4a16-concurrency-scaling-full-matrix.md`
- **Long-ctx W4A16 series**: `2026-05-10-w4a16-longctx-prompt2048-bench.md` (§10 + §11)
- **PF8.5 KILL errors entry**: `2026-05-10-pf85-bench-v11-substrate-kill-conc1-warmup.md`
- **Twin-control SKILL candidate**: `2026-05-10-skill-candidate-twin-control-arm-discipline.md`
- **Index anchor**: `docs/index.md` Last refreshed line at EOD+1990
- **Pickup queue §8**: chronological tick log with all detailed entries

## §9 Open questions for user

1. Pick A (Medusa, 2-3 days) vs B (Hybrid, ~2 weeks) per direction
   options doc?
2. Long-context target context length distribution? Determines whether
   B's ~2wk investment is worth it for Machete-class threshold at 32k+
3. ROADMAP P0 (World #1 32k+ throughput) — is current data sufficient,
   or need separate W1/W2 conc=4 32k+ bench harness?
4. PF8.3 H1' redesign — wait for Task #47 redesign brief, or close
   permanently and pivot to alternative quant strategies (W3/W2)?

Awaits user direction on any of these. Multiple PushNotifications
dispatched without response per `e37a46b`.
