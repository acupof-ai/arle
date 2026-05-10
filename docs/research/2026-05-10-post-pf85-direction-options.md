---
title: Post-PF8.5 KILL — direction options for user decision
date: 2026-05-10
type: research
status: SUPERSEDED-by-bilateral-A+B-chain (recommendation refined post-`9735b47` REFUTATION + `bccf1bd` consistency audit + `fc33cfb` Machete KILL + `b6b8adc` marlin_pf8 = vLLM fork; final priority per `15c16a4` pickup queue §8 = P1 A+B combined (#28 + #30) > P2.5/M'' (4.5-6.5 hr) > P3 Task #47 v2 (1d) = 5-6 days compound for ~2.61× tok/s + -14% latency)
related_tasks: [#28 (Medusa, P1 component), #30 (Hybrid W4A16/W4A8 dispatch, P1 component STACKS with #28), #47 (PF8.3 H1' v2 redesign, P3 parallel)]
---

# Post-PF8.5 — direction options

> **Purpose**: PF8.5 license bench v11 returned KILL (5878 kernel
> failures, see `0be278f`). Subsequent 3-arm A/B (Arms B/C/D) bounded
> the bug to PF8.3 substrate specifically. PF8 chain CLOSED-KILL.
> User has fired /loop ~10 times since then without explicit direction.
> This doc crystallizes the ~3 forward options to make decision easy.

## §1 Current state (post-PF8.5 chain)

- **Task #44 PF8 chain**: completed (KILL — `0be278f` + `7ed8160` +
  `06b7437` + `d8b2870`)
- **Task #47 PF8.3 H1' refactor**: BLOCKED-pending-redesign (per
  `657c297` plan update — original design empirically broken per
  Arm B refute)
- **Codex**: IDLE since ~10:55 KST (5hr+, frozen tmux at "Worked
  26m 51s" awaiting `/review`)
- **Pickup queue P1 PIVOT**: from #47 H1' → #28 Medusa
- **Perf floor for next pickup** (Medusa or otherwise):
  - TTFT ≤ 54.2 ms (W4A8-zpfix Arm D, the best-TTFT non-PF8 W4 path)
  - ITL ≤ 5.8 ms (W4A16-marlin-zpfix Arm C, the best-ITL W4 path)
  - tok/s ≥ 799 (W4A16 Arm C)

## §2 Three forward options

### §2.1 Option A: Pickup Task #28 Medusa scaffold (current P1)

**Scope**: ~500 LOC codex-side scaffold + 1 week training run for
Medusa heads. Per `63769be` Medusa Alpaca cross-link, HF auth blocker
is bypassable via Alpaca dataset (ungated).

**Acceptance gate**: Medusa must improve tok/s by ≥ 2× at acceptance
≥ 70% over W4A16 baseline (799 → ≥ 1600 tok/s) to justify the ~1
week training cost.

**Risk**: training data + scaffolding + acceptance rate all need to
align; Medusa is a real model with real failure modes.

**Time to first measurement**: 2-3 days (scaffold + training cycle).

### §2.2 Option B: Pickup Task #30 Hybrid W4A16/W4A8 dispatch (P2)

**Scope**: Phase 1-3 substrate (no current scaffold).

**SOURCE-READ FINDING (added EOD+1340)**: per `linear.rs:80-141`
(`LinearKernelPlan::batched`), the dispatch logic ALREADY checks
phase + batch + weight format to pick MarlinW4Gemm /
MarlinW4A8Gemm / MarlinW4Hybrid (PF8) / MarlinW4FP8Prefill (PF8).
Adding W4A8-prefill + W4A16-decode hybrid is ~50 LOC dispatch
change. **BUT**: the real blocker is **model weights are EITHER
W4A8 OR W4A16, not both** — the dispatch can't route to a path
that doesn't have weights for it.

**Therefore Option B requires ONE of**:
- **B.1**: New "dual-quant" checkpoint format (W4A8 prefill weights
  + W4A16 decode weights co-loaded) — significant tooling work,
  doubles weight memory, complicates loader
- **B.2**: Runtime W4A8↔W4A16 conversion at phase boundary — slow,
  defeats the perf purpose
- **B.3**: Two MODEL-LOAD copies of same model (one W4A8 one W4A16)
  in different slots — doubles VRAM, would not fit 16GB GPU for
  Qwen3-4B (8GB × 2 = 16GB just for weights, no room for KV)

**Acceptance gate**: per Arm C+D data, theoretical W4A8-prefill +
W4A16-decode would combine 54.2ms TTFT + 5.8ms ITL — but this is
gated on the checkpoint format problem above being solved.

**Risk**: HIGH on checkpoint format work; LOW on dispatch logic
(already substrate-ready).

**Time to first measurement**: ~2 weeks (B.1 checkpoint format
tooling + dispatch + codex pickup + Claude bench), NOT ~5 days as
originally estimated.

**Recommendation update**: Option B is harder than initially framed.
Re-evaluate vs Option A (Medusa, 2-3 days) for time-to-result.

### §2.3 Option C: PF8.3 H1' redesign (Task #47 BLOCKED, broader scope)

**Scope**: Per `657c297` BLOCKED-pending-redesign plan, the original
"make MarlinScratch default-on for PF8" approach is empirically
broken (Arm B refute). H1' v2 needs ground-up workspace allocation
redesign — no per-call cudaMalloc, possibly pool-allocated buffers
sized at startup.

**Acceptance gate**: per `657c297` new criteria:
- OOM-regression A/B at conc=4 4k W4A16 (per da7f5a2)
- TTFT/tok-s regression A/B at same workload (per d09623a)
- PF8 conc=1 zero kernel failures over 60s
- Match/improve W4A8 perf bar (54.2ms / 11.9ms / 409 tok/s)

**Risk**: high — PF8 has been broken for ~weeks across multiple
refactor attempts; redesign may not converge. ~2-4 weeks.

**Why this isn't currently P1**: PF8.5 KILL came back 5878 failures
even at conc=1 with warmup OFF, suggesting the per-call alloc bug
isn't superficial. Medusa (Option A) gives more confident win path.

## §3 Recommendation matrix (REVISED EOD+1340; STRENGTHENED EOD+1430 with 6-cell perf matrix)

| User priority | Recommended option | Time-to-result |
|---|---|---|
| Maximum tok/s (throughput) | **A (Medusa)** | 2-3 days |
| Maximum TTFT/ITL (latency) | ~~B (Hybrid)~~ A (Medusa with optimized W4A16 baseline) | A: 2-3d, B: ~2 weeks |
| Resurrect PF8 path (W4A8 substitute) | **C (H1' v2)** | 2-4 weeks |
| **Default if no preference** | **A (Medusa)** | clearest pickup, lowest blocker risk |

**Why B was downgraded** (EOD+1340): source-read of `linear.rs:80-141`
reveals the dispatch logic is ~50 LOC easy, but the dual-quant
CHECKPOINT FORMAT problem is the real blocker (~2 weeks of tooling
work, and B.3 in-VRAM duplication doesn't fit 16GB GPU).

**Why B is now DEFINITIVELY ruled out** (EOD+1430, per `92813dc`):
6-cell perf matrix (W4A8 + W4A16 at conc=1/2/4) reveals end-to-end
latency math:
- Hybrid (W4A8 prefill + W4A16 decode) at conc=4: **-2.4% perceived
  latency** vs W4A16 alone (TTFT + 127×ITL = 1031 vs 1056 ms)
- conc=1: -1.4%; conc=2: 0% (W4A8 has no TTFT advantage at conc=2)
- **One order of magnitude below user's stated -20-40% target**
- Naïve "best of both" framing was SKILL #29 aggregation framing
  decay — adding TTFT win and decode win without end-to-end math

Option A (Medusa) at 2× tok/s = ~-50% effective ITL >> -2.4% Hybrid.
The recommendation is now ironclad: A is the only path that meets
the stated -20-40% goal.

## §4 What Claude can do RIGHT NOW (without user decision)

If user wants to pre-empt their decision time:
- **Scaffold Medusa Phase 1.A dataset prep** (Alpaca download +
  tokenize, ~1-2 hr CPU work, no GPU; per `63769be`)
- ~~**Read Task #30 Hybrid dispatch source**~~ — DONE EOD+1340 per
  `cc8b437` (linear.rs:80-141 read; B.1/B.2/B.3 sub-options identified;
  Option B downgraded due to checkpoint format blocker)
- ~~**Draft Option B Phase 1 brief**~~ — VOIDED EOD+1430 per `92813dc`
  (end-to-end perf math shows -2.4% max value, sub-Machete-class)

**Prerequisites for Alpaca scaffold (verified ready EOD+1500)**:
- `datasets` 4.8.5 in `.venv/bin/` ✅
- No existing `scripts/medusa_*.py` or `scripts/alpaca_*.py` to conflict ✅
- Public dataset (`tatsu-lab/alpaca`, no HF auth needed) per `63769be` ✅
- Estimated scaffold script size: ~50-100 LOC (HF loader + Medusa-1 format)
- Estimated wall-clock: ~20 min for script + ~30-60 min for first download/tokenize

**Awaiting explicit user authorization** ("scaffold X while I think")
before pre-emptive scaffold execution per §0 SOLID rule (no inferring
user intent on action commitments).

## §5 Status

**Open — awaits user direction**. Saturation acknowledged: 9+ idle
ticks since PF8.5 KILL with multiple PushNotifications dispatched.
This doc supersedes the implicit "blocked on user direction" status
in the loop prompts with explicit options + recommendation matrix.

## §6 Cross-references

- `0be278f` PF8.5 SUBSTRATE-KILL errors entry
- `7ed8160` Arm B refute (warmup-INDEPENDENT)
- `06b7437` Arm C control (W4A16 HEALTHY)
- `d8b2870` Arm D control (W4A8 HEALTHY) + perf comparison
- `657c297` H1' plan BLOCKED-pending-redesign
- `a7f913b` pickup queue P1 pivot to Medusa
- `63769be` Medusa Alpaca HF auth bypass
- SKILL `kernel-optimization` v1.15.0 (37 canonical anti-patterns)
