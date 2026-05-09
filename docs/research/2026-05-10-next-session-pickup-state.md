---
title: 2026-05-10 next-session pickup state — quick orientation
date: 2026-05-10
type: research
status: session-end-checkpoint-for-next-pickup
---

# 2026-05-10 next-session pickup state — quick orientation

> One-page orientation for the next agent (codex reactivation OR
> fresh Claude session) to start from after this 16+ hour session.
> Read this first, then see `docs/index.md` for full context.

## §0 Read order on session start

1. This entry (one page, 3 minutes)
2. `docs/index.md` Last refreshed line (current state of all axes)
3. `de36538` retrospective (4 hallucinations + bilateral cooperative
   discipline working)
4. Skill v1.11.0 catalog: `.claude/skills/kernel-optimization/SKILL.md`
   — anti-patterns #29-32 are session-tested rules, not theoretical
5. Pickup queue (§3 below)

## §1 What's LANDED (today, 2026-05-10)

- **Phase 1 Substep 1.1 LICENSED** (codex `f86d0fd` + Claude
  `4f1b036` σ-tight n=2): TTFT -7.0%, ITL -3.2%, tok/s +2.1% on W4A16
  4k/c=4 vs 2026-05-08 baseline. Matches `e59beb5` -3-8% conservative.
- **PF8.1 + PF8.2 LANDED + smoke-verified**:
  - `940f49e` substrate (BF16→FP8 e4m3 quant + INT4 weight preprocess
    Apache 2.0 port, 181 LOC total, both feature gates clean)
  - `b628eca` PF8.1 runtime smoke PASS (max rel err 5.99% < 12.5%
    FP8 floor)
  - `451d094` PF8.2 runtime smoke PASS (caught 5th hallucination
    BY the smoke itself — bit-pack arithmetic memory-recall error)
- **PF8.4 dispatch wiring LANDED** (`db063ff`, +38 LOC): opt-in
  `INFER_MARLIN_W4_FP8_PREFILL=1` env var, bail at call site pending
  PF8.3 GEMM kernel.
- **#34 RESOLVED** (`df37a68`): `arle model download <id>` CLI surface
  unblocks P0 #28 spec decode hypothesis path.
- **Skill v1.11.0 LANDED** (`b551bea`): canonicalized 4 anti-patterns
  (#29-32) from session retrospective. Now load-bearing for future
  sessions.
- **#36 KILLED** (`9bbc441`): PrefixAware Layer 2 — substrate works
  but op-point fails (warm p95 +17%, cold p95 +114%, starvation
  4.56→8.33×). QueueBound stays default; opt-in CLI retained.
- **#40 Tier 1 wins LANDED** (`c44788f`): -92.5% engine TTFT (this
  session sealed via cooperative codex+Claude chain).

## §2 What's KILLED (with reasoning)

- **Path B-Phase2' Phase 0 P0.A** (`67f18b9` codex + `61c9666` Claude
  architectural synthesis): cutlass FP8 GEMM smoke decode 1.86× <
  2× kill threshold. **W4 decode HBM-bound on weights; FP8 mma is
  wrong lever**. User's "-20-40% ITL via FP8" is **structurally
  infeasible** on sm_89. Same memory-bound ceiling explains why
  Machete (Hopper) wouldn't help on sm_89 even if backportable.
- **Substep 1.2 atomic_add** (in design, `0d63a52`): raw grep proves
  W4A16 `marlin_kernel.cu` has only `int* locks` (no
  `max_par × 64 × n` reduce buffer). W4A8 alt deferred to prefill-only
  FP8 axis.
- **Machete sm_89 backport** (`e65a096` 5-pt convergent evidence):
  `arch::Sm90` hardcoded throughout (collective_builder + mainloop +
  generate.py + Readme + 2026-05-09 prior survey all confirm
  Hopper-only). Default Path B-Phase2' (W4+FP8) tried instead —
  also KILLED for ITL but prefill-only TTFT axis is viable (PF8 chain).
- **M_spec classical external draft** (`#27` closed at -73%/-46%
  tok/s on 4k random text per M_spec plan).

## §3 Pickup queue (priority order)

### Codex's natural pickup (highest leverage)

**PF8.3 FP8 marlin GEMM kernel** (~800-1200 LOC, 1-2 days codex) —
**STATUS: codex briefed via tmux paste-buffer THIS tick + Working
(2s)**. Brief: `/tmp/codex_brief_pf83.txt`. Strategy B selected
(single-template mirror, NOT verbatim port — m16n8k16→m16n8k32 mma
shape mismatch per `259277c`).
- Brief in `a66d99a` §1 + scope analysis in `259277c`
- Dispatch wiring already landed (`db063ff`); just plug kernel call
  into bail site at `infer/src/ops/linear.rs:1966+`
- Reuses cutlass sm_89 FP8 template from P0.A spike (per `d5a6679`
  unstick: `GemmUniversalWithAbsMax` + `arch::Sm89` +
  `LinearCombinationGenericWithScalingAndAbsMax`)
- KEY: shape mismatch m16n8k16 → m16n8k32 (k dim doubles, inner-loop
  changes substantially) — NOT a verbatim port (per `259277c`)
- License gate: TTFT p50 Δ ≥ -8% σ < 5% n=3 (per `a66d99a` §2)
- Strategy A (verbatim cascade marlin_template.h ~2000-3000 LOC) or
  Strategy B (single-template mirror marlin_w4a8_kernel.cu ~800-1200 LOC)
- Recommended: Strategy B for this scope

After PF8.3 lands: **PF8.5** = end-to-end TTFT bench A/B
(W4+INT8 baseline vs W4+FP8 prefill treatment).

### Long-term ITL win path (P0 hypothesis, blocked by training cost)

**#28 Medusa scaffold** (~500 LOC + 1 week training, `a66d99a` §5
P4): only remaining hypothesis for -50%+ ITL on sm_89 W4 decode per
`61c9666` architectural analysis. UNPROVEN until executed. Now
unblocked via `df37a68` #34 CLI surface.

### Research / planning (Claude-doable)

- **W3/W2 quantization research** (P3 in `09ae5a5`): direct weight
  footprint reduction for ITL ceiling. -25-50% ITL ceiling per quant
  level. No immediate impl path; needs PPL gate methodology.
- **#36 PrefixAware revisit**: 3 follow-up paths documented in
  `9bbc441` (cold_headroom sweep / session_id workload / c=32).
  None P0 since #40 already delivered single-stream gap closure.

## §4 Open decisions awaiting user

1. **PF8.3 strategy**: A (verbatim cascade) vs B (single-template
   mirror). Recommended B per `259277c` Strategy C analysis.
2. **#28 Medusa investment**: 1-week training cost + UNPROVEN
   acceptance rate. Worth it given M_spec classical KILL evidence?
3. **Machete name disambiguation** (still open per `e65a096`):
   user reissued "Machete W4 移植" 4+ times despite Hopper-only
   evidence. Default = Path B-Phase2' (W4+FP8 sm_89 native). If user
   means literal Machete sm_89 backport: 1800-3300 LOC + multi-week
   + KILL near-certain.

## §5 Anti-pattern reminders (skill v1.11.0)

Load-bearing for next session:

- **#28**: tool-output-vs-peer-claim → re-run + raw quote in same response
- **#29**: default test fixtures may be known-broken (verify before relying)
- **#30**: git status BEFORE commit (not just before add) in cooperative session
- **#31**: ANY ARLE surface claim needs raw evidence in same response
  (extends #28 beyond contesting peer; covers CLI flags, kernel
  internals, HTTP routes, baseline checkpoint match, model variants)
- **#32**: peer "Waiting >5min" warrants direct ps/log/curl verify
  (4b30c15 33min wedge evidence)

5 hallucinations sedimented this session — pattern: confident claim
about ARLE/bench surface based on internal recall instead of raw
verification. Even "deterministic computation" (bit-pack arithmetic)
can be hallucinated.

## §6 Session productivity summary

Claude commits today: ~30+ across substantial scope. Codex idle
~18 ticks since `f86d0fd` Phase 1.1 wins entry. Bilateral cooperative
discipline established + working when both agents engaged. Solo
Claude productive but reaching diminishing returns without new user
direction OR codex reactivation.

**Recommended next user action**: pick from §4 open decisions OR
let loop continue self-driving (Claude will keep producing PF8 chain
incremental progress, plus auxiliary research entries).

## §7 Cross-references (start here)

- `docs/index.md` — full Last refreshed line covers all axes
- `de36538` — session retrospective with 4 hallucinations chain
- `b551bea` — skill v1.11.0 canonical anti-patterns
- `a66d99a` — NEW prefill-only FP8 directive (PF8.1-5 substep plan)
- `259277c` — PF8.3 scope analysis (shape mismatch finding)
- `61c9666` — architectural P0.A KILL synthesis (FP8 wrong lever for
  decode)
- `e65a096` — Machete sm_89 BLOCKER 5-point convergent evidence
- `09ae5a5` — strategic priority revision
