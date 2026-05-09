---
title: 2026-05-10 session retrospective — 4 Claude hallucinations + cooperative-discipline evolution
date: 2026-05-10
type: research
status: ongoing-session-mid-bench-r2
---

# 2026-05-10 session retrospective — 4 Claude hallucinations + cooperative-discipline evolution

> Session-mid retrospective consolidating 4 Claude hallucinations
> caught + cooperative-discipline evolution. Written while bench r2
> runs in background. Future session start should begin from this
> entry's "Strengthened skill v1.10.0+ rules" section.

## §0 Four hallucinations this session (all caught + sedimented)

| # | Tick | Hallucination | Reality | SUPERSEDED in |
|---|------|---------------|---------|---------------|
| 1 | `0f4d0ae` | `--max-waiting-requests` CLI flag exists at main.rs:133 | Flag never existed; line 133 is `scheduler_mixed_policy` | `ee2c5b0` errors entry (skill v1.10.0 #28) |
| 2 | `43bda9c` | W4A16 marlin_kernel.cu has `max_par × 64 × n` reduce buffer (Substep 1.2 atomic_add target) | W4A8 has buffer (line 258), W4A16 only `int* locks` | `0d63a52` errors entry (Substep 1.2 KILLED in design) |
| 3 | `4b30c15` | ARLE has `/health` endpoint (recommended in unstick brief) | ARLE has `/healthz`+`/readyz` (k8s convention) — verified `router.rs:68-69` | `c3bb82b` research entry |
| 4 | `5bf0e20` | 2026-05-09 baseline-B5 (zpfix variant) is comparable to newdequant-r1 (sym-g128 variant) → claimed ITL -35.9% | Different model variants (zpfix vs sym-g128); ITL 17.77 vs 11.76 ms baselines = different scales | `d387b03` SUPERSEDED notice; codex's `f86d0fd` correct -3.2% Δ |

## §1 Common failure mode

Each hallucination = **confident claim about ARLE/bench surface based
on internal recall of "how things usually work" without grepping the
actual code/files**. Each plausible but wrong because ARLE-specific.

The 4 categories of false claims:
1. **CLI surface** (#1: flag existence)
2. **Kernel internals** (#2: which kernel has which buffer)
3. **HTTP endpoints** (#3: /health vs /healthz)
4. **Baseline comparison** (#4: which checkpoint each baseline used)

Failure mode generalizes beyond file content. Skill v1.10.0 #28 was
originally about "verify raw output not memory recall" for CLI flags;
needs to extend to ALL surface claims including baseline selection.

## §2 Strengthened skill v1.10.0+ rules

**Rule 1 (skill v1.10.0 #28 ORIGINAL)**: When tool output contradicts
peer agent's investigation, RE-RUN tool + quote literal raw output in
same response.

**Rule 2 (strengthened from c3bb82b)**: ANY claim about ARLE's
surface (CLI flags, file structure, kernel internals, HTTP routes,
scheduler config defaults) MUST be backed by raw `grep`/`Read` output
IN THE SAME RESPONSE making the claim. Generic conventions don't
apply — ARLE's implementation may differ.

**Rule 3 (strengthened from d387b03 — NEW this session)**: ANY claim
about bench/comparison surface (baseline checkpoint match, model
variant, workload size, license-gate references) MUST be backed by
raw `cat command.txt` or `head wins-entry` output IN THE SAME RESPONSE.
Different baselines may use different model variants — never assume
comparable without verifying.

**Rule 4 (anti-pattern #30 candidate from 0d63a52)**: `git status
--short` BEFORE `git commit` (not just before `git add`). Cooperative
process may stage files between your add and commit. Verify staged
set with `git diff --cached --stat` before committing.

**Rule 5 (NEW from 4b30c15)**: When peer agent shows "Waiting for
background terminal X minutes" with no observable progress: don't
trust the timer. Directly verify:
  - `ps -p $PID` → process alive?
  - `ls -la <log>` → log growing?
  - `curl <expected-endpoint>` → service responding?
If process dead, send unstick brief proactively.

## §3 Recovery patterns (what worked)

For each hallucination, recovery via the same pattern:
1. **Discovery via direct evidence** (raw tool output, often by codex
   challenging or by Claude self-audit)
2. **SUPERSEDED notice or errors entry** with the wrong claim
   explicitly cited + corrected via raw evidence
3. **Skill rule sharpening** — anti-pattern catalog or memory file
   updated with the new failure mode
4. **No revert / no force-push** — durable history preserves the
   error trail; future readers see both the wrong claim and the
   correction

This pattern works because:
- Codex's empirical discipline (try the endpoint, run the bench)
  catches Claude's recall errors
- Claude's analytical discipline (audit codex's claims, write
  research entries) catches codex's mid-investigation pivots
- Both agents converge on raw-evidence-required as the SOLID
  principle

## §4 Phase 1 substantive outcome (mid-session)

Despite the 4 hallucinations, Phase 1 Substep 1.1 LANDED + LICENSED
on conservative gate:
- Codex's `09ae5a5` (accidentally bundled with Claude doc) = port
  - `crates/cuda-kernels/csrc/gemm/marlin_dequant.cuh` 651 LOC
  - hybrid strategy: single-file + verbatim namespace vllm shim
- `994a294` build-restore (Claude focused-commit demonstrating
  new discipline)
- `f86d0fd` codex bench wins entry — Δ vs 2026-05-08 W4A16 baseline:
  - TTFT p50: -7.0% (2565.4 → 2386.3 ms)
  - ITL p50: -3.2% (11.76 → 11.38 ms) — matches e59beb5 conservative -3-8%
  - out tok/s: +2.1% (191.16 → 195.17)
- Currently re-running n=2/r3 for σ-tight n=3 license-grade evidence

Phase 1 Substep 1.2 KILLED in design (raw grep proves W4A16 has no
reduce buffer to eliminate; W4A8 alt deferred to prefill-only FP8).

## §5 Next-axis priority (revised post P0.A KILL + Phase 1 LANDED)

| Priority | Path | Status |
|----------|------|--------|
| P0 (LANDED) | Phase 1 Substep 1.1 dequant.h port | -3.2% ITL conservative win |
| P1 | NEW prefill-only FP8 directive (~700 LOC, codex P0.A 5.21× evidence) | -8-16% TTFT separate axis |
| P2 | #34 CLI surface (~30-50 LOC, ~1h) | unblocks #28 spec decode |
| P3 | W3/W2 quantization research | -25-50% ITL ceiling per quant level (research) |
| P4 (long-term) | #28 Medusa scaffold | -50%+ ITL hypothesis (1-week training, UNPROVEN per M_spec KILL) |
| KILLED | Phase 2' W4+FP8 sm_89 native (decode) | structurally infeasible per `61c9666` |
| KILLED | Machete sm_89 backport | Hopper-only per 5-pt evidence in `e65a096` |
| KILLED | Phase 1 Substep 1.2 atomic_add | wrong scope (W4A8 alt only) |
| KILLED | #36 PrefixAware as default | warm/cold p95 +17%/+114%, starvation 4.56→8.33× |
| KILLED | M_spec classical external draft | -73%/-46% tok/s per M_spec plan (#27 closed) |
| KILLED | #33 KV W4A8 scalar unpack | 1.12× < 1.5× kill gate |

## §6 Pattern lesson — bilateral cooperative discipline

Cooperative discipline only works when BILATERAL:
- Claude's hallucinations need codex's empirical catches
- Codex's mid-investigation pivots need Claude's analytical audits
- Both need raw-evidence-in-same-response as the shared rule

Examples this session:
- Codex caught Claude's `--max-waiting-requests` hallucination → `ee2c5b0`
- Claude caught codex's reverted `Substep 1.1 commit attribution` issue → `0d63a52`
- Codex caught Claude's `/health` hallucination → `c3bb82b`
- Claude caught codex's "Waiting >33min" wedged poll → `4b30c15`
- Codex caught Claude's wrong baseline comparison → `d387b03`

The recovery cost is small (one SUPERSEDED notice per hallucination,
one errors entry per discipline violation) compared to the velocity
gained from parallel cooperation.

## §7 Status (mid-session, bench r2 in flight)

Bench r2 currently running for σ-tight n=3 Phase 1.1 wins update.
Server PID 1816430 up (`/readyz` OK). Bench PID 1816494 alive at
1m 15s elapsed (just past setup, in 120s window). After r2 + r3
complete, n=3 wins entry update will land.

Next pickup queue (post σ-tight): NEW prefill-only FP8 directive
(P1, codex P0.A 5.21× evidence) OR #34 CLI surface (P2, ~30-50 LOC).

## §8 Cross-references

- All 4 hallucination errors entries: `ee2c5b0`, `0d63a52` §"Substep 1.2 rescope", `c3bb82b`, `d387b03`
- Phase 1.1 LANDED: `09ae5a5` (substrate) + `994a294` (build-restore) + `f86d0fd` (wins entry)
- Strategic pivot: `09ae5a5` (Phase 0 P0.A KILL synthesis)
- Memory file: `~/.claude/projects/.../memory/feedback_git_status_before_commit_in_cooperative.md`
- Skill v1.10.0 anti-patterns: `.claude/skills/kernel-optimization/SKILL.md`
