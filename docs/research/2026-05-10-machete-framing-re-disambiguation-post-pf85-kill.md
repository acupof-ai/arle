---
title: Machete W4 framing re-disambiguation post PF8.5 KILL — canonical mapping (aa9f72e) broken, need alternative path
date: 2026-05-10
type: research
status: open (user-blocked re-disambiguation; cron-loop directive reissued 4× verbatim, current canonical Machete = PF8 path is dead)
related_tasks: [#28 (Medusa, current P1), #44 (PF8 KILL closure), #47 (PF8.3 H1' BLOCKED)]
related_memory: [aa9f72e (canonical disambiguation), e65a096 (Machete sm_89 BLOCKER), 61c9666 (architectural insight)]
---

# Machete W4 framing re-disambiguation post PF8.5 KILL

> **Purpose**: User has reissued the cron-loop directive 4× consecutively
> at 13:11 KST stating the main axis is "Machete W4 kernel 移植 from
> vLLM ... for sm_89 W4A8 优化 (预估 -20-40% ITL vs current Marlin)".
> Per CLAUDE.md memory `aa9f72e`, this was canonically mapped to PF8
> chain Strategy A'. **PF8 chain CLOSED-KILL yesterday** per `0be278f`
> + `7ed8160` + `06b7437` + `d8b2870`. The canonical mapping is now
> broken; need explicit re-disambiguation.

## §1 The literal directive

Verbatim from user's reissue:
> "当前主轴: Machete W4 kernel 移植 from vLLM — port machete from
> vllm-project/vllm to ARLE crates/cuda-kernels for sm_89 W4A8 优化
> (预估 -20-40% ITL vs current Marlin)"

## §2 Why the canonical PF8 mapping is dead

Per memory `aa9f72e` (CANONICAL DISAMBIGUATION 2026-05-09):
> "5+ user reissuances of literal 'Machete W4 移植' formally mapped
> to Path B-Phase2' (PF8 chain Strategy A') per `e65a096` Hopper-only
> blocker, future ticks won't re-litigate"

But the PF8 chain status changed yesterday:
- `0be278f` — PF8.5 SUBSTRATE-KILL at conc=1 (5878 kernel failures)
- `7ed8160` — Arm B confirms warmup-INDEPENDENT (5959 failures even with warmup off)
- `06b7437` — Arm C W4A16 control HEALTHY (rules out infer binary)
- `d8b2870` — Arm D W4A8 control HEALTHY (rules out W4 quant family)

**Per §0 SOLID rule 1 ("推断 ≠ SOLID")**: "future ticks won't
re-litigate" was a context-bound rule. When the underlying context
(PF8 substrate viable) changes, the rule must be revisited. The
canonical mapping is broken; ignoring this would persist a stale
rule against current evidence.

## §3 Architectural reality on sm_89 (per memory 61c9666)

Per `61c9666` ARCHITECTURAL INSIGHT 2026-05-09:
> "W4 decode HBM-bound on weight read (already 4× smaller than
> BF16); FP8 mma helps compute not bandwidth; activation is 0.2%
> of memory traffic in W4 GEMM → user's '-20-40% ITL via FP8 path'
> **structurally infeasible** on sm_89 W4 decode"

This was established before PF8 substrate work began. PF8 work
proceeded on Strategy A' assumption that prefill (compute-bound,
M=2048) could benefit from FP8 mma even if decode couldn't. PF8.5
KILL eliminates THAT axis too.

## §4 Three alternative paths to Machete-class gains on sm_89

Given:
- Literal Machete is sm_90+ (Hopper TMA + WGMMA, blocked per `e65a096`)
- PF8 path is dead (KILL evidence above)
- Decode is HBM-bound (architectural per `61c9666`)
- W4A16 already at 5.8 ms ITL (vs INT8 6.8 ms = -15% gap closed)

Three paths to Machete-class **effective tok/s** improvement:

### §4.1 Path I: Medusa speculative decoding (Task #28, current P1)

**Mechanism**: parallel head verification → 2-3× tok/s at acceptance
≥ 70%. Not literal Machete but **same effective speedup magnitude**
via different mechanism (parallelizes verification, not per-op compute).

**Cost**: ~2-3 days codex + ~1 week Medusa head training.

**Risk**: medium — Medusa training quality + acceptance rate need to
align.

**Time-to-result**: ~2-3 days for first measurement.

### §4.2 Path II: Lower-bit quantization (W3 / W2 / NVFP4)

**Mechanism**: smaller weight footprint → less HBM bandwidth needed
per token → faster decode.

**Cost**: significant — W3/W2 substrates don't exist in tree at
production quality; NVFP4 is sm_100+ only (per `61c9666`).

**Risk**: HIGH — accuracy gates more aggressive than W4A16; quant
calibration risk.

**Time-to-result**: ~2-4 weeks for substrate + bench.

### §4.3 Path III: W4A16 with optimization (closer to physical limit)

**Mechanism**: keep W4A16 path but optimize the marlin kernel
(e.g., fix Hopper-default `BLOCK_M=64, BLOCK_N=64` → tune for sm_89
per skill kernel-optimization Phase 2 hardware sheet). Currently
might be missing 10-20% perf.

**Cost**: low — kernel tuning, ~1-2 weeks.

**Risk**: low — same kernel, different params; well-trodden territory.

**Time-to-result**: ~1-2 weeks.

## §5 Recommendation matrix

| Goal | Recommendation | Time |
|---|---|---|
| Maximum effective tok/s improvement | **Path I (Medusa)** | 2-3 days |
| Resurrect literal "Machete-class" magnitudes | Path II (W3/W2) | 2-4 weeks |
| Squeeze W4A16 closer to physical limit | Path III (kernel tuning) | 1-2 weeks |
| **Default if "Machete 预估 -20-40% ITL" is the goal** | **Path I (Medusa)** | 2-3 days, lowest blocker risk |

## §6 Why Path I (Medusa) is the closest to user's stated goal

User's directive: "预估 -20-40% ITL vs current Marlin".

W4A16 baseline (Arm C): 5.8 ms ITL.

- Path I (Medusa, 2× tok/s at 70% accept): effective ITL halved → ~2.9 ms (~-50%)
- Path II (W3 weight): theoretical ~25% bandwidth reduction → ~4.4 ms (~-25%)
- Path III (W4A16 tune): ~10-20% improvement → ~4.6-5.2 ms (~-10 to -20%)

**Path I most exceeds the -20-40% target.** Path II matches but with
much higher risk + time. Path III matches lower bound with lowest
risk + medium time.

Per "持续累积" + the user's own "world-first" goal: Path I + Path III
in parallel (different code surfaces, no conflict) maximizes coverage.

## §7 Action requested

User to pick:
- **A**: Pickup Medusa (Path I, current P1) — Claude can pre-emptively
  scaffold Alpaca dataset prep (~1-2 hr CPU work) without commitment
- **B**: Pickup W4A16 kernel tuning (Path III) — Claude can scaffold
  the bench harness for tile-param sweep
- **C**: Both A + B in parallel — different code surfaces, no risk
- **D**: Different direction (literal Machete on Hopper, NVFP4 on
  Blackwell, etc.) — would need explicit hardware target

## §8 Memory update needed

Per CLAUDE.md auto-memory rules: when this re-disambiguation is
acknowledged by user, update memory `aa9f72e` to note:
- Canonical mapping (Machete → PF8) was BROKEN by PF8.5 KILL 2026-05-10
- New canonical mapping (Machete → Medusa Path I OR W4A16-tune Path III)
  per user's choice from §7 above

Until user acknowledges, this re-disambiguation stays as research
note (this doc).

## §9 Cross-references

- Memory `aa9f72e` — original Machete = PF8 canonical mapping
- Memory `e65a096` — Machete sm_89 BLOCKER (5-pt evidence)
- Memory `61c9666` — sm_89 W4 decode HBM-bound architectural insight
- `0be278f` PF8.5 SUBSTRATE-KILL errors entry
- `7ed8160` Arm B refutes warmup framing
- `06b7437` + `d8b2870` Arm C+D HEALTHY controls
- `2026-05-10-post-pf85-direction-options.md` — earlier 3-option doc
  (this doc supersedes that with Machete-specific framing)
