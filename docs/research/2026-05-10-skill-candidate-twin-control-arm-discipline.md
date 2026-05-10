---
title: SKILL kernel-optimization Phase 5 sub-rule candidate — twin-control-arm discipline for KILL verdicts
date: 2026-05-10
type: research
status: candidate (n=1 evidence, awaits n+1 from future KILL/A-B chains)
related_skills: [#34 (greedy single-request not sufficient), #36 (grep + behavioral A/B both required), #29 (default broken fixtures / framing decay)]
---

# SKILL Phase 5 sub-rule candidate — twin-control-arm discipline

> **Purpose**: sediment the procedural learning from the PF8.5 4-arm
> A/B chain (Arms A/B/C/D over `0be278f` → `7ed8160` → `06b7437` →
> `d8b2870`) as a SKILL candidate for future evidence accretion.

## §1 The pattern

When a single-arm bench produces KILL (kernel failure, OOM, broken
output), the temptation is to commit a root-cause framing immediately
based on log inspection. **Single-arm KILL is hypothesis-grade evidence
about the cause.** Confounders that look ruled-out from log alone:

- Hardware capability (could be GPU OOM at this scale)
- Binary correctness (could be infer crash on this model class)
- Tool quirk (guidellm metric reporting could be broken at edge cases)
- Workload scale (could be conc=1 has different memory profile)
- Warmup interaction (could be Pass 3 surfaces failure first)

Each of these requires a SEPARATE control arm to definitively rule out.

## §2 The proposed sub-rule (Phase 5 extension)

**When a single-arm KILL is observed, IMMEDIATELY (before committing
root-cause framing) run TWO control arms**:

1. **Nearest-relative control**: same substrate family, different
   variant (e.g. PF8 hybrid → W4A8 marlin). Tests "is the bug
   specific to THIS substrate variant or does it span the family?"
2. **Architecturally-different control**: same workload/binary/
   hardware, completely different substrate (e.g. PF8 hybrid →
   W4A16 marlin). Tests "is the bug substrate-specific or
   binary/hardware/tool-broken?"

**Both controls HEALTHY → bug isolated to single-arm substrate**
(IRONCLAD attribution).

**Either control unhealthy → broaden the search**. Refuse to commit
root-cause framing until at least one control is HEALTHY (proves
the test infrastructure CAN produce healthy signal at this config).

## §3 PF8.5 evidence (n=1)

| Arm | Path | Failures | TTFT | Notes |
|---|---|---:|---:|---|
| A (single-arm KILL) | PF8 hybrid + warmup ON | 5878 | 0.0 (broken) | Initial framing: "warmup-DEPENDENT" (`0be278f`) |
| B (warmup escape hatch) | PF8 hybrid + warmup OFF | 5959 | 0.0 (broken) | REFUTES warmup framing (`7ed8160`) |
| **C (architecturally-different)** | **W4A16-marlin** | **0** | **66.0 ms** | **Confirms binary/hardware healthy (`06b7437`)** |
| **D (nearest-relative)** | **W4A8-marlin** | **0** | **54.2 ms** | **Confirms W4 quant family healthy (`d8b2870`)** |

Without Arms C+D, the framing in `0be278f` could plausibly have been:
- "infer binary broken on sm_89 16GB" (refuted by Arm C)
- "Pass 3 warmup interaction breaks PF8" (refuted by Arm B)
- "W4 quantization can't work on this hardware" (refuted by Arm D)

With all 4 arms, the framing is bounded: **PF8.3 hybrid substrate
specifically is broken; the bug is in `gemm_w4_fp8_marlin_cuda`
per-call workspace allocation**, not in any of: infer binary, Pass 3
warmup, hardware, tool, W4 quant family at large.

## §4 Cost-benefit

### §4.1 Cost of running both controls

- **Wall-clock**: ~10 min (2 arms × ~5 min each)
- **GPU time**: ~2 min (server bench windows)
- **Disk**: ~70 MB bench artifacts
- **Cognitive**: trivial (modify 1-2 env vars / model paths)

### §4.2 Cost of NOT running both controls

- **Framing decay**: documented at SKILL #29 n=4/5/6 — single-source
  artifact framing decays into wrong root-cause attribution
- **Downstream wasted work**: Task #47 H1' refactor was scoped against
  warmup-DEPENDENT framing; without Arm B the refactor design would
  target warmup interaction (wrong), wasting codex pickup time
- **Trust erosion**: KILL framings that turn out wrong require
  follow-up self-corrections (caught 3× in this session-tail)

### §4.3 ROI calculation

Pre-empting one wrong-framing follow-up = preventing ~30 min of
self-correction docs work + ~3-4 hours of misdirected codex pickup
on Task #47 H1' redesign. Cost ~10 min wall-clock to gain ~4 hours
saved ≈ **24:1 ROI**, before counting downstream cooperative loop
benefits.

## §5 Why this isn't covered by existing SKILL items

- **#29** (default broken fixtures): about TEST FIXTURES specifically,
  not bench-arm design patterns
- **#34** (single-request necessary not sufficient): about SHAPES /
  CONCURRENCIES, not control-arm topology
- **#36** (grep + behavioral A/B both required): about static-vs-
  behavioral evidence, not bench-arm count

This sub-rule complements all three: #36 says "use behavioral A/B",
this candidate says "ONE behavioral arm is not enough when the result
is KILL — use TWO controls to bound the broken surface."

## §6 Detection rule

Reviewer checklist for "errors entry from KILL bench":
- [ ] Does the entry cite at least ONE healthy control arm with same
      hardware + binary + workload, different substrate?
- [ ] Does the entry cite a NEAREST-RELATIVE control proving the
      broken surface is bounded?
- [ ] If only one control: does the entry explicitly say WHY a
      twin control isn't needed (e.g. "well-known healthy config
      from prior wins entry")?
- [ ] Does the framing's root-cause claim sit WITHIN the bounded
      surface (not generalized beyond)?

## §7 Status

**Candidate (n=1)**. Evidence accretion plan:
- n=2: any future KILL bench in this repo where twin-control
  discipline is applied + produces refined framing vs naive single-arm
- n=3: external case (e.g. Metal backend KILL surface, KV-tier KILL)
  where the same pattern manifests
- Graduation criteria: at least n=2 INDEPENDENT KILL diagnoses where
  twin-control changed the framing meaningfully

## §8 Cross-references

- `0be278f` original PF8.5 KILL (Arm A) — this is the "naive single-arm
  KILL" before the discipline was applied
- `7ed8160` Arm B (escape hatch refuted warmup framing)
- `06b7437` Arm C (architecturally-different control)
- `d8b2870` Arm D (nearest-relative control + 4-arm matrix)
- SKILL `kernel-optimization` v1.12.0 #34 + #34b (single-request
  not sufficient — companion at shape/concurrency axis)
- SKILL `kernel-optimization` v1.14.0 #36 (grep + behavioral A/B —
  companion at evidence-type axis)
- SKILL `kernel-optimization` v1.15.0 #35 (root-cause-TBD canary —
  companion at acceptance-gate axis)
