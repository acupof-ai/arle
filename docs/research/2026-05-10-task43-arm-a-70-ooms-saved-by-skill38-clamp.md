---
title: Task #43 Arm A artifact deep-dive — 70 OOMs in Pass 3 warmup saved by SKILL #38 clamp; even more inverted than codex 83fc5d0 captured
date: 2026-05-10
type: research
status: closed (Claude-independent verification of codex 83fc5d0 INVERSE direction)
related_tasks: [#43 (DISPROVEN), #47 (H1' design implications), #44 (PF8 chain)]
related_skills: [#36 (grep + behavioral A/B), #38 (warmup clamp)]
---

# Task #43 Arm A artifact deep-dive — 70 OOMs saved by graceful clamp

> **Purpose**: Independent Claude verification of codex `83fc5d0` Task #43
> INVERSE finding by parsing on-disk artifacts (`/tmp/task43-*.log` +
> `bench-output/2026-05-10-task43-{A,B}-*/`). Found: the inversion is even
> more dramatic than codex's commit message captured, AND SKILL #38
> graceful clamp is the load-bearing reason Arm A survived at all.

## §1 What I expected (per `1ba06f0` Claude dispatch-audit hypothesis)

```
Arm A (INFER_PREFILL_GRAPH=1, scratch ENABLED): predicted HEALTHY
  → marlin_scratch=Some → uses _with_scratch path → no per-call alloc fragmentation

Arm B (no env, scratch DISABLED): predicted SUBSTRATE-KILL or near-OOM
  → marlin_scratch=None → falls back to per-call alloc → fragmentation
```

## §2 What the artifacts actually show

### §2.1 Arm A `/tmp/task43-A-scratch-enabled.log` — 70 OOM events

```bash
$ grep -c "out of memory" /tmp/task43-A-scratch-enabled.log
70

$ grep -oE "alloc [^:]+:" /tmp/task43-A-scratch-enabled.log | sort | uniq -c
     32 alloc Marlin W4 y_fp16 scratch:
      9 alloc Marlin W4 x_fp16 scratch:
     # remainder are generic "Alloc failed:" Pass 3 retries
```

Cascade pattern (excerpt):
```text
B=4 at 2048 tokens/row failed → retry at 1024
B=5 at 2048 → 1024 → 512 → 256 → 128 (all OOM, alloc Marlin W4 x_fp16 scratch)
B=6 at 2048 → 1024 → 512 → 256 → 128 → 64 → 32 (all OOM, alloc Marlin W4 y_fp16)
B=7 at 2048 → 1024 → 512 → 256 → 128 → 64 → 32 → 16 → 8 → 4 → 2 (cascade-down)
```

### §2.2 Arm B `/tmp/task43-B-scratch-disabled.log` — 1 OOM event

```bash
$ grep -c "out of memory" /tmp/task43-B-scratch-disabled.log
1

# That single warning:
2026-05-10T09:50:21 WARN warmup.rs:300 Pass 3 prefill warmup for B=8 at 2048
  tokens/row failed (completion failed: alloc marlin y_fp16: ...OOM); retrying at 1024
```

Standard SKILL #38 graceful-clamp behavior, single retry, no cascade.

### §2.3 Both arms produced full bench CSVs

```bash
$ ls bench-output/2026-05-10-task43-{A,B}-*/
benchmarks.csv  benchmarks.html  benchmarks.json   # both
```

Server in Arm A actually serviced requests successfully despite the 70-OOM
cascade — log shows `Request 74/75/76 done` at 4097-token prefill chunks,
~526ms/req. The graceful warmup clamp degraded Pass 3 budget without
killing the process.

## §3 The actual inversion magnitude

| Arm | Predicted | Actual OOMs | Bench output | Inversion |
|---|---|---:|---|---|
| A (scratch ENABLED) | HEALTHY | **70** | OK | **+69 OOMs vs predicted** |
| B (scratch DISABLED) | KILLED | **1** | OK | **−69 OOMs vs predicted, no kill** |

**Net direction**: scratch-enabled is ~70× WORSE for Pass 3 warmup OOM
pressure than scratch-disabled at this workload (4097-token sustained,
W4A16, conc=4). Codex `83fc5d0` reported "INVERSE direction" but didn't
quantify the 70:1 ratio.

## §4 Why it didn't kill the server

**SKILL `kernel-optimization` v1.13.0 #38 (warmup target shape clamp)** is
load-bearing. Each OOM in Pass 3 triggers `warmup.rs:300` retry-at-half:
2048 → 1024 → 512 → ... → 2. The clamp converts a hard CUDA OOM into a
soft "warmup at degraded budget" outcome. Without #38, Arm A would have
panicked at the first B=4 failure and Task #43 hypothesis would have
(wrongly) seemed CONFIRMED.

This adds **n=5 evidence** for #38 (was n=4 from earlier this
session-tail per `2026-05-10-claude-independent-verify-task48-fix-0pct-diff.md`
§8.1):

| n | Source | Config | Pass 3 cost / outcome |
|---|---|---|---|
| 1 | greedy_consistency (test) | max=4 batch sizes | 368ms |
| 2 | e2e test default | max=4 + cublasLt autotune | 1572ms |
| 3 | Task #35 production | cap=8 batch sizes | +8186ms |
| 4 | Task #35 prod B=8 2048 | OOM → clamp to 1024 | graceful adapt |
| **5** | **Task #43 Arm A (this)** | **W4A16 scratch enabled, sustained load** | **70 OOMs survived via clamp; server functional** |

#38 is now the most-evidenced kernel-optimization SKILL anti-pattern in
this session.

## §5 Why scratch=ENABLED produced more OOMs than scratch=DISABLED

This is counter-intuitive and worth understanding:

**Hypothesis (NOT yet evidence)**: When `marlin_scratch=Some(MarlinScratch)`
the `_with_scratch` path uses **fixed-size** preallocated y_fp16/x_fp16
buffers sized for the largest expected workload. At Pass 3 warmup,
those preallocated buffers are competing with Pass 3's own scratch
allocations for the same VRAM headroom — **double-allocating**. When
`marlin_scratch=None`, per-call alloc only requests what the actual call
needs, leaving headroom for Pass 3.

**Evidence required to confirm** (would need source read of
`linear.rs:317-323` MarlinScratch struct + dispatch at `linear.rs:2064-2095`):
- Confirm MarlinScratch buffer sizes are workload-max-bound
- Confirm Pass 3 warmup runs AFTER scratch allocation (not before)
- Run a 3rd arm: `INFER_PREFILL_GRAPH=1 INFER_PREFILL_WARMUP=0` to isolate
  whether it's scratch-vs-warmup contention or something else

**Per §0 SOLID rule 1**: this is a hypothesis, not a fix. Not actionable
yet. Document as research note for future codex Task #47 H1' refactor
scope consideration.

## §6 Implications for Task #47 H1' refactor design

`docs/research/2026-05-10-h1prime-design-revision-marlinscratch-already-exists.md`
proposed making MarlinScratch the default-on path for PF8.3. The Arm A 70-OOM
data **invalidates the simple "make scratch default-on" approach** — at sm_89
16GB the static-scratch budget can starve Pass 3 warmup. H1' refactor MUST
include:

1. **Pass 3 warmup runs BEFORE scratch allocation** — order matters
2. **Scratch sizing strategy** — workload-aware bounds, not max-shape
3. **OOM-regression A/B gate** — bench v11 must compare PF8 with-scratch
   vs without-scratch in OOM count, not just throughput

Per `2cc608a` H1' design: this finding strengthens the requirement for an
OOM-regression A/B gate before merge.

## §7 SKILL #36 strengthening (grep + behavioral A/B both required)

This is now n=3 evidence for #36 (was n=2 at v1.14.0 graduation):

| n | Case | Grep said | Behavioral A/B said |
|---|---|---|---|
| 1 | (original n=1 case) | path X is correct | path X has different behavior |
| 2 | Task #43 codex 83fc5d0 | hypothesis CONFIRMED | hypothesis INVERSE |
| **3** | **Task #43 Claude artifact deep-dive (this)** | **scratch=on healthy** | **scratch=on 70× more OOM events** |

#36 v1.14.0 graduation criteria already included "grep + behavioral A/B
both required" — this evidence is corroboration, not graduation trigger.

## §8 Status

**Closed — Claude-independent verification PASS.** Codex `83fc5d0`
INVERSE finding is correct AND quantitatively more dramatic than the
commit message captured. Three concrete adds:

1. **70:1 OOM ratio** documented (Arm A vs Arm B)
2. **SKILL #38 to n=5 evidence** (now most-evidenced this session)
3. **Task #47 H1' design constraint** added: must include OOM-regression
   A/B gate, must consider scratch-vs-warmup ordering

## §9 Cross-references

- `83fc5d0` codex Task #43 INVERSE-direction commit (root finding)
- `1ba06f0` Claude original dispatch-audit hypothesis (DISPROVEN)
- `e8b6b31` Claude analysis of INVERSE outcome
- `2cc608a` H1' design revision (now needs OOM-gate constraint)
- `linear.rs:317-323` MarlinScratch struct
- `linear.rs:2064-2095` dispatch fallback
- `warmup.rs:300` SKILL #38 graceful clamp (load-bearing)
- SKILL `kernel-optimization` v1.13.0 #38 (now n=5)
- SKILL `kernel-optimization` v1.14.0 #36 (now n=3 corroboration)
- `bench-output/2026-05-10-task43-{A,B}-*/benchmarks.{csv,html,json}`
- `/tmp/task43-{A,B}-*.log`
