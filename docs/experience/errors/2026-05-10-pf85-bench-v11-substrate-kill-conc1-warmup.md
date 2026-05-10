# PF8.5 license bench v11 — SUBSTRATE-KILL at conc=1 from Pass 3 warmup

## Context

Date: 2026-05-10 11:33-11:34 KST
Bench: `bash scripts/pf85_bench_v11_user.sh` (Claude runs, see
`docs/research/2026-05-10-pf85-claude-runs-bench-v11.md`)
Substrate under test: PF8.3 (commit 11763ba), enabled via env vars
`INFER_HYBRID_W4A8_PREFILL=1` + `INFER_MARLIN_W4_FP8_PREFILL=1`
Workload: guidellm conc=1 60s sustained-load, prompt=512 / output=128
Model: `infer/models/Qwen3-4B-W4-hybrid-zpfix` (PF8 hybrid checkpoint)

## Symptoms

```text
HEALTH CHECK
BENCH:   success=5385  fail=0
SERVER:  kernel_failures=5890  live_kernel_failures=5878
VERDICT: SUBSTRATE-KILL
```

Bench-tool reported 5385 "successful" requests but server log reveals
5878 `gemm_w4_fp8_marlin_cuda failed with code 2` errors. TTFT median
in stats table reports **0.0 ms** — the broken-signal artifact (failed
requests have 0 latency).

Earliest failures (all at warmup, before any user request):

```text
2026-05-10T11:33:10.769533+08:00 WARN warmup.rs:300
  Pass 3 prefill warmup for B=1 at 1540 tokens/row failed
  (completion failed: gemm_w4_fp8_marlin_cuda failed with code 2);
  retrying at 770 tokens/row
2026-05-10T11:33:10.770489+08:00  ... B=1 770 → 385
2026-05-10T11:33:10.770666+08:00  ... B=1 385 → 192
2026-05-10T11:33:10.794793+08:00  ... B=2 1540 → 770
...
```

After warmup, every user request also fails:

```text
2026-05-10T11:33:27.039141+08:00 ERROR prefill.rs:627
  Request 0: prefill batch failed: gemm_w4_fp8_marlin_cuda failed
  with code 2
2026-05-10T11:33:27.054767+08:00 ERROR ... Request 1
2026-05-10T11:33:27.067630+08:00 ERROR ... Request 2
[5878 total failures]
```

## Root Cause

**Hypothesis (n=1 evidence, needs codex H1' verification)**:
PF8.3 `gemm_w4_fp8_marlin_cuda` requires per-call workspace scratch
allocation that fails immediately under Pass 3 warmup pressure. Same
class as Task #43 Arm A finding (`da7f5a2` + `d09623a`):

| Workload | OOMs | TTFT |
|---|---:|---:|
| Task #43 Arm A: W4A16 with `INFER_PREFILL_GRAPH=1` (static scratch) | 70 | 1502 ms |
| Task #43 Arm B: W4A16 default (lazy alloc) | 1 | 94 ms |
| **PF8.5 v11: PF8.3 substrate (this entry)** | **5878** | **0.0 ms (broken)** |

PF8.3 is the WORST of the three: hard kernel-failure with `code 2`
even at warmup B=1, not just degraded perf. **Static-scratch path is
broken on sm_89 16GB GPU**, regardless of whether the static buffer
is W4A16 (Task #43 Arm A degradation) or PF8.3 (this — outright kill).

## Reconciliation with prior findings

- `57c37b5` H8 DISPROVEN said "kernel WORKS at conc=1 single requests"
  — that was a SINGLE manual curl, no Pass 3 warmup pressure. With
  Pass 3 warmup running first (which the bench script enables), the
  kernel fails immediately.
- `0cde63d` PF8.3 RUNTIME KILL framed the failure as "load-dependent"
  — this bench refines: load-INDEPENDENT, but warmup-dependent.
  Pass 3 warmup (Task #35 substrate) is the trigger.
- Task #47 H1' refactor was designed to make MarlinScratch default-on
  for PF8. **This bench evidence says default-on would AMPLIFY the
  failure**, not fix it. H1' design needs revision before pickup.

## Fix

**No code fix this entry.** This is the LICENSE bench that the user
has been blocked on for many ticks. KILL outcome triggers:

1. **Task #44** (PF8 chain) → CLOSED with KILL
2. **Task #47** (PF8.3 H1' static-scratch refactor) → BLOCKED pending
   redesign; static-scratch default-on path is empirically broken
3. **Pickup queue P1 pivot**: from `#47 H1' refactor` → `#28 Medusa
   scaffold`
4. **PF8.3 substrate stays in tree** as opt-in (default off per
   `db063ff`) for future H1' redesign work
5. PF8.5 license tooling (`scripts/pf85_bench_v11_user.sh` +
   companions) preserved for re-bench after future fix

## Rule

When testing substrate that requires per-call workspace allocation
on memory-constrained GPUs (sm_89 16GB), **Pass 3 warmup must
complete cleanly before declaring kernel correctness**. A single-
request manual curl (per H8 DISPROVEN) is NECESSARY but NOT
SUFFICIENT — Pass 3 warmup adds memory pressure that exposes
allocation failures before any user request arrives. This generalizes
SKILL `kernel-optimization` v1.12.0+ #34 ("greedy single-request not
sufficient"): now requires Pass 3-warmup-up bench BEFORE the per-
request workload, since warmup is itself a stress test.

This bench would have produced n=4 evidence for SKILL `kernel-
optimization` v1.13.0+ #38 (warmup target shape clamp) — the retry-
at-half cascade fired 5+ times for B=1 alone. But the cascade
ultimately couldn't save the substrate (vs Task #43 Arm A which
served degraded but functional). This case strengthens the framing
of #38: graceful clamp converts hard-kill into degraded survival
ONLY when the per-call path is intermittently broken; if the per-
call path is **systematically broken** at all shapes, clamp doesn't
save the substrate.

## Cross-references

- `0cde63d` — original PF8.3 RUNTIME KILL (sustained-load framing)
- `57c37b5` — H8 DISPROVEN (single-request, no Pass 3 framing)
- `11763ba` — PF8.3 substrate landing (12 files +3936/-13)
- `da7f5a2` — Task #43 Arm A 70:1 OOM ratio
- `d09623a` — Task #43 self-correction 16× TTFT degradation
- `2cc608a` — H1' design REVISION (now needs PF8.5 kill rebound)
- `2026-05-10-pf85-claude-runs-bench-v11.md` — this run's plan + risk
- `bench-output/2026-05-10-pf83-treatment-conc1-FINAL/` — raw outputs
- `/tmp/pf83-FINAL-treatment.log` — server log
- SKILL `kernel-optimization` v1.12.0 #34 + #34b (server log first)
- SKILL `kernel-optimization` v1.13.0 #38 (warmup clamp framing)
