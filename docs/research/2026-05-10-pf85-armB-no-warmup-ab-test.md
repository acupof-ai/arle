---
title: PF8.5 Arm B (`INFER_PREFILL_WARMUP=0`) — single-variable A/B testing the warmup-DEPENDENT framing from 0be278f
date: 2026-05-10
type: research
status: in_progress (bench running 11:43-11:45 KST, expected ~11:45 result)
related_tasks: [#44 (closed KILL — this strengthens or refines the framing), #47 (H1' — design implications differ by outcome)]
related_skills: [#34 + #34b (server log first), #38 (warmup clamp)]
---

# PF8.5 Arm B — `INFER_PREFILL_WARMUP=0` A/B

> **Purpose**: convert HYPOTHESIS-grade claim in `0be278f` errors entry
> ("warmup-DEPENDENT not load-DEPENDENT") to evidence via single-
> variable A/B per skill kernel-optimization Phase 5.

## §1 The hypothesis being tested

`0be278f` §"Reconciliation with prior findings" wrote:
> "`0cde63d` PF8.3 RUNTIME KILL framed the failure as 'load-dependent'
> — this bench refines: load-INDEPENDENT, but warmup-dependent.
> Pass 3 warmup (Task #35 substrate) is the trigger."

This was inferred from the log pattern (failures start at Pass 3
warmup B=1, before any user request), but never directly tested. Per
§0 SOLID rule 1 ("推断 ≠ SOLID"): inference from log pattern is
hypothesis-grade evidence. A direct A/B with the warmup escape hatch
converts it.

## §2 Single-variable A/B design

| Arm | INFER_PREFILL_WARMUP | Other env | Expected per hypothesis |
|---|---|---|---|
| A (baseline, prior bench) | 1 (default) | INFER_HYBRID_W4A8_PREFILL=1 INFER_MARLIN_W4_FP8_PREFILL=1 | KILL (5878 failures from B=1 warmup) |
| **B (this run)** | **0** (escape hatch) | same | **HEALTHY at conc=1, possibly KILL at sustained load** |

**If hypothesis correct**: Arm B should produce healthy bench output
(real TTFT/ITL numbers, no kernel failures during user requests at
conc=1) since the warmup phase is skipped.

**If hypothesis wrong**: Arm B should also fail with kernel errors
either at first user request or at sustained-load. Then the framing
needs revision — the per-call workspace alloc itself is broken,
warmup is just the first place it surfaces.

## §3 Bench config (matches Arm A exactly except the env)

```bash
RUST_MIN_STACK=33554432 \
  INFER_PREFILL_WARMUP=0 \                          # ← single-variable change
  INFER_HYBRID_W4A8_PREFILL=1 \
  INFER_MARLIN_W4_FP8_PREFILL=1 \
  setsid target/release/infer \
    --model-path infer/models/Qwen3-4B-W4-hybrid-zpfix \
    --port 8000

guidellm benchmark run \
  --target http://127.0.0.1:8000 \
  --model infer/models/Qwen3-4B-W4-hybrid-zpfix \
  --processor infer/models/Qwen3-4B-W4-hybrid-zpfix \
  --profile concurrent --rate 1 --max-seconds 60 --warmup 5 \
  --random-seed 20260416 \
  --data 'prompt_tokens=512,...,output_tokens=128,...' \
  --output-dir bench-output/2026-05-10-pf85-armB-no-warmup \
  --backend openai_http \
  --backend-kwargs '{"validate_backend": "/v1/models", ...}'
```

## §4 Decision matrix

### §4.1 Outcome 1: Arm B HEALTHY (real TTFT, 0 kernel failures)

→ Confirms warmup-DEPENDENT framing in `0be278f`
→ Implies a non-warmup PF8.3 deployment IS viable at conc=1
→ Path forward: design a PF8.3 substrate that skips Pass 3 warmup
  (or uses different warmup shapes that don't trigger the alloc bug)
→ Task #47 H1' refactor remains BLOCKED but with narrower scope:
  fix the Pass 3 warmup interaction, not the per-call alloc path
→ Updates `0be278f` errors entry §Rule with confirmed framing

### §4.2 Outcome 2: Arm B also KILL (failures at first user request)

→ REFUTES warmup-DEPENDENT framing — per-call alloc is the actual
  bug, warmup just surfaces it first
→ Updates `0be278f` errors entry §Rule with corrected framing
→ Confirms PF8.3 substrate is fundamentally broken on sm_89 16GB
  regardless of warmup state
→ Task #47 H1' refactor BLOCKED with broader scope: per-call
  workspace alloc cannot work; needs ground-up redesign
→ Strengthens "static-scratch path is broken across W4A16 + PF8.3"
  conclusion from da7f5a2/d09623a Task #43 Arm A
→ Adds n=2 evidence for SKILL #29 framing-decay pattern (my own
  errors entry's hypothesis was wrong — caught within 30 min by
  follow-up A/B)

### §4.3 Outcome 3: Arm B partial (HEALTHY at conc=1 but degraded)

→ Mixed signal — warmup is contributing but not exclusive cause
→ Need additional A/B at conc=2,4 to disambiguate
→ Records as REVIEW window

## §5 Skill discipline applied

This A/B addresses §0 SOLID rule 6 (framing 多角度交叉, wall-clock
ground truth) by NOT trusting my own inferred framing without testing.
The previous PF8.5 bench (Arm A) reported "5385 successful" but those
were actually all-failed (broken signal); this Arm B will produce
either HEALTHY (real numbers) or KILL (real numbers + log failures) —
both are evidence-grade outcomes not dependent on bench-tool framing.

## §6 Status

**CLOSED — Outcome 2: REFUTES warmup-DEPENDENT framing.**

## §7 Result (11:43 KST)

```text
=== POST-BENCH HEALTH CHECK (Arm B INFER_PREFILL_WARMUP=0) ===
Pass 3 warmup status: DISABLED by INFER_PREFILL_WARMUP=0
Total kernel failures in server log: 5959 (vs Arm A 5878 — even more)
First 3 failures (immediately at user requests, 11:42:17.710-732):
  Request 0: prefill batch failed: gemm_w4_fp8_marlin_cuda failed code 2
  Request 1: prefill batch failed: gemm_w4_fp8_marlin_cuda failed code 2
  Request 2: prefill batch failed: gemm_w4_fp8_marlin_cuda failed code 2

Bench-tool stats: TTFT/ITL still 0.0 ms (broken-signal artifact, all reqs failed)
Throughput: 99.3 req/s (of failed requests)
```

### §7.1 Verdict per §4.2 decision matrix

**Outcome 2: Per-call workspace alloc IS the bug, not warmup.**

| Arm | Warmup | Kernel failures | Failures start | TTFT mdn |
|---|---|---:|---|---:|
| A (baseline) | ON | 5878 | warmup B=1 | 0.0 ms (broken) |
| **B (this)** | **OFF** | **5959** | **Request 0** | **0.0 ms (broken)** |

The 81 extra failures in Arm B (5959 vs 5878) suggest warmup at least
absorbed SOME failure attempts in Arm A, but:
1. Warmup did NOT cause the kernel bug — Arm B has more failures
2. Warmup did NOT prevent it — both arms produce identical per-request
   failure mode (failure at Request 0, code 2 cudaErrorMemoryAllocation)
3. Both arms produce **broken bench-tool signal** (5385/5959 "successful"
   = failed requests with 0 latency)

### §7.2 Implications

**`0be278f` errors entry framing CORRECTION required**:
- ❌ "load-INDEPENDENT but warmup-DEPENDENT (Pass 3 is the trigger)"
- ✅ "**Per-call workspace allocation in `gemm_w4_fp8_marlin_cuda` is
     systematically broken on sm_89 16GB**; warmup is just the first place
     it surfaces because warmup runs before user requests; load-INDEPENDENT
     and warmup-INDEPENDENT — the kernel itself is broken regardless of
     when first invoked"

**Strengthens `da7f5a2` + `d09623a` Task #43 Arm A finding**: static-scratch
/ per-call workspace alloc path is broken across W4A16 (degraded) +
PF8.3 (outright kill) on sm_89 16GB.

**Task #47 H1' refactor scope expansion** (per §4.2):
- BLOCKED with BROADER scope — per-call alloc cannot work on this
  hardware, needs ground-up redesign
- Original H1' design (make MarlinScratch default-on for PF8) is
  empirically broken whether warmup is on or off
- Future redesign must address: workspace allocation must NOT use
  per-call cudaMalloc on sm_89 16GB; needs pool-allocated buffers
  with size known at allocation time

### §7.3 SKILL implications

**SKILL #29 (default broken fixtures / framing decay) → n=6 evidence**
(was n=5 from `d09623a`). Pattern: my own `0be278f` errors entry's
"warmup-DEPENDENT" framing was inferred from log pattern without direct
A/B; 30-minute follow-up A/B refuted it. **Single-source artifact
inference (server log alone) misled me into a directionally-wrong root-
cause framing.** Same pattern as `b956f3a` (test/fixture mismatch) and
`d09623a` (CSV not parsed).

**Procedural rule strengthening for future ticks**: when an errors entry
makes a load-bearing claim about *which* of multiple correlated phases
caused a failure (warmup vs first-request vs sustained-load), run a
single-variable A/B with the relevant escape hatch BEFORE committing
the framing. The 30-minute cost of the A/B prevents downstream design
work (Task #47 H1' redesign) from being scoped to the wrong root cause.

### §7.4 Why `0be278f` PF8.5 KILL verdict still stands

The KILL outcome itself is unchanged — both arms produce the same
"server cannot serve any PF8 request" outcome at conc=1. What changes
is the EXPLANATION:
- Original framing: "warmup is the trigger" → fix warmup, kernel works
- Corrected framing: "kernel is broken regardless" → cannot fix via
  warmup tuning; must redesign workspace allocation
- Decision: Task #44 PF8 chain stays CLOSED-KILL; Task #47 H1' BLOCKED
  with BROADER scope; pickup pivot to #28 Medusa unchanged

The verdict is more robust than originally framed.

