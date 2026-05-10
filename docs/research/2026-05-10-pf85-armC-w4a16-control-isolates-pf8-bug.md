---
title: PF8.5 Arm C (W4A16 control) HEALTHY — definitively isolates PF8.3 substrate as the bug
date: 2026-05-10
type: research
status: closed (3-arm A/B complete)
related_tasks: [#44 (closed KILL — strengthened by control), #47 (H1' redesign scope confirmed broader), #43 (Arm A finding now triangulated)]
related_skills: [#34 + #34b (server log first), #36 (grep + behavioral A/B both required)]
---

# PF8.5 Arm C — W4A16 control HEALTHY, isolates PF8.3 substrate as bug

> **Purpose**: complete the n=3 single-variable A/B chain by running
> a control arm with W4A16 (no PF8 env vars), identical config
> otherwise. If W4A16 control works while both PF8.3 arms fail, then
> the bug is definitively isolated to PF8.3 substrate (not infer
> binary, not Pass 3 warmup, not workload, not hardware).

## §1 Hypothesis being tested

After Arm A (PF8.3 + warmup ON) and Arm B (PF8.3 + warmup OFF) both
produced KILL with kernel failures, the framing in `7ed8160` was:
"Per-call workspace allocation in `gemm_w4_fp8_marlin_cuda` is
systematically broken on sm_89 16GB."

But this still leaves a confounder: what if the infer binary itself
is broken at conc=1 on this hardware regardless of which kernel?
A W4A16 control run on the same binary, same hardware, same workload,
same warmup behavior would rule that out.

## §2 Bench config (matches Arms A/B exactly except model)

```bash
RUST_MIN_STACK=33554432 \                            # same
  setsid target/release/infer \                      # same binary
    --model-path infer/models/Qwen3-4B-GPTQ-W4A16-marlin-zpfix \
                                                     # ← only diff: W4A16 not PF8
    --port 8000

guidellm benchmark run --rate 1 --max-seconds 60 ... # same conc=1 60s
```

Notably **NO** `INFER_HYBRID_W4A8_PREFILL=1` and **NO**
`INFER_MARLIN_W4_FP8_PREFILL=1` env vars. Pass 3 warmup ON (default).

## §3 Result (11:48 KST)

```text
=== POST-BENCH HEALTH CHECK (Arm C W4A16 control) ===
Total kernel failures in server log: 0

Pass 3 warmup status:
2026-05-10T11:47:29 WARN warmup.rs:300
  Pass 3 prefill warmup for B=8 at 2003 tokens/row failed
  (completion failed: alloc marlin y_fp16:
  DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory"));
  retrying at 1001 tokens/row
2026-05-10T11:47:30 INFO warmup.rs:308
  Pass 3 prefill warmup done in 8051ms (8 batch sizes, max 8)

=== Bench-tool TTFT/ITL/TPOT (REAL numbers, not broken signal) ===
TTFT mdn:  66.0 ms  p95: 67.1 ms
ITL mdn:    5.8 ms  p95:  5.8 ms
TPOT mdn:   6.3 ms  p95:  6.3 ms
Throughput: 1.3 req/s, 799 tok/s
```

## §4 3-arm A/B verdict

| Arm | Model | PF8 env | Warmup | Kernel failures | TTFT mdn | tok/s | Verdict |
|---|---|---|---|---:|---:|---:|---|
| A | PF8 hybrid-zpfix | ON | ON | **5878** | **0.0 ms** (broken) | 99 (failed) | KILL |
| B | PF8 hybrid-zpfix | ON | OFF | **5959** | **0.0 ms** (broken) | 99 (failed) | KILL |
| **C** | **W4A16-marlin-zpfix** | **OFF** | **ON** | **0** | **66.0 ms** (real) | **799** | **HEALTHY** |

### §4.1 What this triangulates

| Comparison | Conclusion |
|---|---|
| A vs B (warmup ON vs OFF, both PF8) | Not warmup-dependent — same kill profile |
| **A vs C (PF8 vs W4A16, both warmup ON)** | **NOT infer-binary-broken — same binary serves W4A16 fine** |
| **B vs C (PF8 no-warmup vs W4A16 warmup)** | **NOT hardware-broken — same GPU serves W4A16 fine** |
| A vs B vs C | The ONLY differing variable producing KILL is PF8.3 substrate |

### §4.2 Confounding variables ruled out by Arm C

- ✅ infer binary correctness (W4A16 produces real numbers)
- ✅ Pass 3 warmup correctness (W4A16 warmup OOMs at B=8 then gracefully clamps once, completes in 8051ms)
- ✅ guidellm tool correctness (produces real TTFT/ITL when kernel works)
- ✅ Hardware capability (same RTX 4070 Ti SUPER, same 16GB)
- ✅ Workload pattern (same conc=1 60s 512/128 tokens)
- ✅ Test harness (same `pf85_bench_v11_user.sh`-derived config)

### §4.3 What's NOT ruled out (still hypothesis-grade)

- Could be specifically the cublasLt FP8 path (`cublasLtMatmul` invocation in marlin_w4_fp8_kernel.cu)
- Could be specifically the cudarc allocator under FP8 weight layout
- Could be a kernel-launch-config bug in marlin_w4_fp8 (smem/reg/stages misconfigured)
- Could be a per-call cudaMalloc pattern unique to PF8 path

These are all WITHIN the PF8.3 substrate boundary — not WHICH part of PF8.3 is broken. n=3 A/B nails the boundary; further bisection within PF8.3 would need codex source-read + microbench scope.

## §5 Comparison: W4A16 vs INT8 baseline

Arm C TTFT mdn 66.0 ms vs INT8 baseline (per `pf85_bench_v11_user.sh`
license matrix) 53.6 ms = **+23% slower**. Not an A/B comparison
(different bench windows, different days), but informative:
- W4A16 is the next-best-perf path after INT8 baseline
- PF8 was supposed to recover the gap (-8 to -16% TTFT vs INT8)
- With PF8 KILLed, the gap to INT8 stays at +23% for W4A16 path

This sets the bar for what Medusa (P1 pivot) needs to beat: at least
match W4A16 perf while improving ITL.

## §6 SKILL implications

### §6.1 #36 (grep + behavioral A/B both required) — n=4 corroboration

3-arm A/B with control arm is the IDEAL behavioral A/B per #36 v1.14.0
graduation. Static code audit of `marlin_w4_fp8_kernel.cu` would have
suggested per-call alloc pattern was suspect (per `da7f5a2` Task #43
Arm A grep finding), but ONLY behavioral A/B with W4A16 control proves
it's PF8-specific not infer-binary-broken. n=4 evidence (was n=3 from
prior accumulation).

### §6.2 #34 (greedy single-request not sufficient) — strongly reinforced

Arm C produces real TTFT/ITL because W4A16 actually computes; Arms A/B
produce 0.0 ms ttft because PF8 kernel fails on every request. The
contrast IS the validation that broken-signal (TTFT=0) is detectable
when paired with control arm having real signal.

### §6.3 #34b (server log first) — yet another data point

Arm C log shows ONE Pass 3 warmup OOM at B=8 (graceful clamp, normal),
then 0 errors during the 60s bench. Arm A/B logs show 5878/5959
failures in the same window. The bench-tool stats wouldn't tell you
which arm worked without server log inspection.

## §7 Implications for downstream work

1. **Task #44 PF8 chain CLOSED-KILL stands ironclad** — 3-arm A/B
   eliminates infer-binary / hardware / warmup confounders
2. **Task #47 H1' redesign scope CONFIRMED broader** — must address
   per-call allocation in PF8 path specifically; W4A16 path doesn't
   need similar treatment
3. **Pickup queue P1 pivot to #28 Medusa unchanged** — perf bar set
   by W4A16 (66 ms TTFT, 799 tok/s) is the floor Medusa must improve
4. **Future PF8 work**: any PF8.3 redesign should run W4A16 control
   arm BEFORE landing as license gate — confirms the kernel surface,
   not the substrate plumbing

## §8 Procedural learning

The 3-arm A/B took **~5 min wall-clock total** (Arm C only — Arms A/B
were prior). Should have included Arm C in the original Arm A bench
plan instead of inferring "kill is PF8-specific" from log evidence
alone. Adding to procedural rule:

**Skill `kernel-optimization` Phase 5 sub-rule (proposed)**: when
single-arm KILL is observed, IMMEDIATELY run a control-arm bench
(same hardware/workload/binary, no substrate-under-test) before
committing root-cause framing. The control arm cost is small (~5
min); it pre-empts the framing-decay pattern caught at SKILL #29
n=4/5/6 in this session-tail.

## §9 Cross-references

- `0be278f` original PF8.5 KILL (Arm A) — framing now fully validated
- `7ed8160` Arm B REFUTE warmup-DEPENDENT framing
- `da7f5a2` + `d09623a` Task #43 Arm A static-scratch broken framing — strengthened
- `bench-output/2026-05-10-pf85-armC-w4a16-control/benchmarks.{json,csv}`
- `/tmp/pf85-armC-w4a16-control.log`
- SKILL `kernel-optimization` v1.14.0 #36 (n=4 evidence)
- SKILL `kernel-optimization` v1.12.0 #34 + #34b

## §10 Arm D follow-up (added 11:53 KST) — W4A8-zpfix control

Per skill kernel-optimization Phase 5 single-var A/B sub-rule (no
single-control assumption — test BOTH non-PF8 W4 paths to fully
isolate), Arm D extends the matrix with W4A8-zpfix (Task #48
qzeros-fixed default).

### §10.1 Arm D config

```bash
RUST_MIN_STACK=33554432 \
  setsid target/release/infer \
    --model-path infer/models/Qwen3-4B-GPTQ-W4A8-zpfix \   # ← W4A8 not W4A16
    --port 8000
# guidellm: same conc=1 60s 512/128 tokens, default warmup ON
```

### §10.2 Arm D verdict

```text
Total kernel failures: 0
Pass 3 warmup: completed 4532ms (B=7+B=8 OOM at 2048→1024 graceful clamp, normal)
TTFT mdn: 54.2 ms (real)  / p95: 55.0 ms
ITL  mdn: 11.9 ms (real)  / p95: 11.9 ms
TPOT mdn: 12.2 ms
Throughput: 0.6 req/s, 409 tok/s
```

### §10.3 Final 4-arm A/B matrix

| Arm | Model | PF8 env | Warmup | Failures | TTFT | ITL | tok/s | Verdict |
|---|---|---|---|---:|---:|---:|---:|---|
| A | PF8 hybrid-zpfix | ON | ON | 5878 | 0.0 (broken) | 0.0 | 99 (failed) | KILL |
| B | PF8 hybrid-zpfix | ON | OFF | 5959 | 0.0 (broken) | 0.0 | 99 (failed) | KILL |
| **C** | **W4A16-marlin-zpfix** | OFF | ON | **0** | **66.0 ms** | **5.8 ms** | **799** | **HEALTHY** |
| **D** | **W4A8-zpfix (Task #48)** | OFF | ON | **0** | **54.2 ms** | **11.9 ms** | **409** | **HEALTHY** |

### §10.4 NEW perf comparison (Claude-first measurement at conc=1 sustained)

This is a substantive new measurement chain — first time these paths
have been benched against each other at conc=1 sustained:

| Path | TTFT mdn | ITL mdn | tok/s | Notes |
|---|---:|---:|---:|---|
| INT8 v3 baseline (`pf85_bench_v11_user.sh` matrix) | 53.6 ms | 6.8 ms | 697 | from 2026-05-08 |
| **W4A8-zpfix** | **54.2 ms** | **11.9 ms** | **409** | +1% TTFT, +75% ITL, -41% tok/s vs INT8 |
| **W4A16-marlin-zpfix** | **66.0 ms** | **5.8 ms** | **799** | +23% TTFT, -15% ITL, +15% tok/s vs INT8 |
| PF8.3 hybrid (Arms A/B) | KILL | KILL | KILL | substrate broken |

Architectural insight (n=1 hypothesis, needs source-read confirmation):
- **W4A8** wins prefill (FP8 activation reduces prefill compute by ~10%)
  but loses decode (per-token quant overhead amortized over fewer
  ops at decode batch=1)
- **W4A16** loses prefill (more weight bandwidth at large M=512) but
  wins decode (no per-token quant overhead, just W4 unpacking)
- Neither beats INT8 baseline at TTFT × ITL × tok/s simultaneously
- PF8 was supposed to get -8 to -16% TTFT vs INT8 (license matrix);
  with PF8 KILLed, the gap is unbridged

### §10.5 Implications for Medusa pivot (Task #28)

Medusa Phase 1.A perf floor to beat:
- **TTFT**: 54.2 ms (W4A8) — must hold or improve
- **ITL**: 5.8 ms (W4A16) — must hold or improve
- **Throughput**: 799 tok/s (W4A16) — must hold or improve

Medusa primarily attacks **decode tok/s** via parallel head verification
— so the relevant axis is improving on W4A16's 799 tok/s (or W4A8's
409 tok/s on the W4A8 path). Acceptance threshold for Medusa: ≥ 2×
on tok/s at acceptance ≥ 70%, otherwise diminishing return on
training cost (~1 week).

### §10.6 Skill implications additional

**SKILL #34 reinforced n+1**: 4-arm A/B with TWO healthy controls
provides much stronger evidence than single-control. Arms C+D rule
out "infer binary specifically broken when serving W4 quants" — both
W4A16 and W4A8 paths work fine.

**Skill `kernel-optimization` Phase 5 sub-rule (proposed, n=1
evidence)**: when KILL observed in one substrate variant, run TWO
controls (one nearest-relative + one architecturally-different) to
fully bound the broken surface. Arm C (W4A16, different bit-width
from PF8) + Arm D (W4A8, same bit-width, different impl) together
isolate "PF8.3 specifically, not W4 quantization in general".
