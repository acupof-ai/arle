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
