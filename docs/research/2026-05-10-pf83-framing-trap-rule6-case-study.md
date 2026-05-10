---
title: PF8.3 framing-trap case study — narrow "100% bench failure" vs wider "kernel works at conc=1" (§0 SOLID rule-6 reinforcement)
date: 2026-05-10
type: research
status: closed (case study, sediments SKILL v1.12.0 #34 + #34b)
---

# PF8.3 framing-trap case study — narrow "100% bench failure" vs wider "kernel works at conc=1"

> **Purpose**: capture the 2026-05-10 PF8.3 chain as a concrete §0 SOLID
> rule-6 framing-trap case study, parallel to the 2026-05-08 NVTX
> framing trap. Both show: aggregate-failure-rate framing ≠ per-request
> functional framing; using only one framing for license-or-kill leads
> to wrong conclusions.

## §1 The trap

### Narrow framing (would-have-been wrong conclusion)

After PF8.3 substrate landed `11763ba` (12 files, +3936/-13 LOC), bench
v3-v10 sequence ran with `INFER_MARLIN_W4_FP8_PREFILL=1`. Bench v10
final results showed:

```
PF8.3 path requests: 101380 / 101380 failed
gemm_w4_fp8_marlin_cuda failed with code 2 (cudaErrorMemoryAllocation)
```

Documented as PF8.3 RUNTIME KILL in `0cde63d`. **Narrow framing
conclusion**: substrate is fundamentally broken, kernel doesn't work,
should revert the entire PF8 chain (~3936 LOC + 4 days work).

### Wider framing (actual ground truth)

Per `81672c3` H8 diagnostic patch (added `cudaGetLastError()` clear at
wrapper entry) + `de314d2` verify script + `57c37b5` H8 verdict:
single-request curl test at conc=1 produced:

```
$ scripts/pf83_h8_verify.sh
=== curl /v1/completions ===
{"id":"cmpl-9c215c25-...","choices":[{"text":" fox器的使用,比如在代码中使用",
 "finish_reason":"length"}],"usage":{"prompt_tokens":4,
 "completion_tokens":10,"total_tokens":14}}
=== curl second request ===
{"id":"cmpl-149153b9-...","choices":[{"text":" beginningations, and the like.
 The first step",...}]...}
=== gemm_w4_fp8_marlin_cuda failure count ===
Failures: 0
```

**Wider framing conclusion**: kernel works correctly at conc=1 (valid
multilingual completions, 0 errors). The KILL is **load-dependent**,
not kernel-broken. Root cause = per-call `CudaSlice::alloc_zeros` × 5
buffers × 252 ops × N concurrent × 60s sustained = ~580 GB cumulative
cudarc churn → CachingAllocator pool fragmentation under sustained
load → small-block requests fail with `cudaErrorMemoryAllocation`
despite > 14 GB free VRAM (per `cd7732a` §7 H1').

## §2 Why narrow framing was tempting (and wrong)

Three reasons engineers reach for narrow framing first:

1. **Bench tool reports aggregate metrics**: guidellm output table
   shows "0 successful requests" and "all-zero latency" — visually
   loud, easy to over-weight.
2. **Greedy_consistency PASS at conc=1 was earlier evidence**: codex
   commit `ace3cbe` already ran greedy_consistency on PF8 path and
   PASSED — but at conc=1 single-thread. The narrow framing trap is
   forgetting that "greedy at conc=1" answers a different question
   than "bench at conc=4 sustained 60s".
3. **Confirmation pressure**: large bench failure rate is unambiguous
   *if* you don't ask "compared to what?" — i.e. against a per-request
   functional baseline.

## §3 Wall-clock / per-request framing as ground truth

Per §0 SOLID rule 6, when framings diverge, wall-clock / per-request
framing wins. Apply to PF8.3:

| Framing | Result | License decision |
|---|---|---|
| Bench failure-rate aggregate (conc=4 60s) | 100% fail | KILL substrate |
| **Per-request functional (conc=1 curl)** | **PASS — valid completions** | **Kernel works, KILL is load-mechanism** |
| Per-request functional + sustained-load (conc=1+2+4) | NEEDED for license | Both gates required |

**Lesson**: per-request functional framing answers "does the kernel
produce correct output for any request?" — this is the **prerequisite
ground truth** before sustained-load bench is meaningful. If
per-request fails → kernel-broken, KILL substrate. If per-request
passes but sustained-load fails → kernel works, KILL is load-mechanism
(allocator/scheduling/concurrency), fixable without reverting kernel.

## §4 Pattern parallel to 2026-05-08 NVTX trap

| | 2026-05-08 M_pf-graph Phase 0v2.B | 2026-05-10 PF8.3 |
|---|---|---|
| Narrow framing | NVTX dispatch = 55.7% of prefill window | bench failure = 100% of conc=4 requests |
| Narrow conclusion | LICENSE Phase 0v2.B (high % win available) | KILL PF8.3 (substrate broken) |
| Wider framing | wall-clock = 191 ms / 60s trace = 0.32% per-request | conc=1 single-request curl = 0 failures |
| Wider conclusion | KILL Phase 0v2.B (< 10% wall-clock < kill threshold) | KEEP substrate (load-dependent, fixable via H1' static-scratch) |
| Cost of narrow framing | wasted Phase 0v2 cycle (already paid) | would have wasted ~3936 LOC + 4 days (avoided) |

Both show: **narrow framing X% ≠ ground-truth X%; license-or-kill must
use wall-clock / per-request framing**.

## §5 Skill v1.12.0 sediments

`0be7220` codified two anti-patterns from this evidence:

- **#34**: greedy_consistency single-request PASS NECESSARY but NOT
  SUFFICIENT for new GEMM kernel substrate. Pair with sustained-load
  bench at conc 1+2+4. Codex implementing Task #35 cap=8 prefill
  warmup THIS session is already internalizing this — the implicit
  "short sustained-load smoke" step appeared in codex's own work plan
  without explicit brief instruction. Compound learning visible.
- **#34b**: bench reports 0 successful → CHECK SERVER LOG FIRST.
  v3-v10 wasted 30+ min on guidellm CLI quirks (PATH, --backend-kwargs,
  --outputs html, absolute --output-dir, pre-mkdir) when the actual
  cause was kernel 100% failure visible in
  `/tmp/<server>.log` line 627: `prefill batch failed:
  gemm_w4_fp8_marlin_cuda failed with code 2`.

## §6 How to apply (for future engineering decisions)

When a bench reports a high failure rate or a wide regression:

1. **Functional gate FIRST**: can a single curl produce a valid
   response on the same code path? If no → kernel/substrate broken,
   KILL is correct. If yes → KILL is load-mechanism, investigate
   allocator / scheduler / concurrency.
2. **Cross-check framings**: aggregate-pct vs per-request, narrow
   window vs wall-clock. Diverging framings ⇒ ground-truth framing
   wins.
3. **License-or-kill on ground-truth framing only**: don't kill
   substrate based on aggregate metrics if per-request functional
   passes; conversely, don't license substrate based on narrow window
   metrics if wall-clock impact is sub-threshold.
4. **Server log first when bench reports 0 success**: tool quirks vs
   substrate failure are visible in different places.

## §7 Cross-references

- `11763ba` PF8.3 substrate landed
- `0cde63d` PF8.3 RUNTIME KILL aggregate framing (101380 failures)
- `81672c3` H8 diagnostic patch (clears sticky cudaGetLastError)
- `de314d2` `scripts/pf83_h8_verify.sh` (per-request functional verify)
- `57c37b5` H8 verdict: per-request PASS, aggregate KILL → load-dependent
- `cd7732a` §7 H1' refined hypothesis (per-call alloc fragmentation)
- `05e2135` `docs/plans/M_pf83_h1prime_static_scratch.md` (the fix design)
- `0be7220` SKILL v1.12.0 anti-patterns #34 + #34b (sediments)
- `847a132` AGENTS.md/CLAUDE.md §0 SOLID rule 6 (the framing-trap rule)
- `feedback_first_principle_solid_or_deeper.md` memory (now references this case)
