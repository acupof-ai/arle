---
title: 2026-05-10 P2.5 Step 1 LICENSE — QQQ Hunk 8 IS the M'' (Marlin schedule auto-tune); concrete codex Step 2 brief
date: 2026-05-10
type: research
status: open (P2.5 LICENSED for codex Step 2 pickup; ~50-100 LOC port)
related_docs: [`b2cccb9` P2.5 diagnostic, `f8b8174` QQQ upstream finding, `d8ebe73` Machete-inspired reframing §3.2 M'']
---

# P2.5 Step 1 — LICENSE: QQQ Hunk 8 IS schedule auto-tune; codex Step 2 ready

> **2026-05-10 later update**: any priority-table reference to
> A+B/Medusa in this document is historical. Current Qwen3.5 Medusa is
> blocked on recurrent-state rollback; P2.5 remains separate and
> unaffected.

> **Why now**: `b2cccb9` §4.1 specified P2.5 Step 1 = "diff + categorize
> 119-line delta". Executed this tick. Result: hunk 8 (+99/-42) IS the
> schedule auto-tune pattern that M'' (`d8ebe73` §3.2) was supposed
> to deliver. **QQQ has already done what ARLE wanted to do for M''.**

## §1 Diff stats

```
$ diff -u marlin_w4a8_kernel.cu /tmp/qqq_gemm.cu | wc -l
446

$ grep -c '^+' /tmp/qqq-arle-w4a8.diff   # 225 added lines
$ grep -c '^-' /tmp/qqq-arle-w4a8.diff   # 106 removed lines
                                          # net +119 in QQQ direction
```

12 hunks total. Sizes (ranked):
1. Hunk 8 (L827-868): **+99/-42 = +57 net** — biggest, MOST CHANGED
2. Hunk 9 (L885+92): +28 net — kernel dispatch logic
3. Hunk 12 (L957+62): +31 net — Torch FFI entry point
4. Hunk 7 (L774+10): cosmetic / minor
5-12: smaller hunks, mostly comments / minor refactor

## §2 Hunk 8 — the M'' pattern, by another name

### §2.1 What ARLE has (lines 820-868)

Single-purpose `CALL_IF` macro that hardcodes one tile config per
template instantiation. Dispatch is "if-elseif" cascade matching
`(thread_m_blocks, thread_n_blocks, thread_k_blocks, group_blocks)`.

```c
#define CALL_IF(THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, GROUP_BLOCKS) \
  else if (thread_m_blocks == THREAD_M_BLOCKS && ...) { \
    Marlin<THREADS, THREAD_M_BLOCKS, ...><<<blocks, THREADS, ...>>>(...); \
  }
```

### §2.2 What QQQ has (lines 820-961)

Replaced with **explicit thread-config table + auto-selection**:

```c
static constexpr int min_thread_n = 64;
static constexpr int min_thread_k = 64;
static constexpr int tile_size = 16;
static constexpr int max_par = 16;
static constexpr int pack_factor_4bit = 8;  // 8 4-bit vals inside a 32 bit

typedef struct {
  int thread_k;
  int thread_n;
  int num_threads;
} thread_config_t;

thread_config_t small_batch_thread_configs[] = {
    // Ordered by priority
    // thread_k, thread_n, num_threads
    ...  // (table of configs auto-selected per workload shape)
};
```

### §2.3 This IS the M'' pattern

Per `d8ebe73` §3.2:
> "Option M'' (cutlass-style schedule auto-tune ON existing Marlin):
> Apply Machete's ScheduleConfig pattern to ARLE's existing Marlin —
> auto-pick BLOCK_M/N/STAGES per problem shape."

QQQ Hunk 8 IS exactly this. The `small_batch_thread_configs[]` array
+ runtime selection IS the "ScheduleConfig pattern" applied to Marlin
(without WGMMA/TMA, sm_89-compatible).

## §3 Updated assessment for P2.5 + M''

P2.5 (QQQ diff-port) and M'' (Marlin schedule auto-tune) MERGE into
ONE pickup:

| Property | Value |
|---|---|
| **Source** | QQQ Hunk 8 (lines 820-961 of qqq_gemm.cu) |
| **Target** | ARLE marlin_w4a8_kernel.cu lines 820-868 (replace CALL_IF with thread_config_t table) |
| **LOC port** | ~100-140 (the new struct + table + dispatcher logic) |
| **Risk** | LOW (data-driven config table replacing hand-tuned macro) |
| **Expected gain** | 2-8% on shape-mismatched paths (per M'' estimate) |
| **A/B benches needed** | conc=1+conc=4 W4A8 sustained 60s + greedy_consistency |

This is a SOLID LICENSE outcome — well within the
`kernel-optimization` skill Phase 8 threshold (small port + medium
risk + measurable gain).

## §4 Concrete Step 2 codex pickup brief

```
Task: P2.5/M'' merged pickup — port QQQ thread_config_t pattern to ARLE marlin_w4a8

Step 2.A (codex, 1-2 hr):
  - Read /tmp/qqq_gemm.cu lines 820-961 (the new dispatch logic)
  - Read crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu lines 820-868
    (the existing CALL_IF cascade)
  - PRESERVE ARLE-specific:
    * cp_async4_stream + cp_async1_stream (sm_89 streaming variants)
    * extern "C" gemm_w4a8_marlin_cuda FFI (NOT void qqq_gemm Torch entry)
    * cudaStream_t parameter signature
  - REPLACE: ARLE's CALL_IF cascade with QQQ's thread_config_t struct +
    small_batch_thread_configs[] table + auto-selection logic

Step 2.B (codex, 1-2 hr):
  - Update Rust dispatcher in infer/src/ops/linear.rs (~30 LOC) to pass
    thread_k/n/num_threads as -1 (sentinel for auto-select) by default
  - Add env-var override for explicit tile config (per kernel-optimization
    skill Phase 7 tradeoff: don't lose A/B-tuning ability)

Step 2.C (Claude, 1 hr):
  - cargo build --release --features cuda
  - cargo test --release --features cuda --test e2e_w4a8 (greedy_consistency)
  - bash scripts/bench_guidellm.sh w4a8-qqq-port-after \
      --concurrencies 1,4 --max-seconds 60 \
      --data 'prompt_tokens=512,output_tokens=128'

Step 2.D (Claude, 30 min): A/B vs current ARLE marlin_w4a8 bench:
  - License: TTFT/ITL Δ within ±5% (neutral or better) AND greedy 0.0% diff
  - Soft win: TTFT/ITL Δ ≥ -2% with σ < 5% across n=3
  - Kill: any regression > +5% OR greedy diff > 0.5%
```

## §5 Refined priority table (final)

| Priority | Path | Wall-clock | Status | Expected |
|---|---|---:|---|---|
| P1 | A+B combined (Medusa + Hybrid) | 4-5 days | gated on user GO | 2.61× tok/s + -14% latency |
| P2.5/M'' | **QQQ Hunk 8 port (schedule auto-tune)** | **3-5 hr (Step 2.A-D)** | **LICENSED Step 1; Step 2 ready** | **2-8% TTFT on shape-mismatched paths** |
| P3 | Task #47 H1' v2 | 1 day | gated on diagnostic logging | unblocks PF8 path |
| ~~P2~~ | ~~vLLM Marlin diff-port~~ | DONE (`b6b8adc`) | maintenance only | |
| ~~P3.5~~ | ~~M''' (W4-FP8 preprocess)~~ | DONE (`86b28c7`) | already integrated | |
| ~~P4 M''~~ | ~~Marlin schedule auto-tune~~ | **MERGED INTO P2.5** | (was 3-5 days, now 3-5 hr) | |
| P5 | Option M' (full cutlass rewrite) | 2-3 weeks | open, HIGH risk | 5-15% best-case |
| KILLED | Literal Machete port | impossible | KILLED `fc33cfb` | 0% on sm_89 |

### §5.1 Compound win wall-clock (revised)

| Item | Days |
|---|---:|
| P1 A+B combined | 4-5 |
| P2.5/M'' merged QQQ port | 0.5 (3-5 hr) |
| P3 Task #47 H1' v2 (parallel codex track) | 1 |
| **Total compound win wall-clock** | **5-6 days** |

Net compound expected: **2.61× tok/s + -14% latency + 2-8% TTFT
shape-tune + PF8 path unblock**.

P2.5/M'' is the cheapest win on the queue — 3-5 hr for 2-8% gain.
**Recommend codex pickup IMMEDIATELY** (parallel to A+B if possible);
no user GO needed, no architectural decision pending.

## §6 SKILL #43 application

Per canonical anti-pattern #43 v1.16.0:
- "Verified absent at" pointers throughout this entry:
  - QQQ Hunk 8 source: `/tmp/qqq_gemm.cu` lines 820-961
  - ARLE target: `marlin_w4a8_kernel.cu` lines 820-868
  - LOC measured: 119 net delta, 12 hunks total
- Future codex pickup brief follows same template

## §7 Cross-references

- `b2cccb9` P2.5 diagnostic (this entry executes Step 1)
- `f8b8174` QQQ upstream finding
- `d8ebe73` Machete-inspired reframing §3.2 M'' (now merged into P2.5)
- `b6b8adc` ARLE marlin_pf8 = vLLM fork (P2 DONE)
- `6577ba6` SKILL v1.16.0 #43 graduation
- `kernel-optimization` skill Phase 8 license-or-kill thresholds
- ARLE `crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu` lines 820-868 (target)
- QQQ `csrc/qqq_gemm.cu` lines 820-961 (source)
- `/tmp/qqq-arle-w4a8.diff` (full 446-line diff)
