---
title: 2026-05-10 ARLE marlin_w4a8_kernel.cu upstream is HandH1998/QQQ (NOT vLLM) — P2.5 candidate diff-port surfacing
date: 2026-05-10
type: research
status: open (identifies a real upstream-tracking opportunity for sm_89 W4A8 path)
related_docs: [`b6b8adc` ARLE marlin_pf8 = vLLM fork, `2f19a3c` Marlin parity survey, `d8ebe73` Machete-inspired reframing]
---

# ARLE marlin_w4a8 upstream lineage — HandH1998/QQQ, NOT vLLM

> **2026-05-10 later update**: priority-table references to A+B/Medusa
> are superseded for Qwen3.5 by the recurrent rollback blocker. This
> upstream-lineage finding remains independent.

> **Why now**: `b6b8adc` closed P2 (vLLM Marlin diff-port) as DONE
> at file level. This tick surveys whether the W4A8 axis has a
> SEPARATE upstream worth tracking. Yes: HandH1998/QQQ is the actual
> upstream of ARLE marlin_w4a8_kernel.cu, and it's been updated as
> recently as 2026-04-23 (~17 days before today's audit).

## §1 Evidence

### §1.1 ARLE marlin_w4a8_kernel.cu attribution (L1-5)

```c
/*
 * Adapted from https://github.com/IST-DASLab/marlin/blob/master/marlin/marlin_cuda_kernel.cu
 * Modified by HandH1998
 * Copyright (C) 2024 HandH1998
 * Copyright (C) Marlin.2024 Elias Frantar (elias.frantar@ist.ac.at)
 */
```

ARLE forked the **HandH1998-modified** Marlin variant, not the
IST-DASLab original. This is a separate lineage from vLLM's
`csrc/quantization/marlin/marlin.cu` (which Neural Magic-modifies the
same IST-DASLab base).

### §1.2 HandH1998/QQQ project status

```
name: QQQ
description: QQQ is an innovative and hardware-optimized W4A8
              quantization solution for LLMs.
updated: 2026-04-23T13:21:33Z
```

Active project, last update ~17 days before today (2026-05-10).
Specifically W4A8-focused (matches ARLE's W4A8 path).

### §1.3 vLLM does NOT have an internal W4A8 marlin variant

gh code search results for `vllm-project/vllm + filename:marlin + W4A8`:
- `csrc/quantization/marlin/marlin.cu` (general Marlin, not W4A8-specific)
- `vllm/model_executor/layers/quantization/utils/marlin_utils.py` (Python utils)

gh code search for `HandH1998` in vllm-project/vllm:
- `benchmarks/kernels/benchmark_w8a8_block_fp8.py` (W8A8 not W4A8)
- `.buildkite/lm-eval-harness/configs/Meta-Llama-3-8B-QQQ.yaml`
  (eval config for QQQ models, NOT a kernel adaptation)

So vLLM consumes QQQ at the model-config level (loads
QQQ-quantized checkpoints) but does NOT vendor the QQQ marlin kernel
internally. The kernel lives at the upstream HandH1998/QQQ repo.

## §2 Implication for ARLE upstream-tracking

### §2.1 Two independent upstream sources

| ARLE file | Upstream | Last upstream activity |
|---|---|---|
| `marlin_kernel.cu` (W4A16) | IST-DASLab + Neural Magic via vLLM marlin | tracked via marlin_pf8/ fork (per `b6b8adc`) |
| `marlin_w4a8_kernel.cu` (W4A8) | HandH1998/QQQ | active 2026-04-23 |
| `marlin_pf8/*.h` (PF8 substrate) | vLLM marlin/ headers | up-to-date per attribution |

### §2.2 New P2.5 candidate: QQQ diff-port

This is the FIRST upstream-diff opportunity surfaced where ARLE may
genuinely be behind:

| Option | Source | Wall-clock | Expected | Risk |
|---|---|---:|---|---|
| P2.5 | HandH1998/QQQ diff-port | ?d (need diff first) | unknown | LOW-MED |

Effort sizing requires:
1. (Claude, 0.5d) Fetch QQQ marlin_cuda_kernel.cu, diff vs ARLE
   marlin_w4a8_kernel.cu
2. License-or-kill on diff size
3. (Codex, 1-2d) Port if delta < 100 LOC of meaningful kernel changes

Unlike P2 (vLLM Marlin already ported), this could deliver real
sm_89-specific W4A8 improvements if QQQ shipped tile/scheduler tunings
in the last ~17 days.

### §2.3 Status of QQQ-vs-ARLE diff (THIS audit)

NOT YET PERFORMED — would need next-tick or codex pickup. Listing
as "candidate-pending" per SKILL #43 discipline, with explicit
"verified absent at" pointer:

- ARLE last touch on marlin_w4a8_kernel.cu: per `git log` ~2026-05-08
  (HandH1998 mods + W4A8 path)
- QQQ upstream last touch: 2026-04-23
- **Time gap**: ~15 days (QQQ updated AFTER ARLE's last sync)
- **Diff status**: not measured this audit — P2.5 diagnostic step
  pending

## §3 Updated priority table (refines `b6b8adc` §3)

| Priority | Path | Wall-clock | Status | Expected |
|---|---|---:|---|---|
| P1 | A+B combined (Medusa + Hybrid) | 4-5 days | gated on user GO | 2.61× tok/s + -14% latency |
| ~~P2~~ | ~~vLLM Marlin diff-port~~ | DONE | b6b8adc | maintenance only |
| **P2.5** | **HandH1998/QQQ diff-port** | **0.5d diag + ?d port** | **NEW — diff pending** | **unknown, LOW-MED risk** |
| P3 | Task #47 H1' v2 | 1 day | gated on diagnostic logging | unblocks PF8 path |
| ~~P3.5~~ | ~~M''' (W4-FP8 preprocess)~~ | DONE | already integrated | |
| P4 | Option M'' (Marlin schedule auto-tune) | 3-5 days | open | 2-8% conditional |
| P5 | Option M' (full cutlass rewrite) | 2-3 weeks | open | 5-15% best-case, HIGH risk |
| KILLED | Literal Machete port (sm_90+) | impossible | KILLED `fc33cfb` | 0% on sm_89 |

### §3.1 P2.5 vs P1 priority

P1 still dominant (4-5d, 2.61× tok/s = bigger lever). But P2.5 is a
new entry that could complement P1 OR be a quicker pickup if user
wants kernel-axis verification before committing to A+B's user-GO gate.

## §4 Cross-references

- `b6b8adc` ARLE marlin_pf8 = vLLM fork (P2 DONE finding)
- `2f19a3c` ARLE Marlin parity survey
- `86b28c7` M''' completion correction
- `fc33cfb` Machete KILL
- `494ad3a` Task #47 H1' v2 redesign brief
- `d8ebe73` Machete-inspired reframing
- ARLE `crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu` (HandH1998/QQQ-derived)
- HandH1998/QQQ: <https://github.com/HandH1998/QQQ> (active 2026-04-23)
- vLLM `benchmarks/kernels/benchmark_w8a8_block_fp8.py` (W8A8, not W4A8)
- vLLM `.buildkite/lm-eval-harness/configs/Meta-Llama-3-8B-QQQ.yaml` (eval config, not kernel)

## §5 SKILL #43 application

This entry is a positive APPLICATION of the canonical anti-pattern
#43 (graduated v1.16.0 per `6577ba6`):
- Item LISTED as "candidate-pending"
- Evidence pointer INLINE: "diff status not measured this audit —
  P2.5 diagnostic step pending"
- Explicit "verified absent at <upstream URL> + ARLE file path"
  comparison

This is the bare-minimum step that #43 requires for any
"pending pickup" entry. Future briefs should follow this template.
