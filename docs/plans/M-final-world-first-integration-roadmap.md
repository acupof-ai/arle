# M-final — Integration roadmap to "world-first long-sequence inference engine"

> Manager-level synthesis. Captures all in-flight + pending
> milestones from M3.6 onward, projects combined throughput
> trajectory, identifies what's needed BEYOND known plans to hit
> "world-first" status.

## Priority & ROI

> **2026-05-07 EOD update**: M_world1 strategic redirect — target
> = world #1 (lead #2 by 30%+), not just beat vLLM. See
> [`M_world1-30-percent-lead-roadmap.md`](M_world1-30-percent-lead-roadmap.md).
> Phase 0 (bench SGLang + TRT-LLM locally) is now the gating
> step; per-shape priorities below are derived from vLLM-only
> data and will be re-anchored after Phase 0 baseline lands.

| Track | Status | Priority | ROI basis | Kill criteria |
|---|---|---|---|---|
| F4-Small (decode async sync) | **✓ shipped** (`2a534c4`) | done | High-conc +82.5% out tok/s vs F4-Small-pre baseline. ARLE 843 vs vLLM 647 = +30.3%. | n/a (shipped) |
| **M3.9 Phase 1A v3** (multi-slot async readback) | **✓ shipped, default Split** (codex `5cacdcb`) | done | Substrate kept under `--scheduler-mixed-policy mixed`. Default Split+ring incidentally gave **+25.6%** vs F4-Small longctx baseline (153.9 vs 122.5). Mixed opt-in still has secondary bugs (workspace + per-prefill prep loop) — flagged experimental. | Mixed opt-in re-bench post-future fix; if no improvement → defer indefinitely. |
| **M_pf** (prefetch wiring) | plan; P0 done (`699002f` peek_prefix_classify) | **P3** *(deprioritized)* | Honest ROI re-estimate: 5–15% TTFT improvement at multi-tenant shared-prefix only. Substrate exists (`submit_prefetch_plan`). | < 5% TTFT improvement at multi-tenant bench → revert wiring. |
| **M_b.2 Phase 1** (TileLang FP8 prefill TileLang) | A0 smoke landed (`c865f4b`); full integration pending | **P1 (conditional, post-Phase 0)** | Long-ctx 4k TTFT gap remaining (ARLE 1976 vs vLLM 1177 ms = vLLM 1.68× faster). Likely scheduler-side first (codex investigating chunk policy); kernel-axis only if scheduler fix insufficient. | After scheduler-side longctx fix: if ARLE TTFT < 1.5× #2 → demote. |
| Spec-decode + F4-Small compound | future | **P2** | Speculation theoretical 1.5-2× decode throughput. Substrate at `scheduler/cuda/spec_path.rs`. Compound with F4-Small async pattern. | Spec acceptance < 0.6 in A0 smoke → revert. |
| **M_world1 Phase 0** (bench SGLang + TRT-LLM locally) | not started | **P0** | Without #2 baseline measurement, all "30% lead" priorities are unanchored guesses. Mandatory for Strategy. | If SGLang + TRT-LLM > ARLE by > 50% at 2+ shapes on RTX 4080S → reframe to "competitive niche". |
| **M_nsys P1** (defer first graph capture until cuProfilerStart) | P0 substrate done (`9b1fb8c`); P1 pending | **P0 diagnostic infra** | Required for proper long-ctx trace; nsys 2025.6 silently drops kernel data when graph captured before profile window. Without it, optimization at long-ctx is blind. | If NVIDIA fixes nsys upstream → redundant. |
| INT4 KV compression | future | P3 | 30% memory bandwidth reduction. Long-ctx attention bandwidth-bound. ~400 LOC. | MMLU drop > 0.5% → abandon. |
| ~~M_ibp~~ (in-batch prefix dedup) | **ABANDONED** (`9432289`) | n/a | Phase 0 license-or-kill: ARLE already 1.80× past vLLM at multi-tenant shared-prefix. Cascade pattern adequate. | (Already killed.) |

**Tier ordering**: P0 (Phase 0 baseline + M_nsys P1) → P1 (per-shape gap fixes) → P2 → P3, gated by per-tier ROI evidence.

**2026-05-07 EOD note**: original "P0 = M3.9 Phase 1A v3" framing
proved wrong — Phase 1A v3 implementation REGRESSED long-ctx
(TTFT +132%) before fix landed. The fix (default Split, multi-slot
ring as substrate) ended up being NET POSITIVE +25.6% via incidental
gain. Lesson recorded in
[`feedback_docs_priority_roi_evidence.md`](../../../.claude/projects/-home-ckl-projects-arle/memory/feedback_docs_priority_roi_evidence.md).

## Current state (2026-05-07 EOD)

### Confirmed bench numbers (2026-05-07 EOD, post-Phase-1A-v3-fix)

| Workload | ARLE | vLLM | Δ |
|---|---:|---:|---|
| **High-conc 1k/256/c=64 out tok/s** | **843** | 647 | **ARLE +30.3% ✓** |
| High-conc per-row ITL | **0.99 ms** | 1.43 ms | **ARLE 1.45× faster ✓** |
| **Long-ctx 4k/c=4 out tok/s** (post-fix default Split) | **153.9** | 159.1 | vLLM +3.4% |
| Long-ctx 4k/c=4 TTFT mdn | 1976 ms | **1177 ms** | **vLLM 1.68× faster (remaining gap)** |
| Long-ctx 4k/c=4 ITL | 19.3 ms | 18.8 ms | ~equal |
| Long-ctx 8k/c=4 out tok/s (pre-fix) | 92.2 | 105.6 | vLLM +14.5% (need re-bench post-fix) |
| Long-ctx 8k/c=4 ITL | 23.9 ms | 26.7 ms | ARLE 1.12× faster |
| **Multi-tenant shared-prefix TTFT mdn** | **318 ms** | 573 ms | **ARLE 1.80× faster ✓** |
| c=1 latency 512/128 ITL | 14.0 ms | unmeasured | (single-row baseline) |

**For "world #1 (lead #2 by 30%)"**: vLLM may not be #2.
SGLang + TRT-LLM bench pending (M_world1 Phase 0). Above table
is vs vLLM only; the canonical comparison shifts post-Phase-0.

### Architecture state

**Decode axis (high-conc dominant)**:
- ✓ F4-Small (`2a534c4`): decode async readback — eliminated 65 ms
  per-tick `cuStreamSynchronize`, +82.5% high-conc throughput.
- ✓ M_b.1 Phase B (`45e1d0c`): TileLang HD128 BF16 decode kernel
  for `max_qlen==1` paths. No measurable high-conc delta (kernel
  is fast enough; not the bottleneck).

**Prefill axis (long-ctx dominant)**:
- ✓ B.1.2 (`14a48e9`): prefill async chunk completion. Failed to
  move TTFT (-28 ms vs -800 ms target). Root cause analysis
  surfaced the actual TTFT bottleneck.
- ✓ **M3.9 Phase 0** (`786a20a`): instrumentation to measure
  Mixed/Split routing + Ok(false) reasons.
- ✓ **M3.9 Phase 1A v3** (codex `5cacdcb`, shipped 2026-05-07):
  multi-slot async readback substrate kept; default policy = Split
  (Mixed flagged experimental due to workspace + per-prefill prep
  loop residual bugs). Default Split + ring substrate gave +25.6%
  vs F4-Small longctx baseline (153.9 vs 122.5 out tok/s).
- ⏳ **Long-ctx 4k/c=4 prefill TTFT 800 ms gap**: codex license-
  or-kill bench (`c219434`) ruled out H_LP1 (full-row packing →
  TTFT -21.5% but out tok/s -12.1%, net negative tradeoff) and
  H_LP2 (chunk-size 4096 → TTFT +0.4%, kill criteria fired).
  Remaining: **H_LP3 (per-chunk launch overhead)** needs nsys
  trace — gated on M_nsys P1 (graph capture timing fix).

**Kernel-axis future**:
- ⏳ M_b.2 A0 (`c865f4b`): TileLang HD128 FP8 decode kernel A0
  smoke. Phase 1 (full integration) pending.
- ⏳ M_e.1: Metal Qwen3.5 KVPool unification (cross-backend).

**Correctness substrates landed**:
- ✓ M_d.1 (5 steps): RadixCache namespace + tokenizer fingerprint
  → silent corruption defense for hot-swap scenarios.
- ✓ NVTX scaffolding (`998bfee`): trace observability.
- ✓ vllm_serve_control.sh (`998bfee`): apples-to-apples vLLM
  bench wrapper.

## Projected throughput trajectory

Assuming each pending milestone hits its expected gain:

| Cumulative state | High-conc out tok/s | Long-ctx 4k/c=4 TTFT | Long-ctx 4k/c=4 out tok/s |
|---|---:|---:|---:|
| Baseline (Phase 1 trace) | 462 | (n/a) | 122.5 |
| ✓ F4-Small (landed) | 843 | (n/a) | 122.5 |
| ✓ M_b.1 Phase B (landed) | 843 | (n/a) | 122.5 |
| ✓ M3.9 Phase 1A v3 default Split (`5cacdcb`) | 843 | 1976 ms | **153.9** (+25.6%) |
| vLLM s8 same shape (control) | 647 | 1177 ms | 159.1 |
| **Remaining gap to vLLM** | ARLE +30.3% ✓ | vLLM 1.68× ✗ | -3.4% ✗ |

**Real target: lead #2 by 30%+** (per `M_world1-30-percent-lead-roadmap.md`).
vLLM may not be #2 — Phase 0 baseline (SGLang + TRT-LLM bench) is
mandatory before further optimization investment. See
[`M_world1-30-percent-lead-roadmap.md`](M_world1-30-percent-lead-roadmap.md).

## What's missing for "world-first" beyond known plans

### Opportunity A — Speculative decode + F4-Small compound

ARLE has spec-decode infrastructure
(`scheduler/cuda/spec_path.rs` + `forward_spec_verify_batch`).
Current path doesn't benefit from F4-Small's async readback —
spec verify still has its own per-step sync chain.

**Proposed**: M3.9 sequel applies Phase 1A v3's multi-slot
pattern to spec verify. Combined with high acceptance rate
(>0.6), spec-decode can 1.5-2× effective decode throughput.

LOC est: ~150. Risk: spec-decode invariants are subtle
(B=1 vs B=N consistency); needs strong correctness gating.

### Opportunity B — Long-context KV compression (INT4)

ARLE supports FP8 KV. SGLang and recent literature show INT4 KV
gives another ~30% memory bandwidth reduction with minimal
accuracy loss (≤ 0.5% PPL on Qwen-class).

For long-context where KV bandwidth dominates, INT4 → 2× TTFT
improvement potential via:
- 2× more requests fit in same KV pool (raise concurrency)
- ½ memory bandwidth per attention read

LOC est: ~400 (new kernel + dispatch + pool layout). Risk:
correctness across ALL workloads, FP4 silently degrades MMLU.

### Opportunity C — Full continuous batching (vLLM v1 parity)

Currently ARLE has `step_mixed_launch` → unified mixed batch BUT
the scheduling decision (when to mix vs pure-prefill vs split)
is greedy/per-step. vLLM v1's continuous batching makes the
SAME decision but **smarter**: pre-allocates token budgets across
upcoming steps to maximize per-tick utilization.

Concretely: ARLE step takes whatever decode rows + prefill
candidates fit. vLLM step uses a token-budget oracle that
considers expected ITL across requests to choose admission rates.

LOC est: ~250 (scheduler refactor). Risk: correctness invariants
in admission ordering.

### Opportunity D — KV prefetch from RadixCache

ARLE has RadixCache for prefix sharing. But the prefix HIT KV
loads from CPU/host pool to GPU pool ON-DEMAND when admission
runs. For long-ctx prompts with shared system prompts, this
forces a stall.

**Pre-stage**: when scheduling sees prefix MATCH on admission,
fire async H2D copy IMMEDIATELY (before the request reaches
prefill). By the time prefill runs, KV is already on GPU.

LOC est: ~80 (new admission hook). Risk: extra GPU memory
pressure when speculative prefetch evicts useful blocks.

## Process retrospective (from F4-Small to here)

### What worked

1. **Trace before fix** (F4-Small): Phase 1 nsys trace identified
   65 ms `cuStreamSynchronize` as 48.6% of CPU API time → fix
   delivered +82.5% throughput.
2. **Multi-shape validation**: long-ctx vLLM control surfaced the
   prefill TTFT gap that high-conc trace alone hid.
3. **Server-log decomposition**: `step breakdown` lines
   substituted for nsys at low-concurrency workloads where nsys
   capture is unreliable.
4. **Source archaeology + git blame**: located F4-Small's
   `deferred_decode_emit.is_none()` precondition as the structural
   cause of 10× split tax.
5. **Hypothesis verification before fix**: corrected analysis
   (`28056b9`) showed naive 1-line removal would cause silent
   token loss → forced multi-slot design.
6. **Parallel work pattern**: codex implements + multi-round
   review while Claude reads source / does math / does cross-system
   bench. Both make progress without conflict.

### What failed (and why)

1. **B.1.2 traceless fix attempt**: assumed prefill sync was
   bottleneck without trace evidence → -28 ms TTFT vs -800 ms
   target. Cost: 1 day of codex work + 1 review cycle.
2. **M3.8 v1 plan-without-survey**: 156-line plan to "implement
   cross-request prefill batching" → 30-min experiment showed
   it already worked. Plan wasted.
3. **M3.9 26b7f86 1-line-fix hypothesis**: blame analysis pointed
   to F4-Small precondition; jumped to "remove the line" →
   another 30 min of source read showed protected invariant
   (silent token loss).

### Rule synthesis

| Rule | Source incident |
|---|---|
| Trace before fix | F4-Small ✓, B.1.2 ✗ |
| Survey source before plan | M3.8 v1 ✗ |
| Blame analysis: understand the invariant, not just remove the guard | M3.9 26b7f86 ✗ → 28056b9 ✓ |
| Multi-shape validation before declaring victory | F4-Small +82.5% high-conc was true; long-ctx -14% only surfaced via vLLM control |
| Parallel work = independent layers (codex impl, claude survey/bench) | All recent ticks |

## Path to "world-first" — sequencing

**Tier 1 (1-2 days,bound by codex review cycles)**:
- M3.9 Phase 0 commit (codex,~30 min more)
- Validation bench (Claude,5 min)
- M3.9 Phase 1A v3 (codex,~1-2 hr per F4-Small precedent)
- Validation bench + nsys (Claude,15 min)

**Tier 2 (2-3 days)**:
- M_b.2 Phase 1 full integration (codex)
- Cross-system bench at every shape (Claude)
- Decision: which Opportunity (A/B/C/D) is highest ROI

**Tier 3 (1 week+)**:
- Selected opportunity from A/B/C/D
- Full bench gauntlet vs vLLM/SGLang at all shapes
- Wins entry: "world-first parity confirmed"

## Cross-references

- F4-Small: `2a534c4`, wins `8f83c80` + `c63c31c`
- M3.6 plan: `68965e0` → `53a2061`
- M3.7 overlap architecture: `6300851`
- M3.8 cross-request batching (canceled): `e592634` + `2530ad6` + `67f9bcb`
- Split tax confirmed: `4a3612b`
- F4-Small Mixed-disable root cause: `26b7f86` → `28056b9`
- M3.9 plan: `63af21f`
- M_b.1: `b42da5d` (Phase A) + `45e1d0c` (Phase B) + `2e60844` (no-delta wins)
- M_b.2: `3a896f3` (plan) + `c865f4b` (A0 smoke)
- M_d.1 (tokenizer fingerprint): 5 commits including `5ae6b83` + `0e1bc3d`
- vLLM longctx control: `9afcd57` + `f7146d4`
- B.1.2: `14a48e9` + `c711b85` (after-snapshot)
