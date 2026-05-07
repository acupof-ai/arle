# M-final — Integration roadmap to "world-first long-sequence inference engine"

> Manager-level synthesis. Captures all in-flight + pending
> milestones from M3.6 onward, projects combined throughput
> trajectory, identifies what's needed BEYOND known plans to hit
> "world-first" status.

## Priority & ROI

| Track | Priority | ROI basis | Negative case | Kill criteria |
|---|---|---|---|---|
| **M3.9 Phase 1A v3** (multi-slot async readback) | **P0** | Long-ctx 8k TTFT 4961→~2500 ms (50% reduction) — measured against 4961 ms F4-Small baseline + 2367 ms vLLM control. Blocker chain: F4-Small precondition `deferred_decode_emit.is_none()` (verified `2a534c4` git-blame) blocks Mixed at long-ctx steady-state, forces Split (10× tax measured `4a3612b`). | If ARLE long-ctx workloads stop being a target (project pivot to high-conc only). High-conc bench unchanged → no regression. | Phase 0 metrics show `mixed_ok_false_count / total < 5%` after F4-Small. Means deferred event resolves quickly enough that Mixed is rarely blocked → Phase 1A v3 has nothing to fix. |
| **M_pf** (prefetch wiring) | **P1** | Long-ctx multi-tenant TTFT cliff: 6k cached prefix → ~144 ms H2D. Overlap with prefill removes that wait. **Confirmed substrate**: `submit_prefetch_plan` exists, zero call sites (`50ae808`). | Single-tenant or no-cache workloads see no benefit. May regress if prefetch evicts useful T0 blocks. | Phase 1 bench at multi-tenant shared-prefix shape shows < 20% TTFT improvement → revert. |
| **M_b.2 Phase 1** (TileLang FP8 attn full integration) | **P1 / P2 conditional** | Phase 1 trace showed `decode_attention_varlen_quantized_partial_kernel` 41.6% GPU. But F4-Small + Phase 1A v3 likely move bottleneck off attention (per `2e60844` data showing kernel is fast at batched mode). Re-evaluate after P1A v3 lands. | M3.9 Phase 1A v3 closes long-ctx gap → kernel work no longer needed for parity. | After P1A v3 lands, if kernel still > 30% GPU AND ARLE < vLLM at 2 of 4 standard shapes → proceed. Else demote. |
| **Spec-decode + F4-Small compound** | **P2** | Speculation theoretical 1.5-2× decode throughput. ARLE spec infrastructure exists (`scheduler/cuda/spec_path.rs`). | Acceptance rate < 0.6 → speculation negative. Long-ctx KV size scaling makes draft/verify per-token cost equal → no compound gain. | A0 smoke at high-conc shows `spec_acceptance_rate < 0.6` after 30s warmup → revert. |
| **INT4 KV compression** | **P3** | 30% memory bandwidth reduction. Long-ctx attention is bandwidth-bound. | Numerical regression > 1% PPL on Qwen-class. ~400 LOC, deep code change. | Pre-merge eval at MMLU shows > 0.5% drop → abandon. |

**Tier ordering**: P0 → P1 → P2 → P3, gated by per-tier ROI evidence.
P1 tracks are run AFTER P0 lands and bench data refreshed.

**Why P0 = M3.9 Phase 1A v3 (not M_pf)**: Phase 1A v3 closes a
known production-blocking 10× tax with high-confidence path
(infrastructure-level fix); M_pf is opportunistic gain that only
materializes for shared-prefix workloads. Plus Phase 1A v3 is in
codex's active pipeline.

**Retroactive update note**: this section was added on
[next-visit-after-rule] per the new memory rule
`feedback_docs_priority_roi_evidence.md`. Original draft (commit
`d16effe`) lacked explicit ROI/kill criteria.

## Current state (2026-05-07 EOD)

### Confirmed bench numbers

| Workload | ARLE | vLLM | Δ |
|---|---:|---:|---|
| **High-conc 1k/256/c=64 out tok/s** | **843** | 647 | **ARLE +30.3% ✓** |
| High-conc per-row ITL | **0.99 ms** | 1.43 ms | **ARLE 1.45× faster ✓** |
| Long-ctx 4k/c=4 TTFT | 3403 ms | unmeasured | TBD |
| Long-ctx 8k/c=4 TTFT | 4961 ms | 2367 ms | **vLLM 2.10× faster ✗** |
| Long-ctx 8k/c=4 ITL | 23.9 ms | 26.7 ms | ARLE 1.12× faster |
| Long-ctx 8k/c=4 out tok/s | 92.2 | 105.6 | vLLM +14.5% |
| c=1 latency 512/128 ITL | 14.0 ms | unmeasured | (single-row baseline) |

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
- ⏳ **M3.9 Phase 0** (in flight): instrumentation to measure
  Mixed/Split routing + Ok(false) reasons.
- 🔜 **M3.9 Phase 1A v3** (designed, not yet briefed): multi-slot
  async readback buffers → unblocks Mixed path at long-ctx.

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

| Cumulative state | High-conc out tok/s | Long-ctx 8k TTFT | Long-ctx 8k out tok/s |
|---|---:|---:|---:|
| Baseline (Phase 1 trace) | 462 | 4961 ms | 92 |
| **+ F4-Small** (landed) | **843** | 4961 ms | 92 |
| + B.1.2 (landed, marginal) | 843 | 4933 ms | 92 |
| + M_b.1 Phase B (landed, marginal) | 843 | 4933 ms | 92 |
| **+ M3.9 Phase 1A v3** (pending) | 850 (no change) | **~2500 ms** | ~150 |
| + M_b.2 Phase 1 (TileLang FP8 attn) | 900 (kernel speedup) | ~2200 ms | ~170 |
| **+ M_b.2 Phase 2** (kernel+autotune) | **1000+** | ~1900 ms | ~200 |

**Stretch goal**: ARLE 1000+ tok/s high-conc + ARLE long-ctx
faster than vLLM = "**world-first**" at this hardware tier.

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
