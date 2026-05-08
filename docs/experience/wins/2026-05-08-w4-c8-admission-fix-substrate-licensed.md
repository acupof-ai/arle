# W4 c=8 admission-fix substrate LICENSED — 0% → 100% turn success,40× batch occupancy gain

> Codex `369292f` page_budget hypothesis(reserve growth for ALL active
> slots vs only runnable decode slots)→ implementation in
> `infer/src/scheduler/cuda/execution.rs`(uncommitted at HEAD,verified via
> codex local build)。Codex's W4 c=8 verify bench landed at 16:31 with
> **256/256 turns succeeding**(vs W4 c=8 deadlock 0/256 in `e3669d4`)。
>
> First production-spec W3/W4 baseline data from ARLE under master §2.1。
> Substrate fix land-ready;wins entry per CLAUDE.md mandatory bench rule。

## Phase 1 — Target

| Field | Value |
|---|---|
| Metric | turn success rate + ITL on W4 c=8 canonical(8 × 8K prompt + 256-token resume) |
| Baseline | W4 c=8 deadlock(`e3669d4`):0/256 turns,active=8 prefill_queue=7 prefill_rows=0 stuck |
| License threshold | ≥ 95% turn success(out of 256)+ ITL p50 ≤ 30 ms |
| Kill threshold | < 50% turn success OR deadlock signature(prefill_rows=0 + tokens_out=0) |

## Phase 5 — Single-variable A/B

**Single variable**:`PrefillBudget::from_scheduler_for_decode_slots`
iteration scope:**all `running_batch` slots(buggy)** vs **only runnable
decode_slots(fixed)**。

All else unchanged:
- Same model:`Qwen3-4B-W4A16-sym-g128-marlin`
- Same workload:`agent-w4-tool-resume`(128 sessions × 2 turns,8K prompt)
- Same harness:`scripts/bench_agent_trace.py --num-concurrent 8`
- Same server config:`--num-slots 16 --max-seq-len 9216`
- Same hardware:sm_89 4070 Ti SUPER

## Results

| Metric | W4 c=8 deadlock(`e3669d4`) | **W4 c=8 admission-fix** | Δ |
|---|---:|---:|---:|
| Turn success rate | **0/256(0%)** | **256/256(100%)** | **+100 percentage points** |
| ITL p50 | n/a(deadlock) | **16.47 ms** | new datapoint |
| ITL p99 | n/a | 17.04 ms | tight σ ~3.5% |
| TTFT p50 | n/a | 11768 ms | 8 × 8K prompt admission backlog |
| TTFT p99 | n/a | 72515 ms | last sessions waited |
| engine_ttft_us(server-side) | n/a | 2000 ms | true TTFT on stream |
| engine_itl_p50_us | n/a | 35 ms | server-side step ITL |
| **engine_batch_occupancy** | **0.0200(2%)** | **0.8253(82.5%)** | **+40×** |
| Tokens out | 0 | **44665** | new datapoint |
| prefix_hit_rate | n/a | 97.0% | RadixCache fully effective |
| session_affinity_hit | 0 / 16 miss | 645 / 20 miss(97%) | session reuse working |
| engine_kv_tier_hit_T0 | n/a | 96.99% | KV tier reuse |

Bench artifact:`bench-output/2026-05-08-arle-w4-c8-admission-fix.json`
(local;gitignored)。

## Phase 8 verdict — LICENSED

| Threshold | Result | Verdict |
|---|---|---|
| ≥ 95% turn success | 100% | ✅ LICENSE |
| ITL p50 ≤ 30 ms | 16.47 ms | ✅ LICENSE |
| No deadlock signature | engine_batch_occupancy 82.5% | ✅ LICENSE |

**LANDED** as W4 c=8 admission-fix substrate baseline。Master §7.1 P0.0
binding workload P0 axis production-shape evidence is established for W4。

## What this proves

1. **Codex page_budget hypothesis CORRECT**:reserving growth for ALL
   `running_batch` slots(including 7 still in `prefill_queue` waiting)
   exhausts page budget before any candidate prefills → deadlock。
   Limiting reservation to runnable `decode_slots` only allows admission
   to rotate naturally。

2. **Workload-dependent deadlock confirmed in fix**:both W3(1K × c=16,
   total 16K)and W4(8K × c=8,total 64K)hit the same fingerprint
   pre-fix(`cb087c7` + `e3669d4`),and both axes share the same root
   cause(now empirically verified for W4 — W3 c=16 verify TBD).

3. **Production-shape baseline now usable**:
   - W4 c=8 ITL 16.47 ms is the no-spec baseline for Medusa(`528844c` plan)
   - Throughput baseline:44665 tokens / wall-time = real per-second metric
   - Master §7.4 P1.1 spec axis re-test prerequisite(W3/W4 baseline first)
     is now satisfied for W4

## Phase 7 — Tradeoffs(post-fix)

| Axis | Status | Note |
|---|---|---|
| LOC complexity | ⚠ ~5-10 LOC fix in `execution.rs` | minor surgical change |
| Hardware specificity | ✅ none | scheduler invariant fix |
| **TTFT under load** | ⚠ p50 11768 ms | admission queue backlog at burst-of-8 large-prompts;NOT a deadlock,just queue wait |
| Memory budget | ✅ same | no extra reservation |
| Numerical correctness | ✅ no model output changes | scheduler-only fix |
| **Generality** | ⚠ verify W3 c=16 too | both shapes must unblock per `e3669d4` rule |

Major remaining axis:**W3 c=16 verification still pending**(codex bench
order was W4 first;W3 c=16 should follow with same server config or
re-launched with different `--max-seq-len`)。

## Cross-references

- Codex root cause hypothesis: `369292f`(`docs/research/2026-05-08-w3-c16-deadlock-page-budget-hypothesis.md`)
- W3 c=16 deadlock initial: `cb087c7`(`docs/experience/errors/2026-05-08-w3-c16-deadlock-not-just-admission.md`)
- W4 c=8 deadlock confirmation: `e3669d4`(`docs/experience/errors/2026-05-08-w4-c8-deadlock-confirms-workload-dependent.md`)
- W3 c=4 baseline(off-spec workaround): `370a267`
- W4A16 LICENSED: `f6f3af3`
- Master §7.1 P0.0 baseline mandate
- Bench artifact:`bench-output/2026-05-08-arle-w4-c8-admission-fix.json`(local)
- Codex implementation:`infer/src/scheduler/cuda/execution.rs`(uncommitted at HEAD,5-10 LOC)
- Pre-commit verification:codex `bea90bb` Phase 1b smoke

## Status note — pending codex commit

Codex's hot-path source changes(5 files including `execution.rs`)are
uncommitted at the time of this entry。This wins entry documents the
empirical W4 c=8 verification numbers from codex's pre-commit local build。
Once codex commits + pushes the substrate fix:
1. This entry's commit-SHA references will resolve concretely
2. Reproducibility verified via `git checkout <fix-sha> && cargo build && bench`
3. W3 c=16 verify bench will be run against the committed binary

If for any reason the codex commit doesn't land(e.g.,e2e regression
forces revert),this entry stays as evidence the fix WAS implementable
and produced correct numbers in pre-commit form。

## Rule

When a substrate fix produces empirical evidence of correctness BEFORE
its commit lands(uncommitted hot-path source + verified locally):
**document the win with explicit pre-commit attribution**。Don't wait for
the commit landing because:
- Bench artifacts go stale fast
- Codex iteration may produce multiple verify cycles
- Cross-workload rule(W3 + W4 both unblock)demands single-shot evidence
  preservation

The wins entry with explicit "pending-codex-commit" status is preferred
over silent loss of evidence。
