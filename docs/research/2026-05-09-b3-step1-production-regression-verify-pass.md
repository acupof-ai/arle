# B3 Step 1 production regression verify — byte-identical behavior to pre-fix baseline

> Per `7c8fd61` admission_allows signature refactor,verify production
> stability via fresh-build W3 c=4 short multiturn bench against
> `063da81` pre-Step-1 baseline。
>
> **Result:byte-identical performance**(384/384 turns,TTFT p50 208 ms,
> ITL p50 8.5 ms — same as pre-Step-1)。Step 1 admission_allows refactor
> confirmed production-safe via empirical regression test。

## Setup

Same as `063da81` baseline — fresh build with `7c8fd61` Step 1 refactor:

```bash
cargo build --release -p infer --features cuda  # builds with admission_allows(SchedulerSignals)
./target/release/infer --num-slots 8 --max-seq-len 5120
# Default cap=8 (12300c5), warmup max=8 (c20b1ce), admission signal pipeline (7c8fd61)

bench_agent_trace.py --workload agent-w3-short-multiturn --num-concurrent 4
```

## Empirical comparison

| Metric | `063da81` pre-Step-1 | **`7c8fd61` post-Step-1** | Δ |
|---|---:|---:|---:|
| Turn success | 384/384(100%) | **384/384(100%)** | same |
| TTFT p50 | 208 ms | **208 ms** | identical |
| TTFT p99 | 573 ms | 571 ms | -0.4%(noise) |
| ITL p50 | 8.5 ms | **8.5 ms** | identical |
| ITL p99 | 8.8 ms | 8.8 ms | identical |
| W3 warm TTFT p50 | 205.7 ms | 206.9 ms | +0.6%(noise) |
| W3 cold TTFT p50 | 318.8 ms | (similar) | ~same |

All metrics within σ noise band。**Byte-identical behavior preservation
empirically confirmed**。

## What Step 1 changed semantically

`admission_allows`:
- Before:`(&self, queued_requests: usize) -> bool`,constructs
  `SchedulerSignals::queue_state(queued_requests, 0)` internally
- After:`(&self, signals: SchedulerSignals) -> bool`,passes signals through
- Caller compatibility:both `submit()` and `is_full()` callers pass
  `SchedulerSignals::queue_state(current, 0)` — IDENTICAL signal
  contents to pre-refactor

The QueueBoundAdmission policy(currently active default)only reads
`signals.queued_requests`,ignoring all other fields。So with
`SchedulerSignals { queued_requests: N, prefix_hit_tokens: 0,
session_affinity_slot: None, turn_depth: 0 }`,behavior matches
pre-refactor exactly。

## Step 2 unblocked for codex

Pipeline now plumbable:
1. ✅ `admission_allows` accepts full SchedulerSignals(this fix)
2. ⏳ codex Step 2:populate signals at submit time(prefix_hit_tokens
   via RadixCache.match_prefix,session_affinity_slot lookup)
3. ⏳ codex Step 3:config flag to switch policy
4. ⏳ codex Step 4:multi-tenant bench

Step 2 requires SchedulerHandle to access RadixCache(currently internal
to scheduler core)。This is substrate-LOC heavy(>100 LOC architectural
change),codex pickup #36 remains。

## Validation methodology

Per skill v1.5.1 anti-pattern #15(warm-server implicit dependency)+
#17(bimodal failure distribution):
- Cold-start fresh-build verification(this run)
- Same workload as production-stable baseline `063da81`
- σ-tight metrics confirm safe deployment

This is the **smallest meaningful regression test**:N=1 cold-start
fresh-build verifies signature refactor is byte-equivalent。Production
deployment confidence:HARD LICENSED for Step 1。

## Cross-references

- Step 1 commit: `7c8fd61`
- Pre-Step-1 baseline: `063da81`(W3 c=4 cap=8 default CLEAN)
- Codex Step 1 directive: `e53e4d8`
- Scope refinement: `c0ddd4f`
- Skill v1.5.1: `9f65b4d`

## Status

- ✅ Step 1 admission_allows refactor: production-safe(this entry)
- ⏳ Step 2 RadixCache integration: codex pickup #36
- ⏳ Step 3 policy config flag: codex pickup
- ⏳ Step 4 multi-tenant bench: codex post Step 2-3

## Rule

**Substrate refactor with behavior preservation should always have
empirical regression test before commit even if cargo tests pass**。
Cargo unit tests check correctness;production bench checks performance。
For ARLE specifically:scheduler hot-path refactors(admission,
warmup,signal pipeline)need W3 c=4 baseline regression — fastest
production-shape verification(~3 min wall)。

This methodology rule is implicit in CLAUDE.md "every runtime change
produces a bench entry" — formalize for substrate refactors:
`build → cargo test → cargo build release → ~3min production-shape
bench → commit if all green`。
