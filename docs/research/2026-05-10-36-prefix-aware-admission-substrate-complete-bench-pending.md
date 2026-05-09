---
title: #36 PrefixAwareAdmission — substrate FULLY WIRED, only bench evidence pending
date: 2026-05-10
type: research
status: scope-rescoped-impl-done-bench-pending
---

# #36 PrefixAwareAdmission — substrate FULLY WIRED, only bench evidence pending

> P0 survey before plan body (per `feedback_p0_survey_before_plan_body.md`)
> applied to #36 this tick. Finding: **all three M_b3 implementation
> steps are already in main**. Original task scope (~200-400 LOC
> scheduler-side wiring) **collapses** to a bench-validation A/B run.
> Survey saved a stale brief that would have re-implemented what exists.

## Verified existing state (2026-05-10 main `08d9b7e`)

### Step 1 — admission_allows signature change (LANDED `7c8fd61`)

```bash
$ git log --oneline -S "fn admission_allows(&self, signals: SchedulerSignals)" -- infer/src/scheduler/types.rs
7c8fd61 fix(scheduler): admission_allows takes SchedulerSignals — B3 Step 1 wiring
```

Current state at `infer/src/scheduler/types.rs:688`:

```rust
fn admission_allows(&self, signals: SchedulerSignals) -> bool { ... }
```

Two legacy callers at `types.rs:798` and `:841` still pass
`SchedulerSignals::queue_state(current, 0)` (i.e., `prefix_hit_tokens=0`)
— that is M_b3's intended Step 1 endpoint ("leave prefix_hit_tokens=0
defaults at all callers — same behavior as before").

### Step 2 — RadixCache signal pipeline (LANDED, commit TBD)

`infer/src/scheduler/cuda/runtime/admission.rs:409-435`:

```rust
fn prefix_aware_admission_allows_plan(
    &self,
    plan: &PrefixAdmissionPlan,
    queued_requests: usize,
) -> bool {
    if !matches!(self.config.admission_policy, SchedulerAdmissionPolicy::PrefixAware)
        || !self.config.prefix_cache_enabled { return true; }

    let signals = self.prefix_aware_admission_signals(plan, queued_requests);
    if !signals.is_cold_request() || self.config.max_waiting_requests == 0 { return true; }

    let cold_headroom = self.config.cold_headroom
        .unwrap_or(self.config.max_waiting_requests / 4);
    let cold_soft_cap = self.config.max_waiting_requests
        .saturating_sub(cold_headroom);
    signals.queued_requests < cold_soft_cap
}
```

Real `PrefixAdmissionPlan` → `SchedulerSignals` conversion via
`prefix_aware_admission_signals(plan, queued)` populates
`prefix_hit_tokens` from the plan's reusable prefix length. Cold-request
detection works. **Anti-starvation `prefix_aware_fail_open_candidate`
also wired** (`admission.rs:437-458`) so cold-only traffic still fills
idle slots without warm-only deadlock.

### Step 3 — Default policy switch via CLI (LANDED)

`infer/src/main.rs:124`:

```rust
admission_policy: String,    // CLI flag --admission-policy
```

`infer/src/main.rs:702`:

```rust
let admission_policy = SchedulerAdmissionPolicy::parse(&args.admission_policy)?;
```

`infer/src/scheduler/types.rs:478-493`:

```rust
pub enum SchedulerAdmissionPolicy {
    QueueBound,
    PrefixAware,
}

impl SchedulerAdmissionPolicy {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "queue-bound" => Ok(Self::QueueBound),
            "prefix-aware" => Ok(Self::PrefixAware),
            other => bail!("unknown admission policy: {other}"),
        }
    }
}
```

Default at `types.rs:214`: `SchedulerAdmissionPolicy::QueueBound`. Tests
at `types.rs:886-927` cover both branches with explicit signal shaping.

### Dispatch site at `types.rs:371-387`

```rust
pub fn admission_policy_allows(&self, signals: SchedulerSignals) -> bool {
    match self.admission_policy {
        SchedulerAdmissionPolicy::QueueBound => QueueBoundAdmission { ... }.allow(signals),
        SchedulerAdmissionPolicy::PrefixAware => {
            PrefixAwareAdmission::with_cold_headroom(...).allow(signals)
        }
    }
}
```

Two parallel wiring paths exist:
- `admission_policy_allows()` (new, dispatches by enum) — used by main scheduler loop
- `admission_allows()` (legacy, always QueueBound, signals=zero) — used at lines 798/841

Both behaviors are correct **for the default `QueueBound` path**; when
the user passes `--admission-policy prefix-aware`, the new dispatcher
takes over and the prefix-aware fast-path at `admission.rs:350` runs.

## What was wrong with the existing M_b3 plan

`docs/plans/M_b3-prefix-aware-admission-step1-directive.md` last status
section (per `tail -60`):

> Status:
> Step 1 directive ready for codex pickup(0.5 day,30-50 LOC)。
> Steps 2-3 follow sequentially。Total B3 work 2-3 days codex,closes
> SGLang multi-tenant 2× gap = -50% TTFT on multi-tenant axis。

This was accurate **at the time of authoring** (pre-`7c8fd61`) but is
now stale. Task #36 in the task list (description "wire
PrefixAwareAdmission to admission policy stack. Likely 200-400 LOC
scheduler-side work") is also stale.

## Rescoped #36 — bench-validation A/B

Original scope: 200-400 LOC scheduler-side wiring + tests + bench.
Actual remaining scope:
- 0 LOC implementation (substrate complete)
- 1 bench run pair (QueueBound baseline vs PrefixAware) on multi-tenant
  burst workload
- 1 wins or errors entry per bench outcome

### Concrete bench command

Per `M_world1` multi-tenant burst protocol (cited in M_b3 plan):

```bash
# Baseline arm — QueueBound (current production default)
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
  scripts/bench_guidellm.sh 36-bench-A-queuebound \
    --concurrencies 8 --max-seconds 120 --warmup 10 \
    --data 'prompt_tokens=2048,prompt_tokens_stdev=512,output_tokens=128,output_tokens_stdev=32' \
    -- --admission-policy queue-bound

# Treatment arm — PrefixAware (the wiring under test)
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
  scripts/bench_guidellm.sh 36-bench-B-prefixaware \
    --concurrencies 8 --max-seconds 120 --warmup 10 \
    --data 'prompt_tokens=2048,prompt_tokens_stdev=512,output_tokens=128,output_tokens_stdev=32' \
    -- --admission-policy prefix-aware
```

Matched controls (per kernel-optimization skill Phase 5):
- Same model + weights + KV format
- Same `--num-slots` / `--max-seq-len` / `--max-waiting-requests`
- Same prompt distribution (must include some session-affinity overlap
  to exercise prefix-cache hit path; pure-cold traffic gives `is_cold`
  = true for every request and PrefixAware degrades to QueueBound +
  fail-open noise)
- Same n=3 σ-tight for license/kill decision

### License or kill gates

Per kernel-optimization skill Phase 8 + M_b3 KILL criteria:

| Outcome | License | Kill |
|---------|---------|------|
| Multi-tenant TTFT | Δ ≥ +20% improvement at p50 | Δ < +5% or any regression |
| Throughput | Δ ≥ +10% | Δ < 0% |
| Tail latency p99 | Δ ≥ -10% | Δ > +20% |
| Cold-request fairness | cold p95 within 1.5× warm p95 | cold p95 > 3× warm p95 (starvation) |

If license: target a Tier 1 wins entry mirroring `c44788f` format
(matched-control 60s window, server-side `engine_ttft_us` ground truth,
cardinality evidence — here, prefix hit-rate distribution from
`/v1/stats`).

If kill: errors entry must name (a) which sub-gate failed, (b) what
`/v1/stats` shows about prefix-hit distribution under load, (c) whether
`cold_headroom` default (`max_waiting/4`) is the actual lever, and (d)
whether to retire `--admission-policy prefix-aware` flag or just
default `QueueBound`.

### Open question — does this workload actually exercise the gate?

`prefix_aware_admission_allows_plan` only takes effect when
`is_cold_request()` AND `signals.queued_requests >= cold_soft_cap`. In
a low-pressure bench (queue depth never approaches `max_waiting`), the
gate never fires and the bench measures nothing. Two mitigations:

1. **Set `--max-waiting-requests` low** (e.g. 4) so cold_soft_cap
   activates at modest queue depth. Cite this as a deliberate gate
   trigger, not a default-behavior comparison.
2. **Cite `/v1/stats` `prefix_aware_admit_deferrals` counter** (if
   exists; if not, that itself is a sub-task — add internal counter for
   "deferrals due to PrefixAware cold_soft_cap") to prove the gate
   actually fired during the bench.

If neither holds, the bench is null and #36 reverts to "needs
counter-instrumentation first" before validation A/B is meaningful.

## Pickup brief (next tick or codex)

When the GPU window opens for #36:

1. Pre-flight: `grep -n "prefix_aware_admit_deferral\|cold_soft_cap_hits" infer/src/`
   to find / add an internal counter that proves the gate fired.
2. Run baseline + treatment per the bench commands above.
3. Decide license vs kill per the gate matrix.
4. Land wins or errors entry; cross-link from this research entry.

Estimated wall time: 1.5h GPU-side (baseline + treatment + 2 retries
for σ < 5%) + 0.5h doc work. Much smaller than the original 2-3 day
implementation estimate.

## Cross-references

- M_b3 directive: `docs/plans/M_b3-prefix-aware-admission-step1-directive.md`
  (substrate finding obsoletes its Status section but plan body
  remains useful for KILL-gate framing and SGLang gap context)
- M_e2 prompt-trie prefix cache plan:
  `docs/plans/M_e2-prompttrie-prefix-cache.md`
- M_ibp in-batch prefix caching plan:
  `docs/plans/M_ibp-in-batch-prefix-caching.md`
- Step 1 commit: `7c8fd61` (2026-05-09)
- Substrate sites:
  - `infer/src/scheduler/policy.rs:80-130` PrefixAwareAdmission impl
  - `infer/src/scheduler/types.rs:371-387` admission_policy_allows dispatcher
  - `infer/src/scheduler/types.rs:478-493` SchedulerAdmissionPolicy enum + parse
  - `infer/src/scheduler/cuda/runtime/admission.rs:350` fast-path call site
  - `infer/src/scheduler/cuda/runtime/admission.rs:409-458` plan-aware gate + fail-open
  - `infer/src/main.rs:124,702` CLI flag wiring
- Closes 4k/c=4 SGLang +76.6% gap delivered separately by #40 Path B.2
  (`c44788f`); #36 axis is the multi-tenant 2× gap

## 状态

#36 wiring substrate **complete**. Task pending = bench-validation
A/B + license-or-kill decision + counter-instrumentation if needed.
Survey saved a brief that would have re-implemented existing code.
Next tick: when GPU window opens, run the bench command pair above.
