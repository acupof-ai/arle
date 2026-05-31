# Control Plane — making the scheduler tick obvious (`TickPhase` + folded batching policy)

> Root-cause treatment of "scheduler tick 不显然" — the one plane the
> attention/KV, dispatch-governance, and operator-library docs do **not** touch.
> Every claim below is source-verified against HEAD `621c84c4` (2026-05-31).
> Companion: [`attention-kv-architecture.md`](attention-kv-architecture.md),
> [`state-plane-slot-extent-contract.md`](state-plane-slot-extent-contract.md).
> Status: **approach-first artifact, awaiting sign-off.** Cross-cutting (>5
> files); per root `AGENTS.md` no runtime code lands until accepted. This is a
> **deletion/convergence refactor** — it names a state machine that already
> exists and folds three scattered policy inputs into the one site that already
> decides plan shape. It introduces **no new abstraction for hypothetical
> consumers** (`memory/feedback_no_speculative_interface_shaping`).

## TL;DR — the tick is already a state machine; it just has no name

A reader who opens `step()` (`execution.rs:926`) to answer *"what does one
scheduler tick do?"* cannot read it linearly. The honest control flow is:

```
run_inner loop  (scheduler_loop.rs:117-215)
  drain_*  →  wait_for_wakeup  →  assign_slots  →  step(assign_us)  →  cleanup  →  metrics
                                                      │
                                                      ▼  step() (execution.rs:926)
   ┌─ if pending_prefill.is_some():  step_prefill_readback();  if !ready → sleep(100us); RETURN
   ├─ if pending_decode.is_some()  :  step_decode_readback();   if !ready → sleep(50us);  RETURN
   └─ else                          :  snapshot → build_candidate_plan → launch_gpu_command → dispatch_decode_emits
```

The three branches are **three states of a machine** — `ReadbackPrefill`,
`ReadbackDecode`, `PlanAndLaunch` — but they are encoded implicitly as **three
`Option` fields** (`core.rs:160-162`: `pending_decode`, `pending_prefill`,
`deferred_decode_emit`) plus early-return + `sleep`. Two consequences, both
source-verified:

1. **Sample + state-update are invisible at the launch site.** `step()` ends at
   `launch_gpu_command` (`execution.rs:1030`); the lines after it
   (`:1044-1060`) are only metrics. The actual token sampling and
   `req.phase`/`generated_tokens` mutation happen **on the next tick** inside
   `step_decode_readback` / `finish_prefill_batch` (`prefill.rs:607`). A reader
   sees "launch" and reasonably expects "sample" next — it is a full loop turn
   away, behind an early-return.

2. **The machine invariant is enforced by a runtime panic, not the type
   system.** `execution.rs:1017-1024` asserts `pending_decode.is_none()` and
   `pending_prefill.is_none()` "before the next launch" — i.e. *you may only
   enter `PlanAndLaunch` from a cleared state*. That is exactly a state-machine
   transition rule, currently a `panic!` waiting to fire instead of an
   unrepresentable state.

The naive fix — "extract a `tick()` that does readback then launch in one
function" — does **not** survive the source: readback is deliberately split
across loop turns so intake/admission overlaps the previous tick's GPU compute
(`scheduler_loop.rs:140-143` comment; the 100us/50us backoffs at
`execution.rs:953/986` are load-bearing anti-spin fixes from the 2026-05-25
nsys H5 finding). Collapsing the turns would **reintroduce a regression**. So
the fold is not "merge the phases" — it is "**name** the phases."

## The grounded boundary

| Today (implicit) | Grounded form | Why |
|---|---|---|
| 3 `Option` fields + early-returns + `assert!` | **`enum TickPhase { ReadbackPrefill, ReadbackDecode, PlanAndLaunch }`** derived by **one pure fn** `tick_phase(&self) -> TickPhase` over `(pending_prefill, pending_decode, deferred_decode_emit)`; `step()` becomes a `match` over it | the machine exists; a named phase makes the readback→launch cycle legible in one place and the "cleared before launch" invariant a `match`-arm fact, not a runtime panic |
| batching policy split across **plan_step (shape) + PrefillBudget (token budget) + candidate selection (rows)** | fold the budget + selection inputs **into the `plan_step` call site** (`execution.rs:599`) which *already* decides `Decode/Mixed/Prefill/Split` — one struct in, one `StepPlan` out | `plan_step` is already the home for "what shape"; "how many tokens / which rows" should be decided **next to it**, not in two sibling files |

`tick_phase` is a **pure function** (over three `Option::is_some()` reads + the
`deferred_decode_requires_readback_before_launch` predicate). Same killer
property as `oplib::linear::plan`: *"given this pending-state, which phase
runs?"* becomes a GPU-free `assert_eq!(tick_phase(state), PlanAndLaunch)` unit
test — the tick's control flow gets the same CPU-testability the compute plane
already has via the landed `oplib`.

## What it collapses — the improvement that drives it

The traced real improvement is **decode-aware prefill chunking** (a named want
in the half-done scan: "smaller prefill chunks while decode is running, scaled
by KV utilisation"). Today it must edit **6-8 sites**, source-verified:

1. `scheduler/types.rs` — `SchedulerConfig` new field
2. `scheduler/cuda/budget.rs` — `PrefillBudget::from_scheduler` (token cap)
3. `scheduler/cuda/execution.rs:605` — `collect_prefill_candidates` (pass hint)
4. `scheduler/cuda/execution.rs:454` — `capped_prefill_reservation` (use hint)
5. `scheduler/cuda/execution.rs:606` — `select_prefill_candidates` (row cut)
6. `scheduler/cuda/execution.rs:599` — `plan_step` (shape, already here)
7-8. optional `policy.rs` + per-session variant

The decision is **smeared across budget.rs + 3 execution.rs helpers + plan_step**
because there is no single "given (has_decode, kv_util, config), produce the
admission budget *and* plan shape" site. After the fold:

- **`plan_step` owns the whole decision.** A new batching policy is **1 site**:
  one `match`/branch inside `plan_step`, reading a `BatchInputs { has_decode,
  kv_util, config, candidates }` it already assembles (`execution.rs:600-606`).
  `PrefillBudget` + candidate selection become *inputs computed for* `plan_step`,
  not independent decision points.

This is convergence, not a new trait: there is exactly **one** batching policy
today, so a `BatchPolicy` *trait* would be speculative. The fold is "put the
three scattered inputs where the shape decision already lives," which a real
second policy (decode-aware chunking) immediately exercises.

## Migration — independently shippable, each revertible

- **Step 1 (PURE, CPU-tested, no behaviour change).** Add `enum TickPhase` +
  `fn tick_phase(&self) -> TickPhase`. Add a CPU unit test sweeping the eight
  `(pending_prefill, pending_decode, deferred_emit)` combinations →
  expected phase. **Do not** rewire `step()` yet. Pure addition.
- **Step 2 (rewrite `step()` as `match tick_phase()`).** Replace the three
  `is_some()` early-returns with one `match`. The `ReadbackPrefill`/
  `ReadbackDecode` arms keep their exact backoff `sleep`s and early `return`s
  (load-bearing). The `PlanAndLaunch` arm absorbs the `assert!(pending.is_none())`
  as a *match-arm precondition that can no longer be violated* (you only reach
  it when both are `None`). Behaviour bit-identical; verify with the existing
  `runtime/tests.rs` scheduler tests + one guidellm c-sweep (no numeric delta
  expected — pure restructure).
- **Step 3 (fold batching inputs into `plan_step`).** Move
  `PrefillBudget::from_scheduler` + `collect/select_prefill_candidates` to be
  computed inside / immediately adjacent to `plan_step`, returning the budget
  alongside the `StepPlan`. Verify byte-identical plan choices on a replay of
  the `runtime/tests.rs` fixtures.
- **Later (gated, separate):** the decode-aware-chunking policy itself lands as
  the *first consumer* of the folded site — proving the 1-site claim end-to-end
  before any further policy is added.

## License-or-kill

- Step 1-2 are **deletion/legibility refactors**: the gate is *behaviour-identical*
  (bit-identical greedy output + no >2% guidellm TTFT/ITL delta on the affected
  backend). If a phase-machine rewrite cannot stay bit-identical, it is wrong —
  kill and keep the implicit form.
- Step 3's fold is licensed **only** by the decode-aware-chunking consumer
  landing as 1 site. If folding the inputs does not actually reduce that
  improvement to one site (e.g. the budget genuinely needs to live in `budget.rs`
  for a reason this design missed), kill the fold and keep `tick_phase` alone —
  `tick_phase` stands on its own legibility/invariant merit.

## Honest gaps (not self-deception)

- `tick_phase` makes the *cycle* legible; it does **not** by itself make the
  deferred sample/update *location* obvious. A reader still has to know that the
  `ReadbackDecode` arm is where sampling happens. Mitigation is doc + arm naming,
  not a structural guarantee. If this proves insufficient in review, a follow-up
  could thread the sampled-token commit through an explicit return value rather
  than a side-effecting readback — but that is **not** proposed here (it would be
  speculative until the legibility-via-naming attempt is shown to fail).
- This design covers the **CUDA** scheduler (`scheduler/cuda/`). The Metal
  scheduler runtime has its own loop; whether `TickPhase` generalises is **out
  of scope** until a Metal consumer forces it — same trip-wire discipline as the
  kernel-crate anti-goals in [`architecture.md`](../architecture.md).
- Spec-decode adds a fourth implicit sub-state via `route_spec_plan`
  (`execution.rs:661/679`) + `spec_path.rs`. Whether it becomes a `TickPhase`
  variant or stays inside `PlanAndLaunch` is **deferred** to when spec-decode
  produces a real throughput lift (currently paused, per
  [`architecture.md` §Speculative Decode Framework](../architecture.md)).
