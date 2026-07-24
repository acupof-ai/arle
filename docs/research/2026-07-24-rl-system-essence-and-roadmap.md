# The RL system's essence: a correspondence invariant, then an economics ratio

Research + planning insight, 2026-07-24. Sources: THUDM/slime 358-issue landscape
(external map of where the mainstream Megatron+SGLang RL stack bleeds) crossed
with ARLE's own decoded incidents (two null-gradient episodes, the group-stagger
KILL, the 5/28 sidecar miss). Explanation doc; the executable plan is
[plans/2026-07-24-sweetspot-corpus-first-gradient.md](../plans/2026-07-24-sweetspot-corpus-first-gradient.md).

## Thesis — two axioms, strictly ordered

**Axiom 1 (correctness): the training side must see, token-exact, what the
behavior policy actually generated.** Every RL-system failure with a long tail
is a silent break of this correspondence, and every silent break traces to a
piece of state with more than one owner.

**Axiom 2 (economics): the system's real output is gradient-bearing tokens per
GPU-hour** — not tok/s. Rollouts discarded for length, zero-variance groups, and
saturated corpora all multiply throughput by zero.

Order matters: optimizing axiom 2 while axiom 1 is broken accelerates the
production of wrong gradients.

## Evidence

### External — slime's four load-bearing clusters are one disease

| Cluster | Closure rate | Shared root |
|-|-|-|
| Weight-refresh chain (Megatron→HF→SGLang) | 39% | weights owned by 3 abstractions |
| OOM / memory lifecycle | 24% | VRAM split across 3 CUDA contexts, no ledger |
| Long-seq + context parallel | 16% | correctness state sharded, no single-box repro |
| Agentic rollout (loss-mask misalign, multi-turn advantage) | — | forest trajectories forced into a matrix abstraction |

The issue-closure diagonal (fp8 89% ↔ longseq_cp 16%) shows closure rate tracks
single-machine reproducibility and inversely tracks abstraction-layer count.
The zombie-issue pool (P90 25.5 d, maintainers "looking into it") sits exactly
where state ownership is distributed: there is no cheap fix because the root is
structural.

### Internal — the same axioms, decoded in ARLE

- **Null gradient ×2** (errors/2026-07-03 corpus saturation; errors/2026-07-22
  length-wall skips): axiom-2 violations that each cost a debug cycle because
  they present as "training runs fine, loss = 0".
- **Sidecar miss 5/28 (~18% of distill windows in the stagger A/B)**: axiom-1
  crack. Code-anchored: the serve writes the token sidecar only when the engine
  stream reaches `delta.finish` (coordinator.rs:1221); the `delta.error`,
  client-disconnect, and stream-error break paths all leave a dump without a
  sidecar, and `cc_convert` skips those requests (cc_convert.rs:172). Whether
  the skipped requests carry unique trajectory content (signal loss) or are
  client-retry duplicates (benign dedup) is exactly the unattributed question.
- **Group-stagger KILL** (errors/2026-07-24): a premise inferred from code
  reading, falsified by one baseline counter read — the method-level lesson
  (measure the baseline observable first) applies to every roadmap item below.

## ARLE's structural position

Single-process train-infer unification gives axiom 1 by construction where
slime prays across glue: weight update is a LoRA-merge function call +
`invalidate_prefix_cache`, the student borrows the rollout engine's frozen base
zero-copy, and VRAM is explicitly modeled (`ckpt-gate modeled=…B`). The two
places axiom 1 is NOT yet constructive are the serve↔harness boundary (dumps +
sidecars, i.e. the 18%) and the harness↔trainer boundary (window matching).
Those boundaries are file-based, not process-based — which is why they are
where our cracks appear.

Standing design vetoes derived from the map:

1. **No tri-state memory handoff** (slime's torch_memory_saver: 8 crashes in
   its latest 60 issues). Any "train releases VRAM → infer takes over → hand
   back" proposal is rejected on this evidence.
2. **No fix that needs >1 box to reproduce.** Our answer to 30K-token
   trajectories is filter + early-stop (single-box verifiable), not repairing
   the 30K backward (our longseq_cp — it would inherit its 16% closure rate).
3. **One first-class lane** (ThinkingCap-27B agent-OPD). slime's GLM-vs-Qwen
   response-time gap (6 h vs 34 h) shows what pretending otherwise costs.

## Roadmap (priority order, each single-variable)

| # | Item | Axiom | Cost | Acceptance |
|-|-|-|-|-|
| P0 | **Sidecar-miss attribution**: audit the stagger-A/B dumps — classify each of the 5 missing sidecars (error / client abort / retry-duplicate); add a drop-reason counter at the three break paths | 1 | zero GPU, read-only | every miss attributed; counter lands only if misses carry unique content |
| P1 | **Trajectory-level seq early-stop**: harness knows token counts per turn; abort a sample once it exceeds `max_update_seq` instead of rolling 400 s then discarding (update_strategy.rs:650 wall) | 2 | small, harness-only | zero `SKIP … > max_update_seq` GPU-seconds |
| P1 | **Sweet-spot corpus → first real gradient** (existing plan, #173) | 2 | ~1 day pod | mean_loss > 0, held-out Δ vs envelope |
| P2 | **Mega-rollout width** (>1 group concurrently; GPU busy-frac GO already measured, wins/2026-07-24) | 2 | scheduler-level | rollout wall/group ↓ at equal pass-rate |
| P2 | **Eval amortization**: baseline envelope cached per (model, config) across runs; eval cadence every other round (eval was 24/56 of the A/B's sample walls) | 2 | script-level | run wall ↓, verdicts unchanged |

Method rule binding all items (from the stagger KILL): **before building, read
the baseline's existing counter for the waste you believe exists.** P0 and P1
both have their observables already emitted (`lack a token sidecar` warnings,
`SKIP … seq` lines); P2 items get one profiling read before any code.
