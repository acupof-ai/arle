# The RL systems substrate: a correspondence invariant, then an economics ratio

Research + planning insight, 2026-07-24 (adversarially reviewed same day: 3
independent attack passes; every surviving attack is folded in below). Sources:
THUDM/slime 358-issue landscape (external map of where the mainstream
Megatron+SGLang RL stack bleeds) crossed with ARLE's own decoded incidents (two
null-gradient episodes, the group-stagger KILL, the 5/28 sidecar miss).
Explanation doc; the executable plan is
[plans/2026-07-24-sweetspot-corpus-first-gradient.md](../plans/2026-07-24-sweetspot-corpus-first-gradient.md).

## Scope

These axioms bound the **systems substrate** only. Algorithmic validity —
reward hacking, advantage-estimator bias, entropy/KL collapse, exploration —
is a third, independent axis this doc does not cover; a substrate can satisfy
both axioms perfectly while training a worse model, and axiom 2's ratio would
even score hacked rollouts as productive. Substrate health is necessary, never
sufficient.

## Thesis — two axioms

**Axiom 1 (correctness): the training side must see, token-exact, what the
behavior policy actually generated.** In all four of our decoded incidents the
silent break traced to state with more than one owner. That is an observed
pattern, not a law — a single-owner serializer bug would break correspondence
too — but it is where the evidence to date points, and multi-owner boundaries
are where we post guards first.

**Axiom 2 (economics): the system's real output is gradient-bearing tokens per
GPU-hour** — not tok/s. Rollouts discarded for length, zero-variance groups,
and saturated corpora all multiply throughput by zero.

**Ordering is conditional, not absolute.** "Fix axiom 1 first" holds when the
correspondence breaks are *selection-biased* (e.g. misses correlating with
long/failing trajectories — exactly the high-signal tail); an unbiased break
is only an axiom-2 sample-size cost. Whether our sidecar misses are biased is
the open question P0 exists to answer — until then the ordering is a working
default, not a settled fact.

## Evidence

### External — slime's four load-bearing clusters (hypothesis-grade)

| Cluster | Closure rate | Shared root |
|-|-|-|
| Weight-refresh chain (Megatron→HF→SGLang) | 39% | weights owned by 3 abstractions |
| OOM / memory lifecycle | 24% | VRAM split across 3 CUDA contexts, no ledger |
| Long-seq + context parallel | 16% | correctness state sharded, no single-box repro |
| Agentic rollout (loss-mask misalign, multi-turn advantage) | — | forest trajectories forced into a matrix abstraction |

The closure-rate diagonal (fp8 89% ↔ longseq_cp 16%) is consistent with
"closure tracks single-machine reproducibility" but is an **uncontrolled
correlation** from a multi-team OSS tracker (maintainer priority, user count,
and difficulty are all unmodeled confounders) — treat it as a source-survey
hypothesis, per §0, not evidence.

### Internal — the same axioms, decoded in ARLE

- **Null gradient ×2** (errors/2026-07-03 corpus saturation; errors/2026-07-22
  length-wall skips): axiom-2 violations that each cost a debug cycle because
  they present as "training runs fine, loss = 0".
- **Sidecar miss, 5 of 28 requests in one A/B window** (count from the
  uncommitted stagger-A/B run logs; dumps preserved on the pod, 446+432
  files): the streaming path writes the token sidecar only on engine
  `delta.finish` (coordinator.rs:1221; the non-streaming branch at :1271
  always writes one, but the CC harness streams); the `delta.error`,
  client-disconnect, and stream-error break paths all leave a dump without a
  sidecar, and `cc_convert` skips those requests (cc_convert.rs:172).
  **Bounded severity**: a skipped *mid-turn* request loses only that turn's
  mask-1 supervision — its content recurs mask-0 in later requests' resent
  prompts (cc_convert.rs:152) — while a *session-final* miss loses content
  and a retry-duplicate miss loses nothing. Attribution (biased loss vs
  benign dedup) is the roadmap's step 3.
- **Group-stagger KILL** (errors/2026-07-24): a premise inferred from code
  reading, falsified by one baseline counter read — the method-level lesson
  (measure the baseline observable first) applies to every roadmap item below.

## ARLE's structural position

Single-process train-infer unification gives axiom 1 by construction inside
the process boundary: weight update is a LoRA-merge + `invalidate_prefix_cache`
in one control closure (serve_engine.rs:373), the student borrows the rollout
engine's frozen base zero-copy (`--share-frozen-base`), and VRAM is explicitly
modeled (`ckpt-gate modeled=…B`). Two caveats keep this honest:

- The clean record on slime's clusters is partly **structural non-exposure**
  (a single-box lane cannot exhibit context-parallel bugs), not demonstrated
  immunity — our own multiproc socket-timeout teardown at TP=N c8 was exactly
  a bug class invisible to single-box repro.
- Axiom 1 is NOT yet constructive at the two file-based boundaries —
  serve↔harness (dumps + sidecars, the 5/28) and harness↔trainer (window
  matching) — which is precisely where our cracks appear.

Standing design defaults derived from the map (scoped, not absolute):

1. **No tri-state memory handoff** (slime's torch_memory_saver: 8 crashes in
   its latest 60 issues). Any "train releases VRAM → infer takes over → hand
   back" proposal is rejected on this evidence.
2. **No new training-side feature whose correctness requires >1-box repro.**
   The existing multi-GPU inference lane (DSv4/GLM TP=8/EP=8) is exempt and
   keeps its own verification routing; this veto governs the OPD loop, where
   our answer to 30K-token trajectories is filter + reject (single-box
   verifiable), not repairing the 30K backward.
3. **One first-class lane** (ThinkingCap-27B agent-OPD). slime's GLM-vs-Qwen
   response-time gap (6 h vs 34 h) shows what pretending otherwise costs.

## Roadmap (reordered by the review; each step feeds the next)

| # | Item | Axiom | Cost | Acceptance |
|-|-|-|-|-|
| 1 | **Sidecar drop-reason counter**: tag the three streaming break paths (error / disconnect / stream-closed) with a per-reason counter, ships with the next binary | 1 | ~10 LOC | every future miss self-attributes |
| 2 | **Sweet-spot corpus → first real gradient** (existing plan, #173) — the same pod run produces the counter's first live data | 2 | ~1 day pod | mean_loss > 0, held-out Δ vs envelope |
| 3 | **Offline miss attribution**: classify the preserved stagger-A/B misses (pod read-only) + step 2's counter data; decide biased-loss vs benign-dedup | 1 | pod read-only, 0 GPU | each miss classified; bias verdict recorded |
| 4 | **Conditional — length early-stop, serve-side**: only if step 2's banded run still shows `SKIP … > max_update_seq` lines (the corpus filter may already absorb them); implement as a per-session token cap in the serve that rejects over-limit requests — NOT harness-side dump-watching, which would rebuild the machinery the stagger revert just deleted | 2 | small, serve-only | zero residual SKIP GPU-seconds |
| 5 | **Mega-rollout width** (>1 group concurrently; GPU busy-frac 0.30–0.34 GO, wins/2026-07-24) | 2 | scheduler-level | rollout wall/group ↓ at equal pass-rate |
| 6 | **Eval cost**: cadence every other round; envelope reuse only *within* a run keyed (binary, config, corpus) and invalidated on rebuild — cross-run caching is rejected (it re-imports the cross-day drift the matched-A/B rules exist to exclude); keep a 1-seed cross-run re-anchor as a drift tripwire | 2 | script-level | run wall ↓; anchor stays in band |

Method rule binding all items (from the stagger KILL): **before building, read
the baseline's existing counter for the waste you believe exists.** Steps 3
and 4 are explicitly gated on observables steps 1–2 produce.
