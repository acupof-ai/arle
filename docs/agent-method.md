# ARLE — working method (on-demand)

Loaded from [`AGENTS.md`](../AGENTS.md) when a task needs evidence discipline,
decomposition, or a verdict. Not needed for routine edits.

---

## SOLID — evidence, not inference

**Not SOLID enough → keep digging.** The quality bar, not a suggestion.

- **Inference ≠ evidence.** Source survey / grep / doc / callgraph are
  *hypothesis*; evidence = measured nsys / bench numbers / runtime log counter /
  controlled-variable A/B. No evidence → label it hypothesis, no conclusion.
- **Isolate confounders.** One experiment changing N variables at once (buffer
  pool + scheduler clamp + KV format + graph capture) is **unattributable**.
- **Root-cause hypotheses get verify-or-reject too** — not just fixes. Wrong root
  cause → every sub-experiment wasted.
- **80% SOLID is not enough.** Dig to 95%+, or explicitly declare "deferred,
  accepting the uncertainty". No silent pass.
- **Wall-clock / per-request framing is ground truth.** A narrow-window X% share
  ≠ the actual wall-clock impact; acceptance uses wall-clock.
- **Case-as-fact.** A negative result (regression, failed metric, a subagent's
  "it's structural") is a case to debug at the token level, NOT a license to
  generalize into a KILL. ① decode the actual model outputs on the *failing*
  slice; ② audit the eval harness for artifacts (timeouts bucketed as a class,
  request-errors counted wrong, a metric rewarding the wrong thing). Aggregate
  **and** mechanism can both lie. Attribute first, overturn second.

Anchors:
- **Agentic-OPD false-KILL** (2026-06-20): a −14pp "structural KILL" was wrong —
  its "+42pp teacher abstention" gate was 14/17 teacher TIMEOUTS miscounted.
- **M_pf-graph Phase 0** (2026-05-08): launch share not nsys-verified, trigger
  count uncontrolled, 4 variables at once → verdict void.
- **M_pf-graph v2** (2026-05-08): nsys 55.7% of the prefill window = **0.32%
  wall-clock**.

---

## Decompose to the implementation level

Atomic tasks → dependency DAG → critical path. Fine-grained enough, the plan
falls out on its own.

- **Implementation level, not principle level.** "Pre-allocate, don't copy big
  buffers" is a principle, not a spec — name the exact buffer / size / call site
  / precondition. Claude produces the line-level spec, the executor copies it
  verbatim; free improvisation drops fields.
- **Every state change enumerates each mutated buffer and proves each.** Rollback
  / cache / scratch / fusion / quant: list **every** device buffer written, give
  each a disposition + exact precondition — never "should self-heal":
  ① rolled back by a named existing path; ② self-heals (write the precondition —
  a ring's speculative write self-heals **only** for seq_len < ring_size);
  ③ snapshot/restore.
- **Inline speed into correctness.** Pre-allocate once and reuse; copy at the
  smallest grain (ring moves one slot → store that slot, not the ring); fold into
  the opt-in path with the default baseline byte-for-byte unchanged.
- **Correct inference ≠ baseline identity.** The gate for spec-decode / quant /
  kernel-swap is correct inference (needle retrieval + same-config-twice
  non-determinism floor + self-consistency), NOT token-exact-vs-baseline (MoE
  non-determinism). A degenerate (looping) prompt is not a valid test case.
- **Root-cause on a clean baseline.** Isolate the precondition first.

Anchor — **DSv4 EAGLE rollback** (2026-06-06): `truncate_decode_len` restored
only `compressed.seq_len`, missing `pending_kv`/`prev_overlap`; full enumeration
then exposed the missing `sw_window` + `fp8_kv_pool` ring slots.

**Gate before any implementation code** (skill
[`understand-until-simple`](../.claude/skills/understand-until-simple/SKILL.md)):
decompose to a concrete file:line / kernel / loop → get measured evidence → let
measurement correct the hypothesis → state the fix in **one sentence + one
measured number** and name the next wall. Still saying "hard / multi-day" means
not yet decomposed — "hard" is a confession, not a property.

**Cost comes AFTER the concrete work, never before** (ckl 2026-06-20). A cost
guessed ahead of decomposition biases the work down and pre-commits a wrong size.
Empirical: DSv4 batched decode was hand-waved "very hard, multi-day new infra" —
reading the code collapsed it to "one per-row `for` at `dsv4.rs:1872` → batch it
→ ~2×".

---

## Recurring lessons (each distilled from ≥3 experience entries)

- **SLO verdict from the SLO workload, not a smoke shape** — a c=1 short-prompt
  nsys "2× win" routinely flips on production prompt length. The workload is the
  multi-turn long-agent dataset at the TraceLab medians (bench spec §3.3):
  119K prefix, 875 append, 214 output, 95.7% prefix hit. Anything shorter, or
  anything that cannot hit the prefix cache, is a smoke
  shape, whatever it measures.
- **`plan_label=mixed` is reachability, not acceptance** — a c-sweep must clear
  TTFT *and* ITL before a default flip, on ≥2 binding
  production shapes.
- **A/B: same-binary, same-shell, same-prompt, two env flips, side-by-side** —
  cross-day claims drift backend / KV dtype / scheduler tuning.
- **Smoke-output garbage is config-suspect first** — A/B the prod backend on the
  same config before staring at new code.
- **Launch-count source-survey is hypothesis** — a fused-kernel rewrite for tiny
  CUDA ops is accepted only by a paired component A/B under the runtime's sync
  framing.
- **Capability claims <5pp on small-n (≤200) evals need multi-seed (≥5) + mean±σ
  + Wilson 95% CI** — "best ckpt across save-every-10" is positively biased.
- **Pod-side probe trust is conditional on git+symbol checks** — the pod tree is
  a git repo at HEAD *and* `strings target/release/arle | grep <symbol>`; the
  binary proves some tree was current whenever last built.
- **Greedy-decode the actual generation when a metric looks catastrophic** — 3
  weeks of "FP8 KV broken" collapsed when one `eprintln!` showed a test-framework
  artifact.
- **Cross-rank state that feeds a collective or a shared budget must be aligned
  explicitly** — a rank-local fallback (prefix-attach failure, sidecar miss,
  budget solve) degrades one rank while peers proceed; without an align step the
  next collective desyncs (garble or hang). Align once per admission, off the
  hot path: min-reduce for lengths (prefix matched + restored), divide-by-world
  for budgets (KV tier), lockstep for shapes (`seq_len` — under CP only page
  ownership may diverge, never the lengths themselves). Three hits in one week:
  tier-budget-not-divided-by-world, prefix-match-no-cross-rank-min-reduce,
  prefix-restore-min-reduce.

## Execution hygiene (Claude and delegated agents alike)

- Surface known failure logs upfront so the same blocker isn't re-discovered.
- Pin SKU / shape / scope at exact granularity, not by fuzzy name.
- Before patching an upstream component, grep the raise point and lock the root
  cause; verify the patched lib in an isolated dir, never the dev install.
- Probing install / directory / env layout: enumerate candidate paths upfront,
  not fail-then-retry.
- Upstream patch crossing a size or cross-cutting-policy threshold → pause + ack.
- Regression tests mirror the failure mode with a minimal in-component kernel,
  not by importing caller code.
