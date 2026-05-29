# GPU Dispatch Governance — Improvement Plan

> Diagnosis: [`../reviews/2026-05-29-gpu-dispatch-governance-analysis.md`](../reviews/2026-05-29-gpu-dispatch-governance-analysis.md).
> Status: **approach-first artifact, awaiting sign-off.** This is an architectural,
> cross-cutting change (>5 files); per root `AGENTS.md` no runtime code lands until
> the approach is accepted. Each phase below is independently shippable and
> independently revertible.

## The principle

Make the execution path a **first-class artifact** across four mechanically-enforced
gates. Each gate closes one symptom and generalizes a primitive that already exists
(§4 of the analysis), so the net change is *convergence and deletion*, not new layers.

```
DECLARE  →  one resolver answers "what will run on this GPU"      ▶ 链路不清晰
OBSERVE  →  every dispatch + every fallback reports what fired    ▶ 链路不通 (detect)
ASSERT   →  tests & benches fail if the path didn't fire / fell back ▶ 链路不通 (prevent)
GOVERN   →  a registry tracks chosen-vs-best-vs-roofline per op    ▶ 算子不是最好的
```

Sequencing is cheapest-highest-leverage first. Observe before Assert (you cannot
assert a counter that does not exist). Declare can proceed in parallel. Govern is last
because it depends on the counters Observe produces.

---

## Phase 1 — OBSERVE (highest leverage, lowest cost, ship first)

**Problem.** Resolved dispatch decisions are computed and discarded (`LinearKernelPlan`
never logged; `StepPlan` collapses `SpecDecode`→`Decode`); silent fallbacks demote at
`trace` level (`linear.rs:139,169`) or with no log at all (graph→eager
`batch_decode.rs:1004-1024`, FP8→BF16 `main.rs:2025-2029`, Metal C++→Rust
`qwen35/qwen35.rs:2784`). Theme 3 (path never fired) and the ~17-commit c=4 miss are
direct consequences.

**Intervention.**
1. **Kernel-fired counters.** Generalize `MIXED_BATCH_FALLBACK_REASONS`
   (`metrics.rs:136-150`) into a `dispatch_kernel_total{op,variant}` counter family on
   `ServerMetrics`. One increment at each existing launch site, keyed by the
   already-resolved `LinearKernelPlan` variant / `KVFormat` / graph-vs-eager / attention
   kernel id. Render on `/v1/stats` next to `plan_label`.
2. **Loud fallbacks.** Add `dispatch_fallback_total{from,to,reason}`. Every silent
   demotion (the five sites above) increments it and emits one `log::warn!`. A fallback
   is now a counter you can gate on, not a `trace` line nobody sees.
3. **Fix the blind spot.** Split `SchedulerPlanLabel::Decode` so `SpecDecode` is
   distinguishable (`execution.rs:95-103`).

**Builds on:** `ServerMetrics`, `MIXED_BATCH_FALLBACK_REASONS`, `/v1/stats` renderer
(`metrics/render.rs`). **Files:** `metrics.rs`, `metrics/render.rs`, launch sites in
`ops/linear.rs` + `model/qwen35/batch_decode.rs` (+ Metal `backend/metal/ops.rs`).
**Acceptance:** `/v1/stats` of any guidellm run shows non-zero
`dispatch_kernel_total` for every kernel the workload exercised, and
`dispatch_fallback_total == 0` on the happy path. **Cost:** ~1 counter family + N
one-line increments; no hot-path branch added (the resolve already happens).
**License-or-kill:** counters must add < 0.5% to per-token wall-clock (atomic increment
on an already-taken branch — expected ≈0; verify with one before/after guidellm c=1
and c=16 run, kill if regression > 0.5%).

---

## Phase 2 — DECLARE (parallel with Phase 1)

**Problem.** No artifact answers "for `(model, shape, batch, quant, features, SKU)`,
what kernel chain runs." The answer is spread across `StepPlan`, `LinearKernelPlan`, a
`KVFormat` match in three files, the graph heuristic, and inline `env::var` reads
(`INFER_MARLIN_W4_FP8_PREFILL`, `INFER_R4_W4A16_GEMV_OVERRIDE`,
`INFER_HYBRID_W4A8_PREFILL_ENABLED`, `INFER_BYPASS_TILELANG_PREFILL`). No SM-tier
resolver exists.

**Intervention.**
1. **`DispatchPolicy` struct, resolved once at bootstrap.** Collect every scattered
   `env::var` dispatch knob into one typed struct built in
   `backend/cuda/bootstrap.rs`. Hot-path code reads fields, never `env`. This is a pure
   deletion-style refactor — same behavior, one home. (Honors
   `feedback_use_industry_env_vars`: rename ad-hoc `INFER_*` knobs only where an
   ecosystem standard exists; otherwise keep names, just centralize.)
2. **`ExecutionPlan` descriptor.** A read-only struct assembled per step from the
   existing resolvers — `{scheduler_plan, kv_format, attn_kernel, per-layer
   linear-kernel class, graph|eager, spec_path}`. Reuses `LinearKernelPlan` /
   `StepPlan` outputs; adds no new decision logic.
3. **`arle explain-dispatch --model … --shape … --quant …` introspection.** Prints the
   resolved `ExecutionPlan` for a hypothetical request *without serving it*, plus the
   SKU capability check (which TileLang head-configs are AOT-compiled for this SM, and
   whether the requested shape hard-fails at `attention.rs:1477`). This is the literal
   answer to "每个 GPU 走的链路不清晰", available before bench day.

**Builds on:** `LinearKernelPlan`, `StepPlan`, the envelope log
(`bootstrap.rs:538-551`). **Files:** new `infer/src/dispatch_policy.rs`,
`bootstrap.rs`, `ops/linear.rs` (read policy field instead of `env`), a CLI subcommand.
**Acceptance:** `grep -rn 'env::var("INFER_' infer/src/ops infer/src/model` returns
zero hot-path reads (all moved to `DispatchPolicy`); `arle explain-dispatch` output for
the canonical Qwen3.6 + DSv4 shapes matches the kernels Phase-1 counters report at
runtime. **Cost:** refactor, no new behavior. **License-or-kill:** if centralizing the
`env` reads changes *any* dispatch decision (verify by diffing Phase-1 counters
before/after on the canonical 4-shape set), the refactor introduced a bug — revert and
isolate.

---

## Phase 3 — ASSERT (the gate that stops "链路不通" at the door)

**Problem.** Nothing fails when an implemented path doesn't fire. Theme 1 (reachability
≠ license), Theme 3, Theme 4 (perf on garbage), Theme 5 (substrate misaligned) all land
because the bench is the *first* check, and it runs last.

**Intervention.**
1. **Path-fired unit gate.** A new path lands with a test that asserts its Phase-1
   counter increments under a representative request — the corpus's single most-reached-for
   missing tool (`feedback_path_probe_before_perf_claim`). Generalizes the ad-hoc Metal
   `Once` probes into a real assertion. Add to `infer/tests/`.
2. **Bench harness gate** in `scripts/bench_guidellm.sh`: a `--expect-kernel <variant>`
   flag that, post-run, reads `/v1/stats` and **fails the run** unless
   `dispatch_kernel_total{variant} > 0` **and** `dispatch_fallback_total == 0` for the
   path under test. "链路不通" becomes a red bench, not a 0-delta surprise three weeks in.
3. **Harness precondition checks** (codify the Theme-5 lessons mechanically): the bench
   script refuses to run unless the served binary's commit is clean and non-dirty (§4 of
   bench-spec already wants this) and — for pod runs — `strings target/release/infer |
   grep <expected_symbol>` passes (`errors/2026-05-28-dsv4-flashmla-decode-parity-precond-fail.md`).
4. **Correctness-before-perf gate.** Wire the existing KV-precision-parity harness
   (`cargo test … kv_precision_parity`) and a decoded-token sanity print into the
   "new quant/kernel path" checklist so a perf number can never be recorded on garbage
   output (`errors/2026-05-08-w4a8-quantize-broken-100pct-token-diff.md`,
   `errors/2026-05-26-fp8-kv-catastrophic-was-test-artifact.md`).

**Builds on:** Phase-1 counters, `bench_guidellm.sh`, the KV-parity test, bench-spec §3.
**Files:** `infer/tests/dispatch_reachability.rs` (new), `scripts/bench_guidellm.sh`,
bench template. **Acceptance:** a deliberately-misrouted path (e.g. flag left OFF)
makes both the unit gate and `--expect-kernel` bench fail; the happy path passes.
**Cost:** test + script logic, no runtime change. **License-or-kill:** if the gate
produces false positives (fails a path that *did* fire), the counter granularity from
Phase 1 is wrong — fix the counter, do not weaken the gate.

---

## Phase 4 — GOVERN (operator optimality becomes a visible, owned gap)

**Problem.** The dispatcher resolves *a* kernel, never the *best*. Suboptimal hot-path
kernels (quantized GEMV scalar-FMA ~16% decode; MoE routing 255/256 threads idle ~4.4%;
FP8 fused-attention scalar softmax) are documented only in one-off audits, consulted by
nobody, tracked nowhere. Theme 2 (launch-count survey licenses dead-end fusions) is the
flip side: no roofline context, so "fewer launches" keeps looking like a win.

**Intervention.**
1. **Checked-in kernel registry** — `docs/reviews/kernel-registry.md` (living successor
   to the 2026-04-14 six-principles review). One row per `(op, shape-class, SKU, quant)`:
   chosen `LinearKernelPlan`/attention variant · impl type (hand-rolled / TileLang /
   library) · measured roofline position · **best-known alternative + why it isn't
   wired** + owner. Makes "best kernel exists but isn't wired" a tracked item, not a
   silent default.
2. **Roofline-share gate for kernel work** (mechanize Theme 1 + 2): before any
   operator-optimization commit, the wins entry must state the op's measured **% of
   per-request wall-clock** at the binding SLO shape. < ~5% share → not licensed for
   kernel work regardless of any narrow-window or launch-count argument. This is the
   §0 framing rule (`M_pf-graph v2 framing trap`) turned into a required field.
3. **Autotune coverage as a registry column.** Today only `Bf16Gemv` is tuned (opt-in).
   The registry tracks which variants are hardcoded vs tuned, surfacing the gap so it is
   prioritized deliberately, not rediscovered each audit.

**Builds on:** the six-principles review, `kernel-vs-sota` audit, Phase-1 counters
(which now tell you which variants actually run, so the registry tracks *live* paths,
not theoretical ones). **Files:** `docs/reviews/kernel-registry.md` (new), bench
template (add the roofline-share field). **Acceptance:** every hot-path variant that
Phase-1 counters show firing has a registry row with a roofline number; the three known
P0 gaps have owners. **Cost:** documentation + one bench-template field.
**License-or-kill:** the registry is killed if it drifts stale — it must be a required
update in the verify-phase exit, or it becomes another forgotten audit doc.

---

## Process mechanization (the meta-fix)

The corpus proves the rules are known and unfollowed. Three changes move enforcement
out of human memory:

1. **Verify-phase exit condition gains a clause** (root `AGENTS.md` §Benchmarks):
   a runtime change isn't done until its wins entry cites (a) the path-fired counter
   value and (b) `dispatch_fallback_total == 0`. The bench template adds these fields so
   the check is mechanical.
2. **CI lint:** a diff that adds a `LinearKernelPlan` variant / `KVFormat` arm / new
   dispatch branch but no `dispatch_reachability.rs` assertion → fail. Closes Theme 3 at
   review time.
3. **`feedback_*` memories become tooling, not reminders:** `feedback_path_probe_before_perf_claim`
   → the Phase-1 counter + Phase-3 gate; the wall-clock-at-SLO rule → the Phase-4
   roofline field; paired-single-variable-A/B → the `--expect-kernel` + fallback==0
   bench gate.

---

## Non-goals / explicitly out of scope

- **Not** rewriting any kernel. Govern *surfaces* the gaps; fixing them is separate,
  licensed work gated by the new roofline field.
- **Not** building MLX kernel introspection. The Metal C++/MLX kernel choice stays a
  black box below the FFI line; Observe stops at the Rust dispatch boundary and the
  documented MLX-side selection (`feedback_mlx_async_eval_is_caller_thread`).
- **Not** a new abstraction layer. `ExecutionPlan` is a read-only view assembled from
  existing resolvers; `DispatchPolicy` is centralization of existing `env` reads. If
  either grows decision logic of its own, the refactor failed.
- **Not** changing default dispatch behavior. Phases 1–2 are observability + refactor;
  any default flip remains a separate c-sweep-gated decision
  (`errors/2026-05-25-axis2-mixed-default-kill.md`).

## Risks

- **Counter overhead on the hot path.** Mitigated: increments sit on already-taken
  branches; Phase-1 license-or-kill measures it (kill > 0.5%).
- **Refactor regression in Phase 2.** Mitigated: Phase-1 counters become the diff oracle
  — if centralizing `env` reads changes any counter on the canonical shapes, it's a bug.
- **Registry rot in Phase 4.** Mitigated: tied to the verify-phase exit; a stale row is
  a CI-visible omission, not a silent drift.
- **Scope creep into kernel rewrites.** Mitigated: explicit non-goal; Govern produces
  owned items, not patches.

## ROI

The corpus puts a concrete price on the status quo: ~17 wasted commits on one
unreachable c=4 path, ~40 micro-tune kills in a single day, 3 weeks on a non-bug FP8-KV
investigation. Phases 1–3 are days of work (counters, a refactor, a test file, a bench
flag) and turn each of those classes into a same-day red signal. Phase 4 is documentation
discipline. The payback is the recovered effort, and the second-order win is that every
future bench on the existing roadmap (DSv4, Qwen3.6, quant, spec-decode) starts from
"the path provably fired" instead of "let's hope it did."

---

## Suggested execution order

1. **Phase 1** (Observe) — ship first; everything else depends on the counters.
2. **Phase 2** (Declare) — parallel; the `env`-centralization refactor and `explain-dispatch`.
3. **Phase 3** (Assert) — once counters exist; the unit gate + bench `--expect-kernel`.
4. **Process mechanization** — land alongside Phase 3 (template + CI lint).
5. **Phase 4** (Govern) — registry seeded from Phase-1 live-path data.

Each phase is a small tranche, lands as its own commit, and carries its own bench entry
(Phase 1 counters change the runtime → in-scope per §Benchmarks; Phases 2–4 are
refactor/docs/tooling → state the exemption in the commit body).

**Awaiting your sign-off on the approach before any runtime code lands.**
