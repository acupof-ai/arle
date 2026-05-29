# GPU Dispatch Governance — Root-Cause Analysis

> Companion improvement plan: [`../plans/gpu-dispatch-governance.md`](../plans/gpu-dispatch-governance.md).
> Evidence base: 5 parallel code surveys (CUDA + Metal dispatch chains, observability
> inventory, operator-quality audit, 544-entry experience-corpus mining), 2026-05-29.
> All claims carry `file:line` or a corpus filename; inference is marked *hypothesis*.

## TL;DR

Three reported pains — "做完才发现 GPU 链路不通", "算子不是最好的、性能差",
"每个 GPU 走的链路不清晰" — are **one root cause seen from three angles**:

> **The execution path is an *emergent* property of ~15 scattered,
> compute-and-discard dispatch decisions. It is never *declared*, *surfaced*,
> *asserted*, or *governed* as a first-class artifact. So a path's reachability,
> its operator optimality, and its per-SKU shape are all invisible until bench day —
> after the work is already done.**

The fix is not more kernels and not more discipline-by-memory (the corpus proves
memory-enforcement already fails). It is to make the execution path a first-class
artifact across four mechanically-enforced gates: **Declare → Observe → Assert →
Govern**. The plan doc specifies them; this doc proves why they are needed.

---

## 1. One root cause, three symptoms

| Reported symptom | Governance failure | Why it stays invisible until bench day |
|---|---|---|
| **链路不通** — finished a path, then found it never ran | **Reachability is not *asserted*.** Dispatch decisions are computed and discarded; no test or CI gate proves the new path fires; feature flags default OFF; fallbacks demote silently at `trace` level. | Compiles + unit-tests pass + lands in `main`. None of that exercises the path under load. The bench measures the *old* path; a 0-delta A/B is the only signal, and it arrives last. |
| **算子不是最好的** — operators underperform | **Optimality is not *governed*.** The dispatcher resolves *a* kernel, never the *best* one. Only BF16 GEMM has autotune, and it is opt-in. Known-suboptimal kernels are not tracked anywhere. | The slow kernel is *correct*, so tests are green. Its sub-roofline position only shows up in a one-off profiling pass, then is forgotten until the next audit. |
| **链路不清晰** — per-GPU path unknown | **The path is not *declared*.** No single artifact says, for `(model, shape, batch, quant, features, SKU)`, what kernel chain will run. There is no SM-tier resolver at all. | The answer is spread across `StepPlan`, `LinearKernelPlan`, a `KVFormat` match in three files, a graph-vs-eager heuristic, and inline `std::env::var` reads. Nobody can print "what runs on this GPU." |

All three reduce to: **the path is implicit.** Make it explicit and all three close.

---

## 2. How dispatch actually works today (evidence)

### 2.1 The resolvers exist — but each is local, and each throws its answer away

The runtime *does* have real dispatch resolvers. They are just (a) per-subsystem,
not unified; (b) **compute-and-discard** — the resolved choice is never surfaced;
(c) gated by `env::var` reads buried in the hot path; (d) silent on fallback.

- **Scheduler plan** — `StepPlan` enum, `infer/src/scheduler/cuda/execution.rs:74-82`:
  `Idle | Decode | SpecDecode | Prefill | Split | Mixed`. This is the *only* dispatch
  decision that reaches a counter (§2.3), and even then `metrics_label()`
  (`execution.rs:95-103`) **collapses `SpecDecode` into `Decode`** — so the one
  observable signal cannot distinguish a speculative-decode step from a normal one.

- **Linear / GEMM operator** — `LinearKernelPlan`, `infer/src/ops/linear.rs:64-101`:
  a genuine 30-variant resolver (`Bf16Gemv`, `MarlinW4A8Gemm`, `Dsv4Fp8BatchGemv`,
  `Q4KBatchGemv`, …). It correctly dispatches by weight-format × batch × phase. But:
  - the resolved variant is **never logged or counted** — it is used to launch, then dropped;
  - policy is read inline from the hot path — `INFER_MARLIN_W4_FP8_PREFILL`
    (`linear.rs:95,126`), `INFER_R4_W4A16_GEMV_OVERRIDE` (`linear.rs:158`),
    `INFER_HYBRID_W4A8_PREFILL_ENABLED` (`linear.rs:134`);
  - alignment-fail fallbacks demote via `log::trace!` (`linear.rs:139,169`), which is
    **off at the default log level** — a W4 weight that fails Marlin alignment silently
    runs a different, slower kernel and nothing visible says so.

- **KV format → attention kernel** — no resolver at all; a `match kv_pool.format`
  is duplicated across `infer/src/model/qwen35/batch_decode.rs:1220-1520`,
  `prefill.rs:463`, and `forward.rs:461`. The TileLang HD256 head-config selection
  (`infer/src/ops/attention.rs:1473-1487`) hard-fails on any `(qo,kv)` pair not
  AOT-compiled at build time — `(8,2),(16,2),(16,4)` only.

- **CUDA graph vs eager** — runtime heuristic at `batch_decode.rs:969`; on capture
  failure it **silently falls through to eager** (`batch_decode.rs:1004-1024`) with no
  counter and no warn.

- **KV dtype** — `KVFormat::Auto` resolves to `BF16` for correctness (`main.rs:2104`),
  and an FP8 candidate that fails the envelope check **silently retries as BF16**
  (`main.rs:2025-2029`).

- **Metal** — the Rust side is traceable, but the C++ Qwen3.5 model
  **silently falls back to the Rust path** on load failure (`backend/metal/qwen35.rs:2784`)
  and DFlash **silently disables itself** on config mismatch (`backend/metal/dflash.rs:750-760`).
  Below the FFI line, MLX picks the actual Metal kernel inside `mx::eval` — a black box
  with no hook to report which kernel fired.

**Net:** for one decode step there are ≈15 branch points across ≥6 files. The chosen
path is fully determined but **never assembled into one inspectable answer.**

### 2.2 No SM/SKU tier resolver

The CUDA survey found **zero** runtime SM/compute-capability dispatch (no `sm_`,
`compute_capability`, or arch-based selection in the dispatch chain). Per-SKU
specialization happens only at *build time* via TileLang AOT, and the precompiled
head configs hard-fail at runtime if your SKU's shape was not baked in
(`attention.rs:1477-1486`). "每个 GPU 走的链路不清晰" is literally true: there is no
code path that answers "what runs on this GPU," and the build-time answer fails
closed on an unanticipated shape.

### 2.3 Observability stops at the scheduler tick

- **`plan_label`** (`SchedulerPlanLabel`, `infer/src/metrics.rs:127-134`) is the single
  best "which path ran" signal — but it is 5-way and scheduler-tick-level. It cannot
  see FP8-vs-BF16 attention, TileLang-vs-CUDA-C, graph-vs-eager, or
  tensor-core-vs-scalar GEMV. It answers "what did the scheduler *decide*," not
  "what kernel *ran*."
- **`MIXED_BATCH_FALLBACK_REASONS`** (`metrics.rs:136-150`) is the pattern done right:
  13 named, counted reasons for why a mixed-batch step fell back. **This is the template
  to generalize** — it exists for exactly one path and proves the approach works.
- **CUDA path probes: none.** Metal has three ad-hoc `std::sync::Once` + `log::info!`
  probes (`request_state.rs:2588,2807,917`) added reactively after the misses below.
- **NVTX** ranges exist (`scheduler/cuda/nvtx_scopes.rs`) but fire only under `nsys`.
- **Tests asserting a path ran: none.** `e2e.rs` / `e2e_qwen35.rs` assert output
  correctness only. A kernel can be implemented, tested, landed, and never invoked.
- **Feature flags default OFF:** `spec_enabled`, `spec_sparse_kv_enabled`,
  `t2_disk_tier_enabled` (`infer/src/scheduler/types.rs`). Implement-then-flag-off is a
  structural trap.

### 2.4 Operator optimality is ungoverned

- Only `Bf16Gemv` has shape-aware autotuning, and it is **opt-in** (`INFER_GEMM_AUTOTUNE=1`,
  off by default). Every quantized operator uses **hardcoded launch configs**.
- Documented-suboptimal hot-path kernels with no tracking artifact:
  quantized batch-GEMV is scalar FMA with no tensor cores (**~16% of decode**),
  MoE routing runs `threadIdx.x==0` only — 255/256 threads idle (**~4.4% of decode**),
  FP8 fused-decode attention uses scalar softmax with no `cp.async`
  (`docs/research/2026-05-28-arle-kernel-vs-sota-audit.md`,
  `docs/reviews/2026-04-14-cuda-kernel-six-principles-review.md`).
- These facts live in one-off audit docs, not in any artifact the dispatcher or a
  reviewer consults. "Best kernel exists but isn't wired" is invisible by default.

---

## 3. The corpus quantifies the cost

Mining `docs/experience/{errors,wins}/`: **544 entries across a 29-day window**
(2026-05-01 → 2026-05-29, ~19/day). The same failure themes recur across the *entire*
window — i.e. the rules are already written down and still not followed. The five
ranked root-cause themes map exactly onto the four governance gaps:

| # | Theme | Entries | Maps to gate |
|---|---|---|---|
| 1 | Narrow-window framing ≠ wall-clock; reachability ≠ license | 11 | **Assert** (wall-clock-at-SLO gate) |
| 2 | Launch-count survey licensed a fusion that didn't move wall-clock | 8 (+~40 in the `2026-05-16-p3-*` series) | **Govern** (operator registry + roofline) |
| 3 | Implemented path never fired (hidden seam / missing branch / default-OFF / dead wiring) | 8 | **Observe** + **Assert** |
| 4 | No correctness gate / broken reference → perf measured on garbage | 5 | **Assert** (correctness gate) |
| 5 | Bench/test substrate misaligned (stale pod tree, wrong envelope, default-flip without c-sweep) | 6 | **Assert** (harness preconditions) |

Representative, expensive instances:

- **~17 commits** of a paged-KV plan validated "at c=4 ±5%" — c=4 never dispatched the
  touched path; three consecutive audits missed it
  (`errors/2026-05-07-p3-1c-bench-c4-was-not-on-paged-path.md`,
  `errors/2026-05-07-three-layer-audit-miss-c4-real-path-is-packed-batch.md`).
- **DeepGEMM built, cached, holding 6 GiB** — `forward_native_deepep_routed_gpu` never
  branched to it; 0-delta A/B was the only tell
  (`errors/2026-05-27-b33-deepgemm-not-wired-on-native-deepep.md`).
- **3 weeks of "FP8 KV is catastrophic"** collapsed when one `eprintln!` of decoded
  tokens showed the *reference* was a degenerate loop
  (`errors/2026-05-26-fp8-kv-catastrophic-was-test-artifact.md`).
- **~40 micro-tune kills in one day** (`2026-05-16-p3-*`): launch-shape / row-pointer
  hoist / smem-input / pair-load fusions, almost all KILL with no measured win.

This is the budget the governance layer recovers. The interventions are cheap; the
status-quo waste is not.

---

## 4. What is already right — build on it, don't replace it

The proposal is deletion-friendly: every gate generalizes a primitive that already
exists and works for one case.

- `MIXED_BATCH_FALLBACK_REASONS` (counted fallbacks) → generalize to all fallbacks.
- `SchedulerPlanLabel` / `plan_label` (a path counter on `/v1/stats`) → extend granularity
  from scheduler-tick to kernel-fired.
- `LinearKernelPlan` (a real resolver) → surface its answer; lift the inline `env` reads
  into one typed policy.
- NVTX scopes (annotated ranges) → the offline counterpart of the runtime counters.
- bench-spec §3 already *mandates* internal-source counters in every wins entry — the
  gate just makes that mandate mechanical instead of honor-system.

---

## 5. Root-cause statement

> Dispatch correctness, reachability, optimality, and per-SKU shape are treated as
> things you *discover by benchmarking*, when they should be things the system
> *declares, reports, and asserts by construction*. The runtime already computes the
> right dispatch decisions — it just immediately forgets them, never proves they fire,
> and never records whether the kernel it picked is the best one available.

Memory-enforcement has been tried and has failed at scale: the rules
(`feedback_path_probe_before_perf_claim`, wall-clock-at-SLO, paired-A/B) are all
written down, yet the corpus shows the same five themes recurring across 544 entries
in 29 days. The only durable fix is to move enforcement from human memory into the
type system, the metrics surface, the test suite, and CI. That is the plan.

→ [`../plans/gpu-dispatch-governance.md`](../plans/gpu-dispatch-governance.md)
