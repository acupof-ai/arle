# Codex task queue — pickup after 2026-05-31 21:20 rate-limit lifts

**Context**: codex was rate-limited 2026-05-28; this Claude /loop session
has produced 12 commits + 5 research/errors docs while codex was offline.
Below is a pre-scoped queue codex can work through in parallel when
quota returns. Each task is self-contained, has explicit kill criteria,
and cites file:line.

Order is **ROI-ranked**; not strict dependencies. Codex may pick any in
any order. Each task should land as its own commit (`<type>(<scope>):
<subject>`, no `Co-Authored-By: Claude` footer).

---

## Task A — Microbench validating the student_rollout 1.44 s/tok claim

**Scope**: small new bench, no train-crate changes.

**Goal**: confirm or refute the perf hypothesis in
[`docs/research/2026-05-28-opd-rollout-perf-208s-bottleneck.md`](../research/2026-05-28-opd-rollout-perf-208s-bottleneck.md).
The doc's claim: train-crate `forward_rollout_cached_device_token` is
~16× slower per token than batched `student_forward`, with the gap
caused by host-side per-op autograd-tape bookkeeping over ~300 ops per
token × 144 tokens.

**Deliverable**: `crates/train/examples/opd_rollout_token_bench.rs`
that:

1. Loads Qwen3.5-0.8B-Base (with or without LoRA — flag-controlled)
   via the same path the OPD train uses.
2. Runs warm-up: 5 prompts of `prompt_max_tokens=16`,
   `rollout_len=64`, dropping into the same code path
   `opd.rs:1654-1722` (the `use_device_rollout_argmax` branch).
3. Measures per-token latency across N=10 prompts at rollout_len=64
   and N=10 at rollout_len=128. Reports:
   - mean / median / p50 / p99 per-token latency
   - per-phase breakdown using the existing
     `Qwen35RolloutForwardProfile` machinery (cache_select,
     embedding, layers, final_norm, lm_head)
   - per-layer min/max/mean to spot outlier layers
4. Same harness with `tape.set_enabled(true)` for one control run
   to confirm the "tape disabled" claim isn't accidentally still
   recording.

**Acceptance**:
- Bench runs in ≤10 min wall-clock on the 4070 Ti SUPER at
  `--mem-fraction-static 0.30` (~6 GB peak).
- Output saved to `bench-output/2026-MM-DD-opd-rollout-token-bench/`.
- Compare numbers against the v4 run.txt 1.44 s/tok point estimate.

**Kill criteria**:
- Pass: bench reproduces ≥1.0 s/tok at rollout_len=128, with per-layer
  profile showing layer-uniform cost (no single outlier > 2× the
  median).
- Kill (reduces task ROI): bench measures < 0.3 s/tok — the v4 run's
  208 s/step came from something other than the per-token forward
  (e.g., the retain bookkeeping at `opd.rs:1755`). Investigate that
  next instead.

---

## Task B — Identify the dominant host-side cost inside the per-token forward

**Depends on**: Task A confirming the perf bottleneck.

**Scope**: nsys / cuda-events instrumentation of one rollout step.

**Goal**: pin the host-side cost to a specific op or call. Hypotheses
from this session's investigation:

1. **`TensorStore` per-op overhead** — `store.alloc_device_tensor()` +
   `store.get()` × ~300 ops per token. Hash lookups, ID allocation,
   metadata cloning.
2. **`select_cache_rows(cos_cache, &[position])` host→device upload**
   per token × 2 (cos and sin). Could batch up-front for the whole
   rollout.
3. **Lazy graph build dispatch** — backend_cuda.rs comments say ops
   are lazy. If host-side graph construction is per-op slow, that
   compounds.
4. **`retain_rollout_step_tensors` at `opd.rs:1705`** — O(n) keep-set
   each invocation, called every 2 tokens.

**Deliverable**: appended to
`docs/research/2026-05-28-opd-rollout-perf-208s-bottleneck.md` with
nsys / cuda-events numbers identifying the dominant cost. One
sentence per hypothesis with the actual measured % of per-token
time.

**Acceptance**: top hypothesis named with evidence; the other three
either ruled in (with %) or ruled out.

**Kill criteria**:
- Pass: one hypothesis accounts for ≥40% of per-token time, with
  evidence.
- Kill: no single hypothesis > 25% (cost is diffuse), pivot to
  Task C directly.

---

## Task C — Quick-win micro-optimizations (pre-refactor)

**Scope**: ≤3 files, no architectural change.

**Goal**: chase the ROI tail before the big refactor. Candidates:

1. **Pre-compute cos/sin for the entire rollout window once** at the
   start of the rollout loop, then slice per-token instead of
   re-uploading `&[position]` each call. Touches `opd.rs:1648-1722`
   and possibly `qwen35.rs:1933-1934`. Estimated ROI: 2 × cache_select
   per token saved. ~10-20 ms × 144 = 1.4-3 s per step if cache_select
   is hot.
2. **Lift `rollout_keep_base.clone()` out of the
   `retain_rollout_step_tensors` per-step loop** (`opd.rs:1755`).
   Currently the base set is cloned each retain; can be mutated in
   place if the retain logic allows.
3. **Reduce the retain frequency for non-final steps** — currently
   `ROLLOUT_RETAIN_INTERVAL=2` (`opd.rs:413`). If most retained
   tensors are unreachable from the final backward, every-4 might be
   safe.

**Deliverable**: one commit per candidate landed, each with a before/
after rollout=128 timing measurement (use Task A's bench).

**Acceptance**: cumulative ≥20% step-time reduction at rollout=128.

**Kill criteria**:
- Pass: net step time ≤ 250 s at rollout=128 (vs 310 s today).
- Kill per-candidate: < 5% reduction → revert.

---

## Task D — Audit other train-crate forward paths for the same overhead

**Scope**: read-only audit of `crates/train/src/qwen35.rs`.

**Goal**: are the FULL-SEQUENCE forwards (`forward_batch_indices`,
`forward_batch_hidden_indices` at lines 1976-2030) doing the same
host-side bookkeeping per token? They're called for student_forward
(87 ms/tok = 12.5 s for 143 tok). If the host-side cost is uniform,
the batched cost should be similar — but it's not. Either:

- Batched path has lower per-op cost (good — that's the inspiration
  for the per-token fix)
- Or per-op cost dominates but is amortized by batching (suggesting
  the per-token path needs to batch what it currently doesn't)

**Deliverable**: appendix to the perf research doc with a comparison
table. ~100 lines.

**Acceptance**: clear answer on whether per-token forward CAN be made
batched-fast without crossing the train↔infer boundary.

**Kill criteria**:
- Pass: doc commits with conclusion.
- (No kill — pure-research task.)

---

## Task E — Approach-first design doc for the rollout↔infer hand-off

**Depends on**: Task A confirms bottleneck AND Task C exhausts
quick wins AND Task D rules out a pure train-crate fix.

**Scope**: design doc + minimal-scaffolding commit. NO functional
refactor without user approval.

**Goal**: scope the work to route OPD's student rollout through the
infer engine's CUDA-graph decode + paged KV. Crosses `crates/train` ↔
`infer/src/backend/cuda/*`, > 5 files, so CLAUDE.md approach-first
applies.

**Deliverable**: `docs/projects/2026-05-XX-opd-rollout-through-infer.md`
covering:
- Lifecycle: where infer engine is constructed, where LoRA in-memory
  mirror lives, who owns the student weights between rollout and
  backward.
- Numerical-parity test plan (bit-identity vs train-crate forward at
  the same seed, ≥95% token-match gate).
- Step-time projection at rollout=128 and 256 with kill criteria.
- Implementation order: 5-10 atomic commits each landable in
  isolation.

**Acceptance**: approach doc committed; **no code changes** in this
task. User signoff required before any actual refactor.

---

## Side tasks (parallel, low-priority)

### Task F — extractor improvement for MMLU invalid responses

The `arle_capability_eval.py` MMLU extractor fails on 18% of responses
(30/171 per seed). The tick-6 patch adds `mmlu_invalid.json` dumping
EVERY invalid response (commit `e68aa26c`). After running the next
multi-seed eval (whenever that happens), the
`runs/.../seed_N/mmlu_invalid.json` files will give empirical failure
modes. Patch `_mmlu_extract_letter` at
`scripts/arle_capability_eval.py:195` with new layers covering those
modes. Kill criterion: bring n_scored / n_samples from 145/171 to
≥160/171 on the same prompts.

### Task G — kv-tier audit follow-ups not landed in tick 1

From `docs/research/2026-05-28-kv-tier-current-state.md` `(c) Code-quality
audit`, two deletion candidates were deferred for approach-first review:

1. `LocalCudaTransport` (kv_tier/transport/local_cuda.rs) — dead skeleton,
   no consumers, AGENTS.md:49 documents it as "future P0' NVLink peer hop".
   Removal touches 4 files. Worth deciding: keep as design-frozen skeleton
   or delete and document the gap.
2. `impl KvTierAdapter for TieredKvPolicy` (scheduler/cuda/policy.rs:70-82)
   — dead trait impl on CUDA (only Metal consumes the trait). Removal
   touches 3 files.

Both need user-facing trade-off explanation before deletion.

### Task H — add session_id support to `scripts/bench_guidellm.sh`

The kv-tier audit's primary "unlock T1 on default workload"
recommendation. Today guidellm has no native session_id field; either
wrap it with a small Python preprocessor that injects session_id into
each request body, or write a separate Rust bench tool. Scope this
before committing — could be 20 LOC wrapper or 500 LOC new bench.

---

## What this session already shipped (don't redo)

- 5-seed paired verdict: OPD null effect on MMLU & GSM8K
  (`docs/experience/errors/2026-05-28-mmlu-cross-base-was-noise.md`).
- `--seed` knob in arle_capability_eval.py (commit `c55db536`).
- Multi-seed driver `eval_opd_ckpt_seeds.sh` (`60c146dd` + `7c592054`).
- Paired analyzer `analyze_multi_seed.py` with Wilson CI, paired delta,
  t-test (`b17061e0` + `3bae5cfc`).
- kv-tier audit (`f82415a5` deletion + `0e702f17` audit doc).
- Mac CUDA-Rust typecheck fix (`f952834d`).
- mmlu_invalid.json dump (`e68aa26c`).
- Perf-axis research doc (`6fce6562`).
