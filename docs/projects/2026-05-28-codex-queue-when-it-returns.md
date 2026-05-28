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

## Task A — DONE 2026-05-28 tick 10 — perf scaling characterized

**Skip this**. Claude ran a 2-step rollout-scale sweep across
rollout_len ∈ {8, 16, 32} against the v4 (rollout=128) production
point using the existing `opd_step_cuda_infer_teacher_train` binary
— no new bench needed. Raw data:
`runs/2026-05-28-rollout-scale-bench/run.log`. The findings are in
[`docs/research/2026-05-28-opd-rollout-perf-208s-bottleneck.md`](../research/2026-05-28-opd-rollout-perf-208s-bottleneck.md)
"Microbench update" section.

**Key result**: per-token rollout cost is NOT constant — it scales.
Two-term fit:

```
student_rollout(n_gen) ≈ 0.31 · n_gen + 0.0099 · n_gen²
```

At n=130 the quadratic term is 81% of student_rollout (169 s out of
208 s). Predicted n=256 → 728 s student_rollout/step (matches the
v7-dryrun kill at 19 min/step).

The original "host-side per-op bookkeeping dominates" hypothesis
from tick 7 was wrong at large n. Both terms matter; the quadratic
attention math dominates above n≈30.

Route-through-infer projection revised to ~5× speedup (vs original
70×) — more conservative, accounts for the quadratic term that
infer's flash/paged-attention kernels can only attack the *constant*
of, not the *shape* of.

## Task A' — Pin the quadratic source (was Task B)

**Promoted from Task B because Task A is done; Task A' is the next
diagnostic step.**

The 0.0099·n² coefficient is consistent with several causes:

1. **Attention math + KV cache append** — each decode step's
   `QK^T` and `attn @ V` over the growing cache → O(t) work at step
   t, O(n²) total.
2. ~~`store.retain_ids(&keep)` at `opd.rs:1755`~~ — **RULED OUT**
   2026-05-28 tick 11. `memory_summary live_tensors=370` flat
   across the rollout sweep at rollout ∈ {8, 16, 32}; max-live
   doesn't grow with n. Total retain cost at n=130 ≈ 2.4 ms,
   negligible.
3. **Backward graph buildup** — even with `tape.set_enabled(false)`
   the autograd backend might still walk a growing structure per
   forward.

**Deliverable**: nsys profile OR cuda-events instrumentation on one
rollout step at rollout_len=64 (mid-range — clear quadratic signal,
fast iteration). Report: which of (1)-(3) accounts for what
fraction of the 0.01·n² coefficient.

**Acceptance**: top hypothesis named with evidence; if (1) attention
is >70% of the quadratic term, the only path is route-through-infer
(Task E). If (2) or (3) is >50%, there's a cheaper pure-train-crate
fix worth a separate task.

**Kill criteria**: as before — no single source > 25%, the cost is
diffuse, pivot to Task E.

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

### Task F — REPLACE MMLU letter generation with logprob scoring (was: extractor patch)

**Updated 2026-05-28 tick 9 after empirical data.** Ran a fresh
base seed=5 with the new `mmlu_invalid.json` dump (commit `e68aa26c`)
to characterize the 18% invalid rate. Findings from
`runs/2026-05-28-base-extractor-data/capability_seeds/seed_5/mmlu_invalid.json`
(26 invalids):

- **24/26** are letter-enumeration responses (' ABCD', ' ACD', ' BCD',
  ' ABC', etc.) — the model never commits to a single letter.
- **2/26** are " None of the above" — model commits to "none".
- **0/26** are responses an improved extractor could legitimately
  rescue without introducing false positives.

The extractor at `scripts/arle_capability_eval.py:195` is **already
correct** on these — None is the right return on a genuine
enumeration. Patching to extract the first letter would be a false
positive (picking A from "ABCD" doesn't reflect the model's choice).
Max potential gain from extractor work: ~1pp better n. Not worth it.

**Real fix path** — bigger scope:

1. **Constrained decode**: restrict the completion to A/B/C/D only at
   the HTTP layer. Either add a stop-sequence on whitespace + non-
   letter, or use the existing xgrammar matcher
   (`crates/xgrammar-sys`) to constrain to `(A|B|C|D)`.
2. **Logprob scoring**: compute log P(letter|prompt) for each of A/B/
   C/D directly from the model's first-position output distribution.
   The infer engine already returns logprobs through `/v1/completions`
   with `logprobs=N` — wire that into MMLU and rank A-D by logprob.
   This eliminates extraction entirely and is the standard MMLU eval
   methodology (cf. lm-evaluation-harness `loglikelihood` mode).
3. **Prompt change**: append "Answer with exactly one letter (A, B,
   C, or D):" to push the model toward commitment. Lower-effort,
   smaller gain than the above.

Kill criterion for the logprob path: at n_samples=200, invalid rate
should drop to ≤2% (from current 15-18%). Also enables true paired
comparison since logprobs are deterministic across the same model
state.

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

### Task I — FlashMLA SM89/SM90 build gate (blocking local rebuild on SM89)

**Discovered 2026-05-28 tick (post-/loop, infra work).** Recent FlashMLA
SM90 commits (`b3a33188`, `ed4a7b38`, `8ebe3ff5` and predecessors) added
`csrc/sm90/decode/sparse_fp8/instantiations/*.cu` to the build and FFI
entrypoints in `cuda-kernels/src/ffi.rs` (`arle_flashmla_sm90_sparse_decode_fwd`
etc.). The build env var `ARLE_CUDA_DISABLE_FLASHMLA=1` correctly skips
the `.cu` files BUT the Rust-side FFI calls aren't cfg-gated, so on
SM89-only boxes (RTX 4070 Ti SUPER, Ada Lovelace) the build either:
- compiles `.cu` files and fails with `cannot specify max blocks per
  cluster for this GPU architecture` (the `launch_bounds(...,
  Kernel::CLUSTER_SIZE)` attribute is SM90+), OR
- with `ARLE_CUDA_DISABLE_FLASHMLA=1` skips compile but fails linker
  on `arle_flashmla_sm90_sparse_decode_fwd` and friends.

Local OPD train rebuild is fully blocked on SM89 as of 2026-05-28.
Commit `01d07bf6` (LoRA rank/alpha/target-set CLI args) IS in the
tree but local binary is stuck at the pre-FlashMLA-SM90 build from
2026-05-26.

**Goal**: make `ARLE_CUDA_DISABLE_FLASHMLA=1` a complete disable —
both .cu skip AND Rust FFI either cfg-gated or stub-impl returning
"not built" errors. Alternatively a proper cargo feature
`cuda-kernels/flashmla` that defaults on for SM90 builds and off on
SM89-only.

**Acceptance**: SM89 box rebuilds `opd_step_cuda_infer_teacher_train`
clean with `ARLE_CUDA_DISABLE_FLASHMLA=1` and the resulting binary
runs at parity with the 2026-05-26 binary on the OPD CUDA path.

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
