# Metal DFlash Qwen3.5 Layer 2c.4 — concurrent-DFlash bucket B=2 bit-ident — 2026-04-19

> Layer 2c.4 of the Qwen3.5 DFlash batched-verify roadmap
> ([`docs/plans/metal-dflash-qwen35-2c4-mixed-mode-dispatch.md`](../../plans/metal-dflash-qwen35-2c4-mixed-mode-dispatch.md)).
> Phase 2B of that plan — the pure-DFlash concurrency bucket. The full
> mixed-mode two-subgraph `async_eval` story from the plan is deferred
> (see §Problems observed) because `qwen35_dflash_speculative_block_batched`
> performs a mid-sequence host-side sampling eval that can't be coalesced
> into a single `async_eval` alongside plain-decode. Correctness-only;
> first concurrent-DFlash throughput delta lands with a follow-up bench run.

## Goal

**Type:** regression-check.
When ≥2 DFlash-enabled slots are concurrent, does the new
`execute_qwen35_dflash_packed_batch → try_decode_qwen35_dflash_speculative_batch`
path produce the same sampled token (and therefore the same first-step
output trajectory) as a scalar `qwen35_dflash_speculative_block` run for
an equivalent single-row state? Acceptance: for two rows starting from
bit-identical prefilled states with greedy sampling, each batched row's
first sampled token equals the scalar path's sampled token.

## Hypothesis

The batched path composes three primitives already proven bit-ident at
B=1/B=2 by the prior 2c.1/2c.2/2c.3 entries:

- Draft forward: `dflash_draft_forward_batched` (2c.2 — B=1
  `max_abs_delta = 0`, rank-1 RoPE offset + rank-3 reshape are
  numerically identical to the scalar rank-2 path).
- Target verify: `qwen35_compiled_verify_block_batched` (2c.1 — B=2
  `max_abs_delta = 0`, left-padded additive mask + per-row RoPE offsets
  collapse to the scalar path when all rows share the same active
  prefix length).
- GDR rollback: `qwen35_rollback_to_accepted_varlen` (2c.3 — B=1
  `max_abs_delta = 0` across `k ∈ {0, 1, 2, block_size=4}`, varlen
  kernel's `steps[b]` broadcast collapses to scalar at B=1).

At B=2 with identical prefilled states and greedy sampling, the stacked
tensors are `concatenate([state_0, state_1], axis=0)` where `state_0 ==
state_1`. Every primitive is a pure axis-0 broadcast at this fixture, so
the first-step sampled token on every row must equal the scalar path's
sampled token.

If it does not match, the bug is in one of:

1. The per-row eligibility gate in
   `try_decode_qwen35_dflash_speculative_batch` allowing a mismatched row
   into the batch (checks: phase=Decode, backend=Cpp, DFlash enabled,
   empty token_buffer, captured target_hidden, committed last_token,
   matching `cache_len`, matching `target_hidden.shape()[0]`, matching
   `draft_state.active_len()`, and pointer-equal runtime/weights/config).
2. Per-row K/V active-prefix slicing before stacking for the draft
   forward — scalar path consumes `active_kv_flat()`; batched path must
   slice before `concatenate_axis(0)` to avoid attending over inactive
   zero-filled capacity slots.
3. Scatter of sampled tokens back to each row's `token_buffer` +
   `record_sampled_token` ordering (first popped token vs retained
   speculative tail).

## Parameters

Phase 2B scope (Phase 1 landed earlier this session; line numbers below
are post-fix HEAD):

- New config field: `MetalSchedulerConfig.metal_dflash_concurrency_off:
  bool` at `infer/src/backend/metal/scheduler.rs:45` (default **`true`**
  — legacy behavior, see §Problems observed P1 round-2). When `true`,
  preserves the legacy `open.len() >= 2` permanent-downgrade at
  `runtime.rs:1078-1097`. When flipped to `false`, the three-bucket
  partition below exposes the new path; this is currently bench/opt-in
  only because admission still caps DFlash to solo requests.
- New partition in `execute_decode_batch` at `runtime.rs:1090-1130`:
  plain rows → existing `execute_qwen35_packed_decode_batch`; DFlash
  rows (`len ≥ 2`) → new `execute_qwen35_dflash_packed_batch`;
  single-DFlash row falls through to per-row `execute_decode_single`.
- New dispatcher: `execute_qwen35_dflash_packed_batch` at
  `runtime.rs:1190-1250`. Gathers `&mut [&mut MetalRequestState]` from
  the DFlash bucket, calls the new wrapper, scatters sampled tokens.
- New wrapper:
  `MetalRequestState::try_decode_qwen35_dflash_speculative_batch` at
  `request_state.rs:947+`. Eligibility gate enforces the 10 conditions
  above; on failure returns `Ok(None)` and the dispatcher falls back to
  per-row `execute_decode_single` (graceful, not a scheduler cancel).
- Batched primitive: `qwen35_dflash_speculative_block_batched` at
  `dflash.rs:2065+` (landed Phase 1; `#[allow(dead_code)]` retired in
  Phase 2B). Slices each row's draft K/V to `active_len` via
  `slice(_, [0,0,0,0], [1, n_kv, draft_len, head_dim], [1,1,1,1])`
  before `concatenate_axis(0)` (codex P1 finding #2, now fixed).
- Routing gate for batched path: new
  `MetalDflashRuntime::batched_draft_path_eligible()` at
  `dflash.rs:331+`. Returns true only when `DFLASH_DRAFT_CPP=1` AND
  `draft_cpp_model.is_some()` AND `draft_attention_mask != "causal"` —
  mirroring the scalar draft-path routing predicate (codex P1 finding
  #3, now fixed).
- Correctness test:
  `tests::qwen35_dflash_packed_batch_b2_matches_scalar_runs` at
  `dflash.rs:3094`. Constructs three `MetalRequestState` instances with
  identical prompts, prefills each to Decode phase, takes one scalar
  `decode_step` on state C to capture the expected first token, then
  runs `try_decode_qwen35_dflash_speculative_batch(&mut [&mut A, &mut
  B])` and asserts `sampled[0] == sampled[1] == scalar_first_token`.
- Model: `mlx-community/Qwen3.5-4B-MLX-4bit` (target) + local
  `z-lab/Qwen3.5-4B-DFlash` cache (draft). Test gates on
  `env::var("QWEN35_MODEL_PATH").is_ok()` and sets
  `DFLASH_DRAFT_CPP=1` via `env::set_var` under `metal_test_guard` so
  both scalar and batched paths take the C++ draft path (otherwise the
  Phase 2B eligibility gate would reject the batched call and the test
  would fail).

Command:

```bash
QWEN35_MODEL_PATH="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3" \
cargo test --release --no-default-features --features metal -p infer \
  --lib -- --test-threads=1 qwen35_dflash_packed_batch_b2_matches_scalar_runs
```

## Environment

- Hardware: Apple M4 Max (40 GPU cores, ~400 GB/s UMA).
- macOS: 26.3.1 (build 25D771280a); Metal 3 via MLX 0.29.x
  (pinned in `crates/mlx-sys/CMakeLists.txt`).
- Commit: HEAD `29fa6e4` (train Metal bf16 roundtrip) + uncommitted
  Phase 1+2B+P1-fixes diff. Combined stat:
  ```
   crates/mlx-sys/src/lib.rs                     |   13 +
   crates/mlx-sys/src/mlx_dflash_draft_model.cpp |  238 +++++
   infer/src/backend/metal/dflash.rs             | 1409 +++++++
   infer/src/backend/metal/mlx.rs                |  124 +++
   infer/src/backend/metal/qwen35.rs             |  329 +++++
   infer/src/backend/metal/request_state.rs      |  340 +++++
   infer/src/backend/metal/runtime.rs            |  127 ++
   infer/src/backend/metal/scheduler.rs          |    8 +
   8 files changed, 2493 insertions(+), 95 deletions(-)
  ```
  (This bundles the full 2c.1 + 2c.2 + 2c.3 + 2c.4-Phase-2B code diff
  under one commit; the four docs/wins entries ship as separate docs
  commits alongside.)
- Feature set: `cargo build --release --no-default-features --features metal`.

## Results

Full DFlash subset:

```
running 4 tests
test backend::metal::dflash::tests::dflash_qwen35_verify_batched_matches_two_single_row_runs ... ok
test backend::metal::dflash::tests::draft_forward_batched_matches_forward_for_b1 ... ok
test backend::metal::dflash::tests::qwen35_dflash_packed_batch_b2_matches_scalar_runs ... ok
test backend::metal::dflash::tests::qwen35_rollback_to_accepted_varlen_matches_scalar_for_b1 ... ok

test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 330 filtered out; finished in 3.32s
```

- `qwen35_dflash_packed_batch_b2_matches_scalar_runs`:
  `scalar_first_token=24218`, `batched_first_tokens=[24218, 24218]` —
  identical first-step output on both rows.
- Upstream primitives (regression check under the same test run):
  2c.1 `overall_max_abs_delta=0`; 2c.2 `max_abs_delta=0`; 2c.3
  `overall_max_abs_delta=0`.
- Full lib suite under `--features metal`: **326 passed, 0 failed, 8
  ignored**. No regression in the 322 non-DFlash tests.
- Build: zero Rust warnings under
  `cargo build --release --no-default-features --features metal --tests -p infer`.
- Wall time: 3.32 s for the 4-test DFlash subset.
- **Δ vs baseline:** zero numerical delta. No throughput number —
  Phase 2B is correctness-only; the first concurrent-DFlash guidellm
  sweep is a follow-up bench entry (see §Follow-ups).

## Problems observed

- **Mixed-mode (plain ⨯ DFlash) async fan-out deferred.** The plan at
  [`docs/plans/metal-dflash-qwen35-2c4-mixed-mode-dispatch.md`](../../plans/metal-dflash-qwen35-2c4-mixed-mode-dispatch.md)
  targeted a two-subgraph design: build a plain-decode lazy subgraph +
  build a DFlash lazy subgraph, then one `mlx_async_eval` fans out
  across Metal streams for parallel execution. The first Phase 2 agent
  (aa867a0690aaa8652) correctly flagged that
  `qwen35_dflash_speculative_block_batched` performs a **mid-sequence
  host-side sampling eval** in step 4 (draft argmax via `sample_rows`
  at `dflash.rs`), which forces Metal materialization inside the
  batched block and breaks the lazy-throughout contract the
  two-subgraph design requires. Making the block purely lazy requires
  splitting it into `_build_draft_logits` → `eval` → `host_argmax` →
  `_build_verify_and_rollback` — an API refactor outside Phase 2B
  scope. Phase 2B therefore ships the pure-DFlash bucket only; DFlash +
  plain rows run sequentially (same behavior as today's serialized
  per-row `execute_decode_single`, but with concurrent DFlash rows
  batched together when ≥2).
- **Three [P1] codex findings on first Phase 2B implementation** (all
  fixed before commit):
  1. Eligibility gate only checked `cache_len` equality across rows;
     added `target_hidden.shape()[0]` and `draft_state.active_len()`
     equality checks so mismatched rows fall back to scalar instead of
     erroring mid-batch.
  2. Draft forward stacked raw `k_caches[i]`/`v_caches[i]` including
     zero-filled inactive capacity; added per-row active-prefix slice
     before `concatenate_axis(0)`.
  3. Batched path unconditionally used C++ draft with `attn_mask=None`;
     added `batched_draft_path_eligible()` that gates on
     `DFLASH_DRAFT_CPP=1` AND `draft_attention_mask != "causal"` to
     mirror the scalar routing.
- **Two more codex findings on the round-2 diff** (also fixed before
  commit — documented here so the bench doc captures the full review
  trail):
  1. [P1] `runtime.rs` admission gate (`admit_request` at ~920)
     still caps DFlash to solo requests (`active.is_empty()`), so
     flipping `metal_dflash_concurrency_off=false` would leave a lone
     surviving DFlash row routing through per-row
     `execute_decode_single` instead of joining the batched path —
     regressing the A+B mixed case vs the legacy permanent-downgrade.
     Flipped the new config field's **default back to `true`** so the
     Phase 2B path is opt-in for benchmarking until admission is lifted
     and the throughput gain is measured. The three-bucket partition is
     live but unreachable under defaults.
  2. [P2] `qwen35_build_updated_target_hidden` fell back by cloning
     `target_hidden_per_row[0]` for every row on capture mismatch,
     which silently broadcasts row 0's hidden state onto rows 1..B and
     diverges the state of the rejecting rows. Changed the fallback
     parameter to `&[MlxArray]` (scalar caller passes
     `std::slice::from_ref(target_hidden)`; batched caller passes
     `&target_hidden_per_row`) so each row's fallback is its own
     pre-block hidden state.
- **Two [P1]/[P2] codex findings on the round-3 diff.** The [P1]
  "plain-decode cache rollback on singleton fallback" was **retracted**
  by a follow-up codex review (2026-04-19): the
  `invalidate_qwen35_decode_batch_cache` sync on the `Ok(None)` arm is
  the ONLY mechanism that propagates `packed_kv_flat`/`packed_gdr_flat`
  updates from batched decode into each request's per-row KV state —
  bypassing it (as an earlier attempted fix did) leaves the singleton
  survivor decoding on stale caches. The original unconditional invalidate
  is correct; no fix needed. The [P2] finding below was only reachable
  when `metal_dflash_concurrency_off=false`; default-`true` production
  paths were unaffected.
  1. [P2] **All-or-nothing DFlash demotion on buffered-speculative
     rows** (`request_state.rs:1704-1713`). The eligibility gate returned
     `Ok(None)` as soon as any row had `!dflash.token_buffer.is_empty()`
     or missing `target_hidden`, which demoted the whole bucket to
     scalar `execute_decode_single`. After a successful speculative
     block (`accepted_inputs > 1` with tail kept), that row carried a
     non-empty buffer on the next tick, so in practice any mixed-
     acceptance pattern collapsed the bucket. **Fixed in follow-up
     commit FIXME-commit-sha**: the function now partitions rows into a
     ready subset (empty buffer + captured `target_hidden` + cross-row
     shape/handle agreement with the first ready row) and a stale
     subset; the batched kernel runs on the ready subset iff `len >= 2`,
     and the caller in
     `runtime.rs::execute_qwen35_dflash_packed_batch` routes stale rows
     through `execute_decode_single`. The new return type is
     `Option<DflashBatchOutcome { ready_indices, tokens }>`; see
     `request_state.rs` for the helper predicates
     `row_passes_dflash_batch_per_row_predicates` and
     `rows_agree_on_dflash_batch_cross_row_predicates`.
- **Single-row DFlash still serializes.** The dispatcher requires
  `dflash_rows.len() >= 2` to batch; one DFlash row falls through to
  `execute_decode_single` (per-row). The batched-stack overhead isn't
  worth paying for B=1 on an already-batched primitive. This is
  intentional.
- **§6 watch items:**
  - Warmup (§6.1): single test invocation; N/A.
  - Thermal (§6.2): 3.32 s total; N/A.
  - Determinism (§6.7): synthetic deterministic inputs, greedy
    sampling (temp=0), compiled graphs; delta is exactly zero.

## Learnings

- **Mid-sequence host-side evals defeat the async fan-out story.** The
  original 2c.4 plan assumed speculative block + plain decode could
  both run as lazy subgraphs and fan out across Metal streams via one
  `async_eval`. Draft sampling requires host-visible token ids to
  build the verify step, so the draft-forward → argmax boundary is a
  hard host-sync point. Any future mixed-mode design must either
  (a) split the speculative block into pre-sample and post-sample
  lazy halves with an explicit eval+argmax between, (b) push draft
  argmax into a GPU kernel that writes token ids back to Metal buffers
  (`mlx` does not currently expose this as a batched op), or (c) give
  up on fan-out parallelism and accept serial DFlash+plain dispatch —
  which is what Phase 2B does.
- **Per-row active-prefix slicing is load-bearing for batched SDPA
  parity.** Phase 1's B=2 test constructed both rows with identical
  `active_len` (same prefill length), so the inactive-capacity slots
  in the stacked K/V happened to also be identical and the bug didn't
  manifest. This is the classic "correctness holds on fixture but not
  in production" trap from `feedback_matched_ab_for_small_bench_effects.md`.
  The codex P1 finding caught it before it shipped; defensive fix is
  now in place for arbitrary `active_len` mismatches (which the
  eligibility gate also refuses at the entry boundary — double
  defense).
- **Agent architectural-flag escape > agent trying to ship anyway.**
  The first Phase 2 agent stopped at Explore, identified the lazy-eval
  bottleneck, and presented three options (narrow/medium/full) instead
  of attempting a mid-scope compromise. The narrow option (Phase 2B)
  landed cleanly in a second single-round attempt with zero bit-ident
  regressions. This matches the CLAUDE.md `>5 files → stop + flag`
  rule and the 2-strike rule on hard problems.
- **Stopping rule hit** (spec §7.3): variance N/A for correctness;
  hypothesis confirmed with strictly-better-than-tolerance result
  (batched and scalar first-token IDs match exactly); §6 clean; no
  prior-snapshot delta (first 2c.4 entry). One run suffices.

## Follow-ups

- **Concurrent-DFlash throughput bench** (task #13 on the queue):
  `scripts/bench_guidellm.sh metal-qwen35-dflash-c-sweep` with c ∈ {1,
  2, 4, 8}, compared against the 162.9 tok/s c=8 baseline from
  [`2026-04-19-metal-qwen35-final-state.md`](2026-04-19-metal-qwen35-final-state.md).
  Expected: Phase 2B removes the `open.len() >= 2` permanent DFlash
  downgrade, so c=2/4/8 should retain the DFlash per-row K̄≈3.6
  acceptance rate instead of collapsing to plain decode at K̄=1.
  Theoretical ceiling `~ (1 + K̄) × plain_c_throughput` before the
  GDR kernel 6.1 ms/row computation cap kicks in.
- **Mixed-mode (plain ⨯ DFlash) parallel fan-out** (future phase):
  requires splitting `qwen35_dflash_speculative_block_batched` into
  `_build_pre_sample` + `_build_post_sample` halves with an explicit
  eval between, then a two-subgraph `async_eval` across plain + post-
  sample-DFlash. Measure first whether Phase 2B's serialized mixed
  dispatch is already under the GDR-kernel cap (likely: the 6.1 ms/row
  number is per-DFlash-row, plain rows don't touch GDR). If serialized
  mixed dispatch already hits ≥2× the c=8 baseline, the mixed-mode
  refactor may not be worth the API surface churn.
- **Layer 2d**: `speculative_block` batched sibling (parallel to
  `verify_block_batched`) for the reject-path. Scalar today;
  low-priority since the reject path is cold.
- **Test gating hygiene** (task #15): `draft_forward_batched_…` +
  `qwen35_rollback_to_accepted_varlen_…` + `qwen35_dflash_packed_batch_…`
  all gate on `QWEN35_MODEL_PATH` but also load `z-lab/Qwen3.5-4B-DFlash`
  unconditionally via `MetalDflashRuntime::load`. Environments with
  base target but no cached draft checkpoint regress. Add
  `QWEN35_DFLASH_DRAFT_PATH` guard or local-cache probe.

## Cross-links

- Plan:
  [`docs/plans/metal-dflash-qwen35-2c4-mixed-mode-dispatch.md`](../../plans/metal-dflash-qwen35-2c4-mixed-mode-dispatch.md)
  (Phase 2B is a strict subset; mixed-mode deferred).
- Prior 2c.3 entry (varlen rollback B=1 bit-ident):
  [`2026-04-19-verify-metal-qwen35-dflash-2c3-varlen-rollback-b1-bit-ident.md`](2026-04-19-verify-metal-qwen35-dflash-2c3-varlen-rollback-b1-bit-ident.md).
- Prior 2c.2 entry (batched draft forward B=1 bit-ident):
  [`2026-04-19-verify-metal-qwen35-dflash-2c2-draft-batched-b1-bit-ident.md`](2026-04-19-verify-metal-qwen35-dflash-2c2-draft-batched-b1-bit-ident.md).
- Prior 2c.1 entry (verify_block_batched B=2 bit-ident):
  [`2026-04-19-verify-metal-qwen35-dflash-2c1-b2-bit-ident.md`](2026-04-19-verify-metal-qwen35-dflash-2c1-b2-bit-ident.md).
- Layer 1 baseline (packed decode 162.9 tok/s at c=8 — concurrent-
  DFlash throughput target):
  [`2026-04-19-metal-qwen35-final-state.md`](2026-04-19-metal-qwen35-final-state.md).
- Runtime dispatch: `infer/src/backend/metal/runtime.rs:1090-1250`.
- Request-state wrapper: `infer/src/backend/metal/request_state.rs:947+`.
- Batched primitive: `infer/src/backend/metal/dflash.rs:2065+`
  (`qwen35_dflash_speculative_block_batched`).
