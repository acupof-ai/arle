# DSv4 FlashMLA per-layer budget fix — needle_gate PASS, exact-fit admission-boundary proof (#85 Route A step 2)

**Date:** 2026-07-08. **Backend:** CUDA, DeepSeek-V4-Flash-FP8, TP=4 (GPUs 4-7,
GPU1 held by a concurrent legitimate job). **Commit:** `3ebc763f9`.
**Scope:** `crates/infer-cuda/src/dsv4.rs` (`kv_budget_plan`),
`crates/infer-cuda/src/attention/kv_layout.rs` (`Dsv4LayerKvLayout::new`).

## Goal

Correctness gate for the FlashMLA per-layer KV budget fix — `kv_budget_plan()`
now sums each layer's real FlashMLA page need instead of `.max()` +
uniform-divide-by-`num_layers`. Per
`docs/plans/2026-07-08-dsv4-route-a-page-granular-prefix-reuse.md`'s KV-budget
section: pure allocation-accounting change, expected byte-identical decode
output — the gate is about catching an off-by-one in the new arithmetic, not
numerics drift.

## Setup deviation (necessary)

DeepSeek-V4-Flash-FP8 (274 GB) structurally requires multi-GPU TP; ran TP=4 on
GPUs 4-7 (GPU1 occupied by a concurrent `arle serve` job, untouched). Built in
an isolated tree (`/host/arle-build-needlegate`, cloned from `/host/arle-build`)
after `pod.sh sync`'s default dirty-file push nearly pulled in ~850 lines of
unrelated uncommitted Route-A WIP from the local tree — reverted before
building. `cargo build --release --features cuda,nccl,deepep --bin arle`
(bare `cuda` misses TP=4 multiproc deps).

## Results — needle gate, 3 context lengths

| length | prompt_tokens | exact | partial | miss | det |
|---|---:|---:|---:|---:|---|
| 512 | 499 | 3/3 | 0 | 0 | NONDET |
| 2048 (max/2) | 1879 | 3/3 | 0 | 0 | NONDET |
| 3800 (near-max) | 3448 | 3/3 | 0 | 0 | NONDET |

**PASS** at all three. NONDET is the documented MoE non-determinism floor
(same-config-twice envelope), not a defect.

## Results — admission-boundary exactness proof

`max_total_tokens=32768`, TP=4: `pool_total 109MB, affordable 68` (per-slot
gate) → `pool-band-affordable 1` → clamped to exactly `num_slots=1`, the
tightest possible fit. Server booted clean, **no `ensure!` panic** in either
`dsv4.rs`'s pre-check or `kv_layout.rs`'s pool-constructor invariant
(`pool.max_total_pages >= num_slots*flashmla_slot_pages`) — confirms the
plan's proof that the new sum-per-layer construction is exact, zero slack.

## Results — measured `num_slots` at three budgets (TP=4, GPUs 4-7)

| `max_total_tokens` | per-slot-state affordable | pool-band-affordable | final `num_slots` |
|---:|---:|---:|---:|
| 4096 | 295 | — (not binding) | 129 |
| 20000 | 103 | 2 | 2 |
| 32768 | 68 | 1 | 1 |

Pre-fix comparison at the same shapes not captured (would need a second
full build+boot at the parent commit; time-boxed out). The admission-boundary
exactness proof is the load-bearing correctness evidence, not the pre/post
delta.

## Problems — orthogonal finding, NOT a `3ebc763f9` defect, newly *reachable* because of it

At the tight `num_slots ∈ {1,2}` boundary this fix makes reachable for the
first time, the scheduler's admission-reject path
(`crates/infer-core/src/lib.rs:1268`, the 2026-07-05 "reject-not-hang" fix
for oversized single prompts) aborts a too-large request cleanly, but the
DSv4 slot's FlashMLA fixed-band reservation is **not released** on that
abort. The next request assigned to the same slot then hits
`HostPagedKvPool::alloc_fixed_band` with a shortfall ("slot 0 needs 130, free
2" / "needs 81, free 4") and `bail!`s — crashing **all 4 worker ranks**.
Reproduced identically at `num_slots=1` (32768) and `num_slots=2` (20000);
did NOT reproduce at `num_slots=129` (4096) — there the oversized prompt hit
a different (graceful, 0-token-completion) path instead.

Root cause lives in DSv4's slot-abort cleanup lifecycle
(`infer-core/src/lib.rs:1250-1281`), not in `kv_budget_plan`/
`Dsv4LayerKvLayout::new`'s arithmetic. Pre-fix, `max_seq_len` values yielding
`num_slots ≤ 2` were rejected outright at *startup* (per
`2026-07-06-dsv4-flashmla-budget-reconciliation-verified.md`) — a live server
in this tight-budget regime never previously existed to expose this path.
Filed as `docs/experience/errors/2026-07-08-dsv4-slot-abort-band-leak-crash.md`.

## Learnings

- An "exact fit, zero slack" budget construction is a genuine correctness
  improvement (confirmed: the `ensure!` fired as a true equality, never
  spuriously) — but tightening a budget makes previously-unreachable
  low-`num_slots` configurations reachable, which can expose latent bugs in
  code paths that only ever ran with headroom before. Verify admission/abort
  lifecycle at the new tight boundary, not just the arithmetic itself.
- `pod.sh sync`'s default dirty-file push can contaminate an isolated
  verification build with unrelated local WIP — clone to a fresh tree and
  check `git status`/`git log` before building when verifying a specific
  commit in isolation from concurrent local work.
