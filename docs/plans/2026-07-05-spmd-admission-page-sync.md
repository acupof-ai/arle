# SPMD admission divergence — sync the free-pages check across TP ranks

> Status: Shipped (`5fd6a8984`) and pod-verified — all 4 ranks now call the
> admission collective symmetrically, no divergence. This closed the
> SPMD-livelock class of bug, but the same repro hung again via a separate,
> unrelated mechanism (a plain capacity shortfall with no reject path) — see
> "Round 5" in
> [errors/2026-07-05-multiproc-lockstep-ack-hang-no-timeout.md](../experience/errors/2026-07-05-multiproc-lockstep-ack-hang-no-timeout.md),
> fixed and pod-verified separately in `eeac3d2b9`. **The original
> user-reported hang (`prompt_tokens≈8106` at TP=4/EP=4) is now fully
> resolved end-to-end** — both fixes confirmed on real hardware.

## Root cause — proven, not inferred

Serving DeepSeek-V4-Flash-FP8 TP=4/EP=4, a `prompt_tokens=8106` request hangs
the entire server forever (`prompt_tokens=7661` works, ~5s). Full incident
history: [errors/2026-07-05-multiproc-lockstep-ack-hang-no-timeout.md](../experience/errors/2026-07-05-multiproc-lockstep-ack-hang-no-timeout.md)
(rounds 1-3: two real, kept relay-layer hardening fixes that turned out not
to be the mechanism). Round 4 caught it live: simultaneous gdb snapshots of
all 4 worker PIDs + coordinator, 3 timepoints ~15-40s apart, show the
**stuck rank rotating** (rank3 → {rank1,rank2} → rank1) while the others sit
idle — proof against a static all-4-ranks NCCL deadlock, proof for a moving
cross-rank mismatch.

`crates/infer-core/src/lib.rs:1087` `admit_waiting`:
```rust
let mut remaining_pages = self.kv.free_pages();   // per-rank LOCAL, line 1084
...
while self.active.len() < running_cap {
    match self.try_admit_front_waiter(slot, ..., &mut remaining_pages)? { ... }
}
```
`try_admit_front_waiter` (`lib.rs:1128`) unconditionally calls the NCCL
min-reduce collective `cached_prefix_match_len` (`infer-cuda/src/executor.rs:2489`
DSv4, `:3604`-ish Qwen3.6 → `tp_min_usize` → `tp.rs:432
all_reduce_min_scalar_i32`) for the front waiter — every rank must call this
symmetrically every tick. **After** that collective returns, the Admit vs.
Throttle decision uses `pages_needed > *remaining_pages`
(`lib.rs:1177`/`prefix.rs`), where `remaining_pages` traces back to
`self.kv.free_pages()` — `infer-seam/src/host_paged_kv_pool.rs:118`, **purely
rank-local state, never synced**.

The control request's `host_demoted_pages:13` (a per-rank KV-tier residual)
is exactly the kind of state that can differ across ranks. When it does, one
rank Admits (→ `active.len() >= running_cap` forever after → its `while`
loop condition goes false → **it never calls the admission collective
again**) while another Throttles (→ `waiting` stays non-empty →
**it keeps calling the same collective every tick**). NCCL matches
collective calls by call order, not by type/content — a rank that stopped
calling and one still calling can never realign, except by accidental tick
coincidences (which is the "rotating stuck rank" observed). The request
never reaches uniform admission and the server hangs forever.

## Fix — sync the one divergent value, minimal footprint

Same pattern already used for `cached_prefix_match_len` (compute locally,
min-reduce, decide identically on all ranks): sync `remaining_pages`'s
starting value **once per `admit_waiting()` call**, not per-candidate — the
loop's subsequent arithmetic only *decrements* it from
already-TP-consistent `pages_needed` values, so a single sync at the top
keeps the whole tick's decisions symmetric.

1. **`infer-seam/src/lib.rs`** — new `BackendExecutor` method, default
   identity (correct no-op for single-rank/non-TP backends, same convention
   as `cached_prefix_match_len`'s default-`0`):
   ```rust
   /// Cross-TP-rank min-reduce of a per-rank-local scalar. Single-rank/no-TP
   /// backends (default) return `local` unchanged. TP backends MUST call
   /// this symmetrically on every rank, every tick that reaches it — same
   /// discipline as `cached_prefix_match_len`'s reduce (#146-class hang if
   /// one rank stops calling while another keeps calling).
   fn tp_sync_min(&self, local: usize) -> anyhow::Result<usize> {
       Ok(local)
   }
   ```
   (Name deliberately distinct from the existing private
   `Dsv4Executor::tp_min_usize`/`Qwen35Executor::tp_min_usize` inherent
   methods in `infer-cuda/src/executor.rs` — no shadowing ambiguity.)

2. **`infer-cuda/src/executor.rs`** — implement for both `Dsv4Executor`
   (near :2489, alongside its existing `cached_prefix_match_len` impl) and
   `Qwen35Executor` (near :3604-ish), delegating to each executor's existing
   private `tp_min_usize` inherent helper (already wraps
   `all_reduce_min_scalar_i32` with the right error context):
   ```rust
   fn tp_sync_min(&self, local: usize) -> Result<usize> {
       self.tp_min_usize(local, "admission free pages")
   }
   ```

3. **`infer-core/src/lib.rs:1084`** — sync the starting value once:
   ```rust
   let mut remaining_pages = self.executor.tp_sync_min(self.kv.free_pages())?;
   ```
   (`admit_waiting` already has `&mut self.executor` reachable the same way
   `try_admit_front_waiter` calls `self.executor.cached_prefix_match_len`.)

4. **Other backends** (`infer-metal`, `infer-hip`, `infer-vulkan`): no change
   needed — the seam default (`Ok(local)`) is correct wherever there's no TP
   group (Metal doesn't TP today; HIP/Vulkan are experimental single-box
   lanes). Confirm at implementation time that none of them already run
   TP>1 in a way that would need a real impl instead of the default.

## Why this is the minimal fix, not the only conceivable one

- **Rejected: gate re-entry into the admission collective instead** (e.g. a
  broader per-tick barrier forcing every rank to reach the same decision
  boundary before any proceeds). Bigger surface, redundant with syncing the
  one proven divergent value; only reach for this if syncing `free_pages`
  turns out to be an incomplete fix (see Not yet ruled out below).
- **Deliberately not fixed here**: `prefix_match.matched_len` from
  `self.radix.peek_longest_prefix_match(...)` (`lib.rs:1142`, page-radix
  cache) is a SEPARATE rank-local read that round 4 did not implicate and
  this plan does not touch. If a future hang shows the same rotating-stuck-
  rank signature with `free_pages` already synced, check this next —
  same failure class, different value.

## Verification plan (before declaring shipped)

1. Local: `cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib` (Mac typecheck), `cargo test -p infer-core --lib` (existing scheduler tests must stay green — this touches a hot decision path).
2. Pod: rebuild at TP=4/EP=4, repeat the EXACT repro (`prompt_tokens=8106` immediately after a `prompt_tokens=7661` control, so `host_demoted_pages` residual state is present) — must complete instead of hang. Repeat ≥3x (the divergence was itself intermittent-looking across snapshots; one clean run is not sufficient evidence).
3. Regression: the `prompt_tokens=7661` control alone, and a plain single-request TP=4 smoke test, must show no behavior change (this is a correctness fix on a rarely-hit divergence, not a default-flip — no perf claim, no bench entry needed per CLAUDE.md's benchmark-exemption for correctness-only changes, but note it explicitly in the commit body).
4. Write the wins/errors entry closing out
   [errors/2026-07-05-multiproc-lockstep-ack-hang-no-timeout.md](../experience/errors/2026-07-05-multiproc-lockstep-ack-hang-no-timeout.md)'s
   "Status — paused here" section with the actual fix commit.
