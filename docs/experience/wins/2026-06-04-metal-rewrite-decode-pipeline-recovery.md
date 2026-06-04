# Metal rewrite ~−17% decode regression RECOVERED: cross-step decode pipeline (one step deep), matched A/B, bit-identical

## Context

`infer-metal` (rewrite, `Engine<MetalExecutor, MetalKvPool>`, branch
`arch/ideal-inference-engine`) decoded ~−16 to −18% slower than legacy
`infer/src/backend/metal` on the canonical Qwen3.6-35B-A3B-4bit MoE c=1
(`~70` vs legacy `84.3` pure-decode tok/s) — the residual after the wired-limit
and publish/drain-cadence levers were KILLED
([`2026-06-04-metal-rewrite-decode-deconfound-wired-limit-kill.md`](2026-06-04-metal-rewrite-decode-deconfound-wired-limit-kill.md)).

Root cause confirmed by source: the engine loop is strictly
`submit(N) → poll(N) blocks on eval → apply(N) → submit(N+1)`
(`infer-core/src/lib.rs:367-400`). The rewrite Metal `poll` does a blocking
`mlx::eval(sampled)` and returns `Ready`, so step N's GPU work fully drains
before step N+1 is even built. The GPU idles for the host gap each token
(apply_output + admission + plan-N+1 build + a fresh `begin_session`). Legacy
hides this gap with `pending_sampled` double-buffering
(`request_state.rs:6100-6149`): it issues step N+1's forward *before*
materializing step N.

**Architectural finding (why a pure engine-loop tweak can't fix it):** the
engine derives the next decode row's `last_token` from
`request.generated_tokens.last()` (`infer-core/src/planner.rs:23`), populated in
`apply_output` only after step N's token is materialized. So `build_forward_plan(N+1)`
*requires* step N's value — the engine can never have two steps in flight, and
`PollResult::NotReady` is a re-poll signal, not an overlap signal. The overlap
must live inside the executor, exactly like legacy. The seam contract is
unchanged.

## What Worked

A Metal-local, env-gated (`INFER_METAL_PIPELINE=1`, default OFF) cross-step
decode pipeline in `RealMetalExecutor`. The decode session is held one step
ahead: `submit_decode` issues the NEXT greedy step's `step_session` (async)
before returning the current token, so step N+1's forward overlaps step N's
poll/eval and the scheduler's plan-N+1 build. Per-token sequence:

- **cold (first greedy decode of a turn):** run step N → drain → publish (valid
  page ids, correct gdr) → sample N → record `last_sampled` → prequeue step N+1
  (session left OPEN, async). Return N.
- **fast (`pending` for slot, greedy):** drain + publish the now-committed
  prequeued step (page ids still valid, gdr correct), prequeue the following
  step, return the already-sampled token. **No forward on the engine's critical
  poll path.**

Correctness invariants kept: `committed_len` (engine `kv_seq_len`) tracked
separately from session `cache_len` (one ahead under pipeline) so the seam
length assertion holds; the K/V + gdr prefix snapshot is always published for
the COMMITTED step (never the one-ahead prequeue), so radix reuse stays exact;
prequeue is capacity-bounded and greedy/single-slot-only (any other shape drains
and falls to the HEAD path); a stale prequeue is dropped on epoch change /
prefill. A `std::sync::Once` eprintln proves the path is live.

**`codex review` caught a P2 the 3-turn bench could not** (each turn here goes
through prefill, so cross-request slot recycling never fired): an exact
prefix-cache hit can admit a NEW request straight into `Decoding` on a slot
index a finished request left a `pending` on, and the fast path runs before
`reset_slot_if_epoch_changed` — returning the prior request's stale token and
committing its old session into the new slot. Fixed by gating the fast path on
`pending_matches_live_slot`: same slot, **unchanged slot epoch** (not recycled),
still-open one-ahead session, and `row.kv_seq_len == committed_len`. Any miss
drops the stale pending and falls to the cold path (which resets + drains the
orphaned session). Re-verified bit-identical + win intact after the fix.

### Matched A/B (same freshly-built binary, c=1, interleaved, M4 Pro 48 GB)

`agent-bench` `synthetic(256,3,32,48)`, 4 runs each, `pure_decode_tok_s`
(aggregate) + per-turn steady decode_tok_s.

**Qwen3.6-35B-A3B-4bit MoE (canonical):**

| run | A: OFF (HEAD) | B: ON (pipeline) | paired Δ |
|----:|--------------:|-----------------:|---------:|
| 1 | 66.1 | 71.8 | +5.7 |
| 2 | 63.0 | 73.7 | +10.7 |
| 3 | 72.1 | 82.5 | +10.4 |
| 4 | 74.9 | 83.5 | +8.6 |
| **mean** | **69.0** | **77.9** | **+12.9%** |

Every pair positive. B's clean-thermal runs (82.5-83.5) recover to within ~1% of
legacy 84.3 — the ~−17% regression is closed.

**Qwen3.5-0.8B sanity:**

| run | A: OFF PURE | B: ON PURE |
|----:|------------:|-----------:|
| 1 | 246.3 | 287.0 |
| 2 | 206.1 | 295.8 |
| 3 | 236.2 | 242.5* |
| 4 | 195.1 | 279.6 |
| **mean** | **220.9** | **276.2 (+25%)** |

B > A in all 4 pairs; B median ~283 ≈ legacy 282.5 (full recovery). (*run 3 B
thermal dip; still ahead of paired A.) A is noticeably noisier (thermal
195-246); B is tighter, confirming the overlap removes a variable host-gap stall.

### Correctness gate — bit-identical, both models

FNV-1a of each turn's greedy tokens, A vs B:

- Qwen3.6: `875087ddac2cf3ef` / `eb0b8243920d2ad2` / `409375e5fa866588` — identical.
- Qwen3.5-0.8B: `ff7b9f15227012db` / `7f797566848864e5` / `6bdd8a612548c4b5` — identical.

TTFT 6→3 radix reuse fires in every run (ON and OFF). No hang/deadlock.
Pipeline-fast-path probe confirmed LIVE on both models.

### Guardrails

- `infer-cuda` (shares the seam) stays green:
  `CUDARC_CUDA_VERSION=12060 cargo check -p infer-cuda --no-default-features --features cuda,no-cuda`.
  Seam types untouched — change is Metal-internal (new struct fields + methods).
- `cargo test -p infer-metal --features metal`: 6/6 pass (placeholder submit/poll
  plumbing tests prove HEAD/default behavior preserved).
- `cargo test -p infer-core -p infer-seam`: 28 + 5 pass (engine loop unaffected).
- `cargo clippy -p infer-metal -p agent-bench ... -D warnings`: clean.

## Files changed

- `crates/infer-metal/src/executor.rs` — `pipeline_decode_enabled()` env gate +
  `std::sync::Once` fast-path probe; `PendingStep`; `RealMetalExecutor.pending`;
  `MetalSlotState.{committed_len,last_sampled}`; pipelined `submit_decode`
  (cold-seed + fast-path) + `pending_matches_live_slot` (recycled-slot guard) +
  `commit_pending_then_prequeue` + `prequeue_decode`; prefill clears stale
  pending. HEAD path unchanged when flag OFF. **KEPT.**

Default is OFF (opt-in `INFER_METAL_PIPELINE=1`); runtime default behavior
unchanged → no default-flip. Left in tree, no git (sole committer is ckl).

## Rule

- When the engine loop owns the per-token feedback (next `last_token` ←
  materialized step N), it can NEVER overlap decode steps itself; `NotReady` is a
  re-poll, not a pipeline. Cross-step overlap MUST live in the executor
  (legacy `pending_sampled` shape), driven through the unchanged seam.
- A one-ahead decode session is safe for greedy iff you track committed vs
  session length separately AND publish the K/V + recurrent (gdr) prefix
  snapshot for the COMMITTED step only — publishing the prequeued step's gdr
  corrupts the radix snapshot (recurrent state is sequence-position-specific).
- Prove the overlapped path is live with a `Once` probe AND a bit-identical
  greedy-token FNV A/B before crediting the tok/s delta; interleave A/B runs —
  the HEAD baseline alone swung 195-247 from thermal drift, which would have
  masked the +13% on a non-interleaved comparison.
