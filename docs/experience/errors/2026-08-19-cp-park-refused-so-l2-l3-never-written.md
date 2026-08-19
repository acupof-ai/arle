# Under CP the whole-slot park is refused, so KV pressure always recomputes — 2026-08-19

## Context

`--kv-recall` was deleted (`3f826c204`); L2/L3 are wanted as a **lossless**
capacity extension instead, with tier I/O only at prefill/scheduler boundaries
and never in decode. Verifying that the surviving path actually moves KV into
L2/L3 under pressure.

TP=2 CP=2 on Qwen3.6-27B, `--mem-fraction-static 0.25 --kv-dram 32GiB
--kv-disk ... --kv-oversubscription`, L1 profiled to 128 local pages
(2048 local tokens). Workload: 6 concurrent sessions × ~1500-token prefix ×
2 turns × 256 generated tokens, 65 s wall — several times over capacity.

## Phenomenon

Every KV-tier counter stayed at exactly 0: `demoted_pages`, `promoted_pages`,
`demoted_slots`, `host_demoted_pages`, `reuse_hit_*`, `useful_read_bytes`,
`useful_write_bytes`. Nothing was ever written to L2 or L3.

The engine was not idle about it — the log carries **9,970** preempt/park lines,
all of the same shape:

```
WARN infer_core::planner: planner.rs:243
whole-slot KV demote failed for slot 0:
Qwen3.6 whole-slot swap: slot 0 ran B2 CP decode; preempt via recompute
```

## Cause

`Qwen35SlotState::swap_out_image` (`qwen35_state.rs:743`) refuses outright once
the slot has run a B2 CP decode step:

```rust
ensure!(
    !self.decode_recurrent_live,
    "Qwen3.6 whole-slot swap: slot {slot} ran B2 CP decode; preempt via recompute"
);
```

The refusal is about the **recurrent** state, not the KV: under B2 the decode
pair is a 1/cp head subset and the full pair is stale at the scatter point, so
neither reconstructs the full-dim image the slot record wants. The KV pages
themselves are ordinary paged KV and are perfectly swappable.

`requeue_preempted_decode` (`planner.rs:299`) treats the failure as "no tier"
and falls through to recompute, so under 2D every preemption pays a full
re-prefill and L2/L3 stay empty.

## Impact

This is the blocker for using L2/L3 as a capacity extension under CP. The
capacity mechanism exists and fires — it is refused every time. Single-GPU and
plain TP are unaffected (no B2 decode, so `decode_recurrent_live` stays false).

One smaller observation from the same run:

- Turn-2+ TTFT still collapsed (1.71 s → 0.07 s median) despite zero
  prefix-cache activity. Cause unknown.

The inert prefix cache in that run was not a defect: it is disabled by design
under 2D (`lib.rs:1593`, `lib.rs:1674`, `planner.rs:417`) because the ring pass
recomputes the whole prompt and the match+attach collectives deadlock
cross-communicator. `prefix_cache_lookups_total` 0 is the intended reading.

## Rule

A capacity mechanism that is *invoked* is not a capacity mechanism that *works*.
The park path had been reachable and reached for months; it refused on every
call under 2D and the only evidence was a WARN nobody counted. Any tier whose
success counters can sit at exactly 0 through a full pressure run needs a
failure counter next to them — a refusal that only logs is invisible at the
metric surface where capacity decisions are actually read.

## Status

Fixed 2026-08-19 — see
[`wins/2026-08-19-cp-slot-park-works-l2-l3-nonzero.md`](../wins/2026-08-19-cp-slot-park-works-l2-l3-nonzero.md).
Each rank round-trips its own 1/cp shard (`b2cc9b783`); capture no longer
destroys the slot before the tier accepts it (`8e325e8dc`); refusals are
counted as `kv_tier_slot_demote_failures_total`.
