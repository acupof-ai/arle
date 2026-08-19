# Whole-slot park works under CP: L2/L3 goes from 0 bytes to 390 round-trips — 2026-08-19

## Context

Fixes [`2026-08-19-cp-park-refused-so-l2-l3-never-written.md`](../errors/2026-08-19-cp-park-refused-so-l2-l3-never-written.md).
Under TP=2 CP=2 every whole-slot park was refused (9,970/9,970 in one 65 s
pressure run) and L2/L3 received nothing. `swap_out_image` rejected any slot
that had run a B2 CP decode step.

Commits: `b2cc9b783` (park the shard), `8e325e8dc` (capture before release),
`cb9a53373` (comment trim).

## What Worked

**Each rank round-trips its own shard.** The refusal was about the recurrent
pair, not the KV: under B2 the live state is the 1/cp decode pair and the full
pair is frozen at the scatter point, so no rank can produce a full-dim image.
But park and promote are lockstep across the cp group and every rank owns its
own tier, so a rank can capture and restore exactly the shard it holds. No
collective, no full-dim reconstruction — the earlier plan to all-gather the
shards was more machinery for the same result.

`Qwen35SlotImage` records which pair it carries. Swap-in lands a decode-pair
image back in the decode pair and marks every layer scattered, so the resumed
decode reads it instead of re-scattering from the zeroed full pair. The full
pair has no reader under 2D — MTP/spec is refused there
(`executor/qwen35.rs:642`) and the spec snapshot is gated off (`:2453`).

**Capture must not destroy the slot.** With CP parks succeeding, the next park
attempt exposed a latent ordering bug: `swap_out_image` freed the device state
and only then did `demote_slot` try to store the bytes. A refused insert
returned `Ok(false)` — "not parked" — and the planner left the victim decoding
against a slot with no KV:

```
Qwen3.5 materialized state len 0 != DecodeRow.kv_seq_len 1493 for slot 0
```

which unwinds the whole worker group. Capture is now read-only;
`release_swapped_out` frees the state after the insert is confirmed on every
rank. DSv4 had the identical shape and got the same split.

**A refusal counter.** `kv_tier_slot_demote_failures_total` sits next to the
success counters. It paid for itself in the first run after: the tier-off arm
reports 43 refusals as a number instead of a WARN.

## Result

TP=2 CP=2, ThinkingCap-Qwen3.6-27B-FP8, 4×H20, `--mem-fraction-static 0.25
--kv-dram 32GiB --kv-disk --kv-oversubscription --max-running-requests 4`,
6 sessions × ~1500-token prefix × 2 turns × 256 generated tokens.

| Quantity | Before | After |
|---|---|---|
| Park attempts refused | 9,970 / 9,970 | 0 |
| `demoted_slots` / `promoted_slots` | 0 / 0 | 390 / 390 |
| `slot_demote_failures` / `slot_promote_failures` | not counted | 0 / 0 |
| `fallback_recompute` | every preemption | 0 |
| `reuse_hit_host_demoted` | 0 | 2,210 |
| Needle @ depth 50, conc 16 × 3 rounds | not reachable | 48/48 exact |

Promote of a 10,130-token slot: **125–130 ms**, consistent across all four
ranks (n=390).

`kv_tier_io_useful_write_bytes` stayed 0 — the whole working set fit in the
32 GiB L2, so L3 was never reached. Correct behaviour, not a gap.

Graceful degradation with `--kv-dram 0`: 43 refusals, 0 recomputes, needle
still 48/48, wall clock within 0.2% of the tier-on arm (65.5 s vs 65.4 s in a
regime where neither arm had to rotate).

## Pending

The wall-clock A/B in a regime that genuinely forces rotation is **not
measured**. Two attempts produced null results:

- `--max-running-requests 8` with 8 sessions: nothing had to rotate, 0 parks
  in both arms.
- `--max-running-requests 6` with 16 sessions: another party's TP=4 serve took
  GPUs 4-7 mid-run, leaving 11,021 MB free against ab2's 81,427 MB. The KV
  budget clamped `num_slots` to 1 and `max_prompt_tokens` to 2048, so every
  8,000-token prompt aborted. Contention artifact, not a code result.

So the promote cost is measured (130 ms) but the counterfactual — a full
re-prefill of the same context — is not measured in the same run. Rerun on a
clean box.

## Rule

A run's own log carries the free-VRAM figure it was budgeted from
(`qwen35_forward.rs:259` "KV budget: free NMB"). Read it before believing a
null result on a shared box: an 8× drop in free memory silently clamps
`num_slots` and `max_prompt_tokens`, and the workload then aborts instead of
exercising anything.
