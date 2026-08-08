# The agentic writeback drops most turn records, and its mitigation is inert under CP

**Date:** 2026-08-08 · **Pod:** 8×H20 GPUs 4–7, ThinkingCap-Qwen3.6-27B-FP8,
cp=4 × G=2, `fulltrain11` (449-task production run, tree `5b1cd473d`)

## Context

Read off the running production run at 76/449 groups (5.5 h, `fulltrain11.log`
and `fulltrain11-rounds.jsonl`):

| Quantity | Value |
|---|---:|
| samples rolled | 304 (90 passing, 29.6%) |
| turn records dropped by `max_update_seq 23000` | 2084 |
| dropped record length | min 23003, median 27570, max 48485 |
| turn records trained | 299 |
| supervised tokens trained | 37913 (127 per record) |
| prompt tokens generated | 75.1 M |
| completion tokens generated | 0.97 M |
| update wall | 7542 s = 38% of run wall |
| update GPU-busy share | 92.3 s of 178 s median = 52% |

## Phenomenon

**A supervised span of 127 tokens sits inside a sequence of 23000–48485.** The
cc harness accumulates the agent's context into every turn's prompt, so a turn
record's length is set by the session context, not by what is being trained on.
The writeback forward captures the whole prefix and backpropagates through a
0.5% suffix.

`max_update_seq` therefore drops records **by session depth**: the cap is
crossed after a handful of turns regardless of the response, so training is
concentrated on early turns of sessions that stayed short. 2084 records dropped
against 299 trained.

The pass share of the dropped set is not recorded — the drop line carried only
the length. It now carries prompt length, reward, and supervised-token count, so
the next run measures the lost signal directly instead of bounding it.

## Root Cause

The cap is a real VRAM wall, not an arbitrary knob: the writeback peak is set by
the forward prefix-capture transient of the 48 GDN linear-attention layers
([2026-07-25 entry](2026-07-25-s1a-frozen-prefix-kv-bf16-no-vram-win-kill.md)).
Raising the cap OOMs.

The mechanism that removes the wall already exists —
`--writeback-frozen-prompt-kv` forwards only the gen segment
(`crates/train/src/opd.rs:3115`, `:3393`) — but it is **gated off under CP**:

```rust
let frozen = writeback_frozen_prompt_kv() && prompt_len > 1 && !cp.is_enabled();
```

cp=4 is the accepted production config
([fleet entry](../wins/2026-08-07-agent-opd-rollout-fleet.md)), so the flag is a
silent no-op on every production run. The two features are mutually exclusive
today: CP shards the full sequence, the frozen path forwards a single-rank gen
segment.

## Fix

Not applied to the running job. Recorded as the next writeback lever, in the
order the evidence supports:

1. Make the drop measurable (done — the log line now carries reward and
   supervised length).
2. Decide the CP/frozen-prefix interaction. Two paths, different costs: shard
   the gen segment across CP ranks (small segment, ~127 supervised tokens, so
   sharding buys little and the collective cost may dominate), or run the
   writeback at cp=1 and keep CP for the rollout fleet only. The second is
   plainly cheaper and needs no new kernel work; it needs a measurement of the
   cp=1 writeback wall against the current 178 s median.

## Rule

In an agentic lane the trained span and the sequence length are unrelated
quantities — a length cap is a filter on session depth, so read it as a
selection bias on the training set, not as a throughput knob. And a flag that
exists to lift that cap must be checked against the config actually shipped:
`&& !cp.is_enabled()` made the mitigation inert on every production run without
any error surfacing.
