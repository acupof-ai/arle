# DSv4 extension-prompt prefix reuse — frontier no longer clobbers the chunk-end boundary (#166)

## Context

#166 escalated twice past its filed surface. Layer 1: the `prefix_reuse` gate
read a nonexistent stats key (`prefix_hit_tokens`; real key
`prefix_cache_hit_tokens`) and `stat_delta` silently returned 0 (f6cd2ca3a +
fail-loud d9765b7cf). Layer 2 (the real engine bug, instrumented round-2 pod
probes, `dsv4.rs:1191`): with `--dsv4-decode-reuse` ON, the finish
write-through ALWAYS recaptured the frontier page, replacing the prefill
chunk-end's tail-less boundary entry (carry at the aligned page end, licenses
ANY continuation) with a finish-frontier entry (carry at finish_len, licenses
only an exact tail continuation). For prompts shorter than one prefill chunk
(2048) that was the only boundary page → every diverging suffix — the
canonical multi-turn shape — licensed 0 blocks (`raw_blocks=26,
licensed_blocks=0`), and later finishes on shared pages retroactively
destroyed previously-hitting shapes (republish clears the frontier tail).

## What Worked

Fix b6f34a564: `capture_finish_frontier` skips the frontier recapture when the
page already holds a tail-less boundary entry — the finish forfeits only the
< page_tokens sub-page tail (exact repeats floor at the aligned boundary).
Round-3 pod battery (8×H20, DSv4-Flash-FP8 TP=4, b6f34a564):

- Probes: EXT + EXT-REPEAT license 26 blocks (was 0), REPEAT2 keeps licensing
  (no retroactive destruction), zero boundary-downgrade republishes logged.
- Gates: `prefix_reuse` 2000 **and** 2003 PASS (reuse_hit 1792t/28p ×3);
  restore output == full-recompute output (matched A/B, 3 salts); needle
  27/27 exact.
- Bench (round 4, champion fingerprint `bench-prompts-64.jsonl` 90 s/point,
  raw `bench-output/2026-07-17-r4-prefix-fix/` on the pod):

c32 TTFT p50 3680 vs champion 4519 ms (−18.6%).

Wash, licensed. Champion row unchanged.
Protocol note: a bare `1,32` grid made c32 look regressed — the champion c32 was
measured warm at the end of a `1,4,16,32` sweep; fingerprint includes the
grid order.

## Rule

A frontier tail must EXTEND reuse, never GATE it: an entry whose only carry
sits at finish_len turns every diverging continuation into a licensing zero.
Keep the aligned tail-less boundary as the durable commit point; and when a
reuse metric reads zero, instrument the decision path (raw → licensed →
attached) before theorizing — every zeroing stage was silent here.
