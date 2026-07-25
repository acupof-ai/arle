# DSv4 finish-write-through decode-region reuse — `--dsv4-decode-reuse`

> Status: pod-verified opt-in (crash-fix gate PASS, `28b8cd7bb`, binary
> `b1d9f968…`, TP=8 H20). OFF byte-identical; ON no-crash + reuse engages.
> Default flip still pending a token-id-preserving perf harness (below).

## Context

Landed the finish-write-through decode-region reuse plan
(2026-07-10-dsv4-finish-writethrough-reuse.md)
behind `--dsv4-decode-reuse` (default OFF → byte-identical). On a request finish,
the slot's full frontier state (generated-region content + the live carry —
overlap/idx_overlap/ring/pending — captured at the exact finish position
`finish_len`) is written THROUGH to the content-keyed `Dsv4PrefixStatePool` at the
finish sync point (graph-safe). A later agentic turn restores to `finish_len`
(radix-aligned `matched_len` + the sub-page `tail_len` carried on the frontier
entry) and prefills only the new suffix.

## What Worked (local gates only — perf is pod-gated)

- Seam widened to `restore_prefix_sidecar -> Result<usize>` (extra tokens beyond
  `matched_len`, default 0) + new `capture_finish_frontier` seam. Every non-DSv4
  backend returns 0 / no-op → byte-identical (`cargo test -p infer-core`, 92
  passed).
- Decode-lane per-tick publish is a no-op under the flag (the D2H+sync was the
  CUDA-graph trap); the whole generated region is captured once at finish.
- Frontier tail = `staging` tail rows `[matched_len/ratio, finish_len/ratio)` +
  DSA tail + sub-`ratio` `pending` (`finish_len % ratio` tokens; NO kernel
  change — the forward derives `pending_len` from `start_pos`,
  `attention.rs:7638`), anchored on the last radix page's entry (the sub-page
  tail has no radix id of its own).
- Gates: `cargo clippy -p infer-cuda … -D warnings` clean; `cargo check -p
  infer-api …` clean; `cargo test -p infer-core` green; `cargo fmt` clean.

## Pod verification (TP=8 H20, DSv4-Flash-FP8, `--dsv4-decode-reuse true`)

Two rounds. v1 (`79b5dbb17`): mechanism engaged (multi-turn match 640→704, +1
page into the decode region) but the ON path CRASHED the TP serve — `pool
seq_len 494 != append_pos 485`: a shorter request restored a prior turn's
sub-page tail (the tail has no radix content identity;
[errors](../errors/2026-07-10-dsv4-finish-writethrough-tail-content-identity.md)).
v2 (`28b8cd7bb`) added the continuation guard and re-verified:

| Lane | Result |
|---|---|
| Baseline OFF (needle_gate 446–512 ×3) | 15/15 exact DET — byte-identical |
| **Crash-repro ON** (24 alternating pt499/pt485) | **24/24 exact, serve alive, ZERO `seq_len != append_pos`, zero panic** — 23 hits / 8832 hit_tokens / 7 write-through pages. v2 fixed the v1 crash. |
| Multi-turn ON (637-tok prompt) | no crash, guard does NOT over-restrict — write-through published **10 pages** (past the 9-page/576-tok prefill floor, into the decode region); fresh 640-tok needle 738291 exact 3/3 DET |
| Repeat-storm ON (8-concurrent ×2, the v1-crashing lane) | no crash, serve alive; a `738231` flip attributed to pre-existing concurrent-MoE non-determinism by OFF/ON A/B (OFF 8/16 vs ON 7/16, identical batch pattern), NOT decode-reuse |

## Still pending (default flip)

- **Perf Δ%**: the token-id-preserving multi-turn driver is now BUILT
  (`3461a37c8`: server accepts a token-id `prompt` array + returns
  `prompt_token_ids`; `scripts/eval_harness/token_reuse.py` replays turn-1's
  exact `prompt_ids + generated_ids` into turn-2). The clean OFF-vs-ON delta run
  is **blocked on infra, not the feature**: the pod's GPU 1 is pinned by foreign
  jobs (~13 GB free; DSv4 TP=8 needs ~37 GB/rank), so the serve can't boot.
  Harness + serve recipe staged on the pod (`reuse_measure.sh`); rerun when
  GPU 1 frees ≥40 GB through the ~25s weight-load window. Mechanism already
  confirmed twice: v1 measured 640→704 (+1 page), v2 published 10 pages into the
  decode region (past the 9-page prefill floor).
- Graph lane is code-structural (no DSv4 decode graph under TP/MoE): the
  decode-lane publish is a no-op under the flag, so no per-step D2H is added —
  graph-safe by construction, not runnable against a DSv4 graph today.

## Rule

Finish write-through captures the carry LIVE at `finish_len` (never rebuilt) —
dodges the replay-tail stale-`prev_overlap` KILL. The sub-page tail
`[matched_len, finish_len)` has NO radix content identity, so it is reused ONLY
for a VERIFIED continuation (`prompt[matched_len..finish_len] == entry.tail_tokens`)
— trusting it verbatim was the v1 crash. Default flip still needs the perf Δ%.
