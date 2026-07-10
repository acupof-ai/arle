# DSv4 finish-write-through decode-region reuse — `--dsv4-decode-reuse`

> Status: pending-remote (opt-in flag; needs the pod correctness+perf license).

## Context

Landed the finish-write-through decode-region reuse plan
([2026-07-10-dsv4-finish-writethrough-reuse.md](../../plans/2026-07-10-dsv4-finish-writethrough-reuse.md))
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

## Pending (remote / pod)

- `needle_gate.py` x3 same-config vs baseline envelope, TP=4 DSv4-Flash-FP8.
- W1 multi-turn (`prefix_reuse_gate.py`): P + R + follow-up → turn-2 restores to
  `finish_len` (whole prior turn), reuse length ≈ `finish_len` not the prompt
  floor; needle exact.
- Graph lane: single-GPU decode graph ON + a reuse turn → decode graph captures
  (no eager fallback) AND reuse works.
- Perf: restore vs cold-prefill of the reused span; Δ% vs baseline.

## Rule

Finish write-through captures the carry LIVE at `finish_len` (never rebuilt) —
dodges the replay-tail stale-`prev_overlap` KILL. Content divergence within the
sub-page tail `[matched_len, finish_len)` is trusted (agentic turns extend
verbatim); the pod needle gate is the correctness license before any default flip.
