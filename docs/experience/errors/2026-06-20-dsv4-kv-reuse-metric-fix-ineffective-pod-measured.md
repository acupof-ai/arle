# WS2 KV reuse-metric "fix" (a45f3a29) is ineffective for DSv4 — reuse is real (49×) but counters still read all-miss

## Context

Pod TP4 (GPUs 0-3) serve of DSv4-Flash at 5af252bc, port 18234, no-spec. Sent
the SAME ~1.2k-token prefix twice and timed it + read `/v1/stats kv_system`:

```
REUSE_TIMING cold=22.470s warm=0.455s speedup=49.43x
KV: reuse_hit_resident=0  prefix_match_full_blocks=0  resident_pages=0  reuse_miss=7
```

Prefix reuse is unmistakably WORKING (49× — the warm request skips prefill). But
the corrected counters still read all-miss. The L1/L2/L3 → self-documenting
RENAME (a45f3a29) is correct and live in `/v1/stats` (names confirmed); the
reuse-metric MIS-CLASSIFICATION is NOT fixed for DSv4.

## Root Cause (measured, not yet code-traced)

a45f3a29 hooked `record_attached_prefix_metrics` (infer-core/src/lib.rs:957) on
the generic admit/attach path, incrementing `reuse_hit_resident` only when
`reused_prefix_pages.len() >= 1`, and wired `resident_pages` to the
`HostPagedKvPool` occupancy. DSv4's 49× reuse demonstrably takes a path that
leaves `reused_prefix_pages` empty at that hook, and `resident_pages` reads a
pool DSv4 does not populate (DSv4 uses `Dsv4KvAdapter` per-slot KV, not
`HostPagedKvPool`). So the hook fires `reuse_miss` every request despite the
real prefix-cache hit. The agent's fix was callgraph-inference-based and never
pod-verified; measurement refutes it.

## Fix

RE-OPEN WS2. Find the ACTUAL DSv4 prefix-reuse decision (the RadixCache token-
match that yields the 49× prefill skip) and increment the reuse-hit +
`prefix_match_full_blocks` counters THERE; wire `resident_pages` to DSv4's real
resident KV occupancy (the `Dsv4KvAdapter` / slot pool), not `HostPagedKvPool`.
Re-verify on the pod with the same `reuse_timing.py` until a 49× warm request
shows `reuse_hit_resident >= 1` + `prefix_match_full_blocks > 0` +
`resident_pages > 0`. The RENAME stays; only the metric wiring is wrong.

## Rule

Pod-VERIFY a metric fix by MEASURING both the underlying behavior (here: the
49× reuse timing) AND the counter side-by-side — never claim a metric fix from
callgraph inference. A counter that reads "all-miss" while the behavior it
counts is 49× live is the loudest possible "your hook is on the wrong path"
signal; the in-process unit tests (58 green) passed precisely because they
exercised the wrong path too. §0: measurement is evidence; the admit/attach
callgraph was hypothesis.
