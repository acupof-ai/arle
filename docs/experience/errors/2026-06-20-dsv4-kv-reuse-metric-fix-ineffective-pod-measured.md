# RESOLVED (NOT A BUG): DSv4 has NO cross-request prefix reuse — the all-miss counter is HONEST; the "49×/14× reuse" was JIT/CUDA-graph warmup

## RESOLUTION (2026-06-20, warm-JIT controlled probe — supersedes everything below)

A clean probe with the DeepGEMM JIT + CUDA-graph cache fully warm
(`A_cold=4.00s`, not the earlier 22s) and 5 DISTINCT same-length prefixes:

```
A_cold=4.00  A_warm=0.30  B=0.30  C=0.32  D=0.39  E=1.55  A_after_BCDE=0.30  B_after=0.30
```

`A_warm` (prefix identical to A) = `B`/`C`/`D` (different prefixes) = ~0.30s.
**If prefix reuse existed, identical would beat different. They are EQUAL ⇒ no
reuse.** DSv4 re-prefills every request at ~0.3s warm; the cold/warm "speedups"
(22.5s→0.46s, 5.64s→0.40s) were entirely DeepGEMM-JIT + CUDA-graph *warmup*
amortization, NOT KV reuse. The re-fix agent's code trace was correct:
`reusable_prefix_blocks=0` (infer-cuda/src/executor.rs:302), radix off, no
cross-request KV reuse — by design (DSv4 owns its MLA KV inside the forward).

**Verdict: the `kv_system` all-miss reading is CORRECT. WS2 is closed — there is
no metric bug; there is no reuse to count.** The L1/L2/L3→self-documenting rename
(a45f3a29) stays. Separately, "DSv4 re-prefills every request" is a *feature gap*
(no cross-request prefix cache) — adding one is a substantial correctness feature,
not a monitoring fix.

**Lesson (the load-bearing one): JIT/CUDA-graph warmup state is a massive
confound for ANY cold/warm timing on DSv4.** I flip-flopped THREE times (49×
reuse → no reuse → 14× reuse → no reuse) because each 2-point cold/warm probe sat
at a different warmup state. The ONLY clean reuse test is identical-vs-different
prefix at FULLY-warm JIT (`A_warm` vs `B_diff`); there they're equal. §0: a
measurement at an uncontrolled confound is not evidence — isolate warmup first.

---

## (Superseded) Original framing — kept for the record

WS2 KV reuse-metric "fix" (a45f3a29) is ineffective for DSv4 — reuse is real (~14×, after separating ~17s one-time warmup) but counters still read all-miss

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

## Root Cause — REFINED by a 3-prefix control (2026-06-20)

The first "49×" conflated two effects. A decisive control — prefix A twice, then
a DIFFERENT same-length prefix B — separates them:

```
A_cold=22.90s  A_warm=0.40s  B_diff=5.64s
```

- **One-time warmup ≈ 17s** = A_cold − B_diff (DeepGEMM JIT + CUDA-graph capture
  for the prefill shape; paid once, not per request).
- **Real prefix reuse ≈ 14×** = B_diff (5.64s, different prefix, full prefill) vs
  A_warm (0.40s, identical prefix, prefill SKIPPED). An identical prefix
  genuinely skips the ~5.6s prefill — reuse EXISTS.

So both my first claim ("49× reuse") and the re-fix agent's counter-claim
("zero reuse, all warmup") were partly wrong. The agent correctly traced that the
RADIX path is off for DSv4 (`reusable_prefix_blocks` returns 0 at
`infer-cuda/src/executor.rs:302`, radix trie never populated) — but wrongly
concluded "re-prefills every request", which A_warm=0.40s refutes. DSv4 reuses
via a NON-radix path (slot-level: the slot that served A_cold still holds its KV;
an identical next prompt matches and skips prefill). a45f3a29 (radix attach) and
the agent's swap-restore counter BOTH hook the wrong path → `reuse_miss` every
request despite the real ~14× hit. `resident_pages=0` is a sampling artifact
(`/v1/stats` read after the slot freed).

## Fix

WS2 STILL OPEN. The counter must hook the SLOT-LEVEL prefix-match (the path that
makes A_warm skip prefill), not radix-attach nor swap-restore. Find where the
DSv4 decode/admit path detects "this prompt's prefix matches the slot's resident
KV → skip prefill" and increment reuse_hit_resident + prefix_match_full_blocks
there. Re-verify with the 3-prefix control: A_warm must show the hit, B_diff must
NOT (it's a genuine miss). DEFERRED as lower priority: reuse is functional (~14×),
the bug is cosmetic (monitoring under-reports); the perf headline (MTP) took
priority. The RENAME stays correct.

## Rule

A single cold-vs-warm ratio conflates one-time warmup with per-request reuse —
add a DIFFERENT-input control (B_diff) to separate them before attributing the
speedup. Here it converted "49× reuse" / "0× reuse (all warmup)" into the true
"~17s one-time warmup + ~14× genuine reuse." §0: even a measurement needs the
right control; a 2-point cold/warm comparison is a confounded measurement, not
ground truth.
