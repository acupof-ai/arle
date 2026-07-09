# DSv4 Route A FlashMLA-lane needle regression — bisected to origin + amplifier

## Context

Discovered while pod-verifying the #150 `ARLE_DSV4_PROJ_BATCHED_BF16` lever
(2026-07-09): the A/B was unreadable because HEAD's own baseline had
collapsed — solo n=1 needle miss 98.3% (59/60) on the numeric-salt prompt
shape, ~82% at n=2 on both arms. Attributed the same day via a
discriminator matrix + commit bisection on the pod (TP=4, GPUs 4/5/6/7,
`concurrent_needle_v3.py` rebuilt-harness pt=462 solo lane, baseline output
`'The secret access code stated earlier is **738291**.'`, 943bacda = 8/8
byte-identical).

## Root Cause

**Two-tier structure inside the FlashMLA decode lane — origin + amplifier,
not independent bugs** (signatures distinguish them):

| Verdict | Commit(s) | Evidence |
|---|---|---|
| GOOD baseline | `943bacda`, `714643e2a`, `55a74d870` | 48/48 exact, zero wording drift |
| **Origin** | `0198c3ba7` (reset per-layer compressor/dsa_official row counters on Route A restore) | digit-substitution class appears (`738292`/`738249`), 4/50 miss + determinism loss (wording drift, upstream had zero); boundary clean: 0/63 upstream vs 5/75 at/after |
| **Amplifier** | Route A page-sharing series `1354650f7`→`92e02e081`, effective at `3b0d7eda1` (middle commits don't build — series is one unit) | truncation class appears (`'7382.'`/`'738.'`, zero instances before), miss 4-8% → 27% |
| Worst point | `e05a467e6` (pad FlashMLA page table to full lsp) | 47% miss (7/15) |
| Not fixed at HEAD | `5fbcbbac7` | 11/16 miss variants persist |
| **Lane proof** | HEAD + `ARLE_DSV4_FLASHMLA_DECODE=0` | **15/15 exact** — whole failure surface is FlashMLA-decode-lane-only |

`ARLE_DISABLE_PREFIX_CACHE=1` does NOT mitigate at HEAD — the amplifier
mechanism does not require a prefix hit (mechanism-level contradiction with
`0198c3ba7` nominally touching only the restore path is unresolved at
commit-level attribution; left to the fix's own verification).

Mechanism (source-level, matching the measured signatures):
- Amplifier: `e05a467e6` pads host band mirrors with repeated-last-page
  aliases while the per-(slot,layer) device page table is synced "exactly
  once per slot lifetime" (`dsv4.rs:1165-1169`) — decode growth leaves the
  device table stale-aliased, FlashMLA reads/writes band pages ≥ n onto the
  last real page, clobbering prompt-tail KV (where the needle lives). The
  pre-padding "host N vs device 18" `ensure!` + full-recompute fallback had
  been masking this by accident.
- Origin: `0198c3ba7` sets restored counters (`compressed.seq_len`,
  `packed_rows`) without rebuilding the per-slot staging content those
  counters describe — the loud `ensure!` crash it fixed became silent
  wrong-data reads perturbing near-tie logits.

## Fix

Not in this entry — in flight on two tracks: (a) another session's fix
series (`be123bcc`/`1b993da1`, "prefix reuse tail prefill" naming — note it
may only cover the cache-on path; acceptance must use this entry's pt=462
lane WITH `ARLE_DISABLE_PREFIX_CACHE=1` as well); (b) the structural
B→C dirty-bit contract in
[docs/plans/2026-07-09-dsv4-kv-reuse-seam-refactor.md](../../plans/2026-07-09-dsv4-kv-reuse-seam-refactor.md)
Phase 0. Immediate production mitigation: `ARLE_DSV4_FLASHMLA_DECODE=0`
(correct, slower).

## Rule

- **A "fix" that silences an `ensure!` must restore/rebuild every piece of
  state the invariant was guarding** — setting the counter without the
  content converts a loud crash into silent corruption (§0.1 full-enumeration
  discipline; same class as the DSv4 EAGLE rollback anchor).
- **An error-then-fallback path can be load-bearing for correctness** —
  `e05a467e6` removed a size-mismatch error whose fallback (full recompute)
  was the only thing keeping stale device tables from being consumed. Before
  deleting an error branch, prove the happy path holds the invariant the
  error was catching.
- **Bisect discipline that made this attribution stick**: independent pod
  tree + per-run binary sha256 (a shared tree got reset mid-bisect by a
  parallel session, voiding one arm); alternating trial-id shapes exposed
  that "first-3-requests-clean" was a prompt-length (462 vs 461 token)
  artifact, not warm-state decay; non-building middle commits collapse a
  series into one attributable unit — report it as such, don't guess.
- The pt=461 lane (`'...**738**. (The full code was '` budget-truncation) is
  fragile even on the pre-window baseline (5/7 miss at 943bacda) — it is a
  model/prompt-shape behavior, NOT a regression indicator; gate on pt=462.
