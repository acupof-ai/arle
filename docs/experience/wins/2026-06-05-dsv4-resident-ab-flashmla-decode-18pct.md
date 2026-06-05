# DSv4-Flash decode: resident A/B harness → FlashMLA +18% same-load (occupancy still pending)

**Date:** 2026-06-05. **Backend:** CUDA, DSv4-Flash FP8 TP=8/EP=8, 8×H20.
**Status:** harness landed; same-load decode A/B is SOLID (+18%), occupancy ncu
still gated. Supersedes the cross-run +9% smoke in
[`2026-06-05-dsv4-flashmla-decode-wireup.md`](2026-06-05-dsv4-flashmla-decode-wireup.md).

## Context

The FlashMLA wire-up's perf claim was velocity-blocked: every scalar-vs-FlashMLA
A/B reloaded the 149 GB DSv4 model (~110 s), so we only had a cross-run smoke
(27.8 vs 25.5 tok/s, different loads — not a matched A/B per
[[feedback_matched_ab_for_small_bench_effects]]). The fix is a **resident A/B
harness**: load the TP=8/EP=8 executor once, then flip FlashMLA on/off in the
same process via a new process-local override, and time each variant's
steady-state decode after a warmup window.

## What Worked

`crates/infer-cuda/examples/dsv4_resident_ab.rs` (+ `scripts/dsv4_resident_ab.sh`)
drives N decode variants after one load. The enabler in `attention.rs`:
- `set_dsv4_flashmla_decode_override(Option<bool>)` — a process-local `AtomicI8`
  that overrides the `ARLE_DSV4_FLASHMLA_DECODE` env gate (`None` restores env),
  so dispatch flips without a restart.
- `dsv4_flashmla_decode_alloc_enabled()` (new `ARLE_DSV4_FLASHMLA_DECODE_ALLOC`
  env) — allocates the FP8-KV arena up front for *both* variants, so the
  FlashMLA path is reachable after load even when dispatch started scalar.
  Falls through to the decode gate when unset → **production behavior unchanged**.

Both gates are inert by default (override = env, alloc = decode gate), so the
default serving path is byte-identical.

## Results (same load, B=1, 128-token decode, warmup=16 excluded)

| variant  | steady tok/s | decode tok/s | oracle16 | vs scalar bf16 |
|----------|-------------:|-------------:|----------|----------------|
| scalar   | 23.670       | 23.840       | PASS     | reference      |
| flashmla | **27.940**   | 27.914       | PASS     | **+18.0%**     |

- **+18.0% same-load** is the headline — a matched A/B (one load, two dispatch
  flips), not the earlier cross-run +9% smoke.
- **`prefill_ms` is NOT a valid A/B here** (scalar 5548 ms cold-first vs flashmla
  66 ms warm-second): the first variant pays CUDA context init + first-kernel JIT;
  decode steady is warmup-excluded per-variant so it is order-robust, prefill is
  not. Cited decode only.
- **`scalar_ref=DIFF@122`**: FlashMLA (FP8) matches scalar (bf16) for 121 decode
  tokens then diverges at 122/128 — deep FP8-vs-bf16 drift at depth, both
  `oracle16=PASS`. Not a correctness bug; flagged for the KV-precision-parity gate.

## Honest read / not-done

- **Single same-load run per variant** — the harness now makes repeats cheap
  (seconds after load); an order-swap control (flashmla-first) + 3-run variance
  are the next step before any default flip.
- **Occupancy still pending.** ncu caught the FlashMLA main kernel
  (`flash_fwd_splitkv_mla_fp8_sparse_kernel`, grid **78 CTA/rank** vs the scalar
  tiny-grid) but `--set speed-of-light` returned *"No metrics to collect"* — only
  launch/device-attr metrics expanded, no SM%/achieved_occupancy. The 78-CTA grid
  is the one SOLID datapoint (structurally larger than the scalar 1-3% grid); the
  occupancy SOL proof needs an explicit-metric rerun (`pending-remote`).

## Closeout (matched A/B + wrapper breakdown + DIFF@122)

Lever #1 closed out via the harness (no reload). **Default flip still NOT
licensed** — occupancy unproven.

- **Matched A/B is solid** — both orders (scalar-first *and* flashmla-first),
  3 reps each, warmup-excluded: **scalar 23.713 ± 0.047** vs
  **flashmla 27.988 ± 0.061 tok/s = +18.03%**. The +18% is order-robust and
  low-variance, not a single-run artifact.
- **Occupancy: precond-fail (again).** The explicit-metric ncu rerun
  (`sm__throughput`, `sm__warps_active`, `gpu__time_duration`, `dram__throughput`)
  hit the FlashMLA kernel but failed with `UnknownError`
  (`dsv4_ab_ncu_flashmla_explicit.log`). The `.ncu-rep` still only proves launch
  shape (78 CTA/rank). **No SM%/occupancy claim** → the kernel isn't licensed for
  a default flip yet; needs a working ncu metric path (`pending-remote`).
- **Wrapper ops measured (NVTX per-token)** — `dsv4_flashmla_wrapper_nvtx_stats`:
  FlashMLA fwd ~0.64 ms, TP Q all-gather ~0.61 ms, Q repack ~0.23 ms, FP8 KV pack
  total ~0.56 ms. Real costs (~1.4 ms/token of wrapper around a 0.64 ms kernel),
  but **trim deferred** — it needs a new kernel ABI (local-head support) or
  pack-into-kernel fusion; not enough ROI to start before the occupancy proof.
- **DIFF@122 = precision margin, not a bug.** At position 127 scalar top1 `11111`
  (margin 0.25 over `8760`); FlashMLA flips to top1 `8760`=28.0 with `11111`=27.875
  (margin 0.125). FlashMLA (FP8) flips a tiny local margin vs scalar (bf16);
  oracle16 still PASS. Consistent with FP8-vs-bf16 precision delta, not corruption.

## Rule

A resident, load-once A/B harness is the right fix when a model reload
(149 GB / ~110 s) is the iteration bottleneck — it converts a cross-run smoke
into a matched same-load A/B and makes ncu/variance loops seconds-scale. Cite
**decode steady-state only** from such a harness: the first variant pays cold
CUDA init, so `prefill_ms` (and any non-warmup-excluded slice) is order-confounded;
the warmup-excluded decode window is the order-robust comparison.
