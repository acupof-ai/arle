# DSv4 FlashMLA-decode default-on — adopt the vendored SGLang decode-attention kernel (+24%)

**Date:** 2026-06-06. **Backend:** CUDA, DSv4-Flash FP8 TP=8/EP=8, 8×H20.
**Status:** flip landed (default-on, opt-out `ARLE_DSV4_FLASHMLA_DECODE=0`),
**licensed by a same-load resident A/B**. Task #16 / wholesale-adopt plan step 1.

## Context

FlashMLA SM90 sparse decode (`arle_flashmla_sm90_sparse_decode_fwd`) is the *same
vendored kernel SGLang uses* for DSv4 decode attention — already wired in ARLE
(`try_flashmla_decode_attention`) but gated off behind `ARLE_DSV4_FLASHMLA_DECODE`,
so the default decode ran the scalar `dsv4_hybrid_attention` path. `mla_attn` is the
largest decode kernel bucket (~23% wall). Adopting it = flip the gate, not write a
kernel ("先用最好的不要闭门造车").

## What Worked

- **Resident same-load A/B** (`dsv4_resident_ab`, one 149 GB load, both legs via the
  `DSV4_FLASHMLA_DECODE_OVERRIDE` atomic — no per-config cold-load): 64-token decode,
  **scalar 29.47 tok/s → FlashMLA 36.59 tok/s = +24.2%**, and **token-exact** (FlashMLA
  output bit-identical to scalar at the deterministic 64-tok shape — the correctness
  gate; cf. [[reference_dsv4_moe_nondeterminism_confounds_4096_parity]] for why we
  verify at the deterministic short shape, not 4096-token-exact).
- **Default flip is allocation-safe:** `dsv4_flashmla_decode_alloc_enabled()` falls
  through to `dsv4_flashmla_decode_enabled()` (attention.rs:1110), so flipping the one
  gate makes the FlashMLA arena allocate at load under the default — no
  "enabled but no arena" crash.
- Kept the opt-out env + the override atomic (resident A/B harness path intact).

## Honest read

- Verified on the **real production SKU** (TP=8/EP=8, 8×H20) at the deterministic
  64-tok shape, token-exact + +24%. Per ckl's "已经有基线了直接测优化后的" — the
  optimized config was measured directly against the known scalar baseline; no
  baseline re-run.
- Decode-graph (default-off, +1.5% wall) is superseded by this on the wall-clock
  axis; the `dsv4.rs:656` gate already makes them mutually exclusive (graph only when
  FlashMLA-decode is off), so no conflict.

## Rule

Adopting a vendored kernel that's already wired = flip the gate + license with a
same-load resident A/B (token-exact at the deterministic shape + tok/s Δ), not a
cold-load-per-config sweep. When flipping a runtime gate default-on, check every
*sibling* gate it feeds (here the `_alloc_` gate) flips with it, or the default
path hits an "enabled but unallocated" panic.
