# hc_enter fusion KILLED by matched e2e pair (−4.9%) despite −29%/inst microbench — SMEM-carveout switching is the prime suspect

**Date:** 2026-06-11 (dawn). Matched back-to-back pair, md5-verified pod
files, co-tenant-clean both arms.

## Verdict

| arm | B=1 p50 | note |
|---|---|---|
| B' (pair: params@1024 + prologue@1024, d7be8c9b) | 41.63 | |
| C' (hc_enter one-kernel, d457ad1b) | **39.61 (−4.9%)** | output correct |

Component microbench said hc_enter WINS (10.47µs vs pair 13.1+gap ≈
14.7µs, −29%). The e2e wall says it LOSES by ~+1.2ms/token. Reverted
(`git revert d457ad1b`); the kernel remains in history for Rung 2.

## Prime suspect (hypothesis, licensed for ONE cheap probe before Rung 2)

hc_enter uses **40KB dynamic shared memory** (stages the whole stream row);
the kernels around it (FlashMLA, DeepGEMM, the 14KB-smem prologue) run at
default L1/SMEM carveout. Alternating launches with very different smem
footprints can force **per-launch carveout reconfiguration** (~µs each,
×172 transitions/token ≈ the missing ~1.2ms) — invisible to the microbench
(same kernel back-to-back keeps one carveout). Fix shape if confirmed:
stage only the mixed row (8KB) and re-read the stream from global like the
pair does.

## Rules

- **A microbench measures the kernel; the wall measures the NEIGHBORHOOD.**
  Same-kernel-in-a-loop timing misses launch-context effects (carveout
  switches, L2 displacement). Component license is necessary, not
  sufficient — the matched e2e pair is the final gate, in BOTH directions.
- **Big-SMEM fused kernels are not free**: Rung-2 segment kernels must keep
  dynamic smem within the default carveout (≤~48KB... ideally ≤16KB) or
  pin a uniform carveout across the chain.
- Cross-session drift hit ±6% again (same config: 44.04 → 41.63 three hours
  apart). Matched pairs or nothing, both directions of the verdict.
- **Pod tree got externally reset mid-campaign** (to 82ea0ef6; the "arm C"
  39.46 was actually the 256/256 state — fake arm). md5-verify the pod
  files against local HEAD before EVERY build (pod-probe-trust rule, now
  with teeth).

## State

Landed Rung-1 (all matched-pair-verified): mhc_params warp tail + fused
hc_pre+rms_norm + both@1024 = **+9.3% over the 256 pair; campaign
39.51 → ~44 (best clean session)**. Next: pack_quantize epilogue fusion /
splitK / Rung 2 — each with the carveout lesson applied.
