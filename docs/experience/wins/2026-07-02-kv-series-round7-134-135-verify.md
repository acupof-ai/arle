# Round 7: #134 verified clean; #135 needed a completion fix — group now self-tears-down in 5.8s

## Context

Round 7 on the 8×H20 pod, machine-verifying `cdaacf55` (#134 swap-in vacate +
#135 group teardown) per the round-6 handoff. Build `958536e9` + `14db14f9`,
snapshot bin `arle-r7b-bin` (cuda,nccl). Unit gates first: infer-cuda lib
(prefix_index/tier_key/chunked/chunk_manifest/oversized) 11/11, infer-core lib
(oversubscription/slot_) 8/8.

## What Worked

- **#134 VERIFIED** — Qwen3.6-27B-FP8 single-GPU, `--max-running-requests 1
  --kv-oversubscription --kv-disk --kv-disk-limit 32GiB --kv-dram 0`
  (features `prefix,park`), two simultaneous A(128)+B(64) pairs:
  `demoted_slots 14→28, promoted_slots 14→28, slot_promote_failures = 0`
  across both pairs (round 6: exactly 1 per pair), zero WARN lines in the
  whole serve log, L3 disk really written (du mid-run 348M/366M, disk_pages
  11 mid-run), both outputs coherent. promoted == demoted exactly.
- **#135 VERIFIED, but only after a completion fix.** `cdaacf55` alone
  measurably did NOT tear down the group: kill -9 rank 1 mid-decode on DSv4
  TP=4 → curl 500 in 0.26s and the `coordinator.rs:183` teardown marker
  logged, yet `/v1/models` still answered 5 minutes later and ranks 0/2/3
  spun at 100% GPU. Root cause: `ServeShutdown::request()` sets an
  AtomicBool that `shutdown_signal` (infer-api/src/serve.rs) never polled —
  it only selected on SIGINT/SIGTERM, so the HTTP loop never unwound and the
  worker guard never dropped. `14db14f9` adds a 200ms `is_requested()` poll
  arm to the select. Re-test on the fixed bin: reader fail → teardown marker
  (+1ms) → `shutdown signal received` (+172ms) → 5s guard grace → ranks
  0/2/3 SIGKILLed → RUN_EXIT=0; kill → **group fully self-exited in ~5.8s**,
  GPUs 4-7 back to 0 MiB, zero manual kills.

## Problems

- **DSv4-Flash forward NaNs from position 4** on `958536e9` (TP=4, tier
  flags irrelevant — control without `--kv-disk` reproduces). Decoded via
  `--probe-out`: prefill entropy sane for pos 0-3, NaN from pos 4 (= first
  position consuming a compress-ratio-4 chunk); every decode step then
  samples token id 0 (bos, special) → empty text. Worktree bisect
  **exonerated the round-7 window**: `a25922b9^=16a95fe0` AND the round-6
  build `5cafb308` itself reproduce it — pre-existing, so round-6's "DSv4
  regression completion clean" cannot have checked visible text on this
  shape. See the errors entry of the same date.
- Dense-arm coverage gap: radix page-tier `demoted_pages/promoted_pages`
  cannot be pressured on a 97GB card with a 0.6B model — pool =
  `max(profiled, floor)` with `mem_fraction_static` clamped ≥0.5 (~47GB
  floor). Only the recall write-through moved (`host_demoted_pages` 718).
- Durable KV recall cannot survive a restart by construction:
  `durable_namespace` embeds `std::process::id()` (kv_tier.rs:514), so
  `tier.load()` (executor.rs:4075) can never see a previous boot's store;
  in this round the durable lane never even attached (no `arle-kv-recall-*`
  dir created despite `--kv-recall --kv-disk`).

## Rule

A teardown fix isn't verified by its own log line — verify the *effect*
(processes gone, GPUs 0 MiB, no manual kill), not the intent marker. The
round-7 kill-test passed the marker and still left 3 workers spinning.
