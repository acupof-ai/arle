# KV T2 Disk Spill on kv-native-sys (#83) — store host-verified, CUDA pending-remote

## Goal

Give the T1 tier (#82) an opt-in disk level (#83): when the host-RAM map
fills, the coldest entry spills to a fingerprint-named block file under
`--kv-ssd-path`, so demoted-prefix capacity becomes T1 + T2 instead of
rejecting. SSD stays **opt-in** per ckl 2026-06-11 (T1 is the default-on
level).

## Hypothesis

The engine never needs to know about levels: `kv_tier_capacity_pages`
reports T1+T2 and the store places/spills internally, so #82's engine
machinery (demote/promote/rotate) works unchanged. `kv-native-sys`
(`write_block_atomic`/`read_block`/`block_path`) is sufficient as the disk
substrate — this is its **first consumer** since the rewrite orphaned it.

## Params

- `crates/infer-cuda/src/kv_tier.rs` (not cuda-gated — pure host,
  CPU-testable): two-level `CudaKvTierStore`; T1 = touch-stamped map; T2 =
  fingerprint(`tier_key`)-named block files with a tracked resident set;
  spill-coldest-on-insert; reads check T1 (recency bump) then disk;
  removal unlinks. Spill write failure keeps the entry in RAM and reports
  no room (never lose a payload).
- Wiring: `--kv-ssd-path` (+ `--kv-ssd-max-bytes`, default 20 GiB) now
  passes structural validation and attaches pre-traffic via the
  engine-thread control seam (`run_on_executor` →
  `CudaExecutor::set_kv_tier_disk`). Fail-closed moved to the consumption
  boundary: non-CUDA backends and non-dense CUDA arms reject an explicit
  request instead of silently serving without the tier.
- `/v1/stats` `ssd_recall` stays `available=false` with an updated reason:
  per-level recall counters are not split out yet; T2 activity is inside
  the `kv_tier` block. Splitting per-level counters is follow-up scope on
  #83.

## Env

Local Apple Silicon; store logic host-tested, CUDA copies typechecked
(`CUDARC_CUDA_VERSION` `cuda,no-cuda` lane).

## Results

- `cargo test -p infer-cuda --release` — **58 passed** (3 new kv_tier
  tests: T1 cap, spill-coldest + disk read-back byte-identical + unlink,
  both-levels-full).
- `cargo test -p infer-api` (cpu lane) — 14 passed (fail-closed test
  rewritten to structural-validation-passes; consumption gating asserted
  at engine build).
- infer-server 23 passed; metal cli lane compiles.

**pending-remote**: the disk path under real CUDA traffic (demote → spill
→ promote-from-disk → needle gate) rides the same pod session as #82's
gate; wall-clock license: disk promote must beat re-prefill at the SLO
shape, else T2 stays a capacity extender for cold prefixes only.

## Problems

Per-level observability is coarse (one `kv_tier` counter set across
T1+T2). Acceptable for the first cut; the counters live engine-side and
levels are executor-internal — splitting needs an executor→stats getter,
deferred to keep the seam minimal.

## Learnings

Reporting T1+T2 as one capacity number kept the engine completely
level-agnostic — no scheduler change at all for a whole new storage tier.
The tier seam's "opaque key + capacity" design paid for itself one tranche
later.
