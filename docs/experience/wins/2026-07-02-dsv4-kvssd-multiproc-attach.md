# DSv4 TP=4 `--kv-ssd-path`: config-carried L3 attach verified end-to-end

## Context

`--kv-ssd-path` lived in serve-layer `ServeKvSsdOptions` plus a post-spawn
`run_on_executor(set_kv_tier_disk)` hook, but DSv4 multi-GPU serves through
multiproc workers that build engines solely from `ARLE_WORKER_ENGINE_CONFIG`
— so workers never attached the disk tier. With `--kv-t1-budget-bytes 0` the
slot tier had zero pages and every `demote_slot` returned `Ok(false)`: no
demoted slots, no disk pages, no mmap files. Fix series `da0f40e2` +
`a80e0aa8` + `35109d64`: `EngineLoadConfig` carries
`kv_ssd_root`/`kv_ssd_max_bytes`, every rank attaches inside
`build_cuda_engine` (after the budget setters that rebuild the store),
`ServeKvSsdOptions` and both post-spawn hooks deleted, DSv4 `slot_image_bytes`
fit gate deleted (chunk room is the only real bound), Qwen3.6 G3 slot tier
now actually attaches its disk level.

## What Worked

Pod verify ran the fix-series tree (pre-rebase head `651b3759`; the series
now lives on main as `da0f40e2`+`a80e0aa8`+`0ef6a72e`), 8×H20 box,
DSv4-Flash-FP8,
`INFER_CUDA_DEVICES=1,3,4,6 INFER_TP_SIZE=4 INFER_DSV4_MAX_SEQ_LEN=4096`,
serve flags `--max-total-tokens 4096 --max-running-requests 1
--slot-oversubscription --kv-ssd-path /host/arle-kv-ssd-verify
--kv-ssd-max-bytes $((16<<30)) --kv-t1-budget-bytes 0`:

- **All 4 worker ranks attach at build**: each rank's log shows
  `kv_ssd_root: Some("/host/arle-kv-ssd-verify")` in its parsed config; at
  engine-ready the root holds 4 per-PID namespaces
  (`arle-kv-tier-<pid>-<nanos>-<n>/kv.mmap`, 16 GiB sparse each).
- **Demote/promote round-trips through disk**: two back-to-back request
  pairs → `demoted_slots=21, promoted_slots=21, slot_promote_failures=0`.
  With `--kv-t1-budget-bytes 0`, `host_capacity_pages==0` forces every chunk
  through `write_to_disk`; physically `du` shows 129 MiB of written blocks in
  each rank's sparse mmap (513 MiB total).
- **Parked request coherent**: the victim resumed across ~10 park/promote
  rotations and produced a coherent 256-token essay (title-level variance
  matches MoE cross-run non-determinism, reproduced on an unparked control).
- Perf posture: the default serve path is untouched (no `kv_ssd_root` → no
  attach; demote only runs under opt-in `--slot-oversubscription`), so no
  guidellm delta run — correctness-gated opt-in feature, same treatment as
  [2026-06-30-kv-mmap-tier-e2e](2026-06-30-kv-mmap-tier-e2e.md).

Repro clients: two `POST /v1/chat/completions` fired **back-to-back** (same
coordinator drain window), A `max_tokens=256`, B `max_tokens=64`.

## Problems

- **Multiproc admission latency (separate product issue)**: a request
  arriving mid-decode queues FIFO behind ~kHz empty `TickAdmissions`
  (`coordinator.rs` lockstep loop; ~608k ticks for ~600 engine steps) and
  reaches the engine only after the running generation finishes — B sent
  ~2s after A never parks A, and `/v1/stats` is unserviceable during a
  decode (StatsQuery rides the same FIFO; measured 12.3s stall, answered
  8ms after completion). Park requires B to be engine-side before A's 8th
  token (`OVERSUBSCRIPTION_MIN_SLICE`).
- 4 workers share interleaved stderr; `STEP_DIAG` lines are
  rank-unattributable — per-rank log prefixes would help.

## Rule

Anything a multiproc worker must honor MUST ride `EngineLoadConfig` (the
only state that crosses `ARLE_WORKER_ENGINE_CONFIG`) and be applied inside
`build_cuda_engine` — a post-spawn `run_on_executor` hook only ever reaches
the single-proc rank. Same class as the MTP lowering (`serve.rs`); check
this FIRST when a serve flag "works single-GPU but not TP".
