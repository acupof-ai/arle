# DSv4 band exhaustion parks instead of fatal — per-row per-layer device-fit gate (#160)

> Status: Code landed (1b80dc724 + 389464585); runtime park-verify
> **pending-remote** — real band-exhaustion c-sweep needs the 8×H20 box
> (GPUs 4-7 occupied by another lane at land time).

## Context

DSv4's `band_extend` hard-errored on device-pool exhaustion mid-forward with
no recovery (no demote/park; the planner retract loop was device-blind), so
the first over-admitted slot aborted the serve — the admission watermark's
benefit regime was exactly its unsafe regime (#160).

## What Worked

Generalized a9d0c5412's park-not-fatal gate instead of a DSv4 special case.
Three codex-review rounds shaped the seam:

- **Seam**: `kv_device_fit(rows: &[DeviceRowDemand], unfit: &mut Vec<usize>)`
  + `kv_device_gate_active()` (default false → inert backends build nothing).
  `DeviceRowDemand { slot, target_tokens, pages_hint }` carries the row's
  known final span and the engine's legacy formula.
- **DSv4**: exact per-layer pairing — per demand-paged layer,
  `need_L = band_pages_for(target + spec_depth + 1) − have`; a row fits iff
  every layer fits; fitting rows debit cumulatively. Host-side bookkeeping
  only (CUDA-graph safe). `band_extend` stays as the unreachable backstop.
- **Qwen3.6**: same numbers as before via `pages_hint` debit — behavior
  unchanged, now through the shared hook.
- Codex round 1 killed the scalar min-free/max-need projection (saturated
  sliding-window ring free=0/need=0 + compressed layer need=1 read as
  permanent exhaustion). Round 2 killed first-unfit tail-drain (a stuck
  big-need chunk starved later fitting rows forever) and per-step demand
  allocs on inert backends (capability gate + Engine-owned scratch vecs).

Gates: `device_fit_pairs_need_with_pool_not_extrema`,
`device_fit_unfit_row_does_not_starve_later_fitting_rows`, 113 infer-core
tests green, clippy -D clean, both Mac cuda-lane checks clean.

## Rule

- A multi-pool backend's admission projection must pair need and free
  per pool — combining unrelated extrema (min free × max need) manufactures
  exhaustion a saturated fixed-capacity pool never has.
- Shed exactly the unfit rows; draining the tail turns one stuck row into a
  whole-plan livelock.
