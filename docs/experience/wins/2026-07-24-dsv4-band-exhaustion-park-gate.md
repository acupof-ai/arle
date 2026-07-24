# DSv4 band exhaustion parks instead of fatal — per-row per-layer device-fit gate (#160)

> Status: Code landed (1b80dc724 + 389464585). Runtime park-verify attempted
> 2026-07-25 on 4×H20 and **the gate never fires — it is unreachable by
> config**, for a structural reason worth keeping (see §Runtime verify).

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

## Runtime verify (2026-07-25, `d0525cb06`, 4×H20 GPUs 0-3 TP=4/EP=4)

**Negative, and structurally so.** `device KV pool exhausted` count = 0 (both
variants) across four escalating pressure configs, all with
`--max-running-requests` deliberately omitted:

| config | c | complete | errors | gate | KV-overflow preempts |
|---|---|---|---|---|---|
| 64×3.4k tok | 32, 64 | 59, 83 | 0 | 0 | 164 |
| 96×~10k tok | 96 | 116 | 0 | 0 | +88 |
| 96×~1.2k tok | 96 | 164 | 0 | 0 | +64 |
| 96×~1.2k + MTP-2 | 96 | 125 | 0 | 0 | 0 |

The gate is live, not inert (`dsv4_flashmla_demand_paged` is true here:
head_dim 512 ≠ 576; the log confirms `demand-paged bands`). The third config
drove the engine to full saturation — `/v1/stats` `active_requests 59,
queue_depth 37, kv_free_pages 0` — and it still did not fire.

Attribution: host admission and the device band pool are sized from the SAME
solved capacity (`kv_layout.rs:1101` — the engine-facing admission page count
IS the solved shared token capacity in 64-token pages), so host admission binds
first by construction. `kv_device_fit` is therefore a **drift/rounding
backstop**, not a path a well-formed config reaches. No config was found that
makes the device pool scarcer than admission believes it is.

What the runs DO confirm: the old fatal path never executes. Zero `band_extend`
errors, zero panics, zero worker exits; the serve survived all four runs and
answered coherently after saturation. The adjacent park path is heavily
exercised — 316 `KV-overflow preempt → requeued for recompute`, every submitted
request completed, 0 correctness failures.

Raw: pod `/host/arle-build/bench-output/2026-07-24-a160-*`, `/host/a160-serve.log`.

## Rule

- A multi-pool backend's admission projection must pair need and free
  per pool — combining unrelated extrema (min free × max need) manufactures
  exhaustion a saturated fixed-capacity pool never has.
- Shed exactly the unfit rows; draining the tail turns one stuck row into a
  whole-plan livelock.
- When two limits derive from one solved number, the second can be
  **unreachable by construction** — say so instead of leaving its verify
  "pending" forever. A backstop that a well-formed config cannot reach is
  still worth keeping (it catches drift), but it must be labeled a backstop,
  not claimed as a verified path.
