# Three-layer audit miss — the c=4 real path is `decode_qwen35_packed_batch` — 2026-05-07

## TL;DR

The c=4 batched decode bench actually fires **`decode_qwen35_packed_batch`** (`request_state.rs:2411`), the varlen path with `left_padding`. Path probe test (one-time `log::info!` at fn entry) confirmed:

```
INFO infer::backend::metal::request_state: request_state.rs:2455
qwen35_path_probe: decode_qwen35_packed_batch FIRED (varlen path with left_padding)
```

`decode_qwen35_batch` (line 2976), where I recently landed P3.1c.3a (dual-write) and P3.1c.3c (step_batch_paged routing), **never fires** from the production scheduler. Both functions are reachable but the scheduler runtime always dispatches through the packed varlen path via `try_decode_qwen35_packed_batch` (`runtime.rs:2481`).

This is the **third audit miss in a row** on this issue. Each layer fixed a wrong assumption only to reveal another:

1. `2026-05-07-p3-1c-bench-c4-was-not-on-paged-path.md` — discovered c=4 doesn't go through `Qwen35StepDriver::run_step` (single-stream path).
2. `2026-05-07-perf-model-audit-c4-not-leftpad.md` — claimed `decode_qwen35_batch` has no left-padding (TRUE) and concluded "the morning gap analysis is wrong" (WRONG — see #3).
3. **(this entry)** — `decode_qwen35_batch` is also never called. The actual c=4 hot path is `decode_qwen35_packed_batch`, which DOES have `left_padding`. The morning analysis of left-pad as the gap is back to plausibly correct.

## Implications for the 17 P2.x / P3.1c.x commits

| Commit | Path | Status under audit |
|---|---|---|
| P2.0 (e25d617) — pool alloc in Qwen35StepDriver | c=1 path | live & correct |
| P2.2 (60a9b32) — decode dual-write in run_step | c=1 path | live & correct |
| P3.1c.1 (77b5c2a) — prefill→pool batch write | both c=1 prefill_tokens AND c≥2 prefill (need to verify) | partial — c=1 confirmed; c≥2 prefill path not probed |
| P3.1a (23fa52d) — step_session_paged FFI | c=1 path | live & correct |
| P3.1b (df90ff0) — Rust call switch | c=1 path | live & correct |
| P3.1c.2 (ee2d097) — Rust gather + pass | c=1 path | live & correct |
| P3.1c.3a (ee5ad4e) — pool dual-write in decode_qwen35_batch | **NOT the c≥2 hot path** | **dead code** for production |
| P3.1c.3b (75e3d76) — step_batch_paged FFI | targets the wrong batch path | **dead code** entry point |
| P3.1c.3c (c97afc5) — Rust call switch in decode_qwen35_batch | **NOT the c≥2 hot path** | **dead code** for production |
| P3.1c.3d impl note (df04488) | targets wrong path | revise to target packed path |
| Perf model audit (7cab2de) | conclusion wrong (saw absence of left-pad in dead-code path) | **partially withdrawn** by this entry |

The c=1 commits (P2.0, P2.2, P3.1a/b, P3.1c.2) plus the prefill→pool write (P3.1c.1) DO touch the live single-stream path. Pool dual-write at c=1 was bench-validated (commit 7cab2de §"Verification") — single-stream ITL within ±5% of OFF baseline. Those commits stand.

The four "batched-path" commits (P3.1c.3a/b/c + perf-audit) are structurally correct Rust/C++ code that targets a function nobody calls from production. They don't HARM (no behavior change for c≥2 since `decode_qwen35_batch` doesn't run), but they do NOT progress the c=4 unlock.

## Where the c=4 work actually needs to land

| Real c=4 hot-path component | File:line |
|---|---|
| Scheduler dispatch | `metal/runtime.rs:2481` `try_decode_qwen35_packed_batch` |
| Rust orchestrator | `metal/request_state.rs:2411` `decode_qwen35_packed_batch` |
| Per-state batch carrier | `metal/request_state.rs:773` `Qwen35PackedDecodeBatch` (with `packed_kv_flat: Vec<MlxArray>` + `left_padding: Vec<i32>`) |
| C++ FFI entry | `crates/mlx-sys/src/mlx_qwen35_model.cpp:2372` `qwen35_compiled_step_batch_packed` |

For paged-KV to actually move c=4 ITL, the work has to flow through these. Specifically:

- Pool dual-write would need to fire from `decode_qwen35_packed_batch` (after `step_batch_packed` returns) to mirror the per-row K/V into the corresponding pool. Currently no pool dual-write fires here.
- The C++ `step_batch_packed` would need a paged variant that reads SDPA inputs from gathered K/V instead of `slice_update`-on-shared-cache + per-row left-pad.
- The Rust `Qwen35PackedDecodeBatch::admit_rows` flow would need to seed each new row's pool with that row's prefill K/V, similar to how P3.1c.1 does it for `Qwen35StepDriver`.

This is at least as much work as the existing `decode_qwen35_batch` thread, except correctly aimed.

## Lesson — third instance of "the path I'm targeting isn't the path running"

Pattern: change the code, bench, see ±5% noise, conclude "no regression, plumbing works". The bench was meaningless because the changed code never fired.

Three distinct misses with the same root cause: I assumed without verifying which hot path the bench exercises.

**Rule going forward**: ANY bench-driven change to a Metal hot path must include a one-time `log::info!` probe at the top of the targeted function. Bench then greps for the probe in server logs to confirm the function fires. Without that, "no regression" is meaningless.

Stack with `feedback_substrate_audit_grep_full_tree.md`, `feedback_ffi_session_owns_data.md`, and `feedback_perf_model_unverified.md` (next memory) as the fourth audit-discipline rule.

## Recommended next steps

1. **Stop adding paged-KV pieces on `decode_qwen35_batch`**. The path is dead.
2. **Don't revert the dead commits immediately**. They're behaviorally inert; reverting churns history. Mark them as "preparation, not exercised in production" in commit body of next change.
3. **Re-target paged-KV at `decode_qwen35_packed_batch` + `step_batch_packed`**. This is the actual c=4 unlock surface.
4. **Profile FIRST.** Per the previous audit's action items, profile the actual hot path with metal capture / mlx instruments. Identify whether left-pad is really the dominant cost. Only THEN commit to a kernel cutover.
5. **Add path-probe logs** at every Metal hot-path function entry as a permanent diagnostic. Cheap, idempotent, prevents repeating this audit class.

## What to roll back

Probably nothing for now. The dead-code commits don't hurt and removing them creates churn. But the perf-model-audit doc (7cab2de) needs an erratum line saying its "no left-pad" conclusion was about the wrong function.

The master-analysis doc and M_e.1 plan should add a "Hot-path verification" section citing the path-probe technique so future readers don't re-do this miss.

## Cross-references

- Probe code in commit (this tick, not yet committed when this entry was written): `request_state.rs:2455` (`DECODE_QWEN35_PACKED_PROBE`) + `request_state.rs:2987` (`DECODE_QWEN35_BATCH_PROBE`).
- Earlier audit errata:
  - [`2026-05-07-p3-1c-bench-c4-was-not-on-paged-path.md`](2026-05-07-p3-1c-bench-c4-was-not-on-paged-path.md)
  - [`2026-05-07-perf-model-audit-c4-not-leftpad.md`](2026-05-07-perf-model-audit-c4-not-leftpad.md)
- Master analysis (needs hot-path-verification annotation):
  [`2026-05-07-metal-optimization-master-analysis.md`](../../projects/2026-05-07-metal-optimization-master-analysis.md)
- M_e.1 plan (parent):
  [`M_e1-metal-paged-kv-hot-path.md`](../../plans/M_e1-metal-paged-kv-hot-path.md)
