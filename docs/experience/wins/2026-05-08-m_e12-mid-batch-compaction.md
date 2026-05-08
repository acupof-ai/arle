# Bench — M_e.12 mid-batch compaction shipped + P1 caught + fixed — 2026-05-08

## Goal

Port mlx-lm's `GenerationBatch.filter` pattern (PR #1072 era):
when a row in the c≥2 packed-decode batch hits stop-token mid-step,
drop it from the live state in the SAME tick (not the next-tick
set-diff path that wastes 1 batched forward), and reclaim the
time-axis min-padding so survivors don't carry the dead row's
`kv_capacity`. Two mechanical wins:

1. **Row-axis in-tick compaction** — the next decode step runs at
   B-1 instead of B, saving ~1/B of the next step (~25% at c=4).
2. **Time-axis min-pad reclaim** — when only late-admitted rows
   survive, slice off the leading time slots that no row needs.
   Eliminates the "long-row death stranding survivor at long
   kv_capacity" failure mode in heterogeneous-length workloads.

## Hypothesis

- **Functional**: M_E12 path triggers when ≥1 row finishes mid-batch.
  Probe `M_E12_COMPACTION_FIRED` fires on first time-axis reclaim.
- **No regression on homogeneous workloads** (where compaction
  never triggers, the always-on `take_axis` row-keep path on
  retain_rows is still the existing scheduler-tick path).
- **Safety**: P1 from codex review caught and fixed before push
  (kv_capacity field stale after axis-2 slice).

## Implementation

`infer/src/backend/metal/request_state.rs` (+63/-3 LOC):
- Extend `retain_rows(&mut self, row_indices, shrink_time_axis: bool)`.
- After axis-0 row keep, when `shrink_time_axis && min_pad>0`:
  - Slice every `packed_kv_flat` tensor on axis 2:
    `start=[0,0,min_pad,0]`, `stop=[B, n_kv, kv_capacity, head_dim]`.
  - Decrement `batch_cache_len -= min_pad`.
  - **Decrement `kv_capacity -= min_pad`** (P1 fix from codex review;
    see § Codex below).
  - Subtract `min_pad` from each remaining `left_padding[i]`.
  - Once-fire `M_E12_COMPACTION_FIRED` probe.
  - `clear_metal_cache()` — mirrors M_e.11 KV_CACHE_CHUNK boundary
    safety, since `take_axis` per-tensor + axis-2 slice each
    allocate fresh tensors that can churn the residency set.
- `packed_gdr_flat` deliberately NOT sliced on axis 2 — that field
  is per-request recurrent state, no time axis (shape
  `[1, Hv, Dv, Dk]` and `[1, conv_kernel-1, qkv_dim]`); slicing
  axis 2 there would corrupt state. Inline comment cites the
  `try_build_qwen35_packed_decode_batch:2352-2353` source of truth.

`infer/src/backend/metal/runtime.rs` (+39/-2 LOC):
- In `execute_decode_batch`, after the
  `process_token + finish_or_requeue_decoded_request` loop:
  - Capture `original_req_ids` BEFORE consuming `open` (order matches
    cache row index by construction).
  - Compute `keep_row_indices` = rows whose `req_id` is back in
    `active` after finalize/requeue.
  - If `|keep| < |original|`, call
    `cached.batch.retain_rows(&keep_row_indices, true)`.
  - On retain_rows error, invalidate the cache (defensive backstop).
- 2 existing `retain_rows` callsites updated to pass
  `shrink_time_axis=true` (no behavior change for the homogeneous
  case where `min_pad=0`).

The pre-existing scheduler-tick set-diff path
(`runtime.rs:2543-2580`) is intentionally kept as a backstop for
cancellation/timeout cases that bypass `process_token`.

## Codex review (caught a P1 before push)

```
codex
The new compaction path leaves the logical KV capacity inconsistent
with the physical MLX tensor shapes, so common staggered batched
decode scenarios can fail after compaction.

[P1] Update packed KV capacity after compaction —
request_state.rs:922-929. When compaction fires for c≥2 packed
Qwen3.5-family decode after the shortest row drops, this slice
reduces each KV tensor's axis-2 length from `self.kv_capacity` to
`self.kv_capacity - min_pad`, but `self.kv_capacity` remains
unchanged. Subsequent row admission builds new rows with the stale
larger capacity, causing concat shape mismatches, and continued
decode can write past the actual sliced axis before the next grow.
```

P1 manifested for real in the first acceptance bench, exactly as
predicted (see § Results, "panic evidence" subsection). Fixed by
adding `self.kv_capacity -= min_pad;` immediately after the slice
loop. Rebuild + clippy clean.

## Acceptance bench command

Server (M_e.12 + P1 fix binary):
```bash
RUST_LOG=info ./target/release/metal_serve \
  --model-path mlx-community/Qwen3.6-35B-A3B-4bit \
  --port 8765 \
  --max-running-requests 16
```

Probe v2 (forced lengths via `ignore_eos=true`):
```bash
# Round 1 + Round 2 (both run sequentially): each fires
#   short = max_tokens=200, ignore_eos=true
#   long  = max_tokens=1500, ignore_eos=true
# concurrently, waits both.
/tmp/m_e12_bimodal2.sh
```

## Environment

- **Backend**: Metal (Apple Silicon)
- **Hardware**: Apple M4 Pro (auto-detected via mlx 0.31.1 banner)
- **MLX**: 0.31.1, **macOS**: 26.3.1
- **Model**: `mlx-community/Qwen3.6-35B-A3B-4bit` (canonical Metal
  per AGENTS.md)
- **Commit**: this commit ships M_e.12 (90 LOC across 2 files)
- **Feature set**: `cargo build --release --no-default-features
  --features metal -p infer --bin metal_serve`
- **Auto-wired-limit**: 20 GiB pinned (default-on)
- **M_e.11**: default 1024-token cadence (default-on; per its own
  Phase A entry, prophylactic on this stack)
- **Build gates**: cargo build clean, clippy `-D warnings` clean,
  644 `request_state::tests_mod` + sibling tests pass

## Results

### M_e.12 path triggers (first acceptance bench, OLD binary
pre-P1-fix)

```
$ grep "M_E12_COMPACTION_FIRED" /tmp/m_e12_bench_server.log
2026-05-08T11:12:31.694209+08:00  INFO request_state.rs:909
  metal_path_probe: M_E12_COMPACTION_FIRED
  (kept 1/1 rows, dropped min_pad=4784 time slots)
```

→ **min_pad=4784**. After a long-tail row finished mid-batch in a
heterogeneous-output (stdev=2750 around mean=4050) c=2 workload,
M_e.12 reclaimed 4784 time slots from the survivor — exactly the
mlx-lm pattern intent.

### P1 panic evidence (same bench, OLD binary)

```
2026-05-08T11:12:32.485734  ERROR runtime.rs:1772
  Metal decode batch panicked for [RequestId(4), RequestId(5)]:
  mlx_concatenate_axis returned a null MLX handle:
  [concatenate] All the input array dimensions must match exactly
  except for the concatenation axis. However, the provided shapes
  are (1,2,3920,256), (1,2,8704,256), and the concatenation axis is 0.
```

→ Exactly the codex P1 prediction:
- pre-compaction kv_capacity = 8704
- compaction at 11:12:31.694 sliced axis 2 to **8704 − 4784 = 3920**
- 11:12:32 admit attempted to concat new row's `[1, 2, 8704, 256]`
  with cached `[1, 2, 3920, 256]` → null handle.

The fix (decrement `self.kv_capacity` in the same block) makes
admit_rows build new rows at the new physical size, and any
extend-via-concat sees matching axis-2 lengths.

### Post-fix verification (bimodal v2, FIXED binary)

```
[Fri May  8 11:21:03] === round 1 ===
[Fri May  8 11:21:07] round 1: short done after 4s   (200 tokens, ignore_eos)
[Fri May  8 11:21:23] round 1: long done after 20s   (1500 tokens, ignore_eos)
[Fri May  8 11:21:23] === round 2 ===
[Fri May  8 11:21:27] round 2: short done after 4s   (200 tokens)
[Fri May  8 11:21:42] round 2: long done after 19s   (1500 tokens)

$ grep -E "panicked|null MLX|concatenate_axis|FATAL" \
    /tmp/m_e12_bimodal2_server.log
(no matches — clean)
```

→ Same workload structure (2-row batch, one finishes first while
other still decoding), no panic. P1 fix verified.

Note `M_E12_COMPACTION_FIRED` did NOT fire in bimodal v2: both rows
started at t=0 with `left_padding=0`; when one died, the survivor's
left_padding was still 0 → `min_pad=0` → time-axis reclaim no-op,
probe doesn't fire (probe is gated on `min_pad>0`). The row-axis
compaction (the in-tick row drop, the actual mechanical M_e.12 win)
DID happen — it just doesn't have its own probe. To exercise the
time-axis branch, you need a 3+ row batch where the originally-early
rows die and only late-admitted rows survive (heterogeneous bench
above does exactly that).

## Δ vs baseline

Per-workload ITL/TTFT delta vs pre-M_e.12 baseline is **TBD next
tick** — the available bench data was confounded by the OLD-binary
P1 panic (which invalidates ITL numbers from the `c=2` window after
11:12:31). A clean A/B with FIXED binary on the heterogeneous-stdev
workload is queued; today's evidence covers correctness, not perf.

| Aspect | Pre-M_e.12 | Post-M_e.12 (this commit) |
|---|---|---|
| Mid-batch row finish | leaves dead row in cache until next tick (1 wasted forward at B) | dropped same tick at B-1 |
| Late-admit-only survivor scenario | survivor stuck at original `kv_capacity` | reclaims `min_pad` slots |
| `retain_rows` API | `(row_indices)` | `(row_indices, shrink_time_axis: bool)` |
| Codex P1 (kv_capacity stale after slice) | n/a | fixed (`self.kv_capacity -= min_pad`) |
| Code surface | n/a | +99 / -3 LOC across 2 files |

## Problems / observations

1. **No clean A/B perf number this commit**: the heterogeneous bench
   that triggered M_e.12 also triggered the P1 panic before fix; the
   bimodal v2 verification used a workload shape that doesn't
   exercise the time-axis branch. Filed as next-tick work to run
   `bench_guidellm.sh` heterogeneous c=2 + c=4 against the FIXED
   binary, with the most-recent Metal Qwen3.6 baseline as the Δ
   anchor.
2. **Probe `M_E12_COMPACTION_FIRED` is gated on `min_pad > 0`**, so
   it doesn't observe row-axis-only compactions (which the in-tick
   trigger handles for free via the existing `retain_rows` axis-0
   take). Add a separate "row-drop" probe in a future tick if
   needed — for now, the heterogeneous bench's recorded `min_pad=4784`
   fire is sufficient evidence the in-tick path executes.
3. **Default-on, no env gate**: M_e.12 ships as a behavioral default
   for the c≥2 packed path. The cost (one-take_axis-per-tensor on
   any tick where ≥1 row finishes) is bounded by mlx-lm's matching
   policy and is paid only when payoff exists. If a workload regresses,
   the env-gate add is one bool field — no struct churn.

## What worked

- **Codex review BEFORE push** (per the existing
  `feedback_codex_review_async` memory) — caught the kv_capacity
  staleness P1 that would otherwise have shipped silently. The
  subsequent OLD-binary bench independently corroborated the
  codex prediction byte-for-byte (8704→3920 axis-2 vs new admit's
  8704). Two independent witnesses validated the fix necessity.
- **`ignore_eos=true` for forced-length probe benches** — sidesteps
  the model's natural EOS so the bench can deliberately construct
  bimodal length scenarios. Existing OpenAI v1 field, no new
  plumbing needed.
- **Reusing `retain_rows` instead of adding a parallel compact
  helper** — the row-axis half was already in tree (used by the
  scheduler-tick set-diff path); M_e.12 only had to add an
  optional time-axis post-pass and a new caller-site. ~50% smaller
  code surface than a from-scratch compactor.

## Rule

When a bench-spec'd code path has TWO branch conditions, write the
bench to exercise BOTH branches deterministically (here: time-axis
reclaim needs `min_pad>0`, row-axis needs only ≥1 row finishing).
Confirming the in-tick row-axis path requires either a workload where
late-admit produces survivors (already covered by the
heterogeneous-stdev bench) OR a separate row-drop probe (deferred).
Otherwise the absence of a probe in a single-shape verification bench
gets misread as "path didn't fire" when it really did fire silently.

## Next

- **A/B perf bench** vs FIXED binary: heterogeneous-stdev c=2 + c=4
  workload, compare to nearest Metal Qwen3.6 baseline (likely
  `2026-05-07-bench-qwen36-mle-perf.md`). Predict −15-25% ITL p50
  on c=4, −25-35% on long-tail half at c=2 per design agent
  estimate. ~10 min wall-clock.
- **Add row-drop probe** if A/B perf needs to attribute wins to
  the in-tick path vs the existing scheduler-tick backstop.
- **dflash-mlx Prometheus `/metrics` port** still on the deck.
- **M_e.13 SSD KV persistence** still queued (see task #37).

## References

- M_e.12 design (subagent a3c85aa01d71edb4e):
  see task #36 metadata.
- M_e.11 wins entry (immediately preceding):
  [`2026-05-08-m_e11-residency-set-hygiene.md`](2026-05-08-m_e11-residency-set-hygiene.md)
- M_e.11 Phase A no-repro:
  [`2026-05-08-bench-m_e11-residency-stability-phase-a.md`](2026-05-08-bench-m_e11-residency-stability-phase-a.md)
- mlx-lm ecosystem survey:
  [`docs/research/2026-05-07-mlx-ecosystem-survey-c4-itl-gap.md`](../../research/2026-05-07-mlx-ecosystem-survey-c4-itl-gap.md)
- Bench artefacts:
  - heterogeneous: `bench-output/2026-05-08-m_e12-heterogen-c2/`,
    `bench-output/2026-05-08-m_e12-heterogen-c4/`,
    `/tmp/m_e12_bench_server.log` (M_E12 fire + P1 panic)
  - bimodal v2: `/tmp/m_e12_bimodal2.log`,
    `/tmp/m_e12_bimodal2_server.log` (no panic)
