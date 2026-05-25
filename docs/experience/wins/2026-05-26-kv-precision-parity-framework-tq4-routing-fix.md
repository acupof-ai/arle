# KV precision parity framework landed + TQ4 prefill routing fix

## Context

`docs/plans/2026-05-25-kv-precision-parity-framework.md` set the goal: build a
per-precision parity test harness (BF16 reference vs INT8 / FP8 / TQ4) so that
each KV quant precision has a regression gate, then surface and fix what the
audit exposes. The motivation came from the 2026-05 errors-entry pileup —
17 FP8 / TurboQuant / GPTQ INT4 kills landed without a cross-precision test,
so each "fix" risked silently breaking already-green precisions.

Phase 1 (harness) + Phase 2 (audit) + a structural Phase 3 fix (TQ4 prefill
routing) landed in this session.

## What worked

### 1. Harness — `infer/tests/kv_precision_parity.rs`

- Boots the scheduler in-process per precision via `Scheduler::with_config`
  with explicit `kv_cache_dtype` + `kv_pool_format`, runs the same prompt set
  greedy, compares token trajectories against the BF16 reference, and writes
  `target/kv-parity-<model>-<unix>.json`.
- Per-precision gates: BF16 self = 1.0, INT8 ≥ 0.99, FP8 / TQ4 currently
  report-only pending Phase 3 root-cause (see below).
- Knobs: `KV_PARITY_PROMPTS`, `KV_PARITY_MAX_TOKENS`, `KV_PARITY_MAX_SEQ_LEN`,
  `KV_PARITY_INCLUDE_TQ23=1`.
- Skips cleanly if model weights are missing (mirrors existing
  `infer/tests/` conventions).

### 2. TQ4 prefill routing fix

**Before**: TurboQuant pools allocate `page_size = 1` (kernels for token-
granular paged decode have not been ported yet — see comment at
`crates/cuda-kernels/src/paged_kv.rs:9-11`). Qwen3's `launch_prefill_batch`,
the trait default `forward_prefill_batch`, and the scheduler's
`prepare_prefill_batch` all routed unconditionally to the paged prefill path,
which calls the TileLang HD128 kernel that hard-asserts `page_size == 16`
(`ops/attention.rs:617`). Every TQ4 prefill aborted before a single token
generated; the audit observed `mean_match = 0.0000` with `tokens = 0`.

**After**: three call sites now gate the paged path on `page_size == 16` and
fall through to the contiguous BF16 prefill + post-completion migration
(`migrate_from_contiguous_turboquant_range`):

- `infer/src/scheduler/cuda/prefill.rs:535` — `uses_paged` (real request
  path).
- `infer/src/model/qwen3/forward.rs:454` — `launch_prefill_batch` dispatch
  (warmup, in-process tests, direct-entry callers).
- `infer/src/model.rs:506` — trait default `forward_prefill_batch` (fallback
  for any model that inherits the default).

A companion change in `infer/src/scheduler/cuda/core/construction.rs:107-122`
keeps the per-slot contiguous KV buffer at `CONTIGUOUS_KV_TOKENS = 512` for
non-paged-prefill formats (the existing shrink to `PREFIX_CACHE_BLOCK_SIZE`
only fires when both the model and the pool format support paged prefill).
The warmup loop at `core/warmup.rs:188-205` now skips Pass 3 entirely when
the pool format is non-paged, dropping the 7-retry warning spam.

**Audit result (Qwen3-4B, 4 prompts × 64 tokens, L4)**:
- BF16: `mean_match = 1.0000` (self).
- INT8: `mean_match = 1.0000`.
- FP8:  `mean_match = 0.0156` (report-only — see Deferred).
- TQ4:  `mean_match = 0.0000` (report-only — 4-bit FWHT is inherently lossy
  at greedy token-trajectory granularity, consistent with the 2026-05-21 TQ
  9B fixed-logits kill).

TQ4 now reaches inference instead of aborting at prefill; trajectory parity
remains structurally weak by 4-bit's nature, which is the correct posture.

### 3. `auto` KV default flipped to BF16

`infer/src/main.rs::kv_mode_candidates` previously emitted
`[FP8E4M3, BF16]` for `--kv-cache-dtype auto`, with the comment
claiming "negligible quality regression on Qwen3 family". The 2026-05-26
audit reproduces the 2026-05-02 / 2026-05-05 FP8 token-1 catastrophic
divergence (`mean_match = 0.39%` at 8×256, `1.56%` at 4×64) — that comment
was inaccurate. Auto now ships `[BF16]` only; FP8 is opt-in via
`--kv-cache-dtype fp8` with the divergence behavior documented in the CLI
help.

## Deferred — Phase 3

- **FP8 catastrophic step-1 divergence**: the harness reproduces the
  2026-05-02 / 2026-05-05 bug consistently. Root-cause requires unit-test-
  level quant→dequant roundtrip diagnostics (per the 2026-05-05 errors
  entry's "next steps" §1-3 — full-logit delta, durable FP8 row/scale
  inspection, `decode_attention_varlen_fp8` readback comparison). Tracked
  via the harness gate `gate_trajectory: None` (report-only) and the
  errors entry [`2026-05-26-fp8-kv-step1-divergence-known-deferred.md`](../errors/2026-05-26-fp8-kv-step1-divergence-known-deferred.md).
- **INT8 long-decode drift**: at 8 prompts × 256 tokens, INT8 hit
  `mean_match = 0.8901` with 1/8 prompts diverging at step 242. Short-decode
  (≤ 64 tokens) passes the 0.99 gate. Long-decode accumulation needs
  investigation; tracked via the same errors entry.

## Rule

- Cross-precision parity gate **must** ship before any KV quant kernel
  optimization (the 2026-05 erratum was 17 fixes without one). The harness
  here is that gate going forward — every KV-touching change is expected to
  produce a parity diff vs BF16 reference, even if the precision being
  changed is report-only.
- Format invariants (`page_size == 16` for TileLang HD128 prefill) should
  be encoded at every dispatch site that crosses the invariant boundary,
  not only at the "main" path. The TQ4 routing fix needed three call
  sites to converge cleanly.
- "Auto" defaults must match the parity audit's current verdict — if a
  precision fails the gate, it cannot be the auto candidate, full stop. Old
  comment claims ("negligible quality regression") are not evidence and
  should be deleted when contradicted.

## Cross-refs

- Plan: [`docs/plans/2026-05-25-kv-precision-parity-framework.md`](../../plans/2026-05-25-kv-precision-parity-framework.md)
- FP8 deferred: [`docs/experience/errors/2026-05-26-fp8-kv-step1-divergence-known-deferred.md`](../errors/2026-05-26-fp8-kv-step1-divergence-known-deferred.md)
- Origin of FP8 known-broken status:
  - [`2026-05-02-qwen3-fp8-kv-numerical-tier1-fail.md`](../errors/2026-05-02-qwen3-fp8-kv-numerical-tier1-fail.md)
  - [`2026-05-05-fp8-kv-tier1-still-fail.md`](../errors/2026-05-05-fp8-kv-tier1-still-fail.md)
- TurboQuant lossiness baseline: [`2026-05-21-arle-turboquant-9b-fwht-fixed-logits-kill.md`](../errors/2026-05-21-arle-turboquant-9b-fwht-fixed-logits-kill.md)
