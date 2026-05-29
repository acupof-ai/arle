# OPD P4 — default OPD rollout to the infer path + base-cache hardening + dead-code audit

**Date**: 2026-05-29
**Track**: CUDA / train (OPD throughput enabler — capstone)
**Plan**: [`docs/plans/2026-05-29-opd-student-rollout-via-infer.md`](../../plans/2026-05-29-opd-student-rollout-via-infer.md) §P4
**Builds on**: [`2026-05-29-opd-infer-rollout-ab-p3.md`](2026-05-29-opd-infer-rollout-ab-p3.md) (5.0× step / 60.9× rollout, flagged OFF)
**Status**: PASS — default flipped, infra hardened, zero dead code safely deletable

## Context

P3 confirmed routing OPD student rollout through the in-process infer engine
(CUDA graph + paged KV) is 4.99× faster end-to-end and 60.9× faster on the
rollout phase, behind `ARLE_OPD_INFER_ROLLOUT=1` (default OFF). P4 is the
capstone: make it the default, harden the base-cache infra P3 worked around,
and delete only genuinely-dead code.

## What Worked

### 1. Base-cache hardening (`feat(train,cuda)` 59180e01)

P3 had to seed the infer student engine's `lora_base_cache` via a temp-PEFT-dir
+ `INFER_LORA_PATH` dance at engine construction: `cache_lora_base()` was
populated **only** by `load_and_attach_lora`, and `remerge_lora` bailed without
it. Fragile for a default path (disk write + env mutation in a single-threaded
setup window).

**Fix** (`infer/src/model/qwen35/weights.rs`): make `remerge_lora` capture the
pristine base **lazily and idempotently** on its first call. A student engine
constructed without `INFER_LORA_PATH` is pristine (no adapter merged), so the
first `remerge_lora` (driven by the per-step `sync_lora_from_store`) snapshots
the un-merged q/v weights via the existing idempotent `cache_lora_base()`
(early-returns once set). Also exposed `cache_lora_base_pub()` for callers
wanting a deterministic snapshot point.

This let the example
(`crates/train/examples/opd_step_cuda_infer_teacher_train.rs`) drop the
temp-PEFT-dir + `INFER_LORA_PATH` juggling entirely (and the now-dead
`write_infer_adapter_dir` / `peft_name` / `F32Tensor` helpers + `safetensors` /
`tempfile` / `Cow` / `BTreeMap` / `Dtype` / `View` imports). Engine now loads
clean from the student dir; first per-step sync seeds the base.

Verified by the P2 canary (100% floor + 100% sync, below) and the P4
confirm-bench (`infer_sync` succeeds every step with no disk/env setup).

### 2. Default flip (`feat(train)` b84ba740)

`crates/train/src/opd.rs::infer_rollout_flag_enabled()` inverted:
`ARLE_OPD_INFER_ROLLOUT` unset or any value **except** `0`/`false` selects the
fast infer path; set `=0` (or `false`) to opt **out** to the train-crate
fallback. The train-crate rollout stays fully reachable (verified: opt-out arm
runs, finite loss, no second engine VRAM).

### 3. Inventory-gated deletion audit — NOTHING safely deletable

Per the SOLID constraint, grepped all callers of every train-crate decode-path
symbol across the workspace before deleting anything.

| symbol | callers found | deleted? | why |
|---|---|---|---|
| `forward_rollout_cached` (+`_profiled`, `_device_token`, `_device_token_profiled`) | `opd.rs` fallback rollout (1737/1755/1815); examples `realckpt_train`/`_profile`/`_diag`; `qwen35.rs` `#[cfg(test)]` (3162) | **NO** | live opt-out fallback + examples + unit test |
| `causal_sdpa_decode_gqa_cached` | `qwen35.rs` (758/939) — the train-crate decode attention | **NO** | only reachable via the live fallback rollout |
| `Qwen35KvCache` | `opd.rs` fallback (1726/1796); examples; test (3148) | **NO** | live fallback + examples + test |
| `append_cached_kv` | `qwen35.rs` (737/744/915/922) | **NO** | only reachable via the live fallback |
| `causal_sdpa_decode_gqa_cache_online_f32_hd256` (online kernel) | `backend_cuda.rs` dispatch (5642), active for `head_dim==256` | **NO** | active fast path |
| `causal_sdpa_decode_gqa_cache_f32` (legacy two-pass kernel) | `backend_cuda.rs` dispatch (5678) as the **`head_dim != 256` fallback** | **NO** | online kernel is hardcoded `_hd256` and only used when `head_dim==256`; legacy covers all other head_dims in a model-generic autograd backend — NOT covered by online |
| `ARLE_AUTOGRAD_DECODE_ATTN_LEGACY` escape hatch (`env_force_legacy_decode_attn`) | `backend_cuda.rs` (5707) | **NO** | forces legacy kernel; only meaningful while legacy kernel exists (kept) |
| `cpu_causal_sdpa_decode_gqa_cache` (CPU impl) | `backend.rs` (1215/3428) — `Backend` trait method | **NO** | non-CUDA path of the public autograd `Backend` contract |

**Conclusion: zero deletions are safe.** The entire train-crate rollout is the
live opt-out fallback (default flip kept it reachable — deleting it = half-state
+ broken opt-out + broken examples/test). The legacy two-pass decode kernel is
the `head_dim != 256` fallback in a model-generic autograd backend; the
`online_f32` kernel is `head_dim==256`-only, so it does **not** fully cover the
legacy kernel's role. The `ARLE_AUTOGRAD_DECODE_ATTN_LEGACY` hatch stays
because the legacy kernel it selects stays. Conservatism over deletion volume,
exactly as the constraint anticipated. The dead BF16-KV kernels referenced by
the plan were already removed in `03cf1bc8` (verified absent). The only code
removed in P4 is the example's temp-dir helpers, now obsolete after task 1.

## Results — P4 confirm-bench (NEW default, infer on, standalone)

RTX 4070 Ti SUPER, sm89, CUDA 13.2. In-process teacher + student =
Qwen3.5-0.8B-Base, `examples/opd/sample-prompts.jsonl`, **rollout=128**,
**steps=2**, r=8 α=16 AttentionQv, lr=1e-5, `mem_fraction_static=0.05`, CUDA
graph on. No `ARLE_OPD_INFER_ROLLOUT` set (= default infer path). No temp dir /
no `INFER_LORA_PATH`.

| metric | value | P3 infer-arm |
|---|---:|---:|
| **mean step (s)** | **50.18** | 50.07 |
| step 1 (s) | 52.27 | — |
| step 2 (s) | 48.10 | — |
| total wall (2 steps, s) | 103.33 | — |
| **student_rollout (s)** | **3.34 / 3.15** | 3.34 |
| infer_sync (s) | 0.196 / 0.205 | 0.20 |
| backward (s) | 40.80 / 37.35 | 38.89 (dominant) |
| student_forward (s) | 7.33 / 6.89 | 7.06 |
| loss | 1.036e-4 / 1.051e-4 | 1.036e-4 (step 1) |

Step-time matches the P3 infer arm (~50 s). KL/loss finite and same magnitude.
`infer_sync` runs every step with **no disk/env setup** — confirms the lazy
base-cache works end-to-end.

### Peak VRAM (16 GB card, `nvidia-smi` 2 Hz)

`PEAK_VRAM_MIB=8806` (2 in-process infer engines + train student + ~131 MiB
desktop baseline). ~7.5 GB headroom. No OOM.

### Opt-out fallback (`ARLE_OPD_INFER_ROLLOUT=0`, steps=1 rollout=4)

EXIT=0, loss 7.90e-5 finite, **no `infer_student_loaded`** (no second engine
VRAM paid), rollout via the train-crate path. No half-state — opt-out is fully
functional.

## Verification

- `cargo check -p infer --no-default-features --features cuda,no-cuda` — clean
  (pre-existing warnings only).
- `cargo check -p train --no-default-features --features cuda,no-cuda
  --examples` — clean.
- CPU `cargo test --release -p train --no-default-features --features no-cuda
  --test test_opd_step` — **14/14 pass**.
- P2 canary `test_infer_student_lora_sync` (GPU, --ignored) — **PASS**, floor
  100.0% / sync 100.0% over 64 tokens.

## Rule

- A default-path infra prerequisite must not depend on a disk/env side-channel:
  capture the pristine LoRA base **lazily on first re-merge** from the pristine
  in-memory weights, not via a temp PEFT dir + `INFER_LORA_PATH`.
- "Delete the slow path on the fast path's win" is wrong when the slow path is
  the live opt-out fallback and the kernel is a head-dim-generic fallback. Grep
  every caller across the workspace before deleting; a `head_dim==256`-only
  fast kernel does NOT cover a model-generic legacy kernel. Zero safe deletions
  is a valid, SOLID outcome — conservatism over deletion volume.

## Scope / next

- OPD rollout default is now the infer path; train-crate rollout is the
  reachable opt-out fallback (`ARLE_OPD_INFER_ROLLOUT=0`).
- Next axis: backward (~38–41 s, ~78% of step) — the plan's anticipated next
  OPD throughput bottleneck.
