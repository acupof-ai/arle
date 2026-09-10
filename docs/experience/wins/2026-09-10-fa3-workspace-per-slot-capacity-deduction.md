# FA3 quantized-attention workspace becomes a per-slot line item in the qwen35 capacity solve

Date: 2026-09-10 · Lane: lane/fa3-ws-solve · Task #18 (fa3-workspace-in-resource-solve)

## Context

The quantized-FA3 split-KV workspace (`paged_attention_quantized_fa3_workspace_bytes`)
is allocated at `PagedKVPool` construction for `INT8 | FP8E4M3` pools and is
charged in `device_bytes()`, but the capacity solve's budget breakdown
(`compute_budget_breakdown`, the input to `budget_bytes_for_tokens`) had no
workspace term. The solve therefore under-counted pool VRAM by the workspace
size, so the Qwen3.8-27B-NVFP4 / 32K / 256-slot default fit the solve and OOMed
mid-serve instead of being capped or refused at boot.

## What worked

- **The workspace is linear in `num_slots`, so it is a per-slot charge, not a
  fixed reserve.** The FFI formula is `num_splits × total_q_tokens × num_q_heads
  × (head_dim + 2) × 4B`, and pool construction passes `total_q_tokens =
  num_slots`. The per-slot cost `num_splits × num_q_heads × (head_dim + 2) × 4B`
  is a constant (~2 MB/slot for a 27B quantized pool, ~545 MB at 256 slots), so
  it folds into the same per-slot deduction the solve already makes for
  recurrent state — no circular dependency, no iterative solve. This is the
  opposite shape from DSv4's `prefill_transient_reserve_bytes` (a fixed constant
  deducted before the solve): that reserve is count-independent, this workspace
  is not.
- **The deduction lives in the infer-model solve core; the byte count stays in
  the executor.** `kv_slot_budget_local` / `kv_pool_pages_local` gain an
  `attn_ws_per_slot` param and deduct `(per_slot + attn_ws_per_slot) × n` from
  free VRAM, itemizing the workspace in the budget log. The executor computes
  the bytes from the pool shape (`local_kv_heads × 8`, `head_dim`, 64 splits) via
  the existing `cuda_kernels::kv_quant` FFI wrapper — the single source of the
  formula — gated on the quantized formats, 0 otherwise.
- **The reject guard and clamp log name the workspace.** A config that now
  affords 0 slots fails at boot with the recurrent and workspace per-slot costs
  both printed, so the shortfall is named instead of surfacing as a mid-serve
  OOM.

## Rule

A pool allocation whose size scales with `num_slots` is a per-slot charge in the
capacity solve: deduct it alongside the recurrent per-slot grant and itemize it
in the boot log. A count-independent allocation (DSv4's prefill transient) is a
fixed reserve deducted before the solve. Match the deduction shape to the
allocation's scaling, and keep the byte formula in one place (the kernel FFI),
not duplicated into the solve.

## Net

No GPU on this box: the boot/refusal outcome on the 27B config is
pending-remote. Local gates: `cargo test -p infer-model` 15/15 (incl.
`kv_slot_budget_charges_fa3_workspace_per_slot`, which proves a 2 KB/slot
workspace sheds the solved slot count 60 → 50); Mac CUDA clippy gate exit 0;
`check_repo_hygiene.py` green; `cargo fmt --check` clean.

`git diff --stat`: +82/−26 = +56 net across 2 files (infer-model +68 with 1 new
test, qwen35_forward.rs +40 — the workspace computation, log itemization, and
the two solve call sites).
