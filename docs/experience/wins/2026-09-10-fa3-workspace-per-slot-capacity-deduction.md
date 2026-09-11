# FA3 quantized-attention workspace becomes a per-slot line item in the qwen35 capacity solve

Status: **pending-remote** — code landed on `lane/fa3-ws-solve` (PR #281); the
GPU boot/refusal outcome below has not run. Do not read the correctness claims
as verified. What IS verified locally (Mac, no GPU): the infer-model solve
tests (15/15, incl. the per-slot workspace deduction), the Mac CUDA clippy
gate, hygiene, and fmt. What is NOT verified: that the Qwen3.8-27B-NVFP4 /
32K / 256-slot config now boots with reduced slots or is refused at boot with
the shortfall named — that needs the model on a GPU.

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

## Remote validation command (1×H20)

The config from the task: `Qwen3.8-27B-NVFP4`, 32K length, 256 slots.
Copy the checkpoint onto the container's local NVMe and serve from that
path.

```bash
arle serve --backend cuda \
  --model-path <model-dir>/Qwen3.8-27B-NVFP4 \
  --max-total-tokens 32768 \
  --max-running-requests 256
```

`max_seq_len` is `max_total_tokens`; requested slots is
`max_running_requests` (`crates/infer-cuda/src/executor/qwen35.rs:619`,
`kv_budget_plan`). The FA3 workspace term is non-zero only for an
INT8/FP8-E4M3 pool; Qwen3.8-27B-NVFP4 serves FP8 KV, so leave
`--kv-cache-dtype` at its default.

Success is one of two outcomes printed at boot
(`crates/infer-cuda/src/qwen35_forward.rs`):

1. clamps — a WARN containing
   `Qwen3.5 KV budget: requested 256 slots × ~NMB/slot (recurrent NMB +
   FA3 attn workspace NMB) exceeds the cross-rank-min joint-affordable …
   clamping num_slots to AFF`, with the preceding INFO line
   `Qwen3.5 KV budget: free …MB, per_slot …MB … + FA3 attn workspace
   <nonzero>MB/slot`; the server then serves with AFF < 256 slots; or
2. refuses — the startup error
   `Qwen3.5 KV budget rejected startup: post-weights free VRAM affords 0
   slots at max_seq_len 32768 (per_slot ~NMB, recurrent NMB + FA3 attn
   workspace NMB, exceeds …)`.

Either outcome names the workspace; a boot that proceeds with 256 slots
and OOMs mid-serve means the deduction did not engage.

## Net

Local gates: `cargo test -p infer-model` 15/15 (incl.
`kv_slot_budget_charges_fa3_workspace_per_slot`, which proves a 2 KB/slot
workspace sheds the solved slot count 60 → 50); Mac CUDA clippy gate exit 0;
`check_repo_hygiene.py` green; `cargo fmt --check` clean. The 27B/32K boot
outcome runs on an H20 with the command above; no result has been recorded
yet.

`git diff --stat`: +82/−26 = +56 net across 2 files (infer-model +68 with 1 new
test, qwen35_forward.rs +40 — the workspace computation, log itemization, and
the two solve call sites).
