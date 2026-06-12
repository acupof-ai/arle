# CUDA KV-quant wiring — #68 T3/T4 pod-session plan

Turnkey spec for the one pod session that lands the CUDA INT8/FP8/TQ4 KV path
and runs the Qwen3.5 4-precision license-or-kill matrix. #68 T1 (neutral gate
harness) and T2 (seam-level dtype dispatch) are landed and pushed; this doc is
the remaining T3 (executor wiring) + T4 (CLI threading) + T5 (validation).

## Verdict — why this is a pod session, not a Mac commit

The wiring touches the CUDA attention hot path and is gated by a correct-
inference needle ladder on the real serve. It cannot be validated on a Mac, and
FP8 KV has a multi-kill history (2026-05 errors ×4, one a 3-week chase that
collapsed to a test-framework artifact). Landing unvalidated kernel-integration
code would violate license-or-kill, no-half-states, and correctness-gate-on-
real-path. So: Claude owns the line-level spec below; the pod session executes
and validates each dtype before it counts as landed.

## Current state (evidence, grep-confirmed 2026-06-12)

- The seam carries the request: `infer_seam::KvCacheDtype {Auto,Bf16,Int8,Fp8,Tq4}`
  (`crates/infer-seam/src/kv_dtype.rs`), re-exported by infer-api, resolved
  per-backend at construction. Metal does `MetalKvCacheDtype::resolve` (T2).
- **CUDA consumes the dtype nowhere.** `grep kv_cache_dtype crates/infer-cuda/src`
  → 0 hits. Executor constructors take `(model_path, num_slots, total_pages)`
  only (`executor.rs:84-117`).
- **Qwen3.5 KV is contiguous BF16 per-layer caches**: `Qwen35KvState.k_caches/
  v_caches: Vec<DeviceVec>`, each `DeviceVec::zeros(ctx, max_seq_len * kv_dim)`
  bf16 (`qwen35.rs:147-172`); read by the nonpaged BF16 attention kernels.
  `per_slot_kv_bytes()` (`qwen35.rs:780-801`) hard-codes the bf16 byte budget.
- **DSv4 FP8 KV is a bespoke MLA latent arena** (`Dsv4MlaKvArena`, 584 B/token,
  FlashMLA MODEL1 NoPE=448/RoPE=64, kv_heads=1; `dsv4.rs:28-72`) — model-
  specific, NOT a generic INT8/FP8/TQ4 substrate Qwen3.5 can reuse.
- **cuda-kernels already has the quant kernels + Rust wrappers, unwired into
  either executor**: `kv_quant.rs` (24 pub fns: `quantize_paged_kv_int8_per_channel`,
  `quantize_paged_kv_fp8_per_channel`, `quantize_paged_kv_int4_per_channel`,
  the `*_per_channel` scale calibration, and fused `decode_attention_{int8,fp8}_
  per_channel_k`), `kv_turboquant.rs` (TQ4: `turboquant_quantize_paged_single`,
  `turboquant_fused_decode_attention`), `turboquant_state.rs` (rotations +
  Lloyd-Max codebook). These have cuda-level tests but zero infer-cuda callers.

So the math is ported at the kernel level; the gap is executor integration.

## The architecture decision — SETTLED by the kernel source (evidence, not hypothesis)

I first hypothesized "A — contiguous-with-scales, reuse the kernels unchanged."
Reading the `.cu` source on 2026-06-12 **refutes A**. The two halves of the
quant path disagree on what they need:

- **Write** (`csrc/kv/kv_quant.cu`): `quantize_paged_kv_*_kernel` takes
  `new_token_indices` and computes `page_idx = token_row / kPageSize` with
  `kPageSize = 16` baked as a compile-time constant, into a page-blocked **NHD**
  layout (`page_idx*16*kv_dim + kv_head*16*head_dim + …`). For a contiguous
  buffer this reduces to flat indexing — so the write side *alone* looks
  A-compatible.
- **Read** (`csrc/attention/decode_attention_quantized.cu`): the fused
  `decode_attention_{fp8,int8,int4}_per_channel_k` kernels read KV through a
  **page-table indirection** — `kv_indices` (block table) + `kv_indptr`
  (per-request page offsets): `page_idx = kv_indices[page_start_global + g];
  row_base = page_idx * kQuantPageSize`. This is genuine FlashInfer-style paged
  attention. Qwen3.5's contiguous per-layer cache has no `kv_indices`/`kv_indptr`.

**Verdict: B is required.** The quant decode path is page-table-driven, so
Qwen3.5's quant-KV must live in a paged NHD quant pool (data + per-channel
scales) addressed by `kv_indices`/`kv_indptr`, and the quant decode replaces the
nonpaged BF16 attention. This is materially more than "change `DeviceVec`'s
element type": Qwen3.5 today runs **nonpaged** full-attention; the quant path
brings paged attention (page table build per step) to it. That raises T3's cost
and is the strongest reason it is a pod session, not a Mac diff.

Scope guard: the host seam stays unchanged — `CudaKvPool = HostPagedKvPool`
already allocates page ids; the page-blocked NHD *device* quant pool +
`kv_indices`/`kv_indptr` build are a backend-internal CUDA detail below the seam.
The model-generic dispatch (T2) and the gate (T1) are untouched.

Page-table reuse — SETTLED (read `dsv4_page_table.rs` 2026-06-12): its core
math (`physical_page` logical→physical lookup; `contiguous_page_table_byte_range`
contiguity proof) is **shape-agnostic** — it operates on a generic `table: &[u32]`
+ `page_bytes`; only the error strings and the 584 B test fixture are DSv4-
flavored, and it is pure host code (CPU-testable, no nvcc). So: **lift it to a
backend-neutral `paged_kv_table.rs`** (rename the DSv4 error strings) per the
unified-abstraction rule, and reuse it for Qwen3.5. What it does NOT yet have —
and what is new code either way — is the FlashInfer-style `kv_indices`/`kv_indptr`
per-request flattening the `decode_attention_*_per_channel_k` kernels consume;
add that builder into the lifted module (CPU-testable, same discipline).

## T4 — thread the dtype to the executor (line-level; do this FIRST, it is
Mac-typecheckable and unblocks T3)

1. `crates/infer-cuda/src/executor.rs`: add a `CudaKvCacheDtype` resolved enum
   (mirror `MetalKvCacheDtype`) + `resolve(infer_seam::KvCacheDtype)`: `Auto|Bf16
   → Bf16`; `Int8 → Int8`; `Fp8 → Fp8`; `Tq4 → Tq4`. Until a dtype's kernel
   path lands (T3), its `resolve` arm `anyhow::bail!`s "CUDA <dtype> KV not yet
   wired (#68 T3)" — fail loud, never silent-downgrade.
2. Add `kv_cache_dtype: CudaKvCacheDtype` to each executor constructor signature
   (`from_qwen3_bf16_safetensors`, `from_qwen35_moe_safetensors`, DSv4) and pass
   it to `Qwen35KvState::new` / the DSv4 arena builder.
3. `crates/infer-api/src/loaded.rs`: at the CUDA `metal_serve_handle` sibling,
   call `infer_cuda::CudaKvCacheDtype::resolve(config.kv_cache_dtype)?` and pass
   it into the constructor. (Bf16 today → byte-identical; bail on the rest.)
4. `crates/cli/src/serve.rs`: add `Fp8`/`Tq4` to `ServeKvCacheDtypeArg` +
   `args.rs:452`; widen the int8-Metal-only guard at `serve.rs:339` to "quant
   dtypes route to the backend that supports them" (Metal: int8; CUDA: int8/fp8/
   tq4 once T3 lands). Until then the CUDA arms bail at resolve.
5. Bench-exempt (no runtime behavior change; all new arms bail). Unit-test
   `CudaKvCacheDtype::resolve` like the Metal one.

## T3 — wire the quant hot path (per dtype, each gated by T5 before it counts)

Per dtype, in this order (cheapest correctness first): **INT8 → FP8 → TQ4.**
INT4/TQ2/TQ3 stay report-only (monolith gate precedent). All four steps assume
the **B verdict above**: a page-blocked NHD quant pool, not the contiguous cache.

1. Stand up the per-layer **paged NHD quant pool** for the full-attn layers
   (replaces the contiguous `Qwen35KvState.k_caches/v_caches` for dtype ≠ Bf16):
   quant-byte data buffer (`kQuantPageSize`-blocked NHD) + per-channel
   `k_static_scales` (and `v_scales` where the kernel takes them). Build/maintain
   the `kv_indices`/`kv_indptr` page table per step (reuse `dsv4_page_table.rs`
   if it is not MLA-shape-locked, else a Qwen-local builder — see open sub-Q).
   Update `per_slot_kv_bytes()` (`qwen35.rs:780-801`) to the quant width so the
   unified `SlotBudget` sizes slots correctly. Bf16 keeps the existing
   contiguous path byte-for-byte.
2. KV append (after q/k/v projection + RoPE, before store): call the matching
   `cuda_kernels::kv_quant::quantize_paged_kv_<dtype>_per_channel` with
   `new_token_indices` = the written rows into the paged pool. INT8/FP8 need the
   two-step per-channel scale calibration (`compute_k_per_channel_absmax` →
   `finalize_k_per_channel_scales{,_int8}`) on the prefill pass; cache static K
   scales per layer.
3. Decode read: replace the nonpaged BF16 decode attention with the
   page-table-driven `decode_attention_{int8,fp8}_per_channel_k`
   (`kv_indices`/`kv_indptr` from step 1; TQ4: `turboquant_fused_decode_attention`
   + `turboquant_state`). Size partial/merge scratch via the `*_workspace_bytes`
   helpers. Prefill read: dequant-to-bf16 (`dequantize_paged_kv_<dtype>_to_hnd`)
   into a bf16 HND work buffer for the existing prefill attention, unless a fused
   quant prefill kernel exists.
4. TQ4 only: build `TurboQuantLayerState` (rotations + Lloyd-Max codebook) at
   load; rotate Q with `turboquant_rotate_query` before the fused decode.

## T5 — license-or-kill (the gate, already built in T1)

Single GPU, Qwen3.5-4B. BF16 reference first, then each dtype:

```
GATE_PROFILE=generic MODEL=/data01/models/Qwen3.5-4B scripts/lever_gate.sh bf16_ref
GATE_PROFILE=generic MODEL=/data01/models/Qwen3.5-4B SERVE_FLAGS="--kv-cache-dtype int8" scripts/lever_gate.sh qwen_int8
GATE_PROFILE=generic MODEL=/data01/models/Qwen3.5-4B SERVE_FLAGS="--kv-cache-dtype fp8"  scripts/lever_gate.sh qwen_fp8
GATE_PROFILE=generic MODEL=/data01/models/Qwen3.5-4B SERVE_FLAGS="--kv-cache-dtype tq4"  scripts/lever_gate.sh qwen_tq4
```

Gate = needle ladder ×3 same-config repeats within the BF16 same-config
envelope (±1/length, zero garbage class), NOT byte-identity (MoE non-det).
**Greedy-decode and READ the actual tokens** before trusting any miss-rate
metric (distilled lesson — the 3-week FP8 artifact). Per-dtype verdict; default
flip needs a separate wall-clock perf license per the bench spec. Ship one wins
entry `docs/experience/wins/2026-MM-DD-cuda-kv-quant-qwen35-matrix.md` with the
BF16 ref envelope, each dtype's ladder, and the license/kill call.

## Risks

- **FP8 first-token garbage** is config-suspect first (per-channel scale
  calibration window, NoPE/RoPE split) — A/B the same prompt on BF16 before
  staring at kernel code.
- **Paged attention for Qwen3.5 (B above)** is the big-diff risk: bringing the
  `kv_indices`/`kv_indptr` page-table build to a model that runs nonpaged today.
  The page-table build is the new correctness surface — validate it on Bf16
  first (a paged-Bf16 decode that matches the nonpaged Bf16 reference) BEFORE
  layering quant on top, so a quant miss isn't confounded by a page-table bug.
- **Budget interaction**: the quant width flows into `per_slot_kv_bytes` →
  the unified `SlotBudget`; verify slot count rises ~proportionally to the
  byte saving and that the C2 clamp / C4 reject messages still name real knobs.
