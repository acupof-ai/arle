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

## The architecture decision (resolve on the pod, with device evidence)

The quant kernels address KV by `new_token_indices` into a quant buffer +
per-channel scale buffer; the fused decode-attention kernels read that quant
layout. Qwen3.5 stores contiguous per-layer BF16. Two ways to bridge:

- **A — contiguous-with-scales (recommended starting hypothesis).** Keep the
  per-layer cache, change its element type to the quant byte width, add a
  sibling per-channel scale buffer per layer. Token `t` lives at `t*kv_dim`
  (page_size = whole cache), so `new_token_indices` is just the write rows —
  the "paged" kernels apply unchanged. Quant-on-append after RoPE/projection;
  attention switches to the fused quant-decode kernel. Smallest diff, reuses
  the existing slot/cache lifecycle.
- **B — move Qwen3.5 KV to the paged pool.** Larger; only justified if the
  fused quant-decode kernels hard-require true paging (a page table indirection
  the contiguous layout can't satisfy). Decide by reading the kernel `.cu`
  source on the box, not by inference here (§0: 推断 ≠ SOLID).

Pick A unless the kernel source forces B. Either way the substrate stays a
backend-internal CUDA detail — the seam (`KvPool = HostPagedKvPool` page-id
allocator) and the model-generic dispatch are unchanged.

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
INT4/TQ2/TQ3 stay report-only (monolith gate precedent).

1. `Qwen35KvState::new` (`qwen35.rs:168-172`): when dtype ≠ Bf16, allocate the
   cache at the quant byte width + a per-channel `k_static_scales` buffer per
   full-attn layer. Update `per_slot_kv_bytes()` (`qwen35.rs:780-801`) to the
   quant width so the unified resource budget sizes slots correctly.
2. KV append (after q/k/v projection + RoPE, before cache store): call the
   matching `cuda_kernels::kv_quant::quantize_paged_kv_<dtype>_per_channel`
   with `new_token_indices` = the written rows. INT8/FP8 need the two-step
   per-channel scale calibration (`compute_k_per_channel_absmax` →
   `finalize_k_per_channel_scales{,_int8}`) on the prefill pass; cache the
   static K scales per layer.
3. Attention read: replace the nonpaged BF16 decode attention with
   `decode_attention_{int8,fp8}_per_channel_k` (TQ4: `turboquant_fused_decode_
   attention` + `turboquant_state`). Size scratch via the `*_workspace_bytes`
   helpers. Prefill attention path: dequant-to-bf16 (`dequantize_paged_kv_<dtype>
   _to_hnd`) into scratch if no fused prefill kernel exists, else fused.
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
- **Contiguous-vs-paged (A/B above)** is the one decision that can force a
  larger diff; settle it from the kernel `.cu` on the box before writing append.
- **Budget interaction**: the quant width flows into `per_slot_kv_bytes` →
  the unified `SlotBudget`; verify slot count rises ~proportionally to the
  byte saving and that the C2 clamp / C4 reject messages still name real knobs.
