# CUDA KV-quant wiring — #68 T3/T5 pod-session plan

Turnkey spec for the pod session that lands the CUDA INT8/FP8/TQ4 KV path and
runs the Qwen 4-precision license-or-kill matrix. #68 T1 (neutral gate harness),
T2 (seam-level dtype dispatch), and T4 (CUDA-boundary fail-loud threading) are
landed and pushed; this doc is the remaining **T3** (executor wiring) + **T5**
(validation).

## What landed (truth, with commits)

- **T1** `9afeeb58` — the gate harness is model-neutral: `scripts/lever_gate.sh`
  (boots a serve, runs the matrix, tears down) + `scripts/needle_gate.py`
  (needle ladder). `GATE_PROFILE=dsv4` carries the TP=8 env; any other profile
  (`generic`) is single-GPU and brings its config via `SERVE_FLAGS`.
- **T2** `60a27fc8` — `infer_seam::KvCacheDtype {Auto,Bf16,Int8,Fp8,Tq4}` is the
  backend-neutral *request*; each backend resolves it against its own matrix at
  construction. Metal: `MetalKvCacheDtype::resolve`.
- **T4** `bb0ed611` — `infer_cuda::CudaKvCacheDtype` + `resolve()`, called in
  `build_cuda_engine` to fail loud at the engine boundary; CLI `--kv-cache-dtype`
  gains `fp8`/`tq4`; the serve guard is symmetric (int8→Metal-target,
  fp8/tq4→CUDA-target). **T4 landed minimally**: `CudaKvCacheDtype` has only the
  `Bf16` variant today and `resolve` bails on int8/fp8/tq4 ("pending #68 T3").
  The dtype is **not** threaded through the executor constructors — that lands
  with T3, when a non-BF16 path actually consumes it (interfaces trail real
  callers). The CLI int8-on-CUDA arm also widens in T3.

## Verdict — why T3/T5 is a pod session, not a Mac commit

The wiring touches the CUDA attention hot path and is gated by a correct-
inference needle ladder on the real serve. It cannot be validated on a Mac, and
FP8 KV has a multi-kill history (2026-05 errors ×4, one a 3-week chase that
collapsed to a test-framework artifact). Landing unvalidated kernel-integration
code would violate license-or-kill, no-half-states, and correctness-gate-on-
real-path. So: Claude owns the line-level spec; the pod session executes and
validates each dtype before it counts as landed.

## Current state (evidence, grep-confirmed 2026-06-12)

- **CUDA consumes the dtype only at the boundary (T4).** `grep kv_cache_dtype
  crates/infer-cuda/src` → resolve-gate only; the executor constructors still
  take `(model_path, num_slots, total_pages)` (`executor.rs`), no dtype param.
- **The Qwen3.5/3.6 MoE KV is contiguous BF16 per-layer caches**:
  `Qwen35SlotState.k_caches/v_caches: Vec<DeviceVec>`, each
  `DeviceVec::zeros(ctx, max_seq_len * kv_dim)` bf16 (`qwen35.rs:147-186`); read
  by the nonpaged BF16 attention kernels. `per_slot_kv_bytes()` hard-codes the
  bf16 byte budget. **Only the `full_attention` layers have these caches** — see
  the model-routing note below.
- **The dense Qwen3 KV** lives in `QwenCudaExecutor` (`executor.rs`), a separate
  path on the shared `PagedKVPool`; every layer is full-attention.
- **DSv4 FP8 KV is a bespoke MLA latent arena** (`Dsv4MlaKvArena`, 584 B/token,
  FlashMLA MODEL1 NoPE=448/RoPE=64, kv_heads=1; `dsv4.rs`) — model-specific, NOT
  a generic INT8/FP8/TQ4 substrate the Qwen paths can reuse.
- **cuda-kernels already has the quant kernels + Rust wrappers, unwired into any
  executor**: `kv_quant.rs` (24 pub fns: `quantize_paged_kv_{int8,fp8,int4}_
  per_channel`, the `*_per_channel` scale calibration, fused
  `decode_attention_{int8,fp8}_per_channel_k`), `kv_turboquant.rs` (TQ4:
  `turboquant_quantize_paged_single`, `turboquant_fused_decode_attention`),
  `turboquant_state.rs` (rotations + Lloyd-Max codebook). cuda-level tests, zero
  infer-cuda callers.

So the math is ported at the kernel level; the gap is executor integration.

## Model routing on the pod (verified 2026-06-12)

`classify_cuda_model` (`loaded.rs:99-150`) maps the HF config to the executor:

| On-pod model | arch / `model_type` | CUDA path | KV state | Gate quality |
|---|---|---|---|---|
| `Qwen3-0.6B` | `Qwen3ForCausalLM` / `qwen3` (dense) | `from_qwen3_bf16_safetensors` (`QwenCudaExecutor`) | dense, **all** layers full-attn, no MoE non-det | cleanest correctness signal; 0.6B may miss long needles |
| `Qwen3.6-35B-A3B` | `Qwen3_5MoeForConditionalGeneration` / `qwen3_5_moe` | `from_qwen35_moe_safetensors` (`Qwen35CudaExecutor` → `Qwen35SlotState`) | MoE; **only 1-in-4 layers full-attn** (`full_attention_interval: 4`; the rest linear-attention recurrent, no KV cache); MoE non-det widens the gate | the issue's literal target family; slow boot (~70 GB BF16) |

**There is no `/data01/models/Qwen3.5-4B` on the pod** (the earlier spec assumed
it). The qwen3_5_moe family is present as `Qwen3.6-35B-A3B`.

## The architecture decision — SETTLED by the kernel source (evidence, not hypothesis)

I first hypothesized "A — contiguous-with-scales, reuse the kernels unchanged."
Reading the `.cu` source on 2026-06-12 **refutes A**:

- **Write** (`csrc/kv/kv_quant.cu`): `quantize_paged_kv_*_kernel` takes
  `new_token_indices`, `page_idx = token_row / kPageSize` with `kPageSize = 16`
  compile-time, into a page-blocked **NHD** layout. For a contiguous buffer this
  reduces to flat indexing — the write side *alone* looks A-compatible.
- **Read** (`csrc/attention/decode_attention_quantized.cu`): the fused
  `decode_attention_{fp8,int8,int4}_per_channel_k` kernels read KV through a
  **page-table indirection** — `kv_indices` (block table) + `kv_indptr`
  (per-request page offsets): `page_idx = kv_indices[page_start_global + g];
  row_base = page_idx * kQuantPageSize`. Genuine FlashInfer-style paged
  attention. The contiguous per-layer caches have no `kv_indices`/`kv_indptr`.

**Verdict: B is required.** Quant-KV must live in a paged NHD quant pool (data +
per-channel scales) addressed by `kv_indices`/`kv_indptr`, and the quant decode
replaces the nonpaged BF16 attention. This is materially more than "change
`DeviceVec`'s element type": the Qwen paths run **nonpaged** full-attention
today; the quant path brings paged attention (page-table build per step) to
them. That is the strongest reason T3 is a pod session.

Scope guard: the host seam stays unchanged — `CudaKvPool = HostPagedKvPool`
already allocates page ids; the page-blocked NHD *device* quant pool +
`kv_indices`/`kv_indptr` build are a backend-internal CUDA detail below the seam.
T1/T2/T4 (dispatch + gate + boundary) are untouched.

Page-table reuse — SETTLED (read `dsv4_page_table.rs` 2026-06-12): its core math
(`physical_page` logical→physical lookup; `contiguous_page_table_byte_range`
contiguity proof) is **shape-agnostic** — it operates on a generic `table: &[u32]`
+ `page_bytes`; only the error strings and the 584 B test fixture are DSv4-
flavored, and it is pure host code (CPU-testable, no nvcc). So: **lift it to a
backend-neutral `paged_kv_table.rs`** (rename the DSv4 error strings) per the
unified-abstraction rule, and reuse it for the Qwen paths. What it does NOT yet
have — and what is new code either way — is the FlashInfer-style
`kv_indices`/`kv_indptr` per-request flattening the
`decode_attention_*_per_channel_k` kernels consume; add that builder into the
lifted module (CPU-testable, same discipline).

## Which gate lane first — dense Qwen3, then the MoE full-attn layers

The substrate (paged NHD quant pool + page-table builder + quant append/decode)
is **shared and model-generic** (the #68 mandate). Wire it once; both Qwen paths
consume it. Validate it on the **cleanest correctness lane first**:

1. **Dense Qwen3** (`QwenCudaExecutor`): all layers full-attn, **no MoE
   non-determinism**, fast boot → the tightest gate envelope and fastest
   iteration on the fraught FP8 path. The 0.6B on-pod model may not pass the
   long-needle rungs in BF16; gate quant against whatever rungs BF16 passes (the
   gate is "quant within the BF16 same-config envelope," so the reference is the
   same model in BF16, not an absolute miss-rate). Fetch a stronger dense Qwen3
   (e.g. Qwen3-4B/8B) if the 0.6B BF16 envelope is too narrow to detect quant
   regressions.
2. **Qwen3.6-35B-A3B** (`Qwen35SlotState`, the qwen3_5_moe family the issue
   names): confirm the shared substrate on its `full_attention` layers (1-in-4).
   Cheap once the substrate exists; the MoE non-det envelope is the gate's
   designed-for floor.

Rationale: root-cause on a clean baseline (`§0.1`), and the substrate being
shared is what makes this model-generic rather than a per-model fork.

## T3 — wire the quant hot path (per dtype, each gated by T5 before it counts)

Per dtype, in this order (cheapest correctness first): **INT8 → FP8 → TQ4.**
INT4/TQ2/TQ3 stay report-only (monolith gate precedent). All steps assume the
**B verdict**: a page-blocked NHD quant pool, not the contiguous cache. Land in
the chosen first lane (dense Qwen3), then graft onto `Qwen35SlotState`.

0. **T4 carry-over**: when the first non-BF16 dtype lands, add its
   `CudaKvCacheDtype` variant + `resolve` arm, thread `kv_cache_dtype:
   CudaKvCacheDtype` through the executor constructor it touches, and widen the
   CLI int8-on-CUDA guard (`serve.rs`). These were deliberately deferred from T4.
1. Stand up the per-layer **paged NHD quant pool** for the full-attn layers
   (replaces the contiguous caches for dtype ≠ Bf16): quant-byte data buffer
   (`kQuantPageSize`-blocked NHD) + per-channel `k_static_scales` (and `v_scales`
   where the kernel takes them). Build/maintain `kv_indices`/`kv_indptr` per step
   via the lifted `paged_kv_table.rs`. Update `per_slot_kv_bytes()` to the quant
   width so the unified `SlotBudget` sizes slots correctly. Bf16 keeps the
   existing contiguous path byte-for-byte.
2. KV append (after q/k/v projection + RoPE, before store): call
   `cuda_kernels::kv_quant::quantize_paged_kv_<dtype>_per_channel` with
   `new_token_indices` = the rows written into the paged pool. INT8/FP8 need the
   two-step per-channel scale calibration
   (`compute_k_per_channel_absmax` → `finalize_k_per_channel_scales{,_int8}`) on
   the prefill pass; cache static K scales per layer.
3. Decode read: replace the nonpaged BF16 decode attention with the
   page-table-driven `decode_attention_{int8,fp8}_per_channel_k` (TQ4:
   `turboquant_fused_decode_attention` + `turboquant_state`). Size partial/merge
   scratch via the `*_workspace_bytes` helpers. Prefill read: dequant-to-bf16
   (`dequantize_paged_kv_<dtype>_to_hnd`) into a bf16 HND work buffer for the
   existing prefill attention, unless a fused quant prefill kernel exists.
4. TQ4 only: build `TurboQuantLayerState` (rotations + Lloyd-Max codebook) at
   load; rotate Q with `turboquant_rotate_query` before the fused decode.

## T5 — license-or-kill (the gate, built in T1)

Single GPU. BF16 reference first, then each dtype (replace `<MODEL>` with the
chosen lane's path — `/data01/models/Qwen3-0.6B` for the dense lane, or a fetched
Qwen3-4B/8B; `/data01/models/Qwen3.6-35B-A3B` for the MoE confirm):

```
GATE_PROFILE=generic MODEL=<MODEL> scripts/lever_gate.sh bf16_ref
GATE_PROFILE=generic MODEL=<MODEL> SERVE_FLAGS="--kv-cache-dtype int8" scripts/lever_gate.sh qwen_int8
GATE_PROFILE=generic MODEL=<MODEL> SERVE_FLAGS="--kv-cache-dtype fp8"  scripts/lever_gate.sh qwen_fp8
GATE_PROFILE=generic MODEL=<MODEL> SERVE_FLAGS="--kv-cache-dtype tq4"  scripts/lever_gate.sh qwen_tq4
```

Gate = needle ladder ×3 same-config repeats within the BF16 same-config envelope
(±1/length, zero garbage class), NOT byte-identity (MoE non-det). **Greedy-decode
and READ the actual tokens** before trusting any miss-rate metric (distilled
lesson — the 3-week FP8 artifact). Per-dtype verdict; default flip needs a
separate wall-clock perf license per the bench spec. Ship one wins entry
`docs/experience/wins/2026-MM-DD-cuda-kv-quant-qwen-matrix.md` with the BF16 ref
envelope, each dtype's ladder, and the license/kill call.

## Risks

- **FP8 first-token garbage** is config-suspect first (per-channel scale
  calibration window) — A/B the same prompt on BF16 before staring at kernel code.
- **Paged attention for the Qwen paths (B above)** is the big-diff risk: bringing
  the `kv_indices`/`kv_indptr` build to a model that runs nonpaged today. The
  page-table build is the new correctness surface — validate it on Bf16 first (a
  paged-Bf16 decode that matches the nonpaged Bf16 reference) BEFORE layering
  quant, so a quant miss isn't confounded by a page-table bug.
- **MoE-lane layer coverage**: on Qwen3.6-35B-A3B only ~25% of layers are
  full-attn, so the quant-KV memory/perf delta there is bounded to those layers —
  don't read the dense-lane win as the MoE-lane win.
- **Budget interaction**: the quant width flows into `per_slot_kv_bytes` → the
  unified `SlotBudget`; verify slot count rises ~proportionally to the byte
  saving and that the C2 clamp / C4 reject messages still name real knobs.
