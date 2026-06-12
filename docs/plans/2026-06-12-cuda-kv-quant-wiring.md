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

## Current state — SOLID re-grounding (code-read 2026-06-12, supersedes the first draft)

The first draft of this plan claimed "the Qwen paths run nonpaged full-attention
today; the quant path brings paged attention to them." **Reading the source
refutes that for the dense lane** and changes the whole T3 sizing. The corrected
truth, per file+line:

- **The dense Qwen3 path is ALREADY paged.** `QwenCudaExecutor` holds
  `kv: PagedKVPool` (`executor.rs:410`); `paged_attention` (`attention.rs:2469`)
  → `prefill_attention`/`decode_attention` → `run_tilelang_paged`
  (`attention.rs:2676`) read the pool through FlashInfer-style
  `meta.kv_indices` + `meta.kv_indptr` page tables, built in `loader.rs:274-322`
  + `decode_graph.rs`. So the page-table build + paged read **already exist** for
  dense Qwen3; the "bring paged attention to a nonpaged model" work does NOT
  apply here.
- **`PagedKVPool` is ALREADY format-aware.** It carries `format: KVFormat`
  (`{BF16, INT8, FP8E4M3, INT4, TurboQuant{..}, PackedBytes{..}}`,
  `paged_kv.rs`); BF16/INT8/FP8E4M3 all share `page_size = 16`
  (`paged_kv.rs:2601-2603`); per-token byte sizing is already format-branched
  (`paged_kv.rs:413-415`); a `KVCacheDtype → KVFormat` map exists for BF16/INT8
  (`paged_kv.rs:357-358`). `from_qwen3_bf16_safetensors` simply **hardcodes
  `KVFormat::BF16`** at its two pool-construction sites (`executor.rs:491,500`).
- **The dense store is two kernels, and only the second changes.**
  `prefill_attention_paged_prep_cuda` (`prefill_attention_paged_prep.cu`) runs
  (1) `prefill_attention_paged_qk_norm_rope_hd128_kernel` — RMSNorm+RoPE applied
  **in place** on `q_batch`/`k_batch` (the kernel writes the RoPE'd value back to
  the same buffer, line 79), then (2)
  `prefill_attention_paged_kv_write_hd128_kernel` — scatters the now-RoPE'd
  `k_batch` + raw `v_batch` into the **BF16** pool through the page table
  (lines 108-109). Decode mirrors this via `decode_prep_paged_cuda`. So the quant
  store is a **drop-in replacement of kernel (2) only**: keep RoPE-in-place, then
  call `quantize_paged_kv_<dtype>_per_channel(k_batch, v_batch → pool,
  k_static_scales, new_token_indices)`.
- **`Qwen3.5/3.6 MoE KV is the genuinely nonpaged lane.** `Qwen35SlotState`
  (`qwen35.rs:14` — "this model OWNS its KV state (no `PagedKVPool`)") holds
  contiguous BF16 per-layer caches (`k_caches/v_caches: Vec<DeviceVec>`,
  `qwen35.rs:147-186`); `per_slot_kv_bytes()` hard-codes the BF16 budget. Only
  the `full_attention` layers (1-in-4, `full_attention_interval: 4`) have caches.
  **This is the lane where "bring paging to a nonpaged model" really applies** —
  a materially bigger diff, deliberately split off (see lane order below).
- **DSv4 FP8 KV is a bespoke MLA latent arena** (`Dsv4MlaKvArena`, 584 B/token,
  FlashMLA MODEL1; `dsv4.rs`) — model-specific, not the generic substrate.
- **The quant kernels + Rust wrappers exist, unwired into any executor.**
  `kv_quant.rs`: `quantize_paged_kv_{int8,fp8,int4}_per_channel`, the
  `compute_k_per_channel_absmax` → `finalize_k_per_channel_scales{,_int8,_int4}`
  calibration, `dequantize_paged_kv_{int8,fp8}_to_hnd`, the fused
  `decode_attention_{int8,fp8}_per_channel_k` (read side takes the SAME
  `kv_indices` + `kv_meta`(=kv_indptr) + pool ptrs + `k_static_scales` +
  `v_scales` + workspace). `kv_turboquant.rs`: TQ4
  (`turboquant_quantize_paged_single`, `turboquant_fused_decode_attention`),
  `turboquant_state.rs` (rotations + Lloyd-Max codebook).

**Net: the substrate is far more complete than the first draft assumed.** For the
dense lane the gap is purely executor glue: thread the format into pool creation,
replace the BF16 store-scatter with the quant-scatter, swap the BF16 read kernel
for the quant read kernel, and manage the per-channel scale buffers. No new paged
attention, no new page-table build.

## Model routing on the pod (verified 2026-06-12)

`classify_cuda_model` (`loaded.rs:99-150`) maps the HF config to the executor:

| On-pod model | arch / `model_type` | CUDA path | KV state | Gate quality |
|---|---|---|---|---|
| `Qwen3-0.6B` | `Qwen3ForCausalLM` / `qwen3` (dense) | `from_qwen3_bf16_safetensors` (`QwenCudaExecutor`) | **already paged** (`PagedKVPool`), all layers full-attn, no MoE non-det | cleanest correctness signal; 0.6B may miss long needles |
| `Qwen3.6-35B-A3B` | `Qwen3_5MoeForConditionalGeneration` / `qwen3_5_moe` | `from_qwen35_moe_safetensors` (`Qwen35CudaExecutor` → `Qwen35SlotState`) | **nonpaged** contiguous caches; only 1-in-4 layers full-attn; MoE non-det widens the gate | the issue's literal target family; needs paging first; slow boot |

**There is no `/data01/models/Qwen3.5-4B` on the pod.** The qwen3_5_moe family is
present as `Qwen3.6-35B-A3B`.

## Architecture decision — SETTLED (dense lane needs only glue; quant pool layout already chosen)

The quant kernels are FlashInfer-style page-table-driven (read via
`kv_indices`/`kv_indptr`, write into a page-blocked NHD quant pool with
`kQuantPageSize = 16`). The dense Qwen3 pool is *already* that layout in BF16
(`[max_total_pages, num_kv_heads, page_size, head_dim]`, `page_size = 16`,
`paged_kv.rs:40-42`), addressed by the same `kv_indices`/`kv_indptr`. So the
quant pool is the **same pool with `format != BF16`** — `PagedKVPool` already
sizes its bytes per-format. No layout migration; flip the format at construction.

One subtlety in the write kernel: `quantize_paged_kv_*_per_channel`
(`csrc/kv/kv_quant.cu`) computes `page_idx = token_row / kQuantPageSize`, i.e. it
assumes an **identity** logical→physical page map (the DSv4 case). The dense pool
uses a **real** (non-identity) page table, so pass `new_token_indices` =
**physical flat rows** (`physical_page * page_size + token_in_page`) rather than
logical positions, and the contiguous-assumption write lands in the right
physical slot. That physical-row builder is the one piece of new host code.

Page-table reuse — `dsv4_page_table.rs` core math (`physical_page` logical→
physical lookup; `contiguous_page_table_byte_range` contiguity proof) is
**shape-agnostic** (operates on a generic `table: &[u32]` + `page_bytes`; only
the error strings + the 584 B test fixture are DSv4-flavored; pure host code,
CPU-testable). **Lift it to a backend-neutral `paged_kv_table.rs`** (rename the
DSv4 strings) per the unified-abstraction rule, and add the physical-row builder
there (CPU-testable, same discipline).

Scope guard: the host seam is unchanged — `CudaKvPool = HostPagedKvPool` already
allocates page ids; the device quant pool format + the physical-row builder are
backend-internal CUDA details below the seam. T1/T2/T4 untouched.

## Which gate lane first — dense Qwen3 (already paged → small diff), MoE deferred

1. **Dense Qwen3** (`QwenCudaExecutor`): already paged, all layers full-attn, no
   MoE non-determinism, fast boot → the smallest diff, the tightest gate
   envelope, and the fastest iteration on the fraught FP8 path. The gate is
   "quant within the same model's BF16 same-config envelope," so the 0.6B's
   absolute miss-rate doesn't matter — the BF16 run on the *same* model is the
   reference. If the 0.6B BF16 envelope is too narrow to detect quant
   regressions on the long rungs, fetch a stronger dense Qwen3 (Qwen3-4B/8B).
2. **Qwen3.6-35B-A3B** (`Qwen35SlotState`): the issue's literal family, but its
   KV is **nonpaged contiguous caches** — wiring quant there first requires
   bringing paged attention to that path (the big diff the first draft mis-
   attributed to the dense lane). **Deferred to a fast-follow** once the shared
   substrate (format-flip + quant store/read + physical-row builder + scale
   state) is proven on the dense lane. Tracked as a #68 follow-on, not a blocker
   for landing the dense-lane matrix.

Rationale: root-cause on a clean baseline (`§0.1`); the substrate being shared is
what keeps this model-generic rather than a per-model fork — the MoE lane reuses
the exact same quant store/read/scale code once its caches are paged.

## T3 — wire the quant hot path (dense Qwen3 lane, per dtype, each gated by T5)

Per dtype, cheapest correctness first: **INT8 → FP8 → TQ4.** INT4/TQ2/TQ3 stay
report-only (monolith gate precedent). All steps are dense-lane glue on the
already-paged pool — no new paged attention.

0. **T4 carry-over** (when the first non-BF16 dtype lands): add its
   `CudaKvCacheDtype` variant + `resolve` arm, thread `kv_cache_dtype:
   CudaKvCacheDtype` (→ `KVFormat`) through `from_qwen3_bf16_safetensors` into the
   two `PagedKVPool` construction sites (`executor.rs:491,500`, replacing the
   hardcoded `KVFormat::BF16`), and widen the CLI int8-on-CUDA guard
   (`serve.rs`). Deferred from T4 by design.
1. **Pool format flip.** `from_qwen3_bf16_safetensors` constructs the pool with
   the resolved `KVFormat` instead of `KVFormat::BF16`. Byte sizing + page_size
   are already format-driven — nothing else to change pool-side. Bf16 keeps the
   exact existing path (format == BF16 → no quant branch taken anywhere).
2. **Per-layer scale state.** Allocate, per full-attn layer, `k_static_scales:
   DeviceVec<f32>` (length `kv_dim`) and the `v_scales` buffer the decode kernel
   takes. Calibrate K static scales on the prefill pass:
   `compute_k_per_channel_absmax` over the RoPE'd K → `finalize_k_per_channel_
   scales_int8` (INT8) / `finalize_k_per_channel_scales` (FP8). Confirm the V
   scale convention from the decode kernel (per-token dynamic vs per-channel
   static) before wiring — read `decode_attention_int8_per_channel_k` semantics.
3. **Store (replace kernel 2).** After the in-place RMSNorm+RoPE kernel, instead
   of `prefill_attention_paged_kv_write_hd128_kernel` (BF16 scatter), call
   `quantize_paged_kv_<dtype>_per_channel(k_batch, v_batch → pool,
   k_static_scales, new_token_indices = physical rows)`. Decode store mirrors via
   the single-token path. Build `new_token_indices` (physical flat rows) from the
   page table via the lifted `paged_kv_table.rs`.
4. **Decode read (swap).** Replace `run_tilelang_paged` (BF16) with
   `decode_attention_{int8,fp8}_per_channel_k` (TQ4:
   `turboquant_fused_decode_attention` + `turboquant_rotate_query`), passing the
   existing `meta.kv_indices`/`meta.kv_indptr`, the pool ptrs, the scale buffers,
   and a workspace sized by `decode_attention_<dtype>_workspace_bytes`.
5. **Prefill read.** The TileLang prefill kernel reads BF16 from the pool; with a
   quant pool it must read quant. Simplest correct path: after the quant store,
   `dequantize_paged_kv_<dtype>_to_hnd` the touched pages into a BF16 HND work
   buffer and run the existing `run_tilelang_paged` prefill against that buffer
   (a fused quant-prefill kernel is a perf follow-on, not a correctness need).
6. **TQ4 only:** build `TurboQuantLayerState` (rotations + Lloyd-Max codebook) at
   load; rotate Q with `turboquant_rotate_query` before the fused decode.

## T5 — license-or-kill (the gate, built in T1)

Single GPU. BF16 reference first, then each dtype (`<MODEL>` =
`/data01/models/Qwen3-0.6B` for the dense lane, or a fetched Qwen3-4B/8B):

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
- **Physical-row `new_token_indices` builder** is the new correctness surface on
  the dense lane (the write kernel assumes identity pages; the dense pool's table
  is non-identity). Validate it in isolation: a CPU unit test on the lifted
  `paged_kv_table.rs` builder, then an INT8 store→dequant→compare round-trip on a
  short prompt before trusting the full needle ladder.
- **Prefill dequant correctness**: the dequant-to-HND prefill path is new glue;
  confirm an INT8 prefill matches the BF16 prefill on a short prompt before the
  ladder so a prefill bug isn't read as a quant-quality miss.
- **MoE lane is a separate, larger diff** (nonpaged → paged); don't read the
  dense-lane license as the MoE-lane license. Fast-follow, tracked under #68.
- **Budget interaction**: the quant width flows into `per_slot_kv_bytes` → the
  unified `SlotBudget`; verify slot count rises ~proportionally to the byte
  saving and that the C2 clamp / C4 reject messages still name real knobs.
