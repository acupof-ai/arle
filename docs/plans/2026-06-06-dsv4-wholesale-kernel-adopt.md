# DSv4 wholesale vendored-kernel adoption → SGLang-on-H20 (adopt, no rewrite)

**Date:** 2026-06-06. **Target:** SGLang-on-H20 parity (6 ms is an H100 number;
H20 floor for this 671B-class FP8 MoE is ~25–40 ms/token, decode GPU-bound).
**Principle:** every lever is wiring a *vendored* kernel — zero hand-rewrites.
**Verification (per ckl):** baseline already measured (~33 tok/s eager); test the
**optimized config directly** and compare to the known baseline — no baseline
A/B re-run. **And DSv4 is run-to-run non-deterministic at 4096**
([[reference_dsv4_moe_nondeterminism_confounds_4096_parity]]) — correctness =
needle/greedy-lossless at the deterministic short shape, never 4096-token-exact.

Synthesizes the 4 parallel adoption specs (the workflow synth 529'd; specs
recovered from the agent transcripts).

## Sequence (cheap flips → FFI wires → big adopt)

### 0. [LANDED] gpu-router default-on — everything-on-GPU (`a4d8bee6`).

### 1. Gate-flips (already vendored + wired, just gated) — licensing in flight
- **FlashMLA-decode** (`ARLE_DSV4_FLASHMLA_DECODE`, +18%, biggest kernel bucket):
  flip default after the kv-precision-parity gate is green.
- **contiguous-MoE** (`ARLE_DSV4_MOE_CONTIG_DECODE`, +13%): flip default after the
  decode-optimized-direct check (token-exact at 64-tok deterministic shape + tok/s ≥ baseline).
- Both: one-line gate flips mirroring `use_gpu_router`; then delete the masked/unpad
  + host-route fallbacks (no half-states).

### 2. mhc-fuse (`mhc_pre_big_fuse` TileLang — vendored, ZERO FFI today) — WRITABLE NOW
Collapses 3 scalar HC launches/sub-block (`gen_mhc_params` GEMM + 2 mhc launches) → 2
fused TileLang kernels. Files:
- `crates/cuda-kernels/tools/tilelang/mhc_pre_big_fuse.py` (new adapter exposing `get_kernel`)
- `tools/tilelang/gen_tilelang_aot.py` (`mhc` family + 2 `WrapperSpec`, `block=96,1,1`)
- `build.rs` (decl consts + `mhc_specs` table + stub entries)
- `crates/cuda-kernels/src/ffi/misc.rs` (2 extern fns: `mhc_pre_norm_fn_fwd_mul_cuda`, `mhc_pre_big_fuse_cuda`)
- `crates/infer-cuda/src/hc.rs` (`hc_pre_fused` + `hc_pre_fused_enabled` on `ARLE_DSV4_HC_FUSE`)
- `crates/infer-cuda/src/dsv4.rs` (3 eager-prefill call-sites at ~706/780, gated; decode-graph B=1 paths out of scope)
- **Dtype gotcha:** `mix_fn`/`base`/`scale` are stored bf16 but the fused kernel
  declares f32 → add a one-shot bf16→f32 mirror at load via `arle_bf16_to_f32_cuda`
  (cheapest, no kernel edit). Keep `keepalive` on `layer_input`/`post`/`comb`
  ([[reference_disabled_event_tracking_premature_buffer_free]]).
- Verify: needle/greedy-lossless at deterministic shape + decode-optimized-direct tok/s vs baseline.

### 3. EAGLE/MTP (DSv4 ships `num_nextn_predict_layers=1`; load skipped today) — WRITABLE after 2 control experiments
Load the shipped MTP head (`deepseek-spec` `mtp_tensor_names` scaffold exists) + the
greedy-lossless draft-verify loop (1 base forward over [last, draft] verify batch →
accept if draft==argmax → 2 tokens/forward). Files: `dsv4.rs` (Dsv4MtpLayer + load +
`mtp_forward` + `forward_tokens_with_hidden`), `executor.rs` (mtp_slots + verify loop
in `submit`), `attention.rs` (KV rollback in `advance_decode_len`), scheduler seq_len
accounting (+2-on-accept). Gate `ARLE_DSV4_SPEC_DECODE`, mutually exclusive with
decode-graph/deepep for v1.
- **Two design unknowns — control-experiment FIRST (§0):** (1) MLA KV rollback on
  reject produces bit-identical state to a fresh forward; (2) host/device seq_len
  lockstep when 2 tokens land/step. Both are cheap-experiment-verifiable; coding the
  loop before they pass is a guess.
- Acceptance gate: greedy-lossless parity (spec-on == spec-off greedy, non-negotiable)
  + measured α on the SLO shape + wall-clock tok/s Δ. ~1.9× is the α≈1 ceiling, not the claim.

### 4. DeepEP-LL (low-latency dispatch/combine) — BLOCKED, prereqs first
NOT a blind copy. Prereqs before any code:
- (a) **Process-model gate:** NVSHMEM same-process init timed out (May errors); LL is
  downstream of the multi-process launcher — confirm the current multiproc launcher
  clears the same-process NVSHMEM init gate.
- (b) **Read real signatures** on the pod: `csrc/kernels/legacy/internode_ll.cu` +
  `deep_ep/buffer.py` — lock arg order/names before writing the C wrapper
  (`feedback_deepep_kernel_api_inverted_naming` / `_combine_uses_recv_channel_prefix`).
- Then: `deepep-sys` LL params + `dispatch_ll`/`combine_ll` (drop `-DDISABLE_NVSHMEM`),
  `deepep.rs` second LL buffer + drop defensive syncs (171/244), `moe.rs`
  `dsv4_moe_forward_deepep_ll` (FP8-once → DeepGEMM masked GEMM), `dsv4.rs` seq_len==1→LL.
- Bonus: LL packed layout also kills the `recv_topk_idx` D2H (`moe.rs:1589`, per-layer).
- ~1500–2000 LOC. Largest piece, real blocker — sequence last.

## Driving
Perf/design = Claude (these are all perf). Codex = deterministic build/test +
pod-side source reads (internode_ll signatures, builds, optimized-direct runs).
Verify each optimized config directly vs the known baseline; no baseline re-runs.
