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
- **Two design unknowns — GROUNDED 2026-06-06 (both tractable; control-experiment still gates landing):**
  1. **KV rollback on reject** — `advance_decode_len` (attention.rs:411) is just
     *logical counters* (`compressor/indexer.compressed.seq_len = total_len/ratio`).
     So rollback is: SW ring **self-heals** (the next real token overwrites slot
     `pos%W`); compressor/indexer counters **recompute** from the accepted `total_len`;
     only the FlashMLA FP8-pool packed counters (`fp8_kv_comp_packed_rows`) need an
     explicit decrement. Control experiment: spec-write pos p+1 → reject → next real
     forward at p+1 is bit-identical to a no-spec forward.
  2. **seq_len lockstep** — decode today does `kv.alloc_tokens(slot,1)` +
     `position=kv_seq_len+1` + asserts `kv.seq_len(slot)==row.kv_seq_len`
     (executor.rs:356-371), emits exactly 1 `SlotToken`. EAGLE: `alloc_tokens(slot,2)`
     on the verify batch, emit 1-or-2 `SlotToken`s, on reject `free`/decrement the 2nd,
     scheduler advances `row.kv_seq_len` by the emitted count; the existing
     `kv.seq_len==row.kv_seq_len` assertion is the consistency gate. Control experiment:
     drive accept(+2) then reject(+1) and assert materialized==logical each step.
  Coding the verify loop before these two controls pass is still a guess (§0).
- **Phase 1 LANDED** (`2e0cde16`, gated `ARLE_DSV4_SPEC_DECODE`): `Dsv4MtpLayer` load
  (dsv4.rs:206/607), `mtp_forward` (dsv4.rs:1032), `forward_tokens_with_hidden`
  (dsv4.rs:687); validated on the TP=8/EP=8 pod (MTP loads, drafts `base=11111
  draft=16`, no crash). The executor seam is the one-shot probe at executor.rs:608.
- **Phase 2 spec (grounded in the real primitives):**
  - Rollback primitive EXISTS: `truncate_slot(slot, new_len)` (lib.rs:258) rolls the
    host+device KV back; pair it with `advance_decode_len(mode, ratio, accepted_total)`
    (attention.rs:411, resets compressor/indexer `compressed.seq_len`) + a FP8-pool
    `fp8_kv_comp_packed_rows` decrement → the full per-slot rollback helper.
  - Verify loop (replaces the executor.rs:608 probe, gated): when `pending_draft`
    is Some — `kv.alloc_tokens(slot, 2)`; forward the 2-token batch `[last_accepted,
    draft]` (a new "decode-2" forward that returns argmax at BOTH positions + the
    hidden); `real_next = argmax@pos`. **Accept** (`draft == real_next`): emit
    `[real_next, real_next2]`, both KV positions stay, re-draft from
    `mtp_forward(hidden, real_next2)`. **Reject**: `truncate_slot(slot, kv_seq_len+1)`
    + counter rollback, emit `[real_next]`, re-draft from `mtp_forward(hidden,
    real_next)`. Greedy accept is lossless (verify is the base model's own forward).
  - Cross-crate: `StepOutput.tokens` is already a `Vec`, so emit-1-or-2 is
    schema-compatible, but the scheduler must advance `row.kv_seq_len` by the emitted
    count and the `kv.seq_len(slot)==row.kv_seq_len` assertion (executor.rs:356) must
    hold across +2 — the lockstep control experiment gates this.
  - Control experiments first: (1) rollback parity — forward `[..tN]` then a wrong
    speculative `tN+1` then `truncate_slot(slot, N+1)`+counter-reset, assert KV state
    bit-identical to a no-spec forward to N+1; (2) emit-1-or-2 seq_len lockstep.
  - Verify (DSv4 non-deterministic → not 4096-token-exact): greedy-lossless at the
    deterministic short shape (spec-on == spec-off) + measured α + wall tok/s Δ.
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
