# Board #4 — adopt FA3 hopper fwd for Qwen3.5/3.6 HD256 full attention

**Date:** 2026-06-11. **Verdict: ADOPT FA3, KILL the TileLang HD256 revival.**
Survey sources: flash-attention @ `fc8cbad6` (hopper/), sglang @ `22c7285a`
(qwen3-next attention backend), in-tree FlashMLA shim
(`build.rs:1763`, `arle_flashmla_shim.cu`) as the vendoring precedent.

## Why adopt (per "先抄业界最好的")

- SGLang runs **exactly this shape** (q16/kv2/HD256, causal, partial-rope
  0.25, sigmoid output gate) through FA3: `flash_attn_varlen` prefill +
  `flash_attn_with_kvcache` decode
  (`docs/reviews/2026-06-11-sglang-qwen3next-pd-comparison.md` row 全注意力).
  Gate and partial RoPE live OUTSIDE the kernel — stock FA3 fwd suffices.
- Our `nonpaged_prefill_attention` is 42.1% of prefill GPU time (avg 5.0 ms,
  max 30.8 ms/launch, no tensor cores, ~10× off the 148-TFLOPS roofline) and
  4.5% of decode (44.7 µs × 10 layers)
  (`docs/reviews/2026-06-11-qwen35-post-license-reprofile-rerank.md`).
- The in-tree TileLang HD256 tiles were never Hopper-tuned; "reviving" them
  means re-deriving FA3's TMA+WGMMA warp-specialized pipeline by hand —
  exactly the closed-door anti-pattern.

## Feasibility (verified against source, not vibes)

1. **Torch-free core.** `hopper/flash.h` (224 lines) is a POD
   `Flash_fwd_params` + `cudaStream_t` — zero torch. Entry points are the
   launch templates (`run_mha_fwd_<Arch, T, kHeadDim, ...>` in
   `flash_fwd_launch_template.h`); the torch surface (`flash_api.cpp`) is
   replaced by our shim, same as FlashMLA's `arle_flashmla_shim.cu` replaced
   its `pybind`.
2. **Instantiation pruning.** `FLASHATTENTION_DISABLE_{BACKWARD, SM8X, FP16,
   FP8, HDIM64/96/128/192, HDIMDIFF*, SOFTCAP, LOCAL, APPENDKV, PACKGQA?}`
   leaves ~4–6 `.cu` units: `flash_fwd_hdim256_bf16{,_paged,_paged_split,_split}_sm90.cu`
   + `flash_fwd_combine.cu` + `flash_prepare_scheduler.cu`. Keep VARLEN,
   SPLIT, PAGEDKV (paged feeds the later pool migration); decide PACKGQA by
   measuring decode both ways (q16/kv2 → pack ratio 8 helps small-B decode).
3. **cutlass.** FA3 pins `csrc/cutlass @ 71275920` — vendor that `include/`
   snapshot under `vendor/flash-attention/csrc/cutlass/`, do NOT reuse
   flashmla's `147f5673` pin (cross-version breakage risk for zero savings).
4. **KV layout maps as-is.** Qwen35 full-attn caches are per-slot contiguous
   `max_seq_len × kv_dim` bf16, token-major rows of [kv_heads=2, 256]
   (`qwen35.rs:95-120`) = FA3 kvcache `[1, max_seq, 2, 256]` + `seqused_k`.
   Step 1 needs ZERO layout migration: prefill chunk = one batch=1 varlen-q
   call per layer; decode = one batch=1 call per row (replaces today's
   per-(head,token) hand kernel). Heuristics we must re-implement in the
   shim: `num_splits` selection (lift logic from `flash_api.cpp`, it is
   plain C++ over the params).
5. **Build precedent.** Same `build.rs` shape as FlashMLA: env-gated
   (`ARLE_CUDA_ENABLE_FA3`), sm_90a-only object list, stub-marker check so a
   stale stub object cannot silently link (`build.rs:1584` pattern).

## Plan (atomic steps, each its own commit + pod check)

1. Vendor `hopper/` (kernel headers + the pruned instantiation set) +
   cutlass-pin `include/` under `vendor/flash-attention/`; LICENSE + README
   pin note. No build wiring yet.
2. `arle_fa3_shim.cu`: fill `Flash_fwd_params` from a C ABI struct
   (q/k/v/o pointers, strides, seqused_k, cu_seqlens_q, scale, causal,
   num_splits), dispatch to `run_mha_fwd_`. build.rs gate + marker symbol.
3. Rust FFI + `qwen35.rs` wiring behind `ARLE_QWEN35_FA3` (default OFF):
   prefill path first (42.1% target), decode second.
4. Gate + license: needle ladder ×3 same-config + c=1/2/4 sweep vs the
   down-tile baseline; nsys mechanism check (attention µs/layer at 3k:
   expect ~5 ms → ~0.5–0.8 ms class). Default-flip only on the full bar
   (TTFT and ITL and tok/s, multi-shape).
5. After license: batched-decode attention (one FA3 call over the whole
   decode batch needs pooled or paged KV — fold into the stage-2 batching
   design rather than a separate migration).

## Killed alternatives

- **TileLang HD256 revival** — never Hopper-tuned, duplicates FA3.
- **FlashInfer** — JIT+torch-coupled runtime; AOT mode still heavier than
  6 instantiation units; no advantage over FA3 for this shape on sm_90a.
- **Reusing flashmla's cutlass pin** — saves ~20 MB of vendor at the cost of
  an unvalidated cross-version build matrix.
