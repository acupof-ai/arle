# Qwen3.6-35B MoE FFN → SGLang fused_moe triton lane (U3) — opt-in, pending-remote

**Date:** 2026-06-12. **Backend:** CUDA, Qwen3.6-35B-A3B, H20, TP=1.
**Status:** `pending-remote` — code complete + Mac-typecheck/clippy green; the
on-device needle gate + A/B runs in the **one-shot validation pass** that also
covers U1+U2 (GDN decode) and U4 (fused add-RMSNorm), per ckl's "写完代码一把验证"
directive (#88). No default flip lands without that pass. **Lowest-confidence
swap of the four** — bar is correctness + clean opt-in + byte-identical flag-OFF;
no perf claim is made here (the lead runs the pod A/B).

## Context

Third tranche of the Qwen-lane SGLang kernel alignment
([plan](../../plans/2026-06-12-qwen-lane-kernel-alignment-sglang.md), ckl
directive "kernel 全对齐 sglang"). Qwen3.6-35B-A3B is an MoE (E=256, top_k=8,
hidden=2048, moe_inter=512); the in-tree MoE FFN runs a hand pack → grouped-GEMM
→ scatter/combine path (decode-band weight-read-bound kernels by default).
This tranche adopts SGLang's `fused_moe` triton stack — `moe_align_block_size`
(native CUDA) + `fused_moe_kernel` grouped GEMM ×2 + `act_and_mul` + `moe_sum_reduce`
— behind opt-in `ARLE_QWEN35_MOE_FUSED_SGLANG` (default OFF → byte-for-byte the
current hand decode-band path).

## What landed

- **moe_align (native)** (`crates/cuda-kernels/csrc/moe/moe_align_block_size.cu`):
  vendored SGLang FULL path stripped of torch → C shim
  `arle_moe_align_block_size_cuda`. int4-vectorized pad fill (numel sentinel),
  warp-scan prefix sum, binary-search expert_ids. `cumsum_buffer` +
  `total_tokens_post_pad` are caller-zeroed atomicAdd accumulators. Auto-picked
  up by the recursive `collect_cu_files` in `build.rs`.
- **fused_moe triton AOT** (`crates/cuda-kernels/tools/triton/kernels/`):
  `arle_fused_moe.py` (`fused_moe_kernel`, 51 params — vendored UNMODIFIED, every
  quant branch kept), `arle_fused_moe_act.py` (`act_and_mul_kernel`, silu),
  `arle_fused_moe_sum.py` (`_moe_sum_reduce_kernel`). The one `fused_moe_kernel`
  source compiles to **two cubins**: GEMM1 (top_k=8, MUL_ROUTED_WEIGHT=0) and
  GEMM2 (top_k=1, MUL_ROUTED_WEIGHT=1). `build.rs` U1 triton manifest extended
  to 7 specs (3 GDN + 4 fused MoE); `none`/`None` signature token → constexpr
  None for branch DCE so the bf16 cubin drops all quant/TMA/bias args.
  **bf16-only for now** — fp8 is a 1-shape AOT signature add (follow-up).
- **FFI** (`crates/cuda-kernels/src/ffi/triton.rs` + `ffi/moe.rs`): 4 fused
  triton externs (+`_load_cuda`) + the moe_align extern. Grid (gX,gY,gZ) first /
  stream last, block dims baked.
- **Safe wrappers** (`crates/cuda-kernels/src/moe.rs`): `moe_align_block_size`,
  `fused_moe_gemm1/gemm2/act_and_mul/sum_reduce`, `fused_moe_max_num_tokens_padded`,
  block-size consts. `.result()?` → CUresult; NOT_SUPPORTED maps to the loud
  opt-in failure in the caller.
- **Lazy stacked weights** (`crates/infer-cuda/src/loader.rs`): per-layer
  `OnceCell<MoeFusedSglangWeights>` on `MoeLayerWeights`. `w1 [E, 2N, K]`
  (gate rows then up rows per expert — matches SGLang stacked `w13_weight`),
  `w2 [E, K, N]` (down verbatim). Pure D2D gather-concat from the existing
  per-expert `DeviceMatrix` Vecs; **+~1.5 GB** on the single-GPU shard
  (w1 ≈ 256·1024·2048·2 B ≈ 1.0 GB, w2 ≈ 256·2048·512·2 B ≈ 0.5 GB),
  materialized only on the first fused forward → **load-time byte-identical**.
  Single-threaded `!Sync` executor → `OnceCell` over `&self` weights is sound.
- **Fused forward + dispatch** (`crates/infer-cuda/src/moe.rs`):
  `moe_forward_fused_sglang` runs moe_align → GEMM1 (`a`=normed, **token-major
  `[T, K]`** → `stride_am=hidden_dim, stride_ak=1`) → act_and_mul → GEMM2
  (carries route_weights, MUL_ROUTED_WEIGHT) → sum_reduce (cache3 `[T, topk, K]`
  → out `[T, K]` via `output_stride_0=hidden_dim, output_stride_1=1`) →
  `add_shared_expert_gated` (reused). Gated in
  `moe_forward_into` after step 2 route: `qwen35_moe_fused_sglang_enabled() &&
  device_route_eligible(cfg)`. **Single-GPU only** (`ep_size==1` loud-fail — the
  lane consumes GLOBAL expert ids directly). `routed_scaling_factor == 1.0`
  asserted (baked in the sum cubin). 7 new scratch slots
  (`fm_sorted_token_ids/expert_ids/total_padded/cumsum/cache1/cache2/cache3`),
  added to `release()`, no per-step alloc (exact-shape `SliceSlot` reuse;
  decode `numel = topk` hits the cache every step). Write-before-read proof
  table documented per slot.

## Resolved design questions

- **Activation layout (caught + fixed in review):** the `HiddenStates` buffer
  fed to GEMM1 (`normed`) and written by sum_reduce (`out`) is **token-major /
  token-contiguous** — element (token t, feature k) at `t*hidden_dim + k`, so a
  token's K features are contiguous. The first cut of this lane assumed it was
  "hidden-major `[K, T]`" and passed transposed strides (`stride_am=1,
  stride_ak=num_tokens`; `output_stride_0=1, output_stride_1=num_tokens`). That
  silently works at `num_tokens=1` (c=1) — `stride_ak=num_tokens=1` happens to
  equal the correct `stride_ak=1` — but **corrupts c≥2 batched decode and all
  prefill**. A c=1-only needle would have passed and shipped the corruption.
  Corrected to `stride_am=hidden_dim, stride_ak=1` (GEMM1 A) and
  `output_stride_0=hidden_dim, output_stride_1=1` (sum_reduce out). Confirmed
  against THREE independent ground-truth kernels on the same buffer:
  `rms_norm_batched_kernel` (`x + blockIdx.x*hidden_dim`), the hand grouped-GEMM
  (`input + route*K + feature`), and the validated hand pack kernel
  `dsv4_pack_local_experts_kernel` (`hidden[token*hidden_dim + col]`). Internal
  caches (cache1/2/3) and the repacked weights were already correct.
- **Weight layout:** stacked per-expert (NOT grouped), gate-then-up (NOT
  interleaved), down verbatim. Repack = pure D2D gather-concat, +1.5 GB.
- **topk_ids/topk_weights tap:** step-2 `dsv4_route` output (token-major
  `[T*topk]` global ids + renormed weights) == SGLang's `topk_ids`/`topk_weights`
  exactly. Single-GPU → global == local.
- **MUL_ROUTED_WEIGHT stage:** GEMM2 (down) carries it (top_k=1 baked).
- **eager-only:** wired through `moe_forward_into`, which both the prefill and
  batched-decode call sites already drive; CUDA-graph capture wiring is a
  follow-up (A/B runs eager-vs-eager). The `_load_cuda` symbols exist for the
  capture-safe path when wired.

## Flag-OFF byte-identity

CONFIRMED. The only code reaching ahead of the original step 3 when the flag is
OFF is the two `cache_ptr` raw-ptr conversions, which the prior code already
computed at the same point (moved one line earlier, side-effect-free — no
launch / alloc / mutation). The fused branch is skipped entirely; the hand
decode-band path is byte-for-byte unchanged.

## Verification (Mac, no nvcc)

- `CUDARC_CUDA_VERSION=12080 cargo check -p infer-api --release
  --no-default-features --features cuda,no-cuda --lib` — Finished, 0 errors.
- `cargo check -p agent-infer --release --no-default-features
  --features cpu,no-cuda,cli` — Finished (fused path is `cfg(cuda)`-gated, excluded).
- `cargo clippy -p infer-cuda ... --features cuda,no-cuda -- -D warnings` —
  Finished, 0 warnings.
- `cargo clippy -p cuda-kernels ... --features no-cuda -- -D warnings` —
  Finished, 0 warnings.
- build.rs no-cuda path skips kernel compile (links stubs) — confirmed.

## Pending (one-shot pod pass, #88)

1. Build on 8×H20 pod with `INFER_TRITON_PYTHON` set (`CARGO_NET_OFFLINE=1`);
   confirm the 4 fused cubins compile (not stubs) via the runtime loud-fail probe.
2. Needle gate ×3 DET vs the locked 2026-06-12 envelope (len 2000/8000 exact) —
   MoE non-determinism confound: gate on needle + same-config-twice floor +
   self-consistency, NOT byte-identity. **MUST include a c≥2 (batched-decode)
   correctness check, not just c=1** — the stride class of bug fixed in review is
   c=1-invisible (transposed activation strides coincide at `num_tokens=1`); a
   batched needle or a coherence check on a c=4 run is the gate that catches it.
   The perf A/B (item 3) measures tok/s, not output correctness, so it does
   **not** substitute for the c≥2 correctness check.
3. Same-binary same-shell A/B `ARLE_QWEN35_MOE_FUSED_SGLANG` OFF vs ON, c=1/2/4/8,
   vs the locked baseline. Δ% per c. License-or-kill on wall-clock per shape; a
   losing A/B keeps the lane opt-in with the verdict recorded. Note +1.5 GB
   device cost in the ON arm.
4. fp8 cubin (1-shape signature add) is a separate follow-on once the bf16 lane
   is licensed.

## Rule

Lowest-confidence kernel swaps land opt-in + flag-OFF byte-identical + correctness
gate first; vendor triton source UNMODIFIED (every quant branch kept, compile only
the bf16 cubin); enumerate EVERY scratch slot with a write-before-read proof and a
no-per-step-alloc guarantee before claiming the lane is wired. Perf license comes
from the pod wall-clock A/B, never the source survey.

**A transposed activation stride is a c=1-invisible bug.** When a new GEMM lane
takes the per-token (M) stride as a runtime arg, derive it from a *validated*
kernel that reads the SAME buffer (here `dsv4_pack_local_experts_kernel` reads
`normed` at `token*hidden_dim + col` → token-major), never from a verbal layout
guess. `stride_am=1, stride_ak=num_tokens` and the correct `stride_am=hidden_dim,
stride_ak=1` are identical at `num_tokens=1`, so a c=1 needle can pass on a
silently-transposed lane — the correctness gate for any batched-GEMM stride
change MUST include a c≥2 shape.
