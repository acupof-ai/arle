# Qwen3.6-35B MoE FFN → SGLang fused_moe triton lane (U3) — opt-in, 2 load-path bugs fixed, A/B pending-remote

**Date:** 2026-06-12 (impl); **2026-06-13** OOM root-cause + build-and-replace
fix + expert-count validator fix.
**Backend:** CUDA, Qwen3.6-35B-A3B, H20, TP=1.
**Status:** `pending-remote` — code complete + Mac-typecheck/clippy green. The
#88 validate3 pod pass found two load-path bugs in sequence: a lazy-build
**OOM** (memory doubling) → build-and-replace at load (`59cea517`,
resolved-design-question #1); then the freed-Vecs form tripped
`moe_forward_into`'s expert-count precheck → `|| fused_sglang.is_some()`
(`ad39dc77`, codex-confirmed). Both fixed. The no-OOM confirm +
needle-envelope + dgoff-vs-moe perf A/B re-run is the remaining pending-remote
item — **deferred: all 8 H20s occupied by the lead's DSv4 TP=8/EP=8 serve
(~54 GB/GPU 2026-06-13); the 35B BF16 needs a free single GPU.** No default flip
lands without that pass. **Lowest-confidence swap of the four** — bar is
correctness + clean opt-in + byte-identical flag-OFF; no perf claim is made here.

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
- **Stacked weights — build-and-replace at LOAD** (`crates/infer-cuda/src/loader.rs`):
  `Option<MoeFusedSglangWeights>` on `MoeLayerWeights`. `w1 [E, 2N, K]`
  (gate rows then up rows per expert — matches SGLang stacked `w13_weight`),
  `w2 [E, K, N]` (down verbatim). Pure D2D gather-concat from the per-expert
  `DeviceMatrix` Vecs, **then the Vecs are freed** (`ctx.sync()?` →
  `gate/up/down.clear()`), exactly mirroring the DeepGEMM grouped block in the
  same loader. Restacked size ≈ source size (w1 ≈ 256·1024·2048·2 B ≈ 1.0 GB,
  w2 ≈ 256·2048·512·2 B ≈ 0.5 GB/layer) — so resident routed-expert VRAM does
  **not** grow. **Superseded the first-forward-lazy `OnceCell` design**, which
  OOM'd the 35B BF16 shard (see the OOM resolved-design-question below). The
  fused lane reads w1/w2 directly; the per-expert ptr tables are empty on this
  path. Mutually exclusive with DeepGEMM (the grouped cache already freed the
  Vecs) — needs `ARLE_QWEN35_DEEPGEMM=0`; both-on loud-fails at the first
  forward via the cache getter.
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

- **Lazy first-forward build OOM'd → build-and-replace at load (caught in pod
  validation, #88 validate3):** the original `OnceCell` "materialize on first
  fused forward" design doubled routed-expert VRAM. The fused `w1 [E,2N,K]` +
  `w2 [E,K,N]` BF16 cache is a SECOND full copy of the MoE weights (~1.6 GB/layer:
  w1 1.07 GB + w2 0.54 GB) built lazily per layer ON TOP of the still-resident
  per-expert Vecs (DeepGEMM-off keeps them). On the 35B-A3B single-GPU shard
  (~70 GB weights on a 96 GB H20, ~30 GB free after load), the lazy build hit
  `CUDA_ERROR_OUT_OF_MEMORY` around layer 20 (20 × 1.6 GB ≈ 32 GB > 30 GB free).
  The lazy build takes `weights: &MoeLayerWeights` (immutable, pool-shared
  `&self`) so it **cannot** free the Vecs — the fix had to move to the loader
  where the Vecs are owned `mut`. Now `MoeFusedSglangWeights::build` runs in
  `load_moe_layer_experts` (mirror of the DeepGEMM `concat → ctx.sync()? →
  gate/up/down.clear()` block): restack → sync → free Vecs per layer. Resident
  = restacked size (replacing the freed Vecs); peak = baseline + one layer's
  transient (~1.6 GB) → fits. The fused forward's `build_fused_sglang_weights`
  is now a pure cache getter (`fused_sglang.as_ref()`), and the forward gate
  loud-fails if the flag is on for a non-device-route-eligible config (the
  non-fused fallback is gone — its Vecs were freed). The doc comment's
  "+1.5 GB total" was wrong by ~40× — it is +1.5 GB *per layer* before the fix,
  and net-zero growth after.
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
- **moe_align expert-id +1 shift (caught + fixed in review, codex P1):** the
  vendored `moe_align_block_size` kernel reindexes `expert_id = topk_ids[i] + 1`
  (a dummy expert-0 slot; the E real experts occupy `[1, E]`). The device code is
  byte-equal to upstream — but upstream is correct **only because its python
  caller pads the count**: `sgl_moe_align_block_size(topk_ids, num_experts + 1,
  …)` with `cumsum_buffer = empty(num_experts + 2)`. The first cut of our Rust
  caller passed the bare `E` (256) and sized cumsum `E+1`. With `num_experts=E`
  the kernel's `shared_counts[E]` (where the highest expert 255's count lands,
  since `255+1=256`) aliases `prefix[0]` — never zeroed, never read into the scan
  → **expert 255 silently dropped**, every token routed to it produces zero MoE
  output. Resolved by tracing the python wrapper
  (`moe_runner/triton_utils/moe_align_block_size.py:79`): pass `align_experts =
  num_experts + 1` to the kernel, size `fm_cumsum` to `num_experts + 2`, and
  compute `max_padded` over `num_experts + 1` block-groups. The `.cu` header now
  documents the load-bearing `+1` convention so the next caller can't re-strip it.
- **top_k==8 baked-cubin guard (caught + fixed in review, codex P2):** the GEMM1
  cubin bakes `top_k=8` (A's row offset = `offs_token // top_k`); the fused
  dispatch branch previously ran for any greedy MoE config. Qwen3.6-A3B *is*
  top_k=8 so the intended model was fine, but a model with a different top_k
  would silently mis-index activation rows. Added an `ensure!(topk == 8, …)`
  loud-fail alongside the existing `ep_size==1` / `routed_scaling_factor==1.0`
  cubin-bake guards.
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

**validate3 progress (2026-06-13):** lane-RUNS probe fired (fused_moe lane
engaged). Two load-path bugs were surfaced in sequence and **both fixed**: (1)
the lazy-build **OOM** (memory doubling) → build-and-replace at load (`59cea517`);
(2) with that fixed, the re-run still failed `MoE expert count mismatch: gate=0
up=0 down=0` because `moe_forward_into`'s `!use_deepgemm` precheck rejected the
freed-Vecs form → added `|| weights.fused_sglang.is_some()` to the validator
(`ad39dc77`). Codex review independently flagged bug 2 and prescribed the same
fix. Both Mac-typecheck/clippy green; pod tree staged + rebuilt non-stale. The
remaining items 1–3 re-run once the H20s free (lead's DSv4 8×H20 serve holds all
8 GPUs at 100% util; the 35B BF16 single-GPU validation has no room until then;
per "等等 h20 的使用" no GPU is grabbed mid-run). The U3 control is `dgoff`
(`ARLE_QWEN35_DEEPGEMM=0`, hand-grouped) vs treatment `moe`
(`ARLE_QWEN35_DEEPGEMM=0 ARLE_QWEN35_MOE_FUSED_SGLANG=1`) — both DeepGEMM-off so
the A/B isolates the fused-kernel swap (DeepGEMM gives no benefit at this shape;
see the 2026-06-13 validation entry).

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

**Byte-equal device code ≠ ported contract.** A vendored `.cu` whose kernel
body is identical to upstream can still be wrong if the *host invocation
convention* didn't come with it. `moe_align`'s `+1` expert shift is correct only
because upstream's python caller passes `num_experts + 1` and sizes
`cumsum_buffer = num_experts + 2`; the kernel device code carries no trace of
that requirement. When porting a kernel, port its caller's argument derivation
(padding, sizing, shift conventions) as a first-class part of the contract, and
document the load-bearing convention in the vendored file's header so a future
caller can't silently re-strip it. Diff the *python/host wrapper*, not just the
`.cu`.

**A transposed activation stride is a c=1-invisible bug.** When a new GEMM lane
takes the per-token (M) stride as a runtime arg, derive it from a *validated*
kernel that reads the SAME buffer (here `dsv4_pack_local_experts_kernel` reads
`normed` at `token*hidden_dim + col` → token-major), never from a verbal layout
guess. `stride_am=1, stride_ak=num_tokens` and the correct `stride_am=hidden_dim,
stride_ak=1` are identical at `num_tokens=1`, so a c=1 needle can pass on a
silently-transposed lane — the correctness gate for any batched-GEMM stride
change MUST include a c≥2 shape.
