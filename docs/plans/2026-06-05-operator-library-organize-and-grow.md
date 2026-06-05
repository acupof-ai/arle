# cuda-kernels operator library — organize + grow (the SGLang-parity operators)

**Date:** 2026-06-05. **Intent:** the kernel library is a *growing operator
library* — keep it organized/elegant **and** add the missing high-performance
operators that close the H20 SGLang gap (decode 39.5 → ~16 ms = 2.5× kernels).
Not a pruning pass. Pairs with `backend-operator-library.md` (the FlashInfer-style
`plan()`/`run()` framework) and the SGLang per-op targets (roadmap §10).

## A. Operators to ADD / complete (the perf-closing work)

Ranked by the measured SGLang H20 per-op gap. Codex implements the `.cu`; this is
the contract + where each slots.

| Operator | Status | Home | SGLang target | Contract |
|---|---|---|---|---|
| **FP8 dense block-scaled GEMM** (attention wq/wkv/wo + compressor + HC) | **adding** (Codex) — DeepGEMM `sm90_fp8_gemm_1d2d` WGMMA, replacing the scalar `dsv4_fp8_gemv_batch` | `csrc/gemm/` (`deepgemm_native.cu` + a dense entry) | 4.94 ms/tok | A[M,K]·B[N,K]ᵀ→D bf16; act (1,128) FP8 + weight (128,128) FP8, FP32 MN-major scales via the native bridge |
| **FlashMLA decode** (splitkv MLA) | **STUBBED** — `attention/arle_flashmla_decode_stubs.cu` + `*_shim.cu` are placeholders; the real `flash_fwd_splitkv_mla` is missing | `csrc/attention/` | 2.02 ms/tok | 576-dim latent (512 NoPE + 64 RoPE), MQA Q-absorb (fold q-heads into seq → read latent once), split-KV + combine; ARLE layers SW/CSA/HCA on top |
| **EAGLE draft + verify** | **missing** (Medusa scaffold only) | new `csrc/spec/` | 1.93× multiplier | draft head fwd + tree-verify; algorithmic, not a single kernel — a separate decode-loop axis after the two above |
| FP8 GEMV decode (M=1) | keep current until proven; route M=1 through the dense GEMM (BW-bound) per the upstream scan — no bespoke GEMV without an ncu GB/s licence | `csrc/gemm/` | — | — |

**Sequence:** FP8 dense GEMM (in progress) → FlashMLA decode (stub → real) →
EAGLE (separate axis). The first two close the 2.5× kernel gap → ~16 ms; EAGLE is
the 1.93× on top → ~8 ms.

## B. 整理 (organization) — make `misc/` not a junk drawer

The biggest navigability problem: `csrc/misc/` holds first-class operators that
belong in named families. Target moves (Codex executes during/after its attention
work, pod-build-verified — `build.rs` globs `csrc/` recursively so a move is
transparent once the `#include`s are fixed):

| File (now in `misc/`) | LOC | → target home |
|---|---|---|
| `dsv4_attention.cu` (the hybrid MLA SW/CSA/HCA core) | 1739 | `csrc/attention/` |
| `dsv4_mhc.cu`, `dsv4_tp_attention_repack.cu` | 434+ | `csrc/attention/` |
| `norm.cu`, `split_qkv.cu`, `elementwise_basic.cu`, `fused_mlp.cu` | — | `csrc/norm/` (new) or `csrc/elementwise/` |
| `sampling.cu` | 632 | `csrc/sampling/` (new) |
| `gated_delta_rule.cu`, `gdr_*` | — | `csrc/gdr/` (new, Qwen3-Next family) |
| `conv1d*.cu` | — | `csrc/conv/` (new) |

After moves: `misc/` should hold only genuine miscellany (`arle_dtype_convert.cu`,
the flashmla shims). Each family dir gets a one-line `README`/header comment
stating its op shapes (per `feedback_file_naming_semantic_alignment`).

## C. Registry — the navigable index (refresh)

`docs/reviews/kernel-registry.md` is **stale** (references deleted
`batch_decode.rs`/`prefill.rs`/`forward.rs`). Rebuild it as the data behind
`plan()` (per `backend-operator-library.md` §5): every live operator → family,
file, op shape, caller, kernel variant. The read-only inventory peer feeds this.

## Constraints / sequencing

- The `.cu` additions + file moves are Codex's (nvcc; the Mac can't compile
  `csrc`). This doc is the **contract + organization** Codex implements against.
- Do moves *with* the attention work (Codex is already in `attention.rs`/
  `dsv4_attention`), not as a separate churn — fold the `dsv4_attention.cu →
  attention/` move into the FlashMLA add.
- No deletion in this pass (per ckl) — the dead-code scan
  (`2026-06-05-cuda-kernels-deadcode-scan.md`) is a *separate, later* decision.
- Each operator add: A/B vs the scalar baseline (double-win prefill+decode),
  16/16 parity, ncu tensor/BW re-check, wall-clock licence.
