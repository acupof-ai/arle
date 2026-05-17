# Candle CUDA-Kernel Vendor Survey — Wave 2 Backward Path

**Date**: 2026-05-17 · **Context**: Post-P3.1 (`65f4753`) — tok/s 91→171 on 4070 Ti SUPER. Wave 2 = port 10 `*_backward` device overrides. User asked whether vendoring from `huggingface/candle` (Apache-2.0 / MIT) saves work.

## 1. Summary

- **Candle has zero hand-written CUDA backward kernels.** Backward is graph-composition of forward primitives in pure Rust (`candle-core/src/backprop.rs`). RoPE is explicitly non-differentiable (`apply_op3_no_bwd` in `candle-nn/src/rotary_emb.rs`). "Vendor candle's `*_backward`" is impossible — those kernels do not exist.
- **Candle is a forward pantry.** `reduce.cu` rmsnorm/layernorm/softmax/rope forwards are well-engineered (warp_reduce_sum, f32 accum, templated bf16/f16) — useful as algorithm reference when ARLE extends to bf16 later, but not Wave 2 work.
- **Two hard incompatibilities** even if we tried: (a) candle RoPE uses interleaved `(2i, 2i+1)` pair layout; ARLE uses NeoX rotate-half `(i, i+half_dim)`. (b) candle `index_add`/`scatter_add` deliberately omits `atomicAdd` (assumes unique destinations), unsafe for `embedding_backward` where duplicate token-ids must accumulate.

**Recommendation**: hand-write Wave 2 using the existing `mean_backward.cu` / `log_softmax_last_axis_backward.cu` style. Total ~5-7 eng-days, vs. ~3 net if we tried to vendor (rewriting incompatibilities eats the savings).

## 2. Per-op table

Surveyed: [`candle-kernels/src`](https://github.com/huggingface/candle/tree/main/candle-kernels/src) (files: `affine`, `binary`, `cast`, `conv`, `fill`, `indexing`, `mmvq_gguf`, `quantized`, `reduce` (contains rms/layernorm/softmax/rope), `sort`, `ternary`, `unary`), `candle-nn/src`, `candle-core/src/backprop.rs`.

| Wave 2 op | Candle forward | Candle bwd kernel | ARLE compatibility | Recommendation | Effort |
|---|---|---|---|---|---|
| `rms_norm_backward` | `reduce.cu::rmsnorm<T>` (warp_reduce_sum, f32 accum, llama.cpp-derived) | **no** — composed from `sqr/sum/sqrt/mul` | math identical; ARLE forward uses tree-reduce, candle uses warp_reduce_sum | **hand-write**; cite candle for bf16 extension later | M (2-pass: sum(grad·x) for d_rms, then d_weight reduce) |
| `rope_backward` | `reduce.cu::rope<T>`, `ropei<T>`, `rope_thd<T>` | **no** — `apply_op3_no_bwd` | layout mismatch: candle interleaved `(2i,2i+1)`, ARLE NeoX `(i,i+half)` | **hand-write**, mirror ARLE `rope.cu` with `-sin` flip | S (≤40 lines) |
| `silu_backward` | `unary.cu::usilu_*` | **no** | dtype only (candle templated, ARLE f32) | **hand-write** `grad * sig * (1 + x*(1-sig))` | S (≤25 lines) |
| `add_broadcast_backward` | `binary.cu::badd_*` (no broadcast in kernel) | **no** | candle pre-broadcasts to contiguous — does not match ARLE stride-0 model | **hand-write**, sum-reduce broadcast axes | M (reuse `mean_backward` row-reduce shape) |
| `embedding_backward` | `indexing.cu::index_add`/`scatter_add` (no atomics) | **no** | **unsafe to vendor**: candle assumes unique destinations; embedding bwd needs `atomicAdd` for duplicate token-ids | **hand-write** with `atomicAdd` | M (thread per (row, dim_chunk) → atomicAdd) |
| `mul_backward` | `binary.cu::bmul_*` | composes `grad.mul(rhs)` + `grad.mul(lhs)` | trivial | reuse existing `elementwise.cu::mul_f32` × 2 launches | XS (Rust dispatch only) |
| `mul_scalar_backward` | — | shipped in P3 | done | — | — |
| `sum_backward` | `reduce.cu::fast_sum` | **no** | broadcast scalar-or-vector | **hand-write**, near-copy of `mean_backward` without `inv_n` | XS (~20 lines) |
| `neg_backward` | `unary.cu::uneg_*` | **no** | trivial | reuse existing `elementwise.cu::neg_f32` | XS (Rust dispatch only) |
| `exp_backward` | `unary.cu::uexp_*` | **no** | needs saved fwd output | reuse `mul_f32` (`grad * exp_out`) | XS (Rust dispatch only) |
| `scatter_add_rows_backward` | `indexing.cu::index_select` | **no** | matches gather pattern | **hand-write**, mirror existing `gather_last_dim_backward.cu` | S (~30 lines) |

Effort: XS ≤½d, S ≤1d, M ≤2d. **Total: 5 × XS + 3 × S + 3 × M ≈ 5-7 eng-days.**

## 3. License + attribution

Candle is **dual Apache-2.0 + MIT** (`LICENSE-APACHE` + `LICENSE-MIT` at repo root). Both permit vendoring with attribution. We are not recommending verbatim vendoring; if we cherry-pick an algorithm later (e.g. warp_reduce_sum for bf16), header:

```cuda
// Adapted from huggingface/candle (Apache-2.0 / MIT):
//   https://github.com/huggingface/candle/blob/<commit-sha>/candle-kernels/src/reduce.cu
// Original copyright: HuggingFace Inc.
```

Pin a commit SHA for reproducibility.

## 4. Architectural lessons (≤200 words)

Candle is the **opposite** of ARLE's host-authoritative-by-default model — and matches what P3.1 forced ARLE toward empirically:

- **Single-device storage, no `Dirty::Host/Both/Device` tristate.** Each `Tensor` lives in exactly one of `Storage::Cpu | Cuda | Metal`; `to_device()` is an explicit deep copy. The class of bug P3 spent weeks on (silent DtoH ping-pong from host-authoritative gradients) is **structurally impossible** in candle.
- **Backward = forward composition.** `Op::Mul` backward is two `Op::Mul` forwards; RMSNorm backward emerges from `sqr/sum/sqrt/broadcast_mul` graph diff. No `*_backward.cu` directory exists.
- **Cost**: fused-backward optimizations (e.g. ARLE's in-kernel `__expf` recompute in `log_softmax_last_axis_backward`) require CustomOp in candle, not free composition. ARLE keeps perf headroom by paying authorship cost.

**Post-Wave 2 architectural revisit (NOT scope)**: collapse `Dirty::*` tristate to single-device storage. `Dirty::Both` has produced 4+ regressions in the experience log. The "from-scratch autograd" cognitive goal does not require a tristate; candle proves single-device suffices. Open a `docs/projects/` ticket after Wave 2 lands.

## 5. Recommended Wave 2 plan (fastest path to >300 tok/s)

Three commits, ordered by bytes-moved-per-step:

1. **A — `embedding` + `add_broadcast`**: the two biggest remaining host-poison sources in P3.1's 287-memcpy residue. Both M-effort. **`embedding_backward` MUST use `atomicAdd`** — do not borrow candle's race-prone scatter. Expected gain ~+50 tok/s.
2. **B — `rms_norm` + `rope`**: next-largest per-step contributors. RMSNorm M (two reductions), RoPE S (sin-flip mirror). Land separately for bisectable numerical regressions. Expected gain ~+40 tok/s.
3. **C — trivial Rust-side dispatch** for `mul`, `neg`, `exp`, `sum`, `silu`, `scatter_add_rows`. Mostly XS, reuses existing `elementwise.cu` / `mean_backward.cu`. Expected gain ~+30-50 tok/s.

**Target**: ≥300 tok/s after B; ≥350 tok/s after C.

**Verify per commit** (CLAUDE.md §Benchmarks): `scripts/bench_guidellm.sh wave2-<a|b|c>` with Δ vs. P3.1 baseline, plus `nsys` step-level memcpy count monotonically dropping. **Stop the wave at the first commit failing ≥+15 tok/s** — bottleneck has moved (likely fwd-kernel fusion or matmul dispatch), warrants fresh nsys attribution before pushing.

**Anti-recommendation**: do **not** create `vendor/candle/`. Per-file header citation is sufficient; a vendored tree creates update-cadence overhead with no benefit at our scale.

---

**Sources**: [candle repo](https://github.com/huggingface/candle) · [candle-kernels/src listing](https://api.github.com/repos/huggingface/candle/contents/candle-kernels/src) · [reduce.cu](https://raw.githubusercontent.com/huggingface/candle/main/candle-kernels/src/reduce.cu) · [unary.cu](https://raw.githubusercontent.com/huggingface/candle/main/candle-kernels/src/unary.cu) · [binary.cu](https://raw.githubusercontent.com/huggingface/candle/main/candle-kernels/src/binary.cu) · [indexing.cu](https://raw.githubusercontent.com/huggingface/candle/main/candle-kernels/src/indexing.cu) · [candle-nn/rotary_emb.rs](https://github.com/huggingface/candle/blob/main/candle-nn/src/rotary_emb.rs) · [candle-core/backprop.rs](https://github.com/huggingface/candle/blob/main/candle-core/src/backprop.rs) · [candle-core/tensor.rs](https://github.com/huggingface/candle/blob/main/candle-core/src/tensor.rs)
