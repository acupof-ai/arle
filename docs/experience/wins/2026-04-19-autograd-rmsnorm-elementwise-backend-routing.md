# autograd: rmsnorm forward + elementwise backward routed through Backend

## Context

Completing the backend-trait dispatch coverage started in
`2026-04-19-autograd-backward-trait-dispatch.md` and `-metal-12-ops-wired.md`.
Systematic audit (user directive: "完整系统的看一下还差什么然后逐个补齐")
flagged two gaps:

1. `ops/norm.rs::rmsnorm` forward still ran inline CPU even under
   `--backend metal` — the biggest remaining hot op without trait routing.
2. `mul_backward`, `exp_backward`, `mul_scalar_backward` still did their
   elementwise muls as inline Rust `zip/map` instead of through
   `backend().mul_forward` / `mul_scalar_forward`.

Earlier in the same session, parallel general-purpose agents had also:
- Added Metal `scatter_add_rows_forward` via a new `mlx_scatter_add_rows_f32`
  FFI binding + C++ bridge helper (using `mlx::core::scatter_add`) — Metal
  had been falling back to CPU because mlx-sys had no binding.
- Added CUDA + Metal + CPU coverage for a new
  `Backend::add_broadcast_forward` method (new NVRTC-compiled
  `add_broadcast.cu` kernel; Metal uses native MLX broadcast on `mlx_add`).

All four pieces land in one commit because they share the same verify matrix.

## What Worked

### 1. rmsnorm forward routed, backward untouched

`ops/norm.rs:29-44` replaced with
`store.backend().rms_norm_forward(&x.data, &w.data, &x.shape, eps)?`.
inv_rms (needed by `rmsnorm_backward` via `SavedContext::RMSNormCtx`) is
now computed in a **separate CPU pass** only when `requires_grad` is true.
This redundant read over `x.data` is memory-bound and trivial compared to
the FLOP-dense normalization itself, so no trait-signature change was
required. The backward path is unchanged.

### 2. Trivial backward muls trait-dispatched

- `mul_backward` → `grad_a = backend().mul_forward(&upstream, &b.data)`
  (same for `grad_b`). Single output allocation per call — no
  intermediate-tensor memory regression (the softmax/log_softmax P1 from
  the prior session does not apply here because these are already a
  single mul, not a 3-op decomposition).
- `exp_backward` → `backend().mul_forward(&output.data, &upstream.data)`.
- `mul_scalar_backward` → `backend().mul_scalar_forward(&upstream, k)`.

### 3. Bit-identical CPU loss preserved

`pretrain_qwen3 --backend cpu --seed 42 --steps 3 --batch 1 --seq 128
--hidden 64 --layers 2 --heads 4 --kv-heads 2 --head-dim 16 --intermediate 128`:

| step | baseline (pre-refactor) | post-refactor |
|------|-------------------------|---------------|
| 0    | 11.8966                 | 11.8966       |
| 1    | 11.8840                 | 11.8840       |
| 2    | 11.8226                 | 11.8226       |
| eval | 11.7904                 | 11.7904       |

Baseline run stashed the refactor; post-refactor run on the same corpus
hash matched exactly. The rmsnorm path `cpu_rms_norm_forward` uses
`(mean_sq + eps).sqrt().recip()` — bit-identical to the old inline
`1.0 / (mean_sq + eps).sqrt()` on f32.

### 4. Green across the matrix

- `cargo test --release -p autograd --no-default-features` — 14 CPU
  backend tests + all backward grad-check tests pass.
- `cargo test --release -p autograd --features metal --test test_backend` —
  **38/38** (includes new
  `metal_backend_scatter_add_rows_gather_shape`,
  `metal_backend_scatter_add_rows_all_oob`,
  `metal_backend_add_broadcast_matches_cpu`,
  plus the pre-existing parity suite).
- `cargo test --release -p train --no-default-features` — all training
  suites green.
- `cargo check -p autograd --no-default-features --features cuda,no-cuda` —
  Mac CUDA typecheck clean (the new `add_broadcast.cu` kernel registration
  + `scatter_add_rows_f32` kernel both compile-gated).
- `cargo clippy -p autograd --no-default-features --tests -- -D warnings`
  — clean.

**PENDING REMOTE CUDA VERIFICATION** — the new `add_broadcast_f32`
NVRTC kernel needs a real GPU run alongside the existing
`scatter_add_rows_f32` verification from the prior session:
`cargo test -p autograd --release --features cuda --test test_backend`.

## Rule

When routing a forward op through the backend trait but the backward
needs a CPU-only side-channel (here: `inv_rms` per row for RMSNorm),
**keep the side-channel computation on CPU as a separate pass gated on
`requires_grad`**, rather than extending the trait signature to return
auxiliary state. The redundant pass over `x` is memory-bound, cheap, and
only runs when autograd is on. Widening the trait to return `(output,
Option<aux>)` would force every backend impl to track it forever for the
sake of one saved-context consumer — bad leverage.

The GELU forward remains intentionally unrouted: `ops/activation.rs`
uses `erff`-based GELU while `Backend::gelu_forward` uses the tanh
approximation, and unifying them would break bit-identical CPU loss.
This is the one known semantic divergence in the ops layer vs the
backend trait, documented here so the next audit doesn't re-flag it.
