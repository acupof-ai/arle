# matmul backward on GPU backends (Metal + CUDA-stub)

## Context

Forward `matmul` already dispatched through `Backend::matmul_forward`
(Metal = MLX `mx::matmul`, CUDA = cuBLAS SGEMM). Backward remained
pure-CPU: `ops/matmul.rs::matmul_backward` did a host transpose + two
`matmul_forward` calls, which under autograd's current topology meant
every training step round-tripped the gradient GEMM to CPU memory, did
the transpose + matmul on CPU, and returned the result. Backward is
~2× forward FLOPs, so this was the dominant Metal hot-path cost once
forward moved to GPU (confirmed iter/s drop trace from the 2026-04-19
RL validation run at 2.80 iter/s TinyLM group=8).

Goal: keep matmul gradients on device for both backends, mirroring the
forward additive-method pattern. CPU path preserved as default so
`CpuBackend` still works.

## What Worked

### Trait addition (`crates/autograd/src/backend.rs`)

Added `matmul_backward` as a sibling of `matmul_forward` on the
`Backend` trait, with a CPU default implementation delegating to
`cpu_matmul_backward`. Non-breaking: existing backends keep working
without override.

```rust
fn matmul_backward(
    &self,
    a: &[f32], a_shape: &[usize],
    b: &[f32], b_shape: &[usize],
    grad_out: &[f32], grad_out_shape: &[usize],
    need_grad_a: bool, need_grad_b: bool,
) -> Result<(Vec<f32>, Vec<f32>)> {
    cpu_matmul_backward(a, a_shape, b, b_shape, grad_out,
                        grad_out_shape, need_grad_a, need_grad_b)
}
```

`cpu_matmul_backward` does host transpose + two `cpu_matmul_forward`
calls (the old inline path, relocated). Returns an empty `Vec` on the
skipped side when `need_grad_*` is false — avoids wasted work when a
model has frozen weights on one side of a matmul.

### Metal override (`crates/autograd/src/backend_metal.rs`)

Imports `mlx_transpose_axes` from `mlx-sys`. `MetalBackend::matmul_backward`
delegates to a new `mlx_matmul_backward` helper that:

1. Holds `MLX_GUARD: Mutex<()>` for the full FFI round-trip (MLX
   globals are not thread-safe).
2. Uploads A, B, grad_out via `mlx_array_from_data` once.
3. For each needed gradient, `mlx_transpose_axes` the appropriate
   operand (`[1, 0]` for rank-2, `[0, 2, 1]` for rank-3), then
   `mlx_matmul` + `mlx_eval` + host readback.
4. Frees all intermediates on every error path.

Return shape matches the CPU reference: empty `Vec` for skipped sides.
No new `.metal` kernels — MLX already has compiled GEMM + transpose.

### CUDA override (`crates/autograd/src/backend_cuda.rs`)

`#[cfg(feature = "no-cuda")]` returns `todo!("GPU required: cuda
matmul_backward is unavailable under feature no-cuda")` so Mac
typecheck (`cargo check -p infer --no-default-features
--features cuda,no-cuda`) still works. Live path: `cuda_matmul_backward`
uses `cudarc::cublas::safe::{CudaBlas, Gemm, GemmConfig,
StridedBatchedConfig}` with the row-major → col-major OP_T trick:

- **grad_a = dC @ B^T**: `transa=OP_T, transb=OP_N`, args `(d_b, d_g)`;
  `m=k, n=m, k=n`; `lda=n, ldb=n, ldc=k`.
- **grad_b = A^T @ dC**: `transa=OP_N, transb=OP_T`, args `(d_g, d_a)`;
  `m=n, n=k, k=m`; `lda=n, ldb=k, ldc=n`.

Rank-3 uses `StridedBatchedConfig` with strides derived from the above
leading dims. No physical device-side transpose — cuBLAS reads
transposed.

### Dispatch (`crates/autograd/src/ops/matmul.rs`)

Replaced the inline host transpose + 2× forward with a single backend
dispatch. `need_grad_a` / `need_grad_b` are honored so backends can
skip work on frozen sides.

## Tests

`crates/autograd/tests/test_backend.rs` — added:

| test | shapes | tol |
|---|---|---|
| `cpu_backend_matmul_backward_matches_reference` | `[8,16]@[16,32]`, `[4,64]@[64,64]`, `[3,8,16]@[3,16,32]` | 1e-5 |
| `cpu_matmul_backward_skips_unneeded_sides` | same, with `need_grad_*=false` | exact |
| `metal_backend_matmul_backward_matches_cpu` | same shapes | 1e-3 |
| `metal_backend_matmul_backward_skip_sides` | same | 1e-3 |
| `cuda_backend_matmul_backward_matches_cpu` | same (gated on `feature="cuda",not="no-cuda"`) | pending-remote |

`cargo test --release -p autograd --features metal --test test_backend`
→ **42/42 passed**.

## End-to-end validation — train_multi_turn on Metal

Matched-params re-run of the 2026-04-19 RL baseline (TinyLM group=8,
30 iters, seq_len=13, `d_model=64 n_layers=2 n_heads=4 d_head=16 d_ff=128`,
vocab=32, lr=5e-3, kl_coef=0.01):

```
target/release/train_multi_turn \
  --iters 30 --group-size 8 --turns 2 \
  --prompt-len 4 --obs-tokens 3 --agent-tokens 3 \
  --vocab 32 --d-model 64 --n-layers 2 --n-heads 4 --d-head 16 --d-ff 128 \
  --lr 5e-3 --kl-coef 0.01 --temperature 1.0 \
  --eval-every 5 --eval-prompts 4 --target-range 12 \
  --backend metal
```

| metric | 2026-04-19 baseline (CPU backward) | 2026-04-20 (Metal backward) | Δ% |
|---|---|---|---|
| iter/s | 2.80 | 2.94 | **+5.0%** |
| token/s | 291 | 305.8 | **+5.1%** |
| wall clock | 10.72 s | 10.20 s | **-4.8%** |
| episode/s | 22.4 | 23.52 | **+5.0%** |

Three-signal RL validation (per 2026-04-19 rule) intact:

- **best_reward** 0.042 → 0.167 (4× climb, identical arc to baseline)
- **loss** 0.000 → -0.0130 at iter 27 (bit-identical sign, same magnitude)
- **KL** +0.05 → -1.05 (reference policy genuinely separate)

Smaller sweep — `--iters 10 --group-size 4 --backend metal` — hit
**7.14 iter/s / 285.7 token/s** with reward climbing 0.0625 → 0.1875,
confirming faster turnaround at smaller group sizes where CPU backward
overhead used to dominate setup cost relative to FLOPs.

## Why the speedup is modest (not 2×)

Backward is ~2× forward FLOPs, but at TinyLM scale (`d_model=64`, group=8)
the per-matmul work is tiny — FFI launch overhead + MLX eval barrier
dominate over arithmetic. MLX-side Metal dispatch is O(µs) per op, and
we issue one transpose + one matmul per gradient side. At Qwen-scale
(`d_model=4096`, real batches), backward matmul is several milliseconds
of GPU arithmetic and the FFI overhead amortizes; expected speedup
there is much larger (baseline for that run is `pending-remote`).

## Not yet validated (remote-pending)

- **CUDA path**: coded + typechecks under `no-cuda` stub; `cuda_backend_matmul_backward_matches_cpu`
  test gated and not exercised locally. Needs CUDA box run to cross
  Metal↔CUDA parity.
- **Qwen3 RL closed loop** with Metal backward on device — the interesting
  throughput number. Needs the existing remote Qwen plan (roadmap 6.3).
- **Full bit-ident** Metal↔CPU on a multi-step RL iter (not just primitive-level
  1e-3 tol).

## Files changed

- `crates/autograd/src/backend.rs` — added `matmul_backward` trait
  method + `cpu_matmul_backward` + `pub(crate) transpose_last_two_ref`.
- `crates/autograd/src/backend_metal.rs` — added `matmul_backward`
  override + `mlx_matmul_backward` helper.
- `crates/autograd/src/backend_cuda.rs` — added `matmul_backward`
  (no-cuda stub + live cuBLAS path) + `cuda_matmul_backward` helper.
- `crates/autograd/src/ops/matmul.rs` — dispatch through
  `store.backend().matmul_backward(...)`; removed inline host transpose.
- `crates/autograd/tests/test_backend.rs` — parity tests (CPU, Metal,
  CUDA-gated).

## Rule

Additive-method pattern scales: when forward moves to a new backend,
**always** schedule the backward trait method as a sibling with a CPU
default, even if you don't implement it same-session. The default keeps
the trait non-breaking while the override lands one backend at a time.
Pure-CPU backward paths look cheap at tiny scale but become the
training bottleneck the moment forward goes GPU — treat them as
first-class backend work, not "we'll get to it."
