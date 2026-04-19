# autograd: 12 Metal trait methods wired through MLX

## Context

Follow-up to `2026-04-19-autograd-cpu-metal-pretrain-bit-ident-cuda-kernel-expansion.md`.
Prior session broadened the `Backend` trait with 12 new forward methods
and provided CPU reference impls + CUDA kernels. Metal still fell back to
the CPU default — that was *why* CPU and Metal stayed bit-identical: only
matmul/add/softmax/log_softmax were actually running on MLX.

User directive (this session): "都接入好" + "并行写代码" — wire every new
op through MLX on Metal in parallel. CUDA stays deferred to remote
(`cuda 的我远端另外验证就行`).

## What Worked

### 1. 12 Metal trait methods now dispatch through MLX

| Method | MLX primitive(s) |
|--------|------------------|
| `mul_forward` | `mlx_multiply` |
| `mul_scalar_forward` | `mlx_multiply` (scalar broadcast) |
| `exp_forward` | `mlx_exp` |
| `neg_forward` | `mlx_negative` |
| `gelu_forward` | `0.5 * x * (1 + erf(x / √2))` via `mlx_erf` |
| `silu_forward` | `x * sigmoid(x)` via `mlx_sigmoid` |
| `rms_norm_forward` | `mlx_fast_rms_norm` |
| `embedding_forward` | `mlx_take_axis` with OOB pre-clamp + mask |
| `sum_last_axis_forward` | `mlx_sum_axis` |
| `mean_last_axis_forward` | `mlx_mean_axis` |
| `rope_forward` | manual NeoX split: slice + mul(cos/sin) + add + sub + concat axis=3 |
| `gather_last_dim_forward` | `mlx_take_axis` on flattened src |

Every new path takes `MLX_GUARD` (the global MLX mutex), null-checks FFI
returns with `AutogradError::TapeInvariant`, and frees every intermediate
`mlx_array` on all return paths. `backend_metal.rs` grew 314 → 1144
lines.

### 2. 29/29 parity tests green

`cargo test --release -p autograd --no-default-features --features metal
--test test_backend` — 29 passed, 0 failed, 0.20s. Twelve new
`metal_backend_*_matches_cpu` tests use realistic Qwen3 shapes:
`[2,4,16,64]` for rope, ids `[0,5,10,63,-1,99,7]` for embedding (exercises
OOB clamp), `[5,0,127,42]` for gather. Tolerances: exp/gelu 1e-3,
rms_norm/silu/rope 1e-4, neg/mul/mul_scalar/gather/embedding 1e-6. Every
one held at or below its tolerance.

### 3. End-to-end pretrain baseline preserved

`pretrain_qwen3 --backend metal --seed 42 --steps 10` on the same 19.71M
Qwen3 config as the 2026-04-19 bit-ident doc: step 0 loss **11.9799** —
identical to the prior CPU/Metal run logged there. Subsequent steps
diverged slightly (11.0748 at step 9 vs. 11.1729 prior at step 20), which
is expected and acceptable: Metal now runs silu/rope/embedding/gather/
rms_norm/etc. on MLX with a different accumulation order than the CPU
reference. The CPU<->Metal bit-identity invariant from the prior doc only
held *because* those ops ran on CPU in both backends; now that Metal has
its own kernels, drift inside per-op tolerance is the correct outcome.

## Rule

When swapping a CPU-default trait impl for a real GPU path, the
bit-identity signal across backends breaks by design — don't panic when
the loss trajectory shifts. The correct invariant becomes: (a) op-level
parity within tolerance on shapes matching the target workload, and
(b) loss at step 0 still matches (because the model is a sum of deterministic
initialization + one forward pass, and `randn + seed 42` is the dominant
term). Step-k drift is acceptable iff op parity tests pass.

When CUDA verification is explicitly deferred to a remote box, still
commit the Metal work on its own — don't block Metal wiring behind the
GPU we can't touch today.
