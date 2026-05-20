# 2026-05-20 — `matmul_bt` op + backward derivation (Axis A of `lm_head` perf)

> **Status:** plan / derivation. Companion to
> [`docs/research/2026-05-20-lm-head-transpose-cache.md`](../research/2026-05-20-lm-head-transpose-cache.md)
> §"Axis A — Transpose-aware forward matmul". Implementation handed to
> codex per the 2026-05-20 work split (Claude = research/plan/docs/deterministic
> code; codex = complex code + verification).

## Why a new op rather than caching the transposed weight

The research doc compared Axis A (`matmul_bt`) vs Axis B (cache the
transposed `lm_head` weight). For training Axis A is structurally cleaner:
the weight mutates after every optimizer step, so the cache would need an
invalidation hook on every `AdamW::step`. Axis A eliminates the transpose
copy entirely — `linear_forward` becomes:

```rust
// Before
let weight_t = transpose(weight, 0, 1, store, tape)?;     // physical 623 MB copy
let projected = matmul(flat_x, weight_t, store, tape)?;

// After
let projected = matmul_bt(flat_x, weight, store, tape)?;  // no copy
```

The op is the natural Rust expression of *"matmul with the second operand
transposed"*, dispatching to the already-implemented
`matmul_a_bt_into` (`crates/autograd/src/backend.rs:1697-1734`) on CPU.

## Forward derivation

For `C = A @ B^T` with shapes:

- `A`: `[m, k]` (row-major)
- `B`: `[n, k]` (row-major — note: K-axis on the inside, this is the
  *natural* lm_head weight layout `[vocab=n, hidden=k]`)
- `C`: `[m, n]`

The math: `C[i, j] = sum_k A[i, k] * B[j, k]`.

CPU dispatch: `matmul_a_bt_into(a, (m, k), b, (n, k), &mut c)` — already
exists. It passes B's stride as `rsb=1, csb=k` to `matrixmultiply::sgemm`,
treating the physical `[n, k]` buffer as the logical `[k, n]` transpose
without copying.

## Backward derivation

Given `C = A @ B^T`, treat the elementwise expression
`C[i, j] = Σ_k A[i, k] · B[j, k]` and apply chain rule for each input.

### `grad_A = ∂L/∂A` — plain matmul

```
grad_A[i, k] = Σ_j (∂L/∂C[i, j]) · (∂C[i, j]/∂A[i, k])
             = Σ_j grad_C[i, j] · B[j, k]
             = (grad_C @ B)[i, k]
```

So **`grad_A = grad_C @ B`** with shapes `[m, n] @ [n, k] = [m, k]`.
This is a *plain* matmul over B's natural row-major layout — **no
transpose-aware kernel needed**.

| Backend | Dispatch |
|---|---|
| CPU host path | `cpu_matmul_forward(grad_out, &[m, n], b, &[n, k])` (existing) |
| CPU device path | `store.backend().matmul(&grad_c, &[m, n], &b, &[n, k])` (existing) |
| Metal / CUDA | inherit via existing `Backend::matmul` |

### `grad_B = ∂L/∂B` — `matmul_at_b_into` reuse

```
grad_B[j, k] = Σ_i (∂L/∂C[i, j]) · (∂C[i, j]/∂B[j, k])
             = Σ_i grad_C[i, j] · A[i, k]
             = Σ_i (grad_C^T)[j, i] · A[i, k]
             = (grad_C^T @ A)[j, k]
```

So **`grad_B = grad_C^T @ A`** with shapes `[n, m] @ [m, k] = [n, k]`.
This is the existing `matmul_at_b_into` kernel pattern, with no need for
a final transpose:

```rust
matmul_at_b_into(
    grad_c, /*a_shape=*/ (m, n),
    a,      /*b_shape=*/ (m, k),
    &mut grad_b,         // shape (n, k) — matches B's natural layout
)
```

This is exactly the existing private helper at
`crates/autograd/src/backend.rs:1747`. It already produces output in
row-major `[K_logical, N]` order where K_logical = n_logical_columns of
the second operand = `m`, and N = `k`. Wait, let me re-check the kernel
contract.

Reading the kernel docstring (`backend.rs:1736-1745`):

> Compute `out = a^T @ b` for row-major rank-2 buffers without
> materialising `a^T`. ... Shapes (caller-enforced):
> - `a`: `[M, K]` (row-major contiguous) — *logical pre-transpose*
> - `b`: `[M, N]` (row-major contiguous)
> - `out`: `[K, N]` (row-major contiguous, pre-zeroed)

So calling `matmul_at_b_into(a=grad_c, a_shape=(m, n), b=a, b_shape=(m, k), out=grad_b)`:
- Treats `a` as `[m, n]` (matches `grad_c`)
- Treats `b` as `[m, k]` (matches `a`)
- Produces `out` as `[n, k]` = `grad_c^T @ a` = `[n, m] @ [m, k]` ✓

`grad_B` shape `[n, k]` matches `B`'s shape `[n, k]`. **No final transpose
needed.**

### Symmetry summary

| Backward grad | Formula | Existing CPU kernel | Shape |
|---|---|---|---|
| `grad_A` | `grad_C @ B` | `cpu_matmul_forward(grad_c, b)` | `[m, k]` ✓ |
| `grad_B` | `grad_C^T @ A` | `matmul_at_b_into(grad_c, a, &mut grad_b)` | `[n, k]` ✓ |

Both grads land in their natural row-major layouts — no scratch transpose
copies anywhere in the backward pass. The forward also avoids the
transpose (via `matmul_a_bt_into`). End-to-end, this Axis A
implementation **eliminates all physical transpose copies** on the
`lm_head` axis.

## Rank-3 (batched) case

`linear_forward` flattens rank-3 input down to rank-2 before the matmul
(see `crates/train/src/qwen35.rs:1016-1022`), so the rank-3 path may not
be strictly needed for `lm_head`. But for completeness — the existing
`cpu_matmul_backward` rank-3 case loops per-batch over the rank-2 kernels
(`backend.rs:1628-1666`). The same pattern applies to `matmul_bt`:

```rust
for bi in 0..batch {
    matmul_a_bt_into(
        &a[bi * (m*k)..],
        (m, k),
        &b[bi * (n*k)..],
        (n, k),
        &mut c[bi * (m*n)..],
    );
}
```

Whether `B` is per-batch or shared across the batch is the only design
decision — for `lm_head` it would be shared, but for KV-cache-style
batched matmuls it might be per-batch. Both can be handled with a shape
parameter check. Codex decides the surface.

## Implementation plan (codex)

1. **Add `matmul_bt` op** in `crates/autograd/src/ops/matmul.rs`:
   - Forward: dispatch CPU host → `matmul_a_bt_into`; device → new
     `Backend::matmul_bt` trait method (default impl: transpose-then-matmul
     for backends that don't override).
   - Record tape entry `BackwardOp::MatmulBT` with
     `SavedContext::MatmulBTCtx { a, b }`.
2. **Add `matmul_bt_backward`** symmetric to `matmul_backward`. Reuse
   `cpu_matmul_forward` for `grad_A` and `matmul_at_b_into` for `grad_B`.
   Upgrade `matmul_at_b_into` to `pub(crate)` (it's already
   `pub(crate)`-grade — currently `fn` private — so the upgrade is a
   visibility-only change).
3. **Add `Backend::matmul_bt` + `Backend::matmul_bt_backward`** trait
   methods, with default implementations that fall back to
   transpose-then-existing-op. Override on CPU backend
   (`crates/autograd/src/backend.rs`); Metal and CUDA inherit the default
   until a separate optimization pass.
4. **Switch `linear_forward`** in `crates/train/src/qwen35.rs:1018-1019`
   from `transpose + matmul` to `matmul_bt`.
5. **Numerical gates**:
   - `test_opd_determinism` — bit-identical (load-bearing — if backward
     symmetry is wrong, this fails first)
   - `test_opd_grad_check` — finite-difference agreement
   - New: add a dedicated `matmul_bt_grad_check` unit test in
     `crates/autograd/tests/` paralleling the existing matmul grad test
6. **Bench**: re-run `rollout_last_logits_ab_bench` (or a new
   `linear_forward_transpose_ab_bench` for cleaner isolation) at
   Qwen3-0.6B shape. Expected wall-clock saving ~22 % of step.

## Risk + safety net

- The backward derivation is fully symmetric with the existing
  `cpu_matmul_backward` (the formulas mirror but swap which operand is
  treated as transposed). If determinism fails, it's almost certainly a
  shape-orientation bug in the `matmul_at_b_into` call — debuggable by
  checking which axis is contracted.
- `test_opd_grad_check` finite-difference (currently at 0.2928 % max
  relative error per codex's 2026-05-20 EOD run) is the numerical safety
  net. Threshold should remain unchanged after the switch.
- No new SIMD kernels — both backward kernels already exist and are
  benched in
  [`docs/experience/wins/2026-05-19-bench-cpu-matmul-backward-transpose-aware-qwen3-06b.md`](../experience/wins/2026-05-19-bench-cpu-matmul-backward-transpose-aware-qwen3-06b.md).

## Cross-links

- Research / decision tree:
  [`docs/research/2026-05-20-lm-head-transpose-cache.md`](../research/2026-05-20-lm-head-transpose-cache.md)
- Existing transpose-aware kernel:
  `crates/autograd/src/backend.rs:1697-1810`
- Existing `linear_forward`:
  `crates/train/src/qwen35.rs:980-1023`
- Existing `cpu_matmul_backward` (template for symmetric design):
  `crates/autograd/src/backend.rs:1588-1672`
