# autograd: backward ops dispatch through trait + scatter_add kernel

## Context

Follow-up to `2026-04-19-autograd-metal-12-ops-wired.md`. Forward ops now
all route through `Backend::*_forward`, but backward paths still held
inline CPU loops — so training's second half (gradient accumulation) was
still 100% CPU even with `--backend metal`.

User directive: "cuda 代码并行先写好, 全部写好代码 然后 让 codex review" —
write all CUDA-accelerating backward code in parallel, then codex-review.

## What Worked

### 1. Three parallel general-purpose agents completed in ~7 min wall

- **Agent 1**: `matmul_backward` → `backend.matmul_forward(upstream, b^T)`
  + `backend.matmul_forward(a^T, upstream)`. Added file-local
  `transpose_last_two` helper (O(N) memory, CPU — not FLOP-bound).
  Unified 2D + 3D-batched into one code path.
- **Agent 2**: New trait method `scatter_add_rows_forward(upstream,
  prefix_rows, feature_dim, indices, vocab) → [vocab * feature_dim]`.
  CPU default, CUDA kernel with `atomicAdd` (embedding backward aliases
  multiple rows onto the same bin). `embedding_backward` and
  `gather_last_dim_backward` both collapse to one call. Gather uses
  index remap `i * vocab + original[i]` so the shared method handles
  both shapes. Metal falls back to CPU default — `mlx-sys` has no
  `scatter_add` binding yet.
- **Agent 3**: `rope_backward` → `backend.neg_forward(sin)` +
  `backend.rope_forward(upstream, shape, cos, -sin)` (backward of NeoX
  rope = forward with sin negated). Softmax/log_softmax backward also
  attempted — reverted per review, see §2.

One type-inference bug at `ops/gather.rs:122` (ambiguous `product()`)
surfaced from combining Agent 2+3 diffs; fixed with explicit turbofish.

### 2. Codex review caught a [P1] memory regression

Agent 3's decomposition of `softmax_backward` and `log_softmax_backward`
used `backend.mul_forward` + `backend.sum_last_axis_forward` + a manual
subtract — semantically correct but it materializes three full output-
shaped intermediate buffers (`tmp`, `diff`, `grad`). On Qwen training
paths with full-vocab 151669 logits and batch×seq in the thousands,
that adds ~3-4× peak memory to the backward step — enough to OOM a
batch that fits today.

Reverted both backward functions to single-pass streaming row-loops
that only allocate the output `grad` buffer. The CPU codegen for these
is tight (one accumulation + one multiply-accumulate per element), and
since they're pure f32 arithmetic with no FLOPs-per-byte to amortize,
putting them on GPU via `*_forward` round-trips would cost more than
the loop it replaces. Matmul/rope backward stay trait-dispatched —
those are genuinely FLOP-dense and amortize the FFI cost.

### 3. Bit-identical CPU baseline preserved

`pretrain_qwen3 --backend cpu --seed 42` step-0 loss = **11.9799**,
step-1 = **11.9369**, step-2 = **11.5274** — identical to pre-refactor.
The CPU path now routes through `backend().matmul_forward` instead of
the free-function `cpu_matmul_forward`, but both are the same code, so
preservation is expected and confirmed.

### 4. Green across the matrix

- `cargo test --release -p autograd --no-default-features` — 13 CPU
  backend tests + 10 m1_ops backward grad-check tests all pass.
- `cargo test --release -p autograd --no-default-features --features
  metal --test test_backend` — 34/34 (parity + scatter_add + the
  refactored Metal ops).
- `cargo check -p autograd --no-default-features --features cuda,no-cuda
  --tests` — Mac CUDA typecheck clean.
- `cargo clippy -p autograd --no-default-features --tests -- -D warnings`
  — clean.

**PENDING REMOTE CUDA VERIFICATION** — new `scatter_add_rows_f32` kernel
needs a real GPU run. User verifies with `cargo test -p autograd
--release --features cuda --test test_backend`.

## Rule

When composing a backward op from trait-level forward primitives,
check: *does the composition introduce intermediate tensors at the same
shape as the largest input?* If yes, and that input can be vocab-sized
or batch×seq-sized, **don't decompose** — keep the backward as a
single-pass Rust loop that only allocates the output. The GPU-via-FFI
overhead is real (upload + readback per call), and f32 elementwise ops
are memory-bound, not FLOP-bound — so the `Backend::*_forward` dispatch
pattern only pays for itself on FLOP-dense ops (matmul, rope, RMS-norm).

When delegating code refactors to 3 parallel agents, bundle a
cross-agent integration check into the next step. Agent 3's revert here
was caught by codex review, not by any single agent's self-test — the
whole-diff view is what surfaced the peak-memory issue.
