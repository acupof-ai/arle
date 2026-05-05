# `autograd` — Agent Guide

Training-tape autograd engine. Host-authoritative `Vec<f32>` by default;
backends can lift tensors to device-resident handles and compose graphs
across ops. Two backends: CPU (reference, always on) and Metal (via
`crates/mlx-sys`). A `cuda` feature compiles on Mac under `no-cuda` but
needs a GPU box to execute. Load this file before editing anything under
`crates/autograd/src/` or adding a `Backend` trait method.

## Refactor posture

- Keep autograd code simple and uniform. Prefer deletion-style refactors:
  remove obsolete parallel APIs, collapse duplicate backend/plumbing logic,
  and keep one canonical tensor/tape contract instead of adapter stacks.

## Module layout

```
crates/autograd/
├── Cargo.toml          — features: default=[] / metal / cuda / no-cuda / safetensors
├── src/
│   ├── lib.rs          — module decls + AutogradError + Result + public re-exports
│   ├── tensor.rs       — Tensor, TensorId, TensorStore (dirty + device_handle fields)
│   ├── tape.rs         — Tape, TapeEntry, BackwardOp, SavedContext
│   ├── backend.rs      — Backend trait, CpuBackend, DeviceHandle, CPU reference impls
│   ├── backend_metal.rs — MetalBackend: MLX FFI + eval counter
│   ├── backend_cuda.rs + backend_cuda/ — CudaBackend: cuBLAS + NVRTC (no-cuda stub on Mac)
│   ├── ops.rs + ops/   — high-level op entry points, one file per op family
│   │                     (activation, attention, broadcast, elementwise, embed, gather,
│   │                      layout, linear_attention, matmul, norm, reduce, rope, softmax)
│   ├── optim.rs        — SGD, AdamW
│   ├── adamw_state.rs  — opaque serializable AdamW moment codec for checkpointing
│   ├── lr_schedule.rs  — `LrSchedule` trait + `ConstantLr`, `LinearWarmup`,
│   │                     `CosineWithWarmup`, `parse_lr_schedule`. Pure step→f32
│   │                     functions (no persisted state).
│   ├── module.rs       — parameter iteration for optimizers
│   └── safetensors_io.rs (feature = "safetensors")
├── tests/
│   ├── test_backend.rs         — backend parity (CPU reference vs Metal/CUDA, 1e-3 tol)
│   ├── test_device_handle.rs   — upload/eval/readback + M5.3a eval-count acceptance
│   ├── m0_ops.rs / m1_*.rs     — tape/op numerical grad_check suites
│   └── helpers.rs              — num_grad, seeded RNG
└── AGENTS.md           — this file
```

The `ops/` directory is intentionally granular. Adding a new op means a
new file in this list — do not stack ops into existing modules to keep
the file count down. Each op file pairs with a `cpu_*_forward` /
`cpu_*_backward` reference under `backend.rs`.

## Invariants (violating these breaks training)

1. **CPU backend is the reference.** Every new op lands with a CPU
   implementation first. Metal/CUDA overrides must match CPU to `≤ 1e-3`
   relative tolerance on the shapes in `tests/test_backend.rs`. The
   `cpu_*_forward` / `cpu_*_backward` free functions in `backend.rs` are
   the authoritative numerical contract; backends may call them as a
   fallback, but a test that fails against the CPU reference is a bug in
   the backend, never in the reference.
2. **Additive-method pattern.** A new op on `Backend` lands with a CPU
   default implementation that delegates to the matching `cpu_*_forward`
   function (or `cpu_*_backward`), so adding an op is non-breaking across
   backends. Metal/CUDA overrides are separate commits — see the "M2b:
   Claude writes, Codex reviews" cadence in memory.
3. **One eval boundary per step (goal), bounded evals always (invariant).**
   See §DeviceHandle contract below. Regressing past the bounded
   `eval_count` check in `metal_single_forward_backward_step_has_bounded_eval_count`
   means we've re-introduced the 1-op-per-eval pattern that made Metal
   1.9× slower than CPU pre-M5.3a. Don't ship until the test passes.
4. **Backend isolation.** `#[cfg(feature = "metal")]` on MLX imports,
   `#[cfg(feature = "cuda")]` on cudarc imports, `#[cfg(feature = "no-cuda")]`
   stub for CUDA execution paths — always `todo!("GPU required: ...")` so
   a CPU-only binary fails loudly rather than silently.
5. **Shared MLX synchronization boundary.** The Metal backend calls
   `mlx_sys::mlx_guard()` for MLX FFI serialization. Do not add a local
   autograd-only mutex around MLX; MLX state is process-global and must share
   the same Rust guard as other `mlx-sys` consumers.
6. **No half-states on device-resident ports.** When you add a lazy
   device-handle path for an op, finish it: forward + backward both go
   through the handle, or neither does. Do not leave the op in a state
   where forward is lazy but backward does its own upload+eval+readback
   unless you document that hybrid state in the wins entry. See
   `feedback_no_half_states.md`.

## DeviceHandle contract (M5.3a)

**Spec**: see [`docs/projects/agent-rl-self-evolving.md`](../../docs/projects/agent-rl-self-evolving.md) §M5 for the device-resident tensor milestone scope.

### Types

```rust
pub enum DeviceHandle {
    Cpu(Vec<f32>),
    #[cfg(feature = "metal")] Metal(MlxHandle),
    #[cfg(feature = "cuda")] Cuda(CudaStorage),
}

pub enum Dirty { Host, Device, Both }
```

`Tensor` carries `data: Vec<f32>`, `device_handle: Option<DeviceHandle>`,
`dirty: Dirty`. Exactly one side is authoritative per `Dirty` value:

- `Dirty::Host` — host `data` is the source of truth; `device_handle` may
  be `None` or stale. Set by `from_slice`, `get_mut`, any CPU op.
- `Dirty::Device` — device handle is authoritative; `data` is empty or
  stale. Set by `alloc_device_tensor` (output of a lazy backend op).
- `Dirty::Both` — both sides are populated and bit-identical. Set after
  `ensure_host` on a device tensor (readback) or after `ensure_device` on
  a host tensor (upload).

### Lifetime & cloning

- `MlxHandle` is `Arc<MlxHandleInner>`; the inner `Drop` runs
  `mlx_array_free` under `mlx_sys::mlx_guard()`. Dropping the last clone is the
  unique free path.
- `Tensor::clone` asserts `dirty != Device` — cloning a device-only
  tensor is a bug (no host data to copy). Call `ensure_host` first. This
  is why `TensorStore::clone_tensor` starts with `ensure_host`.
- The device handle is **not** cloned by `Tensor::clone`; the clone
  starts with `device_handle = None, dirty = Host` and must re-upload if
  it wants device residency again. This keeps clones cheap and
  eliminates two-handle aliasing inside a tape.

### Eval boundary

One explicit flush per step, plus the forced readbacks that CPU-only ops
still cause on their inputs:

```
forward:  lazy ops build graph via Backend::matmul / Backend::add
          (no eval). CPU-only ops (sum, softmax, rmsnorm, gelu, ...)
          call store.ensure_host(input) which evals + readbacks the
          input once. Output of a CPU op is Dirty::Host.
backward: Tape::backward() materializes any still-Dirty::Device output
          of the recorded entries once, then runs the CPU backward
          path on already-host data. Backend-specific overrides
          (matmul_backward on Metal/CUDA) do their own eval+readback
          inside the FFI call — tracked in the eval_count budget.
readback: store.to_host(id) on anything the optimizer / training loop
          wants. Implies eval if dirty == Device.
```

### When an op forces a host readback

Unlike matmul + add, most ops (`sum`, `mean`, `softmax`, `log_softmax`,
`gelu`, `silu`, `exp`, `neg`, `mul`, `mul_scalar`, `rms_norm`, `rope`,
`embedding`, `gather_last_dim`, `scatter_add_rows`) are CPU-only today.
They call `store.ensure_host(input)` on every device-resident input,
which runs one eval per call. This is the gap M5.3b closes — port each
op's forward to a `Backend::<op>` method that takes handles.

Grad accumulation in `tape.rs::merge_grad` + `tensor.rs::accumulate_grad`
also forces a `to_host` on the incoming grad, for the same reason: the
final `iter_mut().zip` sum is host-side. Moving grad accumulation onto
the device is an M5.3b follow-up (see §7.2 M5.3 in
[`docs/plans/rust-agent-rl-single-node.md`](../../docs/plans/rust-agent-rl-single-node.md)).

### Eval counter (Metal only, debug instrumentation)

`backend_metal::{eval_count, reset_eval_count}` expose a
`AtomicU64` incremented on every `mlx_eval` call inside the crate
(`MetalBackend::eval`, `eval_and_readback`, the direct `mlx_eval` in
`mlx_softmax_like`). Used by
`metal_single_forward_backward_step_has_bounded_eval_count` in
`tests/test_device_handle.rs` — bounded at 2 for the reference
`y = x @ w; loss = y.sum(); backward` tape. Strict 1 is the M5.3b-era
goal. Acquire `METAL_TEST_LOCK` if your test uses the counter and runs
alongside other Metal tests in the same binary.

## Active priority — P4 runtime-led train/agent stack

This crate is the autograd substrate underneath P4 (agent/RL/train work
that strengthens the runtime spine). It must stay narrow:

- **Engine + ops + optimizer + LR schedule + checkpoint.** Higher-level
  training surfaces (causal LM heads, GRPO policy, trainers, multi-turn
  eval) live in [`crates/train/`](../train/) and consume this crate, not
  the other way around.
- `adamw_state.rs` is the canonical AdamW checkpoint codec — train-side
  trainers serialize/restore through it, never by reaching into AdamW
  internals directly.
- `lr_schedule.rs` is parse-driven (`parse_lr_schedule`) so train
  binaries can plumb schedules from CLI args without re-implementing the
  curve. New schedules go here.
- M5.3a Metal device-resident port done; M5.3b backend op coverage
  (porting CPU-only ops to lazy device paths) is the next milestone.
- See
  [`docs/projects/agent-rl-self-evolving.md`](../../docs/projects/agent-rl-self-evolving.md)
  §M5 for the device-resident tensor scope and
  [`docs/plans/rust-agent-rl-single-node.md`](../../docs/plans/rust-agent-rl-single-node.md)
  §7.2 for the M5.3b op-coverage plan.

## Tests and benches

- `cargo test -p autograd --release` — CPU-only, ~14 tests.
- `cargo test -p autograd --release --features metal` — adds ~48 tests.
- `cargo check -p autograd --no-default-features --features cuda,no-cuda` —
  Mac typecheck gate for CUDA.
- `examples/bench_step_matmul.rs` — per-step wall-clock sweep CPU vs
  Metal at d_model ∈ {64,128,256,512}. Run:
  `cargo run --release -p autograd --example bench_step_matmul --features metal -- --backend metal --d 128 --iters 200 --batch 32`.
  Acceptance gate from the M5.3a plan: Metal ≥ 1.1× CPU at d_model=128.

## Related memories

- [`feedback_no_half_states.md`](../../.claude/projects/-Users-user-code-agent-infer/memory/feedback_no_half_states.md)
- [`feedback_m2b_claude_writes.md`](../../.claude/projects/-Users-user-code-agent-infer/memory/feedback_m2b_claude_writes.md)
- [`feedback_matched_ab_for_small_bench_effects.md`](../../.claude/projects/-Users-user-code-agent-infer/memory/feedback_matched_ab_for_small_bench_effects.md)
</content>
</invoke>
