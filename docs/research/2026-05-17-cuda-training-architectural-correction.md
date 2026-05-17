# CUDA training stack — architectural correction post Wave 1 KILL

**Date**: 2026-05-17 EOD · **Supersedes** the Wave-1 prediction in
[`2026-05-17-cuda-training-step-nsys-attribution.md`](2026-05-17-cuda-training-step-nsys-attribution.md)
· **Trigger**: Wave 1 shipped (commit on `main` HEAD), tok/s 92.08 →
91.41 (−0.7 %), `nsys` confirms max DtoH single transfer is **still
1 016 MB** — identical to pre-Wave-1. The "Wave 1 = 1 GB DtoH kill"
prediction was falsified.

## What Wave 1 taught us

Wave 1 shipped:
- `Backend::log_softmax_last_axis_backward` + CUDA NVRTC kernel
- `Backend::gather_last_dim_backward` + CUDA NVRTC kernel
- Ops-layer dispatch that routes backward through the device path when
  the upstream gradient is `Dirty::Device`
- `Tensor::clone_tensor` preserves the `DeviceHandle` `Arc` for
  device-resident tensors

Parity gates green (5/5). But the dispatchers **never fire in
production** because the upstream gradient in the CE-loss chain is
host-resident: `mean_backward` (Wave 2) emits a host `Vec<f32>`, and
`Backend::matmul_backward` *signature* itself takes
`grad_out: &[f32]` — a host slice — for the LM-head GEMM's
backward. Every gradient flowing through a matmul backward (which is
"most gradients in a transformer") is forced to host.

## Where the 1 GB DtoH actually is

Re-tracing the CE-loss backward graph with the new evidence:

```
forward:
   hidden:[B,S,H] ── matmul ──→ logits:[B,S,V] (device)
                                  │
                                  └→ log_softmax → [B,S,V] (device, M5.3b)
                                       │
                                       └→ gather(targets:[B,S]) → [B,S]
                                            │
                                            └→ mean → scalar loss

backward:
   d_loss → mean_backward → [B,S] grad        (HOST today)
                              │
                              ↓ ensure_host (small)
                            gather_backward → [B,S,V] grad
                              │
                              ↓ ensure_host (1 GB — but logits-grad
                              │              shape, used by next step)
                            log_softmax_backward → [B,S,V] grad
                              │
                              ↓ ensure_host (1 GB — gets fed into
                              │              matmul_backward as
                              │              grad_out: &[f32])
                          matmul_backward(hidden, W, grad_out)
                              │   ↑
                              │   └─ THIS is the host-slice contract
                              │       that forces the 1 GB readback
                              ↓
                          (grad_hidden, grad_W) — both host Vec<f32>
```

The 1 GB readback is **not** in the log_softmax-grad step. It's the
`matmul_backward` call for the LM-head GEMM consuming its `grad_out`
argument. Wave 1's kernel never gets to materialize a device-resident
output because by the time backward reaches it, the upstream has been
yanked to host by an earlier op in the chain — typically by
`merge_grad` / `accumulate_grad` (which works on host
`Vec<f32>` today) or by the matmul-backward signature itself.

## The actual structural fix

To kill **any** of the 9 032 memcpys per step, we need a
**device-resident gradient tape**. Concrete contract changes:

1. **`Backend::matmul_backward` signature** changes from:
   ```rust
   fn matmul_backward(
       &self, a: &[f32], a_shape: &[usize],
       b: &[f32], b_shape: &[usize],
       grad_out: &[f32], grad_out_shape: &[usize],
       need_grad_a: bool, need_grad_b: bool,
   ) -> Result<(Vec<f32>, Vec<f32>)>;
   ```
   to (additive method, keep the old one for CPU/Metal fallback):
   ```rust
   fn matmul_backward_device(
       &self,
       a: &DeviceHandle, a_shape: &[usize],
       b: &DeviceHandle, b_shape: &[usize],
       grad_out: &DeviceHandle, grad_out_shape: &[usize],
       need_grad_a: bool, need_grad_b: bool,
   ) -> Result<(DeviceHandle, DeviceHandle)>;
   ```
   With CUDA cuBLAS GEMM the math is identical — just two more SGEMM
   calls (grad_a = grad_out @ Bᵀ, grad_b = Aᵀ @ grad_out). The new
   override returns unevaluated `CudaSlice` handles. The old host-host
   path stays for the CPU backend.

2. **`Tape::backward` dispatches device-aware backward when the saved
   forward inputs are `Dirty::Device`.** This requires the per-op
   `BackwardOp` enum entries to know whether their saved context is
   on device, and to call the right backend method. The dispatch is
   per-op, but the change to the `Backend` trait is one new method
   per backward kind.

3. **`accumulate_grad` and `merge_grad` work on device.** These run
   inside `Tape::backward` whenever two backward paths converge on the
   same parameter. Today they do `to_host(grad_id)` + summation on
   host `Vec<f32>`. New version needs an `add_into_device` kernel
   (trivial — 1 SAXPY) and to skip `to_host`. Without this, every
   merge re-pulls the gradient to host even if both inputs were
   device-resident.

4. **Wave 2 host ops** (rms_norm, rope, silu, mul, mul_scalar,
   add_broadcast, embedding, mean, exp, neg, scatter_add_rows) need
   their backward variants going through the same device-lazy
   dispatch as Wave 1. Wave 1's pattern carries over mechanically;
   the difference is that with `matmul_backward_device` and
   `accumulate_grad_device` landed, these will actually fire in
   production instead of getting masked by an upstream host
   readback.

## Why ship Wave 1 anyway

Wave 1 is committed even though it's a measurement no-op because:
- Parity-validated kernels exist (correctness gate green).
- Without them in tree, Wave 2 + `matmul_backward_device` would have
  to ship them anyway. Splitting that future commit gets harder, not
  easier, the more pieces it has.
- The `Tensor::clone_tensor` fix (preserve `DeviceHandle::Arc` for
  `Dirty::Device` tensors) was a real prerequisite bug fix that
  blocked the device-aware grad chain. Even if the rest of Wave 1
  did nothing, this one diff was required.

This is the same SOLID-prerequisite framing as G3 (CUDA adamw_step
override): code shipped, measured win = 0, but the next milestone
can't move without it.

## The right next milestone

Going forward we need to scope this as **one structural project**,
not a stack of one-op-at-a-time wins:

### Project: device-resident gradient tape

| # | Piece | Files | Difficulty |
|---|---|---|---|
| 1 | `Backend::matmul_backward_device` + CUDA cuBLAS override | `backend.rs`, `backend_cuda.rs` | M |
| 2 | `Backend::add_into_device` + CUDA kernel for grad merge | `backend.rs`, `backend_cuda.rs`, `backend_cuda/kernels/add_into.cu` | S |
| 3 | `Tape::backward` device-aware dispatch (`BackwardOp::*` know device-ness) | `tape.rs` | M |
| 4 | `accumulate_grad` / `merge_grad` device path | `tensor.rs` | S |
| 5 | Backward variants for the 11 Wave-2 ops (rms_norm, rope, silu, mul, mul_scalar, add_broadcast, embedding, mean, exp, neg, scatter_add_rows) — kernels + trait + dispatch | many | L |
| 6 | Parity tests per backward op | `tests/test_cuda_lazy_ops.rs` | M |
| 7 | nsys acceptance gate: memcpy count **< 100/step** (was 9 032) | bench dir | gate |
| 8 | tok/s bench across a 10-step run | bench dir | gate |

Estimated total: **3-5 days of focused engineering** for the device-resident
gradient tape, assuming no surprises on `Tape::backward` rewiring.

### Predicted impact on `tok/s`

With memcpy count `9 032 → < 100`, the API-time budget drops from
**45.5 s/step (memcpy) → < 1 s**. The remaining 133 s/step (CPU
compute on uploaded tensors) **also goes away**, because once the
chain is device-resident there are no uploads. Predicted `tok/s`
after the device-resident tape: bounded by actual GPU compute, which
profiled at 888 ms/step → if all of that becomes the bottleneck the
ceiling is `16 384 tokens / 0.888 s ≈ 18 450 tok/s` per step.

That's a **~200×** speedup over today's 92 tok/s, and it lands ARLE
**above the conservative industry baseline target** (≥ 15 000 tok/s
on a 4070 Ti SUPER for a ~40 M-class model).

**This is the milestone that closes the headline target.** Everything
else (FusedLinearCE, bf16, CUDA-graph capture) is incremental on top
of a sane device-resident tape, and brings further wins on the order
of 1.5-2× each but only matters once the host-bound regime is
actually closed.

## Acceptance gate going forward

For the device-resident-tape project, the **single binding gate** is:

> `nsys cuda_api_sum` memcpy count per training step **< 100**.

If memcpy count doesn't drop below 100/step, the device-resident path
is still leaking — somewhere a host roundtrip is happening, and tok/s
gains will plateau before they hit the 18 000 ceiling.

Today: 9 032/step. M5.3b ceiling effectively reached at 92 tok/s.

## What this retires

Items in [`2026-05-17-cuda-training-step-nsys-attribution.md`](2026-05-17-cuda-training-step-nsys-attribution.md)
§Evidence-based ranking that are now wrong or incomplete:

- "Wave 1 expected ~3-5 % wall" — actual 0 %, falsified.
- "Wave 2 expected ~3-5× speedup" — only true if combined with
  `matmul_backward_device` + `accumulate_grad_device`. In isolation,
  Wave 2 has the same Wave-1 failure mode.
- The 4-wave plan was correct in *direction* but wrong in
  *granularity*. The first three rows collapse into one project
  ("device-resident gradient tape"), not three sequential commits.

## Rule (added on top of the nsys-attribution rule)

**Device-residency is a graph property, not a per-op property.** A
single host-only op in a backward chain forces the whole chain
through host. The acceptance gate must measure end-to-end memcpy
count, not the residency of individual ops.

This is the second time we've spent ~half a milestone landing
mechanically-correct code that delivered zero throughput (G3,
Wave 1). Pattern: an op port whose precondition is a missing
contract-level change buys you nothing. Going forward, before
proposing the next per-op port, check whether its **gradient input**
will be device-resident in production — if not, the contract-level
change comes first.
