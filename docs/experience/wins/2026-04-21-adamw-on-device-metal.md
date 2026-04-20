# AdamW on device — Metal lazy composition (M5.3b.10)

**Status:** `pending-remote` — functional parity + eval-count micro-assertion
landed locally; real-model bench on Qwen3.5-scale parameters pending a
remote-capable box (Mac M-series is fine, but the per-step deltas only
show meaningfully when there are 200+ parameters × param-size bytes of
re-upload to kill). Cross-reference this win from the Qwen-class RL plan
once it reruns.

## Context

Through M5.3b.1–9, every forward op on Metal runs inside the MLX lazy
graph — matmul, add, sum, softmax/log_softmax, silu, exp, rope, rmsnorm,
embedding, gelu, gather_last_dim — and backward lowers through the tape's
batch-flush so the host never sees intermediates mid-step. The remaining
hot readback was the optimizer: `AdamW::step` host-iterated every
parameter element-wise, using `get_mut(param_id)` to read the current
weight. `get_mut` auto-triggers `ensure_host`, which does an `mlx_eval`
+ host readback, and the subsequent write flips `Dirty::Host` — which
forces a full upload the next forward pass. On Qwen3.5-class models
(~200 parameters, d_model=2560) this is ~200 × param-size bytes × 2
(down + up) every training step. Tape-batch-flush was already
cheap — optimizer was the biggest remaining churn point per step.

Goal: keep AdamW state + weight updates on the device, collapse every
per-param eval barrier into at most a single terminal `mlx_eval`, and
preserve the existing 15 `AdamW::new(...)` call sites with zero edits.

## What Worked

### Trait addition (`crates/autograd/src/backend.rs`)

Added `Backend::adamw_step` with a signature that takes the current
param bytes + m/v state + grad bytes + scalars, returns updated
(param, m, v). Default impl `cpu_adamw_step_in_place` runs the classic
host loop — so `CpuBackend` and the existing `AdamW::new` (non-device)
path share it, and a backend that doesn't override keeps working.

### Metal override (`crates/autograd/src/backend_metal.rs`)

Full lazy composition — no intermediate `mlx_eval` between arithmetic
ops. Pipeline per parameter:

1. Upload grad via `mlx_array_from_data` (param + m + v arrive already as
   `DeviceHandle`s — their handles are reused verbatim, zero re-upload).
2. Compose `new_m = β1·m + (1-β1)·g` via two `mlx_multiply` + one
   `mlx_add`. Same for `new_v = β2·v + (1-β2)·g²` (plus `g² = g·g`).
3. `m_hat = new_m · (1/bc1)`, `v_hat = new_v · (1/bc2)` — bias correction
   scalars computed host-side and uploaded as 0-d arrays.
4. `denom = sqrt(v_hat) + eps` — `mlx_sqrt` + `mlx_add` with an eps
   0-d array.
5. `ratio = m_hat · reciprocal(denom)` — `mlx_divide` doesn't exist in
   mlx-sys, so compose `a/b = a · 1/b` via `mlx_reciprocal` +
   `mlx_multiply`. Avoids adding a new FFI binding inside this milestone.
6. `scaled_update = lr · ratio`.
7. `decayed_param = (1 - lr·wd) · param` — single `mlx_multiply` with
   the decay scalar.
8. `new_param = decayed_param - scaled_update` via `mlx_subtract`.
9. **Single terminal `mlx_eval([new_param, new_m, new_v])`** — one
   barrier covering all three outputs.

Return: three fresh `DeviceHandle`s. The optimizer then stamps them
back via `TensorStore::replace_device_handle(...)`.

### Moment storage split (`crates/autograd/src/optim.rs`)

`ParamMoments.m`/`v` used to be `Vec<f32>`. Replaced with
`MomentStorage::Host(Vec<f32>) | Device(DeviceHandle)` — host on the CPU
training path, device on the Metal path. `AdamW` gained a new field
`backend: Option<Arc<dyn Backend + Send + Sync>>` and a new constructor
`AdamW::new_with_device(lr, betas, eps, wd, backend) -> Self`. The
existing `AdamW::new(...)` constructor is untouched; all 15 call sites
keep compiling with zero edits.

The `step()` method branches once on `self.backend.is_some()`:
- `step_host` (identical to the pre-M5.3b.10 loop) when no backend is
  attached — the CPU training path.
- `step_device` when a backend is attached — upload grad + param if
  needed (`ensure_device`), call `backend.adamw_step(...)`, stamp the
  three returned handles back with `replace_device_handle`.

### The `get_mut` trap — and how we sidestep it

`TensorStore::get_mut(id)` internally does:

```rust
if tensor.dirty == Dirty::Device { self.ensure_host(id)?; }
// ...
tensor.dirty = Dirty::Host;
```

That's load-bearing for host-mutating callers, but it's exactly the
churn we're trying to kill. After `adamw_step` returns a new
`DeviceHandle` for the updated weight, going through `get_mut` would
(a) evaluate the brand-new graph + read it back, and (b) mark the
tensor `Dirty::Host` — guaranteeing the next forward pass re-uploads.

Added `TensorStore::replace_device_handle(id, handle)` which bypasses
`get_mut`:

- Installs the supplied `DeviceHandle` directly.
- Sets `dirty = Dirty::Device`.
- Clears the host `data` buffer (it's stale — the real data lives on
  the device under the new handle).

Matching helper for moment storage: the optimizer keeps the returned
`new_m` / `new_v` handles in its own `MomentStorage::Device(handle)`
slot — they never enter the store at all, so `replace_device_handle` is
only needed for the param weight.

### Moment readback — `moments_host` accessor

Export/import codec (`adamw_state.rs`) still wants `(Vec<f32>, Vec<f32>)`
for serialization. Renamed the `state_for(id)` accessor to
`moments_host(id)` which:

- For host-backed moments: returns `(m.clone(), v.clone())` (same as
  before; shape preserved).
- For device-backed moments: calls `backend.readback(&handle)` for each
  side, returns the owned Vec<f32>.

The import path (`set_state`) accepts `(m_vec, v_vec, shape)` and
uploads to the device if the backend is attached, else stashes host-only.

## Tests

`crates/autograd/tests/test_device_handle.rs`:

```rust
#[test]
#[cfg(feature = "metal")]
fn metal_adamw_step_stays_device_resident() {
    // 8x8 param, 5 AdamW steps with fixed linear-congruential RNG.
    // lr=1e-3, betas=(0.9, 0.95), eps=1e-8, wd=0.01.
    // (1) Run the same sequence through AdamW::new (host) and
    //     AdamW::new_with_device(backend) (Metal), same grad stream.
    // (2) After step 5: final L2(param_host - param_metal) <= 1e-5.
    // (3) Per-step: store.get(param).unwrap().dirty == Dirty::Device
    //     after AdamW::step returns — no intervening ensure_host.
    // (4) Per-step: eval_count delta <= 2 (observed: 1/step).
}
```

Eval budget micro-assertion — the gate for landing this change. Observed
delta per step: **1 mlx_eval** (the terminal one over
`[new_param, new_m, new_v]`). Budget set to `≤ 2` for headroom if a
future backend rewrite needs one extra barrier (e.g. to split grad
upload from compute); we're well under it.

`cargo test -p autograd --features metal --release` → **15/15 device
tests green (including the new one) + 42 CPU parity + 8 LR schedule +
1 optimizer trait + 5 lib unit**. No collateral regression.
`cargo test -p train --features metal --release` → **36/36 green**.

## Not yet validated (remote-pending)

- **Qwen3.5-class RL closed loop**. The per-step re-upload savings only
  show meaningfully when there are hundreds of params — TinyLM at
  d_model=64 won't move the needle. Re-run the 2026-04-20 matmul-backward
  Metal baseline against this change. Expected: further iter/s climb on
  top of the 2.94 iter/s TinyLM number, with a larger relative delta at
  Qwen3.5 scale.
- **Bench entry proper**. Once that rerun lands we'll have a concrete
  Δ% number for the guidellm wins template, cross-linked from the M5.3
  plan row and this stub promoted to a full wins entry.

## Files changed

- `crates/autograd/src/backend.rs` — added `Backend::adamw_step` trait
  method + `cpu_adamw_step_in_place` default implementation.
- `crates/autograd/src/backend_metal.rs` — added `MetalBackend::adamw_step`
  override composing the entire update on the MLX lazy graph under
  `MLX_GUARD`; single terminal `mlx_eval`.
- `crates/autograd/src/optim.rs` — `MomentStorage::Host | Device`
  enum; `AdamW::new_with_device`; branched `step_host` / `step_device`;
  renamed `state_for` → `moments_host` (device-aware readback).
- `crates/autograd/src/adamw_state.rs` — updated `export_state` to use
  `moments_host` (owned Vec).
- `crates/autograd/src/tensor.rs` — added
  `TensorStore::replace_device_handle(id, handle)` that bypasses
  `get_mut`'s `ensure_host` auto-trigger and clears the stale host buffer.
- `crates/autograd/tests/test_device_handle.rs` — added
  `metal_adamw_step_stays_device_resident` (5-step parity + `Dirty::Device`
  + eval-budget assertions).

## Rule

When an optimizer rides on top of a lazy-graph backend, the three
moving parts are:

1. **State storage** — m/v must live on the device; serializing them
   every step kills the win.
2. **Update composition** — the whole formula (`β1·m + (1-β1)·g`,
   bias correction, denom, scaled_update, decay, subtract) is one lazy
   chain with a single terminal eval barrier.
3. **Handle plumbing** — after the backend returns a fresh param
   handle, install it *without* going through any path that
   auto-flushes to host. `get_mut` is a trap; add a direct
   `replace_device_handle`-style method to bypass it.

Miss any of the three and you get the churn back. The dirty flag state
machine only catches the cases it knows about — it can't know that the
optimizer is about to overwrite the host buffer, so an explicit bypass
is required.
