# OPD Frozen-BF16 LoRA Student Step 0 Audit

## Goal

Unblock 9B GPTQModel teacher -> Qwen3.5-0.8B LoRA OPD on a 16 GB card after
the 2026-05-22 memory kill.

This is an architecture audit, not a fix. The next implementation tranche must
keep the LoRA student base in BF16 when it is frozen; the current train-side
path expands the base to f32 and exhausts the remaining memory after the 9B
runtime teacher is resident.

## Evidence

The licensed 9B GPTQModel teacher path now passes the single-token full-logits
gate:

- top-64 dominant relerr: `0.1242236`
- top-64 RMSE/reference-RMS: `0.0428670`
- ARLE argmax: `11`
- PyTorch BF16 argmax: `11`

The OPD bench then fails before `eval_summary step=0`. A single-token,
rollout-1 control fails in the same place, so prompt length and rollout length
are not the root cause.

Upload diagnostics narrowed the failure to a small first-forward upload:

```text
cuda htod copy failed: shape=[1024, 3584] len=3670016 bytes=14680064 \
err=DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")
```

The failed allocation is only `14.68 MB`; that means the runtime is already out
of usable GPU allocation headroom before the student forward can finish
uploading its f32 base weights.

## Code Path

Current LoRA student load path:

```text
load_qwen35_lora_from_hf_dir
  -> load_qwen35_from_hf_dir_inner(..., LoadMode::LoraStudent)
  -> Qwen35Model::new_with_lora_targets
  -> load_planned_tensor_into_slot
  -> dtype_to_f32
  -> Tensor::new(Vec<f32>, ...)
```

Relevant source facts:

- `crates/train/src/qwen35_loader.rs:534-548` constructs the LoRA student.
- `crates/train/src/qwen35_loader.rs:604-609` allocates a normal train-side
  `Qwen35Model` with LoRA adapters.
- `crates/train/src/qwen35_loader.rs:797-817` calls `dtype_to_f32` and writes
  every checkpoint tensor as `Tensor { data: Vec<f32>, ... }`.
- `crates/train/src/qwen35.rs:1204-1231` allocates embedding and lm_head as
  ordinary f32 tensors even in frozen/LoRA mode.
- `crates/train/src/lora.rs:86-145` stores each base linear weight as a
  `TensorId`; adapters are also `TensorId`s.
- `crates/autograd/src/tensor.rs:247-265` uploads tensors through
  `Backend::upload(&[f32])` on first device use.
- `crates/autograd/src/backend_cuda.rs:88-109` implements that upload as f32
  `clone_htod`.
- `crates/train/src/qwen35.rs:2284-2325` and
  `crates/train/src/lora.rs:147-193` route linear projections through
  `matmul_bt(flat_x, weight)`.
- `crates/autograd/src/ops/embed.rs:13-84` forces CUDA embedding tables through
  `ensure_device(table)`, so tied embedding/lm_head is also uploaded as f32.

Therefore the memory kill is not in the 9B teacher anymore. It is the
0.8B-student frozen base being represented and uploaded as f32.

## Implementation Shape

The fix should be a frozen-base path, not a global BF16 autograd conversion.
LoRA adapters, activations, loss, gradients, AdamW state, and optimizer math can
stay f32. Only checkpoint-backed frozen base weights need BF16 residency.

Recommended implementation order:

1. Add a typed frozen device handle for CUDA BF16 weights.
   - Add a BF16 CUDA storage variant or equivalent wrapper in autograd.
   - It must carry shape/size metadata and never expose a fake f32 host buffer.
   - It is CUDA-only for v1; CPU fallback can keep the existing f32 path.

2. Add explicit frozen BF16 upload APIs.
   - Loader should be able to upload safetensors BF16 bytes directly without
     `dtype_to_f32`.
   - `Dirty::Device` invariant should hold: host `data` empty, device handle
     authoritative.
   - F16/F32 checkpoint tensors can remain widened to f32 until a later tranche;
     Qwen3.5-0.8B BF16 is the current acceptance target.

3. Add mixed matmul for frozen BF16 RHS weights.
   - `flat_x` remains f32.
   - RHS base weight is BF16.
   - output remains f32.
   - no gradient for the frozen RHS.
   - CUDA implementation should use cuBLAS BF16-capable GEMM or a clearly
     documented fallback. A fallback that widens the whole RHS to f32 on device
     does not license the memory axis.

4. Add BF16 embedding gather for frozen embedding tables.
   - Qwen3.5-0.8B tied embedding/lm_head is a large tensor; leaving it f32
     burns much of the expected saving.
   - Output remains f32 so the rest of the model and LoRA path stay unchanged.

5. Teach `LinearWithLora` / `Qwen35Model` to distinguish frozen base weights
   from trainable adapter tensors.
   - Keep existing public `Qwen35Model` methods stable.
   - Prefer a local enum around base weights over changing all tensor math.
   - Only the LoRA adapter ids should be trainable and optimizer-tracked.

6. Add `load_qwen35_lora_bf16_base_from_hf_dir` or a load option with an
   explicit name.
   - Do not silently change the existing f32 loader semantics.
   - The benchmark harness can opt in first.

## Gates

Correctness gates:

- `cargo test -p train teacher_infer --release --features cuda`
- `cargo test -p train --test test_opd_determinism --release`
- Qwen3.5-0.8B self-teach: frozen-BF16-base LoRA logits vs current f32-base
  LoRA logits, dominant top-64 relerr <= `5e-2`.
- One OPD step loss relerr vs current f32-base path <= `5e-2` on the 0.8B
  self-teach configuration.

Memory/perf gates:

- With 9B GPTQModel teacher loaded, Qwen3.5-0.8B LoRA student load + first eval
  must reach `eval_summary step=0` without OOM.
- Peak memory target: <= `15.0 GiB` on the RTX 4070 Ti SUPER 16 GB card.
- 100-step 9B GPTQModel -> 0.8B LoRA OPD must show held-out KL monotonic at
  `0/25/50/100` before any headline switch.

Kill gates:

- If the implementation widens the entire frozen base to f32 on device, KILL;
  it cannot solve the observed memory failure.
- If embedding/lm_head stays f32 and the 9B bench still OOMs, KILL that partial
  tranche and finish BF16 embedding before retrying.
- If dominant-logit relerr exceeds `5e-2` on the 0.8B self-teach control,
  stop and add stage-level attribution before running the 9B bench.

## Deferred

This audit does not implement the BF16 path. It is intentionally separated
because the real fix crosses autograd storage, CUDA backend, Qwen35 model
projection wiring, embedding, and the HF loader. A partial implementation that
touches only one of those layers would create a half-state and would not be
SOLID.

