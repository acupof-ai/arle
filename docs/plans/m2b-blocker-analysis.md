# M2b (LoRA → Qwen3 infer hook) — dependency blocker

**Status**: Blocked · **Opened**: 2026-04-18 · **Owner**: ckl

## Finding

M2.3 as written in [`rust-agent-rl-single-node.md`](rust-agent-rl-single-node.md) §4.2 requires a **CUDA-resident LoRA adapter** that `infer/src/ops/linear.rs` can multiply against and the train crate can update with gradients. That primitive does not exist yet.

## Evidence

- `crates/autograd/src/ops/` — 10 files, **zero CUDA**. All ops are pure-Rust CPU `Vec<f32>` (grep `cudarc|CudaSlice|cuda`: 0 hits).
- `crates/train/src/lora.rs` — LoRAAdapter A/B live as `TensorId` in `autograd::TensorStore` (f32 CPU).
- `infer/src/ops/linear.rs` — consumes `DeviceMatrix { data: CudaSlice<bf16>, … }`. No path takes autograd tensors.
- Plan §4.2 M2.4 explicitly gates on "autograd `GpuTensor` 支持 frozen view 模式" — **frozen view on what storage?** Implicit answer: CUDA. That work is M1.2 + M1.8 in §3.2 and has not been done.

## Options

### (a) Do M1 CUDA path first, then M2b as specified — recommended

- Implement `autograd::ops::matmul` CUDA path (cuBLAS via cudarc, fwd+bwd, row-major/col-major done correctly per `feedback_mlx_rope_layout`-style precedent from mni-ml).
- Implement `autograd::optim::AdamW` CUDA fused kernel.
- Add `autograd::GpuTensor::frozen_view(base_cuda_ptr)` so a base weight `DeviceMatrix` can be wrapped as a non-tape tensor.
- Only then wire `infer/src/ops/linear.rs` with `lora: Option<&LoRAAdapter>`; the adapter's A/B are CUDA-resident `GpuTensor`s that train can `backward` and AdamW-step in place.

Cost: ~10–14 days (M1.2 + M1.8 + frozen-view). Correct, matches plan as designed. Unblocks Qwen3 RL.

### (b) Stub M2b as forward-only hook with CUDA-native LoRAAdapter

- Define `LoRAAdapter { a: DeviceMatrix, b: DeviceMatrix, scale: f32 }` **inside `infer/`** (not shared with train).
- `linear.rs` takes `Option<&LoRAAdapter>`; if present, runs extra `gemm_into(a, x)` → `gemm_into(b, tmp)` and `add_into(y, y_lora)`.
- Training side stays CPU/TinyLM only; Qwen3-LoRA is inference-time only (e.g., loading a pre-trained external LoRA).

Cost: ~2 days. Ships the hook but does not close the train↔infer loop. Deferred gradient integration.

### (c) Skip M2b, proceed to M4 agent-multi-turn on TinyLM

- Treat TinyLM + GRPO (now green) as the end-to-end training target for M4 agent work.
- Qwen3 LoRA RL blocked until M1 CUDA path lands; revisit later.

Cost: 0 days. Lets research/agent-loop work proceed. Best if the immediate goal is "self-evolving agent prototype" not "Qwen3 RL".

### (b′) Merge-time utility — **added 2026-04-18 after industry survey**

- Small CLI / binary that takes `(base_weights_path, lora_adapter_path, out_path)`, computes `W' = W + alpha/rank · B @ A` per-layer in fp32, and writes a new checkpoint. Merge happens **before** the usual Qwen3 load path — zero changes to `infer/src/ops/linear.rs` or any model file.
- Matches the **llama.cpp default** and the mistral.rs "bake-in" path. See `docs/research/2026-04-18-lora-inference-patterns.md` for the survey: llama.cpp merges into GGUF, mistral.rs supports dynamic swap only because they target multi-tenant serving, vLLM needs Punica SGMV only at thousands-of-adapter scale. For a single-researcher self-evolve loop, dynamic swap is not on the path.

Cost: ~1 day, zero risk to the hot path. Doesn't unlock the train↔infer gradient loop (that's still option (a)) but it does ship "Qwen3 + externally-trained LoRA works" with no GPU verification needed from ckl.

## Recommendation (updated 2026-04-18)

Original recommendation was **(a)**. After the industry survey (llama.cpp merges by default, mistral.rs runtime hook exists but adds overhead, Punica is multi-tenant-only), **(b′)** is the pragmatic first move: it ships the user-visible feature without touching the hot path, and leaves (a) fully open for when the project actually needs real-time train↔infer gradient flow.

Pick **(b′)** if the near-term goal is "a Qwen3 binary that can consume a LoRA someone else trained." Pick **(a)** if the goal is "RL-train a Qwen3 LoRA and immediately serve it in the same process." The project's current stated goal is the former.

## Decision pending

Requires user input before coding. Cron loop should not auto-start M2b against the blocker.
