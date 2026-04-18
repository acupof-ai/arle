# M2b (LoRA → Qwen3 infer hook) — dependency blocker

**Status**: Resolved (option **b**) · **Opened**: 2026-04-18 · **Resolved**: 2026-04-18 · **Owner**: ckl

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

## Resolution (2026-04-18)

Shipped option **(b)**: CUDA-native `LoRAAdapter { a: DeviceMatrix, b: DeviceMatrix, scale: f32 }` lives in `infer/src/model/qwen3/lora.rs` — not in `train::autograd`. Three commit waves:

1. `cbe9cba` phase1 — PEFT `adapter_config.json` + `adapter_model.safetensors` loader, f32→bf16 upload, B pre-scaled by `alpha/r` at load time.
2. `adf205a` phase2a — additive `apply_lora_gemv_add` / `apply_lora_gemm_add` (one cuBLAS small GEMM per projection, summed into base output).
3. `1e87b4f` phase2b — wired into prefill + decode hot paths of `infer/src/model/qwen3/forward.rs`.

Review-driven hardening (`146874e` → `09e3685`): LoRA active ⇒ `supports_cuda_graph_decode=false` (eager decode only, because `apply_lora_*_add` allocates per-call temp DeviceVecs that CUDA Graph capture rejects); warmup still runs the autotune pass so cublasLt cache is populated; paged-KV slots are freed on every warmup exit path; synthetic-safetensors integration test at `infer/tests/test_qwen3_lora_loader.rs` validates loader end-to-end (bf16-exact index-dependent fills, disjoint A/B value ranges, scale pre-bake).

Option (a) — autograd CUDA path + in-process train↔infer gradient flow — remains the correct next step when the project actually needs online LoRA training against Qwen3. This resolution only closes "consume a pre-trained LoRA on Qwen3."
