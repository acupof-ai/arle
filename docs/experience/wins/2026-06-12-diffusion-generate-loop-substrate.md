# Diffusion Generate Loop And Metal Bridge

## Context

`mlx-community/diffusiongemma-26B-A4B-it-4bit` is not an autoregressive
Qwen-style MLX checkpoint. Its public config declares a fixed block-diffusion
canvas and an entropy-bound denoising sampler. The current rewrite Metal path is
still a Qwen3.5/Qwen3.6 compiled-step executor, so treating DiffusionGemma as a
normal decode loop would hide the real missing backend work.

## What Worked

Added the backend-neutral block-diffusion loop, then attached a first real
Metal DiffusionGemma bridge:

- `DiffusionGenerationConfig::diffusion_gemma` captures the public generation
  defaults: canvas 256, 48 denoise steps, `t_min=0.4`, `t_max=0.8`,
  confidence threshold 0.005, entropy bound 0.1, and EOS ids `[1, 106, 50]`.
- `DiffusionBlockModel` splits backend work into prompt prefill, canvas
  prediction, and whole-canvas commit.
- `generate_diffusion` owns the fixed-canvas loop, entropy-bound acceptance,
  stability/confidence convergence, stop-token truncation, and per-step traces.
- `diffusion_prediction_from_logits` provides a host bring-up helper for tests
  and first backend wiring; production backends should keep this sampling on
  device.
- `infer-seam::BufferedDiffusionExecutor` adapts a `DiffusionBlockModel` into
  the normal `BackendExecutor` contract: final prompt prefill runs the block
  generator once, the first token is returned immediately, and remaining
  committed tokens are buffered across decode ticks.
- `infer-core` now has an Engine-level completion test proving the buffered
  diffusion executor survives chunked prefill, request sampling params, length
  finish, and the normal completed-request path.
- `infer-core` normalizes the effective Engine `max_tokens` into
  `SamplingParams.max_new_tokens` when callers omit it, so direct
  `Engine::submit_request` callers and HTTP callers reach the diffusion adapter
  with the same generation limit.
- `infer-seam::BufferedDiffusionExecutor` reports zero reusable prefix-cache
  pages. A repeated-prompt Engine regression test proves the adapter does not
  skip diffusion generation on a radix hit.
- `gemma-spec` parses the target DiffusionGemma top-level config, Gemma4 nested
  RoPE parameters (`sliding_attention` / `full_attention`), and MoE fields.
- `infer-metal` now has a dedicated `MetalDiffusionGemmaModel` implementing
  `DiffusionBlockModel`. It loads `model.decoder.*` weights, honors per-weight
  MLX affine quantization overrides (8-bit text/router/embed, 4-bit experts and
  self-conditioning by default), handles full-attention layers with missing
  `v_proj` as K=V, and registers a `mlx-sys` opaque C++ model.
- `mlx-sys` now contains a DiffusionGemma/Gemma4 C++ forward bridge. First
  version recomputes `context + canvas` per denoise pass with a block attention
  mask, Gemma4 `scale=1` attention, RMSNorm/QK norm/V no-weight norm, RoPE,
  dense MLP with tanh GELU, MoE experts with exact GELU, self-conditioning, tied
  embedding logits, softcap, entropy, argmax, and MLX categorical sampling on
  device. It returns only canvas-sized sampled/argmax/entropy arrays to Rust.
- `infer-api` routes Metal DiffusionGemma checkpoints to
  `BufferedDiffusionExecutor<MetalDiffusionGemmaModel>` over a host admission KV
  pool instead of the Qwen `MetalExecutor`. Prefix reuse is disabled and the
  scheduler is clamped to single-flight.
- DiffusionGemma chat is fail-closed: the target tokenizer has
  `chat_template=null`, so `/v1/completions` is licensed first while
  `/v1/chat/completions` returns an explicit unsupported-template error instead
  of silently using Qwen ChatML.
- `infer-vulkan`'s Gemma4 contract now pins plain-scale RMSNorm and the
  DiffusionGemma/Gemma4 MoE shape where dense MLP remains present and router /
  routed experts are additional work.
- `infer-api` CUDA model classification detects DiffusionGemma and fails closed
  instead of falling back to dense Qwen3. CUDA/Vulkan DiffusionGemma forwards
  remain unwired.

Support status is documented as Metal-wired but pending runtime artifact smoke:
the 26B 4-bit checkpoint is not cached on this Mac, so this tranche has compile
and structural evidence but no target-model generation transcript yet.

## Verification

```bash
cargo fmt --check
cargo test -p gemma-spec --release
cargo test -p infer-plan --release
cargo test -p infer-seam --release
cargo test -p infer-core --release
cargo test -p infer-server --release tokenizer::tests
cargo test -p infer-vulkan --release --no-default-features
cargo test -p infer-metal --release --no-default-features --features metal
cargo test -p infer-api --release --no-default-features --features cpu,no-cuda
cargo test -p infer-api --release --no-default-features --features metal,no-cuda
cargo clippy -p gemma-spec -p infer-plan -p infer-seam -p infer-core -p infer-vulkan --release --no-default-features -- -D warnings
cargo clippy -p infer-metal --release --no-default-features --features metal -- -D warnings
cargo clippy -p infer-api --release --no-default-features --features cpu,no-cuda -- -D warnings
cargo clippy -p infer-api --release --no-default-features --features metal,no-cuda -- -D warnings
CUDARC_CUDA_VERSION=12080 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib
git diff --check
```

No benchmark was run because this is a new model bring-up path and the target
checkpoint is not present in the local HF cache. Before any support level higher
than pending-local-smoke, run `/v1/completions` against
`mlx-community/diffusiongemma-26B-A4B-it-4bit` and record the prompt, generated
text, denoise stats, memory guard line, and failure/retry details here or in a
follow-up wins/errors entry. This entry does not cover unrelated DSv4 MTP dirty
work that may be present in the local worktree.

## Rule

Block-diffusion models need a separate canvas denoise/commit loop and a separate
backend model object. Do not bend the autoregressive Qwen executor into a
DiffusionGemma loader; route it as its own `DiffusionBlockModel` and keep chat
disabled until a verified prompt template exists.
