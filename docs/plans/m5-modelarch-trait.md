# M5 — Unified Model Architecture Contract

> Scope retreat from `backend-unification.md` §M5. The original target,
> "Unified `ModelForward` + shared Qwen3 forward path", is not the right next
> cut because `ModelForward` is CUDA-paged-KV shaped and `infer/src/model.rs`
> is only compiled under `feature = "cuda"`. This milestone instead lands the
> backend-neutral architecture contract that both schedulers and telemetry can
> share today.

## 0. Decision

M5 implements the P0 survey recommendation from
[`M5-P0-modelforward-survey.md`](M5-P0-modelforward-survey.md), with one
refinement from the actual codebase:

- Do **not** put the shared trait in `infer/src/model.rs`; Metal-only builds do
  not compile that module.
- Do **not** name the trait `ModelArch`; `infer/src/model_registry.rs` already
  owns `ModelArch` as the architecture enum.
- Add a new backend-neutral module, `infer/src/model_arch.rs`, with a trait
  named `ModelArchInfo` plus a serializable `ModelArchSummary`.

`ModelArchInfo` is the shared "what model shape is this?" contract. It does
not describe forward execution, KV page mutation, CUDA graph state, MLX lazy
arrays, or packed-varlen decode.

## 1. Contract Shape

`ModelArchInfo` should expose only architecture facts that are meaningful on
both backends:

```rust
pub trait ModelArchInfo {
    fn arch_kind(&self) -> crate::model_registry::ModelArch;
    fn hidden_size(&self) -> usize;
    fn vocab_size(&self) -> usize;
    fn num_hidden_layers(&self) -> usize;
    fn num_kv_layers(&self) -> usize;
    fn num_kv_heads(&self) -> usize;
    fn num_q_heads(&self) -> usize;
    fn head_dim(&self) -> usize;
    fn kv_cache_bytes_per_token(&self) -> usize;

    fn arch_summary(&self) -> ModelArchSummary { ... }
}
```

`ModelArchSummary` should be `Clone + Debug + serde::Serialize` and use stable
field names so `/v1/stats?format=json`, future bench traces, and backend tests
can read one shape:

```rust
pub struct ModelArchSummary {
    pub arch: crate::model_registry::ModelArch,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_kv_layers: usize,
    pub num_kv_heads: usize,
    pub num_q_heads: usize,
    pub head_dim: usize,
    pub kv_cache_bytes_per_token: usize,
}
```

If `ModelArch` lacks `Serialize`, implement serialization as its display name
inside `ModelArchSummary` rather than widening unrelated registry semantics.

## 2. Implementation Slices

### S1 — Add Neutral Module

Files:

- `infer/src/model_arch.rs`
- `infer/src/lib.rs`

Work:

- Add `pub mod model_arch;` as an always-available pure Rust module.
- Define `ModelArchInfo` and `ModelArchSummary`.
- Add pure unit tests for summary construction and Qwen3/Qwen3.5/DeepSeek-ish
  static formulas where possible without loading weights.

Commit:

- `feat(model): add backend-neutral architecture contract`

### S2 — Move CUDA `ModelForward` Shape Getters Behind `ModelArchInfo`

Files:

- `infer/src/model.rs`
- `infer/src/model/qwen3/forward.rs`
- `infer/src/model/qwen35/forward.rs`
- `infer/src/model/deepseek/forward.rs`

Work:

- Change `pub trait ModelForward: Send` to
  `pub trait ModelForward: crate::model_arch::ModelArchInfo + Send`.
- Remove the five shape getters from `ModelForward`; callers with
  `M: ModelForward` still see the methods through the super-trait.
- Add `impl ModelArchInfo for Qwen3Model`, `Qwen35Model`, and the DeepSeek
  scaffold.
- Keep formulas unchanged:
  - Qwen3: all hidden layers have K/V pages.
  - Qwen3.5: only full-attention layers count as KV layers.
  - DeepSeek scaffold: latent MLA width remains
    `kv_lora_rank + qk_rope_head_dim`.

Commit:

- `refactor(model): move CUDA architecture shape to ModelArchInfo`

### S3 — Add Metal Architecture Implementor

Files:

- `infer/src/backend/metal/config.rs`
- optional test split under `infer/src/backend/metal/` if existing tests are
  too crowded.

Work:

- Implement `ModelArchInfo` for `MetalModelConfig`.
- Map `MetalModelArch::Qwen3` to `model_registry::ModelArch::Qwen3`.
- Map `MetalModelArch::Qwen35(_)` to `Qwen35` for dense and
  `Qwen3_5_Moe` when the loaded config carries MoE metadata.
- Preserve Metal's full-attention-only KV layer accounting for Qwen3.5-family
  configs.

Commit:

- `feat(metal): expose model architecture summary`

### S4 — Wire One Shared Consumer

Files:

- `infer/src/server_engine/types.rs`
- `infer/src/metrics.rs`
- `infer/src/metrics/render.rs`
- `infer/src/backend/cuda/bootstrap.rs`
- `infer/src/backend/metal/runtime.rs`

Work:

- Add `model_arch: Option<ModelArchSummary>` to `EngineTelemetry`.
- Add `ServerMetrics::set_model_arch(ModelArchSummary)` and include the
  optional summary in `snapshot_engine_telemetry()`.
- CUDA bootstrap sets the summary from the loaded `ModelForward` before
  handing metrics to the scheduler.
- Metal runtime sets the summary from `MetalModelConfig` after load.
- JSON stats include `engine_model_arch`; text `/v1/stats` can stay compact
  unless adding the suffix is trivial and readable.

This is the proof that M5 is not just a trait extraction: both backends project
one architecture shape through the M1 telemetry surface.

Commit:

- `feat(metrics): publish backend-neutral model architecture`

### S5 — Verification + Wins Entry

Files:

- `docs/experience/wins/2026-05-07-m5-modelarch-trait.md`

Work:

- Record line delta, commands, and whether GuideLLM is runnable or pending
  behind M4.5.
- Do not publish perf numbers unless the canonical GuideLLM run drains cleanly.

Commit:

- `docs(wins): record M5 model architecture contract`

## 3. Out of Scope

- Making Metal implement CUDA `ModelForward`.
- Replacing `PagedKVPool` with a cross-backend `KVPoolHandle`.
- Deleting `backend/metal/forward.rs`.
- Sharing Qwen3 forward method bodies across CUDA and Metal.
- Touching attention/recurrent/KV op contracts.
- Changing Qwen3.5 C++ step-driver behavior.

These are deferred until there is a real unified KV abstraction that can model
both CUDA paged KV and Metal packed-varlen/left-padding without forcing either
backend into the wrong execution shape.

## 4. Acceptance

Local Linux gates:

```bash
cargo fmt --all --check
cargo check -p infer --no-default-features --features no-cuda
cargo check -p infer --no-default-features --features cuda,no-cuda
cargo check -p infer --no-default-features --features metal,no-cuda
cargo check -p infer --no-default-features --features cuda,metal,no-cuda
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo check -p infer --features cuda
cargo clippy -p infer --no-default-features --features no-cuda -- -D warnings
cargo clippy -p infer --no-default-features --features metal,no-cuda -- -D warnings
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo clippy -p infer --features cuda -- -D warnings
```

CUDA correctness gates:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
INFER_TEST_MODEL_PATH=infer/models/Qwen3-4B \
cargo test --release -p infer --features cuda --test greedy_consistency -- --test-threads=1

NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
INFER_TEST_MODEL_PATH=infer/models/Qwen3-4B \
cargo test --release -p infer --features cuda --test e2e -- --test-threads=1
```

Metal gates:

- Linux: `metal,no-cuda` check/clippy must pass.
- Apple Silicon remote: Metal scheduler smoke and Qwen3/Qwen3.5 generation
  remain `pending-remote` unless a Metal runner is available.

Bench gate:

- If M4.5 preemption lands first, run canonical
  `scripts/bench_guidellm.sh cuda-m5-modelarch ...` and include deltas.
- If M4.5 is not fixed yet, open the M5 wins entry with bench status
  `pending-M4.5`, referencing
  `docs/experience/errors/2026-05-07-m4-guidellm-canonical-stuck.md`.

Review:

```bash
bun run /home/ckl/.bun/bin/codex review --base <M5-base> -c sandbox.timeouts.exec_seconds=900
```

## 5. Risks

- **Name collision:** `model_registry::ModelArch` already exists. Use
  `ModelArchInfo` for the trait; do not create another public `ModelArch`
  symbol.
- **Cfg leak:** `model_arch.rs` must not import cudarc, cuda-kernels, MLX, or
  Metal types.
- **Formula drift:** Qwen3.5 and DeepSeek have non-standard KV accounting.
  Move existing formulas verbatim before adding new abstraction.
- **Telemetry churn:** `EngineTelemetry` consumers must tolerate
  `model_arch: None` for legacy/CPU mocks.
- **Scope creep:** Do not start `KVPoolHandle` or shared forward-body work in
  this milestone.

## 6. Rollback

Each slice is reversible:

- Revert S4 to remove the telemetry consumer while keeping the trait.
- Revert S3 if Metal config exposure breaks remote Metal.
- Revert S2 to put getters back on `ModelForward`.
- S1 is pure additive and can remain as dead scaffolding if later slices roll
  back.

No feature flag is planned because M5 should be behavior-neutral. If telemetry
shape churn causes downstream issues, gate only the `EngineTelemetry` field
population, not the trait.
