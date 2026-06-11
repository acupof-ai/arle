# Metal serve unified layer and low-impact CLI

## Goal

Make the Metal serving path use the same service layer and scheduler layer as
the other backends, with only backend construction differing below `infer-api`.
Also add a conservative CLI mode for local Apple Silicon serving so the user can
start with bounded slots and smaller prefill chunks instead of the previous
hardcoded Metal router defaults.

## Hypothesis

The first visible smoothness gap was not steady-state disk writes. The code gap
was structural: Metal HTTP serving used a Metal-specific `infer-server` facade
with hardcoded scheduler/KV settings, and `ServeHandle::spawn_with_engine_builder`
signalled readiness before `Engine::warmup()` ran. Moving Metal construction to
`infer-api`, sharing the real host paged KV pool, and warming before ready should
remove the first-request warmup surprise while keeping the service layer
backend-neutral.

## Params

- Backend: Metal functional smoke, `mlx-community/Qwen3.5-0.8B-MLX-4bit`
  (small-model opt-out to avoid stressing the local Mac while validating CLI
  wiring; Qwen3.6/guidellm pending).
- CLI: `arle serve --backend metal --model-path mlx-community/Qwen3.5-0.8B-MLX-4bit --port 8127 --low-impact`
- `--low-impact` engine config: `num_slots=1`, `page_size=16`,
  `chunked_prefill_size=32`. `total_pages`, `max_prompt_tokens`, and
  `max_total_tokens` remain at the normal engine defaults unless explicitly
  set (supersedes the first low-impact cap sketch from this entry).
- Env: `INFER_METAL_WARMUP=1`, `INFER_METAL_PIPELINE` default ON.

## Env

- Host: local Apple Silicon Mac.
- Build: release, `--no-default-features --features metal,no-cuda,cli`.
- Date: 2026-06-11.

## Results

Structural changes:

- Added `infer_seam::HostPagedKvPool`, the production backend-neutral host page
  allocator.
- `infer_metal::MetalKvPool` and `infer_cuda::CudaKvPool` now alias the shared
  host allocator.
- `infer-server` no longer depends on `infer-metal`; it only owns
  `ServeHandle`, tokenizer, and the OpenAI v1 router.
- Metal serve construction moved into `infer-api` and now flows through the
  same `infer_server::openai_router` as CUDA/HIP/Vulkan/CPU.
- `spawn_with_engine_builder` now calls `engine.warmup()` before the ready
  signal, so backends with a warmup hook warm before the server binds.
- `arle serve` now exposes engine budget knobs plus `--low-impact`.

Verification:

| Check | Result |
|---|---|
| `cargo test -p infer-seam --release` | PASS, 8 tests |
| `cargo test -p infer-server --release` | PASS, 18 tests |
| `cargo test -p cli --release --no-default-features --features cpu,no-cuda serve::tests -- --nocapture` | PASS, 15 tests |
| `cargo test -p infer-api --release --no-default-features --features cpu,no-cuda` | PASS, 11 tests |
| `cargo check -p infer-api --release --no-default-features --features metal,no-cuda` | PASS |
| `cargo test -p infer-cuda --release --no-default-features --features no-cuda` | PASS, 53 tests |
| `cargo test -p infer-metal --release --no-default-features` | PASS, 2 tests |
| `CUDARC_CUDA_VERSION=12080 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib` | PASS with pre-existing `dsv4.rs` warnings |
| `cargo check -p cli --release --no-default-features --features metal,no-cuda` | PASS |
| `cargo test -p cli --release --no-default-features --features metal,no-cuda serve::tests -- --nocapture` | PASS, 15 tests |

Functional smoke:

```text
GET /v1/models -> Qwen3.5-0.8B-MLX-4bit
POST /v1/chat/completions, max_tokens=8 -> 200 OK
completion_tokens=8
```

Server log confirmed:

```text
[infer-metal] warmup (INFER_METAL_WARMUP) = true
[infer-metal] decode pipeline (INFER_METAL_PIPELINE) = true
[infer-metal] pipeline fast path LIVE (overlapped decode)
```

## Problems

- No Qwen3.6/guidellm run in this entry. The user reported system stalls, so
  this tranche avoided the 19 GB canonical Metal model during implementation.
  A SOLID smoothness verdict still needs a Qwen3.6 run with `vm_stat 1`,
  `iostat 1`, and request phase timing.
- The first CUDA/no-cuda check without `CUDARC_CUDA_VERSION` failed at the
  `cudarc` build script because this Mac has no `nvcc`; rerunning with the CI
  env `CUDARC_CUDA_VERSION=12080` passed.
- The worktree had unrelated CUDA/Vulkan diffs during validation; they were not
  part of this tranche. The `cuda,no-cuda` pass surfaced warnings from the
  unrelated `crates/infer-cuda/src/dsv4.rs` diff.

## Learnings

- Service-layer backend neutrality is enforceable by dependency direction:
  `infer-server` should receive a `ServeHandle`, not know how to construct a
  Metal executor.
- `EngineLoadConfig` is only useful if every backend router flows through the
  same builder path; a backend-specific router silently turns CLI knobs into
  no-ops.
- Warmup belongs before readiness for serving. Lazy warmup remains idempotent as
  a safety net, but the user-facing server should not bind before the backend
  warmup hook has completed.
