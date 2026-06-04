# `arle serve` in-process serving fix — restores the serve entry post-rewrite

## Context

The `infer/` rewrite (PR #53) deleted the standalone serve binaries
(`infer`, `metal_serve`, `cpu_serve`) — the workspace now builds only the
`arle` binary. But `crates/cli/src/serve.rs` still resolved a
`ServeBackend::binary_name()` ("infer"/"metal_serve"/"cpu_serve") and ran
`Command::new(binary).args(argv).status()`, spawning binaries that no longer
exist. So `arle serve` failed at exec for every backend. Serving is the
product, so this had to move in-process.

## What Worked

Added an in-process server-START (the piece that lived in the deleted
`#[tokio::main]` bins) and rewired the CLI to call it:

- **`crates/infer-api/src/serve.rs` (new)** — `serve_http(ServeHttpOptions)`:
  builds the backend router via `router_for_backend`, owns a tokio
  multi-thread runtime (the CLI is sync), `TcpListener::bind`,
  `axum::serve(...).with_graceful_shutdown(ctrl_c)`. Backend-absent build bails
  with the same message `--doctor` reports.
- **`crates/infer-api/src/loaded.rs`** — `router_for_backend(model_path,
  enable_cuda_graph, config)` mirrors `load_with_config`: Metal reuses
  `infer_server::metal_openai_router_from_model_path`; CUDA/CPU spawn the same
  `ServeHandle` the `load_cuda`/`load_cpu` builders spawn, then
  `infer_server::openai_router(serve, tokenizer, model_id)`. Feature-gated per
  backend, no cfg-leak.
- **`crates/cli/src/serve.rs` (rewrite)** — calls `infer_api::serve_http` for
  the COMPILED backend (`CompiledBackend::detect`). Removed `binary_name()`,
  `ServeInvocation`, the argv builder, and the `Command::new().status()` spawn.
  An explicit `--backend` that does not match the compiled backend is rejected
  up front (in-process can't satisfy a mismatch). Kept the bind warning +
  `--spec-type`/`--mtp-*` Metal-only validation.
- Dropped now-dead `ServeSpecTypeArg::as_backend_value` (was argv-only).

### Smoke evidence (in-process, no spawn)

CPU-built `arle serve --backend cpu --model-path models/Qwen3-0.6B --port 8077`:

```
[ARLE serve] starting cpu backend in-process on 127.0.0.1:8077
GET /v1/models -> {"object":"list","data":[{"id":"Qwen3-0.6B","object":"model","created":...,"owned_by":"arle"}]}
```

Bound + responded on the first poll, clean Ctrl-C shutdown, no child process.
Error paths verified: backend mismatch (`cuda` on a `cpu` binary) and missing
model both error clearly with exit 1.

### Builds / checks (all green)

- `cargo build --release --no-default-features --features metal,no-cuda,cli -p agent-infer --bin arle`
- `CUDARC_CUDA_VERSION=12060 cargo check -p infer-api -p cli --no-default-features --features cuda,no-cuda`
- `cargo build --release --no-default-features --features cpu,no-cuda,cli -p agent-infer --bin arle`
- `cargo clippy ... -p infer-api -p infer-server -p cli -- -D warnings` clean on
  metal + cpu (and on my own crates under cuda).
- 12 serve unit tests + 125/7/4/5 touched-crate tests green (single-threaded;
  one parallel-only flake is the pre-existing `modelscope` shared-env test).

## Not faithfully ported (follow-up)

The rewrite `openai_router` is a smaller surface than the old
`build_app_with_config`. These old serve args have no route in the new stack and
are now **rejected** (not silently ignored):

- `--train-control-url` — no `/v1/train/*` routes in the rewrite router.
- `--pool-model` — no engine-pool `/v1/models` metadata.
- `--spec-type` / `--mtp-draft-model` / `--mtp-draft-tokens` — carried + Metal-
  gated at the CLI, but the Metal router hardcodes its scheduler config and does
  not yet thread speculative routing through `metal_openai_router_from_model_path`.
- Flags after `--` (`extra_args`) — there is no standalone binary to forward to;
  rejected with a clear message.
- `--cuda-graph-max-bs` — top-level arg still parsed but no longer forwarded;
  `EngineLoadConfig` does not expose a decode-graph max-bs knob. `--no-cuda-graph`
  IS honored (flips `enable_cuda_graph` → CUDA decode-graph default).

## Rule

Post-rewrite the runtime ships ONE binary (`arle`); serving is in-process via
`infer_api::serve_http`, never a spawned backend binary. An explicit `--backend`
must equal the compiled backend. Old serve args whose routes the rewrite router
lacks are rejected, not silently dropped.

## Bench note

Correctness/wiring fix that restores a broken entry, not a perf change — no
guidellm sweep. Evidence is the in-process bind+`/v1/models` smoke above plus
the three-target build matrix. A full guidellm sweep on Qwen3.6 Metal is the
right follow-up once the serve path carries real traffic; not gated on this fix.
```
