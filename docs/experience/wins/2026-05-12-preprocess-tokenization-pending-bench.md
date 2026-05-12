# Preprocess tokenization pipeline entry - 2026-05-12

## Goal

- Land the first CPU/GPU pipeline tranche: move request prompt tokenization
  into an explicit preprocess stage before scheduler enqueue, without claiming
  a throughput or latency win yet.

## Hypothesis

- Pre-tokenizing before `IncomingRequest` reaches the scheduler should remove
  tokenizer work from the scheduler admission path and make the next CPU stage
  overlap work easier to isolate. The correctness expectation for this tranche
  is unchanged generated behavior and unchanged scheduler fallback semantics
  for callers that do not provide prompt tokens.

## Command

Correctness and local feature checks:

```bash
rustfmt --check --edition 2024 --config skip_children=true \
  infer/src/server_engine/request_handle_engine.rs \
  infer/src/server_engine.rs \
  infer/src/http_server/handlers.rs \
  infer/src/http_server/types.rs \
  infer/src/http_server/router.rs

cargo test -p infer --no-default-features --features no-cuda \
  request_handle_engine_preprocesses_prompt_tokens_before_submit -- --nocapture

cargo test --release -p infer --no-default-features --features no-cuda

cargo clippy -p infer --no-default-features --features no-cuda -- -D warnings

CUDARC_CUDA_VERSION=13010 \
cargo check -p infer --no-default-features --features cuda,no-cuda

cargo test -p infer --no-default-features --features metal,no-cuda \
  metal_handle_forwards_inner_tokenizer_clone -- --nocapture

cargo test -p infer --no-default-features --features metal,no-cuda \
  pending_metal_request_uses_cached_prompt_tokens -- --nocapture

cargo clippy -p infer --no-default-features --features metal,no-cuda -- -D warnings
```

Deferred serving benchmark:

```bash
scripts/bench_guidellm.sh preprocess-tokenization-cuda \
  --model Qwen/Qwen3.5-4B \
  --processor infer/models/Qwen3.5-4B \
  --concurrencies 1,4,16 \
  --max-seconds 60
```

## Environment

- **Backend:** local compile/test on no-cuda; CUDA serving bench pending.
- **Model:** not loaded for local correctness checks.
- **Hardware:** Apple M4 Pro, macOS 26.3.1, Darwin 25.3.0 arm64.
- **Commit before change:** `b843bce1`.
- **Feature set:** `--no-default-features --features no-cuda` and
  `--no-default-features --features cuda,no-cuda`.
- **Non-default flags / env vars:** `CUDARC_CUDA_VERSION=13010` for
  CUDA-Rust typecheck without local `nvcc`.
- **Server launch:** pending CUDA benchmark.

## Params

| Param | Value |
|---|---|
| Change type | runtime control-plane / request preprocessing |
| Tokenizer ownership | HTTP stores `Arc<Tokenizer>` snapshot in `AppState` |
| HTTP preprocess executor | `tokio::task::spawn_blocking` |
| Scheduler fallback | preserved when prompt tokens are absent |
| Metal runtime | `MetalSchedulerHandle` forwards tokenizer and `PendingMetalRequest` consumes cached prompt tokens |
| Perf status | `pending-bench`, no performance conclusion claimed |

## Results

| Check | Result |
|---|---|
| targeted preprocess unit test | PASS |
| edited-file rustfmt check | PASS |
| no-cuda clippy `-D warnings` | PASS |
| `cuda,no-cuda` typecheck | PASS with unrelated warnings from existing untracked `infer/src/model/deepseek/load.rs` |
| Metal tokenizer-forwarding targeted test | PASS |
| Metal cached-token targeted test | PASS |
| Metal/no-cuda clippy `-D warnings` | PASS |
| full no-cuda release tests | FAIL due unrelated `metal_eval_audit` materialize-boundary classification drift |

Full no-cuda release test failure:

```text
metal_materialize_boundaries_stay_classified:
new Metal materialize boundary file needs docs/experience classification
left includes infer/src/backend/metal/kv_pool.rs, right does not
```

## Problems

- Performance is intentionally not reported in this entry. The change needs a
  CUDA GuideLLM run plus scheduler CPU timing counters before attributing any
  TTFT, ITL, or throughput delta.
- Repository state contains unrelated dirty/untracked DeepSeek/CUDA KV files.
  They were not modified for this tranche and affected broad formatting/check
  surfaces.

## Learnings

- The safe first split is to make `prompt_tokens` a prepared request field at
  the submission boundary while keeping scheduler-side tokenization as a
  compatibility fallback. That changes ownership without changing admission
  semantics.

## Delta vs baseline

- **Baseline:** first runtime-control-plane entry for this pipeline tranche.
- **Delta:** not measured; pending CUDA GuideLLM bench.

## Artefacts

- Raw GuideLLM artefacts: pending.
- Service trace: pending.

## Notes

- This entry is a bench stub per the runtime-change rule. It records the
  correctness gate and explicitly defers performance attribution.
