# Engine pool control plane — guidellm pending, 2026-05-02

> Status: `pending-remote`. This tranche adds control-plane scaffolding for
> multi-model serving metadata and LRU-safe loaded-engine ownership. It does
> not change backend decode kernels or share KV across models.

## Goal

- Regression guard: verify the OpenAI-compatible serving path still works while
  `/v1/models` can list configured pool models and the engine-pool controller
  refuses embedding/reranker stubs as text-generation routes.

## Hypothesis

- Throughput and latency should be unchanged for the primary loaded model
  because requests still route through the same single `RequestHandle`; the new
  pool metadata only affects control-plane model listing and future load/unload
  orchestration.

## Command

```bash
cargo test -p infer --release --no-default-features --features cpu,no-cuda pool_ -- --nocapture
cargo test -p cli --release serve_forwards_pool_model_specs -- --nocapture
cargo clippy -p infer --no-default-features --features cpu,no-cuda --lib -- -D warnings
cargo clippy -p cli -- -D warnings
```

Canonical guidellm command, pending:

```bash
scripts/bench_guidellm.sh engine-pool-control-plane \
  --model Qwen/Qwen3-4B \
  --processor models/Qwen3-4B
```

Invoked via: local control-plane tests only; canonical guidellm pending because
this workspace does not have a running full server bench for every backend.

## Environment

- **Backend:** control plane over cuda/metal/cpu serving binaries
- **Model:** primary text-generation model plus configured pool stubs
- **Hardware:** pending matched server bench host
- **Commit:** pending commit for P2-2 engine pool control plane
- **Feature set:** `cargo test --release --no-default-features --features cpu,no-cuda`
- **Non-default flags / env vars:** none for local tests
- **Server launch:** pending canonical guidellm run

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh <backend-label>`

## Results — sweep headline table

Canonical guidellm sweep: pending.

| rate (req/s) | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---:|---:|---:|---:|---:|---:|
| pending | pending | pending | pending | pending | pending | pending |

## Results — service-side KV / scheduler metrics

| metric | value |
|---|---:|
| peak active | pending |
| peak waiting | pending |
| peak prefill_queue | pending |
| peak kv_util | pending |
| `prefix_hit_rate` | pending |
| `prefix_skip_rate` | pending |
| `kv_fetch_q` | pending |
| `kv_fetch_waiters` | pending |
| `kv_store_q` | pending |
| `kv_store` | pending |
| `kv_bp` | pending |
| `tier_recall` | n/a |
| `tier_src` | n/a |
| `tier_promoted` | n/a |
| `tier_fallback` | n/a |

## Results — request accounting

| metric | value |
|---|---:|
| completed input tokens | pending |
| incomplete input tokens | pending |
| completed output tokens | pending |
| incomplete output tokens | pending |

## Problems

- Canonical throughput/latency numbers are pending. The implementation is
  intentionally control-plane only, so local verification focuses on metadata,
  type stubs, and LRU active-request protection.

## Learnings

- Multi-model support should start as a registry around the existing
  `LoadedInferenceEngine` contract. Embedding/reranker entries must be explicit
  stubs until their own runtime contracts exist.

## Delta vs baseline

- **Baseline:** latest primary-model guidellm snapshot for the selected backend.
- **Delta table:** pending canonical guidellm run.

| metric | baseline | now | delta |
|---|---:|---:|---:|
| TTFT p50 @ synchronous | pending | pending | pending |
| ITL p50 @ saturation | pending | pending | pending |
| out tok/s @ saturation | pending | pending | pending |

## Artefacts

- Raw: pending
- CSV: pending
- HTML: pending
- Service trace: pending

## Notes

- What changed in the code since baseline: added `EnginePool` metadata/LRU
  controller, `--pool-model` serve plumbing, and multi-model `/v1/models`
  listing for configured unloaded stubs.
- Suspected cause of any regression: none expected; requests still enter the
  same primary backend handle.
- Follow-ups: replace this pending entry with a completed guidellm snapshot
  after a primary-model server run with and without pool metadata configured.
