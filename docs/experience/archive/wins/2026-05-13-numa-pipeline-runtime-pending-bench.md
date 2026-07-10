# NUMA pipeline runtime substrate - 2026-05-13

## Goal

- Land P0-P2 NUMA-aware CPU/GPU pipeline architecture support without claiming
  a performance result.

## Hypothesis

- Binding CPU-side workers to the GPU-local NUMA domain before CUDA
  initialization, moving HTTP tokenization onto NUMA worker groups, and routing
  requests by NUMA cost plus queue load should make CPU/GPU overlap measurable
  on Linux CUDA hosts. Local validation should prove API/metrics correctness,
  not throughput.

## Command

Correctness and local feature checks:

```bash
rustfmt --edition 2024 \
  infer/src/runtime_topology.rs \
  infer/src/lib.rs \
  infer/src/metrics.rs \
  infer/src/metrics/render.rs \
  infer/src/http_server.rs \
  infer/src/http_server/types.rs \
  infer/src/http_server/router.rs \
  infer/src/http_server/handlers.rs \
  infer/src/http_server/preprocess.rs \
  infer/src/scheduler/types.rs \
  infer/src/request_handle.rs \
  infer/src/backend/cuda/bootstrap.rs \
  infer/src/scheduler/cuda/core/emit_worker.rs \
  infer/src/scheduler/cuda/core/construction.rs \
  infer/src/scheduler/cuda/runtime/admission.rs \
  infer/src/scheduler/cuda/request.rs \
  infer/src/scheduler/cuda/decode.rs \
  infer/src/server_engine/request_handle_engine.rs \
  crates/cuda-kernels/src/tensor.rs \
  infer/src/bin/metal_serve.rs \
  infer/src/backend/runtime.rs \
  infer/src/backend/metal/runtime.rs \
  infer/src/scheduler/tests.rs \
  infer/src/scheduler/cuda/runtime/tests.rs \
  infer/src/main.rs

cargo test -p infer --no-default-features --features no-cuda \
  runtime_topology -- --nocapture

cargo test -p infer --no-default-features --features no-cuda \
  runtime_topology::tests::configured_cuda_worker_ordinals_parse_single_and_multi -- --nocapture

cargo test -p infer --no-default-features --features no-cuda \
  server_metrics_ -- --nocapture

cargo test -p infer --no-default-features --features no-cuda \
  request_handle -- --nocapture

cargo test -p infer --no-default-features --features no-cuda \
  numa_router -- --nocapture

cargo test -p infer --no-default-features --features no-cuda \
  http_server -- --nocapture

cargo test -p infer --no-default-features --features no-cuda \
  request_handle_engine_preprocesses_prompt_tokens_before_submit -- --nocapture

cargo test -p infer --no-default-features --features no-cuda

CUDARC_CUDA_VERSION=13010 \
cargo check -p infer --no-default-features --features cuda,no-cuda

CUDARC_CUDA_VERSION=13010 \
cargo test -p cuda-kernels --no-default-features --features cuda,no-cuda \
  device_ordinal_override -- --nocapture

codex review --commit 83524c12
```

Deferred serving benchmark:

```bash
scripts/bench_guidellm.sh numa-pipeline-cuda \
  --model Qwen/Qwen3.5-4B \
  --processor infer/models/Qwen3.5-4B \
  --concurrencies 1,4,16 \
  --max-seconds 60
```

## Environment

- **Backend:** local compile/test on no-cuda plus CUDA Rust typecheck; CUDA
  serving bench pending.
- **Model:** not loaded for local correctness checks.
- **Hardware:** Apple Silicon/macOS local development host; Linux CUDA NUMA
  hardware pending.
- **Commit before change:** `74d88283`.
- **Commit before production multi-worker follow-up:** `559d367d`.
- **Commit before review follow-up:** `83524c12`.
- **Feature set:** `--no-default-features --features no-cuda` and
  `--no-default-features --features cuda,no-cuda`.
- **Non-default flags / env vars:** `CUDARC_CUDA_VERSION=13010` for CUDA-Rust
  typecheck without local CUDA runtime.
- **Server launch:** pending CUDA benchmark.

## Params

| Param | Value |
|---|---|
| Change type | runtime topology / HTTP preprocess / CUDA scheduler placement |
| Topology source | Linux sysfs/procfs with non-Linux fallback |
| GPU affinity | GPU PCI bus -> NUMA node -> local CPU set |
| NIC affinity | same-NUMA NICs, then CPU-intersection fallback |
| Tokenizer workers | ARLE-owned NUMA worker groups |
| CUDA workers | `INFER_CUDA_DEVICES=0,1,...` starts one scheduler worker per visible-device ordinal |
| Device selection | thread-local `DeviceContext::new()` override per worker; no global env mutation |
| Detokenizer worker | one CUDA emit worker per scheduler placement, NUMA-bound |
| Request routing | NUMA route cost + queue-load penalty + sticky-session migration |
| New telemetry | topology, affinity, worker groups, numastat, H2D latency, NUMA route/migration/rebalance |
| Perf status | `pending-remote`, no performance conclusion claimed |

## Results

| Check | Result |
|---|---|
| edited-file rustfmt | PASS |
| `runtime_topology` tests | PASS |
| `configured_cuda_worker_ordinals_parse_single_and_multi` | PASS |
| `server_metrics_` tests | PASS |
| `request_handle` tests | PASS |
| `numa_router` tests | PASS |
| `http_server` tests | PASS |
| `request_handle_engine_preprocesses_prompt_tokens_before_submit` | PASS |
| full no-cuda tests | FAIL due unrelated `metal_eval_audit` materialize-boundary classification drift after 582 passing lib tests |
| `cuda,no-cuda` typecheck | PASS with pre-existing DeepSeek reference dead-code warnings |
| `cuda-kernels device_ordinal_override` test | BLOCKED locally: macOS link lacks CUDA C/stub symbols under `cuda,no-cuda`; Rust typecheck passed |
| `codex review --commit 83524c12` | FOUND 2 P2; both fixed in follow-up (`auto_num_slots` ordinal, full error-chain display) |

## Problems

- No Linux CUDA NUMA host was used in this local run, so affinity application,
  sysfs GPU/NIC mapping, numastat locality, and H2D latency under load remain
  pending runtime evidence.
- The first implementation landed the NUMA router substrate while `main` still
  constructed a single router worker. Follow-up production bootstrap now expands
  `INFER_CUDA_DEVICES`, creates one CUDA scheduler worker per ordinal, binds the
  worker before context creation, and passes all workers into
  `NumaSchedulerRouter::new`.
- Full no-cuda test failure is unrelated to this change: `metal_eval_audit`
  reports `infer/src/backend/metal/kv_pool.rs` as an unclassified Metal
  materialize-boundary file. This file is outside the NUMA/CUDA/HTTP paths
  touched here.
- `cuda-kernels` unit test linking is not runnable on this macOS host because
  `no-cuda` skips CUDA C objects while the test binary still references CUDA
  FFI symbols; keep this as remote/Linux CUDA validation.
- This entry intentionally reports no TTFT, ITL, throughput, or GPU idle-time
  conclusion. Those require GuideLLM plus nsys on the target host.

## Learnings

- NUMA routing needs both locality cost and queue-load penalty. Pure locality
  would keep sticky sessions on an overloaded local worker and fail the dynamic
  rebalance requirement.
- The safe tokenizer split is an owned worker pool that receives cloned
  `Tokenizer` handles; using Tokio's global blocking pool cannot express NUMA
  placement.
- H2D latency should be recorded at concrete copy sites. The first landed
  sample point is host-pinned KV promotion back into the paged GPU pool.
- `INFER_CUDA_DEVICE` is still the single-worker compatibility path.
  `INFER_CUDA_DEVICES` is the multi-worker path and is resolved as CUDA
  visible-device ordinals, then mapped back to physical GPU/NUMA topology for
  placement logs and routing cost.
- Automatic slot sizing must use the same selected CUDA ordinal as worker
  bootstrap; otherwise `INFER_CUDA_DEVICES=1` can accidentally probe device 0.
- Worker bootstrap errors need alternate-chain formatting (`{err:#}`) because
  `anyhow::Context` displays only the top context under normal `{err}`.

## Delta vs baseline

- **Baseline:** [`2026-05-12-preprocess-tokenization-pending-bench.md`](2026-05-12-preprocess-tokenization-pending-bench.md).
- **Delta:** not measured; pending CUDA GuideLLM and nsys run.

## Artefacts

- Raw GuideLLM artefacts: pending.
- Service trace: pending.
- Topology log and `/v1/stats.runtime_topology`: pending remote run.

## Notes

- This is a bench stub per the runtime-change rule. It records the correctness
  gate and explicitly defers performance attribution.
