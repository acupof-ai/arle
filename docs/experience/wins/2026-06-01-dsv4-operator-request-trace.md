# DSv4 Operator Trace In Request Logs

## Context

The DSv4 CUDA path already had `ARLE_DSV4_TRACE_LAYER=1` probes around many
attention and MoE phases, but the output was only a stream of per-layer log
lines. It was hard to answer "where did this request spend time" from a local
server log without running nsys and post-processing the whole trace.

## What Worked

Added a request-level DSv4 operator trace aggregate:

- `ARLE_DSV4_OPERATOR_TRACE=1` enables CUDA-synchronizing operator timing and
  writes `dsv4_operator_trace_process_delta` into `infer::request_trace` JSON.
- `ARLE_DSV4_TRACE_LAYER=1` keeps the legacy per-event log lines and also emits
  the new aggregate.
- `ARLE_DSV4_OPERATOR_TRACE_EVENTS=1` emits the legacy event lines when using
  the aggregate-only flag.
- The aggregate includes per-phase totals and per-layer totals with
  `calls`, `tokens`, `total_us`, and `avg_us`.
- `scripts/profile_dsv4_single_decode_nsys.sh` now exports the operator trace
  JSON and CSV beside the nsys summary when the flag is enabled.

This is diagnostic-only. The trace synchronizes the CUDA stream around each
instrumented phase and changes latency; it is not a performance mode.

## Verification

Local:

```bash
cargo fmt --check
cargo check -p infer --no-default-features --features no-cuda
CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,no-cuda
bash -n scripts/profile_dsv4_single_decode_nsys.sh
```

Remote CUDA:

```bash
ARTIFACT_ROOT=/data01/build/arle/docs/trace-artifacts/2026-06-01-dsv4-operator-request-trace-remote \
ARLE_DSV4_OPERATOR_TRACE=1 \
ARLE_DSV4_EXPERT_BACKEND=deepgemm-auto \
ARLE_DSV4_MOE_BACKEND=allreduce \
SERVER_BIN=/data01/build/arle/target-pod/release/infer \
./scripts/dsv4_toolchain.sh smoke \
  --model-path /data01/models/DeepSeek-V4-Flash \
  --max-tokens 32 \
  --port 18188 \
  --expert-backend deepgemm-auto \
  --moe-backend allreduce
```

Result:

- Build: `/data01/build/arle/target-pod/release/infer` from commit
  `bf9c51398d5b52d9dd072b712a2a65d1268def80`.
- Smoke response: 17 prompt tokens, 32 completion tokens, 3.596 s elapsed.
- `request_trace_count=1`.
- `dsv4_operator_trace_process_delta` present.
- `operators=38`, `layers=1308`.
- Service cleanup verified: no `target-pod/release/infer` process and no
  response on ports `18188-18194`.

Top request-level operators from the remote trace:

| phase | calls | tokens | total_us | avg_us |
|---|---:|---:|---:|---:|
| `ffn_total` | 11008 | 16512 | 13050504 | 1185.5 |
| `attn_total` | 11008 | 16512 | 12823300 | 1164.9 |
| `attn_core` | 11008 | 16512 | 11459333 | 1041.0 |
| `ffn_routed_local` | 11008 | 16512 | 5848363 | 531.3 |
| `attn_swa_all_reduce` | 512 | 768 | 4726152 | 9230.8 |
| `ffn_all_reduce` | 11008 | 16512 | 3743755 | 340.1 |
| `attn_hybrid_kernel` | 10496 | 15744 | 2279870 | 217.2 |
| `ffn_expert_loop` | 6256 | 11760 | 2033595 | 325.1 |

## Rule

DSv4 operator conclusions need request-level evidence. Raw source survey and
per-event log snippets are not enough; the local server log must expose a
single request-scoped operator breakdown before using it to drive roofline work.
