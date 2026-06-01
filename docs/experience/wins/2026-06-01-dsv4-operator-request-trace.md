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

Remote CUDA: pending in the same session before using this trace for DSv4
performance conclusions.

## Rule

DSv4 operator conclusions need request-level evidence. Raw source survey and
per-event log snippets are not enough; the local server log must expose a
single request-scoped operator breakdown before using it to drive roofline work.
