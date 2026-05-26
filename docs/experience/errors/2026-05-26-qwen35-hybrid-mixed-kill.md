# Qwen3.5 Hybrid Mixed KILL - mixed triggers, SLO gate still fails

## Context

Follow-up to [`docs/plans/2026-05-25-cuda-perf-codex-collab.md`](../../plans/2026-05-25-cuda-perf-codex-collab.md)
Axis 2 and Axis 3. L4 was unavailable, so the follow-up validation ran on V100
in `/home/chenkailun.c/agent-infer-v100-audit` with cached `Qwen3.5-4B`.

Goal: make Qwen3.5 hybrid support the same mixed decode+prefill contract as
Qwen3 dense, then validate whether `mixed_policy=Mixed` and larger chunked
prefill can be defaulted.

## Evidence

Remote build passed:

- V100-SXM2-32GB, CUDA 12.4, `TORCH_CUDA_ARCH_LIST=7.0`
- `cargo build --release -p infer --bin infer --features cuda`
- server flags shared unless stated: `--num-slots 16 --max-seq-len 5120`
- GuideLLM shape: 4096 input tokens, 256 output tokens, `max-seconds=60`,
  `warmup=5`

Local checks passed:

- `cargo test --release -p infer`: 595 passed, 14 ignored
- `cargo test --release`: 5 CLI tests passed
- `cargo test --release -p infer --test e2e`: 0 local GPU tests
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,no-cuda`

### Cancellation bug fixed during the run

Before the final sweeps, `/v1/completions` streaming cancellation was not
propagating to the scheduler: `trace_streaming_deltas()` kept draining the
scheduler receiver after the HTTP client disconnected, so
`delta_tx.is_closed()` never became true. This polluted c-sweeps because stale
requests kept running after GuideLLM closed a window.

After dropping the scheduler receiver immediately on client disconnect, service
trace showed active requests draining between windows and the c-sweeps became
valid. The unit test added for this bug verifies that the scheduler sender
observes closed after the client receiver is dropped.

### Axis 2: Qwen3.5 hybrid mixed

Raw artefacts:

- invalid cap=4: `/home/chenkailun.c/agent-infer-v100-audit/bench-output/2026-05-26-hybrid-mixed-cancel-v100-csweep-cap3-chunk2048`
- cap=3, chunk=2048: `/home/chenkailun.c/agent-infer-v100-audit/bench-output/2026-05-26-hybrid-mixed-cancel-prefillcap2-v100-csweep-cap3-chunk2048`
- cap=2, chunk=2048: `/home/chenkailun.c/agent-infer-v100-audit/bench-output/2026-05-26-hybrid-mixed-cancel-prefillcap2-v100-csweep-cap2-chunk2048`

Plan labels for the best valid variants:

```text
cap=3/chunk=2048: idle=20773, decode=2049, prefill=25, split=0, mixed=21
cap=2/chunk=2048: idle=23849, decode=2568, prefill=34, split=0, mixed=14
```

V100 c-sweep:

| config | c | TTFT p50 ms | ITL p50 ms | out tok/s | notes |
|---|---:|---:|---:|---:|---|
| cap=3, chunk=2048 | 1 | 4182.4 | 33.50 | 21.52 | valid |
| cap=3, chunk=2048 | 4 | 11264.0 | 178.91 | 14.16 | valid |
| cap=3, chunk=2048 | 8 | 11297.7 | 178.91 | 14.16 | valid |
| cap=3, chunk=2048 | 16 | 11387.5 | 166.43 | 10.74 | valid |
| cap=2, chunk=2048 | 1 | 4195.7 | 33.48 | 21.52 | valid |
| cap=2, chunk=2048 | 4 | 8924.7 | 163.68 | 8.55 | c4/c8 p95 TTFT >52s |
| cap=2, chunk=2048 | 8 | 8927.9 | 163.65 | 8.55 | c4/c8 p95 TTFT >52s |
| cap=2, chunk=2048 | 16 | 9760.2 | 170.10 | 10.52 | best c16 TTFT |

cap=4 is not viable on V100. It produced an invalid GuideLLM result and server
logs showed:

```text
qwen35 prefill_forward_paged_batch requests=3 total_tokens=6145
caused by: Alloc failed: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")
```

Against the original L4 acceptance baseline supplied in the plan
(`c16 TTFT=12913 ms, ITL=71.85 ms, out tok/s=164`), the V100 c16 TTFT can be
made lower, but ITL and output throughput regress far beyond the gate. The GPU
is different, so this is not a clean baseline comparison; it is still enough to
reject defaulting Qwen3.5 hybrid mixed from this evidence.

### Axis 3: chunk=4096 follow-up

Raw artefacts:

- `/home/chenkailun.c/agent-infer-v100-audit/bench-output/2026-05-26-hybrid-mixed-cancel-prefillcap2-v100-c16-cap2-chunk4096`

The 4096 chunk probe did not improve c16:

| config | c | TTFT p50 ms | ITL p50 ms | out tok/s |
|---|---:|---:|---:|---:|
| cap=2, chunk=2048 | 16 | 9760.2 | 170.10 | 10.52 |
| cap=2, chunk=4096 | 16 | 9775.3 | 173.13 | 8.27 |

chunk=4096 also widened c16 TTFT p95 to 53079.9 ms. Do not use it as the
hybrid default.

## Root Cause

The implementation reached a real mixed path: `split=0` and `mixed>0`, with
decode rows staying on Qwen3.5 decode kernels and prefill rows using paged
hybrid prefill. That fixed the earlier false-negative where mixed silently
fell back or used the wrong row semantics.

The remaining failure is performance, not planner reachability:

- Qwen3.5 hybrid prefill is too expensive to colocate with decode in a 60s
  high-concurrency GuideLLM window on V100.
- cap=4 allows 3-row prefill batches and OOMs on V100.
- cap=3 keeps throughput slightly better but c16 TTFT remains ~11.4s and only a
  small number of requests complete.
- cap=2 improves c16 TTFT to ~9.8s but c4/c8 throughput and tail TTFT regress.
- chunk=4096 moves cost into larger prefill steps and worsens throughput/tail.

## Fix

Do not land Qwen3.5 hybrid mixed as the default based on this run. The useful
pieces can be kept as follow-up candidates, but not as a perf win:

- keep the HTTP streaming cancellation fix;
- keep `max_concurrent_prefill_requests <= 2` as the V100 safety bound if
  hybrid mixed remains enabled for experiments;
- keep `chunked_prefill_size=2048` for this workload;
- require a kernel-level hybrid prefill/decode optimization or SLO-aware
  admission policy before revisiting default Mixed for Qwen3.5 hybrid.

## Rule

`plan_label=mixed` is only reachability evidence. It is not a license to land.
The c-sweep must have valid completions and must pass TTFT, ITL, and throughput
gates. Client cancellation must also propagate to the scheduler before using
GuideLLM c-sweeps as evidence, otherwise stale requests contaminate later
concurrency windows.
