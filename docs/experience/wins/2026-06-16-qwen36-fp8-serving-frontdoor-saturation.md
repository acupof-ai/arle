# Qwen3.6 FP8 high-c serving front-door saturation isolated and fixed

## Goal

License-or-kill the hypothesis that the Qwen3.6 FP8 high-c guidellm crawl was a
serving/control-plane cap, not a quant kernel throughput verdict; then remove
the confirmed front-door cap before re-running aggregate throughput.

## Hypothesis

`/v1/stats` timeouts under c=64 are caused by the HTTP layer's global
`Mutex<ServeHandle>`: stats, metrics, and completions all contend on the same
mutex, while completions hold it across `ServeHandle::submit_streaming()` until
the engine thread assigns a request handle. If true, disabling stats polling
alone should not be enough; handle assignment should still serialize behind long
engine steps.

## Verification

Remote H20 GPU0, same FP8 weights and serve environment as the 2026-06-15 QAT gate,
`--num-slots 999 --total-pages 40 --page-size 16 --max-total-tokens 640
--max-prompt-tokens 640`, 512 input / 32 output.

Controlled no-stats run:

```bash
/root/dsv4-venv/bin/guidellm benchmark run \
  --target http://127.0.0.1:8123 \
  --model Qwen3.6-35B-A3B-FP8 \
  --processor /data01/models/Qwen3.6-35B-A3B-FP8 \
  --profile concurrent \
  --data prompt_tokens=512,prompt_tokens_stdev=1,prompt_tokens_min=512,prompt_tokens_max=512,output_tokens=32,output_tokens_stdev=1,output_tokens_min=32,output_tokens_max=32 \
  --max-seconds 120 \
  --output-dir bench-output/2026-06-16-qwen36-fp8-nostats-c32-64-512x32 \
  --backend openai_http \
  --backend-kwargs '{"validate_backend": "/v1/models", "request_format": "/v1/completions"}' \
  --disable-console-interactive \
  --outputs json --outputs csv \
  --rate 32,64 --warmup 5
```

Result: after >4 minutes, no `benchmarks.json`; GPU stayed 100%. After killing
guidellm, the server kept running submitted work and `/v1/stats` still timed out
at 2s and 10s. That killed the narrow "stats polling alone" explanation.

Env-gated trace run (`ARLE_SERVE_SUBMIT_TRACE=1`, c=64, 60s) confirmed the
front-door serialization:

```text
[serve-engine] admitted=1 active=0 waiting=1 pending=1
[serve-engine] step_ms=26345.9 active_before=0 waiting_before=1 pending_before=1 active_after=1 waiting_after=0 pending_after=1
[serve-engine] admitted=1 active=1 waiting=1 pending=2
[serve-submit] mode=streaming handle=1 wait_ms=26284.0 live=2
[serve-engine] step_ms=26355.2 active_before=1 waiting_before=1 pending_before=2 active_after=2 waiting_after=0 pending_after=2
[serve-engine] admitted=1 active=2 waiting=1 pending=3
[serve-submit] mode=streaming handle=2 wait_ms=26355.2 live=3
```

Interpretation: the first request enters immediately, then every later HTTP
handler waits on the global serve mutex while its submit call blocks on
`handle_rx.recv()`. The engine only drains one newly submitted request between
~26s steps, so c=64 degrades to one admission per step.

## Fix

Removed the global `Mutex<ServeHandle>` from `infer-server` HTTP state:

- `/v1/completions` and `/v1/chat/completions` call
  `ServeHandle::submit(_streaming)` directly; the channel send is concurrent and
  the wait for handle assignment no longer blocks every other HTTP handler.
- `/v1/stats` and `/metrics` call `ServeHandle::counters()` directly and no
  longer contend with request submission.
- Kept env-gated submit/engine trace logs under `ARLE_SERVE_SUBMIT_TRACE=1` for
  future high-c diagnosis; default behavior is unchanged.

## Post-fix Gate

Post-fix c=64 direct guidellm still did not produce a clean throughput result,
but it proved the front-door cap is removed:

```text
[serve-engine] admitted=1 active=0 waiting=1 pending=1
[serve-engine] step_ms=26349.8 active_before=0 waiting_before=1 pending_before=1 active_after=1 waiting_after=0 pending_after=1
[serve-engine] admitted=63 active=1 waiting=63 pending=64
```

`/v1/stats` returned while GPU was 100% busy:

```json
{"scheduler":{"active_requests":1,"queue_depth":63,"kv_free_pages":30528}}
```

So the control plane is no longer starved. The remaining c=64 failure is a new
engine/backend wall: after the 63-request admission, the next `engine.step()`
does not return within the 360s timeout. Source read explains why: default
`max_prefill_tokens=16384`, `prefill_max_requests=None`, and Qwen CUDA
multi-row plans execute each prefill row by calling `submit_prefill_row(row)` in
a loop. For 512-token prompts, the scheduler can put about 32 prefill rows in
one tick, and the current FP8 resident correctness path spends ~26s on a
single-row prefill. That makes the post-front-door c=64 sweep prefill-step-bound,
not HTTP-bound and not a valid quant aggregate throughput verdict.

## Verdict

Root-cause hypothesis refined:

- CONFIRMED: global `Mutex<ServeHandle>` serialized request admission and starved
  `/v1/stats`/`/metrics`.
- FIXED: submissions now drain as a batch and stats remains responsive under GPU
  load.
- NEXT WALL: Qwen FP8 high-c prefill is serialized inside the CUDA executor and
  scheduler permits too many prefill rows in one tick. The real c=1..64
  FP8-vs-BF16 aggregate sweep remains blocked until Qwen prefill is capped,
  pipelined, or batched with an adopt-first kernel path.

## Verification Commands

```bash
cargo check -p infer-server --release
cargo test -p infer-server --release
```

Both passed locally.

## Rule

When high-c serving crawls, first separate control-plane starvation from backend
step time. A responsive `/v1/stats` during GPU load proves the HTTP front door is
healthy; if stats is live but one `engine.step()` holds for minutes, the next
lever is scheduler/backend batching, not more HTTP mutex work.
