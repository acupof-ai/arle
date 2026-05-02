# A5 TensorRT-LLM W3 skipped - `session_id` rejected

## Context

A5 requires the same canonical W3 trace across the four-engine panel:
`scripts/bench_agent_trace.py --workload agent-w3-short-multiturn`. The trace
always sends `session_id` on OpenAI-compatible `POST /v1/chat/completions`
requests, per `docs/plans/2026-05-02-agent-load-bench-spec.md` §2 and §3.1.

This run installed pinned TensorRT-LLM `1.2.1` locally and successfully built a
TensorRT backend engine from the local Qwen3-4B HF checkpoint on the NVIDIA L4.
The W3 replay itself is invalid because TensorRT-LLM rejects the required
`session_id` request field.

## Goal

- Baseline TensorRT-LLM on canonical W3 for the A5 four-engine panel, or record the supported fallback if canonical replay fails.

## Hypothesis

- TensorRT-LLM `trtllm-serve serve` should either run the canonical W3 trace through an OpenAI-compatible endpoint or fail clearly on unsupported session metadata.

## Command

Install:

```bash
uv venv --clear /tmp/arle-trtllm-1.2.1
uv pip install --python /tmp/arle-trtllm-1.2.1/bin/python tensorrt-llm==1.2.1
uv pip install --python /tmp/arle-trtllm-1.2.1/bin/python nvidia-cublas==13.1.0.3
```

Server:

```bash
LD_LIBRARY_PATH=/tmp/arle-trtllm-1.2.1/lib/python3.12/site-packages/tensorrt_libs:/tmp/arle-trtllm-1.2.1/lib/python3.12/site-packages/nvidia/cu13/lib:/tmp/arle-trtllm-1.2.1/lib/python3.12/site-packages/nvidia/cublas/lib:/tmp/arle-trtllm-1.2.1/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/tmp/arle-trtllm-1.2.1/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/tmp/arle-trtllm-1.2.1/lib/python3.12/site-packages/nvidia/nvjitlink/lib:/usr/lib64-nvidia:/usr/local/cuda/lib64 \
CUDA_HOME=/usr/local/cuda \
CUDA_VISIBLE_DEVICES=0 \
/tmp/arle-trtllm-1.2.1/bin/trtllm-serve serve /content/workspace/agent-infer/models/Qwen3-4B \
  --backend tensorrt \
  --host 127.0.0.1 \
  --port 30200 \
  --tokenizer /content/workspace/agent-infer/models/Qwen3-4B \
  --max_batch_size 16 \
  --max_seq_len 4096 \
  --free_gpu_memory_fraction 0.85 \
  --enable_chunked_prefill \
  --log_level info
```

Client:

```bash
PYTHONUNBUFFERED=1 python3 scripts/bench_agent_trace.py \
  --workload agent-w3-short-multiturn \
  --server http://127.0.0.1:30200 \
  --label trtllm-w3-h1 \
  --out bench-output/2026-05-02-agent-load-trtllm-w3-h1/results.json \
  --trace-out bench-output/2026-05-02-agent-load-trtllm-w3-h1/trace.jsonl \
  --no-probe-stats
```

## Environment

- **Workload:** `agent-w3-short-multiturn`.
- **Backend / engine:** TensorRT-LLM `1.2.1`, `trtllm-serve serve --backend tensorrt`.
- **Model:** Qwen3-4B, `/content/workspace/agent-infer/models/Qwen3-4B`.
- **Tokenizer / processor:** same local model path.
- **Hardware:** NVIDIA L4, 23,034 MiB VRAM, driver 580.82.07.
- **Commit:** ARLE `c5011cb5`, clean tree before this docs entry.
- **Runtime:** `torch 2.9.1+cu128`, `transformers 4.57.3`, `tensorrt 10.14.1.48.post1`.
- **KV / scheduler:** paged KV cache, `tokens_per_block=32`, `maxNumSequences=16`, `maxBatchSize=16`, capacity scheduler `GUARANTEED_NO_EVICT`.
- **Session / prefix flags:** no accepted `session_id`; no discovered prompt/session cache flag in `trtllm-serve serve --help`.

## Results

TensorRT-LLM setup succeeded:

| setup signal | value |
|---|---:|
| TensorRT engine build time | 47 s |
| engine generation time | 38.9205 s |
| engine serialization time | 20 s |
| loaded engine size | 7707 MiB |
| KV cache allocation | 10.97 GiB |
| KV capacity | 79,904 tokens |
| `/v1/models` | 200 OK |
| smoke request with `model=default` | 200 OK |

Canonical W3 replay failed:

| metric | value |
|---|---:|
| attempted turns | 128 |
| successful turns | 0 |
| failed turns | 128 |
| scored turns attempted | 64 cold |
| scored turns successful | 0 |
| output tok/s | n/a |
| warm TTFT p99 | n/a |
| cold TTFT p99 | n/a |

Representative error:

```text
HTTP 400: {"object":"error","message":"[{'type': 'extra_forbidden', 'loc': ('body', 'session_id'), 'msg': 'Extra inputs are not permitted', 'input': 'w3-warm-000'}]","type":"BadRequestError","param":null,"code"
```

Because the first request in each warm session is rejected, later warm scored
turns are never sent by the sequential per-session trace runner.

## Root Cause

TensorRT-LLM `trtllm-serve serve` validates OpenAI chat request bodies with
extra fields forbidden. The A5 W3 contract requires `session_id` on every
request so server-side session-affinity behavior can be compared. Removing
`session_id` would create a non-canonical trace and hide exactly the protocol
gap A5 is supposed to surface.

## Fix

No code fix was made. Per plan §5, this engine row is documented as unsupported
for canonical W3 trace replay until TensorRT-LLM accepts an ignored or first
class `session_id` field, or the benchmark spec explicitly defines a fallback
request shape that preserves the session-affinity caveat.

## Rule

Do not patch the W3 client to satisfy a competitor server. If an engine rejects
the required OpenAI-compatible request shape, record the protocol failure as the
baseline-panel row and do not claim W3 session-affinity or warm-turn wins for
that engine.

## Artefacts

- Raw result: `bench-output/2026-05-02-agent-load-trtllm-w3-h1/results.json`
- Trace: `bench-output/2026-05-02-agent-load-trtllm-w3-h1/trace.jsonl`
- Client log: `bench-output/2026-05-02-agent-load-trtllm-w3-h1/client.log`
- Server log: `bench-output/2026-05-02-agent-load-trtllm-w3-h1/trtllm_tensorrt_server.log`
- Metrics before: `bench-output/2026-05-02-agent-load-trtllm-w3-h1/metrics_before.prom`
- Metrics after: `bench-output/2026-05-02-agent-load-trtllm-w3-h1/metrics_after.prom`
- TensorRT timing cache: `bench-output/2026-05-02-agent-load-trtllm-w3-h1/model.cache`
- Key sha256:
  - `results.json`: `90b08afec9053f59c38406b228fbb5c7ce31417140e01824c2487b879e16a233`
  - `trace.jsonl`: `97d90c9a251b736b8e8fe2db924e5fa24931b88a73ec6c90ecab7fe8ba2363ee`
  - `client.log`: `1ee08ccb74b57a15cb95a2eff76690fe49eb86f65042e6f12e7829007c0c96cb`
  - `trtllm_tensorrt_server.log`: `806c40de7ff7e54dc07bafb61fc327f39c0d53d91fb17223d85f91cad614c3dd`
  - `metrics_after.prom`: `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`
  - `model.cache`: `bde34cc814f44026b600c9928e1e27b08ffb863871a10b5a993c3382774466ef`
