# 2026-04-15 · Metal Ecosystem Review And Route Correction

## Why This Exists

The Metal plan had started to drift toward single-request tuning as the main
thread. That is no longer the right north star.

As of April 2026, the external Apple Silicon bar is now split into two clear
tracks:

- `mlx-lm`: strong library / CLI ergonomics, prompt caching, rotating KV cache,
  and wired-memory guidance
- `vllm-metal` and Docker Model Runner: scheduler-first serving with the same
  OpenAI-compatible API shape developers already use on Linux

`agent-infer` should compete on the second track for serving. Direct
single-request speed still matters, but it is no longer the roadmap driver.

## External Calibration

### 1. The serving baseline is now scheduler-first, not REPL-first

`vllm-project/vllm-metal` now positions Apple Silicon around the existing vLLM
stack rather than around a standalone single-request runner:

- README headline: "High-performance LLM inference on Apple Silicon using MLX
  and vLLM"
- 2026-04 release note: unified paged varlen Metal attention is the default,
  claiming `83x TTFT` and `3.6x throughput` over `v0.1.0`
- install path is already one-command via `install.sh`

Docker made the same point on 2026-02-26:

- `vllm-metal` plugs into the existing vLLM engine, scheduler, and OpenAI API
- Docker Model Runner exposes the same OpenAI-compatible and
  Anthropic-compatible APIs on macOS as on Linux/WSL
- one-command install is part of the product story

Implication:

- Apple serving is no longer judged as "can it run?".
- It is judged as "does it expose modern scheduler semantics and standard API
  compatibility on Apple hardware?".

### 2. The research bar also centers on continuous batching and reuse

The 2026-01-27 paper `Native LLM and MLLM Inference at Scale on Apple Silicon`
states:

- `21%` to `87%` higher throughput than `llama.cpp` on text models
- continuous batching scaling to `4.3x` aggregate throughput at `16`
  concurrent requests
- up to `525 tokens/s` on Apple M4 Max text workloads

Implication:

- The academic / open-source bar is not "slightly faster single-user MLX
  generate".
- The bar is continuous batching plus prefix-aware reuse on consumer Apple
  hardware.

### 3. `mlx-lm` is still the direct single-request and cache-management baseline

The official `mlx-lm` README is still valuable as the closest Apple-native
baseline for direct execution and memory behavior:

- `pip install mlx-lm`
- rotating fixed-size KV cache
- prompt caching
- wired-memory guidance for large models on macOS 15+

But its public framing remains library / CLI first, not server-first.
Historically, even the built-in server surface had rough edges; issue `#233`
opened on 2025-06-13 reports `mlx_lm.server` returning `404` on a valid
`/v1/chat/completions` request.

Implication:

- For direct `metal_bench`, `mlx-lm` remains the right local engine reference.
- For `metal_serve`, the closer product benchmark is `vllm-metal` /
  Docker Model Runner, not `mlx_lm.server`.

### 4. API compatibility expectations have moved up

Ollama's OpenAI-compatibility docs now document:

- `/v1/chat/completions`
- `/v1/models`
- `/v1/embeddings`
- `/v1/responses`
- streaming, tools, and JSON / `response_format` support on the chat path
- `/v1/responses` support added in `v0.13.3`

Implication:

- `chat/completions` compatibility alone is no longer enough to count as a
  polished local serving surface.
- `responses`, streaming parity, and structured output all belong on the real
  product path, not the nice-to-have path.

## Local Reality Check

These local numbers are directional sanity checks from 2026-04-15, not a
clean commit-isolated perf snapshot. The detached-worktree benchmark path is
currently blocked by unrelated in-flight tiered-KV serde changes, which is its
own benchmark-hygiene problem.

### Direct Metal path is no longer the main problem

Local sanity checks on 2026-04-15 on `Apple M4 Pro / macOS 26.3.1(a)` show the
direct path is already respectable:

### `metal_bench` sanity

`mlx-community/Qwen3.5-4B-MLX-4bit`, `128 / 128`, `warmup=1`, `runs=2`

| Metric | Value |
| --- | ---: |
| prompt TPS | `732.9` |
| generation TPS | `80.2` |
| repo E2E TPS | `72.3` |
| TTFT | `174.7 ms` |

`mlx-community/Qwen3-4B-bf16`, `20 / 256`, `warmup=1`, `runs=3`

| Variant | Prompt TPS | Gen TPS | Repo E2E TPS | TTFT |
| --- | ---: | ---: | ---: | ---: |
| baseline | `247.5` | `26.4` | `26.2` | `81.0 ms` |
| DFlash | `243.6` | `154.5` | `147.2` | `82.1 ms` |

Those numbers are not the bottleneck story anymore.

### `metal_serve` still behaves like a queue, not a scheduler

The interrupted 2026-04-15 HTTP throughput sweep on `Qwen3.5-4B-MLX-4bit`
already showed the problem clearly enough:

| In / Out | C | Aggregate TPS | TTFT p50 |
| --- | ---: | ---: | ---: |
| `512 / 256` | `1` | `66.6` | `586 ms` |
| `512 / 256` | `2` | `62.5` | `4540 ms` |
| `512 / 256` | `4` | `63.1` | `8669 ms` |
| `512 / 256` | `8` | `61.9` | `16416 ms` |
| `512 / 256` | `16` | `63.8` | `32645 ms` |

Interpretation:

- concurrency does not raise total throughput materially
- TTFT grows roughly with queue depth
- the current server is still request-level FIFO serialization with streaming
  on top

That is exactly what the current code says: `metal_serve` is still built around
`BackendRuntimeHandle`, and the existing `MetalScheduler` is not yet the live
execution path.

## Route Correction

### Correct north star

The roadmap should now optimize for this target:

> Production-grade Apple Silicon serving with scheduler-owned request
> lifecycle, prefix-aware reuse, standard OpenAI-compatible API shape, and a
> one-command local install path.

Not this target:

> Keep improving single-request MLX execution until the server becomes
> competitive by accumulation.

That accumulation path is wrong because the current server bottleneck is
architectural, not kernel-local.

### Correct priority order

### P0 · Serving architecture

1. `M0.2` live Metal scheduler
2. `M0.3` live prefix cache + KV pool
3. `M0.4` queue / reuse / memory observability

### P1 · Product surface

4. finish `/v1/responses` streaming parity
5. add structured outputs / constrained decoding
6. add a one-command Apple Silicon install / run path

### P2 · Engine breadth

7. generalize DFlash beyond `Qwen3`
8. expand deliberate model coverage

### Background track, not main track

9. continue direct `metal_bench` tuning only when:
   - it fixes correctness or memory safety, or
   - it produces a measured direct-path win without adding more serving debt

## Operational Corrections

### 1. Stop accepting direct `metal_bench` wins as serving progress

`metal_bench` should remain a required sanity check, but not a milestone exit
for serving milestones.

Serving milestones should require:

- clean `metal_serve` build
- scheduler-backed request lifecycle
- HTTP sweep evidence under concurrency

### 2. Make HTTP sweep a hard gate after `M0.2`

After the scheduler lands, acceptance should require a quick or full
`scripts/bench_throughput_sweep.py` run against `metal_serve`.

Success condition is not "some requests complete". It is:

- aggregate throughput rises materially at `C >= 4`
- TTFT no longer scales roughly linearly with queue depth
- repeated-prefix requests become measurable once `M0.3` lands

### 3. Treat install DX as a real competitive feature

External Apple-native competitors now ship:

- `pip install mlx-lm`
- `curl ... | bash` for `vllm-metal`
- `docker model install-runner --backend vllm`

That means Cargo-feature knowledge is now internal complexity, not acceptable
first-run UX.

## Sources

- `vllm-project/vllm-metal` README and release notes:
  https://github.com/vllm-project/vllm-metal
- Docker blog, `Docker Model Runner Brings vLLM to macOS with Apple Silicon`,
  posted `2026-02-26`:
  https://www.docker.com/blog/docker-model-runner-vllm-metal-macos/
- arXiv `2601.19139`, `Native LLM and MLLM Inference at Scale on Apple Silicon`:
  https://arxiv.org/abs/2601.19139
- `mlx-lm` README:
  https://github.com/ml-explore/mlx-lm
- `mlx_lm.server` issue `#233`, opened `2025-06-13`:
  https://github.com/ml-explore/mlx-lm/issues/233
- Ollama OpenAI compatibility docs:
  https://docs.ollama.com/api/openai-compatibility
