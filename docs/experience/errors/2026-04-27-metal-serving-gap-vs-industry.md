# Metal serving gap versus Apple-native industry baselines

## Context

On 2026-04-27 we compared the current ARLE Metal path against the
Apple-native inference direction represented by oMLX, vLLM Metal, and
vLLM-MLX public benchmarks.

The immediate Qwen3.6 MoE bug was fixed separately: the Qwen3.6 / Qwen3.5-MoE
Metal builder now wires MoE MLP weights into the C++ Qwen3.5 compiled model,
and the local step-driver smoke reports `cpp_batch_prefill` instead of
`rust_scalar_prefill`. That fix removes one wrong fallback, but it does not
close the broader serving gap.

Current local evidence:

- Qwen3.5-0.8B GGUF Q4_K_M on M4 Pro reached 211.7 tok/s decode for
  512 prompt / 1024 decode after Q5_K/Q8_0 affine repack and Q6/group16 qmv
  tuning.
- Qwen3.6-35B-A3B short diagnostics load and run locally around 63-67 tok/s
  decode, but the short DFlash run is only a load/execute diagnostic.
- Qwen3.6 MoE C++ prefill routing has a smoke result, not a completed
  canonical `guidellm` sweep.

External calibration snapshot:

- oMLX reports Qwen3.5-0.8B on M4 Pro (16 GPU cores) at 305.8 tok/s TG for
  1k context, 256.3 tok/s at 8k, and 211.9 tok/s at 16k.
- oMLX reports Qwen3.5-35B-A3B on M4 Pro (16 GPU cores) at 84.5 tok/s TG for
  1k context and 78.9 tok/s for 4k.
- vLLM Metal v0.2.0 made a unified paged varlen Metal kernel the default
  attention backend and claims 83x TTFT plus 3.6x throughput over v0.1.0.
- vLLM-MLX reports continuous batching scaling to 4.3x aggregate throughput at
  16 concurrent requests on Apple Silicon.

## Root Cause

The gap is mostly a runtime-serving gap, not a single-model load bug:

1. ARLE Metal has a live scheduler, but the efficient batched decode path is
   still narrow. Qwen3 and Qwen3.5 same-length decode batching exists, while
   variable-length decode batching has not entered the high-efficiency GPU
   path.
2. Qwen3.5 still pays per-step concat/split around request-local KV and
   recurrent state. That overhead prevents the HTTP concurrency sweep from
   showing the throughput step that same-length batching should otherwise
   unlock.
3. Qwen3.6 / MoE has only just been moved back onto the C++ prefill route.
   The next proof must be a long-context / canonical serving sweep, not a
   short routing smoke.
4. DFlash is not yet integrated into the live scheduler runtime and must be
   judged only on long-context or ultra-long sequence workloads. Short
   32-token diagnostics are not optimization evidence.
5. Metal KV cache remains model-native dtype (`bf16` / `f16` in practice).
   CUDA has quantized-KV paths; Metal does not. This is not the first
   bottleneck, but it limits long-context and multi-request headroom later.
6. Observability is still too coarse for full serving work. Queue, latency,
   and MLX memory gauges exist; the next layer needs reliable prefix hit,
   active/peak KV use, batch shape, varlen fallback, and DFlash acceptance
   counters.

## Fix

Optimization should be ordered as serving closure, then model-specific kernel
work:

1. Make variable-length decode batching the main Metal scheduler path.
2. Remove the Qwen3.5 per-step concat/split state churn or replace it with
   runtime-owned batched state ownership.
3. Run a completed Qwen3.6-35B-A3B `guidellm` sweep after the C++ MoE prefill
   routing fix, using long-context parameters before claiming improvement.
4. Move DFlash evaluation to long-context / ultra-long sequence only, and do
   not optimize from short prompt smoke runs.
5. Expand Metal serving metrics around prefix reuse, batch shapes, KV
   utilization, fallback reason, and DFlash acceptance.
6. Treat Metal KV quantization as a later capacity lever after batching and
   prefix/KV lifecycle are stable.

## Rule

Do not describe Metal as industry-comparable based on single-request decode
alone. For Apple Silicon serving comparisons, require:

- a single-request direct benchmark,
- a canonical `guidellm` serving sweep,
- concurrency scaling evidence,
- long-context evidence for Qwen3.6 / MoE and DFlash,
- and explicit fallback counters showing that the request stayed on the
  intended compiled/batched path.

If only the single-request path improved, say exactly that. The Metal roadmap
priority remains scheduler-first serving, not accumulating isolated direct
bench wins.

## Cross-references

- Roadmap: [`projects/mlx-backend-roadmap.md`](../../projects/mlx-backend-roadmap.md)
- Local Qwen3.5 GGUF result:
  [`wins/2026-04-27-bench-metal-qwen35-0p8b-gguf-q5-q8-q6qmv.md`](../wins/2026-04-27-bench-metal-qwen35-0p8b-gguf-q5-q8-q6qmv.md)
- Qwen3.6 MoE prefill routing smoke:
  [`wins/2026-04-27-bench-guidellm-metal-qwen36-moe-cpp-prefill-pending.md`](../wins/2026-04-27-bench-guidellm-metal-qwen36-moe-cpp-prefill-pending.md)
- DFlash scope correction:
  [`errors/2026-04-27-dflash-long-sequence-only.md`](2026-04-27-dflash-long-sequence-only.md)
- oMLX Qwen3.5-0.8B: <https://omlx.ai/benchmarks/ti2qtqq6>
- oMLX Qwen3.5-35B-A3B: <https://omlx.ai/benchmarks/uvqv3cco>
- vLLM Metal: <https://raw.githubusercontent.com/vllm-project/vllm-metal/main/README.md>
- vLLM-MLX paper: <https://arxiv.org/abs/2601.19139>
