# DSv4 Decode NCCL Bottleneck

Date: 2026-05-14

Scope: true `/root/DeepSeek-V4-Flash` checkpoint, ARLE HTTP server, 8x H20,
`--deepseek-distributed-layers 43`, `--kv-cache-dtype fp8`, `--num-slots 1`,
`ARLE_DSV4_INCREMENTAL_KV=1`.

## Current Result

The current runnable layout is the legacy overlapping `TP=8, EP=8` layout. It
does complete real streaming inference on the true DSv4 weights, but decode is
communication dominated.

| Step | Layout | Communication per generated token per rank | 32-token NCCL kernel instances | Decode performance | Status |
|---|---:|---:|---:|---:|---|
| Baseline before trace | TP=8 / EP=8 | `43 layers * 2 = 86` all-reduces | `32 * 43 * 2 * 8 = 22016` | about 7-8 tok/s, decode step EMA about 105 ms | runnable |
| nsys trace | TP=8 / EP=8 | 86 all-reduces | 22016 observed | 32 tokens in 6.464 s, 4.95 tok/s e2e, decode step about 175-200 ms under profiler | confirmed bottleneck |
| Current code after trace/fanout/B>1 fallback | TP=8 / EP=8 | unchanged, 86 all-reduces | expected unchanged | 32 tokens: 7.50 tok/s e2e; 64 tokens: 7.64 tok/s e2e; decode EMA 117.6 ms | runnable, no comm reduction yet |
| EP-only experiment | TP=1 / EP=8 | target 43 all-reduces | target 11008 | failed warmup: O-LoRA output projection has `wo_a.cols=4096` but full-head local attention is 32768 wide | blocked by attention projection shape |
| Expert replication experiment | TP=8 / EP=1 | target 43 all-reduces | target 11008 | failed model load on H20 102 GB at layer 28 expert upload | memory-prohibitive |

Current benchmark evidence:

- Client output for 32 tokens: `alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau ups phi chi psi omega`
- Client output for 64 tokens continues with deterministic low-entropy text after the Greek sequence; content is not all repeated `2`.
- Content smoke is not single-token or repeated-token only:
  - Math case output starts with `17 + 25 = 42.` and explains the addition.
  - Writing case discusses GPU communication as a MoE decode latency bottleneck,
    though the 48-token cap cuts off mid-sentence.
- [`docs/trace-artifacts/2026-05-14-dsv4-decode/arle-dsv4-default-after.json`](../../trace-artifacts/2026-05-14-dsv4-decode/arle-dsv4-default-after.json):
  32-token run `4.2694s`, 64-token run `8.3799s`.
- Current metrics after 64-token run: `infer_scheduler_step_phase_decode_microseconds=117646`.

nsys evidence:

- [`docs/trace-artifacts/2026-05-14-dsv4-decode/README.md`](../../trace-artifacts/2026-05-14-dsv4-decode/README.md)
- [`docs/trace-artifacts/2026-05-14-dsv4-decode/arle-dsv4-decode-nsys.nsys-rep.gz`](../../trace-artifacts/2026-05-14-dsv4-decode/arle-dsv4-decode-nsys.nsys-rep.gz)
- [`docs/trace-artifacts/2026-05-14-dsv4-decode/arle-dsv4-decode-nsys.sqlite.gz`](../../trace-artifacts/2026-05-14-dsv4-decode/arle-dsv4-decode-nsys.sqlite.gz)
- [`docs/trace-artifacts/2026-05-14-dsv4-decode/arle-dsv4-decode-nsys-stats.txt`](../../trace-artifacts/2026-05-14-dsv4-decode/arle-dsv4-decode-nsys-stats.txt)
- [`docs/trace-artifacts/2026-05-14-dsv4-decode/arle-dsv4-decode-nsys-client.json`](../../trace-artifacts/2026-05-14-dsv4-decode/arle-dsv4-decode-nsys-client.json)

Top nsys signals:

- `ncclDevKernel_AllReduce_Sum_bf16_RING_LL`: 60.7% GPU time, 22016 kernel instances.
- `dsv4_fp8_gemv_batch_kernel`: 8.6% GPU time.
- `dsv4_hybrid_attention_kernel`: 6.4% GPU time.
- `dsv4_fp4_gemv_batch_kernel`: 4.5% GPU time.
- CUDA API overhead is also high: 522765 `cuMemAllocAsync` calls and 522765
  `cuMemFreeAsync` calls in the traced 32-token decode window.

## Why This Is Slow

The default layout currently creates two small BF16 all-reduces per layer per
decode token:

- Attention TP output reduce after the O projection.
- MoE EP routed-expert output reduce before shared expert addition.

For 43 layers this is 86 logical collectives per token per rank. nsys observed
exactly the expected cross-rank kernel count for a 32-token request:

```text
32 output tokens * 43 layers * 2 all-reduces/layer * 8 ranks = 22016 NCCL kernels
```

The decode loop is therefore latency-bound by thousands of small collectives,
not by the sampler. CPU is still visible because the current DSv4 path also
allocates and frees a large number of temporary CUDA buffers per token.

## Industry Direction

The industry path is not "more small all-reduces". It is:

- Use optimized EP token dispatch/combine all-to-all for MoE, not hidden-state
  all-reduce as the main sparse-expert exchange.
- Run grouped GEMM or fused MoE kernels over packed routed tokens.
- Use a low-latency decode communication backend and overlap dispatch/combine
  with GEMM.
- Keep decode memory stable with preallocated scratch and CUDA Graph compatible
  layouts.
- Use MTP/speculative decode for min-latency token throughput once the base
  single-token path is healthy.

Sources checked:

- [DeepEP](https://github.com/deepseek-ai/DeepEP) describes high-throughput and
  low-latency all-to-all kernels for MoE dispatch/combine, including FP8 support
  and a unified low-latency API.
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) provides FP8/FP4/BF16 GEMM,
  grouped GEMM, masked decode grouped GEMM, and Mega MoE that fuses and overlaps
  EP dispatch, FP8xFP4 MoE compute, and EP combine.
- [SGLang EP docs](https://docs.sglang.io/docs/advanced_features/expert_parallelism)
  select all-to-all backends such as DeepEP and runner backends such as DeepGEMM;
  they explicitly split communication and grouped GEMM concerns.
- [vLLM EP docs](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)
  recommend DeepEP low-latency kernels for decode, high-throughput kernels for
  prefill, dual batch overlap, and EPLB/redundant experts when memory allows.
- [TensorRT-LLM DeepSeek-R1 latency write-up](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog01_Pushing_Latency_Boundaries_Optimizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.html)
  reports TP/EP mixed layouts, MTP, CUDA Graph/PDL, custom small-message
  all-reduce, grouped GEMM, RouterGEMM, and fusion as the major latency
  optimizations.

## Optimization Process

1. Added request-level `request_trace` summaries so every HTTP request records
   TTFT, total latency, token throughput, KV/prefix-cache state, scheduler phase
   EMA, and scheduler pipeline counters.
2. Serialized distributed HTTP fanout submissions so rank queues stay ordered
   under concurrent client requests.
3. Added a DSv4 B>1 decode fallback that keeps multi-slot distributed fanout
   semantically alive by running the existing per-slot decode path sequentially.
   This is a correctness step, not a throughput optimization.
4. Added TP/EP axis override plumbing to run controlled layout experiments
   from HTTP launch env vars.
5. Collected nsys decode trace and confirmed NCCL all-reduce dominates GPU time.
6. Tested `TP=1/EP=8`; it avoids the attention all-reduce in theory, but the
   DSv4 O-LoRA projection is grouped, so the current kernels cannot consume the
   full 32768-wide attention output in a single TP=1 rank.
7. Tested `TP=8/EP=1`; it removes MoE EP all-reduce in theory, but replicating
   all experts OOMs on H20 before model load completes.

## Next Fixes

Priority order:

1. Replace the MoE hidden-state all-reduce path with a real DeepEP-style
   dispatch/combine backend and grouped GEMM runner. This is the main decode
   communication fix.
2. Add a DSv4 grouped O-LoRA output projection path so `TP=1/EP=8` can run as
   a valid EP-only profile instead of failing on `wo_a.cols=4096` versus
   32768-wide full-head attention.
3. Preallocate DSv4 per-layer decode scratch to eliminate the per-token
   allocation/free storm shown by nsys.
4. Replace the B>1 sequential fallback with vectorized packed-route decode.
5. Add MTP only after the base decode path has stable low-latency TP/EP
   communication and scratch reuse.
