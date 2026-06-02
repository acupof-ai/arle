# DSv4 Decode Body Graph Start-Position ABI

## Context

Target workload remains DSv4-Flash TP8 + EAGLE with CUDA graph on the single
8-GPU pod, hot GPU cache, 256K/1500:

| Metric | Target |
| --- | ---: |
| TTFT | ~0.44 s |
| TPOT | ~4.85 ms |
| E2E | ~7.7 s |
| Output throughput | ~196 tok/s |

Prior SGLang-path research isolated CUDA graph coverage as the dominant
structural gap. DSv4 already had CUDA graph pieces around decode input/head,
but the 43-layer body still baked host-side decode positions into several
attention-side kernels. That prevents safe replay across token positions.

## What Worked

This tranche adds the graph-safe ABI needed before DSv4 body capture can be
validated:

- DSv4 body graph cache in `DeepseekBatchDecodeBuffers`, keyed by exact batch
  size and slot signature.
- Warm-then-capture behavior: the first matching slot signature runs eager to
  populate lazy scratch and one-shot FP8/SW bootstrap; the next matching step
  captures; later matching steps replay.
- Decode body graph is env-gated by `ARLE_DSV4_DECODE_BODY_CUDA_GRAPH=1`.
- Safety gates keep capture off for debug dumps, operator trace, combine
  overlap, FlashMLA TP overlap, DeepEP EP, and TP>1 unless
  `ARLE_DSV4_NCCL_GRAPH_CAPTURE=1` is explicitly set.
- SWA/hybrid attention, compressor/indexer updates, and FlashMLA FP8
  compressor packing can now consume device-resident `start_pos`.
- Replay advances host compressor/indexer metadata and FP8 compressed-row
  highwater after graph launch.
- DSv4 prebuilt CUDA archive symbol gates include the new start-position ABI
  symbols, so stale prebuilt archives fail closed.

This is not yet a performance claim. It is the smallest reversible substrate
needed to test whether the full DSv4 decode body can replay correctly.

## Verification

Local:

- `cargo fmt --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda`
- `git diff --check`

Remote:

- Pending. This change touches CUDA C and must pass pod-side nvcc build.
- Pending. Correctness gate must show normal EOS output, forced 32-token
  decode, and `scripts/dsv4_batched_decode_validate.py` answer pass.
- Pending. CUDA graph gate must show actual DSv4 body capture/replay in logs.
- Pending. Performance gate must run the matched DSv4-Flash TP8 + EAGLE +
  CUDA graph 256K/1500 hot-cache workload before comparing with the target
  TPOT ~4.85 ms.

## Rule

For DSv4 graph work, a captured graph is not a win by itself. The license order
is: graph-safe ABI, remote build, output correctness, replay evidence, then
matched workload performance. Any shorter path risks treating a debug shape or
fallback eager path as the target CUDA graph result.
