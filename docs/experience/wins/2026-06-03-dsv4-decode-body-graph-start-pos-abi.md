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
- Batched FlashMLA shared-pool slot/layer offsets are pre-staged as a stable
  `[layer][row]` device table outside body capture. The captured body no longer
  records per-layer H2D copies from one reused host scratch buffer.
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

- PASS. Pod build at commit `c7caabe5` completed in 7m10s and harvested a new
  DSv4 prebuilt CUDA archive:
  `/tmp/dsv4_body_graph_20260603/build.log`.
- PASS. The harvested archive exports the new ABI symbols:
  `dsv4_swa_attention_start_pos_ptr_cuda`,
  `dsv4_hybrid_attention_start_pos_ptr_cuda`,
  `dsv4_compressor_update_start_pos_ptr_cuda`, and
  `arle_dsv4_fp8_kv_pack_completed_compressor_row_start_pos_cuda`.
- PASS. Body-graph-off 32-token completions gate passed for c1, c4, and c8:
  every row returned HTTP 200, generated the full token budget, and contained
  `406`. Artifact:
  `/tmp/dsv4_body_graph_20260603/validate_off_32tok/completions32.log`.
- FAIL, contained. The first body-graph-on run captured B=4, then hit
  `CUDA_ERROR_ILLEGAL_ADDRESS` during sampling sync after only 3 generated
  tokens. Artifact:
  `/tmp/dsv4_body_graph_20260603/validate_body_graph_on_32tok/server.log`.
  Root-cause hypothesis is not performance-related: the captured body still
  records per-step owned GPU scratch from the FFN routed/shared path, so graph
  launch can use addresses that are no longer stable.
- PASS. Follow-up FFN scratch fix at commit `ecf819e2` removed the graph-on c4
  `PostMoeExpertAllReduce buffer len 32768 does not match logical len 16384`
  failure and captured B=4 without illegal address. Artifact:
  `/tmp/dsv4_body_graph_20260603/validate_body_graph_exact_scratch_marker32/server.log`.
- FAIL, contained. The same graph-on c4 run completed 32 tokens but generated
  semantically wrong marker text, while the same commit with
  `ARLE_DSV4_DECODE_BODY_CUDA_GRAPH=0` passed c4 marker32. Artifacts:
  `/tmp/dsv4_body_graph_20260603/validate_body_graph_exact_scratch_marker32/completions32.log`
  and
  `/tmp/dsv4_body_graph_20260603/validate_body_graph_off_marker32_c4_control/completions32.log`.
- PASS, partial. Follow-up fix at commit `68f1c7fd` pre-staged all batched
  FlashMLA shared-pool slot/layer block offsets outside body capture and only
  passed stable per-layer device-table pointers through the captured body. The
  body-graph-on marker32 gate then passed c1 and c4: every row generated 32
  tokens and contained `ZZZ406ZZZ`. Artifact:
  `/tmp/dsv4_body_graph_20260603/validate_body_graph_offset_table_marker32/completions32.log`.
- FAIL, contained. The same marker32 gate hung at c8 after all 8 ranks logged
  `Capturing DSv4 body CUDA Graph: B=8 slots=[0, 1, 2, 3, 4, 5, 6, 7]`. There
  was no CUDA illegal address and no decode output after capture. Artifact:
  `/tmp/dsv4_body_graph_20260603/validate_body_graph_offset_table_marker32/server.log`.
- FAIL, isolated. With `ARLE_DSV4_FLASHMLA_DECODE=0`, c8 still hung at the
  same `Capturing DSv4 body CUDA Graph: B=8` point. That rules out the
  FlashMLA slot/layer offset table as the B=8 hang root cause; the remaining
  target is body capture interaction with FFN/NCCL capture or larger graph
  capture topology. Artifact:
  `/tmp/dsv4_body_graph_20260603/validate_body_graph_offset_table_c8_no_flashmla/server.log`.
- PASS, guarded. Current code caps DSv4 body graph capture at
  `ARLE_DSV4_DECODE_BODY_CUDA_GRAPH_MAX_BS` (default `4`) and falls back to
  eager body execution above that cap. Startup contract logs
  `body_graph_enabled` and `body_graph_max_bs`, and the first oversized decode
  batch logs an explicit eager fallback warning. This prevents the default DSv4
  graph path from hanging c8 while preserving the validated c1/c4 graph path.
- PASS. Guarded c8 marker32 gate at commit `679f150a` completed: all 8
  requests returned HTTP 200, generated 32 tokens, and contained `ZZZ406ZZZ`.
  Startup logs showed `body_graph_enabled=true body_graph_max_bs=4`; oversized
  batches logged eager fallback; grep found no `Capturing DSv4 body CUDA Graph`
  lines. Artifact:
  `/tmp/dsv4_body_graph_20260603/validate_body_graph_cap_c8/completions32.log`.
- FAIL. Current HEAD `c21f9d59` with `--spec-enabled --spec-draft-model eagle`
  and MTP weights loaded returned 32 tokens for c4, but the marker outputs were
  corrupted (`ZZZZ406...`, punctuation-heavy garbage, and one row missing the
  marker). That means the current internal MTP/EAGLE path is not yet
  correctness-licensed, even though non-spec decode is. Artifact:
  `/tmp/dsv4_body_graph_20260603/validate_body_graph_cap_eagle_c4/completions32.log`.
- PASS, partial. Follow-up fix at commit `c5832434` passes the
  scheduler-owned decode context into `forward_spec_verify_batch` and removes
  the DSv4 verifier's temporary context. The EAGLE corruption root cause was
  the verifier creating its own short-lived decode context: with
  `ARLE_DSV4_SHARED_KV_POOL=1`, the FP8 KV pool is decode-context-owned, so the
  verifier rebound per-slot attention caches to a temporary pool. After the
  verifier returned, `commit_speculative_target_state` replayed accepted tokens
  through `forward_decode` without rebinding to the scheduler's persistent pool.
  The fixed c4 EAGLE marker32 gate returned HTTP 200 for all 4 requests,
  generated the full 32-token budget, and every output started with
  `ZZZ406ZZZ`. Artifact:
  `/tmp/dsv4_body_graph_20260603/validate_spec_ctx_eagle_c4/completions32.log`.
  This is a correctness gate only: it used the debug-fallback profile and a
  short 32-token decode window, so its per-request throughput is not comparable
  with the 256K/1500 hot-cache target.
- PASS, correctness-only. Follow-up commits `ea2fe1a8`, `2735b177`, and
  `f553d96a` made the internal MTP/EAGLE path fail closed instead of corrupting
  output:
  - speculative draft/bonus tokens now use the same distributed token
    synchronization as normal decode;
  - DSv4 verifier now uses the same per-row target path as commit replay,
    rather than accepting tokens under the batched verifier and replaying a
    different path;
  - internal MTP/EAGLE draft acceptance is disabled by default
    (`ARLE_INTERNAL_MTP_ACCEPT_DRAFTS=1` is experiment-only), so the verifier
    emits target bonus tokens only until MTP draft parity is proven.
  Remote build at `f553d96a` used the prebuilt CUDA fast path and completed in
  17.8 s:
  `/tmp/dsv4_body_graph_20260603/build_spec_mtp_accept_gate/build.log`.
  Spec-on raw completions gate passed with `ANSWER_PASS`: c1, c4, and c8 all
  completed without HTTP errors and every output contained `406`; c8 produced
  64 output tokens at 6.66 aggregate tok/s. Artifact:
  `/tmp/dsv4_body_graph_20260603/validate_spec_accept_gate_raw406_c8/batched_decode_validate.log`.
  Spec-on 32-token chat gate also passed under the weaker marker criterion:
  all 8 requests returned HTTP 200, generated 32 tokens, and contained
  `ZZZ406ZZZ`. Artifact:
  `/tmp/dsv4_body_graph_20260603/validate_spec_accept_gate_marker32_c8/completions32.log`.
  Strict `startswith("ZZZ406ZZZ")` is not a valid sole gate for this prompt:
  current no-spec c8 also produced one `ZZZZ406ZZZ...` row. The strict prompt
  remains a diagnostic only.
- FAIL, performance. The current `--spec-enabled --spec-draft-model eagle`
  path is correctness-safe but not performance-positive: because draft
  acceptance is disabled, it is effectively target verification overhead plus
  fallback normal decode. The marker32 c8 run took about 39.6 s for 32 tokens
  per request, with request traces around 0.81 completion tok/s per request.
  This is not comparable with the target DSv4-Flash TP8 + EAGLE + CUDA graph
  256K/1500 hot-cache line (`~4.85 ms` TPOT, `~196 tok/s`). The next
  performance tranche must fix internal MTP/EAGLE draft parity and re-enable
  acceptance before any 256K/1500 performance claim.
- Pending. Performance gate must run the matched DSv4-Flash TP8 + EAGLE +
  CUDA graph 256K/1500 hot-cache workload before comparing with the target
  TPOT ~4.85 ms.

## Rule

For DSv4 graph work, a captured graph is not a win by itself. The license order
is: graph-safe ABI, remote build, output correctness, replay evidence, then
matched workload performance. Any shorter path risks treating a debug shape or
fallback eager path as the target CUDA graph result.

Full-body capture must fail closed until every tensor pointer recorded by the
graph is owned by decode context or per-layer cache for the lifetime of the
graph. Per-step owned `HiddenStates` are not graph-safe.
