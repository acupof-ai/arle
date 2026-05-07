# SGLang 0.5.11 Prefill Stack Survey

## Context

- Task: explain why SGLang 0.5.11 is 2.03x faster than ARLE on longctx 4k/c=4
  TTFT in `docs/experience/wins/2026-05-07-m_world1-p0-sglang-baseline.md`.
- Source inspected: `layers/linear.py`, all `layers/attention/*.py`,
  `layers/quantization/*.py`, `model_executor/forward_batch_info.py`,
  `launch_server.py`, and the live Qwen3 model file.
- The requested `sglang/srt/model_executor/models/qwen3.py` path does not exist
  in this install; the live Qwen3 implementation is
  `sglang/srt/models/qwen3.py`.
- Baseline runtime notes say `attention_backend=flashinfer`, KV dtype BF16,
  `cuda_graph_max_bs=8`, and likely `chunked_prefill_size=2048`.

## 1. Dense Prefill GEMM Kernel

- For plain BF16 Qwen3, SGLang does not use Triton, CUTLASS, FlashInfer, or a
  custom SGL kernel for dense linear GEMMs.
- The dense unquantized path is `UnquantizedLinearMethod.apply`.
- On CUDA, that method falls through to `torch.nn.functional.linear`.
- In practice this means ATen dispatches the GEMM to PyTorch's CUDA linear
  stack, normally cuBLAS/cuBLASLt for BF16 matrix multiply.
- `ColumnParallelLinear`, `MergedColumnParallelLinear`, `QKVParallelLinear`,
  and `RowParallelLinear` all call their `quant_method.apply(...)`.
- With `quant_config is None`, the method is the unquantized method above.
- I found no hybrid dense BF16 dispatch by N/M/K in `linear.py` or
  `quantization/unquant.py`.
- Quantized FP8 paths can select among FlashInfer TRT-LLM, FlashInfer CUTLASS,
  DeepGEMM, CUTLASS, and Triton-style kernels using shape/device config tables.
- Those quantized paths are not the likely source of this BF16 Qwen3 TTFT delta.
- Conclusion: SGLang's longctx prefill win is probably not from a custom dense
  GEMM kernel in the BF16 baseline.

## 2. Gate-Up and QKV Fusion

- SGLang uses the same high-level fusion pattern as vLLM for Qwen-family dense
  layers.
- Qwen3 attention constructs one `QKVParallelLinear`.
- The fused QKV output is split into `q`, `k`, and `v`, then Qwen3 applies
  Q/K RMSNorm and RoPE before calling `RadixAttention`.
- Qwen3 MLP reuses `Qwen2MLP`.
- `Qwen2MLP` constructs one `MergedColumnParallelLinear` for
  `gate_up_proj`.
- The MLP forward path computes `gate_up`, applies `SiluAndMul`, then uses one
  `RowParallelLinear` for `down_proj`.
- The Qwen3 loader maps HF `q_proj`, `k_proj`, and `v_proj` into `qkv_proj`,
  and maps `gate_proj` and `up_proj` into `gate_up_proj`.
- This matches the vLLM primitive shape: `QKVParallelLinear` plus
  `MergedColumnParallelLinear`.
- SGLang therefore avoids separate gate/up GEMM launches and separate q/k/v
  GEMM launches.
- However, ARLE's Phase 0 gate-up fusion bench did not recover the gap.
- Fusion should be treated as required table stakes, not the main explanation
  for the 2.03x TTFT delta.

## 3. Does SGLang Graph-Capture Prefill?

- Yes, SGLang has a separate prefill/extend graph-capture mechanism.
- This is not the classic decode CUDA graph path.
- `ForwardMode.is_cuda_graph()` covers `DECODE`, `TARGET_VERIFY`, `IDLE`, and
  `DLLM_EXTEND`; it does not classify normal `EXTEND` prefill as classic CUDA
  graph execution.
- The relevant feature is `PiecewiseCudaGraphRunner`.
- `ServerArgs.disable_piecewise_cuda_graph` defaults to `False`, and
  `piecewise_cuda_graph_max_tokens` defaults to `None`.
- During GPU memory setup, SGLang sets the token ceiling from
  `chunked_prefill_size` for non-MLA models if no explicit ceiling is supplied.
- On GPUs below 20 GiB, SGLang defaults `chunked_prefill_size=2048` and
  `cuda_graph_max_bs=8`.
- Therefore, on the 16 GiB RTX 4070 Ti SUPER baseline, the expected piecewise
  prefill graph token ceiling is 2048.
- SGLang auto-generates capture buckets up to that ceiling.
- The token buckets include 1280, 1536, 1792, and 2048 when max tokens is 2048.
- `model_runner.init_piecewise_cuda_graphs` captures piecewise graphs when the
  feature is not disabled and the model has attention layers.
- `PiecewiseCudaGraphRunner.capture_forward_mode` is `ForwardMode.EXTEND`.
- `PiecewiseCudaGraphRunner.can_run` returns true when the current token count
  fits under the captured max token count and unsupported features are absent.
- `replay_prepare` normalizes `ForwardMode.MIXED` to `EXTEND` for the graph
  guard.
- The replay path runs under `enable_piecewise_cuda_graph()`, initializes
  attention metadata, and then calls model forward under the normal context.
- `RadixAttention.forward` has a graph-aware path in extend mode.
- In that path it calls `unified_attention_with_output`, so the graph-safe
  custom op owns the attention call with a preallocated output tensor.
- Important negative finding: flash-attention FA3/FA4 backend comments say CUDA
  graph is only supported for decode/target-verify there, not extend.
- The SGLang baseline uses FlashInfer, where piecewise prefill graphing is the
  path to inspect.
- Conclusion: SGLang likely graph-captures each 4k prompt as graphable
  2048-token prefill pieces while ARLE appears to run prefill eagerly.

## 4. Prefill Attention Backend

- The default attention backend selection picks FlashInfer on this baseline.
- Hopper can pick FA3 and SM100 can pick TRT-LLM MHA, but SM89 with FlashInfer
  available falls to FlashInfer.
- `FlashInferAttnBackend` sets `prefill_backend = "fa2"` and
  `decode_backend = "fa2"`.
- It constructs `BatchPrefillWithPagedKVCacheWrapper`,
  `BatchPrefillWithRaggedKVCacheWrapper`, and
  `BatchDecodeWithPagedKVCacheWrapper`.
- Extend metadata setup computes prefix lengths, sequence lengths, request pool
  indices, KV indices, and page tables for FlashInfer.
- In normal eager extend, SGLang may use ragged prefill when deterministic mode
  is off, no piecewise CUDA graph is active, and the backend is not forced paged.
- Inside piecewise CUDA graph, `is_in_piecewise_cuda_graph()` makes
  `use_ragged=False`.
- That means graph-captured prefill uses the paged FlashInfer wrapper.
- In the paged path, SGLang writes K/V into the token-to-KV pool and calls
  `prefill_wrapper_paged.forward(...)` over that KV buffer.
- With prefix plus new extend chunk in eager mode, SGLang can compute ragged and
  paged states and merge them with FlashInfer cascade state merging.
- The closest ARLE counterpart is `batch_prefill_paged_hd128` in TileLang.
- The API class is similar: paged KV, head-dim-specialized prefill,
  per-request metadata, and variable lengths.
- The likely gap is not just "FlashInfer vs TileLang"; it is FlashInfer paged
  prefill running inside SGLang's piecewise graph-captured full layer stack.

## 5. ARLE Actionable Insights

- P0: add or prototype prefill CUDA graph capture for ARLE's longctx prefill
  path.
- P0: make the capture token-count based, not batch-size based.
- P0: start with 2048-token buckets on 16 GiB GPUs to match SGLang's default
  chunk and graph envelope.
- P0: capture the whole prefill layer loop, not only attention: norms, fused
  QKV, RoPE, KV write, attention, output projection, MLP, and residual flow.
- P0: make ARLE's attention op graph-safe with caller-provided output storage.
- P0: add log counters for graph hit, chunk size, token bucket, and fallback
  reason.
- P0: verify the SGLang baseline actually logs piecewise graph capture and
  captured token buckets before treating this as proven by behavior.
- P1: align ARLE's chunked prefill policy with SGLang for the same experiment:
  cap total newly extended tokens around 2048 and align chunks to page size.
- P1: benchmark ARLE `batch_prefill_paged_hd128` at the same 2048-token chunk
  envelope before changing the attention kernel.
- P1: keep QKV and gate-up fusion, but do not expect fusion alone to close this
  TTFT gap.
- P2: build an attention-only A/B against FlashInfer paged prefill at the same
  shapes after graph capture is controlled.
- P2: compare metadata/index construction overhead separately from kernel time.
- P2: inspect avoidable allocations or dynamic dispatch inside each prefill
  layer that graph capture would eliminate.
- P3: defer BF16 dense GEMM replacement work unless profiling shows cuBLASLt
  itself is the limiter after graph capture.
- P3: treat SGLang FP8/Triton/CUTLASS/DeepGEMM hybrid dispatch as future
  quantized-model input, not a BF16 Qwen3 prefill explanation.
