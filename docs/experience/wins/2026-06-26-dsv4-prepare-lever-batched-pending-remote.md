# DSv4 decode PREPARE lever batched (a+c+b) — written + committed, perf PENDING-REMOTE

Status: all 3 prepare-loop per-row holdouts batched + committed, every gate green
(cuda,no-cuda, n=1 bit-identity, FFI lockstep). The perf measurement (does perrow drop?)
is BLOCKED on GPU availability — honest pending-remote.

## Why this exists
afd6d717 (small GEMMs) was a TP=4 WASH; 30befcd9 (grouped O-LoRA) correct but marginal
(finish only ~12% of the step). Research (wq8xsowti) root-caused: the dominant batchable
per-row cost is the PREPARE (~41ms @ n=8: indexer/compressor cache-write 18 + pack/gathers
~20). Mainstream (vLLM/SGLang/DeepSeek-V3.2) batches ALL of decode via slot-indexed scatter;
per-request is NOT fundamental. Implementation-level research (wxh8qlb39) produced the spec.

## What landed
- **(a) cache-write → P1b pre-pass** (`45599845`): two batched DSA kernels
  (dsv4_dsa_hadamard128_batched + fused_store_index_k_cache_batched) mirroring the per-row
  bodies verbatim + blockIdx.y=slot + per-slot ptr/offset/count arrays (the
  dsv4_compressor_update_batched pointer-array pattern). Hoisted out of the per-row loop,
  gated use_batched_dsa_select && full_flatten && CompressedSparse. key_count for the READ
  preserved.
- **(c) pack batched** (`2e605c3c`): two batched flashmla pack kernels (SW one-token +
  compressed-delta), verbatim bodies + blockIdx.y=row + per-slot page-table-ptr arrays +
  [N] start_pos. 3×N → 2/step. Gated full_flatten && head_dim!=576.
- **(b) gathers via HiddenStatesView** (`1109738b`): borrowed CudaView<bf16> + col(r); deleted
  the ZERO-RIPPLE copies (#5 indexer_query_row + #6 csa_select double re-copy). DEFERRED
  (honest, diminishing): #1-4 (normed/c_q ripple to compressor_forward callers; q/k_prepared
  need a Dsv4MlaPrepared lifetime param).

All single-row / graph / !full_flatten / V32 / SparseIndexed lanes byte-unchanged. No default
behavior flip (runs in the existing batched decode lane). Built on HEAD through the ckl-shared
repo (interleaved commits 93c567a5 VRAM-registry + 72ca0e05 SIGTERM, disjoint).

## Why perf is pending-remote
DSv4-Flash needs TP=4 (4 GPUs × ~74GB weights/rank; TP=2 OOMs). The 8×H20 box stayed
contended throughout (ckl debug-serve on GPUs 0-3, other jobs on 5-6) — never 4 free big
GPUs. Re-verify when 4 ≥74GB GPUs are free: needle exact ×3 (correctness, the MoE-non-det
floor) + perf c-sweep + decode-phase breakdown — the n=8 prep should drop from ~45.8ms
(perrow 17 + pack/gather ~20 hoisted), lifting throughput. If prep drops but throughput
doesn't, the residual is MoE (54ms @ n=8, 45%, physics — needs batch≥32-64) + the deferred
(b) #1-4.

## Rule
Batch the WHOLE per-row decode tail (cache-write + pack + gathers), not just the GEMMs
(afd6d717's mistake) — the GEMMs were a small fraction; the per-slot state ops dominate and
ARE batchable via slot-indexed scatter (the repo's device-pointer-array pattern). Each kernel:
copy the per-row body verbatim, add blockIdx.y=slot + per-slot arrays, gate n=1 bit-identity,
FFI lockstep. Verify needs 4 dedicated TP=4 GPUs — on a shared box, defer pending-remote, do
NOT fake it on an undersized config.
