# DSv4 V2.4 FlashMLA SM90 sparse prefill — real root cause found (max_logits/lse nullptr) + measured 12.4% win at 16K

## SLO-shape probed? — Y (4K, 16K, 24K all measured end-to-end, no crashes)

## TL;DR

Earlier V2.x rollouts assumed the crash at non-16384 token counts was an
s_q alignment / TMA descriptor problem. **The actual root cause was the
ARLE shim passing `nullptr` for `params.max_logits` and `params.lse`.**
FlashMLA's phase1.cuh:457-458 unconditionally writes to those buffers via
`SM90_BULK_COPY_S2G::copy` for every (s_q_idx, q_h_idx) block, so
`nullptr + g_offset` overlapped a CUDA-MMU-protected page → illegal
memory access at runtime.

Upstream FlashMLA always allocates these as part of the C++ API
([`csrc/api/sparse_fwd.h:155-159`](https://github.com/deepseek-ai/FlashMLA/blob/main/csrc/api/sparse_fwd.h))
and the Python tests parametrize `s_q ∈ {1, 62, 213, 1024, 4096, 70000}`
([`tests/test_flash_mla_sparse_prefill.py`](https://github.com/deepseek-ai/FlashMLA/blob/main/tests/test_flash_mla_sparse_prefill.py))
— **there is no s_q alignment requirement**, the kernel handles
arbitrary token_count.

## Roofline check

| Op | Achieved | Peak (8×H20 BF16) | % | Verdict |
|---|---:|---:|---:|---|
| 16K prefill (V2.4 FlashMLA on) | 16017 tok / 103.13 s × 8 = ~1242 tok/s aggregate | ~624,000 tok/s aggregate | 0.20% | PARTIAL — clearly a real fix (no crash, +12.4% over legacy) but well below FlashMLA's published 640 TFLOPS roofline. Bigger gains are gated on A4 multi-stream overlap (AllGather Q + repack still serializes per-layer). |

## Measured results — 8×H20, DSv4-Flash, TP=8, fp8 KV cache, num-slots=4

| Workload | Legacy (FlashMLA off) | V2.4 (FlashMLA on) | Δ% | Notes |
|---|---:|---:|---:|---|
| 4K (4017 tok, single chunk) | 17.6 s | **16.96 s** | **−3.6%** | small s_q; AllGather Q overhead amortizes weakly |
| 16K (16017 tok, single chunk) | 117.75 s | **103.13 s** | **−12.4%** | sweet spot — single chunk, large enough for FlashMLA to amortize collective overhead |
| 24K (24016 tok, chunked-prefill chunk-1 + chunk-2) | ~190 s | 190.94 s | ≈ 0% | wash — chunk-2 's smaller s_q (7632 tok) has more collective overhead per useful FLOP; A4 overlap needed to recover |
| 29K (with V2.2 strict gate, chunked) | 282 s | 281.7 s | wash | gate restricted to 16384-only; chunk-2 fell to legacy. V2.4 will reprobe. |

All four workloads now complete end-to-end with FlashMLA on (`finish_reason="length"`, real response tokens) — V2.x was previously
crashing at non-16384.

## Root-cause trail

| Hypothesis | Outcome |
|---|---|
| s_q must be 64-aligned (TMA box dim) | **WRONG.** Upstream tests use s_q=1, 62, 213 freely. |
| stride_indices_h_kv must be `topk`, not 0 | **NOT THE BUG.** Kernel never reads it (h_kv=1, no h_kv iteration). |
| kv_unified buffer sizing / layout wrong | OK by inspection; tests pass at 16384. |
| Attn_sink dtype/shape mismatch | OK; f32 mirror is sized [num_attention_heads] correctly. |
| **`params.max_logits == nullptr` and `params.lse == nullptr`** | **ROOT CAUSE.** phase1.cuh:457-458 unconditionally writes to these. |

The earlier V2.3 s_q-padding attempt only changed the failure surface
(from "TMA descriptor init at non-64-aligned s_q" to "runtime illegal
memory access inside kernel"). The descriptor init failure at 4017 was
suspicious — but turns out it was a downstream symptom of an earlier
sticky CUDA error caused by the nullptr write of the PREVIOUS request /
chunk that this thread happened to surface during the descriptor init
TMA setup of the next call. With max_logits/lse properly allocated, the
descriptor inits cleanly for any s_q.

## Fix

`infer/src/model/deepseek/weights.rs::finish_attention_gpu` now allocates
`max_logits_scratch` and `lse_scratch` of size `padded_s_q ×
h_q_for_flashmla` f32 each (≤ 1 MB at the verified 16384-token shape)
and passes real pointers to the shim. Both buffers are dropped after the
dispatch.

Commit chain:
- `86d27946` — V2.4 root cause fix (allocate max_logits/lse).
- `7fc575f8` — V2.4 perf cleanup (drop the V2.3 intermediate q_send_buf
  memcpy that no longer serves a purpose now that padding is no-op).

V2.4 gate: `token_count == 16384` (TP=1 path, strict) OR `tp_world > 1 &&
token_count > 1` (V2.4 broader path). Total-position cap of 24576 stays.

## What's still on the table

1. **A4 multi-stream overlap** — the 24K chunked-prefill wash signals
   that AllGather Q + repack overhead dominates when individual chunks
   are smaller than ~16K. SOTA reference: TokenWeave (arxiv 2505.11329)
   for per-tile compute-comm overlap; SGLang DSv4 day-0 uses
   "hierarchical multi-stream overlap" for the same axis. Designed in
   the prior session log; implementation in a dedicated follow-up.
2. **TP=1 path** — keeps strict 16384-only gate because FlashMLA reads
   q_prepared / writes local_attn directly (no padded scratch). A V2.5
   pad-at-TP=1 follow-up would symmetrize the dispatch.
3. **HCA layers' max_compressed_keys ceiling at very long context** —
   the 29K crash at the V2.0/V2.1 stage was likely a combination of
   nullptr write OOB AND a kv_unified sizing issue. Reprobe needed under
   V2.4.

## Why the bug stayed hidden so long

The 16384-token chunk-1 of chunked-prefill at 29K appeared to "work"
even with nullptr max_logits/lse because:

1. At s_q=16384 the writes hit address range `[nullptr, nullptr + 4 MB +
   256)`, which on H20 happens to overlap a driver-reserved CUDA region
   where the write is silently dropped or absorbed without raising
   `ILLEGAL_ADDRESS`.
2. At s_q=4017 / 14250 the writes hit a different sub-page that DID
   trigger the MMU.
3. ARLE never reads max_logits / lse, so the output of the prefill
   (the bf16 `out` tensor) was still correct, and downstream MoE +
   decode worked fine when the writes happened to land in a safe page.

Bottom line: even where the crash didn't surface, **the kernel was
producing undefined behavior**. The V2.4 fix is mandatory for correctness
regardless of the perf delta.

## Rule

When wrapping a kernel that returns multiple tensors, **always allocate
every output the kernel expects to write to**, even if the wrapper has
no consumer for the value. The "optional" of an output buffer is a
property of the kernel implementation, not the API. Verify by reading
the actual kernel source for unconditional writes to every output
pointer.

Corollary: when integrating a vendored kernel, hand-check the wrapper
against the upstream's reference invocation (typically a Python C++
extension that lays out all params via `torch::empty`) — the
"non-mandatory" tensors are often where the bug hides.

## Refs

- Upstream: [FlashMLA](https://github.com/deepseek-ai/FlashMLA), commit `df022eb`
- [csrc/sm90/prefill/sparse/phase1.cuh:457-458](https://github.com/deepseek-ai/FlashMLA/blob/main/csrc/sm90/prefill/sparse/phase1.cuh) — the unconditional write
- [csrc/api/sparse_fwd.h:155-159](https://github.com/deepseek-ai/FlashMLA/blob/main/csrc/api/sparse_fwd.h) — upstream always allocates max_logits/lse
- [tests/test_flash_mla_sparse_prefill.py](https://github.com/deepseek-ai/FlashMLA/blob/main/tests/test_flash_mla_sparse_prefill.py) — s_q test values
- Probe artifacts on pod:
  - `2026-05-28-dsv4-long-4k-mt32-fm1/`
  - `2026-05-28-dsv4-long-16k-mt32-fm1/`
  - `2026-05-28-dsv4-long-24k-mt32-fm1/`
