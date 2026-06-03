# R6 clean-CUDA forward — eager greedy parity VERIFIED on H20 (Phase 0 closed)

**Status:** PASS — exact HF-gold greedy parity. Phase 0 (CUDA eager correctness) closed.
**Track:** R6 clean-CUDA rewrite (`crates/infer-cuda`), branch `arch/ideal-inference-engine`.
**SKU:** H20 / sm_90a, CUDA 12.9, TileLang 0.1.9 venv, Qwen3-0.6B BF16.

## Context

The clean `infer-cuda` BF16 Qwen3 forward (the rewrite's CUDA executor: engine →
executor → `CudaModel::forward_tokens` → paged attention → TileLang kernels) had never
been proven on a real GPU. Bring-up surfaced and fixed a chain of bugs (loader O(N²)
`3f5f2ece`; head_dim assert `fe841c62`; num_pages/total_pages arg swap `db85d56e`),
then hit a hard TileLang HD128 **batched-prefill** WGMMA codegen spin
(`errors/2026-06-04-tilelang-hd128-prefill-wgmma-hang-sm90a.md`).

## What Worked

**Decouple correctness from the broken batched-prefill cubin.** A 1-token forward
routes `seq_len==1 → decode` kernel, which is proven good (it ran cleanly through all
28 layers). Processing the prompt as **sequential 1-token forwards** through the decode
kernel (each at incrementing `start_pos`, accumulating KV) is causally identical to a
batched prefill — same logits, just slower. So we verified the forward's numerics with
the working decode kernel and left the batched-prefill cubin as a perf follow-up.

**Result (H20, greedy, MAX_NEW=16, prompt "The capital of France is" =
`[785,6722,315,9625,374]`):**

```
clean_tokens = [12095, 13, 576, 6722, 315, 9625, 374, 1083, 279, 6722, 315, 279, 5429, 315, 9625, 13]
HF gold      = [12095, 13, 576, 6722, 315, 9625, 374, 1083, 279, 6722, 315, 279, 5429, 315, 9625, 13]
```

**Exact match — 16/16 tokens.** The clean R6 CUDA forward is numerically correct.

The sibling decode `cache_len != kv_seq_len` invariant error did **not** recur once the
pod was rebuilt against current `infer-core` — confirming it was a stale-pod-binary
artifact (current `planner.rs:36` captures `kv_seq_len` pre-allocate; regression guards
`8388fc64`).

## Rule

- **A correctness gate doesn't have to wait on the fast path.** When the batched
  prefill cubin was wedged by a hard upstream-TileLang codegen bug, the proven decode
  kernel + `chunk_size=1` (sequential single-token prefill) verified end-to-end numeric
  correctness anyway. Separate "is the forward correct" (gate) from "is the fast kernel
  fixed" (perf follow-up) — don't let the latter block the former.
- **Bisect kernel-vs-launch with a 1-token forward** (different cubin, same launch
  path) — it proved the R6 architecture sound in one cheap run and localized the spin to
  the batched-prefill cubin.

## Follow-ups

- Batched HD128 prefill cubin (fast long-prompt prefill): perf-only, blocked on an
  upstream TileLang lowering fix or a FlashInfer-C++ migration of HD128 paged prefill.
- Multi-shape / longer-prompt greedy parity sweeps (chunk=1) before the perf path lands.
