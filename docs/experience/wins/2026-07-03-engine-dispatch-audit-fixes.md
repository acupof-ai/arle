# Wrong-regime dispatch audit — 10 candidates, 5 confirmed, 4 fixed+verified

## Context

After four same-shaped bugs in one day (kernel exists, a dispatch gate routes
a production shape onto a wrong-regime path), ckl asked: 还有类似的问题吗?
A six-surface parallel audit (infer-cuda ops / qwen35 / dsv4 / autograd /
infer-core / metal) with adversarial per-finding verification answered it:
10 candidates, 5 confirmed, 5 refuted by the skeptic pass.

## Outcomes

| finding | verdict | action |
|---|---|---|
| `matmul_backward` no ensure_device healing (host-grad demotes the GEMM chain) | confirmed | fixed f54149c0 |
| rope `<=` gate backend-agnostic → Metal partial-rotary hard error | confirmed (my own c51ca26d regression; Metal was SIGSEGV-broken before too) | fixed f54149c0 (CUDA-only partial) |
| sidecar cap-32 evicts `keys().next()` (arbitrary victim) + remerge leaves stale-epoch snapshots | confirmed | fixed 15736c08 (LRU + clear-on-remerge); verified: restore-fail WARNs 0/3 rounds |
| radix publish capped at prompt_len while sidecar captures prompt+generated (boundary mismatch → tool-turn tail re-prefill + dead ~49MiB sidecar entries) | confirmed | fixed 6e820864 (publish takes the token slice; finish publishes the sidecar's boundary); verified: loss band, WARNs 0, no regression |
| frozen-prompt-KV lane linear-attn carry path host-only (+cat_seq/cat_heads host concat) | confirmed as documented deferral whose cost assumption breaks at gen_len≈13.6k | archived as the lane's precondition (plan doc) — opt-in lane, default path unaffected |

With the FP8 dense floor (7089bec9) riding the same build: steady-state toy
round ≈ **10s** (rollout ~3s + writeback 6.8s) — ~44× vs the original 438s.

## Rule

- The audit's five REFUTED candidates earned their keep too: adversarial
  verification against docs/experience licensing records kept measured KILL
  decisions (pooled-MoE, mixed-default) from being "fixed" back in.
- When two caches capture the same event (radix + sidecar), their boundary
  computation must be THE SAME expression — a `min(prompt_len, …)` in one and
  `prompt+generated` in the other is a standing miss generator.
