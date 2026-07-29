# q8kv8 fp8 prefill: fires on the TP path now, but no win on sparse MLA

## Context

sgl-kernel 0.4.4 advertised +258~978% DSv4 prefill from its `sparse_mla_q8kv8_
prefill_sm90` kernel (fp8 q × fp8 kv, ~2x QK GEMM). We vendored it + a bf16→fp8
cast and gated it behind `ARLE_DSV4_Q8KV8_PREFILL=1`.

Two bugs stood between the flag and a measurement:
1. The branch sat BEFORE the TP all-gather and keyed on `local_heads`, so it was
   gated `tp_world==1 && local_heads%64==0`. Under TP=4 a 64-head model shards to
   local_heads=16 → doubly excluded → silent no-op. TP=1 was impossible (274 GB
   checkpoint vs 97 GB card). q8kv8 was structurally untestable on the real path.
2. Fixed by relocating q8kv8 to FlashMLA prefill's execution point (post-gather,
   global-head Q, gate `global_heads%64==0`), an if/else against the bf16 call
   (commit `5f67e285d`).

## What Worked / What Didn't

**The fix works**: on TP=4 DSv4-Flash-FP8 (global_heads=64), `ARLE_DSV4_Q8KV8_
PREFILL=1` now deterministically enters the q8kv8 branch (env confirmed on all 4
worker ranks; pure code gate, no runtime fallback). Output is coherent in both
arms — the identity-scale fp8 cast does not corrupt generation.

**No end-to-end win.** Measured on the pod (TP=4, GPUs 0/2/4/5, ~3000-word
unique prompts, cache defeated):

| prefill TTFT | median |
|---|---|
| baseline (bf16 FlashMLA) | 1.334 s |
| q8kv8 ON | 1.320 s |

Δ ≈ −1%, within noise. Decode is a wash as expected (q8kv8 touches prefill only):
c=1/4/8/16 per-req tok/s identical between arms.

## Rule

**A kernel win measured on a dense workload does not transfer to a sparse one.**
The ~2x is on the QK GEMM. This is DSA *sparse* prefill — QK is topk-bounded, so
it is NOT the prefill bottleneck (MoE FFN + gather + latent KV dominate). Halving
a small term is invisible in wall-clock TTFT. The upstream +258~978% was on a
regime where QK is the bottleneck; ARLE's sparse path already shrank it.

**Verdict: kill the default, keep env-gated.** q8kv8 stays behind
`ARLE_DSV4_Q8KV8_PREFILL=1` (default OFF) — no perf license on sparse MLA. It is
structurally correct + TP-safe + output-verified, so it costs nothing to keep for
a future dense/high-head-count model A/B where QK might dominate. Not worth an
`ncu` isolated-kernel chase on the current model: even a confirmed 2x QK would not
move the TTFT that the FFN/gather terms set.
