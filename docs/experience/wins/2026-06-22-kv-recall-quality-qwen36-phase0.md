# KV-as-memory semantic recall — Phase-0 quality PASS on Qwen3.6-4bit

## Context

Verifying the KV-long-term-memory doc thesis
(larkoffice docx/QHWydhfofob92Yx8NSpcP50Znnc, §6 recall quality) on the canonical
Metal model `mlx-community/Qwen3.6-35B-A3B-4bit`, before committing to any Metal
runtime integration. Gate question: does InfLLM-style semantic block-recall
restore long-context retrieval that StreamingLLM loses — on *this* model's regime
(4-bit, MoE, hybrid attention), which no paper covers (all are full-attn FP16
Llama/Mistral)?

Harness: `scripts/kv_recall_quality_eval.py` — monkeypatches only
`Qwen3NextAttention.__call__` (mlx_lm 0.31.2 `qwen3_next.py:121`), passkey
retrieval across needle depths, 3 conditions on decode attention (prefill always
full): `full` (attend all KV), `stream` (sink+local only), `recall` (mean-key
top-k blocks + sink + local).

## What Worked

**Architecture finding (reshapes the doc).** Qwen3.6 = `qwen3_5_moe` (Qwen3-Next
family) is **hybrid linear+full attention**: 40 layers, `full_attention_interval=4`
→ only **10/40 layers** are full softmax attention with a KV cache (layers
3,7,…,39); the other **30/40 are GatedDeltaNet** (linear attn, fixed recurrent
state, no KV cache, no blocks to recall, and `qwen3_next.py:172-307` confirms **no
RoPE**). So the doc's "全量注意力 KV 线性增长→撞墙" premise governs only 25% of
layers, and `b_kv` ≈ 0.02 MB/token (kv_heads=2, head_dim=256), ~6× below the doc's
Llama-3 0.13 MB/token. Full-attn layers use **partial RoPE: 64/256 dims**
(`partial_rotary_factor=0.25`, `rope_theta=1e7`) → §5 position-compat problem is
structurally small here.

**Result grid** (ctx~5.7K, n_init=32, n_local=256, l_bs=32, top_k=8, greedy):

```
depth     full   stream   recall
0.0        OK      OK       OK
0.25       OK       .       OK
0.5        OK       .       OK
0.75       OK       .       OK
1.0        OK      OK       OK
          1.00    0.40     1.00
```

`recall = full = 1.00 ≫ stream = 0.40`. Stream retrieves only at depth 0/1.0
(passkey inside sink / local window); recall recovers every middle depth. The
mechanism works with the **Metal-feasible** variant: mean-key representative (not
InfLLM attention-influence score, which is unreachable behind the fused
`scaled_dot_product_attention`) and absolute KV positions (no re-encoding). Micro:
stream misses emit "7391" (first 4 of 6 digits) — the GDN linear layers leak
*partial* long-range info; full-attn recall completes the exact match.

## Scale: ctx 16K + top_k floor (single-variable steps from 5.7K)

ctx 5.7K→**15.1K** (depth 0.5, all else fixed): `full=OK, stream=MISS, recall=OK`.
At top_k=8 that is 256 recalled tokens / ~14.8K middle = **1.6% recall fraction** —
mean-key ranks the needle block in the top-8 among ~480 blocks.

top_k floor at 16K/depth-0.5 (sweep top_k, recall mode):

```
top_k:   1     2     4     8
recall:  OK    OK    OK    OK      (full=OK, stream=MISS anchors)
```

**recall holds down to top_k=1** — recalling ONE 32-token block (~0.2% of context)
still returns the full passkey. Mean-key recall ranks the needle block **#1**, not
just in-the-top-k. The Metal-feasible representative is sharply discriminative here.

**Caveat — this is the easy task.** passkey-in-uniform-filler makes the needle's
mean-key trivially distinct from repetitive filler, so #1-ranking is expected-easy.
top_k=1 @ 0.2% is impressive *on a needle among uniform distractors*, NOT a general
claim. A needle among *diverse* distractors (real LongBench/∞-Bench docs) is the
binding test and is not yet run.

## Verdict / Next

Phase-0 PASS licenses the Phase-1 Metal integration. Done so far: depth profile at
5.7K, ctx step to 16K, top_k floor (holds to top_k=1) — all on passkey-in-uniform-
filler, `mean-key` + absolute-position. Still open before a general quality *number*:
the binding test is a needle among **diverse distractors** (RULER/LongBench/∞-Bench),
not uniform filler; plus 32K–128K, multi-needle, multi-seed, and an A/B of
recall-absolute vs recall-repositioned (§5) — expected null on this partial-RoPE model.

Phase-1 in-stack constraints already mapped (Metal): decode attention is
contiguous-range only (`infer-metal/src/executor.rs:2357-2430`), no per-row page
list in `ForwardPlan`/`KvBatchDescriptor` → recall needs Rust-side page-gather
pre-assembly; prefill scores unreachable → keep mean-key representative (validated
here).

## Rule

Before integrating a paper's method into the runtime, reproduce its quality on the
*actual* target regime (here: 4-bit MoE **hybrid** attention) with the
runtime-feasible variant (mean-key, not the paper's score-based key) — a PASS on
the model you can't actually build is a false license. And check the model's
attention topology first: on a hybrid linear+full model, KV-as-memory only governs
the full-attn fraction; the linear layers are already constant-state but
non-retrievable.
