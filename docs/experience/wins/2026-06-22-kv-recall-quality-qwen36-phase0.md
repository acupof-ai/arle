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

## Verdict / Next

Phase-0 PASS licenses the Phase-1 Metal integration. Open (do NOT overclaim from
this): ctx only ~5.7K (not 128K/10M), passkey is the easiest probe (single needle,
exact digits), single seed/key, `mean-key` + absolute-position only. Before a
quality *number* (vs a mechanism license): RULER/LongBench at ≥32K, multi-needle,
top_k sensitivity, and an A/B of recall-absolute vs recall-repositioned (§5) —
expected null on this partial-RoPE model.

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
