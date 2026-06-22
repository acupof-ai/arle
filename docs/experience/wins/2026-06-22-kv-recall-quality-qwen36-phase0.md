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
claim — so the two harder tests below were run to settle it.

## Memory benchmark: diverse distractors + multi-needle

`--distractor diverse` fills the haystack with a varied sentence pool incl.
number-bearing decoys ("serial 48213", "invoice 55820", "balance 71640");
`--needles N` buries N keys at even depths under one list-all Q; thinking disabled
via chat template (`enable_thinking=False`) so the answer is direct (a 0/N from
`<think>` truncation is a harness artifact). `kv_attended` is now MEASURED.

```
ctx~2.9K diverse 3-needle   full 3/3   stream 0/3   recall 3/3   kv/layer: 2937 / 288 / 544
ctx~15K  diverse 4-needle   full 4/4   stream 0/4   recall 4/4   kv/layer: 15156 / 288 / 544
```

recall = full = exact on a diverse 15K haystack with 4 simultaneous objects, at
**544 vs 15156 KV/layer = 27.9× less**. stream confabulates fake keys ("729104,
196906, 196906"). Measured kv_attended matches the formula exactly (full=S,
stream=288=n_init+n_local, recall=544=n_init+top_k·l_bs+n_local).

## Real-content quality: hard QA over a 29K repo doc

`--doc docs/projects/tiered-kv-cache.md` (~29K tok, repo-specific → unknowable
parametrically), two hard comprehension Qs; `full` (attend-all) is the quality ceiling.

```
Q "tiers T0–T3 + storage"    full ✅ HBM/DRAM/NVMe/remote   stream ❌ confabulates (vague T0 only)   recall ✅ exact=full
Q "4 shipped EvictionPolicy" full ✅ Lru/ReuseBiasedLru/HitCountLru/SessionBiasedLru   stream ❌ "L, L, L, L"   recall ✅ exact=full
kv_attended/layer:  full ~29.3K   stream 288   recall 544   →  recall = 53.9× less than full
```

recall reproduces full's answer **quality** (not just a planted needle) at 1.9% of the
KV; stream confabulates (Q1) and degenerates (Q2). This is the binding LongBench-style
test, and recall passes it.

## Agentic: multi-hop next-step decision

Beyond retrieval/QA: a 3-hop dependency chain scattered at depths 0.25/0.5/0.75 in a
diverse 15K haystack — deploy ⇐ smoke-test ⇐ feature-flag ⇐ approval-token — asked
"what is the FIRST action to deploy?" (correct: run `/approve` in #ops, the root
prerequisite, reachable only by chaining all three rules backward).

```
full   (kv 15149)  ✅ "Run /approve in the #ops channel to obtain the approval token"
stream (kv   288)  ❌ "Run ./deploy.sh --env production"   (naive WRONG action; rules reject it)
recall (kv   544)  ✅ "Run /approve in the #ops channel ..."   (exact = full)
```

recall **assembled the transitive 3-hop chain** — it pulled all three scattered rule
blocks (not just the query-similar DEPLOY one) and reproduced full's next-step decision
at **27.8× less KV**. stream, blind to the mid-context preconditions, emits the naive
deploy command the rules forbid — the canonical agentic failure. Single scenario /
single phrasing → directional, not a graded agentic suite.

## Aggregation: the first crack (recall < full)

12 distinct IDs scattered across a diverse 15K haystack, top_k=8 budget, "list all":

```
full    12/12  kv=15420   all exact
stream   0/12  kv=288     garbage loop ("728193, 482917, 192837, ..." repeated)
recall  10/12  kv=544     dropped #5 (472916) + corrupted #11 (629053→629103)
```

First measured `recall < full`: when items (12) exceed the recall budget (top_k=8),
recall degrades to 10/12 — one omission + one digit-corruption — the literature's
aggregation weakness ([SCBench arXiv:2412.10319]). But **graceful** (83%, not collapse
— per-step re-selection recovers most), not a cliff. Retrieval/agentic = recall's sweet
spot; list-all-many / count-all = where it cracks. The late-position corruption (#11)
weakly hints at long-generation drift but isn't cleanly isolated (a ~150-tok list is
too short; real drift needs hundreds of generated tokens). stream collapses (0/12).

## Verdict / Next

Phase-0 PASS, strengthened well past a mechanism license: depth profile @5.7K, ctx→16K,
top_k floor (holds to top_k=1), **diverse-distractor 4-needle (4/4 @15K)**,
**real-doc hard-QA quality (recall = full exact @29K, 53.9× less KV)**, and
**agentic 3-hop next-step (recall = full @15K; stream takes the forbidden action)**.
Open: 32K–128K,
multi-seed, free-form long QA with a graded rubric (vs the checkable-fact QA here), and
a §5 recall-absolute-vs-repositioned A/B (expected null). recall here is scope-only —
flat-VRAM needs the Phase-1 evict path (`executor.rs:2357-2430` contiguous-attn → Rust
page-gather; prefill scores unreachable → keep mean-key, validated).

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
