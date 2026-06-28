# CUDA GQA KV-head replication — Qwen3.5-122B-A10B unlocked at TP4

## Context

Qwen3.5-122B-A10B (2 full-attn KV heads) could not serve on CUDA at **any** TP:
TP4 failed `Qwen3.5 TP full-attention head shard failed: num_kv_heads (2) not
divisible by world_size (4)`, and 234 GB won't fit on TP≤2 (>97 GB/GPU). gap-4 of
the unified model-support plan.

## What worked

`head_shard` (`crates/infer-topo/src/sharding.rs`) gains a **replicate** regime:
when `num_kv_heads < world_size` (and `world % kv == 0`), each KV head is
replicated across `world/kv` ranks instead of sharded (`local_kv_heads = 1`).
New `kv_load_block_index(kv, tp)` maps rank → KV-head block (`rank` in the shard
regime — byte-identical; `rank / (world/kv)` in the replicate regime). The loader
(`infer-cuda/src/loader.rs` `load_qkv_head_sharded*`) takes an explicit
`block_index`: Q callers pass `tp.rank`, K/V callers pass `kv_load_block_index`,
so replica ranks load the **same** K/V projection slice. `qwen35.rs` relaxes the
full-attn divisibility gates for the replicate case. The CUDA attention path was
**already replication-ready** (passes `local_q_heads`/`local_kv_heads` explicitly,
`gqa_ratio = local_q/local_kv`, caches sized from `local_kv_heads`) — no kernel
change. 122B @ TP4: ranks 0,1 → KV head 0; ranks 2,3 → KV head 1; 8 Q + 1
replicated KV head/rank.

**Correctness rationale (o_proj/all-reduce):** each Q head lives on exactly one
rank, so the row-parallel o_proj all-reduce sums per-Q-head partials with no
double-counting; replication only duplicates KV *compute*, never Q output.
Replica ranks load identical K/V weights and see identical hidden states → their
local KV caches are bit-identical. Divisible case (kv ≥ world) stays
byte-identical (verified by `head_shard` unit tests; infer-topo 46 tests pass,
4 new replication tests).

## Verification status (honest)

**LOAD + SERVE verified on 8×H20:** 122B serves at TP4 — `[multiproc-coord] all 4
worker engines ready; opening HTTP` + `serving OpenAI v1 ... Qwen3.5-122B-A10B`,
the `not divisible` error gone. Engine-build succeeds across all 4 ranks with the
replicated head layout — a GQA-replication shape bug would fail engine-build, so
this is strong structural verification (weights shard+load, KV pools size, all
ranks construct).

**Completion (numerical) gate: PENDING a clean re-run.** Repeated 122B TP4
serve/kill cycles this session degraded the multiproc NCCL/IPC state (my teardown
killed the coordinator pid but not the 4 worker ranks → later launches hung at
NCCL init / worker engine-build), so a coherent-completion capture didn't land.
This is harness degradation, not gap-4. To confirm numerically: a fresh
process-namespace serve + a needle/same-config gate (per the KV-precision parity
gate). Low risk given clean all-rank engine-build + the o_proj/cache-identity
reasoning above.

## Rule

GQA at high TP (num_kv_heads < world_size) needs **KV-head replication**, not
sharding — replicate each KV head across `world/kv` ranks (identical weights +
caches), shard only Q. The shape machinery (gqa_ratio, local_kv_heads, cache
sizing) may already support `local_kv_heads=1` — check before assuming a kernel
change. **Teardown of a multiproc TP serve must kill all ranks** (not just the
coordinator pid): leftover worker ranks degrade NCCL state and hang subsequent
launches — use a process-group / all-rank kill (`pod_serve.sh stop`-style).
