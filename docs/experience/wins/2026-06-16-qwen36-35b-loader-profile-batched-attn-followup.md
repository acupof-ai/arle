# Qwen3.6-35B #101 loader profile + batched-attention 35B follow-up

## Context

Follow-up for issue #101 and the `1ed2df6c` Qwen batched full-attention decode
tranche. F2 coherence had already passed for both BF16 and FP8, so the remaining
questions were:

- where the 35B BF16 cold-load wall actually is;
- whether the batched full-attention decode wiring still helps on the real
  35B-A3B shape, not only the fast-load small-model surrogate.

## Loader RCA

The real 35B-A3B BF16 checkpoint takes the stacked expert path, not the
per-expert tensor path:

- per-expert branch: `crates/infer-cuda/src/loader.rs:963`;
- stacked branch taken: `crates/infer-cuda/src/loader.rs:968`;
- evidence: startup profile emitted `loader.moe.stacked_routed_load` for all 40
  MoE layers, with `local_experts=256 gate=256 up=256 down=256`.

Live busy-thread stack while loading showed the `infer-engine` thread in
`read()`; the phase profile agrees:

| Phase | Count | Total ms | Avg ms |
|---|---:|---:|---:|
| `qwen35.total` | 1 | 47,691.6 | 47,691.6 |
| `loader.shard_read` | 27 | 37,807.7 | 1,400.3 |
| `loader.moe.stacked_routed_load` | 40 | 6,943.5 | 173.6 |
| `loader.shard_deserialize` | 26 | 2.2 | 0.1 |

Shard IO read 74.14 GB total. The slow wall is cold shard file reads, not
safetensors metadata parse and not the per-expert loop.

## Killed Experiment

A bounded background shard readahead prototype was tested and removed before
landing:

| Metric | Baseline | Readahead prototype | Delta |
|---|---:|---:|---:|
| `qwen35.total` | 47,691.6 ms | 48,318.1 ms | -1.3% |
| `loader.shard_read` | 37,807.7 ms | 38,427.0 ms | -1.6% |
| `loader.moe.stacked_routed_load` | 6,943.5 ms | 6,858.2 ms | +1.2% |

The root cause is licensed; this specific fix is killed. No runtime code was
kept from the experiment.

## 35B Batched-Attention Follow-Up

Same binary, same prompt, same H20, c=8 streaming probe. The before arm used
`ARLE_QWEN35_BATCHED_DECODE_ATTENTION=0`; default is the after arm.

Correctness:

| Arm | BLUE-73-MANGO c=8 |
|---|---:|
| before per-row | 8/8 PASS |
| after batched | 8/8 PASS |

Throughput:

| Arm | Wall s | Output chunks | Chunks/s | TTFT avg ms | ITL avg ms |
|---|---:|---:|---:|---:|---:|
| before per-row | 3.996 | 1024 | 256.26 | 423.26 | 27.86 |
| after batched | 3.645 | 1024 | 280.91 | 420.77 | 25.14 |
| delta | - | - | +9.62% | -0.59% | -9.76% |

Dispatch geometry:

- before path loops over rows at `crates/infer-cuda/src/qwen35.rs:3608`,
  launching the single-row attention core for each B row;
- after path launches `fused_gqa_attention_decode_batched` once with `b as i32`
  at `crates/infer-cuda/src/qwen35.rs:3548`, then one batched reduce and one
  batched gate.

For B=8 and 10 full-attention layers, the attention core drops from 80
single-row launches per decode step to 10 batched launches per decode step.

## Rule

Do not optimize the branch named in the hypothesis until the live checkpoint
proves it is the branch being taken. For this checkpoint, #101's next real
loader lever is cold shard IO / load-order overlap; per-expert CPU parse
parallelism is not on the BF16 35B critical path.
