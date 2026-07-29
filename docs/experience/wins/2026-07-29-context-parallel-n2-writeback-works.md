# Context-parallel N=2 writeback works end-to-end — CUDA/H20, 2026-07-29

> Status: Shipped (N=2 verified on H20; 256K seq-ladder pending-remote)

## Goal

Make the OPD masked-CE writeback run across N GPUs with the sequence sharded
(per-card activation O(seq/N)) — the only path to 256K, since single-card walls
at ~seq 49152. This entry marks the first end-to-end N>1 run: correct loss,
not just lockstep collectives.

## What worked

Two changes closed the last gap after the launcher/seq-shard/ring-attention
bricks were already in place:

1. **Deterministic LoRA param order** (b8e2ad96b). `adapter_name_map()` was a
   2-entry `HashMap{lora_a, lora_b}`; Rust randomizes hash iteration per
   process, and each CP rank is a separate re-exec, so the two ranks fed
   `all_reduce_cp_grads` the same 64 params in **different order** — pairing
   lora_A `[16,5120]` against lora_B `[12288,16]` into one NCCL collective.
   NCCL has no size rendezvous → GPU spins forever (CPU races to DONE via async
   enqueue, so the host looks finished while the device wedges). Fix:
   `adapter_ordered()` returns fixed A-then-B; register + collect paths use it.
2. **Fail-fast layout guard** (f55c883a3). Before the reduce, each rank gathers
   a fixed-length per-param element-count vector and rejects the step if two
   ranks differ — turns the whole class of order-mismatch bugs from a silent
   24-min spin into a clear error in seconds. world==1 is a no-op.

## Parameters

```bash
arle train agent-opd \
  --model ThinkingCap-Qwen3.6-27B-FP8 \
  --synthetic-writeback-seq 8192 \
  --cp-size 2 --cp-devices 0,1 \
  --lora-rank 16 --lora-alpha 32 --lora-target-set attention-qv
```

- Backend: H20 (sm_90), cuda,nccl release, commit b8e2ad96b + f55c883a3
- Baseline (N=1): `--cp-size 1`, single GPU

## Result

| Config | loss | grad_norm | exit |
|--------|------|-----------|------|
| N=1 (single card) | 10.614354 | — | RUN_EXIT=0 |
| N=2 rank0 shard | 4.770942 | 1.575355e0 | RUN_EXIT=0 |
| N=2 rank1 shard | 5.727130 | 1.575355e0 (bit-identical) | RUN_EXIT=0 |

CP per-rank loss = `local_shard_CE_sum / global_count` (both ranks
total_targets=7936), so the shards are additive shares of one global mean:
**4.770942 + 5.727130 = 10.498 vs N=1 10.614 → 1.1% delta**, within
MoE-nondeterminism + float-add-order slack. The **bit-identical grad_norm**
across ranks is the direct signature that the all-reduce rendezvoused on
matching-sized tensors. Synthetic tokens are deterministic, so kernel/reduction
order is the only variance source. Parity holds — and it incidentally confirms
the RoPE/q_start absolute-position alignment (a wrong shard offset would
diverge the loss, not land at 1.1%).

## Rule

CP shards sequence, replicates weights → weight grads are DP-style
SUM-all-reduced post-backward, with loss scaled by `1/global_targets`. Any
per-rank list handed to a positional collective must be identically ordered on
every rank (see [`errors/2026-07-29-cp-nccl-wedge-is-hashmap-param-order.md`]).
Next: 256K seq-ladder on 8×H20 to find the new memory ceiling (load imbalance
of gather-prefix-KV — rank N-1 does N× rank 0's attention — is the expected
next wall).
