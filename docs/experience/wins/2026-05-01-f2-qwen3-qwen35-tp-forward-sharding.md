# F2 Qwen3/Qwen3.5 TP Forward Sharding Skeleton

## Context

F2 starts the real tensor-parallel forward line while H20 access is pending.
This tranche wires Qwen3 and Qwen3.5 model code to the F0.8
`LayerCommunicator` surface and makes BF16 safetensors loading shard-aware for
the projection weights that TP will split.

## What Worked

- Qwen3 now carries a TP runtime config through model load and shards
  `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and
  `down_proj` for BF16 safetensors when `world_size > 1`.
- Qwen3.5 now has the same sharded-load surface, including hybrid linear
  attention tensors: `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`,
  `conv1d_weight`, `dt_bias`, `A_log`, and `out_proj`.
- Forward hooks are in place:
  - full-attention outputs use `post_attn_all_reduce`.
  - MLP down-projection outputs use `post_mlp_all_reduce`.
  - Qwen3.5 linear-attention outputs use the DP attention gather hook so the
    recurrent path has an explicit communication slot.
- TP=1 remains single-rank no-op. Production TP>1 model load now fails fast
  until the F2 NCCL collective implementation is wired into
  `LayerCommunicator`; the sharded loader math is staged behind that guard and
  covered by CPU-verifiable shard tests.
- TP>1 model instances now hold per-rank effective dimensions so forward
  buffers match local shard rows instead of asserting on full-size buffers.

## Results

Status: implementation verified locally; TP=2/H20 throughput bench is
`pending-remote` until H20 SSH/path is available.

| Check | Result |
|---|---|
| `cargo fmt --check` | PASS |
| `cargo test -p infer qwen3_tp --features cuda` | PASS |
| `cargo test -p infer qwen35_tp --features cuda` | PASS |
| `cargo test -p infer model::layer_communicator --features cuda` | PASS |
| `cargo test -p infer tp::load_context --features cuda` | PASS |
| `cargo test -p qwen35-spec shard` | PASS |
| `cargo check -p infer --no-default-features --features cuda,no-cuda` | PASS |
| `cargo clippy -p infer --features cuda -- -D warnings` | BLOCKED by existing crate-wide lint debt outside this diff |

## Problems

No throughput bench was run locally because this tranche intentionally keeps
production TP>1 rejected until collectives are real, and the H20 machine is
still pending. The required follow-up bench is the H20 TP=1 no-regression smoke
and TP=2 Qwen3/Qwen3.5 forward sharding run after `LayerCommunicator` has NCCL
all-reduce support and SSH/path is provided.

## Learnings

Qwen3.5 needs a separate TP surface from Qwen3 because hybrid linear-attention
weights include sharded 1D recurrent tensors, not only matrix projections.

## Rule

When adding model TP support, wire both Qwen3 and Qwen3.5 together unless the
directive explicitly scopes out the hybrid model.
