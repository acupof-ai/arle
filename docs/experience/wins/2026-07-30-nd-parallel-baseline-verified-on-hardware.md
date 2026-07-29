# N-D parallel training baseline verified on real CUDA+NCCL (8×H20)

Date: 2026-07-30
Scope: `crates/train/src/{context_parallel,grad_clip,opd,update_strategy}.rs`,
`crates/train/src/qwen35.rs` (MoE-TP), `crates/cli/src/{args,train_cli,train_multiproc}.rs`,
`crates/autograd/src/ops/{ring_attention,collective_ep}.rs` + `tape.rs`.

## Context

The N-D parallel training work (mesh convergence, CP, MoE-TP, DP data-plane, plus
CP-ring/EP as live tape ops) was CPU-verified but unproven on hardware. Two
failures surfaced ONLY on real CUDA+NCCL — the CPU suite was green through both:
- CP writeback crashed: a prompt-heavy sequence shard owned zero masked targets →
  empty `position_indices` → fused CE panic. The rank had already run
  `all_gather_seq`, so it must still backprop or the CP group deadlocks.
- MoE-TP finite-diff gate mis-designed: name-seeded shards aren't complementary
  halves, so world=2-vs-1 reconstruction was never a valid identity; the rewritten
  rank-local FD then hit the fp32-loss-readout noise floor (~2.6%).

## What worked

- CP empty shard → zero loss that still depends on `hidden` (`sum·0`): value 0,
  grad 0, collectives in lockstep, grad all-reduce sums a genuine zero. Example
  trajectory reshaped (prompt 8→4) so targets straddle both shards.
- MoE-TP certified EXACTLY on CPU (`lora_shard::moe_tp_composed_swiglu_reconstructs_dense`):
  sum-of-per-rank-partials == dense SwiGLU expert, bit-exact f32 — no GPU, no
  loss-FD noise. The GPU FD gate is a coarse sanity check at its honest fp32
  noise floor (tol 3.5e-2), not the certifier.

## Measured (pod, GPUs 1/4, `--release --features cuda,nccl`)

| Gate | Result | Number |
|---|---|---|
| `nd_parallel_parity` (CP N=2 vs single) | PASS | loss rel_err **4.19e-5** (tol 1e-3) |
| `moe_tp_parity` (rank-local FD) | PASS | rel_err **2.61e-2** (tol 3.5e-2 = fp32-loss floor) |
| `lora_shard::moe_tp_composed_swiglu_reconstructs_dense` | PASS | bit-exact (CPU) |
| `cargo test -p autograd -p train --lib` | PASS | 247 passed, 0 failed |
| build `cuda,nccl` | PASS | 3m55s |

## Rule

CPU-green is not hardware-green: a value-pinned unit suite passed while both
multi-GPU parity gates failed on real NCCL (one a crash, one a mis-designed gate).
For a distributed-training change, the terminal gate is a pod parity run, not the
local suite. And when an fp32-loss finite-diff can't resolve a gradient, the fix
is an exact algebraic reconstruction test (GPU-free), not loosening tol or editing
the production loss to f64.
EOF
echo "written"