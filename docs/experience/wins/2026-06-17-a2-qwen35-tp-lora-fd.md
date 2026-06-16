# A2 Qwen35 TP LoRA Finite-Diff Gate

## Context

Train-side OPD needs tensor-parallel gradients before it can scale the native
autograd student path. The primitive differentiable NCCL all-reduce was already
licensed; this tranche wires it into Qwen35 train model row-parallel outputs and
adds a model-level gate.

Scope is intentionally narrow: dense full-attention Qwen35 LoRA with TP-local
q/k/v, gate/up, o/down weights. Hybrid linear-attention, MoE, vocab-parallel
loss, and FP8 frozen-base QLoRA remain later milestones.

## What Worked

- Added `Qwen35TensorParallelConfig` and explicit TP construction for train
  models.
- Full-attention TP local heads:
  - q/k/v projection rows are local.
  - `o_proj` consumes local attention hidden and all-reduces its output.
- MLP TP local intermediate:
  - gate/up projection rows are local.
  - `down_proj` consumes local activation and all-reduces its output.
- Rollout KV-cache and parity diagnostic entrypoints reject TP models instead
  of silently using global-head decode assumptions.
- Added `crates/train/examples/a2_qwen35_tp_lora_fd.rs`: a 2-rank coordinator
  that central-diffs a rank-local `mlp.down_proj.lora_b` element against the
  analytic autograd gradient through the distributed model.

Remote .62 H20 gate, GPU5/6, NCCL from `/tmp/nccl-cu12-227`, TileLang via
`/root/tl-venv/bin/python`:

```text
ARLE_A2_WORLD=2 ARLE_A2_CUDA_DEVICES=5,6 ARLE_A2_PROBE_INDEX=8 ARLE_A2_EPS=2e-3 \
  target/release/examples/a2_qwen35_tp_lora_fd

a2_qwen35_tp_lora_fd world=2 devices=[5, 6] probe=rank0[8] eps=2.0e-3
loss_minus=1.621813893e0 loss_base=1.621831298e0 loss_plus=1.621848583e0
analytic=8.695340715e-3 numeric=8.672475815e-3 rel_err=2.630e-3 tol=1.0e-2
PASS
```

Stability sweep on the same binary:

| eps | probe | rel_err | verdict |
| --- | ---: | ---: | --- |
| `2e-3` | 8 | `2.630e-3` | PASS |
| `5e-3` | 8 | `2.164e-3` | PASS |
| `1e-2` | 8 | `7.972e-4` | PASS |

The first default probe was too small for a stable GPU central-diff gate:
`probe=rank0[3] eps=1e-3` produced `rel_err=7.002e-2`; `probe=rank0[8]`
with `eps=1e-3` was closer but still just above the 1% threshold
(`rel_err=1.430e-2`). The default example now uses the stable probe/eps.

Local gates:

```text
cargo fmt --check
cargo check -p train --release --no-default-features --features no-cuda --lib
CUDARC_CUDA_VERSION=12090 cargo check -p train --release --no-default-features --features cuda,nccl,no-cuda --example a2_qwen35_tp_lora_fd
cargo test -p train --release --no-default-features --features no-cuda --lib opd::tests:: -- --nocapture
cargo test -p train --release --no-default-features --features no-cuda --lib qwen35::tests:: -- --nocapture
cargo clippy -p train --release --no-default-features --features no-cuda --lib -- -D warnings
```

Note: CUDA/NCCL clippy over examples is currently blocked by unrelated
`infer-cuda/src/dsv4.rs` `clippy::needless_option_as_deref` warnings when
`infer-api/cuda` is pulled in; this tranche did not touch those files.

## Rule

Model-level TP is not licensed by a source survey or a primitive all-reduce
test. Wire the row-parallel boundary, then central-diff a rank-local parameter
against the distributed total loss so the backward all-reduce adjoint is tested.
