# A15 Qwen3.6 MoE Backward Substage Profile

## Context

Path A has OPD rollout generation on infer-core KV decode, so the remaining
35B training work is the differentiable autograd path. A12/A14 made the
Qwen3.6 FP8 MoE layer gate active-expert only, but `MoeGroupedLinear` still
owned about 0.20s of the single-layer backward. Before writing a resident
grouped kernel, this tranche adds an env-gated substage profiler to identify
which part of the grouped backward is actually expensive.

## What Worked

`ARLE_MOE_GROUPED_PROFILE=1` now prints per-call MoE grouped-linear timings:

- shape / active expert count / rank / gradient ownership flags;
- host pack stages;
- helper-level upload / GEMM / eval / readback stages;
- LoRA A/B unpack and input-gradient merge;
- per-call total.

Default behavior is unchanged when the env var is unset.

## Evidence

Remote `.62`, GPU0, model `/data01/models/Qwen3.6-35B-A3B-FP8`, command:

```text
ARLE_MOE_GROUPED_PROFILE=1 qwen36_fp8_lora_fd_gate \
  --model /data01/models/Qwen3.6-35B-A3B-FP8 \
  --device 0 \
  --target-set all-linear \
  --target-adapter auto:routed-up \
  --mode mlp-layer \
  --layer 0 \
  --eps 1e-3 \
  --profile-backward
```

Log:

```text
/data01/arle-track1-route-frozen-fd-fast-20260617095440/moe_grouped_profile_gpu0_20260617_105954.log
```

Correctness stayed licensed:

```text
qwen36_fp8_lora_fd_gate PASS
rel_err=3.170e-3
```

The load-bearing breakdown is the first grouped-linear call, the routed down
projection:

```text
call=1 experts=256 active=24 max_rows=1 in_dim=512 out_dim=2048 rank=8 input_kind=Packed
need_input_grad=true need_weight_grad=false need_lora_a_grad=true need_lora_b_grad=true
pack/base_weight_t=0.279274s
base_backward/upload_b=0.014170s
base_backward/matmul_backward_device=0.000118s
call_total=0.316998s
```

The other two grouped-linear calls, routed gate/up, are much smaller:

```text
call=2 in_dim=2048 out_dim=512 need_input_grad=false call_total=0.014745s
call=3 in_dim=2048 out_dim=512 need_input_grad=false call_total=0.014303s
```

The profiler adds print/timing overhead, so its total should not be compared as
a speed regression against A14. Use it for stage attribution only.

## Verdict

The next MoE backward kernel must target the frozen-base down-projection
input-gradient path:

```text
dX = dY @ W_down
```

The base weight is frozen (`need_weight_grad=false`) but the packed activation
needs a gradient (`need_input_grad=true`). Current code still materializes and
transposes active expert base weights on host (`pack base_weight_t`), which is
the dominant measured wall. The useful GEMM itself is not the wall.

The valid next tranche is a device-resident FP8/BF16 grouped input-gradient
kernel over resident expert weights, returning packed input gradients without
host weight pack/readback. Per-call f32 repack remains killed by
[`../errors/2026-06-17-qwen36-moe-forward-repack-gemm-kill.md`](../errors/2026-06-17-qwen36-moe-forward-repack-gemm-kill.md).

## Verification

Local:

```text
cargo fmt --check
cargo test -p train --release --test test_moe_a0 -- --nocapture
cargo test -p autograd --release --lib
cargo clippy -p autograd --release --no-default-features --features no-cuda -- -D warnings
cargo check -p train --release --no-default-features --features no-cuda --example qwen36_fp8_lora_fd_gate
CUDARC_CUDA_VERSION=12090 cargo check -p train --release --no-default-features --features cuda,no-cuda --example qwen36_fp8_lora_fd_gate
cargo clippy -p train --release --no-default-features --features no-cuda --example qwen36_fp8_lora_fd_gate -- -D warnings
```

Remote:

```text
CUDA_HOME=/usr/local/cuda CUDARC_CUDA_VERSION=12090
ARLE_CUDA_DISABLE_FLASHMLA=1 INFER_TILELANG_PYTHON=/root/tl-venv/bin/python
CARGO_TARGET_DIR=/data01/arle-target-track1-route-frozen-fd
cargo build -p train --example qwen36_fp8_lora_fd_gate --release --no-default-features --features cuda

ARLE_MOE_GROUPED_PROFILE=1 ... qwen36_fp8_lora_fd_gate ... --profile-backward
```

## Rule

Do not write the grouped MoE backward kernel from the operator name alone. For
Qwen3.6 FP8 LoRA, the measured wall is frozen-base down input-gradient weight
materialization, not LoRA A/B math and not generic grouped GEMM latency.
