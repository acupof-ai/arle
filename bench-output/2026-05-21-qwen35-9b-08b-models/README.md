# Qwen3.5-9B -> Qwen3.5-0.8B Phase 1 model smoke

Plan: `docs/plans/2026-05-21-arle-opd-qwen35-9b-to-08b-distillation-plan.md`
at `3cfca71`.

Date: 2026-05-21
Host GPU: NVIDIA GeForce RTX 4070 Ti SUPER, 16 GB
Python: `.venv/bin/python`
Model source: ModelScope only

## Downloads

| Role | ModelScope repo | Resolved path | Size | Result |
|---|---|---|---:|---|
| Teacher AWQ | `tclf90/Qwen3.5-9B-AWQ` | `/home/ckl/.cache/modelscope/hub/tclf90/Qwen3___5-9B-AWQ` | 12G | downloaded; infer load failed |
| Teacher BF16 fallback | `Qwen/Qwen3.5-9B` | `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B` | 19G | downloaded; infer load OOM on 16GB |
| Student | `Qwen/Qwen3.5-0.8B-Base` | `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base` | 1.7G | downloaded; train loader smoke passed |

`modelscope` was already installed in the project venv:
`modelscope==1.37.0`.

## Teacher smoke

AWQ was attempted first, matching the plan. The checkpoint is real AWQ
metadata (`quant_method=awq`, `bits=4`, `zero_point=true`), but the current
infer loader expects dense `.weight` tensors for later layers and fails when
the checkpoint switches to `qweight/qzeros/scales`.

Command:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
target/release/arle serve --backend cuda \
  --model-path /home/ckl/.cache/modelscope/hub/tclf90/Qwen3___5-9B-AWQ \
  --port 8123 -- \
  --num-slots 1 --max-seq-len 128 --chunked-prefill-size 128 \
  --max-num-batched-tokens 128
```

Result: `exit=101`

Failure:

```text
Failed to create scheduler: worker=0 cuda_ordinal=0 gpu=0:
Tensor 'model.language_model.layers.1.mlp.gate_proj.weight' not found in any shard
```

BF16 fallback was then attempted with FP8 KV and the smallest practical slot
budget. It memory-mapped the four BF16 shards, then failed during H2D copy
because the 19.3GB BF16 weight set does not fit a 16GB RTX 4070 Ti SUPER.

Command:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
target/release/arle serve --backend cuda \
  --model-path /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B \
  --port 8123 -- \
  --num-slots 1 --max-seq-len 128 --chunked-prefill-size 128 \
  --max-num-batched-tokens 128 --kv-cache-dtype fp8
```

Result: `exit=101`

Failure:

```text
Memory-mapped 4 shard(s) (19306.3 MB) in 0ms
Failed to create scheduler: worker=0 cuda_ordinal=0 gpu=0:
H2D copy failed: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")
```

Phase 1 teacher acceptance is therefore blocked until one of these lands:

- infer supports this zero-point AWQ checkpoint layout; or
- the teacher is changed to a smaller Qwen3.5 checkpoint that fits 16GB; or
- the run moves to a larger GPU.

## Student smoke

The first `qwen35_loader` run found a checkpoint layout gap:
`linear_attn.conv1d.weight` is stored as `[out, 1, kernel]` while the train
model stores the same data as `[out, kernel]`. The loader now accepts only that
singleton-dimension squeeze. Element count is unchanged.

Command:

```bash
INFER_TEST_QWEN3_06B_DIR=/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base \
cargo test -p train --test test_qwen35_loader --release -- --nocapture
```

Result: `exit=0`

Key output:

```text
loader_smoke_qwen3_0_6b: loaded ok, param_count = 322
loader_smoke_qwen3_0_6b: last-row logits[..5] =
[7.4128284, 18.017591, 8.885979, 12.604919, 2.7894695], all_finite = true
```

## Gates

- `cargo check -p train`: `exit=0`
- `cargo check --workspace`: `exit=0`
- Student real-checkpoint loader smoke: `exit=0`
- Teacher AWQ infer smoke: blocked by unsupported AWQ tensor layout
- Teacher BF16 infer smoke: blocked by GPU memory

Raw artefacts are in this directory.

## Pivot 2026-05-21: runnable 4B teacher

The 9B teacher remains blocked on this 16 GB card:

- `tclf90/Qwen3.5-9B-AWQ` is blocked by ARLE infer's missing zero-point AWQ
  decode path for `qweight/qzeros/scales` tensors.
- `Qwen/Qwen3.5-9B` BF16 is blocked by the 16 GB HBM ceiling before a single
  token can be decoded.

The runnable substitute is `Qwen/Qwen3.5-4B` from ModelScope. This keeps the
same Qwen3.5 family, keeps the distillation ratio at roughly 5x parameters
versus the 0.8B student, and fits on the RTX 4070 Ti SUPER. AWQ zero-point
support is filed as a separate axis, not on the critical path for the
distillation demo.

Download note: the first ModelScope command used `cache_dir='~/.cache/...'`,
which the Python API treated as a literal relative directory. The downloaded
snapshot was moved to the canonical cache path:
`/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-4B`.

4B teacher metadata:

```text
repo=Qwen/Qwen3.5-4B
path=/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-4B
dtype=bfloat16
hidden_size=2560
intermediate_size=9216
num_hidden_layers=32
num_attention_heads=16
num_key_value_heads=4
head_dim=256
vocab_size=248320
shards=2
directory_size=8.8G
```

Exact smoke command requested in the pivot:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
target/release/arle serve --backend cuda \
  --model-path /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-4B \
  --port 8123 -- \
  --num-slots 1 --max-seq-len 128 --chunked-prefill-size 128 \
  --max-num-batched-tokens 128
```

Result: server loaded and decoded one token:

```json
{"choices":[{"text":",","finish_reason":"length"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}
```

Default `mem_fraction_static=0.85` smoke facts:

```text
Memory-mapped 2 shard(s) (9319.8 MB)
GPU transfer complete in 1650ms
GPU memory @ post_model_load: free=6.91 GB / total=16.72 GB
TokenKVPool budget: 4.5 GB
nvidia-smi infer process: 14298 MiB
```

For Phase 2, the more practical same-shape smoke uses
`--mem-fraction-static 0.60`, which keeps the teacher runnable with more HBM
left for the student:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
target/release/arle serve --backend cuda \
  --model-path /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-4B \
  --port 8123 -- \
  --num-slots 1 --max-seq-len 128 --chunked-prefill-size 128 \
  --max-num-batched-tokens 128 --mem-fraction-static 0.60
```

Low-memory result:

```text
TokenKVPool budget: 0.7 GB
nvidia-smi infer process: 10650 MiB
completion: "Hello" -> ",", max_tokens=1
```
