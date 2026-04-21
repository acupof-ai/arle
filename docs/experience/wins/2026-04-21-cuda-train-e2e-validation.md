# CUDA train end-to-end validation on L4

## Context

`docs/experience/wins/2026-04-19-end-to-end-training-flows.md` had already
marked the CUDA training path as "code-complete, pending remote
verification". On 2026-04-21, the workspace had a real NVIDIA L4 box
available, so the goal was to run the train-side CUDA path end-to-end:

1. build the `autograd` CUDA backend and train binaries,
2. run from-scratch Qwen-family pretraining on CUDA,
3. verify checkpoint save + `latest` publication,
4. reload the checkpoint through `eval_lm`,
5. resume from the checkpoint and continue training,
6. rerun CUDA backend tests and clippy after any fixes required for the
   current `cudarc` API.

## What Worked

Two CUDA compile regressions surfaced immediately against `cudarc 0.18.2`:

- `backend_cuda.rs` needed `cudarc::driver::PushKernelArg` imported so
  `LaunchArgs::arg(...)` was in scope.
- The local `launch_1d` / `launch_rows` wrappers in
  `crates/autograd/src/backend_cuda/kernels.rs` had drifted from the current
  `LaunchArgs` lifetime contract. Passing `&mut LaunchArgs<'_>` through a
  closure erased the relationship between the builder and the local kernel
  arguments, so every launch site failed with borrowed-data-escapes / `'static`
  lifetime errors. Fix: bind one explicit lifetime `'a` across
  `stream`, `func`, and `FnOnce(LaunchArgs<'a>) -> LaunchArgs<'a>`, then return
  the builder from each call site in `backend_cuda.rs`.

After those fixes, CUDA build + runtime validation succeeded on this box:

```bash
CARGO_TARGET_DIR=/tmp/agent-infer-target-cuda CUDA_HOME=/usr/local/cuda \
  cargo build -p train --release --features cuda \
  --bin pretrain_qwen3 --bin eval_lm
```

from-scratch Qwen3.5-family pretrain on CUDA:

```bash
CARGO_TARGET_DIR=/tmp/agent-infer-target-cuda CUDA_HOME=/usr/local/cuda \
  cargo run -p train --release --features cuda --bin pretrain_qwen3 -- \
  --model-family qwen35 \
  --corpus /tmp/agent-infer-train/corpus.txt \
  --tokenizer /tmp/qwen3-tokenizer.json \
  --out /tmp/agent-infer-train/pretrain-cuda \
  --steps 6 --batch 1 --seq 16 --lr 1e-3 \
  --log-every 1 --save-every 3 --backend cuda \
  --hidden 64 --layers 2 --heads 4 --kv-heads 2 --head-dim 16 \
  --intermediate 128 --max-pos 64
```

Observed:

- backend: `Cuda`
- family: `qwen35`
- vocab: `151669`
- params: `9,789,120`
- step 1 loss: `11.940998`
- step 6 loss: `11.357897`
- checkpoints published at `step_000003` and `step_000006`
- `/tmp/agent-infer-train/pretrain-cuda/latest -> step_000006`

checkpoint reload through `eval_lm` on CUDA:

```bash
CARGO_TARGET_DIR=/tmp/agent-infer-target-cuda CUDA_HOME=/usr/local/cuda \
  cargo run -p train --release --features cuda --bin eval_lm -- \
  --model-path /tmp/agent-infer-train/pretrain-cuda/latest \
  --data /tmp/agent-infer-train/eval.chat.jsonl \
  --seq-len 32 --backend cuda
```

Result:

- loss: `11.937930`
- ppl: `152959.800651`
- tokens: `27`

resume on CUDA also worked:

```bash
CARGO_TARGET_DIR=/tmp/agent-infer-target-cuda CUDA_HOME=/usr/local/cuda \
  cargo run -p train --release --features cuda --bin pretrain_qwen3 -- \
  --model-family qwen35 \
  --corpus /tmp/agent-infer-train/corpus.txt \
  --tokenizer /tmp/qwen3-tokenizer.json \
  --out /tmp/agent-infer-train/pretrain-cuda-resume \
  --steps 7 --batch 1 --seq 16 --lr 1e-3 \
  --log-every 1 --save-every 7 --backend cuda \
  --hidden 64 --layers 2 --heads 4 --kv-heads 2 --head-dim 16 \
  --intermediate 128 --max-pos 64 \
  --resume /tmp/agent-infer-train/pretrain-cuda/latest
```

Observed:

- resumed from `step_000006`
- trainer optimizer state restored
- continued through step `13`
- `/tmp/agent-infer-train/pretrain-cuda-resume/latest -> step_000013`

Backend verification after the fixes:

```bash
CARGO_TARGET_DIR=/tmp/agent-infer-target-cuda CUDA_HOME=/usr/local/cuda \
  cargo test -p autograd --release --features cuda --test test_backend

CARGO_TARGET_DIR=/tmp/agent-infer-target-cuda CUDA_HOME=/usr/local/cuda \
  cargo clippy -p autograd --release --features cuda --tests -- -D warnings
```

Results:

- `test_backend`: **38/38 passed**
- `clippy`: clean after deleting unused `CudaStorage::len()` and adding a
  narrow `#[allow(clippy::type_complexity)]` on the existing
  `matmul_backward_cases()` test helper.

Follow-up guardrails also landed for the SFT path:

- Added `scripts/train_cuda_e2e.sh`, a one-shot CUDA smoke wrapper that:
  downloads `Qwen/Qwen3-0.6B` into a local dir when missing, builds
  `train_sft + eval_lm + agent-infer`, writes tiny SFT/eval JSONL fixtures,
  then runs `train_sft -> eval_lm -> agent-infer -> resume`.
- Added an exact resume-consistency test to `crates/train/src/bin/train_sft.rs`
  that runs a tiny Qwen3 LoRA SFT job continuously and as `2 + resume-to-4`,
  then proves parity by comparing:
  - `model.safetensors`,
  - `adapter_model.safetensors`,
  - `trainer_state.json`,
  - loaded AdamW optimizer state tensor-by-tensor (`m` / `v` bitwise).
- Verification for those follow-ups:
  - `cargo test -p train --release --bin train_sft`: **5/5 passed**
  - `cargo clippy -p train --release --bin train_sft --tests -- -D warnings`:
    clean
  - `bash -n scripts/train_cuda_e2e.sh`: clean

The one thing that still depended on machine state, not repository state, was
the "real" `Qwen3-0.6B` CUDA SFT smoke on this shared L4. The new wrapper was
able to download the model and finish all builds, but the first actual
`train_sft --backend cuda` failed with `cuda htod copy failed` because an
already-running `infer` server on the same GPU held ~21.5 GiB of the card,
leaving only ~760 MiB free. The failure reproduced even against the tiny
9.8M-parameter local checkpoint, which confirmed the blocker was shared GPU
headroom rather than the SFT code path. Keep `scripts/train_cuda_e2e.sh` as
the rerun command once the box is idle.

## Rule

When `cudarc` moves its launch-builder API, the first breakage may look like a
missing trait import, but the real contract lives in the builder lifetime:
keep one explicit lifetime threaded through `CudaStream`, `CudaFunction`, and
the `LaunchArgs`-building closure, or every kernel argument will collapse into
borrowed-data-escapes / `'static` noise.

For CUDA train verification, don't stop at "it compiles". The minimal useful
acceptance stack is:

1. pretrain on CUDA with real optimizer steps,
2. publish a checkpoint + `latest`,
3. reload the checkpoint through `eval_lm`,
4. resume from the checkpoint,
5. add a one-command rerun harness for the next shared-machine validation,
6. prove resume parity with an exact small-model test,
7. rerun CUDA parity tests and clippy.
