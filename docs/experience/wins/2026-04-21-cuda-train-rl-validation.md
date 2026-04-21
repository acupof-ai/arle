# CUDA RL train validation on L4

## Context

The earlier 2026-04-21 CUDA train validation had already covered the
supervised side of the stack:

- `pretrain_qwen3(cuda) -> checkpoint/latest -> eval_lm(cuda) -> resume(cuda)`
- `train_sft(cuda) -> eval_lm(cuda) -> agent-infer(cuda) -> resume(cuda)`

The remaining question was the RL side: which train binaries are real CUDA
surfaces today, and which ones are only documented as such.

## What Worked

The gap found earlier that day was then closed in code: `train_grpo` now
matches the other train binaries by exposing `--backend cpu|metal|cuda` and
constructing `TensorStore::with_backend(...)` instead of implicitly using the
default CPU store.

First, the RL binaries stayed green under a CUDA-feature build after that
surface change:

```bash
cargo build --release --features cuda -p train \
  --bin train_grpo --bin train_multi_turn --bin eval_lm

cargo test -p train --release --features cuda \
  --bin train_grpo --bin train_multi_turn

cargo clippy -p train --release --features cuda \
  --bin train_grpo --bin train_multi_turn -- -D warnings
```

Observed:

- `train_grpo` bin tests: `4/4 passed`
- `train_multi_turn` bin tests: `3/3 passed`
- `clippy`: clean

`train_grpo` then passed a real CUDA end-to-end smoke on the synthetic dense
Qwen3.5 path.

Fresh run on CUDA:

```bash
target/release/train_grpo \
  --backend cuda \
  --model-family qwen35 \
  --sft-steps 1 \
  --grpo-iters 2 \
  --batch-prompts 2 \
  --group-size 2 \
  --seq 16 \
  --seed 7 \
  --save-path /tmp/train-grpo-cuda \
  --save-every 1
```

Observed:

- backend: `cuda`
- `checkpoint saved to /tmp/train-grpo-cuda/step_000001`
- `checkpoint saved to /tmp/train-grpo-cuda/step_000002`
- `/tmp/train-grpo-cuda/latest -> step_000002` after the fresh run

Checkpoint reload through `eval_lm` on CUDA:

```bash
target/release/eval_lm \
  --backend cuda \
  --model-path /tmp/train-grpo-cuda/latest \
  --data /tmp/train-grpo-eval.jsonl \
  --seq-len 16
```

Result:

- loss: `5.607285`
- ppl: `272.403533`
- tokens: `5`

Exact resume on CUDA:

```bash
target/release/train_grpo \
  --backend cuda \
  --model-family qwen35 \
  --sft-steps 1 \
  --grpo-iters 4 \
  --batch-prompts 2 \
  --group-size 2 \
  --seq 16 \
  --seed 7 \
  --save-path /tmp/train-grpo-cuda \
  --save-every 1 \
  --resume-from /tmp/train-grpo-cuda/latest
```

Observed:

- resumed from `step_000002`
- restored `28` optimizer entries
- advanced to `step_000004`
- `/tmp/train-grpo-cuda/latest -> step_000004`

The shared train control plane also worked on a live CUDA `train_grpo` run:

```bash
target/release/train_grpo \
  --backend cuda \
  --model-family qwen35 \
  --sft-steps 1 \
  --grpo-iters 200 \
  --batch-prompts 2 \
  --group-size 2 \
  --seq 16 \
  --seed 13 \
  --save-path /tmp/train-grpo-serve \
  --serve 19082
```

While that run was active, all four endpoints behaved correctly:

- `GET /v1/train/status` returned a live JSON snapshot (`iter=6`,
  `finished=false`, `dropped_metrics=0`)
- `GET /v1/train/events` returned the buffered `run_start` + metric stream
  with `backend=cuda`
- `POST /v1/train/save` returned `{"save_requested":true}`
- `POST /v1/train/stop` returned `{"stop_requested":true}`

The operator save request flushed a real checkpoint on CUDA before the stop:

- `[train_grpo] checkpoint saved to /tmp/train-grpo-serve/step_000006`
- `/tmp/train-grpo-serve/latest -> step_000006`
- run ended with `status=stopped`, not a crash

`train_multi_turn` remained green on CUDA and now also has the checkpoint
reload contract explicitly verified through `eval_lm`.

Stepwise GRPO on CUDA, fresh run:

```bash
target/release/train_multi_turn \
  --backend cuda \
  --iters 2 \
  --group-size 2 \
  --agent-tokens 2 \
  --obs-tokens 2 \
  --turns 2 \
  --prompt-len 4 \
  --seed 7 \
  --save-path /tmp/train-mt-cuda
```

Observed:

- backend: `Cuda`
- objective: `stepwise-grpo`
- `checkpoint saved to /tmp/train-mt-cuda/step_000002`
- resumed cleanly through `/tmp/train-mt-cuda/latest` to `step_000004`

Sequence-level GSPO on CUDA:

```bash
target/release/train_multi_turn \
  --backend cuda \
  --iters 1 \
  --group-size 2 \
  --agent-tokens 2 \
  --obs-tokens 2 \
  --turns 2 \
  --prompt-len 4 \
  --seed 11 \
  --objective gspo \
  --save-path /tmp/train-mt-gspo
```

Observed:

- backend: `Cuda`
- objective: `gspo`
- `checkpoint saved to /tmp/train-mt-gspo/step_000001`

`train_multi_turn` checkpoint reload through `eval_lm` on CUDA:

```bash
target/release/eval_lm \
  --backend cuda \
  --model-path /tmp/train-mt-cuda/latest \
  --data /tmp/train-grpo-eval.jsonl \
  --seq-len 16
```

Result:

- loss: `3.494860`
- ppl: `32.945689`
- tokens: `5`

The shared train control plane also worked on a live CUDA `train_multi_turn`
run:

```bash
target/release/train_multi_turn \
  --backend cuda \
  --iters 100 \
  --group-size 2 \
  --agent-tokens 2 \
  --obs-tokens 2 \
  --turns 2 \
  --prompt-len 4 \
  --seed 13 \
  --save-path /tmp/train-mt-serve \
  --serve 19081
```

Observed:

- `GET /v1/train/status`, `GET /v1/train/events`, `POST /v1/train/save`,
  and `POST /v1/train/stop` all returned the expected JSON responses
- save flushed `/tmp/train-mt-serve/step_000035`
- `/tmp/train-mt-serve/latest -> step_000035`

## Rule

For RL validation, "CUDA-supported" now means more than a compile-clean CLI:

1. the binary exposes an explicit backend-selection surface,
2. a real CUDA run writes checkpoint dirs plus `latest`,
3. those checkpoint dirs reload through `eval_lm`,
4. `--resume-from latest` restores optimizer state and continues cleanly,
5. live `/v1/train/{status,events,save,stop}` control-plane behavior works on
   the same CUDA surface.

Documentation has to follow that executable proof, not the intended
architecture.
