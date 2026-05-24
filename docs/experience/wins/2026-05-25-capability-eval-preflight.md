# Capability Eval Harness Preflight

Related: `docs/projects/2026-05-24-opd-mainline-task-backlog.md` T12 and
`docs/projects/2026-05-22-eod-opd-cycle-wrap.md`.

## Context

P5 pure OPD is still running under `runs/2026-05-24-p5-pure-opd-5k/`.
The next GPU window needs to launch capability eval immediately after the
`step_005000` adapter lands, without discovering harness or path bugs at that
point.

This preflight stayed CPU-only: no ARLE serve process, no inference, no P5
checkpoint mutation.

## Harness

`scripts/arle_capability_eval.py` is an OpenAI-v1 HTTP eval client, not a model
loader. It does not accept `--lora-path`; the LoRA adapter must be loaded by
the serving process through `INFER_LORA_PATH`.

Supported CLI shape:

```text
python scripts/arle_capability_eval.py \
  --backend arle \
  --base-url http://127.0.0.1:8125 \
  --model-id Qwen3___5-0___8B-Base \
  --tasks mmlu,gsm8k \
  --n-samples 200 \
  --output <dir>
```

MMLU is fixed 5-shot inside the harness; there is no `--n-shot` flag.

## Checkpoint Shape

`crates/train/src/qwen35_checkpoint.rs` saves Qwen3.5 LoRA checkpoints as
PEFT adapter directories:

| File | Purpose |
| --- | --- |
| `adapter_model.safetensors` | LoRA tensors with PEFT keys |
| `adapter_config.json` | base model path, `peft_type=LORA`, rank/alpha, target modules |
| `config.json` | copied base HF config |
| `generation_config.json` | copied or synthesized generation config |
| `tokenizer.json` | copied tokenizer when present |

`crates/train/examples/opd_step_cuda_infer_teacher_train.rs` writes
`step_%06d/` every `--save-every` steps and writes `final/` after the loop.
For P5, use the explicit `step_005000/` directory for the checkpointed eval
table instead of `latest` because `latest` may move to `final/` after training.

Current P5 observed shape:

| Directory | Verdict |
| --- | --- |
| `step_001000/` | all five expected files present |
| `step_002000/` | all five expected files present |
| `latest` | currently points at `step_002000` |

Both existing adapter configs report:

```text
base_model_name_or_path=/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base
peft_type=LORA
r=16
lora_alpha=32.0
target_modules=["q_proj", "v_proj"]
```

This matches the Qwen3.5 serve-side `INFER_LORA_PATH` loader documented in
`docs/experience/wins/2026-05-22-qwen35-lora-serve-load.md`.

## Dry-Run Equivalent

The harness has no explicit `--dry-run` flag, so the preflight used import,
argument, unit-test, path, and offline dataset-cache checks.

```bash
.venv/bin/python -m py_compile scripts/arle_capability_eval.py scripts/arle_capability_compare.py
.venv/bin/python scripts/arle_capability_eval.py --help
.venv/bin/python -m unittest scripts.tests.test_arle_capability_eval -v
HF_DATASETS_OFFLINE=1 .venv/bin/python - <<'PY'
from datasets import load_dataset
for name, cfg, split in [
    ("cais/mmlu", "all", "test[:1]"),
    ("cais/mmlu", "all", "dev[:1]"),
    ("openai/gsm8k", "main", "test[:1]"),
]:
    ds = load_dataset(name, cfg, split=split)
    print(name, cfg, split, len(ds), ds.column_names)
PY
```

Results:

- Eval parser/factory tests: 29 passed.
- MMLU cache hit:
  `/home/ckl/.cache/huggingface/datasets/cais___mmlu/all/0.0.0/c30699e8356da336a370243923dbaf21066bb9fe`
- GSM8K cache hit:
  `/home/ckl/.cache/huggingface/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866`
- Base model path exists:
  `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base`
- Release binary exists:
  `target/release/arle`

## Post-P5 Command

Ready-to-paste after `runs/2026-05-24-p5-pure-opd-5k/step_005000/adapter_model.safetensors`
exists:

```bash
ADAPTER=runs/2026-05-24-p5-pure-opd-5k/step_005000 BASE=/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base OUT=bench-output/2026-05-25-p5-pure-opd-step5000-capability PORT=8125; set -euo pipefail; mkdir -p "$OUT"; test -f "$ADAPTER/adapter_model.safetensors"; test -f "$ADAPTER/adapter_config.json"; INFER_LORA_PATH="$ADAPTER" ./target/release/arle serve --backend cuda --model-path "$BASE" --port "$PORT" -- --num-slots 1 --max-seq-len 4096 --chunked-prefill-size 4096 --max-num-batched-tokens 4096 >"$OUT/serve.log" 2>&1 & server=$!; trap 'kill "$server" 2>/dev/null || true' EXIT; until curl -fsS "http://127.0.0.1:${PORT}/readyz" >/dev/null; do sleep 2; done; .venv/bin/python scripts/arle_capability_eval.py --backend arle --base-url "http://127.0.0.1:${PORT}" --model-id Qwen3___5-0___8B-Base --tasks mmlu,gsm8k --n-samples 200 --output "$OUT"; kill "$server"; trap - EXIT
```

T14 found that sequential checkpoint sweeps should launch `target/release/infer`
directly or kill the full process group; killing only the `arle serve` wrapper
can leave the backend child alive. See
`docs/experience/wins/2026-05-25-p5-pure-opd-5k-capability-sweep.md`.

## Rule

For OPD capability eval, `INFER_LORA_PATH` belongs to the server launch, not
the eval client. The eval client only needs `--base-url`, `--model-id`,
`--tasks`, `--n-samples`, and `--output`; checkpoint shape validation should
happen before spending the GPU window.
