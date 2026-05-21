# OPD example — one-click

`./run-distillation.sh` runs an On-Policy Distillation training step through
ARLE's `train` runtime. Two modes:

## Smoke (default — no download, no GPU strictly required)

```bash
./examples/opd/run-distillation.sh
```

What this does:

- Builds `arle` (release) if not already built — CUDA if `nvcc` is on PATH,
  otherwise CPU-only.
- Runs `arle train opd --smoke` on the embedded tiny Qwen3.5 config.
- Emits per-step `loss / grad_norm / lr` as JSON.

Runtime: < 30 s on a recent laptop. Output lands under
`opd-output/<timestamp>/run.txt`.

This path is for "can I even build and run ARLE OPD?" smoke validation —
the loss curve uses the embedded config, not a real model.

## Real distillation (downloads HF/ModelScope models)

```bash
ARLE_TEACHER=Qwen/Qwen3.5-4B \
ARLE_STUDENT=Qwen/Qwen3.5-0.8B-Base \
ARLE_STEPS=500 \
ARLE_LR=1e-5 \
./examples/opd/run-distillation.sh
```

What this does:

- Resolves model IDs to local cache via `modelscope` (default) or
  `huggingface_hub` Python (`ARLE_SOURCE=huggingface` to switch).
- Auto-installs the resolver package into `./.venv` if missing.
- Calls `arle train opd --teacher-model <dir> --student-model <dir>`.

Downloads land in `~/.cache/modelscope/hub` or `~/.cache/huggingface/hub`
depending on source and resume on retry. First Qwen3.5-4B run pulls ~7 GB.

## Environment overrides

| Var | Default | Notes |
|---|---|---|
| `ARLE_TEACHER` | unset (= smoke mode) | HF / ModelScope ID. Set BOTH this and `ARLE_STUDENT` to enter real mode. |
| `ARLE_STUDENT` | unset (= smoke mode) | HF / ModelScope ID. |
| `ARLE_SOURCE` | `modelscope` | `modelscope` or `huggingface`. |
| `ARLE_STEPS` | `5` | Training steps. |
| `ARLE_ROLLOUT_LEN` | `8` | Greedy student rollout tokens per step. |
| `ARLE_LR` | `1e-4` | AdamW learning rate. For LoRA on a 0.8B student vs 4B teacher, `1e-5` works well. |
| `ARLE_GRAD_CLIP` | `1.0` | L2-norm gradient clip. |
| `ARLE_BACKEND` | `auto` | `auto` (CUDA when built with cuda feature) or `cpu`. |
| `ARLE_VENV` | `./.venv` | Python venv that hosts the model resolver. |
| `ARLE_OUTPUT_DIR` | `./opd-output/<timestamp>` | Where `run.txt` lands. |

## Prompts

`sample-prompts.jsonl` ships 20 short real-text prompts and is referenced by
the example train binaries. The current `arle train opd` CLI accepts only
`--prompt-ids` (comma-separated token IDs); a first-class `--prompts-file`
flag is tracked as Phase 7 of the
[Qwen3.5-9B→0.8B distillation plan](../../docs/plans/2026-05-21-arle-opd-qwen35-9b-to-08b-distillation-plan.md).

## Expected output (smoke)

```json
{
  "step_metrics": [
    {"step": 1, "loss": 0.173..., "grad_norm": ..., "lr": 1e-4, "rollout_len": 11},
    ...
  ],
  "summary": {"final_loss": ..., "step_count": 5, ...}
}
```

If loss doesn't decrease, that's expected for the smoke config — student
and teacher start identical. The point of smoke is to verify the build
and the OPD step itself runs without crashing, not to demonstrate
distillation. Real distillation (next section) is where the loss curve
matters.

## Troubleshooting

- **`arle: command not found`** — script builds `arle` automatically on
  first run; if it failed, check the build log. CUDA builds need `nvcc`,
  `g++-14` (override via `NVCC_CCBIN`), and matching `CUDARC_CUDA_VERSION`
  / `TORCH_CUDA_ARCH_LIST`.
- **`modelscope` install fails** — try `ARLE_SOURCE=huggingface` instead, or
  install manually: `.venv/bin/pip install modelscope`.
- **Resolver hangs** — ModelScope sometimes throttles; cancel and retry,
  or switch to `huggingface`.
- **OOM during real distillation** — start with the smoke path to confirm
  build is sane, then drop `ARLE_ROLLOUT_LEN` to 4 or pick a smaller
  teacher (Qwen3.5-0.6B teacher → Qwen3.5-0.6B student self-distill works
  on a 4 GB GPU).
