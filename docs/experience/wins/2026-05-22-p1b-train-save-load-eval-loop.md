# P1-B: Qwen3.5 LoRA Train Save Load Eval Loop

## Context

P1-A (`a1ad570`) added OPD LoRA adapter saving for Qwen3.5 students, but
serve could not load the saved Qwen3.5 adapter. The gap blocked the user-visible
loop: train a student, save the LoRA adapter, load it in `arle serve`, then run
capability eval against the pre-distillation baseline.

This tranche added Qwen3.5 serve-side LoRA loading in `584f07b` and validated the
full loop with a 4B teacher -> 0.8B LoRA student pilot.

## Setup

- Teacher: `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-4B`
- Student base: `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base`
- Adapter: LoRA r=16, alpha=32, q/v adapters on Qwen3.5 full-attention layers
- Training: 2000 steps, rollout_len=8, lr=2e-5, prompt_max_tokens=16
- Prompts: `examples/opd/sample-prompts.jsonl`
- Eval: `scripts/arle_capability_eval.py`, 200-sample MMLU + GSM8K
- Artifacts:
  - `bench-output/2026-05-22-p1b-distill-pilot/run.txt`
  - `runs/2026-05-22-p1b-distill-pilot/final/`
  - `bench-output/2026-05-22-capability-after-distill-final/`

## Results

OPD KL moved in the intended direction while writing usable checkpoints:

| Step | Train KL | Held-out KL |
| ---: | ---: | ---: |
| 0 | 1.510384544190e-5 | 1.739055323924e-5 |
| 500 | 1.406700874895e-5 | 1.606478099347e-5 |
| 1000 | 1.357229820087e-5 | 1.597982964086e-5 |
| 2000 | 1.317703839732e-5 | 1.598908033884e-5 |

The run wrote `step_001000/`, `step_002000/`, and `final/` PEFT adapter
directories. `final/` loaded through `INFER_LORA_PATH`, and the serve smoke
returned coherent English rather than the earlier long-prompt corruption mode.

Capability eval produced valid before/after rows:

| Label | Backend | Model | GSM8K | MMLU |
| --- | --- | --- | --- | --- |
| base | ? | Qwen3___5-0___8B-Base | 1.5% (3/194, inv 6) | 51.4% (73/142, inv 29) |
| after-2k-distill | arle | Qwen3___5-0___8B-Base | 1.6% (3/188, inv 12) | 50.0% (83/166, inv 5) |
| teacher-4b | ? | Qwen3___5-4B | 2.5% (5/198, inv 2) | 77.3% (116/150, inv 21) |
| Delta after-2k-distill - base | | | +0.05pp | -1.41pp |
| Delta teacher-4b - base | | | +0.98pp | +25.92pp |

## What Worked

- `INFER_LORA_PATH` now works for Qwen3.5 CUDA serve, not only Qwen3.
- The loader parsed the saved PEFT adapter (`24` tensors) and merged q/v LoRA
  deltas into the Qwen3.5 full-attention dense weights.
- The end-to-end loop is now closed: OPD train -> PEFT adapter save -> serve
  load -> capability eval -> before/after compare.
- The pilot did not produce a capability win at 2k steps, but it produced valid
  measurement instead of a tooling blocker.

## Verification

- `cargo check -p infer --no-default-features --features cuda,no-cuda`
- `cargo test -p infer --lib qwen35::lora --release --features cuda`
- Serve smoke with
  `INFER_LORA_PATH=runs/2026-05-22-p1b-distill-pilot/final`
- 200-sample MMLU + GSM8K capability eval on the loaded adapter

## Rule

Checkpoint support is not complete until the saved artifact can be loaded by the
serving path and evaluated with the same capability harness as the base model.
KL-only improvement is useful substrate evidence; capability claims require the
train -> save -> load -> eval loop to produce valid before/after numbers.
