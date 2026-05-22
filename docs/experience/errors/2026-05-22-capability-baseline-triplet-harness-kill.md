# Capability Baseline P0 KILL: Harness Did Not Produce Valid Task Scores

## Context

P0 from
[`docs/plans/2026-05-22-arle-opd-capability-eval-plan.md`](../../plans/2026-05-22-arle-opd-capability-eval-plan.md)
was intended to establish the capability floor and ceiling before OPD
distillation:

- Qwen3.5-0.8B-Base via `arle serve`
- Qwen3.5-4B via `arle serve`
- MMLU + GSM8K through `scripts/arle_capability_eval.py`
- 200 requested samples per task

License threshold:

- PASS: four accuracy numbers land and 4B - 0.8B gap is positive on both
  tasks.
- KILL: any task is below 10% accuracy or the harness/interface is not valid.

## Commands

0.8B:

```bash
./target/release/arle serve --backend cuda \
  --model-path /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base \
  --port 8123 -- \
  --num-slots 1 --max-seq-len 1024 --chunked-prefill-size 256

.venv/bin/python scripts/arle_capability_eval.py \
  --base-url http://localhost:8123 \
  --model-id Qwen3___5-0___8B-Base \
  --tasks mmlu,gsm8k --n-samples 200 \
  --output bench-output/2026-05-22-capability-baseline-08b
```

4B:

```bash
./target/release/arle serve --backend cuda \
  --model-path /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-4B \
  --port 8123 -- \
  --num-slots 1 --max-seq-len 1024 --chunked-prefill-size 256

.venv/bin/python scripts/arle_capability_eval.py \
  --base-url http://localhost:8123 \
  --model-id Qwen3___5-4B \
  --tasks mmlu,gsm8k --n-samples 200 \
  --output bench-output/2026-05-22-capability-baseline-4b
```

Raw artefacts:

- `bench-output/2026-05-22-capability-baseline-08b/`
- `bench-output/2026-05-22-capability-baseline-4b/`

## Results

Datasets loaded successfully with `.venv` `datasets==4.8.4`:

- `cais/mmlu`, config `all`
- `openai/gsm8k`, config `main`

The HTTP surface was reachable for both models. Both models served completion
requests and produced eval outputs. The failure is score validity.

| Model | Task | Requested | Actual samples | Scored | Invalid | Correct | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B-Base | MMLU | 200 | 171 | 4 | 167 | 1 | 25.000% scored / 0.585% raw |
| Qwen3.5-0.8B-Base | GSM8K | 200 | 200 | 185 | 15 | 2 | 1.081% |
| Qwen3.5-4B | MMLU | 200 | 171 | 0 | 171 | 0 | 0.000% |
| Qwen3.5-4B | GSM8K | 200 | 200 | 0 | 200 | 0 | 0.000% |

The 4B - 0.8B gap is not usable:

| Task | 0.8B accuracy | 4B accuracy | Gap |
|---|---:|---:|---:|
| MMLU | invalid-heavy: 1/4 scored | invalid: 0/0 scored | undefined |
| GSM8K | 2/185 scored | invalid: 0/0 scored | undefined |

GPU peaks:

| Model | Peak GPU memory |
|---|---:|
| Qwen3.5-0.8B-Base | 15095 MiB |
| Qwen3.5-4B | 13241 MiB |

## Root Cause

The P0 harness reached ARLE, but the generated outputs did not match the
strict extractors:

- MMLU extractor accepts only a leading `A`/`B`/`C`/`D`.
- GSM8K extractor accepts `#### N` or the last number.
- 4B produced zero valid samples on both tasks.
- 0.8B produced almost no valid MMLU samples and <10% GSM8K.

There is also a sampling mismatch in the harness: `--n-samples 200` produced
171 MMLU questions because the current code uses `200 // 57 = 3` examples per
subject and then truncates. That is acceptable for a smoke run, but it is not
the requested 200-sample task baseline.

Manual 4B probes confirm this is not an HTTP connectivity failure. The server
responded, but the output format was not suitable for the current evaluator:

```text
MMLU probe completion:
'": ethly ofa  your:: about  **e?st'

GSM8K probe completion:
' 16-7 = 7
A: 16-7 = 7
A: 16-7 = 7
...'
```

The GSM8K probe is syntactically parseable but wrong; the MMLU probe is not a
letter answer. In the full GSM8K prompt, 4B produced no parseable numeric
answers under this harness.

## Kill Decision

KILL. The run does not establish a credible 0.8B -> 4B capability gap. The
failure mode is the eval setup, not the OPD pipeline:

- The OpenAI-compatible HTTP path is reachable.
- The datasets load.
- The strict prompt/extractor combination is not producing valid task scores
  for these base checkpoints.

Using these numbers as floor/ceiling would be misleading. In particular, the
teacher cannot be treated as a ceiling when it scores 0/0 valid on both tasks.

## Fix

Next P0 should be a harness repair tranche before another full baseline run:

1. Record per-sample prompt, raw response, extracted answer, and gold answer.
2. Use chat/instruction formatting or a stricter "answer with only A/B/C/D"
   stop pattern for MMLU.
3. For MMLU, either fix the sampler to produce exactly 200 examples or record
   the exact emitted count as the canonical sample size.
4. For GSM8K, add response normalization and save invalid examples. If base
   models remain weak, run a known-good instruct model as a positive-control
   server before blaming OPD.
5. Re-run a small n=50 positive-control first, then scale back to n=200.

## Rule

Capability eval baselines need a validity gate before an accuracy gate. If
invalid outputs dominate, the measured "accuracy" is a harness-format result,
not a model capability result. Save raw responses early so the failure can be
attributed without rerunning the full eval.
