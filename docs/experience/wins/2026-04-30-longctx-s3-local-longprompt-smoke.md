# Longctx S3 Local Long-Prompt Smoke

## Context

After the L4 CUDA bring-up, Phase 1 S3 needed at least a real 32k prompt
non-degeneracy check against the ARLE FP8 server before moving to canonical
GuideLLM S5.

The full S3 comparison matrix still requires simultaneously available FP8,
BF16, and SGLang BF16 services. On this single L4 run, the canonical FP8
server already used about 19.1 GB of 23.0 GB, leaving about 3.5 GB free, so a
second Qwen3-4B service could not be launched beside it.

## What Worked

Server under test:

```bash
/tmp/arle-target/release/infer \
  --model-path infer/models/Qwen3-4B \
  --port 8000 \
  --kv-cache-dtype fp8 \
  --num-slots 16 \
  --max-seq-len 131072 \
  --mem-fraction-static 0.85 \
  --max-num-batched-tokens 16384 \
  --max-prefill-tokens 16384 \
  --schedule-policy fcfs
```

S3 command:

```bash
python3 scripts/longctx_numerical_gate.py \
  --label arle-fp8-longprompt-e2e-local \
  --left-name arle-fp8 \
  --left-url http://127.0.0.1:8000 \
  --tokenizer infer/models/Qwen3-4B \
  --prompt-count 1 \
  --prompt-tokens 32768 \
  --max-tokens 64 \
  --ignore-eos \
  --out-dir bench-output/2026-04-30-longctx-s3-arle-fp8-longprompt-e2e-local
```

Result: pass.

| metric | value |
|---|---:|
| mode | single |
| pairs | 1 |
| ok_pairs | 1 |
| failed_pairs | 0 |
| nonempty_rate | 1.0 |
| status | pass |

Artefacts:

- Raw summary:
  `bench-output/2026-04-30-longctx-s3-arle-fp8-longprompt-e2e-local/summary.json`
- Markdown summary:
  `bench-output/2026-04-30-longctx-s3-arle-fp8-longprompt-e2e-local/summary.md`

## Rule

Single-target 32k S3 smoke is enough to catch context-window clamp and empty
decode failures, but not enough to claim FP8 numerical parity. Token-trajectory
comparison still needs separate FP8/BF16/SGLang services on enough GPU memory
or multiple hosts.
