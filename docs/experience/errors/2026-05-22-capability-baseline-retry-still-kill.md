# Capability Baseline Retry KILL: Fixed Extractor Still Sees Invalid Outputs

## Context

P0-retry re-ran the capability baseline after `d3eb295` repaired the eval
harness:

- MMLU completions now allow `max_tokens=32`.
- The MMLU extractor accepts leading letters, parenthesized letters,
  `answer is X`, and a short-window fallback.
- Each task saves the first five raw responses to `{task}_debug.json`.

The goal was still the P0 baseline from
[`docs/plans/2026-05-22-arle-opd-capability-eval-plan.md`](../../plans/2026-05-22-arle-opd-capability-eval-plan.md):
establish Qwen3.5-0.8B-Base and Qwen3.5-4B MMLU/GSM8K floors and ceilings
through `arle serve`.

License threshold:

- PASS: both models and both tasks have invalid rate <30%, and 4B accuracy is
  above 0.8B accuracy on both tasks.
- KILL: invalid rate remains >50%, or 4B remains 100% invalid.

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
  --output bench-output/2026-05-22-capability-baseline-08b-retry
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
  --output bench-output/2026-05-22-capability-baseline-4b-retry
```

Raw artefacts:

- `bench-output/2026-05-22-capability-baseline-08b-retry/`
- `bench-output/2026-05-22-capability-baseline-4b-retry/`

## Results

| Model | Task | Actual samples | Scored | Invalid | Invalid rate | Correct | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B-Base | MMLU | 171 | 48 | 123 | 71.93% | 15 | 31.25% scored / 8.77% raw |
| Qwen3.5-0.8B-Base | GSM8K | 200 | 183 | 17 | 8.50% | 2 | 1.09% scored / 1.00% raw |
| Qwen3.5-4B | MMLU | 171 | 0 | 171 | 100.00% | 0 | 0.00% |
| Qwen3.5-4B | GSM8K | 200 | 0 | 200 | 100.00% | 0 | 0.00% |

GPU peaks from `nvidia-smi-monitor.csv`:

| Model | Peak GPU memory |
|---|---:|
| Qwen3.5-0.8B-Base | 15095 MiB |
| Qwen3.5-4B | 13241 MiB |

The retry did not produce a usable 4B - 0.8B capability gap. The extractor
fix improved 0.8B MMLU from 4 scored samples to 48 scored samples, but MMLU is
still invalid-heavy. 4B remained 100% invalid on both tasks.

## Debug Evidence

The new debug files show this is not a simple answer-extractor miss. The raw
outputs are malformed for 0.8B and repeated punctuation for 4B, not normal
`(A)`, `The answer is C`, or `A. ...` shapes.

0.8B MMLU first three debug samples:

```text
gold=B extracted=null response="zBuilding\n\n(111?\n then about? Whatart???\n\n\n\n\n  \n\n, 33&m"
gold=C extracted=null response="ّعité二.\n\n What\n1  \nM8/\n50's、5505555551j15"
gold=D extracted=A    response="浦SL\",npmf� a|的，,tr|\n\nbpa， (Symec,IAHA. is.2c73,"
```

0.8B GSM8K first three debug samples:

```text
gold=18    extracted=1 response starts "T\n\n/，yl\n  W  C11. \n1 (1下_ 14-310ththe1?<:德，..."
gold=3     extracted=2 response starts "第一人isc\n)1\n0NA12C **xall0-\n是因为 7S ) -1,..."
gold=70000 extracted=2 response starts "《 Under53(-: ed< s1·[1 \n\n\n\n\n边 //10ED 2\n\n,..."
```

4B MMLU first three debug samples:

```text
gold=B extracted=null response="!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
gold=C extracted=null response="!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
gold=D extracted=null response="!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
```

4B GSM8K first three debug samples:

```text
gold=18    extracted=null response starts "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!..."
gold=3     extracted=null response starts "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!..."
gold=70000 extracted=null response starts "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!..."
```

## Root Cause

The d3eb295 extractor fix is not enough because the served model outputs are
not valid task responses:

- 0.8B emits mixed garbled text for MMLU and GSM8K. The GSM8K extractor often
  finds a number, but the numbers are not useful task answers.
- 4B emits repeated `!` for all captured samples on both tasks.
- The HTTP surface and datasets are working; the failure is downstream of
  request routing and upstream of answer extraction.

For 4B specifically, persistent 100% invalid output points to an ARLE
Qwen3.5-4B serving/model-output problem or a base-checkpoint prompting problem
that must be isolated before capability eval can be trusted.

## Kill Decision

KILL. The retry still fails the validity gate:

- 0.8B MMLU invalid rate is 71.93% (>50%).
- 4B MMLU and GSM8K invalid rates are both 100%.
- 4B accuracy cannot serve as a teacher ceiling when no sample is valid.

Do not use these numbers as a capability baseline or an OPD gap.

## Next Step

Before another full P0 run:

1. Run a positive-control instruct checkpoint through the same `arle serve`
   + harness path. If it produces normal answers, this is base-model/prompt
   mismatch.
2. Run direct PyTorch/HF generation on the same prompts for Qwen3.5-0.8B-Base
   and Qwen3.5-4B. If PyTorch is coherent and ARLE is not, this is an ARLE
   serve/model bug.
3. Add a focused ARLE generation smoke for Qwen3.5-4B with fixed short prompts
   before capability scoring. The repeated-`!` pattern should be debugged
   independently of MMLU/GSM8K.
4. Only after raw generations are coherent, re-run the n=200 capability
   baseline.

## Rule

Extractor generalization is not a substitute for raw-output validity. For
capability baselines, inspect debug samples first; if the model emits malformed
text or repeated punctuation, stop at a serving/prompting diagnosis instead of
turning invalid responses into accuracy claims.
