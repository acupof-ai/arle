# 2026-05-22 — ARLE serve long-prompt corruption: cross-model evidence

> **Status:** research note. Consolidates the surprise finding from the
> P0 capability-eval retry that ARLE serve produces broken output on
> long prompts — *not* just for one quantized model variant, but for
> every Qwen3.5 base model tested so far. This reframes the "9B
> GPTQModel `!` collapse" from an isolated quant-loader oddity into a
> serve-path bug that's blocking capability eval entirely.

## TL;DR

ARLE `arle serve` (CUDA backend, `/v1/completions`) produces broken
output on long prompts for **every Qwen3.5 model checkpoint tested**,
regardless of dtype or quantization:

| Model | Prompt shape | Output | Date evidence |
|---|---|---|---|
| Qwen3.5-9B-GPTQ-4bit | short (1 token / generic) | echo only — exclamation mark | `bench-output/2026-05-22-qwen35-9b-gptqmodel-serve-recheck-after-opd-kill-2/` |
| **Qwen3.5-4B-BF16** | MMLU 5-shot (~2000 tok) | **literal `!!!!!!!!!!...` × max_tokens** | `bench-output/2026-05-22-capability-baseline-4b-retry/mmlu_debug.json` |
| **Qwen3.5-0.8B-Base** | MMLU 5-shot (~2000 tok) | **garbage tokens** (random Unicode + numbers) | `bench-output/2026-05-22-capability-baseline-08b-retry/mmlu_debug.json` |
| Qwen3.5-0.8B-Base | GSM8K 3-shot (~600 tok) | **mostly parseable** (185/200 scored) but only 1% correct | `bench-output/2026-05-22-capability-baseline-08b/gsm8k.json` |

Pattern: short prompts produce something parseable (even if low-quality);
long prompts produce structured garbage (`!` saturation for 4B/9B,
random-Unicode noise for 0.8B). The 9B `!`-only output was previously
written off as a quantized-loader artifact; this run proves it's a
broader serve bug.

## Evidence — verbatim debug samples

### Qwen3.5-4B-BF16 on MMLU (5-shot, ~2 000 tokens)

```json
{
  "i": 0,
  "subject": "abstract_algebra",
  "gold": "B",
  "extracted": null,
  "response_first_200": "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
}
```

171/171 samples produced this exact `!` × 32 output. Different prompts
(different subjects, different questions) all collapse to the same
single-token saturation.

### Qwen3.5-0.8B-Base on MMLU (5-shot, ~2 000 tokens)

```json
{
  "i": 0,
  "response_first_200": "zBuilding\n\n(111?\n then about? Whatart???\n\n\n\n\n  \n\n, 33&m"
}
{
  "i": 1,
  "response_first_200": "ّعité二.\n\n What\n1  \nM8/\n50's、5505555551j15"
}
```

Garbage tokens — Chinese / Arabic / random ASCII mixed with
non-coherent numbers. Different pattern from 4B but same root: the
model is sampling from a distribution that has no relationship to the
prompt structure.

### Qwen3.5-0.8B-Base on GSM8K (3-shot, ~600 tokens) — partially works

185/200 samples produce parseable numeric output (just mostly wrong).
This was the original short-prompt baseline that we DID get scores
from. The contrast with MMLU is informative.

## Hypotheses (in order of likelihood)

Listed with concrete experiments each implies. None are licensed yet;
this note is the evidence base for choosing which to investigate.

### H1: positional embedding / RoPE drift past some length threshold

- The 4B BF16 saturates to `!` (an early-vocab token) at long prompts.
- The 0.8B garbage looks like sampling from a tail distribution.
- Both fail in a way consistent with attention layer producing
  near-random outputs after a length threshold.
- **Experiment:** vary prompt length 100 / 500 / 1 000 / 1 500 / 2 000
  tokens and bisect where the failure starts. The threshold tells us
  which mechanism — RoPE wrapping vs. cache management vs. max_seq_len
  truncation.

### H2: chunked prefill miscalculation

- The repro command uses `--chunked-prefill-size 256` with prompts
  much longer than that.
- If chunked prefill mis-computes positional offsets at chunk
  boundaries, long prompts get progressively-broken KV.
- **Experiment:** rerun with `--chunked-prefill-size 4096` (single-chunk)
  for the same prompts. If the corruption disappears, H2 is confirmed.

### H3: max_seq_len 1024 truncation silently dropping the question

- Command sets `--max-seq-len 1024` but MMLU 5-shot prompts are
  ~2 000 tokens.
- If ARLE truncates from the front (keeping the recent tokens), the
  model sees a partial prompt with no "Answer:" trailer.
- **Experiment:** rerun with `--max-seq-len 4096`. If MMLU starts
  producing letters, H3 is confirmed.
- This is the **cheapest experiment** and should run first.

### H4: BOS / chat template mismatch

- Qwen3.5 base models expect a specific BOS token; ARLE serve may
  prepend a different one.
- **Experiment:** call `/v1/completions` with a known-short prompt
  that does work, then compare the prefix tokens against the
  HuggingFace reference tokenizer.

### H5: KV cache buffer corruption (paged KV bug)

- Less likely given OPD's `forward_token_logits` path works for the
  same models — but `forward_token_logits` is a different code path
  that doesn't use the serving scheduler.
- **Experiment:** disable paged KV (if a flag exists) or use
  `--num-slots 1 --max-seq-len 2048` with no paging.

## Why OPD didn't catch this before

OPD's `train::OpdStep` calls `LoadedInferenceEngine::forward_token_logits`
**directly**, not via HTTP. This API:

- bypasses the serving scheduler entirely
- runs a single forward pass against the in-process engine
- returns logits without going through the autoregressive decode loop

The capability eval is the **first system that exercises long-prompt
autoregressive decode** at scale through ARLE serve. It immediately
hit a structural bug that affects every Qwen3.5 model the project
has tested.

This makes the bug high-priority for:

1. **Capability eval P0** — blocked until fixed
2. **External `ApiTeacher` OPD path** (recommended in
   `2026-05-22-opd-9b-teacher-bf16-cycle-wrap.md` as the only viable
   9B-teacher path on 16 GB) — would route through the same serve
   path
3. **End-user inference** — anyone using `arle serve` for chat
   with prompts > 1 K tokens hits this

## Recommended next step

Run H3 first (it's free):

```bash
./target/release/arle serve --backend cuda \
  --model-path /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-4B \
  --port 8123 -- \
  --num-slots 1 --max-seq-len 4096 --chunked-prefill-size 4096 \
  --max-num-batched-tokens 4096

# then re-run scripts/arle_capability_eval.py MMLU on n_samples=5
```

If still `!` × N → H1 or H2. If now produces letters → H3 confirmed,
the eval plan just needs `--max-seq-len 4096` for MMLU runs.

## Cross-links

- 9B GPTQModel `!` collapse (the first surface of this bug, written off as quant-specific): `docs/experience/errors/2026-05-22-qwen35-9b-gptqmodel-generation-f32load-fix-kill.md`
- 9B serve recheck (1-token-only worked, didn't catch long-prompt path): `docs/research/2026-05-22-qwen35-9b-gptqmodel-live-recheck.md`
- Capability eval P0 kill (this find's immediate trigger): `docs/experience/errors/2026-05-22-capability-baseline-triplet-harness-kill.md`
- Capability eval plan (blocked by this find): `docs/plans/2026-05-22-arle-opd-capability-eval-plan.md`
- ARLE OPD pipeline manual (needs an "OPD bypasses serve" callout): `docs/projects/2026-05-21-arle-opd-cuda-usage-manual.md`
