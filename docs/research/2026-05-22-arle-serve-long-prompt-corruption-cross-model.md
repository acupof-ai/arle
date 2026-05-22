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

## 2026-05-22 14:24 update — H3 partial-confirm + tight bisection

Actually ran H3 on Qwen3.5-0.8B-Base with `--max-seq-len 4096
--chunked-prefill-size 4096` and `--max-num-batched-tokens 4096`.
Result: **H3 partial-confirm** — output shape changed but bug remains.

- max_seq_len=1024 + 2 K prompt: silent prompt truncation → garbage tokens
- **max_seq_len=4096 + 2 K prompt: prompt fits but model still emits `!` × N**

Then bisected collapse threshold with controlled-content prompts
("Hello world. " × N + "The capital of France is", `max_tokens=15`,
`temperature=0`):

| N reps | ~tokens | Output (first 60 chars) | Verdict |
|---:|---:|---|---|
| 5 | 20 | `' Paris. The capital of France is Paris.'` | ✅ coherent |
| 10 | 35 | `'\n\n  \n by, , and\n22\n   3'` | ❌ garbage |
| 15 | 50 | `'fe e'` | ❌ garbage |
| 20 | 65 | `",'tmconfess1anh0icosneZl0\n\n"` | ❌ garbage |
| 25 | 80 | `'0\nThe 0, ...​...1: "Ex"'` | ❌ garbage |
| 30 | 95 | `' the same, N1\n\n),'` | ❌ garbage |
| 40 | 125 | `'00组\n，焦尽+D2+0-.2'` | ❌ garbage |
| 100 | 305 | `'!!!!!!!!!!!!!!!!'` | ❌ `!` collapse |
| 150 | 455 | `'!!!!!!!!!!!!!!!!'` | ❌ `!` collapse |

**The corruption threshold is ~30 tokens.** That's catastrophically
short: virtually every real chat / eval / agent prompt exceeds it.

Cross-temperature probe (T=0.0 / 0.5 / 1.0, fixed seed): bug fires at
every temperature → rules out greedy-decode-specific path. Bug is in
the engine forward pass, not the sampler.

## Why this didn't surface earlier

- Existing `arle serve` smoke tests use single-token or very-short
  prompts (e.g. the 2026-05-22 9B GPTQModel recheck used a 1-token
  `/v1/completions` request). Those prompts are below the ~30-token
  threshold.
- OPD train (`train::OpdStep`) calls
  `LoadedInferenceEngine::forward_token_logits` directly — different
  code path, bypasses the serving scheduler entirely.
- Existing wins entries on 4B-teacher OPD use rollout=8 with
  prompt_max=16 — total scored sequence stays ≤24 tokens. Below
  threshold by construction.

This bug has been latent since whenever the serve path last regressed,
because every routine test stayed under the threshold.

## Revised hypothesis ranking

- **H1 RoPE drift / positional encoding bug** — strongest. The ~30
  token threshold is far below `max_position_embeddings` (32 768 for
  Qwen3.5), so it's not a model limit. RoPE position arithmetic at
  prefill chunk boundaries is the most likely culprit. Possibly an
  off-by-one or wrap that fires at small chunk indexes.
- H2 chunked prefill miscalc — **ruled out** for the ~30-token case:
  with `--chunked-prefill-size 4096`, a 200-token prompt is a single
  chunk. Bug still fires.
- H3 max_seq_len truncation — partial: max_seq_len=1024 produced a
  different garbage shape, but 4096 still fails. Truncation only
  changes the failure mode.
- H4 BOS / chat template mismatch — possible. Worth probing the exact
  request → tokens path that ARLE serve produces vs. HF reference.
- H5 paged KV bug — unlikely with `--num-slots 1`.

## Recommended next code work (real, scoped)

1. **Bisect to the single-token threshold**: try prompts of 25, 26,
   27, ..., 35 tokens to find the exact boundary. RoPE bugs often
   hinge on power-of-2 or window-size constants.
2. **Tokens-eq-prompt control**: send the same SHORT prompt (20 tok
   that works) but prepend padding tokens via the API. If padding
   fixes it, it's a position-id bug. If padding also breaks, it's
   content/structure-driven.
3. **Diff infer code path serve vs forward_token_logits**: which
   layers differ? RoPE cache construction is a common divergence
   point. (Touches dirty `infer/src/model/qwen35/prefill.rs` — codex
   territory.)
4. **Capability eval workaround until bug fixed**: write a
   `scripts/arle_capability_eval_direct.py` that uses the
   `forward_token_logits` API directly via a small Rust binary or
   embedded Python binding. Bypasses the broken serve path while we
   debug it.

## Cross-links update

- ARLE OPD pipeline manual MUST note "OPD bypasses serve" callout:
  `docs/projects/2026-05-21-arle-opd-cuda-usage-manual.md`
- ApiTeacher path now BLOCKED until serve bug fixed:
  `docs/plans/2026-05-21-opd-teacher-api-and-multiteacher-plan.md`

## Cross-links

- 9B GPTQModel `!` collapse (the first surface of this bug, written off as quant-specific): `docs/experience/errors/2026-05-22-qwen35-9b-gptqmodel-generation-f32load-fix-kill.md`
- 9B serve recheck (1-token-only worked, didn't catch long-prompt path): `docs/research/2026-05-22-qwen35-9b-gptqmodel-live-recheck.md`
- Capability eval P0 kill (this find's immediate trigger): `docs/experience/errors/2026-05-22-capability-baseline-triplet-harness-kill.md`
- Capability eval plan (blocked by this find): `docs/plans/2026-05-22-arle-opd-capability-eval-plan.md`
- ARLE OPD pipeline manual (needs an "OPD bypasses serve" callout): `docs/projects/2026-05-21-arle-opd-cuda-usage-manual.md`
