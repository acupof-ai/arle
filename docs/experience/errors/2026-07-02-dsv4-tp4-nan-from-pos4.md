# DSv4-Flash TP=4: forward NaNs from position 4 — every completion detokenizes to empty

## Context

Round-7 coverage on the 8×H20 pod: DSv4-Flash (`/host/DeepSeek-V4-Flash`,
bf16 dir, unchanged since Jun 23) TP=4, `arle serve --backend cuda`. Every
request on every shape returned `text: ""` with `finish_reason: length`
(tokens ARE generated — usage counts them — none visible). Streaming shows
zero content deltas.

## Root Cause

Not yet attributed to a commit — but the case is fully decoded and bounded:

- `--probe-out` (a25922b9) per-position records: prefill entropy/nll sane
  for pos 0, 1, 2, 3 (7.40 / 7.13 / 3.06 / 5.80), **NaN from pos 4 onward**;
  every decode step then records `token: 0` (bos, special) with NaN
  entropy — argmax over all-NaN logits → special token → skip-special detok
  → empty string. Lens records show NaN at EVERY tapped layer with
  `top1: 129279`.
- 1-token prompt emits 3 REAL tokens then goes invisible — the wall is at
  absolute position 4 exactly, matching the first compress-ratio-4 chunk
  (`compress_ratios` layer arm = 4: chunk covers pos 0-3, first consumed at
  pos 4). Mechanistic suspect: the DSv4 compressed-attention chunk path.
- Bisect (worktree boots, same box/model/shape): reproduces on `958536e9`,
  `16a95fe0`, and the round-6 build `5cafb308` — probe commit `a25922b9`,
  LoRA-FP8 promotion `16a95fe0`, and the whole round-7 window exonerated.
  Tier flags exonerated (control without `--kv-disk` reproduces).
- Contradiction to resolve: round-6 recorded "DSv4 regression completion
  clean" on `5cafb308`; measured on `5cafb308` this shape NaNs. That check
  must have used a different shape/criterion (or never looked at text).

## Fix

Open. Next steps for the owner: probe the compressed-chunk write/read at
pos 4 layer-by-layer (the probe JSONL machinery already localizes it);
check the ratio-4 arm's chunk fold for an fp8/bf16 denorm or a zero-length
softmax; establish when (if ever) this exact TP=4 shape last produced text.

## Rule

"Regression clean" claims must pin the exact serve shape + a visible-text
assertion; an HTTP-200/no-crash smoke passes straight through an all-NaN
forward (usage counts tokens even when every one is invisible). Decode the
generation (probe/ids), don't trust the transcript.
