# Metal: stop silently downgrading sampling, and give serve a default — metal, 2026-08-26

> Status: Confirmed (local M4 Pro 48 GB)
>
> Baseline: `0f1fef63c`, `arle serve --backend metal --model-path
> LiquidAI/LFM2.5-8B-A1B-MLX-4bit --no-speculative --kv-cache-dtype bf16`,
> M4 Pro 48 GB. Three loop-prone prompts, `max_tokens=2500`, greedy.

## Context

LFM2.5-8B-A1B loops on long generations. Reproduced at 2500 tokens (10-gram
repeat rate, and the count of the most repeated 10-gram):

| prompt | rep10 | top 10-gram | outcome |
|---|---|---|---|
| count to 200 | 0.211 | 10x | never finished |
| list 60 S-words | 0.128 | 6x | `"ascent"? Not S. "ascent"? Not S. …` |
| 60-line poem | 0.184 | 17x | `The machine knows … The machine knows` |

Two independent reasons the output was greedy, and greedy is what loops:

1. **The checkpoint ships no sampling config.** `SamplingDefaults` reads
   `temperature`/`top_k`/`top_p` from `generation_config.json`;
   LFM2.5-8B-A1B-MLX has none of them, so serve defaults to temperature 0 with
   no penalty. The client cannot fix this either — `eli` sends no sampling
   field at all, so *every* request was greedy no matter what.
2. **Metal ignored the request even when it asked.** `sample_inflight`
   downgraded any non-greedy request to device `argmax` unless
   `--metal-host-sampling`, warning once and then staying quiet. Measured:
   `temperature: 0.7, top_p: 0.95` returned **byte-identical** output to
   greedy — same token counts (2500 / 1983 / 2500), same rep10, same tails.

## What worked

- `infer-metal`: delete the downgrade. `is_raw_argmax()` keeps the device
  argmax + async fast path; everything else takes the host sampler it asked
  for. `--metal-host-sampling` and `MetalRuntimeFlags::host_sampling` are gone
  — the flag only ever controlled whether the backend lied.
- `arle serve --temperature` / `--repetition-penalty`: serve-wide defaults for
  fields the request omits, overriding `generation_config.json`. The only lever
  that reaches a checkpoint with no sampling config and a client that sends no
  sampling fields. `SamplingDefaults` gains `repetition_penalty` (no wire key —
  CLI-set only).

Same three prompts, `max_tokens=2500`:

| config | count | list | poem |
|---|---|---|---|
| greedy (baseline) | 0.211, 2500 tok, unfinished | 0.128 | 0.184 |
| `--repetition-penalty 1.05`, client sends nothing | **0.000, 710 tok, finished 1..200** | 0.154 | **0.004** |
| `temperature 0.7 + top_p 0.95` (now honored) | **0.000, 807 tok, finished** | 0.211, listed all 60 | **0.000, 1868 tok** |

Cost, paid only by requests that leave the raw-argmax path:

| path | tok/s |
|---|---|
| device argmax (greedy, no penalty) | 156-160 |
| host sampler, greedy + penalty | 138-142 |
| host sampler, temperature + top-p | 121-128 |

Correctness: needle ladder 512 / 2000 / 4000 / 8000 x3 exact 3/3 DET (the gate
runs temperature 0 with no penalty, so it still takes the untouched device
path); `cargo test -p infer-metal -p infer-server --features metal` 34/34.

## Recommendation

Serve LFM2.5 with `--repetition-penalty 1.05`. It is the whole fix for the
degenerate loop (the count task went from never terminating to a correct 710-token
answer), it costs ~11% decode, and it needs no client change. Add
`--temperature 0.7` on top if output diversity matters more than the further
~11%.

## Rule

- A backend that silently ignores a request parameter is worse than one that
  refuses it. `temperature: 0.7` returning byte-identical output to
  `temperature: 0` is a bug no caller can see, and the one-time `log::warn!`
  was in a file nobody reads.
- Sampling defaults have three layers — checkpoint, serve, request — and a
  feature only works if all three can express it. Here the checkpoint shipped
  nothing, serve had no override, and the client sent nothing: three layers of
  silence resolving to greedy.
- Repetition penalty and temperature are not interchangeable. Temperature cut
  the repeat rate but still ran to the token cap; the penalty is what let the
  model *finish* — degenerate loops are a max-probability problem, not an
  entropy problem.
