# DSv4 DSpark sliding-window draft latent — c=1 +49%, c=4+ crash

> Status: Active — c=1 win confirmed; c>1 crash under investigation.

## Context

DSpark draft latent KV was sized `max_seq_len + block` (linear in prompt
length), so long prompts OOMed the per-slot budget and the
`--dspark-max-prompt-tokens 64` router disabled DSpark on production ~2.8k tok
prompts. `00224faa0` sized latent KV to `sliding_window + block` (fixed) and
shifted the live context forward (memmove) on overflow.

## Params

- 4×H20 TP=4, GPUs 4,5,6,7; DSv4-Flash FP8 + DSpark FP8 draft, block 5, greedy
- `bench-prompts-64.jsonl` (~3.4k tok), 60s/point, max_tokens 256
- No `--dspark-max-prompt-tokens` (router off)

## Results

| c | No-spec tok/s | DSpark tok/s | Δ% |
|---|-------------:|-------------:|---:|
| 1 | 38.1 | **56.8** | **+49.1%** |
| 4 | 74.1 | CRASH | — |
| 8 | 100.3 | CRASH | — |
| 16 | 117.5 | CRASH | — |

DSpark c=1 accept_rate: 87.1% (drafted 5197, accepted 4527, rejected 670).

## Crash (c=4+)

```
[arle-worker rank=0] failed: worker rank 0 step (tick #1432):
  DSv4 MTP argmax D2H failed: DriverError(CUDA_ERROR_ILLEGAL_ADDRESS,
  "an illegal memory access was encountered")
```

All 4 TP ranks failed simultaneously at c=4 ramp-up. No-spec baseline is stable
across c=1/4/8/16 with 0 errors.

## Hypothesis

The crash is in the spec verify path (`MTP argmax D2H`), multi-slot. The
sliding-window latent shift may leave stale rows that the verify path reads out
of bounds, OR the c>1 DSpark dispatch path (`executor/dsv4.rs:1920`) has a
pre-existing bug that the router masked (long prompts never reached B>1).

## Rule

- DSpark c=1 is a strong win (+49%); do not revert the sliding-window change.
- c>1 crash blocks a default flip; fix before claiming concurrency wins.
