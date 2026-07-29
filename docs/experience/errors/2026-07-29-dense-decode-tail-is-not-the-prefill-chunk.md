# The dense decode tail is not the prefill chunk

## Context

At c=16 the dense arm loses 55.4% of its decode wall to spikes: 17.3% of
inter-token gaps exceed 3× p50, p90 426 ms against p50 66.6 ms. MoE has the same
shape far weaker (6.1% / 21.7%). Dense prefill runs ~2× slower per token, so the
obvious reading was that a prefill chunk blocks the decode batch for its whole
duration.

## Root Cause — the hypothesis, and why it is wrong

The arithmetic was persuasive: `chunked_prefill_size` defaults to 2048, dense
prefill measures 4252 tok/s, so one chunk is **481.7 ms** — and dense p90 is
426 ms with max 667 ms. MoE at 10,991 tok/s gives 186 ms per chunk and a much
lighter tail. Two models, two prefill rates, tails in the predicted ratio.

Swept `--chunked-prefill-size` 512 / 1024 / 2048 on dense, 48 req/point:

| chunk | predicted chunk time | TPOT c=16 | ITL p50 | ITL p90 | gaps > 3× p50 |
|---|---:|---:|---:|---:|---:|
| 512 | 120 ms | 100.93 ms | 63.68 ms | 228.54 ms | 11.36% |
| 1024 | 241 ms | 102.34 ms | 63.50 ms | 236.03 ms | 11.36% |
| 2048 | 482 ms | 99.29 ms | 63.76 ms | 231.44 ms | 11.42% |

**Nothing moves.** A 4× change in chunk size leaves p90 at 228-236 ms and the
tail share identical to three significant figures. p90 is not the chunk time at
any setting — at 2048 it is half the prediction, at 512 it is double.

So the spike is bounded by something the chunk size does not control. Whatever
serializes decode behind prefill is not paying per-chunk-token.

## Fix

None — the lever is rejected, not tuned. `chunked_prefill_size` stays at 2048.

## Rule

**A ratio that matches is not a mechanism.** Two models' tails landed in the
ratio of their prefill rates, and the absolute number sat inside the predicted
range — and the hypothesis was still wrong. The cheap sweep that falsified it
cost one run; the redesign it would have justified would have cost a week.
Sweep the knob the hypothesis names before building on it.
