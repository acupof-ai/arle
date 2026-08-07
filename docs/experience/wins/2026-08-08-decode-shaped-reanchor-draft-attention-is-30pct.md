# Decode re-anchored: draft attention is 30.5% of a tick, not the 4.3% it was priced out at — CUDA, 2026-08-08

> Status: **baseline established.** First decode-shaped capture at c=16, window
> reconciled against run totals to +6.8%. No runtime change; this measures.

## Problem

Every decode number in `docs/perf-qwen36-27b.md` was priced on the long-agent
32K anchor, which is 279:1 prompt:output and in which all decode together is
2–6% of GPU time. Decode levers ranked off it are ranked off a few percent of
the machine.

## Parameters

```bash
python3 scripts/gen_bench_prompts.py bench-decode-16x1.jsonl 16 1024 4096 1

/host/fqbatch/arle-C-70760bc09 serve --backend cuda \
  --model-path ThinkingCap-Qwen3.6-27B-FP8 \
  --spec-type dspark --mtp-draft-model Qwen3.6-27B-DFlash \
  --dspark-block-size 6 --port 8178 --max-running-requests 16

python3 scripts/bench_throughput.py --url http://127.0.0.1:8178 \
  --prompts-jsonl bench-decode-16x1.jsonl --concurrency-grid 16 \
  --requests-per-concurrency 32 --max-tokens 4096 --temperature 0 \
  --seed 20260416 --timeout-seconds 1200

nsys profile -o dec -t cuda,nvtx --delay 130 --duration 60 \
  --sample none --cpuctxsw none <serve>
```

1× H20 GPU 6, TP=1, eager, 16 slots. Artifacts `/host/decodeanchor/`.

- Measured prompt 1107 tok/req, output **1795** tok/req against 4096 requested —
  the driver and serve have **no ignore-EOS option** (checked), so generations
  terminate early. **Prompt:output 1 : 1.62**, inverted from the anchor's 279:1.
- 32 requests, not 128, so there is no queueing ramp to corrupt the window.
- Window at bench elapsed 40–100 s, 60 s, past the ~12 s ramp.
- Rows per tick **16.0** by four independent counts: `rms_norm_batched_offset`
  gridX ÷ 6, `prefill_attention_paged_hd256` launches ÷ 16 layers, `add_native`
  gridX ÷ 30, draft `nonpaged_prefill_attention` launches ÷ 5 taps.

## Window reconciliation — required before quoting any share

| | ticks/s |
|---|---:|
| window: 484 ticks in 59.973 s | **8.070** |
| run-level, empirical chain length 3.375 tok | **7.559** (+6.8%) |
| run-level, `16 × (1 + 0.475 × 6)` | 6.626 (+21.8%) |
| bound: 1556 steps / 140.81 s | 11.05 — window is below it ✓ |

Representative. The formula estimate overstates tokens per tick because a chain
stops at its first rejection, so `accepted/drafted` is not accepted-per-chain;
the empirical `1 + accepted/chains` is the tighter check.

Contrast the anchor capture the same day: 6 ticks against a run-level rate 2–6×
higher, because its window sat inside a queueing ramp.

## Results

Row: 407.97 out tok/s, 659.55 total tok/s, ITL mean 31.08 ms, TTFT p50 1.42 s,
28/32 complete. `accept_rate` 0.4749, prefix hit 0.5152.

```
tick span 112.33 ms | GPU-busy 96.88 ms (86.2%) | period 123.9 ms
draft attention        29.57 ms  30.5%
FP8 GEMM               27.87 ms  28.8%
GDN / gated-delta      20.30 ms  21.0%
dense GEMM (nvjet + splitK)
                       11.68 ms  12.1%
norms + elementwise     5.67 ms   5.9%
full-attn paged         1.75 ms   1.8%
sampling                0.06 ms   0.1%
                                  total 96.88 ms
```

**A decode lever's share is workload-dependent by 17×.** Full-attention is 1.8%
of a tick here at ~2.5K context and 42.5% at 32.5K on the anchor. Neither is
"the" decode budget.

**Draft attention was priced out at 4.3% and is 30.5% here.** The register
entry read `DSpark draft attention — 1.5 ms of 35 ms (4.3%) — priced out, 3
rewrites failed to transfer`. The three rewrites were sized against that 4.3%,
which is why a −33% microbench win was capped at −1.4% end to end. At this
shape the cap is 7× larger.

The kernel is `nonpaged_prefill_attention_kernel`, 39,690 launches in the
window. **This contradicts a standing note that the nonpaged kernel never fires
on serve.** It is true of the trunk, which takes
`prefill_attention_paged_hd256_kernel`; the DFlash draft has no paged full-attn
KV pool, so the draft lane takes the nonpaged branch. `full_attention_into` is
reached with `full_attn_paged() == false` for the draft executor.

**Idle inside a tick is 13.8%**, and 61.4% of that gap time sits in 53 gaps
over 1 ms. Decode ticks on the anchor capture had nothing above 20 µs. Cause
unknown. Beside it, HtoD moves 0.08 GB per tick in **0.486 ms** — latency, not
bandwidth.

Achieved bandwidth on the paged full-attn kernel:

```
16 rows x 2484.3 tok x 65536 B = 2.605 GB / 1.590 ms = 1.638 TB/s = 47% of 3.5
```

Against 1.02 TB/s (29.2%) at 32.5K and 9 rows. Longer context measured *lower*
achieved bandwidth, but row count moved too and the two effects cannot be
separated from two points.

Mean context 2484.3 tokens is reconstructed from the clean run's per-request
`prompt_tokens` + `ttft_s` + `itl_s`, simulating 16 slots in submission order
and sampling live rows every 5 s across the window. It is not read off the
trace, and it ranges 2196 → 2673 within the window.

## Problems

The capture needed two attempts. `nsys` defaults to `--kill sigterm`, which
SIGTERM'd serve at window end and hung the bench, so run 1 has a clean 60 s
trace but no end-of-run stats. Run 2 with `--kill none` aborted with
`free(): invalid pointer` 22 s into the window, 28/32 requests errored, capture
unusable. Analysis uses run 1's trace with run 2's clean row from Step 2, which
is the same workload. Not retried, per the two-failure rule.

## Learnings

**A decode lever has no single share — it has a share per context.** 1.8%
against 42.5% for the same kernel on two workloads on the same day. "Priced
out" is only meaningful with the shape attached, and this register carried
verdicts without one.

**Reconcile the window against run totals as a step in the analysis.** One
division separated a representative window (+6.8%) from a badly unrepresentative
one (2–6× off) on the same box, the same day, with the same tooling.

**Check for an ignore-EOS knob before designing a generation-heavy bench.**
There is none here, so a decode-shaped workload tops out at whatever the model
chooses to emit — 1795 of 4096 requested.
