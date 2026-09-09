# Batched MTP on Qwen3.8-27B-NVFP4: wins at c≤4, loses at c≥8 — break-even needs accept rate 0.75, measured 0.47

> Status: Accepted with bounds. The batched implementation is correct (chains
> engage at every c, needle 12/12, lever gate PASS) and ships; the speculation
> is profitable only at c≤4 on this workload. Qwen3.8-27B-NVFP4 + model_mtp
> (d=2), fp8 KV, 1×H20, 32K agent chain, greedy.

## Context

The first MTP acceptance run (arm A) showed the spec chain-counter delta
exactly 0 for c≥4: the draft was pinned to gate=1 (serial, one row at a time),
so at batch the speculation never engaged. The fix batches the draft
(`mtp_decode_batch`, GEMMs batched, per-row attention on the 1-layer MTP head)
and the verify is one batched prefill-shaped forward over N×(d+1) rows. This
entry is the acceptance rerun on the full model with the fix in.

## Phenomenon

Per-request decode throughput, same binary, `--spec-type auto` vs
`--spec-type none` (256 max-tokens, magnitude run):

| c | decode ms/tok, MTP | decode ms/tok, no-spec | Δ | per-request tok/s Δ |
|---:|---:|---:|---:|---:|
| 1 | 11.386 | 13.775 | −17.3 % | +21 % |
| 4 | 4.641 | 4.704 | −1.3 % | −3 % |
| 8 | 3.911 | 3.062 | +27.7 % | −19 % |
| 16 | 3.099 | 2.397 | +29.3 % | −15 % |
| 32 | 2.999 | 2.148 | +39.7 % | −22 % |

The metric is `decode_forward_busy_micros / decode_forward_tokens` — both
buckets count only pure-decode steps (prefill_rows empty). The old metric
divided by `generated_tokens`, which includes mixed-step tokens; spec changes
the step composition, so the old metric is not A/B-comparable.

Engagement is fixed: ~4200 chains at every c (was 0 at c≥4), accept rate
0.467 flat across c, tok/step ≈1.9 (verify works). Mixed-step counts are equal
across arms (31 vs 31 at c=32) — the loss is on pure decode, not in
prefill/decode mixing, so the 2026-08-22 mixed-step mechanism does not apply.

Per-step cost (ms per engine step, pure decode): 2.4–2.5× at c≥8 (c=32: 162.1
vs 64.5 ms, 2.51×); 1.6× at c=1.

## Root Cause

Break-even. The verify sends 3N rows (d=2 draft + base) through the trunk and
returns 1+d·a tokens per step, where a is the accept rate. It is profitable
iff

```
1 + d·a > cost_ratio(3N / N)
```

Measured: cost_ratio is 1.6 at c=1 (idle GPU) and 2.51 at c=32 (saturated),
and 1+2·0.467 = 1.93. At c=1, 1.93 > 1.6 → win. At c≥8 the saturated
cost_ratio exceeds 1.93 → loss, growing with c. At c=32 the break-even accept
rate is (2.51−1)/2 = 0.755; the measured 0.467 is 0.29 short.

The cost ratio rises with c because the no-spec arm gets cheaper faster under
batching (its single N-row forward amortizes), while the verify's 3N rows are
on the critical path of every step.

## Mechanism (nsys, 64 max-tokens)

NVTX ranges label each trunk op with its row count, so verify (seq=3N) and
decode (seq=N) steps separate in the same run. Within the mtp c=16 run:

| op | decode (N=16) | verify (3N=48) | verify/decode |
|---|---:|---:|---:|
| full_attention (16 layers) | 3.44 ms/step | 13.05 ms/step | 3.8× |
| linear_attention (24 layers) | 3.50 ms/step | 9.74 ms/step | 2.8× |
| dense_ffn | 1.28 ms/step | 1.25 ms/step | 1.0× |
| input/post-attn norms + residuals | 1.08 ms/step | 2.64 ms/step | 2.4× |

The verify's attention cost is at the structural row ratio (3×): the varlen
kernel reads the KV once per request per call, and 3N query rows cost ~3× the
attention time of N rows. There is no pathological re-read. The FFN is flat
across the row ratio — at M=16/48 the small GEMMs are latency-bound, so the
verify's extra rows cost little there. The 2.51× per-step cost ratio at c=32
sits between the attention ratio and the flat FFN, as the step mix predicts.

At c=32 with 64 max-tokens, pure-decode steps do not occur: 32 requests ×
33K prefill tokens against 64 decode tokens each means every engine step is
mixed (the trace's buckets are prefill chunks and partial batches). The c=32
magnitude therefore comes from the 256-max-token run above; the 64-token run
supplies only the per-op structural split.

## Fix

The batched draft and verify shipped (04c3bb76e, 3a0b05e0b); the pure-decode
metric shipped with it (3e72ae8a4). The speculation stays on: it is the
default for NVFP4 checkpoints with an MTP head and wins the c=1 SLO lane
(+21 % per-request tok/s) and washes at c=4. The lever for c≥8 is drafter
quality (accept rate), not the verify kernel — a drafter that raises a above
0.75 at c=32 flips the sign without touching the runtime.

## Parameters

- Model: `/mnt/data02/Qwen3.8-27B-NVFP4` (NVFP4 weights, model_mtp head, d=2).
- Binary: `/host/arle-build-a/target/release/arle` (release profile), lane/a
  merge bb8ee217f + batched MTP (04c3bb76e, 3a0b05e0b).
- Serve: `--backend cuda --kv-cache-dtype fp8 --max-total-tokens 65536
  --max-running-requests 32 [--spec-type auto --mtp-draft-tokens 2
  --spec-max-batch 32 | --spec-type none]`, 1×H20 card 6.
- Magnitude run (A/B table): `scripts/bench_throughput.py`,
  `/host/bench-agent-32k-32.jsonl` (32K agent chain), c-sweep 1/4/8/16/32,
  requests-per-concurrency = c, **256 max-tokens**, temperature 0.
- Mechanism run (nsys): same prompts, c=16 and c=32, **64 max-tokens** —
  short-decode shape for the structural per-op split; not a throughput
  source. `nsys profile --trace cuda,nvtx --sample=none --cpuctxsw=none`,
  ARLE_NVTX=1; per-op times from `nvtx_sum` ranges labeled `seq=<rows>`.
- Correctness: `scripts/needle_gate.py 512,4096,16384,32768 3 0.0` (RAW=1,
  TEMPLATE=qwen3_nonthink) under MTP: 12/12 exact, DET. `lever_gate.sh`
  nospec-baseline vs MTP-lever: correctness PASS (summaries=5, concurrent
  needle PASS, temp arm PASS).
- Dates: 2026-09-09/10, pod `iv-yeozpb5g5cbw80bls64e`.

## Rule

A spec-decode acceptance has two independent questions: does the draft engage
(chains > 0 at every c — the arm-A bug), and does it pay (1+d·a against the
measured cost ratio at the target c). Engagement is a correctness gate;
profitability is arithmetic on measured numbers, and at saturated batch the
cost ratio is set by the no-spec arm's amortization, not by the draft.
