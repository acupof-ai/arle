# FA3 paged was gated on qlen 1 — the spec verify never got it

## Context

The 2026-07-27 champion row measured DSpark net-negative at serving
concurrency (−6.3% at c=8, −7.1% at c=16) and repriced it as "a speculation win
that was really paying for a kernel defect"
(repricing; entry deleted).

That comparison was rigged, by me. `full_attention_paged` selected FA3 with:

```rust
let decode = meta.seq_len == 1;
if decode && pool.format == KVFormat::BF16 && ... { /* FA3 */ }
```

A DSpark verify carries `block + 1 = 17` query rows, so `meta.seq_len == 17`
and the whole spec arm fell through to the TileLang paged **prefill** kernel.
The 2.76× reached the no-spec arm only. I benchmarked a fixed path against an
unfixed one and published the ratio.

The arithmetic said so before the code did: DSpark measured 19.31 ms/token at
E[k] ≈ 2.19, so a verify step cost ≈ 61.6 ms against a 28.64 ms decode step —
**2.15×**. Verifying 17 tokens reads exactly the same KV bytes as verifying 1;
only a few 17-row GEMMs are added. The honest ceiling is ~1.05×.

## What Worked

Per-request `seqlen_q` from `meta.q_offsets`, causal (the shim demotes to
non-causal at qlen 1), split-KV always. The vendored units needed nothing:
`arle_fa3_shim.cu` sizes `out_accum`/`softmax_lse` by `seqlen_q` already, and
both paged dispatches (Split true/false) are compiled.

I first widened it to **every** query length, prefill included. That regressed
TTFT p50 51% at c=8 and is reverted — the gate is `FA3_MAX_QLEN = 64`, so
decode and verify take FA3 and prefill chunks keep the TileLang paged kernel
([entry](../errors/2026-07-28-fa3-prefill-per-request-launch-regression.md)).

Needle gate, `needle_gate.py 512,4096,16384,32768 3 0.0` (`qwen3_nonthink`,
RAW): **exact=3 miss=0 DET at every length**, on the dense 27B **and** on
`Qwen3.6-35B-A3B-FP8`. The MoE needed no work — its `head_dim` is also 256 and
it shares `full_attention_paged`; it only diverges at the FFN. It does exercise
a GQA ratio of 8 against the dense model's 6, which PackGQA had not been
checked at.

Decode, `bench-agent-32k-16x8`, token-weighted mean ITL, capped-gate binary
`arle-fa3c`, all three arms one session (full table in
[baselines](../../baselines.md)):

| | no-spec | DSpark 16 | ratio |
|---|---:|---:|---:|
| c=1, all turns | 28.71 ms (34.8 tok/s) | **9.39 ms (106.6)** | 3.06× |
| c=1, warm turns | 28.99 ms (34.5) | **8.77 ms (114.1)** | 3.31× |
| c=8 | 908.39 ms | 838.64 ms | 1.08× |
| c=16 | 1822.09 ms | 1755.26 ms | 1.04× |

**3.06× per token at c=1**, up from 1.48× before the verify path reached FA3.
The verify step costs 9.39 × E[k+1]=3.19 ≈ 30.0 ms against a 28.71 ms decode
step — **1.04×**, which is the floor: verifying 17 tokens reads the same KV
bytes as verifying 1.

The decay to 1.08× at c=8 is not the spec path failing. Both arms collapse
identically there (ITL p50 66.07 vs 66.19 ms, p90 4035 vs 3823 ms) and the MoE
arm collapses too — it is the scheduler queueing, and it caps every arm's total
tok/s at c=8. Speculation only converts *idle* capacity; a full batch has none.

## Problems

- **The single-request probe used for the 2.76× is unsound and is dropped.** It
  timed `for line in urllib_response`, which stamps buffer fills, not SSE
  arrivals. On one binary it reported mean 72.6 ms against its own p50 of
  26.3, where the harness measured mean 28.64 / p50 28.71 / max 32.3 over
  14208 gaps — no tail at all. The p50 survived the buffering and the mean did
  not. Only `bench_throughput.py`'s `itl_s` is quoted now.
- **`DsparkConfig.layer_types` is parsed and never honored** — every draft
  layer runs the 2048 sliding window. DFlash declares 1 of 5 layers full;
  the DSpark checkpoints declare all 5. Honoring them needs a ctx ring the
  length of the request (671 MB/slot at 32k vs 42), so windowing stays — but it
  sits directly on acceptance and is now a startup warning instead of silence.
- **No `ncu`.** The needle gate and the harness deltas are the evidence.

## Rule

**A/B arms must differ in exactly the thing under test.** A capability
predicate written for one shape (`seq_len == 1`) silently partitions the arms
when another arm has a different shape, and the result reads as a property of
the feature instead of an artifact of the gate. Before comparing two arms,
check that the code path you just changed is reached by both — the per-step
cost ratio will tell you before the profiler does.
