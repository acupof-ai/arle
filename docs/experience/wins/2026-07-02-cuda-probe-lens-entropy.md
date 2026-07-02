# CUDA inference probe: decode logit lens + per-token entropy JSONL

> Status: **Shipped** — verified on the 8×H20 pod 2026-07-02 (smoke PASS,
> probe-off perf guard PASS, probe-on costs measured below).

## Context

Analysis probe for two questions: (1) at which layer does the decode
distribution's perplexity plateau (early-exit depth — logit lens over the last
N layers, default 10); (2) per-token entropy/NLL for every prefill position and
every sampled decode token. `arle serve --probe-out <path>
[--probe-lens-layers N] [--probe-token-entropy BOOL]` → rank-0 JSONL; env
transport `ARLE_PROBE_*` (docs/environment.md).

## Env / Params

8×H20 (115.190.184.36), DSv4-Flash-FP8, TP=8 (`INFER_TP_SIZE=8`, allreduce
MoE + deepgemm experts), eager decode (`ARLE_DSV4_DECODE_GRAPH` unset),
`--spec-type none --max-running-requests 1`, max_seq 4096. Build
`--release --features cuda,nccl` at `9fbc0954`; probe symbols confirmed in the
binary via `strings`. B=1, ~295-token prompts (unique-prefixed), decode tok/s =
128/(median wall₁₂₉ − median wall₁), 3 reqs/arm, wall spread <1%.

## Results

**Smoke (one greedy completion, 328 prompt + 64 gen tokens):** meta line ✓;
328/328 prefill records (nll omitted only at the two prefill-chunk tails) ✓;
64 decode records ✓; 63 lens positions × exactly 10 records, layers 33..42 of
the 43-layer model ✓ (first generated token comes from the prefill forward,
by design not lens-instrumented). Self-check: final-layer lens NLL vs decode
NLL **mean|Δ| = max|Δ| = 0.0000** (6 dp, n=63); `agree` 60/63, the 3 falses are
exact bf16 argmax ties (`top1_logprob == −nll` to 6 dp), not distribution
mismatches.

**Layer-PPL (exp(mean lens nll) over 63 greedy decode tokens):**

| layer | mean_nll | PPL | agree% |
|---|---|---|---|
| 33 | 8.170 | 3532.3 | 19.0% |
| 34 | 7.544 | 1888.9 | 20.6% |
| 35 | 7.225 | 1372.7 | 28.6% |
| 36 | 7.002 | 1098.7 | 31.7% |
| 37 | 5.831 | 340.5 | 34.9% |
| 38 | 5.527 | 251.3 | 39.7% |
| 39 | 4.441 | 84.8 | 42.9% |
| 40 | 3.025 | 20.6 | 49.2% |
| 41 | 3.000 | 20.1 | 52.4% |
| 42 | 0.416 | **1.52** | 95.2% |

PPL collapses 20.1 → 1.52 only at the final layer: **no early-exit plateau in
the last 10 layers** of DSv4-Flash on this workload — the last layer's jump is
load-bearing.

**Perf (same binary, same session, flag-only arms):**

| arm | decode tok/s | Δ vs guard | TTFT p50 |
|---|---|---|---|
| probe absent (guard) | 24.93 | — | 435 ms |
| flags present, lens 0 + entropy off | 24.97 | +0.16% (wash) | 435 ms |
| entropy only | 24.60 | −1.3% | 923 ms (+488 ms @ 295-tok prompt) |
| lens 10 + entropy | 14.54 | −41.7% | 922 ms |

Zero-cost-off **verified** (guard = sanity arm to 0.16%). Lens cost =
2.81 ms/layer wall = 1.64 ms GPU `dsv4/stage/lm_head_project` + 0.12 ms
head_hc/norm + ~1 ms 517 KB D2H + host entropy over 129,280 f32 (stage-profile
attributed, 10 × 2.81 ≈ measured +28.1 ms/step). Entropy TTFT cost scales
linearly with prompt length (chunked head projection of every position).
Guard-vs-parent-binary A/B deferred (shared pod tree under a concurrent
agent); the flag-absent = flags-off match bounds the probe's off-cost at ≈0.

## Consumer caveats

- Each non-final prefill chunk emits one extra `decode` record (the discarded
  boundary sample) — filter `decode` records to `pos ≥ prompt_len`.
- bf16 argmax ties can flag `agree=false` on an equal-probability top-1;
  detect via `top1_logprob == −nll`.
- Positions restart at 0 per request within one serve session (readiness
  probes also write records) — slice by offset for multi-request analysis.
- `--probe-out` with lens 0 + entropy off lazily creates no file at all.

## Rule

Instrument at existing convergence/sync points (sampler entry, post-sampling
sync) instead of inside per-layer hot loops; stash device buffers and defer
D2H to a sync point the path already pays for. Verify a probe's "zero cost
when off" claim with a same-binary flag-absent vs flags-off A/B, not code
inspection alone.
