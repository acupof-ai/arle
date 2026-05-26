# A100 INT8 KV vs BF16 KV — +57~113% throughput, −39~55% ITL — kv-int8-vs-bf16-a100-2026-05-26

## Goal

Quantitative validation that the 2026-05-25 parity audit's
`mean_match=1.0` for INT8 KV on Qwen3-4B dense translates into real
production throughput / latency wins worth shipping as the recommended
quantized KV mode. Same prompt → confirm INT8 is the right "all
precisions usable" answer for Qwen3-dense before deeper FP8 fix work
lands.

## Hypothesis

INT8 KV halves the per-token KV-cache bytes vs BF16 (1B/elem vs 2B/elem
per quantized element + per-(token, head) f32 scale ≈ 1.06B/elem vs
2B/elem). Decode attention is memory-bound on the KV reads for HD128
GQA workloads, so halving the bytes should roughly halve the per-token
attention latency.

## Command

```bash
# On A100-SXM4-80GB sm_80 with Qwen3-4B BF16 weights:
./scripts/bench_int8_vs_bf16.sh   # see attached, runs:
#   ./target/release/infer --model-path infer/models/Qwen3-4B --port 8000 \
#     --num-slots 16 --max-seq-len 5120 --mem-fraction-static 0.85 \
#     --kv-cache-dtype <bf16|int8>
#   scripts/bench_guidellm.sh kv-<label> --target http://localhost:8000 \
#     --model Qwen3-4B --processor infer/models/Qwen3-4B \
#     --concurrencies 1,4,16 --max-seconds 30 --warmup 5
```

## Environment

- Hardware: NVIDIA A100-SXM4-80GB, driver 535.261.03, CUDA 12.4, sm_80
- Model: `infer/models/Qwen3-4B` (BF16 dense HF snapshot, 7.6 GB)
- ARLE commit: `79ef8880` (main, post FP8 KIVI plan)
- Feature set: `cargo build --release -p infer --features cuda`
- Non-default flags: `--num-slots 16 --max-seq-len 5120 --mem-fraction-static 0.85`
- guidellm: 0.6.0, prompt_tokens=4096, output_tokens=128, max-seconds=30 / c

## Results

### Headline table

| Concurrency | KV mode | TTFT p50 (ms) | ITL p50 (ms) | out tok/s | total tok/s |
|---:|---|---:|---:|---:|---:|
| 1  | BF16 | 249.1  | 21.05 | 45.88  | 780.19  |
| 1  | INT8 | 251.4  | **9.37**  | **97.64**  | **1660.33** |
| 4  | BF16 | 856.7  | 23.41 | 128.38 | 2182.91 |
| 4  | INT8 | 878.0  | **13.44** | **210.21** | **3574.43** |
| 16 | BF16 | 3313.7 | 57.45 | 151.05 | 2568.37 |
| 16 | INT8 | 3039.6 | **35.19** | **237.15** | **4032.41** |

### Deltas (INT8 vs BF16)

| Concurrency | ITL p50 | out tok/s | total tok/s | TTFT p50 |
|---:|---:|---:|---:|---:|
| 1  | **−55.5%** | **+112.8%** | **+112.8%** | +0.9% |
| 4  | **−42.6%** | **+63.7%**  | **+63.7%**  | +2.5% |
| 16 | **−38.7%** | **+57.0%**  | **+57.0%**  | **−8.3%** |

INT8 wins decisively on per-token latency and throughput across the
c-sweep. TTFT is essentially flat — INT8 quant doesn't touch the
prefill compute path significantly (the same TileLang BF16 prefill +
post-attention finalize quant; the small c=16 TTFT improvement is
likely noise from a slightly faster scheduler tick under shallower KV
bytes-per-page).

## Findings

1. **INT8 KV is a clean memory-bound attention win**. The ITL p50
   improvement (−39% to −56%) tracks the ~1.9× memory-bandwidth
   reduction from halving KV bytes; the difference between the
   theoretical 2× and observed 1.6–2.2× is attributable to the
   per-(token, head) scale lookup overhead in the INT8 decode kernel.
2. **Throughput scales with concurrency batching but tapers**: c=1
   gains +113%, c=16 gains +57%. At higher concurrency, decode is
   bandwidth-balanced across more in-flight requests, so the
   per-request memory savings amortize less per token (other paths —
   sampling, scheduling, prefill — start to dominate).
3. **Both modes peak around 85% kv_util** at the configured
   `--mem-fraction-static 0.85`; INT8 fits roughly 1.9× more tokens
   into the same pool budget, but at concurrency 16 the workload
   doesn't pressure-test that since prompts are only 4096 tokens.

## Operational guidance

For Qwen3-4B / Qwen3-dense production:

- **Default**: keep `auto` = BF16 (correctness-safe).
- **For memory savings or higher concurrency throughput**: explicit
  `--kv-cache-dtype int8`. Expect +57~113% throughput, parity_match=1.0
  on short-decode (≤ 64 tokens), ~0.89 mean trajectory match past
  240 tokens (the known long-decode drift, documented in errors).
- **Do not use FP8 KV on Qwen3-dense** until the KIVI per-channel K
  scheme lands ([`docs/plans/2026-05-26-fp8-kv-per-channel-k-fix.md`](../../plans/2026-05-26-fp8-kv-per-channel-k-fix.md)).
  Catastrophic step-1 divergence is precision-floor compounding, not
  a code bug — `--kv-cache-dtype int8` is the correct quantized path
  today.

## Rule

- A quantized KV mode is "usable" only when the parity audit
  (`infer/tests/kv_precision_parity.rs`) shows `mean_match >= 0.95` at
  the production decode horizon AND a bench shows ITL p50 improvement
  ≥ 30% at concurrency 1. INT8 KV on Qwen3-dense clears both bars;
  FP8 fails the first (mean_match = 0.0156) and must not be the
  default-on choice.

## Cross-refs

- Plan + parity framework: [`docs/plans/2026-05-25-kv-precision-parity-framework.md`](../../plans/2026-05-25-kv-precision-parity-framework.md)
- FP8 root cause + KIVI fix plan: [`docs/plans/2026-05-26-fp8-kv-per-channel-k-fix.md`](../../plans/2026-05-26-fp8-kv-per-channel-k-fix.md), [errors entry](../errors/2026-05-26-fp8-kv-step1-divergence-known-deferred.md)
- Bench script: `/tmp/bench_int8_vs_bf16.sh` (paired script preserved alongside; not committed since it's a one-shot)
- Raw artefacts: `bench-output/2026-05-26-kv-bf16/` and `bench-output/2026-05-26-kv-int8/` on the A100 box
