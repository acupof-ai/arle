# CUDA L4 fp8 KV Ablation — partial run, 2026-04-29

## Goal

Measure fp8 KV ON vs BF16 KV OFF on CUDA L4 without touching weight
quantization, and close the warmup / overlap / short-request follow-ups.

## Environment

- Backend: ARLE CUDA
- Hardware: NVIDIA L4, 23.66 GB VRAM, driver 580.82.07
- Commit under test: `cb03c1df`
- Feature set: `/tmp/arle-target/release/infer`, release CUDA build
- Canonical request shape: c=16, 4096 prompt tokens, 256 output tokens,
  120s, `scripts/bench_guidellm.sh --profile concurrent --concurrencies 16`
- Note: Qwen3.5 ARLE results are marked invalid because guidellm observed
  TTFT/ITL p50 = 0 and server logs showed 0-token completions, even after
  raising `--max-seq-len` from 4352 to 8192.

## A · Hangover Closure

| item | result | evidence |
|---|---|---|
| Server warmup pipeline | fixed | `047beccf`; HTTP listener opens only after scheduler warmup signal. Smoke: warmup done 13:37:05.856, listener 13:37:05.913. |
| CPU/GPU overlap | instrumented, no thread split | `047ca156`; c=16 micro showed decode 73.1-73.4ms, admission 6-14us, emit 9-13us, cleanup 8us steady / 0.57ms EMA tail. |
| Short bypass revalidation | not fully closed | Qwen3.5 short-only 32/16: `b1cbed19` TTFT p50 296.0ms, ITL p50 47.28ms, out tok/s 277.34; `dc99705f` TTFT p50 295.5ms, ITL p50 47.28ms, out tok/s 328.82. Both runs later logged FlashInfer workspace OOM and 0-token completions, so short workload still needs a correctness fix. |

Short-request artefacts:

- `bench-output/2026-04-29-arle-cuda-l4-qwen35-short-b1cbed19/`
- `bench-output/2026-04-29-arle-cuda-l4-qwen35-short-dc99705f/`

## B · ARLE Qwen3-4B fp8 KV ON/OFF

| model | KV dtype | max tokens in KV pool | KV budget | peak KV util | GPU SM% | TTFT p50 | ITL p50 | out tok/s | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-4B | fp8 | 148,256 | 11.5 GB | 97.0% | 99.7 | 13121.1ms | 72.68ms | 107.56 | valid |
| Qwen3-4B | bf16 | 78,256 | 11.5 GB | 99.6% | 99.1 | 13074.1ms | 119.12ms | 73.62 | degraded; prefix demotion 1.46s and 0-token completion observed |

Delta, fp8 vs bf16:

| metric | fp8 | bf16 | delta |
|---|---:|---:|---:|
| out tok/s | 107.56 | 73.62 | +46.1% |
| TTFT p50 | 13121.1ms | 13074.1ms | +0.4% slower |
| ITL p50 | 72.68ms | 119.12ms | -39.0% |
| KV pool tokens | 148,256 | 78,256 | +89.5% |

Raw artefacts:

- fp8: `bench-output/2026-04-29-arle-qwen3-fp8kv-on-run2/`
- fp8 GPU dmon: `bench-output/2026-04-29-arle-qwen3-fp8kv-on/gpu_dmon.csv`
- bf16: `bench-output/2026-04-29-arle-qwen3-fp8kv-off-bf16/`
- bf16 GPU dmon: `bench-output/2026-04-29-arle-qwen3-fp8kv-off-bf16-dmon/gpu_dmon.csv`

## B · ARLE Qwen3.5-4B fp8 KV

| model | KV dtype | max_seq_len | max tokens in KV pool | KV budget | GPU SM% | guidellm status |
|---|---|---:|---:|---:|---:|---|
| Qwen3.5-4B | fp8 | 4352 | 502,528 | 10.3 GB | 98.0 | invalid: TTFT/ITL p50 = 0 and 0-token completions |
| Qwen3.5-4B | fp8 | 8192 | 502,528 | 10.3 GB | n/a | invalid again with same TTFT/ITL zero failure |
| Qwen3.5-4B | bf16 | n/a | n/a | n/a | n/a | not run; fp8 path was already invalid |

Raw artefacts:

- `bench-output/2026-04-29-arle-qwen35-fp8kv-on/`
- `bench-output/2026-04-29-arle-qwen35-fp8kv-on-seq8192/`

## SGLang And Numerical Spot-Check

Not completed in this tranche. The ARLE Qwen3.5 invalid-result blocker must be
fixed before using its fp8-vs-bf16 numbers or running a meaningful token-level
precision comparison across 16 fixed prompts.

## Conclusion

Do not change the ARLE default based on this partial run. Qwen3-4B strongly
favours fp8 KV for capacity and ITL, but Qwen3.5 still has a correctness /
stream accounting blocker at this bench shape. fp8 KV remains the right
candidate default, but the default should not be promoted further until the
Qwen3.5 0-token completion and guidellm TTFT/ITL-zero failure are fixed and the
SGLang + numerical spot-check rows are filled.
