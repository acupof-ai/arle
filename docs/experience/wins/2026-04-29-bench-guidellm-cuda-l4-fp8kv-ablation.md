# CUDA L4 fp8 KV Ablation — Qwen3 / Qwen3.5 / SGLang, 2026-04-29

## Goal

Measure fp8 KV ON vs BF16 KV OFF on CUDA L4 without touching weight
quantization, and close the warmup / overlap / short-request follow-ups.

## Environment

- Backends: ARLE CUDA and SGLang 0.5.10.post1
- Hardware: NVIDIA L4, 23.66 GB VRAM, driver 580.82.07
- Baseline / prior rows: `cb03c1df`, `dc99705f`, `b1cbed19`
- Fixed ARLE Qwen3.5 rows: `b63712fa` + `c5b4ab7e` + `95d2be64`
- Feature set: `/tmp/arle-target/release/infer`, release CUDA build
- Canonical request shape: c=16, 4096 prompt tokens, 256 output tokens,
  120s, `scripts/bench_guidellm.sh --profile concurrent --concurrencies 16`
- ARLE Qwen3.5 run flags: `--num-slots 16 --max-seq-len 8192
  --kv-cache-dtype {fp8,bf16}`
- SGLang Qwen3.5 run flags: `--dtype bfloat16 --kv-cache-dtype
  {fp8_e4m3,bf16} --mem-fraction-static 0.85 --max-running-requests 16
  --context-length 8192 --chunked-prefill-size 2048 --max-prefill-tokens 16384
  --disable-cuda-graph-padding --disable-piecewise-cuda-graph`

## A · Hangover Closure

| item | result | evidence |
|---|---|---|
| Server warmup pipeline | fixed | `047beccf`; HTTP listener opens only after scheduler warmup signal. Smoke: warmup done 13:37:05.856, listener 13:37:05.913. |
| CPU/GPU overlap | instrumented, no thread split | `047ca156`; c=16 micro showed decode 73.1-73.4ms, admission 6-14us, emit 9-13us, cleanup 8us steady / 0.57ms EMA tail. |
| Short bypass revalidation | partially closed | Qwen3.5 short-only 32/16: `b1cbed19` TTFT p50 296.0ms, ITL p50 47.28ms, out tok/s 277.34; `dc99705f` TTFT p50 295.5ms, ITL p50 47.28ms, out tok/s 328.82. The later 0-token failure was traced to Qwen3.5 prefill workspace over-admission, not the short-bypass path. |

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

## B · ARLE Qwen3.5-4B fp8 KV Blocker And Retest

Root cause and fix are recorded in
`docs/experience/errors/2026-04-29-qwen35-fp8kv-zero-token.md`.

The invalid rows were true server-side zero-token completions. Qwen3.5 admitted
too many long prefill rows, then failed lazy workspace allocation in
FlashInfer/GDR scratch after the fp8 KV pool had already consumed memory.
Fixes:

- reserve model-reported Qwen3.5 scheduler workspace before KV-pool sizing;
- cap Qwen3.5 concurrent prefill rows to 1 while leaving decode at c=16;
- forward scheduler-side generated token ids through the emit `Finish` path;
- add a warning when prefill samples a stop token before any emit.

| model | backend | KV dtype | max_seq_len | KV tokens / slots | KV budget / alloc | peak KV util | GPU SM% | TTFT p50 | ITL p50 | out tok/s | status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3.5-4B | ARLE | fp8 | 4352 | 502,528 | 10.3 GB | n/a | 98.0 | 0.0ms | 0.0ms | invalid | invalid: completed 0-token requests |
| Qwen3.5-4B | ARLE | fp8 | 8192 | 502,528 | 10.3 GB | n/a | n/a | 0.0ms | 0.0ms | invalid | invalid again before fix |
| Qwen3.5-4B | ARLE | fp8 | 8192 | 398,624 | 8.2 GB | 79.2% | 99.4 | 12376.0ms | 60.77ms | 158.25 | valid after workspace reserve + prefill cap |
| Qwen3.5-4B | ARLE | bf16 | 8192 | 249,136 | 8.2 GB | n/a | n/a | n/a | n/a | n/a | blocked: BF16 HD256 FlashInfer decode returns CUDA error 9 during warmup/probe |

Raw artefacts:

- `bench-output/2026-04-29-arle-qwen35-fp8kv-on/`
- `bench-output/2026-04-29-arle-qwen35-fp8kv-on-seq8192/`
- fixed fp8 server: `bench-output/2026-04-29-arle-qwen35-fp8kv-on-fixed-server/server.log`
- fixed fp8 bench: `bench-output/2026-04-29-arle-qwen35-fp8kv-on-fixed/`
- fixed fp8 GPU dmon: `bench-output/2026-04-29-arle-qwen35-fp8kv-on-fixed-dmon/gpu_dmon.csv`
- bf16 blocked server: `bench-output/2026-04-29-arle-qwen35-bf16kv-fixed-server/server.log`
- bf16 no-graph retry: `bench-output/2026-04-29-arle-qwen35-bf16kv-fixed-nograph-server/server.log`

Delta vs the invalid `dc99705f` Qwen3.5 fp8 row cannot be expressed as a
meaningful percent because the before row had TTFT/ITL p50 = 0 and completed
0-token requests. The fixed fp8 row is the first valid ARLE Qwen3.5 result for
this shape.

## B · SGLang Qwen3.5-4B fp8 KV ON/OFF

| model | backend | KV dtype | max_seq_len | KV tokens / slots | KV alloc | GPU SM% | TTFT p50 | ITL p50 | out tok/s | total tok/s | status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3.5-4B | SGLang | fp8_e4m3 | 8192 | 337,910 | K 2.58 GB, V 2.58 GB, Mamba 4.61 GB | 96.3 | 6572.3ms | 70.40ms | 167.32 | 2845.13 | valid |
| Qwen3.5-4B | SGLang | bf16 | 8192 | 168,955 | K 2.58 GB, V 2.58 GB, Mamba 4.61 GB | 96.4 | 6328.6ms | 72.93ms | 162.41 | 2761.65 | valid |

Delta, SGLang fp8 vs bf16:

| metric | fp8_e4m3 | bf16 | delta |
|---|---:|---:|---:|
| out tok/s | 167.32 | 162.41 | +3.0% |
| TTFT p50 | 6572.3ms | 6328.6ms | +3.9% slower |
| ITL p50 | 70.40ms | 72.93ms | -3.5% |
| KV pool tokens | 337,910 | 168,955 | +100.0% |

ARLE fixed fp8 vs SGLang fp8:

| metric | ARLE fp8 | SGLang fp8 | ARLE delta |
|---|---:|---:|---:|
| out tok/s | 158.25 | 167.32 | -5.4% |
| TTFT p50 | 12376.0ms | 6572.3ms | +88.3% slower |
| ITL p50 | 60.77ms | 70.40ms | -13.7% |

SGLang raw artefacts:

- fp8 bench: `bench-output/2026-04-29-sglang-qwen35-fp8kv-on/`
- fp8 server: `bench-output/2026-04-29-sglang-qwen35-fp8kv-on-server/server.log`
- fp8 GPU dmon: `bench-output/2026-04-29-sglang-qwen35-fp8kv-on-dmon/gpu_dmon.csv`
- bf16 bench: `bench-output/2026-04-29-sglang-qwen35-bf16kv/`
- bf16 server: `bench-output/2026-04-29-sglang-qwen35-bf16kv-server/server.log`
- bf16 GPU dmon: `bench-output/2026-04-29-sglang-qwen35-bf16kv-dmon/gpu_dmon.csv`

## SGLang And Numerical Spot-Check

ARLE fp8-vs-bf16 numerical spot-check is blocked by the ARLE Qwen3.5 bf16
server failure above. SGLang spot-check was run with 16 fixed prompts, seed
`20260429`, `temperature=0`, `max_tokens=64`, then both outputs were tokenized
with `infer/models/Qwen3.5-4B`.

| backend | pair | exact output pairs | avg common-token match rate | earliest divergence | status |
|---|---|---:|---:|---:|---|
| ARLE | fp8 vs bf16 | n/a | n/a | n/a | blocked by ARLE bf16 decode warmup/probe failure |
| SGLang | fp8_e4m3 vs bf16 | 8 / 16 | 77.54% | generated token 2 | completed; not numerically identical |

Spot-check artefacts:

- `bench-output/2026-04-29-sglang-qwen35-spotcheck-fp8.json`
- `bench-output/2026-04-29-sglang-qwen35-spotcheck-bf16.json`
- `bench-output/2026-04-29-sglang-qwen35-spotcheck-compare.json`

## Conclusion

Do not flip ARLE fp8 KV to the default in this tranche.

Qwen3-4B strongly favours fp8 KV for capacity and ITL, and Qwen3.5 fp8 now has
a valid ARLE run after the workspace reserve and prefill cap. However, the ARLE
bf16 Qwen3.5 side is blocked by BF16 HD256 FlashInfer decode warmup/probe
failure, so ARLE still lacks a valid fp8-vs-bf16 precision pair. SGLang shows
fp8 KV capacity doubling and a small throughput gain, but its fixed-prompt
spot-check diverges on half the prompts. The default decision should wait for
the ARLE bf16 decode blocker and an ARLE token-level precision pass.
