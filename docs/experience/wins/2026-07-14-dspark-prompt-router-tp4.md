# DSv4 DSpark prompt router — H20 TP=4

> Status: Shipped

## SLO-shape probed? N — 8K prompt, batch 1

The 8K production-length probe ran, but the replicated 19.9 GB draft leaves one
affordable slot per rank, so batch >= 4 was impossible. This entry licenses an
opt-in prompt router, not a DSpark default flip.

## Roofline check

Deferred: no NCU roofline run. The controlled wall-clock sweep already kills
DSpark above 64 prompt tokens; profiling a rejected path would not change the
router decision. Next wall: remove replicated draft weights before a concurrency
sweep.

## Goal

Find the DSv4 DSpark throughput crossover and remove its medium/long-context
regression without changing default behavior.

## Hypothesis

Full-context draft attention grows with prompt length, so one request-start gate
can preserve short-prompt gains and route longer prompts to the target decoder.

## Params

- DeepSeek-V4-Flash FP8 + DSpark FP8 draft, block 5, greedy
- 4x H20 96 GB, TP=4, GPUs 3-6; CUDA 12.9, driver 535.161.08
- Concurrency 1, 128 output tokens, `ignore_eos=true`, 5 s warmup
- 20 distinct natural document slices per shape; 12 requests at 8K
- GuideLLM 0.6.0, concurrent profile, seed 20260416
- Baseline binary: `fee5832ce`, SHA-256 `63925252...f02c6c`
- Router binary: SHA-256 `6b60a206...36e9d6`
- Router flag: `--dspark-max-prompt-tokens 64`
- Profiling off

## Command

```bash
GUIDELLM__MP_CONTEXT_TYPE=forkserver guidellm benchmark run \
  --target http://127.0.0.1:8799 \
  --model /host/DeepSeek-V4-Flash-FP8 \
  --processor /host/DeepSeek-V4-Flash-FP8 \
  --profile concurrent --data <natural-shape.jsonl> \
  --max-seconds 120 --random-seed 20260416 \
  --backend openai_http \
  --backend-kwargs '{"validate_backend":"/v1/models","request_format":"/v1/completions"}' \
  --disable-console-interactive --outputs json --outputs csv --rate 1 --warmup 5
```

Both servers used `INFER_TP_SIZE=4`, `INFER_CUDA_DEVICES=3,4,5,6`,
`--comm-backend nccl`, and `--max-running-requests 1`. Only `--spec-type` and
the router flag changed.

## Results — routed latency

| prompt | variant | TTFT p50 ms | ITL p50 ms | request mean s |
|---:|---|---:|---:|---:|
| 32 | no-spec | 175.17 | 22.13 | 2.991 |
| 32 | router | 218.38 | 20.40 | 2.831 |
| 64 | no-spec | 190.13 | 22.16 | 3.008 |
| 64 | router | 257.09 | 22.16 | 2.937 |
| 128 | no-spec | 230.19 | 22.14 | 3.044 |
| 128 | router | 224.65 | 22.10 | 3.038 |
| 8100 | no-spec | 10513.02 | 22.06 | 13.359 |
| 8100 | router | 10682.10 | 22.00 | 13.454 |

The router removes the 128/8K decode regression within run noise. Eligible
32/64 outputs were token-identical to ungated DSpark. The 128 output was
token-identical to no-spec. The same 8K prompt produced coherent outputs on both
paths; token identity is not required because DSv4 MoE has a non-determinism
floor.

## Request accounting

| shape | measured requests | completed output tokens | errors |
|---:|---:|---:|---:|
| 32 / 64 / 128 | 19 each | 2432 each | 0 |
| 256 / 512 | 19 each | 2432 each | 0 |
| 8100 | 12 no-spec, 11 DSpark, 12 router | 1536, 1408, 1536 | 0 |

## Problems

- A synthetic Qwen-tokenized prompt decoded to a `race race ...` loop; discarded.
- `bench_throughput.py` stopped on EOS and produced unequal output counts; discarded.
- The first DSpark restart raced GPU memory release and OOMed; the clean retry passed.
- One interrupted 8K command targeted the 512 output directory. It was killed
  before finalization; the original JSON timestamp and size remained unchanged.
- DSpark acceptance at 512 was 147/515 (28.5%), yet wall-clock throughput lost
  23.5%; acceptance is not a speed license.

## Learnings

Full-context draft attention makes DSpark prompt-shape dependent. On this H20
TP=4 configuration, 64 prompt tokens is the last measured win. Decide once from
the original prompt length; switching on growing decode length would destroy the
short-prompt gain.

## Artefacts

- Raw JSON/CSV: `/host/arle-dspark-bench/bench-output/2026-07-14-{nospec,dspark-old,dspark-router64}-natural*`
- Dataset SHA-256: 32 `7986875c...239ea`; 64 `b202d16f...22af4`;
  128 `169b7c78...a0f5f`; 256 `8e657e05...2dabd`;
  512 `1a577db5...0cc3`; 8K `ddf60194...e1b3`

## Notes

- `--dspark-max-prompt-tokens` defaults to unset, preserving prior behavior.
- The threshold is measured for this model, output length, and H20 TP=4. Other
  deployments must rerun the crossover sweep.
- Cold-load I/O is unchanged by routing because the draft still loads on every
  rank. Draft sharding is the next structural optimization.
