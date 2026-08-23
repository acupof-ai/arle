# DSv4 precision matrix: NVFP4 wins c=1, quantized KV is not wired — CUDA, 2026-08-24

> Status: Characterized. NVFP4 experts + BF16 KV is the c=1 champion; the four
> quantized-KV arms cannot load by design.

## Goal

Bench and eval DSv4-Flash across every precision configuration the runtime
admits: {NVFP4, FP8} experts x {bf16, fp8, int8} KV.

## Parameters

```bash
arle serve --backend cuda --model-path <ckpt> --tensor-parallel-size 4 \
  --kv-cache-dtype <kv> --spec-type none --port 8300
python3 scripts/bench_throughput.py --concurrency-grid 1,8 \
  --requests-per-concurrency 16 --max-tokens 256 --temperature 0 \
  --prompts-jsonl bench-agent-32k-16x8.jsonl
arle_capability_eval_local.py --tasks mmlu --n-samples 200 --concurrency 1 --seed 0
```

- Binary `gatemiss-v3`; 4xH20 (GPUs 0,1,2,4), TP=4, `--comm-backend nccl`
- Checkpoints `/data00/DeepSeek-V4-Flash-0731` (NVFP4 experts, 156 GB) and
  `-0731-FP8` (287 GB)
- Prompt p50 28568 tok, completions 256 exact

## Results

| experts | KV | status | captures | c=1 decode tok/s | c=1 ITL p50/p99 ms | c=8 decode tok/s | MMLU (200) |
|---|---|---|---:|---:|---|---:|---|
| NVFP4 | bf16 | OK | 76 | **44.4** | **22.2 / 42.0** | 27.0 | 0.855 |
| FP8 | bf16 | OK | 0 | 26.5 | 37.0 / 74.3 | 25.7 | 0.860 |
| NVFP4 | fp8 | LOAD_FAILED | — | — | — | — | — |
| NVFP4 | int8 | LOAD_FAILED | — | — | — | — | — |
| FP8 | fp8 | LOAD_FAILED | — | — | — | — | — |
| FP8 | int8 | LOAD_FAILED | — | — | — | — | — |

Accuracy is a wash: 0.855 vs 0.860 on 200 items is one item, inside the noise
of that sample size. NVFP4 is 1.68x the FP8 checkpoint at c=1 and the two
converge by c=8, which is the compute-bound crossover the family shows
everywhere else.

## Problems

**The four quantized-KV arms cannot load, by design.** Every rank bails at
engine construction:

```
--kv-cache-dtype fp8 is not supported for Dsv4; only Qwen35 supports quantized paged KV
```

`crates/infer-api/src/loaded.rs:2054` admits a non-BF16 KV dtype only for
`CudaModelKind::Qwen35`. `docs/support-matrix.md` claimed "INT8/FP8 paged
quant-KV dispatch landed opt-in, correctness licensed (#68)", which is what
made this a six-arm matrix instead of a two-arm one; the entry is corrected in
the same change. `crates/cli/src/args.rs:944` had it right all along ("wired
one mode at a time under #68 T3").

The coordinator hides this: all four workers die before the boot-ping, so the
only surfaced error is `RelayCoordinator write envelope to worker rank 0:
relay write payload: Broken pipe`. The real message is in the worker stdout,
interleaved four ways because all four ranks write it concurrently.

**The FP8 arm captured zero graphs — a regression introduced hours earlier.**
`16857e541` made an allocating capture fatal. NVFP4 captures at 0 alloc nodes
and was verified that way, but the FP8 checkpoint still records 86 (2 per
layer: the O-LoRA staging resizes per layer rather than once), so every
capture was rejected and the path fell back to eager. Cost: 59.5 -> 26.5 tok/s
at c=1. `ebb1dd89b` restores the warn on that construction site.

## Learnings

**NVFP4 experts are the c=1 configuration for DSv4-Flash**: 44.4 vs 26.5
tok/s, ITL p99 42.0 vs 74.3 ms, with no accuracy cost on MMLU. The FP8
checkpoint is also 1.8x the bytes (287 vs 156 GB).

**A matrix arm that cannot load is worth running anyway.** The four failures
took under two minutes each and produced the finding: a support-matrix claim
that had been wrong long enough to shape an experiment design.

**A strict audit verified on one checkpoint is not verified for the family.**
The alloc-node gate was measured at zero on NVFP4 and shipped as a default;
the FP8 checkpoint of the same model, on the same code path, still allocated
86 nodes. Nothing about "DSv4 captures cleanly" generalized across the
quantization axis, and the run that would have caught it was the one this
matrix ran.

Open: the O-LoRA staging resize thrash on the FP8 path. `in_len`/`out_len`
are passed into `decode_proj_deepgemm_raw`, so a max-sized buffer reused
across layers needs those to be bounds rather than extents — unverified.

## Artifacts

- `/host/arle-ops/runs/c1g/matrix/matrix.tsv`, `bench-*.json`, `mmlu-*/`
- `/host/arle-ops/runs/c1g/kvprobe/serve-{fp8,int8}.log`
