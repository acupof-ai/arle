# Metal Qwen3.6 35B-A3B DFlash Quick Check

## Goal

- Check the current Metal Qwen3.6-35B-A3B path after the Qwen3.5 GGUF decode
  work, and verify whether DFlash improves short decode on the local M4 Pro.

## Hypothesis

- The Qwen3.5 GGUF Q5/Q8 affine repack and Q6 qmv tile work should not be
  assumed to transfer directly to Qwen3.6, because this target is the
  MLX-community 4-bit MoE checkpoint rather than GGUF Q4_K_M.
- DFlash might improve decode TPOT, but Qwen3.6 prefill/TTFT is likely still
  the visible short-request bottleneck.

## Command

```bash
AGENT_INFER_QWEN35_GENERATE_PROFILE=1 ./target/release/metal_bench \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --prompt-tokens 32 \
  --generation-tokens 64 \
  --warmup 0 \
  --runs 1 \
  --json \
  --profile

./target/release/metal_bench \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --dflash-draft-model z-lab/Qwen3.6-35B-A3B-DFlash \
  --prompt-tokens 32 \
  --generation-tokens 64 \
  --warmup 0 \
  --runs 1 \
  --json \
  --profile

AGENT_INFER_QWEN35_GENERATE_PROFILE=1 ./target/release/metal_bench \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --prompt-tokens 32 \
  --generation-tokens 256 \
  --warmup 0 \
  --runs 1 \
  --json \
  --profile

./target/release/metal_bench \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --dflash-draft-model z-lab/Qwen3.6-35B-A3B-DFlash \
  --prompt-tokens 32 \
  --generation-tokens 256 \
  --warmup 0 \
  --runs 1 \
  --json \
  --profile

./target/release/metal_bench \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --baseline-compare \
  --prompt-tokens 32 \
  --generation-tokens 256 \
  --warmup 0 \
  --runs 1
```

## Environment

- Backend: Metal
- Target model: `mlx-community/Qwen3.6-35B-A3B-4bit`
- Draft model: `z-lab/Qwen3.6-35B-A3B-DFlash`
- Local cache: target present under HuggingFace cache (`19G`), draft present
  under HuggingFace cache (`1.8G`)
- Hardware: Apple M4 Pro, 20 GPU cores, 48 GB unified memory, Metal 4
- OS: Darwin 25.3.0 arm64
- Binary: existing `target/release/metal_bench`
- Workspace note: diagnostic quick check run from a dirty workspace with
  unrelated edits present. Treat this entry as directionally useful, not a
  canonical release benchmark.

## Results

| shape | mode | gen tok/s | repo e2e tok/s | prompt tok/s | TTFT ms | total ms | peak RSS MB |
|---|---|---:|---:|---:|---:|---:|---:|
| 32 / 64 | baseline | 63.187 | 4.872 | 2.639 | 12123.7 | 13136.6 | 6541.3 |
| 32 / 64 | DFlash | 66.374 | 5.746 | 3.145 | 10174.9 | 11139.2 | 9546.3 |
| 32 / 256 | baseline | 63.003 | 23.472 | 4.676 | 6843.4 | 10906.7 | 10817.9 |
| 32 / 256 | DFlash | 67.096 | 16.693 | 2.778 | 11520.7 | 15336.1 | 10228.8 |

`--baseline-compare` result:

```text
baseline gen_tps=65.7  DFlash gen_tps=64.7
compare | baseline TPOT 15.23 ms -> DFlash 15.45 ms  (delta +1.4%)
```

## Delta vs Baseline

| shape | metric | baseline | DFlash | delta |
|---|---|---:|---:|---:|
| 32 / 64 | gen tok/s | 63.187 | 66.374 | +5.0% |
| 32 / 64 | repo e2e tok/s | 4.872 | 5.746 | +17.9% |
| 32 / 256 | gen tok/s | 63.003 | 67.096 | +6.5% |
| 32 / 256 | repo e2e tok/s | 23.472 | 16.693 | -28.9% |
| baseline-compare | TPOT ms | 15.23 | 15.45 | +1.4% slower |

## Problems

- Single-run, no-warmup diagnostic only. This was intentionally short to
  answer whether Qwen3.6 shows an immediate win after the Qwen3.5 GGUF work.
- The direct JSON path did not emit DFlash acceptance counters for this run
  shape, so the TPOT compare is the cleaner decode-only signal.
- TTFT/prompt timing variance is large enough that short end-to-end results
  should not be used as a product claim.

## Learnings

- Qwen3.6-35B-A3B loads and runs on the Metal path locally, but DFlash is not
  a reliable win in this quick check: the paired TPOT compare is flat to
  slightly worse.
- The next Qwen3.6 target is prefill/TTFT and MoE/GDR execution, not more
  speculative-decode tuning.
- Do not infer Qwen3.6 gains from the Qwen3.5 GGUF Q4_K_M affine/tiled kernel
  work without a model-specific benchmark.
