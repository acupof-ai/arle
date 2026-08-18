# MXFP4 W4A16 on Metal — Qwen3.5-9B pilot — Metal, 2026-08-18

> Status: Shipped (feature). Replacement of the affine-4bit ladder row: Rejected.

## Goal

Validate MXFP4 weight quantization on the Metal backend as a replacement for
the affine-4bit Qwen3.5-9B row in `scripts/bench_local_metal_all.sh`, gated on
the needle ladder (115–8000 tokens) and a matched c=1 decode A/B.

## Hypothesis

MXFP4 (E2M1 weights, one E8M0 power-of-two scale per 32-element block, no
bias) removes the per-group bf16 scale+bias traffic of affine 4-bit, so decode
should be faster at equal generation quality.

## Parameters

- Needle gate: `RAW=1 TEMPLATE=qwen3_nonthink python3 scripts/needle_gate.py`
  (lengths 115,180,241,300,446,1000,2000,4000,8000; 3 runs; depth 0.0).
- Decode A/B: `python3 scripts/bench_local_metal.py <url> <id> 512 128 6`
  (512 prompt tokens, 128 generated, 6 trials, c=1).
- Baseline: `mlx-community/Qwen3.5-9B-MLX-4bit` (affine, group 64, int8 KV).
- Treatment: `models/Qwen3.5-9B-MXFP4` (converted from the BF16 checkpoint with
  `mlx_lm.convert`, mode mxfp4, group 32).
- Server flags: `--max-prompt-tokens 6144 --max-total-tokens 8192
  --max-running-requests 1`; KV dtype varied as noted below.

## Environment

- Host: Mac (Metal), 48 GB, memory-pressure guard active.
- arle release build, `--features metal,no-cuda`, this commit.
- Reference arm: stock `mlx_lm` 0.31.2 loading the same converted checkpoint.

## Results

Needle ladder (exact / partial / miss per length, 3 runs):

| length | affine 4bit, int8 KV | MXFP4, int8 KV | MXFP4, bf16 KV |
|---:|---|---|---|
| 115–1000 | 3/0/0 | 3/0/0 | 3/0/0 |
| 2000 | 3/0/0 | 3/0/0 | 3/0/0 |
| 4000 | 3/0/0 | 0/0/3 | 3/0/0 |
| 8000 | 3/0/0 | 0/0/3 | 0/0/3 |

(Each cell is exact/partial/miss across 3 runs. The affine column is the
established ladder-row baseline, re-checked at 8000 this session.)

The 8000 miss is a deterministic refusal ("I cannot provide the secret access
code…"). Stock mlx_lm on the same checkpoint, same prompt (8129 tokens),
refuses identically. The degradation is a property of the MXFP4 checkpoint at
this context length, reproduced without any arle code in the path.

Decode A/B (c=1, 512→128):

| arm | TTFT ms | TPOT ms | decode tok/s | e2e tok/s |
|---|---:|---:|---:|---:|
| affine 4bit | 1456.9 | 20.02 | 50.0 | 32.0 |
| MXFP4 | 1459.6 | 19.01 | 52.6 | 33.0 |

MXFP4 decodes 5.2% faster; TTFT is a wash.

Perplexity (wikitext-2 test, 297,047 tokens, seq 2048, stock mlx_lm stack):

| arm | PPL |
|---|---:|
| affine 4bit | 9.5037 |
| MXFP4 | 10.1214 |

MXFP4 pays +0.62 PPL (+6.5%) at 2K context. The E8M0 power-of-two scale is
coarse — small-magnitude weights inside a block share the block max's exponent
— and the error accumulates with depth into the 8K-token refusal above.

## Problems

- MXFP4 with the default int8 KV cache degrades between 2000 and 4000 tokens;
  with bf16 KV it holds to 4000. The affine baseline passes 8000 with int8 KV.
  The two quantizations compound, and the default KV dtype makes MXFP4 look
  worse than it is.
- The memory-pressure guard rejects the default slot count on this box;
  `--max-running-requests 1` drops static state from ~12.6 GiB to ~49 MiB.

## Learnings

MXFP4 support ships as an opt-in format: a checkpoint whose config declares
`"mode": "mxfp4"` loads and serves end-to-end, exact on the needle ladder
through 4000 tokens with bf16 KV, and 5% faster than affine 4bit on decode.
The affine path is unchanged (mode defaults to affine; the custom MMA2 kernel
serves affine only, MXFP4 uses the stock `quantized_matmul`).

The 9B ladder row stays affine: MXFP4 loses the 8000-token rung in both arle
and stock mlx_lm, so the swap trades a small decode gain for a long-context
regression. MXFP4 checkpoints on Metal should be served with `--kv-cache-dtype
bf16`.
