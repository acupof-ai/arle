# Metal DFlash

User guide for running DFlash speculative decoding on the Metal backend.

Status: experimental.

## What it is

Metal DFlash adds a draft-model-assisted decode path on Apple Silicon. The
target model still produces the final tokens; the draft model proposes a block
of candidate tokens, and the target verifies them.

Today this implementation is intended for:

- Apple Silicon
- `--features metal,no-cuda`
- `Qwen3` targets only

It is not a generic speculative-decoding surface yet.

Parameter reference:

- [metal-dflash-params.md](metal-dflash-params.md)

## Current support

Supported now:

- Backend: Metal
- Model family: `Qwen3`
- Entry points:
  - `metal_request`
  - `metal_bench`
  - `metal_serve`

Not supported yet:

- `Qwen3.5`
- CUDA server scheduler integration
- Claims of universal speedup on every workload

Important limitation:

- The current Metal server is serial. DFlash improves single-request decode,
  but it does not turn `metal_serve` into a batched serving runtime.

## Validated model pair

Locally validated pair:

- Target: `mlx-community/Qwen3-4B-bf16`
- Draft: `z-lab/Qwen3-4B-DFlash-b16`

Observed local result on Apple M4 Pro:

- `prompt=20`, `generation=256`
- baseline `generation_tps = 25.9`
- DFlash `generation_tps = 152.0`
- about `5.9x` decode throughput

Raw benchmark record:

- [experience/wins/2026-04-14-metal-dflash-qwen3.md](../experience/wins/2026-04-14-metal-dflash-qwen3.md)

## Build

```bash
cargo build -p infer --release --no-default-features --features metal,no-cuda
```

## One-shot generation

Use `metal_request` when you want a single prompt/response run:

```bash
cargo run -p infer --bin metal_request --release --no-default-features --features metal,no-cuda -- \
  --model mlx-community/Qwen3-4B-bf16 \
  --dflash-draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --prompt "write a quicksort in python" \
  --raw-prompt \
  --max-new-tokens 128
```

## Benchmarking

Use `metal_bench` to compare baseline vs. DFlash on the same workload.

Baseline:

```bash
cargo run -p infer --bin metal_bench --release --no-default-features --features metal,no-cuda -- \
  --model mlx-community/Qwen3-4B-bf16 \
  --prompt-tokens 20 \
  --generation-tokens 256 \
  --warmup 1 \
  --runs 3
```

DFlash:

```bash
cargo run -p infer --bin metal_bench --release --no-default-features --features metal,no-cuda -- \
  --model mlx-community/Qwen3-4B-bf16 \
  --dflash-draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --prompt-tokens 20 \
  --generation-tokens 256 \
  --warmup 1 \
  --runs 3
```

Use generation-heavy workloads first. That is where the current Metal DFlash
path is already validated.

## Serving

`metal_serve` exposes the same DFlash controls as the request/bench tools:

```bash
./target/release/metal_serve \
  --model-path mlx-community/Qwen3-4B-bf16 \
  --dflash-draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --warmup 1 \
  --port 8000
```

## Configuration

### `--dflash-draft-model`

Enables DFlash and points the Metal backend at the draft checkpoint.

Accepted values:

- local model directory
- Hugging Face repo id

### `--speculative-tokens`

Optional block-size override.

Recommendation:

- leave this unset unless you have benchmark data

Why:

- the draft checkpoint already carries a trained default block size
- lowering it can reduce acceptance and throughput
- current runtime will warn if you force a smaller value than the draft default

## Recommended usage pattern

Use DFlash first when:

- the workload is decode-heavy
- prompt is modest and generation is long
- you are validating local Apple Silicon throughput

Do not assume it helps when:

- prompt and generation are balanced
- the prompt is long enough that prefill dominates total wall time
- the target is `Qwen3.5`

## Known limitations

- `Qwen3.5` is intentionally rejected today because recurrent rollback is not
  integrated into the Metal DFlash path yet.
- The current implementation is validated on `Qwen3-4B`; larger `Qwen3`
  targets may work, but should be benchmarked explicitly before making claims.
- Draft-model checkpoints may not ship tokenizer files; this is expected. The
  target model tokenizer remains the source of truth.

## Troubleshooting

If throughput is poor:

1. Remove `--speculative-tokens` and rerun with the draft default.
2. Re-run the same workload as a baseline without DFlash.
3. Compare `generation_tps`, not only total wall time.
4. Prefer `prompt=20`, `generation=256` or another decode-heavy benchmark for
   first-pass validation.

If the backend refuses to load:

1. Check that the target model is `Qwen3`, not `Qwen3.5`.
2. Check that the draft hidden size matches the target hidden size.
3. Rebuild with `--no-default-features --features metal,no-cuda`.
