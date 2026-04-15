# Metal DFlash Parameters

Parameter reference for the Metal DFlash entry points.

Scope:

- `metal_request`
- `metal_bench`
- `metal_serve`

This document is intentionally parameter-first. For support limits, benchmark
results, and usage guidance, see:

- [metal-dflash.md](metal-dflash.md)

## Shared DFlash parameters

These flags exist on all three Metal entry points.

| Parameter | Required | Default | Applies to | Meaning | Notes |
| --- | --- | --- | --- | --- | --- |
| `--dflash-draft-model <PATH_OR_REPO>` | No | disabled | `metal_request`, `metal_bench`, `metal_serve` | Enable Metal DFlash and load the draft checkpoint | Accepts a local path or Hugging Face repo id |
| `--speculative-tokens <N>` | No | draft-config default | `metal_request`, `metal_bench`, `metal_serve` | Override the speculative block size | Leave unset unless benchmark data says otherwise |

Rules:

- If `--dflash-draft-model` is omitted, the backend runs the normal Metal path.
- If `--speculative-tokens` is set lower than the draft checkpoint's default,
  runtime will warn because this can reduce acceptance and throughput.
- `--speculative-tokens 0` is invalid.

## `metal_request`

Purpose:

- one-shot prompt/response run
- smoke test
- quick local validation

Command shape:

```bash
cargo run -p infer --bin metal_request --release --no-default-features --features metal,no-cuda -- \
  --model mlx-community/Qwen3-4B-bf16 \
  --dflash-draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --prompt "write a quicksort in python" \
  --raw-prompt
```

| Parameter | Required | Default | Meaning | Notes |
| --- | --- | --- | --- | --- |
| `--model`, `-m <MODEL>` | Yes | none | Target model path or HF repo id | Example: `mlx-community/Qwen3-4B-bf16` |
| `--prompt <PROMPT>` | Conditionally | none | Inline prompt text | Mutually exclusive with `--prompt-file` |
| `--prompt-file <PATH>` | Conditionally | none | Read prompt from file | Mutually exclusive with `--prompt` |
| `--system <TEXT>` | No | none | System prompt for chat formatting | Ignored with `--raw-prompt` |
| `--raw-prompt` | No | `false` | Skip ChatML formatting | Recommended for low-level throughput checks |
| `--stream` | No | `false` | Stream text to stdout | Summary prints to stderr in stream mode |
| `--warmup <N>` | No | `1` | Warmup requests before the timed request | Use `0` for the fastest smoke test |
| `--max-new-tokens <N>` | No | `256` | Maximum generated tokens | Mapped to runtime `max_new_tokens` |
| `--temperature <F>` | No | `0.0` | Sampling temperature | `0.0` means greedy |
| `--top-k <K>` | No | `-1` | Top-k sampling | Metal currently accepts only `-1` or `1` |
| `--ignore-eos` | No | `false` | Keep generating past EOS | Useful for fixed-length measurement |
| `--dflash-draft-model <PATH_OR_REPO>` | No | disabled | Enable DFlash | Shared DFlash flag |
| `--speculative-tokens <N>` | No | draft default | Override block size | Shared DFlash flag |

Recommended smoke command:

```bash
cargo run -p infer --bin metal_request --release --no-default-features --features metal,no-cuda -- \
  --model mlx-community/Qwen3-4B-bf16 \
  --dflash-draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --prompt "benchmark throughput" \
  --raw-prompt \
  --warmup 0 \
  --max-new-tokens 16
```

## `metal_bench`

Purpose:

- baseline vs. DFlash throughput comparison
- TTFT / prompt TPS / generation TPS / repo-E2E TPS measurement

Command shape:

```bash
cargo run -p infer --bin metal_bench --release --no-default-features --features metal,no-cuda -- \
  --model mlx-community/Qwen3-4B-bf16 \
  --dflash-draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --prompt-tokens 20 \
  --generation-tokens 256 \
  --warmup 1 \
  --runs 3
```

| Parameter | Required | Default | Meaning | Notes |
| --- | --- | --- | --- | --- |
| `--model`, `-m <MODEL>` | Yes | none | Target model path or HF repo id | Same target meaning as `metal_request` |
| `--prompt-tokens <N>` | No | `20` | Exact prompt token count for the synthetic benchmark prompt | Use a longer prompt to test prefill-heavy cases |
| `--generation-tokens <N>` | No | `256` | Exact output token count | Alias: `--max-tokens` |
| `--warmup <N>` | No | `3` | Warmup runs excluded from stats | Use `1` for quick local comparisons |
| `--runs <N>` | No | `5` | Timed runs | Mean / p50 / p99 are computed across these |
| `--profile` | No | `false` | Print per-run detail | Useful when diagnosing variance |
| `--json` | No | `false` | Emit machine-readable JSON | Good for snapshotting |
| `--save-baseline <PATH>` | No | none | Write current results as a baseline JSON file | Does not compare |
| `--compare-baseline <PATH>` | No | none | Compare against an existing baseline | Fails if metrics regress past thresholds |
| `--update-baseline <PATH>` | No | none | Overwrite a baseline only if thresholds pass | Safe update flow |
| `--dflash-draft-model <PATH_OR_REPO>` | No | disabled | Enable DFlash | Shared DFlash flag |
| `--speculative-tokens <N>` | No | draft default | Override block size | Shared DFlash flag |

Recommended benchmark pair:

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

Recommended first workload:

- `--prompt-tokens 20`
- `--generation-tokens 256`

That is the currently validated generation-heavy case.

## `metal_serve`

Purpose:

- local OpenAI-compatible Metal server
- serial runtime validation

Command shape:

```bash
./target/release/metal_serve \
  --model-path mlx-community/Qwen3-4B-bf16 \
  --dflash-draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --port 8000
```

| Parameter | Required | Default | Meaning | Notes |
| --- | --- | --- | --- | --- |
| `--model-path <MODEL_PATH>` | Yes | none | Target model path or HF repo id | Server form uses `--model-path`, not `--model` |
| `--port <PORT>` | No | `8000` | HTTP listen port | OpenAI-compatible API |
| `--max-waiting <N>` | No | `256` | Max queued requests before rejection | Server is still serial, not batched |
| `--dflash-draft-model <PATH_OR_REPO>` | No | disabled | Enable DFlash | Shared DFlash flag |
| `--speculative-tokens <N>` | No | draft default | Override block size | Shared DFlash flag |

Important server limitation:

- `metal_serve` is still a serial runtime. DFlash improves the single-request
  decode path; it does not add CUDA-style continuous batching.

## Supported combinations

Working today:

- target family: `Qwen3`
- backend: `metal`
- build flags: `--no-default-features --features metal,no-cuda`

Rejected today:

- `Qwen3.5` + Metal DFlash
- DFlash without a draft model

## Removed environment variables

These DFlash environment variables are no longer the intended user interface:

- `AGENT_INFER_METAL_DFLASH_MODEL`
- `AGENT_INFER_METAL_DFLASH_SPECULATIVE_TOKENS`

Use explicit CLI flags instead.

## Practical defaults

Use these defaults unless you are actively benchmarking:

- `--dflash-draft-model`: set it explicitly
- `--speculative-tokens`: leave unset
- `metal_request --warmup`: `0` for smoke, `1` for normal usage
- `metal_bench --warmup`: `1`
- `metal_bench --runs`: `3`

## Troubleshooting by parameter

If DFlash does not activate:

1. Check `--dflash-draft-model`.
2. Check that the target is `Qwen3`.
3. Check that the build uses `--features metal,no-cuda`.

If throughput gets worse:

1. Remove `--speculative-tokens`.
2. Re-run the same command without `--dflash-draft-model`.
3. Compare `generation_tps` on the same workload.
