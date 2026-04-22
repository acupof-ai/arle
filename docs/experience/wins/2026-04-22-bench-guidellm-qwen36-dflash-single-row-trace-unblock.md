# qwen36 dflash single-row trace unblock + host acceptance cleanup

## Goal

Unblock `QWEN35_DFLASH_PROFILE=1` on the single-row Qwen3.6 Metal DFlash path
 and remove the last single-row-only MLX prefix-match helper from acceptance.

## Hypothesis

Two issues were blocking useful single-row trace work:

1. The Rust MLX wrapper only recognized a small subset of MLX dtypes, so
   `QWEN35_DFLASH_PROFILE=1` could panic on valid `int64` arrays before the
   timed request finished.
2. Single-row acceptance still routed through a dedicated MLX prefix-match
   kernel and a second helper that rebuilt the accepted token slice as arrays,
   even though the control path only compares a single `15`-token draft prefix.

The right cleanup was:

- make the Rust `Dtype` enum match the bridge constants closely enough for
  tracing/debugging, and
- after sampled verify, materialize the single-row posterior block once,
  compare the prefix on the host, and assemble accepted tokens directly.

## Params

- Target model:
  `mlx-community/Qwen3.6-35B-A3B-4bit`
- Draft model:
  `z-lab/Qwen3.6-35B-A3B-DFlash`
- Prompt:
  raw `"hi"`
- Generation:
  `64`
- DFlash block size:
  `16`

## Env

- Host: `Apple M4 Pro`
- OS: `macOS 26.3.1`
- Build flags: `--release --no-default-features --features metal,no-cuda`

## Results

### Trace unblock

Before this change:

```bash
env QWEN35_DFLASH_PROFILE=1 target/release/metal_request \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --prompt hi --raw-prompt --max-new-tokens 64 \
  --dflash-draft-model z-lab/Qwen3.6-35B-A3B-DFlash
```

Result:

- panicked in `MlxArray::dtype()`
- message: `mlx_array_dtype returned unknown dtype 8`

After this change, the same command completes and emits the DFlash profile
window:

- `qwen35_dflash[agg 50 blocks]`
- `qwen35_dflash[agg K-hist]`
- `qwen35_dflash[agg pos-agree]`

### Single-row exactness guard

Command:

```bash
QWEN35_MODEL_PATH=$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3 \
QWEN35_DFLASH_DRAFT_PATH=$HOME/.cache/huggingface/hub/models--z-lab--Qwen3.5-4B-DFlash/snapshots/96899cc270945f554998309580b08a04a05a3187 \
cargo +stable test -p infer --release --lib --no-default-features --features metal,no-cuda \
  backend::metal::dflash::tests::dflash_qwen35_verify_batched_matches_two_single_row_runs \
  -- --exact --nocapture
```

Result:

- `1 passed`
- `accepted_inputs=[1, 1]`
- `overall_max_abs_delta=0`
- `finished in 1.55s`

### Dtype guard

Command:

```bash
cargo +stable test -p infer --release --lib --no-default-features --features metal,no-cuda \
  backend::metal::mlx::tests::dtype_supports_int64_roundtrip \
  -- --exact --nocapture
```

Result:

- `1 passed`
- `finished in 0.14s`

### Single-row DFlash profile run

Command:

```bash
cargo +stable build -p infer --release --no-default-features --features metal,no-cuda --bin metal_request
env QWEN35_DFLASH_PROFILE=1 target/release/metal_request \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --prompt hi --raw-prompt --max-new-tokens 64 \
  --dflash-draft-model z-lab/Qwen3.6-35B-A3B-DFlash
```

Result:

- `TTFT: 13.7 ms`
- `Gen TPS: 22.5 tok/s`
- `matched K̄=0.60/15`
- `tok/block=1.60`

## Problems

- The trace data says the next real blocker is no longer a Rust-side prefix
  helper. The single-row path is still dominated by sampled verify completion,
  and the acceptance shape itself is poor (`K0` on `64%` of blocks in this run).
- Local standalone `metal_bench` binary reruns are still blocked by unrelated
  compile errors outside this line, so this slice uses `metal_request` plus the
  exactness test as its verification envelope.

## Learnings

- For single-row `block_size=16`, keeping acceptance on host is cleaner than
  launching a dedicated MLX prefix-match kernel for one short prefix.
- The MLX wrapper has to recognize the common integer/floating dtypes or the
  trace/debug path becomes less trustworthy than the hot path it is supposed to
  explain.
