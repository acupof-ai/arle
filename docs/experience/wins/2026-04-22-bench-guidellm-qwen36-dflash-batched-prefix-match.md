# qwen36 dflash batched prefix-match on GPU

## Goal

Delete the last row-by-row prefix-match loop on the packed Qwen3.5/Qwen3.6
Metal DFlash verify path and replace it with one GPU batched primitive.

## Hypothesis

After `88bdd23`, packed verify still sliced each row of the sampled posterior,
ran `prefix_match_len_i32` once per row, and paid one host sync per row for
the accepted-prefix length.

`dflash-mlx` keeps this comparison on device. A native `[B, T] -> [B]`
prefix-match kernel should let Rust materialize only:

1. the matched-length vector, and
2. the single posterior token each row actually emits.

That should shrink packed verify wall time without changing acceptance or
hidden-state numerics.

## Params

- Target model:
  `$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`
- Draft model:
  `$HOME/.cache/huggingface/hub/models--z-lab--Qwen3.5-4B-DFlash/snapshots/96899cc270945f554998309580b08a04a05a3187`
- Qwen3.6 smoke target:
  `mlx-community/Qwen3.6-35B-A3B-4bit`
- DFlash block size:
  `16`

## Env

- Host: `Apple M4 Pro`
- OS: `macOS 26.3.1`
- Build flags: `--release --no-default-features --features metal,no-cuda`

## Results

### New bridge primitive

- Added `mlx_prefix_match_len_i32_batched(lhs, rhs)` to `mlx-sys`
- Contract: int32 contiguous `[B, T]` + `[B, T]` in, int32 `[B]` matched prefix
  lengths out

### Targeted packed-DFlash exact test

Command:

```bash
QWEN35_MODEL_PATH=$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3 \
QWEN35_DFLASH_DRAFT_PATH=$HOME/.cache/huggingface/hub/models--z-lab--Qwen3.5-4B-DFlash/snapshots/96899cc270945f554998309580b08a04a05a3187 \
cargo +stable test -p infer --release --lib --no-default-features --features metal,no-cuda \
  backend::metal::dflash::tests::dflash_qwen35_verify_batched_matches_two_single_row_runs \
  -- --exact --nocapture
```

Before this change:

- `test result: ... finished in 1.82s`

After this change:

- cold first run: `test result: ... finished in 3.86s`
- warm rerun: `test result: ... finished in 0.98s`

Steady-state delta vs the prior committed packed path:

- `1.82s -> 0.98s` on the same exactness test
- `-46.2%` wall time on the warm rerun

Correctness from the same command:

- `accepted_inputs=[1, 1]`
- `overall_max_abs_delta=0`
- `1 passed`

### Bridge correctness test

Command:

```bash
cargo +stable test -p infer --release --lib --no-default-features --features metal,no-cuda \
  backend::metal::mlx::tests::prefix_match_len_i32_batched_counts_each_row_prefix \
  -- --exact --nocapture
```

Result:

- `1 passed`
- `finished in 0.11s`

### Related packed-verify guard

Command:

```bash
QWEN35_MODEL_PATH=$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3 \
cargo +stable test -p infer --release --lib --no-default-features --features metal,no-cuda \
  backend::metal::qwen35::tests::verify_block_batched_matches_independent_verify_block_for_b2 \
  -- --exact --nocapture
```

Result:

- `1 passed`
- `test result: ... finished in 0.45s`

### Qwen3.6 no-regression checks

Command:

```bash
cargo +stable test -p infer --release --no-default-features --features metal,no-cuda \
  --bin metal_bench tests::baseline_compare_default_draft_qwen36_a3b -- --exact
```

Result:

- `1 passed`

Command:

```bash
cargo +stable run -p infer --release --no-default-features --features metal,no-cuda \
  --bin metal_request -- \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --prompt hi --raw-prompt --max-new-tokens 2 \
  --dflash-draft-model z-lab/Qwen3.6-35B-A3B-DFlash
```

Result:

- `exit 0`
- `Metal DFlash enabled`
- `Output tokens: 2`

## Problems

- A speculative `take_axis(axis=1)` gather cleanup was tested first and
  removed. On MLX it produced shape `[B, B, 1]` in the local test instead of
  the row-wise `[B, 1]` gather this path needed, so it was a wrong abstraction
  for posterior-token extraction.
- Building the standalone `metal_bench` binary for a fresh serial throughput
  rerun is currently blocked by unrelated local `kv_tier` type errors in
  `infer/src/kv_tier/coordinator.rs`. This slice therefore uses targeted
  packed-DFlash exact tests plus Qwen3.6 smoke as the validation envelope.

## Learnings

- The profitable follow-through on the previous posterior-readback cleanup was
  not another gather trick; it was deleting the remaining per-row prefix-match
  loop entirely.
- `dflash-mlx` was directionally right here: acceptance wants one device-side
  batched primitive, then the host should only touch the scalar results it
  must emit or branch on.
