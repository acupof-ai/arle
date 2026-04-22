# qwen36 dflash batched posterior gather

## Goal

Delete the last per-row posterior-token extraction loop in packed
Qwen3.5/Qwen3.6 Metal DFlash verify.

## Hypothesis

After commit `4ef3b88`, packed verify already computed matched prefix lengths
for every row in one GPU pass, but it still sliced `posterior_tokens[b]` and
materialized one scalar token per row on the host.

A dedicated `[B, T] + [B] -> [B]` gather kernel should let the packed path
materialize the matched lengths and emitted posterior tokens in one shared
`eval`, removing the last row loop from acceptance.

## Params

- Target model:
  `$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`
- Draft model:
  `$HOME/.cache/huggingface/hub/models--z-lab--Qwen3.5-4B-DFlash/snapshots/96899cc270945f554998309580b08a04a05a3187`
- DFlash block size:
  `16`

## Env

- Host: `Apple M4 Pro`
- OS: `macOS 26.3.1`
- Build flags: `--release --no-default-features --features metal,no-cuda`

## Results

### New bridge primitive

- Added `mlx_gather_axis1_i32(values, indices)` to `mlx-sys`
- Contract: int32 `values[B, T]` + int32 `indices[B]` in, int32 `out[B]` out

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

- `test result: ... finished in 0.98s`

After this change:

- `test result: ... finished in 0.85s`

Delta:

- `-13.3%` wall time on the exact packed DFlash equivalence test

Correctness from the same command:

- `accepted_inputs=[1, 1]`
- `overall_max_abs_delta=0`
- `1 passed`

### Bridge correctness test

Command:

```bash
cargo +stable test -p infer --release --lib --no-default-features --features metal,no-cuda \
  backend::metal::mlx::tests::gather_axis1_i32_picks_one_value_per_row \
  -- --exact --nocapture
```

Result:

- `1 passed`
- `finished in 0.26s`

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
- `test result: ... finished in 0.46s`

## Problems

- The first version failed because `verify_block_batched_sampled` does not
  guarantee `posterior_tokens` arrive as int32. The fix was to cast once to
  `Int32` at the packed DFlash boundary and keep the gather kernel itself
  dtype-simple.
- Fresh standalone `metal_bench` binary rebuilds remain blocked by unrelated
  local `kv_tier` compile errors, so this slice is validated with the packed
  exact tests instead of a new serial end-to-end throughput run.

## Learnings

- The right cleanup was not another reshape/take trick; it was a tiny
  dedicated gather primitive that matches the acceptance contract exactly.
- At this point, packed DFlash acceptance is structurally aligned with
  `dflash-mlx`: one GPU pass for prefix lengths, one GPU pass for emitted
  posterior tokens, then one shared materialization of the two `[B]` result
  vectors.
