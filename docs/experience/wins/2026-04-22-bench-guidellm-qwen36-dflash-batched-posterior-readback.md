# qwen36 dflash batched posterior readback shrink

## Goal

Remove the last obvious full-matrix CPU readback in batched Metal DFlash
verify after commit `17174b0`.

## Hypothesis

`qwen35_dflash_speculative_block_batched` still did:

1. `eval(&posterior_tokens)`
2. `posterior_tokens.as_slice_i32()`
3. CPU prefix matching over the whole `[B, block_size]` matrix

That should be avoidable. The packed verifier already has both the packed
draft tokens and the sampled posterior tokens on device, so it should be able
to do row-wise prefix matching on GPU and materialize only one scalar token per
row.

## Params

- Qwen3.5 target:
  `$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`
- Qwen3.5 draft:
  `$HOME/.cache/huggingface/hub/models--z-lab--Qwen3.5-4B-DFlash/snapshots/96899cc270945f554998309580b08a04a05a3187`

## Env

- Host: `Apple M4 Pro`
- OS: `macOS 26.3.1`
- Build flags: `--release --no-default-features --features metal,no-cuda`

## Results

### Targeted packed-DFlash microbench

Command:

```bash
QWEN35_MODEL_PATH=$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3 \
QWEN35_DFLASH_DRAFT_PATH=$HOME/.cache/huggingface/hub/models--z-lab--Qwen3.5-4B-DFlash/snapshots/96899cc270945f554998309580b08a04a05a3187 \
cargo +stable test -p infer --release --lib --no-default-features --features metal,no-cuda \
  backend::metal::dflash::tests::dflash_qwen35_verify_batched_matches_two_single_row_runs \
  -- --exact --nocapture
```

Before this change:

- `test result: ... finished in 2.61s`

After this change:

- `test result: ... finished in 1.82s`

Delta:

- `-30.3%` wall time on the targeted packed DFlash verify test

### Correctness guard

Same command as above.

Result:

- `1 passed`
- `overall_max_abs_delta=0`
- accepted inputs stayed `[1, 1]`

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
- `test result: ... finished in 0.49s`

## Problems

- A more aggressive “skip explicit packed verify mask when left_padding == 0”
  cleanup looked tempting earlier, but it changed the SDPA route and broke
  exact hidden-state equivalence. That remains reverted.

## Learnings

- The profitable packed-path cleanup was not “remove all host reads”; it was
  “remove the wide host read and keep only the per-row scalars the caller
  actually needs.”
- Keeping prefix matching on GPU and materializing only one posterior token per
  row is a clean follow-through on the earlier packed verify host-fence work.
