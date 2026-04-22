# qwen36 dflash batched verify host cache_pos + packed draft sampling

## Goal

Reduce batched Metal DFlash verify host-side fences on the Qwen3.5/Qwen3.6
packed path without changing numerics.

## Hypothesis

Two hot spots were still visible in code/trace:

1. Batched verify passed `cache_pos_arr` as an MLX array, and
   `mlx_qwen35_model.cpp` eagerly `eval`ed it to read host ints.
2. Batched draft sampling still looped row-by-row, doing
   `linear -> sample_rows_array -> materialize_token_array` once per row.

Passing `cache_pos_arr` as a host `int32_t*` and sampling the whole packed
draft suffix in one pass should reduce per-block host overhead while keeping
the packed DFlash contract bit-exact.

## Params

- Qwen3.5 target model:
  `$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`
- Qwen3.5 DFlash draft:
  `$HOME/.cache/huggingface/hub/models--z-lab--Qwen3.5-4B-DFlash/snapshots/96899cc270945f554998309580b08a04a05a3187`
- Qwen3.6 smoke target:
  `mlx-community/Qwen3.6-35B-A3B-4bit`
- Draft block size: `16`

## Env

- Host: `Apple M4 Pro`
- OS: `macOS 26.3.1`
- Build flags: `--release --no-default-features --features metal,no-cuda`

## Results

### Batched verify microbench

Command:

```bash
QWEN35_MODEL_PATH=$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3 \
cargo +stable test -p infer --release --lib --no-default-features --features metal,no-cuda \
  backend::metal::qwen35::tests::verify_block_batched_matches_independent_verify_block_for_b2 \
  -- --exact --nocapture
```

Before this change:

- `test result: ... finished in 1.63s`

After this change:

- first rerun: `test result: ... finished in 0.97s`
- second rerun: `test result: ... finished in 0.49s`

Interpretation:

- the exact packed verify path got materially faster after removing the
  `cache_pos_arr` MLX-array round-trip
- the second rerun is warmer than the first, but both post-change runs are
  below the pre-change `1.63s`

### Packed DFlash correctness

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
- `overall_max_abs_delta=0`
- `finished in 2.61s`

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
- `TTFT: 184.4 ms`

## Problems

- The more aggressive “same-length packed verify can skip explicit attn mask”
  cleanup was not numerically stable enough for this path: it switched the C++
  full-attention route from masked SDPA to causal SDPA and broke the exact
  `updated_target_hidden` equivalence test. That experiment was reverted.
- A global “force M=16 to qmm/gather_qmm” MLX heuristic change also regressed
  local throughput and was reverted.

## Learnings

- The low-risk packed-path wins are the ones that delete host fences without
  changing attention math.
- `cache_pos_arr` did not need to be an MLX array on the packed verify path;
  carrying it as a host slice is simpler and faster.
- Batched draft sampling was cleaner as one packed `linear + sample` pass than
  a per-row loop.
