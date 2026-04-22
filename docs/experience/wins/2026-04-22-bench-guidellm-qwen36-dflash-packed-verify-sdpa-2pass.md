# Qwen3.6 DFlash Packed Verify SDPA 2-Pass Hook

## Goal

- Optimization / alignment: hook the existing `batched_sdpa_2pass` Metal
  kernel into the compiled Qwen3.6/Qwen3.5 verify hot path, but only for the
  packed batched verify case where the kernel contract is actually satisfied.

## Hypothesis

- If packed verify can replace stock MLX SDPA with the fixed-`M=16` 2-pass
  kernel on full-attention sublayers, the packed DFlash lane moves closer to
  `dflash-mlx` without risking regressions on the single-row local path.

## Command

```bash
cargo +stable test -p infer --release --no-default-features --features metal,no-cuda --lib backend::metal::mlx::tests::batched_sdpa_2pass_matches_causal_sdpa_for_m16 -- --exact
```

```bash
cargo +stable test -p infer --release --no-default-features --features metal,no-cuda --lib backend::metal::qwen35::tests::verify_block_sampled_matches_batched_sampled_for_b1 -- --exact
```

```bash
cargo +stable test -p infer --release --no-default-features --features metal,no-cuda --lib backend::metal::qwen35::tests::verify_block_batched_matches_independent_verify_block_for_b2 -- --exact
```

```bash
cargo +stable test -p infer --release --no-default-features --features metal,no-cuda --bin metal_bench tests::baseline_compare_default_draft_qwen36_a3b -- --exact
```

```bash
cargo +stable clippy -p infer --release --no-default-features --features metal,no-cuda --bin metal_bench --bin metal_request -- -D warnings
```

```bash
cargo +stable run -p infer --release --no-default-features --features metal,no-cuda --bin metal_request -- --model /path/to/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46 --prompt hi --raw-prompt --max-new-tokens 2 --dflash-draft-model z-lab/Qwen3.6-35B-A3B-DFlash
```

```bash
target/release/metal_bench \
  --model /path/to/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46 \
  --use-step-driver \
  --ignore-eos \
  --prompt-tokens 20 \
  --generation-tokens 1024 \
  --warmup 1 \
  --runs 3 \
  --json
```

```bash
target/release/metal_bench \
  --model /path/to/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46 \
  --dflash-draft-model z-lab/Qwen3.6-35B-A3B-DFlash \
  --use-step-driver \
  --ignore-eos \
  --prompt-tokens 20 \
  --generation-tokens 1024 \
  --warmup 1 \
  --runs 3 \
  --json
```

## Environment

- **Backend:** Metal
- **Model:** `mlx-community/Qwen3.6-35B-A3B-4bit`
- **Draft:** `z-lab/Qwen3.6-35B-A3B-DFlash`
- **Hardware:** `Apple M4 Pro`
- **OS:** `macOS 26.3.1 (25D771280a)`
- **Commit base:** `88970ad`
- **Feature set:** `--no-default-features --features metal,no-cuda`
- **Non-default flags / env vars:** none

## Results

Correctness / validation:

- `batched_sdpa_2pass_matches_causal_sdpa_for_m16`: `1 passed`
- `verify_block_sampled_matches_batched_sampled_for_b1`: `1 passed`
- `verify_block_batched_matches_independent_verify_block_for_b2`: `1 passed`
- `metal_bench` unit test: `1 passed`
- `clippy -D warnings`: passed
- `metal_request` smoke: `exit 0`, `Output tokens: 2`

Single-row regression check on the local harness:

| mode | gen tok/s | repo e2e tok/s | TTFT ms |
|---|---:|---:|---:|
| baseline | 81.16 | 80.72 | 69.06 |
| DFlash now | 48.24 | 48.07 | 72.97 |

Additional DFlash stats:

| block_size | avg accepted inputs | acceptance rate |
|---:|---:|---:|
| 16 | 2.50 | 59.96% |

## Problems

- The first attempt at this slice applied `batched_sdpa_2pass` to single-row
  verify as well. On this M4 Pro local lane that was a clear regression, so
  the final code gates the 2-pass hook to packed batched verify only.
- There is no dedicated local packed-DFlash throughput harness in this tranche.
  `metal_bench --use-step-driver` only exercises the single-row path, so the
  table above is a regression check, not a direct measurement of the new
  packed-only fast path.
- Because the packed path is out of scope for that harness, the single-row
  number above should be read only as “no single-row optimization was kept in
  this slice”, not as the payoff of the packed hook itself.

## Learnings

- The right stable cut on this machine is not “enable 2-pass everywhere”.
  The kernel belongs behind an explicit packed-verify gate, not a blanket
  `seq_len == 16` heuristic.
- An explicit verify-context bit in `ForwardContext` keeps the dispatch clean:
  prefill and normal decode cannot accidentally fall into the 2-pass path.
- Factoring `batched_sdpa_2pass_cpp` into a shared array-native helper keeps
  the bridge and compiled-model path aligned without duplicating kernel logic.

## Δ vs baseline

- **Prior committed local snapshot:** [2026-04-22-bench-guidellm-qwen36-dflash-scalar-verify-api.md](./2026-04-22-bench-guidellm-qwen36-dflash-scalar-verify-api.md)
- **Same-session baseline:** included in the main results table above

| metric | prior committed local DFlash | now | Δ% |
|---|---:|---:|---:|
| generation_tps | 51.00 | 48.24 | -5.4% |
| repo_e2e_tps | 50.82 | 48.07 | -5.4% |
| TTFT ms | 70.43 | 72.97 | +3.6% |

## Notes

- Final code in this tranche:
  - threads an explicit `is_verify` bit through the compiled Qwen3.5/Qwen3.6 model
  - factors `batched_sdpa_2pass_cpp` into a shared C++ helper
  - enables the 2-pass kernel only for packed batched verify with no additive mask
  - leaves single-row verify on the prior scalar sampled API
