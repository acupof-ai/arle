# Qwen3.6 DFlash Verify Linear M=16 Dispatch

## Goal

- Optimization / alignment: keep the `Qwen3.6-35B-A3B` single-row Metal
  verify path on one canonical `M=16` quantized-linear flow, instead of
  letting full-attention, GDR, MLP, and final logits fall back to the generic
  3D `quantized_matmul` path layer by layer.

## Hypothesis

- If the compiled target model derives one `prefer_verify_m16` bit from the
  verify context and threads it through all eligible quantized linears, the
  `B=1, S=16` verify block should spend less time in repeated rank-3
  quantized-matmul setup and move closer to the `dflash-mlx` verify-linear
  pattern, with no change to acceptance semantics.

## Command

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
- **Commit base:** `755250c`
- **Feature set:** `--no-default-features --features metal,no-cuda`
- **Non-default flags / env vars:** none
- **Machine window note:** this run was still on a live desktop with background
  storage / endpoint-security services present. Right before the DFlash leg,
  `top` showed `75.46% idle`.

## Results

Correctness / validation:

- `verify_block_sampled_matches_batched_sampled_for_b1`: `1 passed`
- `verify_block_batched_matches_independent_verify_block_for_b2`: `1 passed`
- `metal_bench` unit test: `1 passed`
- `clippy -D warnings`: passed
- `metal_request` smoke: `exit 0`, `Output tokens: 2`

Serial long-generation benchmark:

| mode | gen tok/s | repo e2e tok/s | TTFT ms |
|---|---:|---:|---:|
| baseline | 86.53 | 86.05 | 66.31 |
| DFlash now | 55.97 | 55.77 | 66.00 |

Additional DFlash stats:

| block_size | avg accepted inputs | acceptance rate | blocks |
|---:|---:|---:|---:|
| 16 | 2.50 | 59.96% | 410 |

## Problems

- This slice does not add the upstream `verify_qmm` Metal kernel; it keeps the
  local bridge clean by collapsing eligible verify linears onto one canonical
  `M=16` `quantized_matmul` path first.
- Throughput is still below the plain baseline, so this commit is a verify-path
  cleanup and reduction of wasted linear overhead, not the final parity point.
- The host machine still had background services active, so the numbers above
  should be treated as the current same-window local regression check, not a
  pristine lab run.

## Learnings

- The stable cut is not “invent another verify-only linear stack”. A single
  `ForwardContext -> prefer_verify_m16 -> QWeight::apply` rule keeps the
  compiled model readable and removes layer-by-layer drift between full-attn,
  GDR, MLP, and logits.
- On this path, cleaning up `M=16` quantized-linears is enough to lift
  single-row DFlash without changing acceptance metrics. The accepted-input
  profile stayed flat while throughput improved, which points to real verify
  overhead reduction rather than a sampling artifact.
- The next meaningful gap is now outside this simple dispatch cleanup:
  remaining work should target the still-hot quantized matmul kernel itself or
  MoE-specific verify costs, not another round of routing tweaks.

## Δ vs baseline

- **Prior committed local snapshot:** [2026-04-22-bench-guidellm-qwen36-dflash-packed-verify-sdpa-2pass.md](./2026-04-22-bench-guidellm-qwen36-dflash-packed-verify-sdpa-2pass.md)
- **Same-session baseline:** included in the main results table above

| metric | prior committed local DFlash | now | Δ% |
|---|---:|---:|---:|
| generation_tps | 48.24 | 55.97 | +16.0% |
| repo_e2e_tps | 48.07 | 55.77 | +16.0% |
| TTFT ms | 72.97 | 66.00 | -9.6% |

## Notes

- Final code in this tranche:
  - adds one `should_prefer_verify_m16(ctx)` gate in the compiled target model
  - routes full-attention, GDR, MLP, and final logits through that same gate
  - reshapes eligible `[1, 16, H]` verify inputs to `[16, H]` once inside
    `QWeight::apply`
  - leaves all non-verify and non-eligible shapes on the stock path
