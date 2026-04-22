# Qwen3.6 DFlash Suppress Mask Token During Sampling

## Goal

- Align Metal DFlash sampling with `dflash-mlx` by suppressing the draft
  `mask_token_id` everywhere the speculative path samples tokens:
  prefill `staged_first`, draft suffix sampling, scalar verify summary, and
  packed sampled verify.

## Hypothesis

- If Metal DFlash stops allowing `mask_token_id` back into draft/posterior
  sampling, the path becomes semantically closer to `dflash-mlx` and should
  remove one obvious acceptance-risk difference. Performance impact may be
  small, but the sampling contract becomes cleaner and uniform.

## Command

```bash
cargo +stable build -p infer --release --no-default-features --features metal,no-cuda --bin metal_request
```

```bash
cargo +stable test -p infer --release --lib --no-default-features --features metal,no-cuda backend::metal::qwen35::tests::verify_block_summary_matches_batched_sampled_for_b1 -- --exact --nocapture
```

```bash
cargo +stable clippy -p infer --release --no-default-features --features metal,no-cuda --bin metal_bench --bin metal_request -- -D warnings
```

```bash
target/release/metal_request \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --prompt hi \
  --raw-prompt \
  --max-new-tokens 2 \
  --dflash-draft-model z-lab/Qwen3.6-35B-A3B-DFlash
```

```bash
target/release/metal_bench \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
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
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
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
- **Feature set:** `--no-default-features --features metal,no-cuda`
- **Execution mode:** strict serial, `baseline -> DFlash`

## Results

Strict-serial long-generation benchmark after suppressing `mask_token_id`
across Metal DFlash sampling:

| mode | gen tok/s | repo e2e tok/s | TTFT ms |
|---|---:|---:|---:|
| baseline | 87.53 | 87.04 | 65.62 |
| DFlash now | 33.33 | 33.25 | 75.66 |

Additional DFlash stats:

| block_size | avg accepted inputs | acceptance rate | blocks |
|---:|---:|---:|---:|
| 16 | 2.44 | 58.98% | 420 |

Validation:

- `verify_block_summary_matches_batched_sampled_for_b1`: `1 passed`
- `clippy -D warnings`: passed
- `metal_request` smoke: `exit 0`, `Output tokens: 2`

## Problems

- Acceptance did not improve on this benchmark window; the dominant bottleneck
  is still low realized prefix depth, not token masking.
- Relative to same-window baseline:
  - `generation_tps`: `-61.9%`
  - `repo_e2e_tps`: `-61.8%`
  - `TTFT`: `+15.3%`

## Learnings

- Suppressing the draft `mask_token_id` is still the right contract. It removes
  an obvious semantic mismatch with `dflash-mlx` and keeps every DFlash
  sampling site on one rule.
- On the measured long-generation workload, this change gave only a small
  throughput lift versus the previous local DFlash snapshot; it did not move
  acceptance depth.
- The next useful optimization should target why accepted prefix length stays
  near `2.44`, not more cleanup around sampling wrappers.

## Δ vs prior committed local DFlash

- **Prior snapshot:** [2026-04-22-bench-guidellm-qwen36-dflash-verify-summary-prefetch-cleanup.md](./2026-04-22-bench-guidellm-qwen36-dflash-verify-summary-prefetch-cleanup.md)

| metric | prior DFlash | now | Δ% |
|---|---:|---:|---:|
| generation_tps | 32.14 | 33.33 | +3.7% |
| repo_e2e_tps | 32.06 | 33.25 | +3.7% |
| TTFT ms | 79.25 | 75.66 | -4.5% |

## Rule

- Metal DFlash sampling must suppress the draft runtime's `mask_token_id`
  consistently across prefill, draft, and verify paths. Do not let scalar and
  batched paths drift into different sampling semantics.
