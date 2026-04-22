# Qwen3.6 DFlash GPU-Resident Single-Row Acceptance

## Goal

- Optimization / cleanup: keep the single-row Qwen3.6/Qwen3.5 DFlash
  `drafted / posterior / acceptance` path on GPU until emit time, instead of
  materializing whole speculative blocks on CPU during verify.

## Hypothesis

- If single-row DFlash stops materializing the drafted suffix and posterior
  block into Rust vectors, Metal should remove an avoidable GPU->CPU sync from
  the verify hot path, keep the implementation closer to `dflash-mlx`, and
  slightly improve end-to-end DFlash throughput without changing semantics.

## Command

```bash
cargo +stable test -p infer --release --no-default-features --features metal,no-cuda --lib backend::metal::mlx::tests::prefix_match_len_i32_counts_only_the_matching_prefix -- --exact
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
- **Commit base:** `7ee9aa6`
- **Feature set:** `--no-default-features --features metal,no-cuda`
- **Non-default flags / env vars:** none

## Results

Same-session strict-serial comparison after this change:

| mode | gen tok/s | repo e2e tok/s | TTFT ms |
|---|---:|---:|---:|
| baseline | 74.44 | 74.04 | 74.12 |
| DFlash now | 52.96 | 52.77 | 69.63 |

Additional DFlash stats:

| block_size | avg accepted inputs | acceptance rate |
|---:|---:|---:|
| 16 | 2.50 | 59.96% |

Validation:

- `prefix_match_len_i32` unit test: `1 passed`
- `metal_bench` unit test: `1 passed`
- `clippy -D warnings`: passed
- `metal_request` smoke: `exit 0`, `Output tokens: 2`

## Problems

- Local Metal DFlash still loses to same-session baseline on long generation.
  Relative to the baseline run above:
  - `generation_tps`: `-28.8%`
  - `repo_e2e_tps`: `-28.7%`
  - `TTFT`: `-6.1%` (better)
- This slice also exposed a real dtype mismatch between GPU-sampled token ids
  and the new prefix-match helper; the fix was to normalize sampled token
  arrays to `Int32` before acceptance and any debug materialization.

## Learnings

- The whole-block CPU materialization in single-row verify was removable
  without changing the verify contract: draft samples, posterior samples, and
  longest-prefix acceptance can stay as MLX arrays until the accepted emit
  slice is known.
- Adding a tiny MLX-side `prefix_match_len_i32` helper is enough to keep the
  acceptance decision GPU-resident while still returning a scalar prefix
  length to Rust.
- This is a cleanup win and a small speed win, not the end-state fast path.
  The next real gap is still the single-row verify API itself: `B=1` should
  stop going through the packed `cache_pos_arr` branch and use a native scalar
  cache-position path.

## Δ vs baseline

- **Prior committed local snapshot:** [2026-04-22-bench-guidellm-qwen36-dflash-sampled-full-verify.md](./2026-04-22-bench-guidellm-qwen36-dflash-sampled-full-verify.md)
- **Same-session baseline:** included in the main results table above

| metric | prior committed local DFlash | now | Δ% |
|---|---:|---:|---:|
| generation_tps | 50.39 | 52.96 | +5.1% |
| repo_e2e_tps | 50.21 | 52.77 | +5.1% |
| TTFT ms | 73.32 | 69.63 | -5.0% |

## Notes

- Code change in this tranche:
  - add an MLX bridge helper for longest-prefix token match
  - keep single-row drafted suffix and posterior tokens on GPU
  - materialize only the accepted output slice needed by request-state emit
  - normalize sampled token arrays to `Int32` so the acceptance path is dtype-stable
