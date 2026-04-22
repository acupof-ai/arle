# Qwen3.6 DFlash Native Single-Row Verify API

## Goal

- Cleanup / optimization: stop routing single-row Qwen3.6/Qwen3.5 DFlash
  verify through the packed `cache_pos_arr` entrypoint with `B=1`, and give
  `CppQwen35Model` a native sampled verify API that uses scalar `cache_pos`.

## Hypothesis

- If single-row verify stops paying the packed per-row cache-position plumbing,
  the code path becomes simpler and closer to the real single-row decode
  contract, which should remove one known source of CPU-side verify overhead.

## Command

```bash
cargo +stable test -p infer --release --no-default-features --features metal,no-cuda --lib backend::metal::qwen35::tests::verify_block_sampled_matches_batched_sampled_for_b1 -- --exact
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
- **Commit base:** `5ceed96`
- **Feature set:** `--no-default-features --features metal,no-cuda`
- **Non-default flags / env vars:** none

## Results

Same-window strict-serial comparison after switching single-row verify to the
native scalar-cache API:

| mode | gen tok/s | repo e2e tok/s | TTFT ms |
|---|---:|---:|---:|
| baseline | 79.78 | 79.35 | 70.33 |
| DFlash now | 51.00 | 50.82 | 70.43 |

Additional DFlash stats:

| block_size | avg accepted inputs | acceptance rate |
|---:|---:|---:|
| 16 | 2.50 | 59.96% |

Validation:

- `verify_block_sampled_matches_batched_sampled_for_b1`: `1 passed`
- `metal_bench` unit test: `1 passed`
- `clippy -D warnings`: passed
- `metal_request` smoke: `exit 0`, `Output tokens: 2`

## Problems

- This benchmark window was polluted by unrelated machine load:
  - `target/release/pretrain` at roughly `93% CPU`
  - `StorageManagementService` / `ApplicationsStorageExtension` both above `50% CPU`
- Because of that, this entry is only trustworthy for same-window relative
  comparison, not for absolute throughput claims.
- In this noisy window, DFlash still loses to same-window baseline:
  - `generation_tps`: `-36.1%`
  - `repo_e2e_tps`: `-35.9%`
  - `TTFT`: effectively flat (`+0.1 ms`)

## Learnings

- The single-row verify contract is now structurally correct: single-row
  DFlash no longer fabricates `cache_pos_arr` / `rope_offsets` just to route
  through the packed verifier.
- The dedicated sampled verify API is semantically equivalent to the old
  packed `B=1` path under greedy sampling, as pinned by the new unit test.
- This slice removes an implementation mismatch and one known sync site, but
  it does not by itself fix the larger throughput gap. The next useful work is
  still in the attention kernel and quantized verify path, not more Rust-side
  wrapper churn.

## Δ vs baseline

- **Prior committed local snapshot:** [2026-04-22-bench-guidellm-qwen36-dflash-gpu-acceptance.md](./2026-04-22-bench-guidellm-qwen36-dflash-gpu-acceptance.md)
- **Same-session baseline:** included in the main results table above

| metric | prior committed local DFlash | now | Δ% |
|---|---:|---:|---:|
| generation_tps | 52.96 | 51.00 | -3.7% |
| repo_e2e_tps | 52.77 | 50.82 | -3.7% |
| TTFT ms | 69.63 | 70.43 | +1.1% |

## Notes

- Code change in this tranche:
  - add `qwen35_compiled_verify_block_sampled`
  - add `CppQwen35Model::verify_block_sampled`
  - switch single-row Qwen3.6/Qwen3.5 DFlash to the native scalar-cache verify API
  - update docs so single-row verify no longer claims to use packed `B=1`
