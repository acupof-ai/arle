# Qwen3.6 DFlash Verify Summary + Light Prefetch Cleanup

## Goal

- Keep the single-row Qwen3.6/Qwen3.5 Metal DFlash path closer to
  `dflash-mlx` while making the implementation simpler:
  - return only verify summary data from C++
  - keep next-block prefetch lightweight
  - delete the cloned `draft_state` side path

## Hypothesis

- If single-row verify returns only `(matched_prefix_len, next_token)` and
  prefetch carries only `seed_token + block_tokens`, the hot path stays easier
  to reason about and removes one redundant draft-cache clone without changing
  correctness.

## Command

```bash
cargo +stable build -p infer --release --no-default-features --features metal,no-cuda --bin metal_request
```

```bash
cargo +stable test -p infer --release --no-default-features --features metal,no-cuda --bin metal_bench tests::baseline_compare_default_draft_qwen36_a3b -- --exact
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

Strict-serial long-generation benchmark after switching single-row verify to
the summary API and deleting the cloned prefetch cache path:

| mode | gen tok/s | repo e2e tok/s | TTFT ms |
|---|---:|---:|---:|
| baseline | 86.22 | 85.74 | 66.75 |
| DFlash now | 32.14 | 32.06 | 79.25 |

Additional DFlash stats:

| block_size | avg accepted inputs | acceptance rate | blocks |
|---:|---:|---:|---:|
| 16 | 2.44 | 58.98% | 420 |

Validation:

- `metal_bench` unit test: `1 passed`
- `metal_request` smoke: `exit 0`, `Output tokens: 2`
- release build: passed

## Problems

- This cleanup did not improve throughput.
- Relative to same-window baseline:
  - `generation_tps`: `-62.7%`
  - `repo_e2e_tps`: `-62.6%`
  - `TTFT`: `+18.7%`

## Learnings

- The C++ bridge should expose the narrowest contract the Rust caller needs.
  Returning `(matched_prefix_len, next_token)` is cleaner than handing Rust a
  whole sampled posterior block just to reconstruct the same answer.
- The next-block prefetch state should stay token-only. Cloning a second
  `draft_state` made the control flow harder to trust and did not buy speed.
- This tranche makes the implementation more uniform with `dflash-mlx`, but
  it confirms that the dominant remaining gap is still acceptance depth, not
  Rust-side prefetch bookkeeping.

## Rule

- For single-row Metal DFlash, keep prefetched state as lightweight staged
  tokens plus seed metadata. Do not introduce parallel cloned draft-cache
  flows unless a benchmark shows a clear win.
