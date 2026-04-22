# Qwen3.6 DFlash Verify MoE M=16 Flatten

## Goal

- Optimization / alignment: extend the compiled-model `prefer_verify_m16`
  cleanup into the Qwen3.6 MoE MLP path, since trace now shows
  `qwen35_moe_block_forward_cpp` in the single-row verify hot stack.

## Hypothesis

- If `moe_mlp()` flattens verify-only `[1, 16, H]` hidden states to `[16, H]`
  before entering `qwen35_moe_block_forward_cpp`, the MoE router/shared dense
  linears and gather setup will pay the cheaper `M=16` shape contract without
  changing acceptance behavior.

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

Trace used to pick the change:

```bash
QWEN35_DFLASH_PROFILE=1 target/release/metal_bench \
  --model /path/to/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46 \
  --dflash-draft-model z-lab/Qwen3.6-35B-A3B-DFlash \
  --use-step-driver \
  --ignore-eos \
  --prompt-tokens 20 \
  --generation-tokens 1024 \
  --warmup 1 \
  --runs 1 \
  --json
```

```bash
sample <metal_bench_pid> 5 10 -file /tmp/qwen36_dflash_verify_linear_m16.sample.txt
```

## Environment

- **Backend:** Metal
- **Model:** `mlx-community/Qwen3.6-35B-A3B-4bit`
- **Draft:** `z-lab/Qwen3.6-35B-A3B-DFlash`
- **Hardware:** `Apple M4 Pro`
- **OS:** `macOS 26.3.1 (25D771280a)`
- **Commit base:** `cc266b2`
- **Feature set:** `--no-default-features --features metal,no-cuda`
- **Non-default flags / env vars:** `QWEN35_DFLASH_PROFILE=1` for the trace-only run

## Results

Correctness / validation:

- `verify_block_sampled_matches_batched_sampled_for_b1`: `1 passed`
- `verify_block_batched_matches_independent_verify_block_for_b2`: `1 passed`
- `metal_bench` unit test: `1 passed`
- `metal_request` smoke: `exit 0`, `Output tokens: 2`

Serial long-generation benchmark:

| mode | gen tok/s | repo e2e tok/s | TTFT ms |
|---|---:|---:|---:|
| baseline | 84.48 | 84.02 | 66.25 |
| DFlash now | 58.42 | 58.20 | 66.25 |

Additional DFlash stats:

| block_size | avg accepted inputs | acceptance rate | blocks |
|---:|---:|---:|---:|
| 16 | 2.50 | 59.96% | 410 |

Trace findings that selected this change:

- `sample` still showed `mlx::core::QuantizedMatmul::eval_gpu` and
  `mlx::core::GatherQMM::eval_gpu` in the verify stack.
- The same capture also surfaced `qwen35_moe_block_forward_cpp` directly under
  `Qwen35CompiledModel::forward_impl`, confirming that MoE participates in the
  single-row verify hot path for this checkpoint.

## Problems

- This slice only flattens the verify-only MoE entry shape. It does not add a
  dedicated `gather_qmm` `M=16` kernel, so the expert path remains a likely
  next limiter.
- The acceptance profile stayed unchanged, so this is a fixed-overhead
  reduction only; it does not solve the underlying short-prefix acceptance
  ceiling.

## Learnings

- The compiled-model cleanup had one obvious missing branch: MoE. Once the
  trace showed `qwen35_moe_block_forward_cpp`, extending the same `M=16`
  flatten rule there was the clean follow-on, not another attention-side tweak.
- This change improved throughput without moving acceptance or TTFT, which is
  the expected signature for a real MoE verify overhead reduction.
- The next optimization target is more likely inside `switch_glu_forward` /
  `gather_qmm` than in yet another caller-side routing refactor.

## Δ vs baseline

- **Prior committed local snapshot:** [2026-04-22-bench-guidellm-qwen36-dflash-verify-linear-m16.md](./2026-04-22-bench-guidellm-qwen36-dflash-verify-linear-m16.md)
- **Same-session baseline:** included in the main results table above

| metric | prior committed local DFlash | now | Δ% |
|---|---:|---:|---:|
| generation_tps | 55.97 | 58.42 | +4.4% |
| repo_e2e_tps | 55.77 | 58.20 | +4.4% |
| TTFT ms | 66.00 | 66.25 | +0.4% |

## Notes

- Final code in this tranche:
  - extends `moe_mlp()` with the same `prefer_verify_m16` rule already used by
    full-attention, GDR, dense MLP, and logits
  - flattens verify-only `[1, 16, H]` MoE inputs to `[16, H]` once before the
    existing `qwen35_moe_block_forward_cpp` helper
  - reshapes the MoE output back to `[1, 16, H]` before the residual add
