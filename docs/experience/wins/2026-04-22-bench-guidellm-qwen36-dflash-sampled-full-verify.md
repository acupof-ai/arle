# Qwen3.6 DFlash Sampled Full Verify

## Goal

- Optimization / diagnosis: remove the single-row Qwen3.6 DFlash verify path's
  per-step sampling sync and converge it on the existing sampled packed-verify
  path already used by the batched runtime.

## Hypothesis

- If single-row DFlash stops calling `sample_last_token -> eval` once per
  accepted target step and instead reuses one sampled full-block verify, Metal
  should recover the throughput lost by the rejected chunked-verify attempt and
  keep the implementation closer to the batched fast path.

## Command

```bash
cargo +stable test -p infer --release --no-default-features --features metal,no-cuda --bin metal_bench tests::baseline_compare_default_draft_qwen36_a3b -- --exact
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

```bash
cargo +stable run -p infer --release --no-default-features --features metal,no-cuda --bin metal_request -- --model /path/to/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46 --prompt hi --raw-prompt --max-new-tokens 2 --dflash-draft-model z-lab/Qwen3.6-35B-A3B-DFlash
```

## Environment

- **Backend:** Metal
- **Model:** `mlx-community/Qwen3.6-35B-A3B-4bit`
- **Draft:** `z-lab/Qwen3.6-35B-A3B-DFlash`
- **Hardware:** `Apple M4 Pro`
- **OS:** `macOS 26.3.1 (25D771280a)`
- **Commit base:** `b6040df`
- **Feature set:** `--no-default-features --features metal,no-cuda`
- **Non-default flags / env vars:** none
- **Workspace note:** local validation ran in a dirty workspace with unrelated
  non-Metal edits already present; the DFlash code change in this tranche is
  limited to `infer/src/backend/metal/dflash.rs` plus docs.

## Results

Final same-session serial comparison on the current implementation:

| mode | verify strategy | gen tok/s | repo e2e tok/s | TTFT ms |
|---|---|---:|---:|---:|
| baseline | plain target decode | 71.51 | 71.15 | 71.91 |
| DFlash now | sampled full-block verify (`B=1`) | 50.39 | 50.21 | 73.32 |

Additional DFlash stats:

| block_size | avg accepted inputs | acceptance rate |
|---:|---:|---:|
| 16 | 2.50 | 59.96% |

Immediate local comparison against the rejected chunked-verify attempt from the
same machine window:

| mode | verify strategy | gen tok/s | repo e2e tok/s | TTFT ms |
|---|---|---:|---:|---:|
| DFlash rejected | sampled small chunks | 43.77 | 43.63 | 78.57 |
| DFlash now | sampled full-block verify (`B=1`) | 50.39 | 50.21 | 73.32 |

Validation:

- `metal_bench` unit test: `1 passed`
- `metal_request` smoke: `exit 0`, `Output tokens: 2`

## Problems

- This still does **not** beat same-session baseline on the local Metal lane.
  Relative to the current baseline run:
  - `generation_tps`: `-29.5%`
  - `repo_e2e_tps`: `-29.4%`
  - `TTFT`: `+2.0%`
- Historical comparison to the prior committed
  [prefix-verify-session reuse snapshot](./2026-04-22-bench-guidellm-qwen36-dflash-prefix-verify-session.md)
  is noisier because the machine window was different; that older local run
  recorded `52.78 tok/s` for DFlash and `81.96 tok/s` for baseline, both higher
  than the numbers in this session.

## Learnings

- On Metal Qwen3.6 single-row DFlash, the bad path was not the target verify
  kernel alone; the trace-backed problem was the Rust-side per-step
  `sample_last_token -> eval` sync boundary.
- Reusing the existing sampled packed verifier with `B=1` is cleaner than
  maintaining a separate scalar/session loop and materially better than the
  rejected chunked-verify experiment.
- Even after removing the per-step sample fence, local throughput is still
  bounded by short accepted prefixes (`avg_accepted_inputs ~= 2.5` with
  `block_size=16`), so future gains need to come from acceptance quality and/or
  draft-side cost, not from adding more Rust-side verify branches.

## Δ vs baseline

- **Prior committed local snapshot:** [2026-04-22-bench-guidellm-qwen36-dflash-prefix-verify-session.md](./2026-04-22-bench-guidellm-qwen36-dflash-prefix-verify-session.md)
- **Same-session baseline:** included in the main results table above

| metric | prior committed local DFlash | now | Δ% |
|---|---:|---:|---:|
| generation_tps | 52.78 | 50.39 | -4.5% |
| repo_e2e_tps | 52.60 | 50.21 | -4.5% |
| TTFT ms | 65.83 | 73.32 | +11.4% |

## Notes

- Code change in this tranche:
  - delete the slower chunked single-row verify prototype
  - make single-row Qwen3.6/Qwen3.5 DFlash reuse
    `verify_block_batched_sampled(B=1)` plus the existing rollback helper
  - keep block materialization asynchronous at the end of the speculative block
- This entry records the final local state worth keeping in-tree, not the
  rejected chunked intermediate.
