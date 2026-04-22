# Bench Stub — Qwen3.6 DFlash Prefix Verify on Metal

## Context

`z-lab/Qwen3.6-35B-A3B-DFlash` was functionally working on the local Metal
lane, but the default `block_size=16` path was still too slow because the
single-row verifier paid for a full 16-token target verify even when the draft
usually mismatched at the second speculative position.

This tranche changes the single-row Qwen3.5/Qwen3.6 Metal verifier from
"verify the whole block, then rollback rejected GDR state" to "verify one
target step at a time and stop at the first mismatch-inclusive position".
Batched DFlash stays on the packed full-block verify path.

## What Worked

- The single-row Qwen3.5/Qwen3.6 DFlash path in
  `infer/src/backend/metal/dflash.rs::qwen35_dflash_speculative_block`
  now uses prefix verify via `cpp_model.step()` and no longer records /
  replays GDR tapes for rejected suffixes that were never executed.
- The batched verifier still uses the sampled packed-block path, so the
  implementation keeps one canonical fast path per execution mode instead of
  stacking fallback branches inside the same hot loop.
- Local serial step-driver benchmark on `Apple M4 Pro`, target
  `mlx-community/Qwen3.6-35B-A3B-4bit`, draft
  `z-lab/Qwen3.6-35B-A3B-DFlash`, `prompt_tokens=20`,
  `generation_tokens=1024`, `warmup=1`, `runs=3`:

| mode | verify strategy | gen tok/s | repo e2e tok/s | TTFT ms | avg accepted inputs | acceptance rate |
|---|---|---:|---:|---:|---:|---:|
| baseline | n/a | 81.96 | 81.52 | 68.21 | n/a | n/a |
| DFlash before | full-block verify | 29.06 | 29.00 | 72.95 | 2.44 | 58.98% |
| DFlash now | prefix verify | 48.24 | 48.07 | 72.64 | 2.59 | 61.43% |

- Delta vs the prior DFlash `block_size=16` result on the same workload:
  - `generation_tps`: `+66.0%`
  - `repo_e2e_tps`: `+65.8%`
  - `avg_accepted_inputs`: `2.44 -> 2.59`
- Local profile sample (`QWEN35_DFLASH_PROFILE=1`, `runs=1`) showed the late
  steady-state verify block drop from roughly `62-65 ms` to `28-29 ms` after
  switching to prefix verify, with rollback eliminated on the single-row path.
- Local verification:
  - `cargo test -p infer --release --no-default-features --features metal,no-cuda --bin metal_bench tests::baseline_compare_default_draft_qwen36_a3b -- --exact`
  - `cargo run -p infer --release --no-default-features --features metal,no-cuda --bin metal_request -- --model /path/to/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46 --prompt hi --raw-prompt --max-new-tokens 1 --dflash-draft-model z-lab/Qwen3.6-35B-A3B-DFlash`
  - Both completed successfully; `metal_request` exited `0`.

## Rule

Status: `pending-remote`

On the current local Apple/Metal lane, the right verify optimization is not
"make full-block verify slightly cheaper" but "stop verifying past the first
mismatch when single-row acceptance is short". Keep the batched verifier on the
packed full-block path until separate batch data says otherwise.
